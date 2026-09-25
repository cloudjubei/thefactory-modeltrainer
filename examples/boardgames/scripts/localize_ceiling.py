"""§C.42 item 1 — LOCALIZE THE CEILING: train N independent seeds, dump every state each net misplays, and read
whether the blind spot is systematic, where it lives (net vs search budget), and why (training visits, the
optimal-play manifold). Writes the full per-seed evidence so harness.ceiling can recompute every verdict.

    PYTHONPATH=. .venv/bin/python scripts/localize_ceiling.py --seeds 1-10 --out evidence/tictactoe_ceiling.json.gz
    PYTHONPATH=. .venv/bin/python scripts/localize_ceiling.py --seeds 11-20 --opening-plies 2 \
        --opening-zero-frac 0.5 --out evidence/tictactoe_mixed.json.gz
    PYTHONPATH=. .venv/bin/python scripts/localize_ceiling.py --ab evidence/tictactoe_base.json.gz \
        evidence/tictactoe_mixed.json.gz evidence/tictactoe_ceiling.json.gz

Every state is evaluated with a FRESH search tree (coverage.per_state_act): an agent that keeps its tree across
states scores differently depending on the order states are asked in (§C.42). The raw policy is ALSO scored on
every reachable orientation (`orientation`): `policy_fail_keys` reads one canonical image per state, which called
nets perfect that misplay a rotated board (§C.46 verification)."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


MEASUREMENT_MODULES = ("scripts/localize_ceiling.py", "harness/coverage.py", "harness/targets.py")
ORACLE_MODULES = ("harness/targets.py", "harness/coverage.py")
# The net is stated EXPLICITLY. §C.43-§C.45 recorded {"channels": 32, "blocks": 3, "head_hidden": 32} but, with no
# `residual`, the legacy branch ignored blocks/head_hidden and trained a 12,746-parameter net (§C.46 review).
ARCHS = {"legacy": {"channels": 32},
         "residual": {"channels": 32, "blocks": 3, "head_hidden": 32, "residual": True}}
TRAIN_DEFAULTS = {"epochs": 6, "batch_size": 64, "lr": 1e-3, "buffer_cap": 8000}
# Registered parameter counts per preset, so a drifting preset is refused at launch rather than compared with itself.
PARAMS_EXPECTED = {"legacy": 12746, "residual": 57453}


def _parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def _child_value(game, net, state, action, net_value):
    child = game.step(state, action)
    if game.is_terminal(child):
        return float(game.returns(child)[game.current_player(state)])
    return -net_value(net, game, child)


def run_seed(cfg: dict) -> dict:
    import torch

    torch.set_num_threads(cfg["threads"])
    from harness.coverage import (coverage_failures, failable_keys, optimal_actions, per_state_act, reachable_states,
                                  training_visits)
    from harness.neural import AlphaZeroAgent, encode, net_value, train_alphazero
    from harness.registry import resolve_game

    from contextlib import ExitStack

    from harness.targets import POLICY_TARGETS, record_training_labels, record_training_passes

    game = resolve_game(cfg["game"])
    target_keys = cfg.get("target_keys") or []
    states, complete = reachable_states(game, exact=True, symmetry=True)
    assert complete, "localization needs the full state space"
    knobs: dict = {}
    if cfg.get("reanalyze_frac"):
        knobs.update(reanalyze_frac=cfg["reanalyze_frac"], reanalyze_sims=cfg["reanalyze_sims"])
    if cfg.get("reanalyze_siblings") or cfg.get("steps_matched") or cfg.get("sibling_holdout"):
        knobs.update(reanalyze_siblings=bool(cfg.get("reanalyze_siblings")), steps_matched=bool(cfg.get("steps_matched")),
                     sibling_holdout=cfg.get("sibling_holdout"))
    if cfg.get("policy_target"):
        knobs["policy_target_fn"] = POLICY_TARGETS[cfg["policy_target"]]
    t0 = time.time()
    with ExitStack() as stack:
        written = stack.enter_context(record_training_labels(game, target_keys, encode)) if target_keys else []
        passes = stack.enter_context(record_training_passes(game, states, encode))
        net, history, buffer = train_alphazero(
            game, iterations=cfg["iterations"], selfplay_games=cfg["selfplay"], sims=cfg["train_sims"],
            channels=cfg["arch"]["channels"], net_arch=cfg["arch"], augment=True, gumbel=True, seed=cfg["seed"],
            selfplay_opening_plies=cfg["opening_plies"], opening_plies_zero_frac=cfg["opening_zero_frac"],
            epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"], buffer_cap=cfg["buffer_cap"],
            return_buffer=True, **knobs)
    train_s = time.time() - t0

    def agent(sims):
        return per_state_act(game, lambda: AlphaZeroAgent(net, sims=sims, solve_endgame=0, gumbel=True, c_scale=0.1))

    failures = coverage_failures(game, agent(cfg["eval_sims"]), states)
    by_key = {game.canonical_key(s): s for s in states}
    deep = agent(cfg["deep_sims"])
    net.eval()
    for f in failures:
        s = by_key[f["key"]]
        with torch.no_grad():
            logits, _v = net(encode(game, s).unsqueeze(0))
        prior = {a: float(logits[0, a]) for a in game.legal_actions(s)}
        best_prior_opt = max(prior[a] for a in f["optimal"])
        best_value_opt = max(_child_value(game, net, s, a, net_value) for a in f["optimal"])
        f["deep_ok"] = deep(s) in f["optimal"]
        f["prior_prefers_played"] = prior[f["move"]] >= best_prior_opt
        f["value_prefers_played"] = _child_value(game, net, s, f["move"], net_value) >= best_value_opt
        f["prior_argmax_ok"] = max(prior, key=prior.get) in f["optimal"]
    policy_fail_keys = []
    for s in states:
        with torch.no_grad():
            logits, _v = net(encode(game, s).unsqueeze(0))
        legal = game.legal_actions(s)
        a = max(legal, key=lambda x: float(logits[0, x]))
        if a not in optimal_actions(game, s):
            policy_fail_keys.append(game.canonical_key(s))
    from harness.coverage import orientation_failures
    from harness.symmetry import verified_isometries

    def logits_fn(state):
        with torch.no_grad():
            lg, _v = net(encode(game, state).unsqueeze(0))
        return [float(x) for x in lg[0]]
    isos = {iso.name: iso for iso, _f in verified_isometries(game)}
    orientation = orientation_failures(game, logits_fn, list(isos.values()))
    from harness.coverage import subgroup_failures
    from harness.floor import C47_SUBGROUPS

    subgroups = ({order: subgroup_failures(game, logits_fn, [isos[n] for n in names])
                  for order, names in C47_SUBGROUPS.items()} if set(isos) == set(C47_SUBGROUPS["8"]) else None)
    if cfg.get("save_nets"):
        out = Path(cfg["save_nets"])
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "arch": cfg["arch"]}, out / f"seed{cfg['seed']}.pt")
    universe = failable_keys(game, states)
    tv = training_visits(game, [e for e in buffer if e[2] == e[2]], encode)
    sv = training_visits(game, [e for e in buffer if e[2] != e[2]], encode)
    extra: dict = {}
    if target_keys:
        from harness.targets import search_decomposition, target_error

        target_labels = {"label_ok": 0, "prior_anchor": 0, "search_miss": 0}
        target_net = {}
        for k in target_keys:
            st = by_key[k]
            opt = optimal_actions(game, st)
            for t, draws in ((0.0, [0]), (1.0, list(range(cfg["label_draws"])))):
                for d in draws:
                    dec = search_decomposition(
                        game, lambda: AlphaZeroAgent(net, sims=cfg["label_sims"], gumbel=True, c_scale=0.1),
                        st, seed=d, temperature=t)
                    target_labels[target_error(dec, opt)] += 1
            with torch.no_grad():
                logits, _v = net(encode(game, st).unsqueeze(0))
            legal = game.legal_actions(st)
            probs = torch.softmax(torch.tensor([float(logits[0, a]) for a in legal]), dim=0).tolist()
            prior = dict(zip(legal, probs))
            value_move = max(legal, key=lambda a: _child_value(game, net, st, a, net_value))
            target_net[k] = {"prior_ok": max(prior, key=prior.get) in opt,
                             "prior_opt_mass": sum(prior[a] for a in opt), "value_ok": value_move in opt}
        exact_leaf = {}
        for cs in (0.1, 1.0):
            counts = {"label_ok": 0, "prior_anchor": 0, "search_miss": 0}
            for k in target_keys:
                opt = optimal_actions(game, by_key[k])
                for t, draws in ((0.0, [0]), (1.0, list(range(cfg["label_draws"])))):
                    for d in draws:
                        dec = search_decomposition(
                            game, lambda: AlphaZeroAgent(net, sims=cfg["label_sims"], gumbel=True, c_scale=cs,
                                                         solve_endgame=9), by_key[k], seed=d, temperature=t)
                        counts[target_error(dec, opt)] += 1
            exact_leaf[str(cs)] = counts
        extra = {"target_labels": target_labels, "target_net": [[k, v] for k, v in sorted(target_net.items())],
                 "written_labels": written, "target_labels_exact_leaf": exact_leaf}
    return {"seed": cfg["seed"], "coverage": 1 - len(failures) / len(states), "n_states": len(states),
            "policy_coverage": 1 - len(policy_fail_keys) / len(states), "failures": failures,
            "policy_fail_keys": policy_fail_keys,
            "visits": [[k, tv["visits"].get(k, 0)] for k in sorted(universe)],
            "sibling_visits": [[k, sv["visits"].get(k, 0)] for k in sorted(universe)],
            "buffer_size": len(buffer), "buffer_unmatched": tv["unmatched"] + sv["unmatched"],
            "train_seconds": round(train_s, 1), "params": sum(p.numel() for p in net.parameters()),
            "history": history, "passes": passes, "orientation": orientation,
            **({"subgroups": subgroups} if subgroups is not None else {}), **extra}


def search_alone_control(game, arch: dict, eval_sims: int, deep_sims: int, keys, seed: int = 0) -> dict:
    """The same measurements with an UNTRAINED net: its coverage at the eval budget (what search contributes with
    no learning at all), its raw-policy coverage, and whether it plays each failing state correctly at the deep
    budget. Without this, "deep search fixes the failures" cannot be told apart from search brute-forcing a small
    tree (harness.ceiling marks the reading confounded)."""
    import torch

    from harness.coverage import optimal_actions, per_state_act, reachable_states, state_coverage
    from harness.neural import AlphaZeroAgent, Connect4Net, arch_for_game, encode, legacy_arch_as_built

    torch.manual_seed(seed)
    blank = Connect4Net(**arch_for_game(legacy_arch_as_built(arch), game))
    blank.eval()
    states, _ = reachable_states(game, exact=True, symmetry=True)
    by_key = {game.canonical_key(s): s for s in states}

    def agent(sims):
        return per_state_act(game, lambda: AlphaZeroAgent(blank, sims=sims, solve_endgame=0, gumbel=True, c_scale=0.1))

    policy_ok = 0
    for st in states:
        with torch.no_grad():
            logits, _v = blank(encode(game, st).unsqueeze(0))
        a = max(game.legal_actions(st), key=lambda x: float(logits[0, x]))
        policy_ok += 1 if a in optimal_actions(game, st) else 0
    deep = agent(deep_sims)
    return {"net_seed": seed, "coverage_eval": state_coverage(game, agent(eval_sims), states=states)["coverage"],
            "policy_coverage": policy_ok / len(states),
            "deep_ok": {k: deep(by_key[k]) in optimal_actions(game, by_key[k]) for k in sorted(set(keys))}}


def operator_sanity(game, arch: dict, label_sims: int, net_seeds=range(5)) -> dict:
    """What the floor operator and the labeller give with NO learning: the symmetry-averaged policy of untrained
    nets (failures over the canonical states), and how often the relabelling search, run from an untrained net at
    the recipe's label budget, picks an optimal move at the failable states. The second is the sense in which the
    labels are near-exhaustive at this game size."""
    import torch

    from harness.coverage import failable_keys, optimal_actions, orientation_failures, per_state_act, reachable_states
    from harness.neural import AlphaZeroAgent, Connect4Net, arch_for_game, encode
    from harness.symmetry import verified_isometries

    isos = [iso for iso, _f in verified_isometries(game)]
    states, _ = reachable_states(game, exact=True, symmetry=True)
    failable = failable_keys(game, states)
    targets = [s for s in states if game.canonical_key(s) in failable]
    counts, label_ok = [], None
    for seed in net_seeds:
        torch.manual_seed(seed)
        blank = Connect4Net(**arch_for_game(arch, game))
        blank.eval()

        def logits_fn(state, net=blank):
            with torch.no_grad():
                lg, _v = net(encode(game, state).unsqueeze(0))
            return [float(x) for x in lg[0]]
        counts.append(len(orientation_failures(game, logits_fn, isos)["symmetrized_fail_keys"]))
        if label_ok is None:
            act = per_state_act(game, lambda: AlphaZeroAgent(blank, sims=label_sims, solve_endgame=0, gumbel=True,
                                                            c_scale=0.1))
            label_ok = sum(1 for s in targets if act(s) in optimal_actions(game, s)) / len(targets)
    return {"untrained_symmetrized_failures": counts, "untrained_label_ok_share": label_ok,
            "label_sims": label_sims, "failable_states": len(targets)}


def _attach_control(game, evidence: dict) -> None:
    cfg = evidence["config"]
    keys = {f["key"] for sd in evidence["seeds"] for f in sd["failures"]}
    ctrl = search_alone_control(game, cfg["arch"], cfg["eval_sims"], cfg["deep_sims"], keys)
    for sd in evidence["seeds"]:
        for f in sd["failures"]:
            f["control_deep_ok"] = ctrl["deep_ok"][f["key"]]
    evidence["search_alone"] = {k: v for k, v in ctrl.items() if k != "deep_ok"}
    sanity = operator_sanity(game, cfg["arch"], cfg.get("reanalyze_sims") or cfg["train_sims"])
    evidence["operator_sanity"] = sanity
    print(f"search-alone control (untrained net): coverage@{cfg['eval_sims']} {ctrl['coverage_eval']:.4f}  "
          f"policy-only {ctrl['policy_coverage']:.4f}", flush=True)
    print(f"operator sanity (untrained nets): symmetrized failures {sanity['untrained_symmetrized_failures']}  "
          f"label ok at {sanity['label_sims']} sims {sanity['untrained_label_ok_share']:.3f}", flush=True)


def _report(evidence: dict) -> None:
    from harness.ceiling import localize_report

    rep = localize_report(evidence)
    print(json.dumps({k: v for k, v in rep.items() if k != "recurring"}, indent=1, default=str))
    print("recurring:", rep["recurring"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="tictactoe")
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--selfplay", type=int, default=48)
    ap.add_argument("--train-sims", type=int, default=32)
    ap.add_argument("--eval-sims", type=int, default=48)
    ap.add_argument("--deep-sims", type=int, default=400)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--opening-plies", type=int, default=0, help="random unrecorded self-play opening plies")
    ap.add_argument("--opening-zero-frac", type=float, default=0.0,
                    help="share of self-play games that start from the canonical opening (0 plies)")
    ap.add_argument("--ab", nargs=3, metavar=("BASE", "TREAT", "TARGET"),
                    help="judge a recipe change: two arm evidence files and the evidence the target was localized from")
    ap.add_argument("--treatment-keys", default="opening_plies,opening_zero_frac",
                    help="--ab: the config keys the arms are DECLARED to differ in (comma-separated)")
    ap.add_argument("--manipulation", default="visits", help="--ab: the mechanism the treatment targets (visits|labels)")
    ap.add_argument("--alpha", type=float, default=0.05, help="--ab: split /4 across its four claims")
    ap.add_argument("--target", default="",
                    help="record each seed's self-play LABEL quality at this evidence's failing states (§C.45)")
    ap.add_argument("--label-draws", type=int, default=8, help="Gumbel-noise draws per target state for --target")
    ap.add_argument("--label-sims", type=int, default=32,
                    help="--target: the COMMON search budget the final net's labels are read at in every arm")
    ap.add_argument("--arch", choices=sorted(ARCHS), default="legacy",
                    help="the net, stated explicitly (legacy = the 12,746-parameter net §C.43-§C.45 actually trained)")
    ap.add_argument("--reanalyze-siblings", action="store_true",
                    help="§C.46: add every one-move deviation from the recorded states, relabelled, policy-only")
    ap.add_argument("--steps-matched", action="store_true",
                    help="§C.46: keep the optimisation steps of the self-play rows alone when siblings are added")
    ap.add_argument("--sibling-holdout", default="", help="§C.46: 'mod:salt' — withhold those sibling keys")
    ap.add_argument("--policy-target", default="", choices=["", "exact_uniform_optimal"],
                    help="§C.46 DIAGNOSTIC ONLY: relabel with the solver's policy (never part of the generic process)")
    ap.add_argument("--reanalyze-frac", type=float, default=0.0, help="share of the buffer relabelled each iteration")
    ap.add_argument("--reanalyze-sims", type=int, default=0,
                    help="relabel search budget (0 = the self-play budget); needs --reanalyze-frac > 0")
    ap.add_argument("--out", default="")
    ap.add_argument("--save-nets", default="", help="directory to save each seed's final net (state_dict + arch)")
    ap.add_argument("--control-only", action="store_true",
                    help="attach the search-alone control to existing evidence at --out instead of training")
    args = ap.parse_args()

    from harness.coverage import failable_keys, optimal_play_keys, reachable_states
    from harness.evidence import load_evidence, manifest_entry, save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.registry import resolve_game

    if args.ab:
        from harness.ceiling import ab_report

        base, treat, target = (load_evidence(x) for x in args.ab)
        print(json.dumps(ab_report(base, treat, target, alpha=args.alpha,
                                   treatment_keys=tuple(args.treatment_keys.split(",")),
                                   manipulation=args.manipulation), indent=1, default=str))
        return
    if not args.out:
        raise SystemExit("--out is required unless --ab is given")
    if args.reanalyze_sims and not args.reanalyze_frac:
        raise SystemExit("--reanalyze-sims needs --reanalyze-frac > 0 — otherwise nothing is relabelled and the arm "
                         "would record a treatment it never received")
    game = resolve_game(args.game)
    out = Path(args.out)
    if args.control_only:
        evidence = load_evidence(out)
        replaces = (manifest_entry(out) or {}).get("sha256")
        _attach_control(game, evidence)
        save_evidence(out, evidence, replaces=replaces)
        _report(evidence)
        return
    from harness.neural import Connect4Net, arch_for_game

    arch = ARCHS[args.arch]
    built = sum(p.numel() for p in Connect4Net(**arch_for_game(arch, game)).parameters())
    if built != PARAMS_EXPECTED[args.arch]:
        raise SystemExit(f"the {args.arch} preset builds {built} parameters, not the registered "
                         f"{PARAMS_EXPECTED[args.arch]} — the preset drifted")
    if (args.steps_matched or args.sibling_holdout) and not args.reanalyze_siblings:
        raise SystemExit("--steps-matched / --sibling-holdout act only on siblings — add --reanalyze-siblings")
    holdout = None
    if args.sibling_holdout:
        mod, salt = args.sibling_holdout.split(":", 1)
        holdout = {"mod": int(mod), "salt": salt}
    base = {"game": args.game, "iterations": args.iterations, "selfplay": args.selfplay,
            "train_sims": args.train_sims, "eval_sims": args.eval_sims, "deep_sims": args.deep_sims,
            "arch": arch,
            "params_expected": PARAMS_EXPECTED[args.arch],
            "threads": args.threads, "opening_plies": args.opening_plies,
            "opening_zero_frac": args.opening_zero_frac, "reanalyze_frac": args.reanalyze_frac,
            "reanalyze_sims": args.reanalyze_sims or None, "reanalyze_siblings": args.reanalyze_siblings,
            "sibling_key": "canonical" if args.reanalyze_siblings else None, "sibling_holdout": holdout,
            "steps_matched": args.steps_matched, "policy_target": args.policy_target or None, **TRAIN_DEFAULTS,
            **({"save_nets": args.save_nets} if args.save_nets else {})}
    if args.target:
        tgt = load_evidence(args.target)
        base.update(target_evidence=args.target, label_draws=args.label_draws, label_sims=args.label_sims,
                    target_keys=sorted({f["key"] for s in tgt["seeds"] for f in s["failures"]}))
    seeds = _parse_seeds(args.seeds)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamps = (training_fingerprint(args.game), training_fingerprint(modules=MEASUREMENT_MODULES))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = []
        for r in ex.map(run_seed, [{**base, "seed": s} for s in seeds]):
            print(f"seed {r['seed']:>3}: coverage {r['coverage']:.4f}  policy-only {r['policy_coverage']:.4f}  "
                  f"failures {len(r['failures'])}  [{r['train_seconds']:.0f}s train]", flush=True)
            results.append(r)
    if (training_fingerprint(args.game), training_fingerprint(modules=MEASUREMENT_MODULES)) != stamps:
        raise SystemExit("training or measurement code changed on disk while the seeds ran — the workers ran "
                         "code that no stamp describes, so this evidence is not written; rerun the arm")
    states, _ = reachable_states(game, exact=True, symmetry=True)
    universe = sorted(failable_keys(game, states))
    keep = set(universe)
    import platform

    import numpy
    import torch

    evidence = {"game": args.game, "started": started,
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "numpy": numpy.__version__, "platform": platform.platform()},
                "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "training_fingerprint": stamps[0], "measurement_fingerprint": stamps[1],
                **({"oracle_fingerprint": training_fingerprint(modules=ORACLE_MODULES)} if args.policy_target else {}),
                "config": {**base, "seeds": seeds},
                "universe": universe, "optimal_play_keys": sorted(optimal_play_keys(game)),
                "plies": [[game.canonical_key(s), game.ply(s)] for s in states if game.canonical_key(s) in keep],
                "seeds": results}
    _attach_control(game, evidence)
    save_evidence(out, evidence)
    _report(evidence)


if __name__ == "__main__":
    main()
