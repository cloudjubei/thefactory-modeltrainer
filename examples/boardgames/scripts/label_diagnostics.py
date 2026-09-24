"""§C.45 — WHICH HALF OF THE SELF-PLAY LABEL IS WRONG at the blind-spot states? Retrain the seeds of an evidence
file (deterministically, and refused if a retrained net does not reproduce the failures that file recorded), then
split the self-play target at every failable state into label_ok / prior_anchor / search_miss (harness.targets)
under the arm's own training search, and at the target states under two counterfactual searches: deeper
(more sims) and more Q-trusting (higher c_scale). Non-target failable states are the control.

    PYTHONPATH=. .venv/bin/python scripts/label_diagnostics.py --evidence evidence/tictactoe_base.json \\
        --target evidence/tictactoe_ceiling.json --out evidence/tictactoe_labels.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

CLASSES = ("label_ok", "prior_anchor", "search_miss")


def diagnose_seed(job: dict) -> dict:
    import torch

    torch.set_num_threads(job["threads"])
    from harness.coverage import coverage_failures, failable_keys, optimal_actions, per_state_act, reachable_states
    from harness.neural import AlphaZeroAgent, legacy_arch_as_built, train_alphazero
    from harness.registry import resolve_game
    from harness.targets import search_decomposition, target_error

    cfg = {**job["config"], "arch": legacy_arch_as_built(job["config"]["arch"])}
    game = resolve_game(cfg["game"])
    net, _history, _buffer = train_alphazero(
        game, iterations=cfg["iterations"], selfplay_games=cfg["selfplay"], sims=cfg["train_sims"],
        channels=cfg["arch"]["channels"], net_arch=cfg["arch"], augment=True, gumbel=True, seed=job["seed"],
        selfplay_opening_plies=cfg.get("opening_plies", 0), opening_plies_zero_frac=cfg.get("opening_zero_frac", 0.0),
        return_buffer=True)
    states, _ = reachable_states(game, exact=True, symmetry=True)
    by_key = {game.canonical_key(s): s for s in states}
    evaluate = per_state_act(game, lambda: AlphaZeroAgent(net, sims=cfg["eval_sims"], solve_endgame=0, gumbel=True,
                                                          c_scale=0.1))
    reproduced = sorted(f["key"] for f in coverage_failures(game, evaluate, states))
    if reproduced != sorted(job["recorded_failures"]):
        return {"seed": job["seed"], "reproduced": False, "recorded": sorted(job["recorded_failures"]),
                "retrained": reproduced}
    target = set(job["target"])
    rows = []
    for name, sims, c_scale, only_target in job["searches"]:
        for k in sorted(failable_keys(game, states)):
            if only_target and k not in target:
                continue
            st = by_key[k]
            opt = optimal_actions(game, st)
            classes = dict.fromkeys(CLASSES, 0)
            for t, seeds in ((0.0, [0]), (1.0, list(range(job["noise_draws"])))):
                for s in seeds:
                    d = search_decomposition(game, lambda: AlphaZeroAgent(net, sims=sims, gumbel=True, c_scale=c_scale),
                                             st, seed=s, temperature=t)
                    classes[target_error(d, opt)] += 1
            rows.append({"key": k, "target": k in target, "search": name, "classes": classes})
    return {"seed": job["seed"], "reproduced": True, "rows": rows}


def untrained_ladder(cfg: dict, target: list, budgets: list, n_nets: int, noise_draws: int) -> dict:
    """label_ok at the target states for UNTRAINED nets at each budget — what search alone labels right with no
    learning at all. A relabel budget whose search-alone rate is already high is brute force, not a transferable
    recipe."""
    import torch

    from harness.coverage import optimal_actions, reachable_states
    from harness.neural import AlphaZeroAgent, Connect4Net, arch_for_game, legacy_arch_as_built
    from harness.registry import resolve_game
    from harness.targets import search_decomposition, target_error

    game = resolve_game(cfg["game"])
    cfg = {**cfg, "arch": legacy_arch_as_built(cfg["arch"])}
    states, _ = reachable_states(game, exact=True, symmetry=True)
    by_key = {game.canonical_key(s): s for s in states}
    out = {}
    for b in budgets:
        ok = total = 0
        for seed in range(n_nets):
            torch.manual_seed(seed)
            blank = Connect4Net(**arch_for_game(cfg["arch"], game))
            for k in target:
                opt = optimal_actions(game, by_key[k])
                for t, draws in ((0.0, [0]), (1.0, list(range(noise_draws)))):
                    for d in draws:
                        dec = search_decomposition(game, lambda: AlphaZeroAgent(blank, sims=b, gumbel=True, c_scale=0.1),
                                                   by_key[k], seed=d, temperature=t)
                        ok += target_error(dec, opt) == "label_ok"
                        total += 1
        out[str(b)] = ok / total
    return out


def summarise(rows: list) -> dict:
    out: dict = {}
    for r in rows:
        g = out.setdefault(f"{r['search']}|{'target' if r['target'] else 'other'}", dict.fromkeys(CLASSES, 0))
        for c in CLASSES:
            g[c] += r["classes"][c]
    return {k: {**v, "n": sum(v.values()), **{f"{c}_rate": v[c] / max(1, sum(v.values())) for c in CLASSES}}
            for k, v in sorted(out.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence", required=True, help="the arm whose seeds are retrained and diagnosed")
    ap.add_argument("--target", required=True, help="the evidence the target set was localized from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deep-sims", default="200", help="comma-separated relabel budgets to scan at the target states")
    ap.add_argument("--untrained-seeds", type=int, default=3,
                    help="also scan the same budgets with this many UNTRAINED nets (the search-alone control)")
    ap.add_argument("--trusting-c-scale", type=float, default=1.0)
    ap.add_argument("--noise-draws", type=int, default=8)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    ev = json.loads(Path(args.evidence).read_text())
    tgt = json.loads(Path(args.target).read_text())
    target = sorted({f["key"] for s in tgt["seeds"] for f in s["failures"]})
    cfg = ev["config"]
    budgets = [int(x) for x in args.deep_sims.split(",")]
    searches = ([("train", cfg["train_sims"], 0.1, False)] + [(f"deep{b}", b, 0.1, True) for b in budgets]
                + [("trusting", cfg["train_sims"], args.trusting_c_scale, True)])
    jobs = [{"config": cfg, "seed": s["seed"], "recorded_failures": [f["key"] for f in s["failures"]],
             "target": target, "searches": searches, "noise_draws": args.noise_draws, "threads": args.threads}
            for s in ev["seeds"]]
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(diagnose_seed, jobs))
    bad = [r for r in results if not r["reproduced"]]
    if bad:
        raise SystemExit(f"retraining did not reproduce the recorded failures for seeds {[r['seed'] for r in bad]} "
                         f"— a diagnostic of a different net says nothing about the one that failed: {bad[:1]}")
    rows = [dict(r, seed=res["seed"]) for res in results for r in res["rows"]]
    untrained = untrained_ladder(cfg, target, [cfg["train_sims"]] + budgets, args.untrained_seeds, args.noise_draws)
    out = {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "evidence": args.evidence, "target_evidence": args.target, "training_fingerprint": ev["training_fingerprint"],
           "searches": [{"name": n, "sims": s, "c_scale": c, "target_only": o} for n, s, c, o in searches],
           "noise_draws": args.noise_draws, "target": target, "rows": rows, "summary": summarise(rows),
           "untrained": untrained}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    print("untrained (search-alone) label_ok by budget:", json.dumps(untrained))


if __name__ == "__main__":
    main()
