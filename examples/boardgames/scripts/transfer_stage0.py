"""§C.47 leg C, Stage 0 (design v2, harness.stage0_v2) — train the Connect-4 C_R nets (harness.stage0.C0_RECIPE) with
the self-play recorder on and the exact solver forbidden, then probe each net at non-trivial positions classed by
where they sit relative to its own training data. Writes the evidence harness.stage0_v2.stage0_v2_report judges.

    PYTHONPATH=. .venv/bin/python scripts/transfer_stage0.py --seeds 206-210 --workers 5 \\
        --parts checkpoints/c47_C0v2 --out evidence/c47_C0v2.json.gz

Decision cells, per ply in C0_PLIES: VISITED — a census of every position the net trained on, each flagged with
whether it is still in the final relabel buffer — and SIBLING — exactly `neural.one_ply_siblings` of the final
buffer, the positions a sibling arm would add at its last pass. Descriptive cells: the all-visited one- and two-move
rings, and random-play positions outside all of those. (v1, whose evidence is evidence/c47_C0.json.gz, used a
final-buffer census and was refused by its integrity gate: too few positions per cell.) Per position: the raw policy's move, the mirror-averaged policy's move, and the label the
recipe's own relabel search (64 sims from the final net, `reanalyze_examples`) gives it, each graded against the
exact optimal set. Also descriptive: proven-win conversion by the raw and averaged policies against the exact
defender, from non-trivial random roots and from the net's own buffer, each first error classed by distance.

Each net's training (net, recorded states, history) and each finished row are saved under --parts as they complete,
stamped with the code that produced them; a rerun resumes from parts whose stamps still match, so a crash in the
measurement never forces the hour of training to be repeated. The evidence's `started` is the EARLIEST training
start among the parts it was assembled from, so resuming can never make data look younger than it is."""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

MEASUREMENT_MODULES = ("scripts/transfer_stage0.py", "harness/transfer.py", "harness/coverage.py")
MAX_DRAWS = 200000


def _parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def random_position(game, rng: random.Random, ply: int):
    """A position reached by uniformly random play at exactly `ply`, restarting whenever a game ends first."""
    while True:
        s = game.initial_state(rng)
        while game.ply(s) < ply and not game.is_terminal(s):
            s = game.step(s, rng.choice(game.legal_actions(s)))
        if not game.is_terminal(s):
            return s


def _unique(game, states) -> list:
    return list({game.canonical_key(s): s for s in states}.values())


def _sample(game, candidates: list, n: int, rng: random.Random, exact) -> list:
    """Up to `n` non-trivial positions from `candidates`, in a seeded random order, with their exact values."""
    from harness.transfer import nontrivial

    order = list(candidates)
    rng.shuffle(order)
    out = []
    for s in order:
        vals = exact(s)
        if nontrivial(game, s, vals):
            out.append((s, vals))
            if len(out) == n:
                break
    return out


def _train_part(job: dict, game, stamps: tuple) -> dict:
    """Train one net, or load the part a previous run saved under the same stamps."""
    import torch

    from harness.neural import Connect4Net, arch_for_game, encode, train_alphazero
    from harness.solver import move_values
    from harness.transfer import build_probe, forbid_solver, nontrivial, record_selfplay_states

    recipe, seed, sampling = job["recipe"], job["seed"], job["sampling"]
    path = Path(job["parts"]) / f"seed{seed}.train.pt"
    if path.exists():
        part = torch.load(path, weights_only=False)
        if tuple(part["stamps"]) == stamps and part["recipe"] == recipe:
            net = Connect4Net(**arch_for_game(recipe["arch"], game))
            net.load_state_dict(part["state_dict"])
            return {**part, "net": net}
    prng = random.Random(sampling["probe_seed"])
    probe_states = []
    for ply in job["plies"]:
        while sum(1 for s in probe_states if game.ply(s) == ply) < sampling["probe_per_ply"]:
            s = random_position(game, prng, ply)
            if nontrivial(game, s, move_values(s, weak=True)):
                probe_states.append(s)
    probe = build_probe(game, probe_states, encode, lambda s: move_values(s, weak=True))
    trained_from = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()
    with record_selfplay_states(game, probe) as log, forbid_solver():
        net, history = train_alphazero(
            game, iterations=recipe["iterations"], selfplay_games=recipe["selfplay_games"], sims=recipe["sims"],
            channels=recipe["arch"]["channels"], net_arch=recipe["arch"], augment=recipe["augment"],
            gumbel=recipe["gumbel"], c_scale=recipe["c_scale"], seed=seed,
            selfplay_opening_plies=recipe["opening_plies"], opening_plies_zero_frac=recipe["opening_zero_frac"],
            reanalyze_frac=recipe["reanalyze_frac"], reanalyze_sims=recipe["reanalyze_sims"], epochs=recipe["epochs"],
            batch_size=recipe["batch_size"], lr=recipe["lr"], buffer_cap=recipe["buffer_cap"])
    per_pass = [0] * recipe["iterations"]
    for g in log["games"]:
        per_pass[g["pass"]] += 1
    part = {"stamps": list(stamps), "recipe": recipe, "state_dict": net.state_dict(), "history": history,
            "states": [s for g in log["games"] for s in g["states"]], "games_per_pass": per_pass,
            "probe": log["probe"], "train_seconds": round(time.time() - t0, 1), "solver_forbidden": True,
            "started": trained_from}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(part, path)
    return {**part, "net": net}


def run_net(job: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    from harness import solver
    from harness.coverage import _averaged_move
    from harness.evidence import load_evidence, save_evidence
    from harness.neural import AlphaZeroAgent, encode, one_ply_siblings, reanalyze_examples
    from harness.registry import resolve_game
    from harness.solver import OracleAgent
    from harness.symmetry import verified_isometries
    from harness.targets import _weights_sha
    from harness.transfer import first_error, nontrivial, rings_at

    recipe, seed, sampling = job["recipe"], job["seed"], job["sampling"]
    stamps = tuple(job["stamps"])
    row_path = Path(job["parts"]) / f"seed{seed}.row.json.gz"
    if row_path.exists():
        saved = load_evidence(row_path)
        if tuple(saved["stamps"]) == stamps and saved["recipe"] == recipe and saved["sampling"] == sampling \
                and saved["plies"] == job["plies"]:
            return saved["row"]
    game = resolve_game(recipe["game"])
    part = _train_part(job, game, stamps)
    net = part["net"]
    net.eval()

    cache: dict = {}

    def exact(s):
        """Exact move values, cached by the position ITSELF: a mirror image shares the canonical key but has its
        columns reversed, so a canonical cache would grade a position against its mirror's optimal moves."""
        k = game.state_key(s)
        if k not in cache:
            cache[k] = solver.move_values(s, weak=True)
        return cache[k]

    key = game.canonical_key
    states = part["states"]
    final = states[-recipe["buffer_cap"]:]
    visited_keys, final_keys = {key(s) for s in states}, {key(s) for s in final}
    unique_visited = _unique(game, states)
    siblings, _stats = one_ply_siblings(game, _unique(game, final), key)

    isos = verified_isometries(game)

    def logits_fn(s):
        with torch.no_grad():
            lg, _v = net(encode(game, s).unsqueeze(0))
        return [float(x) for x in lg[0]]

    def raw_move(s):
        lg = logits_fn(s)
        return max(game.legal_actions(s), key=lambda a: lg[a])

    def avg_move(s):
        return _averaged_move(game, s, logits_fn, [i for i, _f in isos], [f for _i, f in isos])

    labeller = AlphaZeroAgent(net, sims=recipe["label_sims"], gumbel=recipe["gumbel"], c_scale=recipe["c_scale"],
                              solve_endgame=0)
    rings_cache: dict = {}

    def rings(ply):
        if ply not in rings_cache:
            r = rings_at(game, unique_visited, ply)
            rings_cache[ply] = {"one": set(r["one_move_off"]), "two": set(r["two_moves_off"]), "states": r}
        return rings_cache[ply]

    def distance(s):
        k = key(s)
        if k in final_keys:
            return "final_buffer"
        if k in visited_keys:
            return "evicted"
        r = rings(game.ply(s))
        return "one_move_off" if k in r["one"] else "two_moves_off" if k in r["two"] else "further"

    positions, containment = [], {}
    crng = random.Random(seed * 1009)
    for ply in job["plies"]:
        r = rings(ply)
        classes = {
            "visited": [s for s in unique_visited if game.ply(s) == ply],
            "sibling": [s for s in siblings if game.ply(s) == ply],
            "one_move_off": list(r["states"]["one_move_off"].values()),
            "two_moves_off": list(r["states"]["two_moves_off"].values()),
        }
        taken = {cls: _sample(game, cands, sampling[cls], crng, exact) for cls, cands in classes.items()}
        rand, draws, inside = [], 0, 0
        while len(rand) < sampling["random"] and draws < MAX_DRAWS:
            s = random_position(game, crng, ply)
            draws += 1
            k = key(s)
            if k in visited_keys or k in r["one"] or k in r["two"]:
                inside += 1
                continue
            vals = exact(s)
            if nontrivial(game, s, vals):
                rand.append((s, vals))
        taken["random"] = rand
        containment[str(ply)] = {"draws": draws, "in_visited_or_rings": inside,
                                 "candidates": {cls: len(c) for cls, c in classes.items()}}
        for cls, picked in taken.items():
            for i, (s, vals) in enumerate(picked):
                best = max(vals.values())
                optimal = {a for a, v in vals.items() if v == best}
                (_x, pi, _v), = reanalyze_examples(game, labeller, [s], random.Random(seed * 100003 + ply * 1000 + i))
                label = max(range(len(pi)), key=lambda a: pi[a])
                positions.append({"cls": cls, "ply": ply, "key": key(s), "raw_ok": raw_move(s) in optimal,
                                  "avg_ok": avg_move(s) in optimal, "label_ok": label in optimal,
                                  "label_mass": sum(pi[a] for a in optimal),
                                  **({"in_final": key(s) in final_keys} if cls == "visited" else {})})

    conversion = []
    for ply in job["plies"]:
        rrng = random.Random(1000 + ply)
        roots, seen = [], set()
        while len(roots) < sampling["conversion_roots"]:
            s = random_position(game, rrng, ply)
            vals = exact(s)
            if key(s) not in seen and max(vals.values()) > 0 and nontrivial(game, s, vals):
                seen.add(key(s))
                roots.append(("random", s))
        own = _unique(game, [s for s in final if game.ply(s) == ply])
        random.Random(seed + ply).shuffle(own)
        picked = 0
        for s in own:
            if picked == sampling["conversion_roots"]:
                break
            vals = exact(s)
            if max(vals.values()) > 0 and nontrivial(game, s, vals):
                roots.append(("own", s))
                picked += 1
        for kind, root in roots:
            for policy, act in (("raw", raw_move), ("avg", avg_move)):
                res = first_error(game, act, root, OracleAgent(), values_fn=exact)
                conversion.append({"kind": kind, "ply": ply, "policy": policy, "converted": res["converted"],
                                   "plies": res["plies"],
                                   **({} if res["converted"] else {"error_class": distance(res["state"]),
                                                                  "error_ply": game.ply(res["state"])})})
    row = {"seed": seed, "started": part["started"], "params": sum(p.numel() for p in net.parameters()),
           "weights_sha": _weights_sha(net),
           "train_seconds": part["train_seconds"], "history": part["history"], "recorded_states": len(states),
           "games_per_pass": part["games_per_pass"], "solver_forbidden": part["solver_forbidden"],
           "probe": part["probe"],
           "sizes": {"visited": len(states), "visited_unique": len(unique_visited), "final_buffer": len(final),
                     "siblings": len(siblings)},
           "containment": containment, "positions": positions, "conversion": conversion}
    save_evidence(row_path, {"stamps": list(stamps), "recipe": recipe, "sampling": sampling, "plies": job["plies"],
                             "row": row})
    return row


def main() -> None:
    import platform

    import numpy
    import torch

    from harness.evidence import save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.stage0 import C0_PLIES, C0_RECIPE
    from harness.stage0_v2 import C0V2_SAMPLING, stage0_v2_report

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="206-210")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--parts", required=True, help="directory for each net's saved training and finished row")
    ap.add_argument("--override", default="{}",
                    help="SMOKE TESTS ONLY: a JSON object over the recipe and sampling keys (the report refuses it)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    override = json.loads(args.override)
    recipe = {**C0_RECIPE, **{k: v for k, v in override.items() if k in C0_RECIPE}}
    sampling = {**C0V2_SAMPLING, **{k: v for k, v in override.items() if k in C0V2_SAMPLING}}
    unknown = set(override) - set(C0_RECIPE) - set(C0V2_SAMPLING)
    if unknown:
        raise SystemExit(f"--override names unknown keys {sorted(unknown)}")
    seeds = _parse_seeds(args.seeds)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamps = (training_fingerprint(recipe["game"]), training_fingerprint(modules=MEASUREMENT_MODULES))
    jobs = [{"recipe": recipe, "sampling": sampling, "seed": s, "plies": list(C0_PLIES), "parts": args.parts,
             "stamps": list(stamps)} for s in seeds]
    rows, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_net, j): j["seed"] for j in jobs}
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as e:
                failed.append(futures[fut])
                print(f"net {futures[fut]} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            print(f"net {r['seed']}: {r['params']} params, {r['train_seconds']:.0f}s train, {r['recorded_states']} "
                  f"states, {len(r['positions'])} positions, probe {[round(a, 3) for a in r['probe']]}", flush=True)
            rows.append(r)
    if failed:
        raise SystemExit(f"nets {sorted(failed)} failed — their finished parts are kept under {args.parts}; rerun "
                         f"to resume")
    if (training_fingerprint(recipe["game"]), training_fingerprint(modules=MEASUREMENT_MODULES)) != stamps:
        raise SystemExit("training or measurement code changed on disk while the nets ran — this evidence is not "
                         "written; rerun")
    evidence = {"game": recipe["game"], "started": min([started] + [r["started"] for r in rows]),
                "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "training_fingerprint": stamps[0], "measurement_fingerprint": stamps[1],
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "numpy": numpy.__version__, "platform": platform.platform()},
                "config": {**recipe, "sampling": sampling, "plies": list(C0_PLIES), "design": "v2", "seeds": seeds},
                "seeds": sorted(rows, key=lambda r: r["seed"])}
    save_evidence(args.out, evidence)
    report = stage0_v2_report(evidence)
    print(report["C0"].get("reading") or report["integrity"], flush=True)


if __name__ == "__main__":
    main()
