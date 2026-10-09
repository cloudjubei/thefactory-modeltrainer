"""§3.6 S1 — search the canonical exact Connect-4 table's frontier for steady states. Samples the registered positions
(harness.floor_s1: per ply, a seeded sample of evidence/c49_frontier_positions.json.gz, a ply it does not hold derived
from the ply two before by harness.floor_s1.next_frontier) or, with `--pilot N`, N other
positions per ply never in that sample. Per position, in parallel: trivial when the rule with an empty map already
wins; otherwise harness.steady_search within the registered budget, the found map simplified, then checked twice — the
solver-free walk (harness.steady_state.verify) and harness.certify against exact values. The exact native solver is
the search's oracle and certify's values; the walk never sees it. Judged by harness.floor_s1.s1_report.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_steady_states.py --workers 8 --out evidence/c49_s1.json.gz \\
        --progress <scratch>/s1_progress.json [--pilot 10]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSITIONS = ROOT / "evidence" / "c49_frontier_positions.json.gz"
MEASUREMENT_MODULES = ("scripts/c4_steady_states.py", "harness/steady_state.py", "harness/steady_search.py",
                       "harness/certify.py", "harness/c4_oracle.py", "harness/native_solver.py")
_SOLVED: dict = {}
SOLVED_LIMIT = 3_000_000


def _value(game, state) -> int:
    from harness.c4_oracle import solve

    key = game.state_key(state)
    if key not in _SOLVED:
        if len(_SOLVED) >= SOLVED_LIMIT:
            _SOLVED.clear()
        _SOLVED[key] = int(solve((state.board, state.to_move)))
    return _SOLVED[key]


def _winning(game, states) -> list:
    out = []
    for s in states:
        me = game.current_player(s)
        moves = set()
        for a in game.legal_actions(s):
            child = game.step(s, a)
            if game.is_terminal(child):
                kept = round(game.returns(child)[me])
            else:
                own = _value(game, child)
                kept = own if game.current_player(child) == me else -own
            if kept == 1:
                moves.add(a)
        out.append(moves)
    return out


def search_one(job: dict) -> dict:
    from games.connect4 import C4State, Connect4
    from harness.certify import certify
    from harness.steady_search import find_steady_state
    from harness.steady_state import Facts, choose, map_bits, simplify, verify

    t0 = time.time()
    game = Connect4()
    root = C4State(tuple(job["board"]), job["to_move"], None, False)
    facts = Facts(game)
    n_levels, budget = job["n_levels"], job["budget"]
    row = {"ply": job["ply"], "index": job["index"], "board": job["board"], "to_move": job["to_move"],
           "empty": len(facts.empty_cells(root))}
    if verify(facts, root, {}, n_levels, budget["cap"])["won"]:
        return {**row, "trivial": True, "status": "trivial", "bits": None, "own_positions": None, "verified": None,
                "certified": None, "levels": None, "seconds": round(time.time() - t0, 2)}
    t_search = time.time()
    found = find_steady_state(facts, root, lambda states: _winning(game, states), n_levels, **budget)
    row.update({"trivial": False, "status": found["status"], "constraints": found["constraints"],
                "iterations": found["iterations"], "search_seconds": round(time.time() - t_search, 2)})
    if found["status"] != "found":
        return {**row, "bits": None, "own_positions": None, "verified": None, "certified": None, "levels": None,
                "seconds": round(time.time() - t0, 2)}
    levels = simplify(facts, root, found["levels"], n_levels, budget["cap"])
    walk = verify(facts, root, levels, n_levels, budget["cap"])
    cert = certify(game, root, game.current_player(root), lambda states: [choose(facts, s, levels, n_levels)
                                                                             for s in states],
                   lambda s: _value(game, s), root_value=1)
    return {**row, "levels": {str(c): k for c, k in sorted(levels.items())}, "levelled": len(levels),
            "bits": map_bits(levels, row["empty"], job["level_bits"]), "own_positions": walk["own_positions"],
            "positions": walk["positions"], "verified": walk["won"],
            "certified": bool(cert["certified"] and cert["complete_game"]), "seconds": round(time.time() - t0, 2)}


def main() -> None:
    import platform

    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    from games.connect4 import Connect4
    from harness.floor_s1 import SPEC, next_frontier, pilot_indices, sample_indices

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", required=True)
    ap.add_argument("--pilot", type=int, default=None, help="search this many positions per ply outside the sample")
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    positions = load_evidence(POSITIONS)
    config = {k: SPEC[k] for k in ("n_levels", "level_bits", "budget")}
    jobs = []
    for ply in SPEC["plies"]:
        rows = (positions["plies"][str(ply)] if str(ply) in positions["plies"]
                else next_frontier(Connect4(), positions["plies"][str(ply - 2)]))
        picked = (pilot_indices(len(rows), SPEC["per_ply"], SPEC["seed"], args.pilot) if args.pilot
                  else sample_indices(len(rows), SPEC["per_ply"], SPEC["seed"]))
        jobs += [{"ply": ply, "index": i, "board": rows[i]["board"], "to_move": rows[i]["to_move"], **config}
                 for i in picked]
    print(f"{len(jobs)} positions ({'pilot' if args.pilot else 'registered sample'})", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(search_one, j) for j in jobs]):
            r = fut.result()
            results.append(r)
            Path(args.progress).write_text(json.dumps({"results": results}))
            print(f"ply {r['ply']} #{r['index']}: {r['status']}"
                  + (f", {r['bits']} bits, {r['own_positions']} own positions, verified {r['verified']}, "
                     f"certified {r['certified']}" if r["status"] == "found" else "")
                  + f" [{r['seconds']}s] ({len(results)}/{len(jobs)})", flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "positions_fingerprint": positions["measurement_fingerprint"],
                             "versions": {"python": platform.python_version()}, "pilot": args.pilot,
                             "config": config, "results": sorted(results, key=lambda r: (r["ply"], r["index"]))})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
