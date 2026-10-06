"""§3.6 option 1, second step — SAT hand-off: the ply-10 positions where size-scored local search ended with exceptions
(evidence/c49_s1_leaf_probe.json.gz) are searched by SAT (harness.steady_search, 8 levels) twice under one budget:
"hinted" starts the solver at local search's best map and constrains the positions where that map needed exceptions
before the first solve; "cold" is the same search without them. A found map is simplified and checked by the
solver-free walk and harness.certify, as in S1.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_steady_handoff_probe.py --workers 4 \\
        --out evidence/c49_s1_handoff_probe.json.gz
"""
from __future__ import annotations

import argparse
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_MODULES = ("scripts/c4_steady_handoff_probe.py", "scripts/c4_steady_states.py", "harness/steady_search.py",
                       "harness/steady_state.py", "harness/certify.py", "harness/c4_oracle.py",
                       "harness/native_solver.py")
N_LEVELS, LEVEL_BITS = 8, 3
BUDGET = {"max_constraints": 200_000, "conflicts": 1_000_000, "seconds": 1800.0, "cap": 1_000_000, "lines": 64}
ARMS = ("hinted", "cold")


def probe(job: dict) -> dict:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from c4_steady_states import _value, _winning

    from games.connect4 import C4State, Connect4
    from harness.certify import certify
    from harness.steady_search import find_steady_state
    from harness.steady_state import Facts, choose, map_bits, simplify, verify

    game = Connect4()
    root = C4State(tuple(job["board"]), job["to_move"], None, False)
    facts = Facts(game)
    hinted = job["arm"] == "hinted"
    hint = {int(c): k for c, k in job["levels"].items()} if hinted else None
    seeds = [C4State(tuple(b), t, None, False) for b, t, _move in job["exceptions"]] if hinted else ()
    t0 = time.time()
    found = find_steady_state(facts, root, lambda states: _winning(game, states), N_LEVELS, **BUDGET,
                              hint=hint, seeds=seeds)
    row = {"index": job["index"], "arm": job["arm"], "status": found["status"], "seeds": len(seeds),
           "constraints": found["constraints"], "iterations": found["iterations"],
           "search_seconds": round(time.time() - t0, 1)}
    if found["status"] != "found":
        return row
    levels = simplify(facts, root, found["levels"], N_LEVELS, BUDGET["cap"])
    walk = verify(facts, root, levels, N_LEVELS, BUDGET["cap"])
    cert = certify(game, root, game.current_player(root), lambda xs: [choose(facts, x, levels, N_LEVELS) for x in xs],
                   lambda x: _value(game, x), root_value=1)
    return {**row, "levels": {str(c): k for c, k in sorted(levels.items())},
            "bits": map_bits(levels, len(facts.empty_cells(root)), LEVEL_BITS), "own_positions": walk["own_positions"],
            "verified": walk["won"], "certified": bool(cert["certified"] and cert["complete_game"])}


def main() -> None:
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pilot = {r["index"]: r for r in load_evidence(ROOT / "evidence" / "c49_s1_pilot.json.gz")["results"]}
    leaves = [r for r in load_evidence(ROOT / "evidence" / "c49_s1_leaf_probe.json.gz")["rows"] if r["exceptions"]]
    jobs = [{"index": r["index"], "board": pilot[r["index"]]["board"], "to_move": pilot[r["index"]]["to_move"],
             "levels": r["levels"], "exceptions": r["exceptions"], "arm": arm} for r in leaves for arm in ARMS]
    print(f"{len(leaves)} positions x {len(ARMS)} arms", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(probe, j) for j in jobs]):
            r = fut.result()
            rows.append(r)
            print({k: v for k, v in r.items() if k != "levels"}, flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the probe ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "settings": {"n_levels": N_LEVELS, "level_bits": LEVEL_BITS, "budget": BUDGET},
                             "rows": sorted(rows, key=lambda r: (r["index"], r["arm"]))})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
