"""SUPERSEDED by scripts/c4_steady_leaf_probe.py (h170) — kept as the record of how its evidence was made; the
harness.steady_local and harness.steady_exceptions APIs it calls have since changed.

§3.6 option 1 — local search (harness.steady_local) on the eight ply-10 positions the S1 pilot left out of budget,
under the SAT diagnostic's conditions (8 levels, 30 minutes each; scripts/c4_steady_budget_probe.py gave SAT 1 of 8).
A found map is simplified and checked by the solver-free walk and harness.certify, as in S1.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_steady_local_probe.py --workers 4 \\
        --out evidence/c49_s1_local_probe.json.gz
"""
from __future__ import annotations

import argparse
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_MODULES = ("scripts/c4_steady_local_probe.py", "scripts/c4_steady_states.py", "harness/steady_local.py",
                       "harness/steady_state.py", "harness/certify.py", "harness/c4_oracle.py",
                       "harness/native_solver.py")
N_LEVELS, SECONDS, CAP = 8, 1800.0, 1_000_000


def probe(job: dict) -> dict:
    import random
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from c4_steady_states import _value

    from games.connect4 import C4State, Connect4
    from harness.certify import certify
    from harness.steady_local import local_search
    from harness.steady_state import Facts, choose, map_bits, simplify, verify

    game = Connect4()
    root = C4State(tuple(job["board"]), job["to_move"], None, False)
    facts = Facts(game)
    t0 = time.time()
    found = local_search(facts, root, N_LEVELS, random.Random(job["index"]), SECONDS, CAP)
    row = {"index": job["index"], "status": found["status"], "best_failures": found["best_failures"],
           "evaluations": found["evaluations"], "search_seconds": round(time.time() - t0, 1)}
    if found["status"] != "found":
        return row
    levels = simplify(facts, root, found["levels"], N_LEVELS, CAP)
    walk = verify(facts, root, levels, N_LEVELS, CAP)
    cert = certify(game, root, 0, lambda xs: [choose(facts, x, levels, N_LEVELS) for x in xs],
                   lambda x: _value(game, x), root_value=1)
    return {**row, "bits": map_bits(levels, len(facts.empty_cells(root)), 3), "own_positions": walk["own_positions"],
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
    pilot = load_evidence(ROOT / "evidence" / "c49_s1_pilot.json.gz")["results"]
    jobs = [{"index": r["index"], "board": r["board"], "to_move": r["to_move"]}
            for r in pilot if r["ply"] == 10 and r["status"] == "budget"]
    print(f"{len(jobs)} positions", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(probe, j) for j in jobs]):
            rows.append(fut.result())
            print(rows[-1], flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the probe ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "settings": {"n_levels": N_LEVELS, "seconds": SECONDS, "cap": CAP},
                             "rows": sorted(rows, key=lambda r: r["index"])})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
