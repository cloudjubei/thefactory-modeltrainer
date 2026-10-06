"""§3.6 S1 input — every first-player position of the canonical exact Connect-4 table's own tree at the given plies
(harness.opening_table.full_table: the first optimal move in column order, the table §C.49's depth study priced),
saved once so later experiments sample the same frontier without re-solving it. Each position carries its exact
move values.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_frontier_positions.py --plies 10 12 --workers 8 \\
        --out evidence/c49_frontier_positions.json.gz
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "books" / "c4_labels.json.gz"
MEASUREMENT_MODULES = ("scripts/c4_frontier_positions.py", "harness/opening_table.py", "harness/exact_values.py",
                       "harness/c4_oracle.py", "harness/native_solver.py")


def main() -> None:
    import platform
    import random

    from games.connect4 import Connect4
    from harness.evidence import load_evidence, save_evidence
    from harness.exact_values import ExactValues
    from harness.fingerprint import training_fingerprint
    from harness.opening_table import full_table

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plies", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()
    game = Connect4()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        exact = ExactValues(game, pool, load_evidence(LABELS)["positions"])

        def move_values(states):
            out = exact.move_values(states)
            print(f"  {len(states)} positions; values {exact.counts}; {time.time() - t0:.0f}s", flush=True)
            return out

        table, _frontier = full_table(game, game.initial_state(random.Random(0)), 0, move_values, max(args.plies) + 1)
        plies: dict = {}
        for (board, to_move), _move in table.items():
            ply = sum(1 for v in board if v)
            if ply in args.plies:
                state = Connect4().initial_state(random.Random(0)).__class__(tuple(board), to_move, None, False)
                plies.setdefault(str(ply), []).append({"board": list(board), "to_move": to_move,
                                                       "values": {str(a): v for a, v in
                                                                  exact.move_values([state])[0].items()}})
    for ply in sorted(plies, key=int):
        print(f"ply {ply}: {len(plies[ply])} positions", flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "table": "full_table, first optimal move in column order", "plies": plies})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
