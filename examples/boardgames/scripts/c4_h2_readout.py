"""§C.49 H2 — read the base and table arms' nets one ply past the exact opening table. The table
(harness.opening_table.full_table, the same one the table arm trained beside) fixes the first player's opening, so
every net faces the same positions one ply past it: each net's raw move there is scored against exact values, and the
hybrid (table + net) is certified through horizon + 4 plies (P-START). Exact values come from the label cache where
it records them and from the native solver otherwise. Judged by harness.floor_h2.h2_report.

    PYTHONPATH=. .venv/bin/python scripts/c4_h2_readout.py --workers 8 --out evidence/c49_H2_readout.json.gz
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "books" / "c4_labels.json.gz"
MEASUREMENT_MODULES = ("scripts/c4_h2_readout.py", "harness/opening_table.py", "harness/exact_values.py",
                       "harness/certify.py", "harness/c4_oracle.py", "harness/native_solver.py")
EMPTY_BOARD_VALUE = 1


def main() -> None:
    import hashlib
    import platform
    import random

    import torch

    from games.connect4 import Connect4
    from harness.certify import certify
    from harness.evidence import load_evidence, save_evidence
    from harness.exact_values import ExactValues
    from harness.fingerprint import training_fingerprint
    from harness.floor_h2 import SPEC
    from harness.neural import load_net
    from harness.opening_table import full_table, hybrid_chooser
    from harness.strategy_tree import raw_chooser

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    game = Connect4()
    root = game.initial_state(random.Random(0))
    runs, nets = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        exact = ExactValues(game, pool, load_evidence(LABELS)["positions"])
        table, frontier = full_table(game, root, 0, exact.move_values, SPEC["horizon"])
        carried = list({game.state_key(c): c for s in frontier for b in game.legal_actions(s)
                        for c in [game.step(s, b)] if not game.is_terminal(c)}.values())
        carried_values = exact.move_values(carried)
        print(f"table {len(table)} entries, frontier {len(frontier)}, carried positions {len(carried)}", flush=True)
        for arm in SPEC["arms"]:
            trained = load_evidence(ROOT / "evidence" / f"{SPEC['prefix']}_{arm}.json.gz")
            runs[f"{SPEC['prefix']}/{arm}"] = {"training_fingerprint": trained["training_fingerprint"],
                                               "config": trained["config"]}
            for row in sorted(trained["seeds"], key=lambda r: r["seed"]):
                path = ROOT / row["net"]["path"]
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                if sha != row["net"]["file_sha256"]:
                    raise SystemExit(f"{arm} seed {row['seed']}: the saved net is not the one its run recorded")
                choose = raw_chooser(game, load_net(str(path)))
                moves = choose(carried)
                optimal = sum(1 for m, v in zip(moves, carried_values) if v[m] == max(v.values()))
                cert = certify(game, root, 0, hybrid_chooser(game, table, choose), lambda s: exact.positions([s])[0],
                               max_depth=SPEC["horizon"] + 4, value_many=exact.positions,
                               root_value=EMPTY_BOARD_VALUE)
                nets.append({"arm": arm, "seed": row["seed"], "net_file_sha256": sha,
                             "carried": {"positions": len(carried), "optimal": optimal},
                             "certificate": {"depth": SPEC["horizon"] + 4, "certified": cert["certified"],
                                             "failures": cert["failures"],
                                             "failures_by_ply": {str(p): c for p, c in cert["failures_by_ply"].items()},
                                             "nodes": cert["nodes"]}})
                print(f"{arm} seed {row['seed']}: optimal at {optimal}/{len(carried)} carried positions; hybrid "
                      f"certified through {SPEC['horizon'] + 4} = {cert['certified']} {cert['failures_by_ply']}; "
                      f"values {exact.counts}", flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the readout ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version(),
                                                                            "torch": torch.__version__},
                             "horizon": SPEC["horizon"], "table_entries": len(table), "frontier": len(frontier),
                             "carried_positions": len(carried), "values": exact.counts, "runs": runs, "nets": nets})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
