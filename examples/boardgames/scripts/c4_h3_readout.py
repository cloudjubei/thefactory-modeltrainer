"""§C.49 H3 — read H2's table-arm nets and H3's deep-tree nets on the fixed position sets the exact opening table
makes the same for every net: s6 (the positions one ply past the table) and s8 (the ply-8 positions reached when the first
player also plays optimally at ply 6, every reply at ply 7). Each net's raw move is scored against exact values; each
net's description is sized as the hybrid certified through 9 plies would be — the opening table plus an exception
wherever the net's move at plies 6 and 8 of its own tree is not optimal (harness.opening_table), costed as
params x 32 bits + entries x (bits to index a first-player position of that tree + 3 bits for the move). H3's
per-pass ply-6 curve is copied from its run. Judged by harness.floor_h3.h3_report.

    PYTHONPATH=. .venv/bin/python scripts/c4_h3_readout.py --workers 8 --out evidence/c49_H3_readout.json.gz
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "books" / "c4_labels.json.gz"
MEASUREMENT_MODULES = ("scripts/c4_h3_readout.py", "harness/opening_table.py", "harness/exact_values.py",
                       "harness/c4_oracle.py", "harness/native_solver.py")
HORIZON = 5
CERTIFIED_THROUGH = 9


def main() -> None:
    import hashlib
    import platform
    import random

    import torch

    from games.connect4 import Connect4
    from harness.evidence import load_evidence, save_evidence
    from harness.exact_values import ExactValues
    from harness.fingerprint import training_fingerprint
    from harness.floor_h3 import SPEC
    from harness.neural import load_net
    from harness.opening_table import exception_table, full_table, hybrid_chooser
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
    key = game.state_key

    def unfinished_children(states):
        return list({key(c): c for s in states for b in game.legal_actions(s) for c in [game.step(s, b)]
                     if not game.is_terminal(c)}.values())

    runs, nets = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        exact = ExactValues(game, pool, load_evidence(LABELS)["positions"])
        table, frontier = full_table(game, root, 0, exact.move_values, HORIZON)
        s6 = unfinished_children(frontier)
        s6_values = exact.move_values(s6)
        after = [game.step(s, next(a for a in game.legal_actions(s) if v[a] == max(v.values())))
                 for s, v in zip(s6, s6_values)]
        s8 = unfinished_children([c for c in after if not game.is_terminal(c)])
        s8_values = exact.move_values(s8)
        print(f"table {len(table)}, s6 {len(s6)}, s8 {len(s8)}; values {exact.counts}", flush=True)
        for arm, run in SPEC["runs"].items():
            prefix, name = run.split("/")
            trained = load_evidence(ROOT / "evidence" / f"{prefix}_{name}.json.gz")
            runs[arm] = {"training_fingerprint": trained["training_fingerprint"], "config": trained["config"]}
            for row in sorted(trained["seeds"], key=lambda r: r["seed"]):
                path = ROOT / row["net"]["path"]
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                if sha != row["net"]["file_sha256"]:
                    raise SystemExit(f"{arm} seed {row['seed']}: the saved net is not the one its run recorded")
                net = load_net(str(path))
                choose = raw_chooser(game, net)
                scores = {}
                for name_, states, values in (("s6", s6, s6_values), ("s8", s8, s8_values)):
                    moves = choose(states)
                    scores[name_] = {"positions": len(states),
                                     "optimal": sum(1 for m, v in zip(moves, values) if v[m] == max(v.values()))}
                exceptions, stats = exception_table(game, root, 0, hybrid_chooser(game, table, choose),
                                                    exact.move_values, CERTIFIED_THROUGH)
                entries = len(table) + len(exceptions)
                params = sum(p.numel() for p in net.parameters())
                bits = params * 32 + entries * (math.ceil(math.log2(stats["positions"])) + 3)
                nets.append({"arm": arm, "seed": row["seed"], "net_file_sha256": sha, **scores,
                             "description": {"table": len(table), "exceptions": len(exceptions), "entries": entries,
                                             "positions": stats["positions"], "params": params, "bits": bits,
                                             "table_only_bits": stats["positions"]
                                             * (math.ceil(math.log2(stats["positions"])) + 3)},
                             **({"carried_curve": row["carried_curve"]} if "carried_curve" in row else {})})
                print(f"{arm} seed {row['seed']}: s6 {scores['s6']['optimal']}/{len(s6)}, s8 "
                      f"{scores['s8']['optimal']}/{len(s8)}; certified-through-9 hybrid {entries} entries over "
                      f"{stats['positions']} positions, {bits} bits; values {exact.counts}", flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the readout ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version(),
                                                                            "torch": torch.__version__},
                             "horizon": HORIZON, "table_entries": len(table), "s6_positions": len(s6),
                             "s8_positions": len(s8), "values": exact.counts, "runs": runs, "nets": nets})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
