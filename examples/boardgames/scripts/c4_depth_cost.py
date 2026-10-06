"""§C.49 after H3 — net cost against table cost as the certified depth grows. The canonical optimal table
(harness.opening_table.full_table: the first optimal move at every first-player position of its own tree) is built
through the deepest registered depth; each H3 net's minimal hybrid (harness.opening_table.exception_table: an exception
only where the raw net is not optimal) through 11 plies, the registered deep nets through 13, each certified
independently (harness.certify) through its depth. Recorded per first-player ply: table entries, each hybrid's
positions and exceptions. Stage one reaches 11 plies for the table and every net; stage two extends the table and the
deep nets to 13, re-walking the shallow plies from cached values. Each net's result is also written to `--progress` as
it lands, so a stopped run keeps what it finished. Judged by harness.depth_cost.depth_report.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_depth_cost.py --workers 8 --out evidence/c49_depth_cost.json.gz \\
        --progress <scratch>/depth_cost_progress.json
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "books" / "c4_labels.json.gz"
MEASUREMENT_MODULES = ("scripts/c4_depth_cost.py", "harness/opening_table.py", "harness/exact_values.py",
                       "harness/certify.py", "harness/c4_oracle.py", "harness/native_solver.py",
                       "harness/strategy_tree.py")


def _ply(key) -> int:
    return sum(1 for v in key[0] if v)


def _by_ply(keys) -> dict:
    out: dict = {}
    for k in keys:
        out[str(_ply(k))] = out.get(str(_ply(k)), 0) + 1
    return out


def main() -> None:
    import hashlib
    import platform
    import random

    import torch

    from games.connect4 import Connect4
    from harness.c4_oracle import EMPTY_BOARD_VALUE
    from harness.certify import certify
    from harness.depth_cost import SPEC
    from harness.evidence import load_evidence, save_evidence
    from harness.exact_values import ExactValues
    from harness.fingerprint import training_fingerprint
    from harness.neural import load_net
    from harness.opening_table import exception_table, full_table, hybrid_chooser
    from harness.strategy_tree import raw_chooser

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()
    game = Connect4()
    root = game.initial_state(random.Random(0))
    trained = load_evidence(ROOT / "evidence" / f"{SPEC['run']}.json.gz")
    rows = {r["seed"]: r for r in trained["seeds"]}
    shallow, deep = SPEC["depths"][-2], max(SPEC["depths"])
    nets: dict = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        exact = ExactValues(game, pool, load_evidence(LABELS)["positions"])

        def move_values(states):
            out = exact.move_values(states)
            print(f"  {len(states)} positions; values {exact.counts}; {time.time() - t0:.0f}s", flush=True)
            return out

        def walk(seed, horizon):
            row = rows[seed]
            path = ROOT / row["net"]["path"]
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if sha != row["net"]["file_sha256"]:
                raise SystemExit(f"seed {seed}: the saved net is not the one its run recorded")
            net = load_net(str(path))
            choose = raw_chooser(game, net)
            table, stats = exception_table(game, root, 0, choose, move_values, horizon)
            cert = certify(game, root, 0, hybrid_chooser(game, table, choose), lambda s: exact.positions([s])[0],
                           max_depth=horizon, value_many=exact.positions, root_value=EMPTY_BOARD_VALUE)
            nets[seed] = {"seed": seed, "horizon": horizon, "net_file_sha256": sha,
                          "params": sum(p.numel() for p in net.parameters()), "certified": cert["certified"],
                          "positions_by_ply": stats["positions_by_ply"], "exceptions_by_ply": _by_ply(table)}
            Path(args.progress).write_text(json.dumps({"nets": list(nets.values())}, indent=1))
            print(f"seed {seed} through {horizon}: {stats['positions']} positions, {len(table)} exceptions "
                  f"{nets[seed]['exceptions_by_ply']}, certified {cert['certified']}; {time.time() - t0:.0f}s",
                  flush=True)

        for horizon, seeds in ((shallow, SPEC["seeds"]), (deep, SPEC["deep"])):
            table, _frontier = full_table(game, root, 0, move_values, horizon)
            table_by_ply = _by_ply(table)
            print(f"table through {horizon}: {len(table)} entries {table_by_ply}; {time.time() - t0:.0f}s", flush=True)
            for seed in seeds:
                walk(seed, horizon)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the readout ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version(),
                                                                            "torch": torch.__version__},
                             "run": {"name": SPEC["run"], "training_fingerprint": trained["training_fingerprint"],
                                     "config": trained["config"]},
                             "table": {"horizon": deep, "by_ply": table_by_ply}, "values": exact.counts,
                             "nets": [nets[s] for s in sorted(nets)]})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
