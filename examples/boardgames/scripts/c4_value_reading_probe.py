"""§C.49 T11c — would the VALUE-AWARE stop fire on Connect-4 nets known to be wrong? For each of the six T10 final
nets (none certified through 10 plies, h91), walk its own first-player tree through 10 plies exactly as training
does, relabel every position with T10's 200-sim search (values included), and read the share rule and the
value-aware rule at several deltas. A zero at the registered delta is a stop on a net the solver does not certify.
No solver is used.

    PYTHONPATH=. .venv/bin/python scripts/c4_value_reading_probe.py --workers 8 --out evidence/c49_T11c_c4_value_reading.json.gz
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NETS = ROOT / "checkpoints" / "c49_sf" / "c49_T10_solver_free"
SEEDS = (361, 362, 363, 364, 365, 366)
MEASUREMENT_MODULES = ("scripts/c4_value_reading_probe.py",)
CONFIG = {"depth": 10, "sims": 200, "deltas": [0.05, 0.1, 0.2]}
SEARCH = {"gumbel": True, "gumbel_m": 16, "c_scale": 0.1, "add_noise": False}


def main() -> None:
    import hashlib
    import multiprocessing as mp
    import os
    import platform
    import random
    import shutil
    import tempfile

    import torch

    import harness.neural as neural
    from games.connect4 import Connect4
    from harness.evidence import save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.strategy_tree import disagreements, raw_chooser, strategy_tree_positions

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    stamps = (training_fingerprint("connect4"), training_fingerprint(modules=MEASUREMENT_MODULES))
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    game = Connect4()
    root = game.initial_state(random.Random(0))
    tmpdir = tempfile.mkdtemp(prefix="t11c_rl_")
    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}
    for k in saved:
        os.environ[k] = "1"
    pool = mp.get_context("spawn").Pool(args.workers, initializer=neural._relabel_worker_init, initargs=("connect4",))
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    rows = []
    for version, seed in enumerate(SEEDS, start=1):
        t0 = time.time()
        path = NETS / f"seed_{seed}.pt"
        net = neural.load_net(str(path))
        choose = raw_chooser(game, net)
        walked = strategy_tree_positions(game, root, 0, choose, CONFIG["depth"])
        labels = neural._parallel_relabel(pool, args.workers, tmpdir, net, version,
                                          {"sims": CONFIG["sims"], **SEARCH}, walked, with_q=True)
        moves, pis, qs = choose(walked), [row[1] for row in labels], [row[3] for row in labels]
        plies = [sum(1 for c in s.board if c) for s in walked]
        by_ply = {}
        for p in sorted(set(plies)):
            idx = [i for i, q in enumerate(plies) if q == p]
            sub = ([moves[i] for i in idx], [pis[i] for i in idx], [qs[i] for i in idx])
            by_ply[str(p)] = {"positions": len(idx), "share": disagreements(sub[0], sub[1]),
                              **{f"value_{d}": disagreements(*sub, d) for d in CONFIG["deltas"]}}
        rows.append({"seed": seed, "net_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "walked": len(walked), "share": disagreements(moves, pis),
                     **{f"value_{d}": disagreements(moves, pis, qs, d) for d in CONFIG["deltas"]},
                     "by_ply": by_ply, "seconds": round(time.time() - t0, 1)})
        print({k: v for k, v in rows[-1].items() if k not in ("by_ply", "net_file_sha256")}, flush=True)
    pool.close()
    pool.join()
    shutil.rmtree(tmpdir, ignore_errors=True)
    if (training_fingerprint("connect4"), training_fingerprint(modules=MEASUREMENT_MODULES)) != stamps:
        raise SystemExit("training or measurement code changed while the probe ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "training_fingerprint": stamps[0], "measurement_fingerprint": stamps[1],
                             "versions": {"python": platform.python_version(), "torch": torch.__version__},
                             "config": CONFIG, "search": SEARCH, "seeds": rows})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
