"""§C.49 D2 — how much search makes the Connect-4 opening labels right? For each net the T10 run saved, walk its own
first-player tree to ply 4, re-run its own search at each budget at every position (the deep budget at plies 0-2
only), and record the move each budget prefers (the argmax of the improved policy — the label training uses) next to
the exact optimal moves. Judged by harness.floor_opening.opening_report.

Optimal moves come from the label cache where it records every move of a position, and otherwise from one native
solve per child position.

    PYTHONPATH=. .venv/bin/python scripts/c4_opening_budget.py --workers 8 --out evidence/c49_D2_opening.json.gz
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NETS = ROOT / "checkpoints" / "c49_sf" / "c49_T10_solver_free"
LABELS = ROOT / "books" / "c4_labels.json.gz"
SEEDS = (361, 362, 363, 364, 365, 366)
MEASUREMENT_MODULES = ("scripts/c4_opening_budget.py", "harness/c4_oracle.py", "harness/native_solver.py")
CONFIG = {"budgets": [200, 2000, 20000], "deep_budget": 100000, "deep_plies": 2, "max_ply": 4}
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
    from games.connect4 import C4State, Connect4
    from harness.c4_oracle import solve
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.strategy_tree import raw_chooser, strategy_tree_positions

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    stamps = (training_fingerprint("connect4"), training_fingerprint(modules=MEASUREMENT_MODULES))
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    game = Connect4()
    root = game.initial_state(random.Random(0))
    full = {}
    for row in load_evidence(LABELS)["positions"] if LABELS.exists() else []:
        s = C4State(tuple(row["board"]), row["to_move"], None, False)
        if set(int(a) for a in row["values"]) == set(game.legal_actions(s)):
            full[game.state_key(s)] = {int(a): int(v) for a, v in row["values"].items()}
    counts = {"recorded": 0, "solved_positions": 0}

    tmpdir = tempfile.mkdtemp(prefix="d2_rl_")
    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}
    for k in saved:
        os.environ[k] = "1"
    relabel_pool = mp.get_context("spawn").Pool(args.workers, initializer=neural._relabel_worker_init,
                                                initargs=("connect4",))
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    version = 0
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers + 2) as solver_pool:
        def move_values(states: list) -> list:
            todo = [s for s in states if game.state_key(s) not in full]
            jobs = [(s, a, game.step(s, a)) for s in todo for a in game.legal_actions(s)]
            open_jobs = [(s, a, c) for s, a, c in jobs if not game.is_terminal(c)]
            solved = dict(zip([(game.state_key(s), a) for s, a, _c in open_jobs],
                              solver_pool.map(solve, [(c.board, c.to_move) for _s, _a, c in open_jobs],
                                              chunksize=1)))
            for s, a, c in jobs:
                mover = game.current_player(s)
                kept = (round(game.returns(c)[mover]) if game.is_terminal(c)
                        else -solved[(game.state_key(s), a)])
                full.setdefault(game.state_key(s), {})[a] = kept
            counts["recorded"] += len(states) - len(todo)
            counts["solved_positions"] += len(todo)
            return [full[game.state_key(s)] for s in states]

        for seed in SEEDS:
            t0 = time.time()
            path = NETS / f"seed_{seed}.pt"
            net = neural.load_net(str(path))
            walked = strategy_tree_positions(game, root, 0, raw_chooser(game, net), CONFIG["max_ply"] + 1)
            plies = [sum(1 for c in s.board if c) for s in walked]
            preferred = [{} for _ in walked]
            for budget in CONFIG["budgets"] + [CONFIG["deep_budget"]]:
                idx = [i for i, p in enumerate(plies)
                       if budget != CONFIG["deep_budget"] or p <= CONFIG["deep_plies"]]
                version += 1
                labels = neural._parallel_relabel(relabel_pool, args.workers, tmpdir, net, version,
                                                  {"sims": budget, **SEARCH}, [walked[i] for i in idx])
                for i, (_x, pi, _v) in zip(idx, labels):
                    preferred[i][str(budget)] = pi.index(max(pi))
            values = move_values(walked)
            positions = []
            for s, ply, pref, vals in zip(walked, plies, preferred, values):
                best = max(vals.values())
                positions.append({"ply": ply, "board": list(s.board), "to_move": s.to_move,
                                  "optimal": sorted(a for a, v in vals.items() if v == best), "preferred": pref})
            rows.append({"seed": seed, "net_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "positions": positions, "seconds": round(time.time() - t0, 1)})
            right = {b: sum(1 for p in positions if p["preferred"][str(b)] in p["optimal"])
                     for b in CONFIG["budgets"]}
            print(f"seed {seed}: {len(positions)} positions; optimal preferred by budget {right}; values {counts} "
                  f"[{rows[-1]['seconds']:.0f}s]", flush=True)
    relabel_pool.close()
    relabel_pool.join()
    shutil.rmtree(tmpdir, ignore_errors=True)
    if (training_fingerprint("connect4"), training_fingerprint(modules=MEASUREMENT_MODULES)) != stamps:
        raise SystemExit("training or measurement code changed while the measurement ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "training_fingerprint": stamps[0], "measurement_fingerprint": stamps[1],
                             "versions": {"python": platform.python_version(), "torch": torch.__version__},
                             "config": CONFIG, "search": SEARCH, "values": counts, "seeds": rows})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
