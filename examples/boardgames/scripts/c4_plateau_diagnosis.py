"""§C.49 D1 — why the Connect-4 stop signal never fired (h88): for each net the T10 run saved, recompute its walk
of its own first-player tree (deterministic: the relabel search draws nothing), let the solver judge every position
where the net's raw move and its 200-sim search disagree, and certify the net through the horizon. Judged by
harness.floor_plateau.plateau_report.

Exact values come from the label cache where it records them (and the empty board's value, a theorem), and from the
native solver otherwise; the evidence counts both.

    PYTHONPATH=. .venv/bin/python scripts/c4_plateau_diagnosis.py --workers 8 \\
        --out evidence/c49_D1_plateau.json.gz
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
MEASUREMENT_MODULES = ("scripts/c4_plateau_diagnosis.py", "harness/c4_oracle.py", "harness/certify.py",
                       "harness/native_solver.py")
CONFIG = {"sims": 200, "depth": 10, "agree_share": 0.5, "gumbel": True, "gumbel_m": 16, "c_scale": 0.1}
EXAMPLES = 30


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
    from harness.c4_oracle import EMPTY_BOARD_VALUE, classify, move_optimal, recorded_values, solve
    from harness.certify import certify
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
    rows = load_evidence(LABELS)["positions"] if LABELS.exists() else []
    known = recorded_values(game, rows)
    for row in rows:
        s = C4State(tuple(row["board"]), row["to_move"], None, False)
        if set(int(a) for a in row["values"]) == set(game.legal_actions(s)):
            known[game.state_key(s)] = max(int(v) for v in row["values"].values())
    known[game.state_key(root)] = EMPTY_BOARD_VALUE
    counts = {"recorded": 0, "solved": 0}

    tmpdir = tempfile.mkdtemp(prefix="d1_rl_")
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
    out_rows = []
    with ProcessPoolExecutor(max_workers=args.workers + 2) as solver_pool:
        def values(states: list) -> list:
            keys = [game.state_key(s) for s in states]
            todo = {k: s for k, s in zip(keys, states) if k not in known}
            for k, v in zip(todo, solver_pool.map(solve, [(s.board, s.to_move) for s in todo.values()], chunksize=4)):
                known[k] = v
            counts["recorded"] += len(states) - len(todo)
            counts["solved"] += len(todo)
            return [known[k] for k in keys]

        for seed in SEEDS:
            t0 = time.time()
            path = NETS / f"seed_{seed}.pt"
            net = neural.load_net(str(path))
            choose = raw_chooser(game, net)
            walked = strategy_tree_positions(game, root, 0, choose, CONFIG["depth"])
            moves = choose(walked)
            labels = neural._parallel_relabel(
                relabel_pool, args.workers, tmpdir, net, seed,
                {"sims": CONFIG["sims"], "gumbel": CONFIG["gumbel"], "gumbel_m": CONFIG["gumbel_m"],
                 "c_scale": CONFIG["c_scale"], "add_noise": False}, walked)
            pis = [pi for _x, pi, _v in labels]
            dis = [i for i, (a, pi) in enumerate(zip(moves, pis)) if pi[a] < CONFIG["agree_share"] * max(pi)]
            needed = []
            for i in dis:
                s = walked[i]
                needed.append(s)
                for a in (moves[i], pis[i].index(max(pis[i]))):
                    child = game.step(s, a)
                    if not game.is_terminal(child):
                        needed.append(child)
            values(needed)
            classes = {"both_optimal": 0, "search_wrong": 0, "net_wrong": 0, "both_wrong": 0}
            by_ply: dict = {}
            examples = []
            for i in dis:
                s, a, pi = walked[i], moves[i], pis[i]
                b = pi.index(max(pi))
                label = classify(move_optimal(game, s, a, lambda st: known[game.state_key(st)]),
                                 move_optimal(game, s, b, lambda st: known[game.state_key(st)]))
                classes[label] += 1
                ply = str(sum(1 for c in s.board if c))
                by_ply.setdefault(ply, {}).setdefault(label, 0)
                by_ply[ply][label] += 1
                if len(examples) < EXAMPLES:
                    examples.append({"board": list(s.board), "to_move": s.to_move, "net_move": a, "search_move": b,
                                     "search_share": [round(p, 4) for p in pi], "class": label})
            cert = certify(game, root, 0, choose, lambda st: values([st])[0], max_depth=CONFIG["depth"],
                           value_many=values, root_value=EMPTY_BOARD_VALUE)
            row = {"seed": seed, "net_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "walked": len(walked), "disagreements": len(dis), "classes": classes, "by_ply": by_ply,
                   "examples": examples,
                   "certificate": {k: cert[k] for k in ("certified", "horizon", "failures", "failures_by_ply",
                                                        "complete", "nodes")},
                   "seconds": round(time.time() - t0, 1)}
            out_rows.append(row)
            print(f"seed {seed}: walked {len(walked)}, disagreements {len(dis)} {classes}; certified through "
                  f"{CONFIG['depth']} = {cert['certified']} (failures {cert['failures_by_ply']}); values {counts} "
                  f"[{row['seconds']:.0f}s]", flush=True)
    relabel_pool.close()
    relabel_pool.join()
    shutil.rmtree(tmpdir, ignore_errors=True)
    if (training_fingerprint("connect4"), training_fingerprint(modules=MEASUREMENT_MODULES)) != stamps:
        raise SystemExit("training or measurement code changed while the diagnosis ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "training_fingerprint": stamps[0], "measurement_fingerprint": stamps[1],
                             "versions": {"python": platform.python_version(), "torch": torch.__version__},
                             "config": CONFIG, "values": counts, "seeds": out_rows})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
