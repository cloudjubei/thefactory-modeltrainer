"""§C.49 T14 — read the pilot's final nets. For each arm's training evidence (evidence/c49_T14_<arm>.json.gz, from
scripts/c4_solver_free.py) and each seed's saved net: walk the net's own first-player tree, and at every White
position through the readout plies record the raw move, the value head's rating of each move (minus its value at
the resulting position, the exact result where the move ends the game), the 200-sim relabel label with T10's search
settings, and every move's exact value — from the label cache where it records the whole position, otherwise one
native solve per resulting position. Judged by harness.floor_c4_value_signal.pilot_report on the `--spec` module's
SPEC; `--prefix` names the training evidence (<prefix>_<arm>.json.gz).

    PYTHONPATH=. .venv/bin/python scripts/c4_pilot_readout.py --workers 8 --out evidence/c49_T14_readout.json.gz
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "books" / "c4_labels.json.gz"
MEASUREMENT_MODULES = ("scripts/c4_pilot_readout.py", "harness/c4_oracle.py", "harness/native_solver.py")
SEARCH = {"gumbel": True, "gumbel_m": 16, "c_scale": 0.1, "add_noise": False}


def _value_head(game, net, states: list) -> list:
    import torch

    from harness.neural import encode

    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(states), 4096):
            _logits, v = net(torch.stack([encode(game, s) for s in states[i:i + 4096]]))
            out.extend(float(x) for x in v[:, 0])
    return out


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
    ap.add_argument("--spec", default="floor_c4_value_signal")
    ap.add_argument("--prefix", default="c49_T14")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from importlib import import_module

    spec = import_module(f"harness.{args.spec}").SPEC
    torch.set_num_threads(2)
    stamps = (training_fingerprint("connect4"), training_fingerprint(modules=MEASUREMENT_MODULES))
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    game = Connect4()
    root = game.initial_state(random.Random(0))
    plies = spec["readout"]["plies"]
    exact = {}
    for row in load_evidence(LABELS)["positions"]:
        s = C4State(tuple(row["board"]), row["to_move"], None, False)
        if set(int(a) for a in row["values"]) == set(game.legal_actions(s)):
            exact[game.state_key(s)] = {int(a): int(v) for a, v in row["values"].items()}

    tmpdir = tempfile.mkdtemp(prefix="t14_rl_")
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
    configs, eras, nets, positions, counts = {}, {}, {}, [], {"recorded": 0, "solved": 0}
    version = 0
    with ProcessPoolExecutor(max_workers=args.workers) as solver_pool:
        for arm in spec["arms"]:
            trained = load_evidence(ROOT / "evidence" / f"{args.prefix}_{arm}.json.gz")
            configs[arm] = {k: v for k, v in trained["config"].items() if k != "certify_depth"}
            eras[arm] = trained["training_fingerprint"]
            nets[arm] = {}
            for row in sorted(trained["seeds"], key=lambda r: r["seed"]):
                path = ROOT / row["net"]["path"]
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                if sha != row["net"]["file_sha256"]:
                    raise SystemExit(f"{arm} seed {row['seed']}: the saved net is not the one its run recorded")
                nets[arm][str(row["seed"])] = sha
                net = neural.load_net(str(path))
                choose = raw_chooser(game, net)
                walked = [s for s in strategy_tree_positions(game, root, 0, choose, max(plies) + 1)
                          if sum(1 for c in s.board if c) in plies]
                todo = [s for s in walked if game.state_key(s) not in exact]
                jobs = [(s, a, game.step(s, a)) for s in todo for a in game.legal_actions(s)]
                open_jobs = [(s, a, c) for s, a, c in jobs if not game.is_terminal(c)]
                solved = dict(zip([(game.state_key(s), a) for s, a, _c in open_jobs],
                                  solver_pool.map(solve, [(c.board, c.to_move) for _s, _a, c in open_jobs])))
                for s, a, c in jobs:
                    exact.setdefault(game.state_key(s), {})[a] = (round(game.returns(c)[0]) if game.is_terminal(c)
                                                                  else -solved[(game.state_key(s), a)])
                counts["recorded"] += len(walked) - len(todo)
                counts["solved"] += len(todo)
                raw = choose(walked)
                children = [(i, a, game.step(s, a)) for i, s in enumerate(walked) for a in game.legal_actions(s)]
                heads = iter(_value_head(game, net, [c for _i, _a, c in children if not game.is_terminal(c)]))
                q_hat = [{} for _ in walked]
                for i, a, c in children:
                    q_hat[i][a] = game.returns(c)[0] if game.is_terminal(c) else -next(heads)
                version += 1
                labels = neural._parallel_relabel(relabel_pool, args.workers, tmpdir, net, version,
                                                  {"sims": spec["readout"]["sims"], **SEARCH}, walked)
                for s, m, q, (_x, pi, _v) in zip(walked, raw, q_hat, labels):
                    positions.append({"arm": arm, "seed": row["seed"], "ply": sum(1 for c in s.board if c),
                                      "raw": m, "label": pi.index(max(pi)),
                                      "q_hat": {str(a): round(v, 6) for a, v in q.items()},
                                      "values": {str(a): v for a, v in exact[game.state_key(s)].items()}})
                print(f"{arm} seed {row['seed']}: {len(walked)} positions, values {counts}", flush=True)
    relabel_pool.close()
    relabel_pool.join()
    shutil.rmtree(tmpdir, ignore_errors=True)
    if (training_fingerprint("connect4"), training_fingerprint(modules=MEASUREMENT_MODULES)) != stamps:
        raise SystemExit("training or measurement code changed while the readout ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "training_fingerprint": stamps[0], "measurement_fingerprint": stamps[1],
                             "versions": {"python": platform.python_version(), "torch": torch.__version__},
                             "readout": spec["readout"], "search": SEARCH, "configs": configs, "eras": eras,
                             "nets": nets, "values": counts, "positions": positions})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
