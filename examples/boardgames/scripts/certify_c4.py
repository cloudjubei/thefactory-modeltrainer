"""§C.49 P-START — certify a Connect-4 net's RAW play (one forward pass per move, argmax over legal columns) as the
first player from the empty board: at every position its strategy reaches, against every reply, its move keeps the
proven win. With --depth the certificate covers the first D plies ("certified through depth D"). Each ply's exact
solves are spread over worker processes with the native solver (harness/native_solver.py), the project book
short-circuiting positions it already proves.

    PYTHONPATH=. .venv/bin/python scripts/certify_c4.py --net checkpoints/scaled_runs/ab302_gpool_s0/ckpt_16.pt \\
        --depth 12 --workers 10 --out evidence/c49_certify_ab302_d12.json.gz
"""
from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

MEASUREMENT_MODULES = ("scripts/certify_c4.py", "harness/certify.py", "harness/native_solver.py", "harness/solver.py")
EMPTY_BOARD_VALUE = 1
EMPTY_BOARD_VALUE_SOURCE = "Connect-4 is a first-player win (Allis 1988; Allen 1988; Tromp's database)"
_BOOK = None


def _solve(job: tuple) -> int:
    global _BOOK
    from games.connect4 import C4State
    from harness import native_solver
    from harness.book import load_book

    if _BOOK is None:
        _BOOK = load_book("connect4")
    board, to_move = job
    return native_solver.solve_position(C4State(tuple(board), to_move, None, False), book=_BOOK)


def main() -> None:
    import platform
    import random

    import torch

    from games.connect4 import Connect4
    from harness import native_solver
    from harness.certify import certify
    from harness.evidence import save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.neural import encode, load_net

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    game = Connect4()
    net = load_net(args.net)
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()

    def choose(states: list) -> list:
        out = []
        for i in range(0, len(states), 4096):
            chunk = states[i:i + 4096]
            legal = torch.zeros(len(chunk), game.num_actions, dtype=torch.bool)
            for j, s in enumerate(chunk):
                legal[j, game.legal_actions(s)] = True
            with torch.no_grad():
                logits, _v = net(torch.stack([encode(game, s) for s in chunk]))
            out += logits.masked_fill(~legal, float("-inf")).argmax(dim=1).tolist()
        return out

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        def many(states: list) -> list:
            t = time.time()
            values = list(pool.map(_solve, [(s.board, s.to_move) for s in states], chunksize=8))
            print(f"  solved {len(states):>8} positions in {time.time() - t:7.1f}s  [{time.time() - t0:7.0f}s total]",
                  flush=True)
            return values

        result = certify(game, game.initial_state(random.Random(0)), 0, choose, lambda s: _solve((s.board, s.to_move)),
                         max_nodes=args.max_nodes, max_depth=args.depth, value_many=many,
                         root_value=EMPTY_BOARD_VALUE)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("the measurement code changed while the walk ran — evidence not written")
    print(f"certified={result['certified']} complete={result['complete']} complete_game={result['complete_game']} "
          f"failures={result['failures']} by depth {result['failures_by_ply']} nodes={result['nodes']} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "net": args.net,
                             "solver_c_sha256": hashlib.sha256(native_solver.SOURCE.read_bytes()).hexdigest(),
                             "root_value_source": EMPTY_BOARD_VALUE_SOURCE,
                             "net_sha256": hashlib.sha256(open(args.net, "rb").read()).hexdigest(),
                             "net_arch": net.arch, "params": sum(p.numel() for p in net.parameters()),
                             "versions": {"python": platform.python_version(), "torch": torch.__version__},
                             "config": {"depth": args.depth, "max_nodes": args.max_nodes, "player": 0},
                             "seconds": round(time.time() - t0, 1), "result": result})


if __name__ == "__main__":
    main()
