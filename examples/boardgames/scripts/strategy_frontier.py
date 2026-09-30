"""§C.49 — the Connect-4 ORACLE FRONTIER through a horizon: for each net setup, grow it round by round on its own
first-player strategy tree (harness.strategy_fit) and report whether it becomes certified through --depth plies.
Exact move values come from the native solver in worker processes and are cached in --labels across widths, rounds
and runs, so the opening's hard solves are paid for once. Labelling is CHECK-FIRST: the net's move is checked with one
solve, and only a move that does not keep the win costs the full label.

    PYTHONPATH=. .venv/bin/python scripts/strategy_frontier.py --depth 8 --setups canon_conv:8,16 residual:16 \\
        --rounds 12 --workers 8 --out evidence/c49_strategy_d8.json.gz
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

MEASUREMENT_MODULES = ("scripts/strategy_frontier.py", "harness/strategy_fit.py", "harness/frontier.py",
                       "harness/native_solver.py", "harness/solver.py")
LABELS = Path(__file__).resolve().parent.parent / "books" / "c4_labels.json.gz"
RECIPES = {"default": {"lr": 2e-3, "batch": 256, "max_epochs": 3000, "check_every": 25, "patience": 600},
           "hold": {"lr": 3e-3, "lr_end": 1e-5, "batch": 512, "max_epochs": 4000, "check_every": 25, "patience": None}}
_BOOK = None


def _values(job: tuple) -> dict:
    global _BOOK
    from games.connect4 import C4State
    from harness import native_solver
    from harness.book import load_book

    if _BOOK is None:
        _BOOK = load_book("connect4")
    board, to_move = job
    return native_solver.move_values(C4State(tuple(board), to_move, None, False), book=_BOOK)


def _kept(job: tuple) -> int:
    """The value the mover keeps by playing `action` — one exact solve of the position it leads to."""
    global _BOOK
    import random

    from games.connect4 import C4State, Connect4
    from harness import native_solver
    from harness.book import load_book

    if _BOOK is None:
        _BOOK = load_book("connect4")
    board, to_move, action = job
    game = Connect4()
    child = game.step(C4State(tuple(board), to_move, None, False), action, random.Random(0))
    if game.is_terminal(child):
        return int(round(game.returns(child)[to_move]))
    return -native_solver.solve_position(child, book=_BOOK)


def _load_labels(path: Path, game) -> dict:
    from games.connect4 import C4State
    from harness.evidence import load_evidence

    if not path.exists():
        return {}
    out = {}
    for row in load_evidence(path)["positions"]:
        s = C4State(tuple(row["board"]), row["to_move"], None, False)
        out[game.state_key(s)] = (s, {int(a): v for a, v in row["values"].items()})
    return out


def _save_labels(path: Path, labels: dict) -> None:
    """Write the whole cache to a temporary file and rename it over the old one, so a run stopped mid-save never
    leaves a corrupt cache behind."""
    from harness.evidence import save_evidence

    path.parent.mkdir(exist_ok=True)
    tmp = path.with_name(path.name.replace(".json.gz", ".tmp.json.gz"))
    save_evidence(tmp, {"positions": [{"board": list(s.board), "to_move": s.to_move,
                                       "values": {str(a): v for a, v in vals.items()}}
                                      for s, vals in labels.values()]})
    tmp.replace(path)


def main() -> None:
    import platform
    import random

    import torch

    from games.connect4 import Connect4
    from harness.evidence import save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.frontier import arch_at
    from harness.strategy_fit import fit_strategy

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--setups", nargs="+", required=True, help="family:width,width (family as in harness.frontier)")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="default",
                    help="hold: refits train until they hold their data (decaying rate, no patience stop)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    game = Connect4()
    labels_path = Path(args.labels)
    cache = _load_labels(labels_path, game)
    known = {k: vals for k, (_s, vals) in cache.items()}
    states = {k: s for k, (s, _vals) in cache.items()}
    stamps = (training_fingerprint("connect4"), training_fingerprint(modules=MEASUREMENT_MODULES))
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    root = game.initial_state(random.Random(0))
    runs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        def many(batch: list) -> list:
            t = time.time()
            out = list(pool.map(_values, [(s.board, s.to_move) for s in batch], chunksize=4))
            for s, vals in zip(batch, out, strict=True):
                states[game.state_key(s)] = s
                known[game.state_key(s)] = vals
            _save_labels(labels_path, {k: (states[k], v) for k, v in known.items() if k in states})
            print(f"    labelled {len(batch):>7} positions in {time.time() - t:7.1f}s", flush=True)
            return out

        def check(pairs: list) -> list:
            t = time.time()
            out = list(pool.map(_kept, [(s.board, s.to_move, a) for s, a in pairs], chunksize=8))
            for (s, a), v in zip(pairs, out, strict=True):
                states[game.state_key(s)] = s
                if v == 1:
                    known.setdefault(game.state_key(s), {})[a] = 1
            _save_labels(labels_path, {k: (states[k], v) for k, v in known.items() if k in states})
            print(f"    checked  {len(pairs):>7} moves     in {time.time() - t:7.1f}s", flush=True)
            return out

        for spec in args.setups:
            family_name, widths = spec.split(":")
            canonical = family_name.startswith("canon_")
            family = {"body": family_name[len("canon_"):] if canonical else family_name, "canonical": canonical}
            for width in [int(w) for w in widths.split(",")]:
                t0 = time.time()
                arch = arch_at(family, width)
                r = fit_strategy(game, arch, root, 0, args.depth, args.seed, RECIPES[args.recipe], args.rounds,
                                 _values_local,
                                 known, many, check, warm_start=True)
                r.update({"family": family_name, "width": width, "arch": arch, "seconds": round(time.time() - t0, 1)})
                print(f"{family_name} width {width} ({r['params']} params): certified through depth {args.depth} = "
                      f"{r['certified']}{' (STALLED: could not hold its data)' if r['stalled'] else ''} after {len(r['rounds'])} rounds; failures by round "
                      f"{[e['failures'] for e in r['rounds']]}; {r['positions']} positions [{r['seconds']:.0f}s]",
                      flush=True)
                runs.append(r)
                _save_labels(labels_path, {k: (states[k], v) for k, v in known.items() if k in states})
    if (training_fingerprint("connect4"), training_fingerprint(modules=MEASUREMENT_MODULES)) != stamps:
        raise SystemExit("training or measurement code changed while the runs ran — evidence not written")
    save_evidence(args.out, {"game": "connect4", "started": started,
                             "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "training_fingerprint": stamps[0], "measurement_fingerprint": stamps[1],
                             "versions": {"python": platform.python_version(), "torch": torch.__version__},
                             "config": {"depth": args.depth, "setups": args.setups, "rounds": args.rounds,
                                        "seed": args.seed, "recipe": RECIPES[args.recipe], "recipe_name": args.recipe,
                                        "player": 0, "warm_start": True},
                             "labelled_positions": len(known), "runs": runs})


def _values_local(state) -> dict:
    return _values((state.board, state.to_move))


if __name__ == "__main__":
    main()
