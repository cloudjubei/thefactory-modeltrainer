"""§3.6 T2 — reads T1's file back, walks every leaf's lines on `--workers` processes for its map's orders, gives the
maps the lowest levels that keep them, merged where they agree, writes the game again, reads it back from the bytes
alone and plays what it read against every reply. `--source` is T1's evidence, or the G3 dry run's with `--out`
elsewhere.

    PYTHONPATH=. .venv/bin/python scripts/c4_map_orders.py --workers 7"""
from __future__ import annotations

import argparse
import base64
import hashlib
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from games.connect4 import COLS, ROWS, C4State, Connect4  # noqa: E402
from harness.evidence import load_evidence, save_evidence  # noqa: E402
from harness.floor_t1 import mirror_cells, t1_differences  # noqa: E402
from harness.floor_t2 import SPEC, t2_orders  # noqa: E402
from harness.map_orders import demands, merged, shrink  # noqa: E402
from harness.steady_state import Facts  # noqa: E402
from harness.strategy_file import decode_strategy, encode_strategy, leaf_states, replay, replay_leaf  # noqa: E402

CELLS = mirror_cells(ROWS, COLS)
N_LEVELS = SPEC["rule"]["n_levels"]
FACTS_LIMIT = 300_000
CAP = 10 ** 9
STATE: dict = {}


def _facts(limit: int = 2_000_000) -> Facts:
    return Facts(Connect4(), limit=limit, **{k: v for k, v in SPEC["rule"].items() if k != "n_levels"})


def _state(key) -> C4State:
    board, to_move = key
    return C4State(tuple(board), to_move, None, False)


def _start(maps: list) -> None:
    STATE.update(facts=_facts(FACTS_LIMIT), maps=maps)


def _walk(job: tuple) -> tuple:
    board, to_move, node = job
    found = demands(STATE["facts"], C4State(board, to_move, None, False), STATE["maps"][node["leaf"]], N_LEVELS,
                    node.get("exceptions"), CELLS, bool(node.get("mirrored")))
    return node["leaf"], sorted(found["strict"]), sorted(found["weak"])


def _leaf(job: tuple) -> bool:
    board, to_move, node = job
    return replay_leaf(STATE["facts"], C4State(board, to_move, None, False), node, STATE["maps"], N_LEVELS, CAP, CELLS)


def _jobs(leaves) -> list:
    return sorted(((s.board, s.to_move, n) for s, n in leaves), key=lambda j: sum(v == 0 for v in j[0]), reverse=True)


def _pool(workers: int, maps: list, fn, jobs: list, label: str, t0: float):
    with Pool(workers, _start, (maps,)) as pool:
        for done, out in enumerate(pool.imap_unordered(fn, jobs), 1):
            if done % 500 == 0:
                print(f"{label} {done}/{len(jobs)} leaves, {time.time() - t0:.0f}s", flush=True)
            yield out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--source", default=str(ROOT / "evidence" / SPEC["source"]))
    ap.add_argument("--out", default=str(ROOT / "evidence" / SPEC["file"]))
    ap.add_argument("--no-replay", action="store_true", help="the dry run's quick pass: write and read back only")
    args = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()
    root = Connect4().initial_state()
    facts = _facts()
    source = load_evidence(args.source)
    read = decode_strategy(facts, root, N_LEVELS, CELLS, base64.b64decode(source["file"]))
    leaves = leaf_states(facts.game, root, read["nodes"])["leaves"]
    print(f"source: {len(read['nodes'])} nodes, {len(read['maps'])} maps, {len(leaves)} leaves, "
          f"{time.time() - t0:.0f}s", flush=True)
    walks = list(_pool(args.workers, read["maps"], _walk, _jobs(leaves), "walked", t0))
    orders = t2_orders(walks)
    walked = time.time() - t0
    plan = merged(orders, N_LEVELS)
    game = shrink(read, plan, {}, facts, N_LEVELS, _state)
    print(f"shrunk: {len(game['maps'])} maps, {sum(len(m) for m in game['maps'])} cells, "
          f"{time.time() - t0:.0f}s", flush=True)
    data, bits, parts = encode_strategy(facts, root, N_LEVELS, CELLS, game)
    print(f"file: {bits} bits, {len(data)} bytes, parts {sorted((k, round(v)) for k, v in parts.items())}", flush=True)
    again = decode_strategy(_facts(), root, N_LEVELS, CELLS, data)
    differences = t1_differences(game, again)
    print(f"read back: {differences} differences, {len(again['maps'])} maps, {time.time() - t0:.0f}s", flush=True)
    result = {"won": None, "leaves": None, "reason": "not played"}
    if not args.no_replay:
        def check(played):
            return all(_pool(args.workers, again["maps"], _leaf, _jobs(played), "played", t0))

        result = replay(facts, root, again, N_LEVELS, CAP, CELLS, check=check)
        print(f"replay: {result}, {time.time() - t0:.0f}s", flush=True)
    save_evidence(args.out, {
        "started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec": "harness.floor_t2", "source_sha256": source["sha256"], "file": base64.b64encode(data).decode(),
        "bits": bits, "sha256": hashlib.sha256(data).hexdigest(), "parts": {k: round(v, 1) for k, v in parts.items()},
        "orders": {str(i): [sorted(s), sorted(w)] for i, (s, w) in sorted(orders.items())},
        "nodes": len(again["nodes"]), "maps": len(again["maps"]), "cells": sum(len(m) for m in again["maps"]),
        "differences": differences, "replay": result, "walk_seconds": round(walked, 1),
        "seconds": round(time.time() - t0, 1)})
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
