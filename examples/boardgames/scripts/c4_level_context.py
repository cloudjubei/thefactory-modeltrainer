"""§3.6 T3 — reads T2's file back, writes the same game with each level coded in context (harness.level_context), reads
it back from the bytes alone and plays what it read against every reply on `--workers` processes. `--source` is T2's
evidence, or the G3 dry run's with `--out` elsewhere.

    PYTHONPATH=. .venv/bin/python scripts/c4_level_context.py --workers 7"""
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
from harness.floor_t3 import SPEC  # noqa: E402
from harness.level_context import decode_strategy, encode_strategy  # noqa: E402
from harness.steady_state import Facts  # noqa: E402
from harness.strategy_file import decode_strategy as decode_plain, leaf_states, replay, replay_leaf  # noqa: E402

CELLS = mirror_cells(ROWS, COLS)
N_LEVELS = SPEC["rule"]["n_levels"]
FACTS_LIMIT = 300_000
CAP = 10 ** 9
STATE: dict = {}


def _facts(limit: int = 2_000_000) -> Facts:
    return Facts(Connect4(), limit=limit, **{k: v for k, v in SPEC["rule"].items() if k != "n_levels"})


def _start(maps: list) -> None:
    STATE.update(facts=_facts(FACTS_LIMIT), maps=maps)


def _leaf(job: tuple) -> bool:
    board, to_move, node = job
    return replay_leaf(STATE["facts"], C4State(board, to_move, None, False), node, STATE["maps"], N_LEVELS, CAP, CELLS)


def _jobs(leaves) -> list:
    return sorted(((s.board, s.to_move, n) for s, n in leaves), key=lambda j: sum(v == 0 for v in j[0]), reverse=True)


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
    game = decode_plain(facts, root, N_LEVELS, CELLS, base64.b64decode(source["file"]))
    print(f"source: {len(game['nodes'])} nodes, {len(game['maps'])} maps, {time.time() - t0:.0f}s", flush=True)
    data, bits, parts = encode_strategy(facts, root, N_LEVELS, CELLS, game)
    print(f"file: {bits} bits, {len(data)} bytes, parts {sorted((k, round(v)) for k, v in parts.items())}", flush=True)
    again = decode_strategy(_facts(), root, N_LEVELS, CELLS, data)
    differences = t1_differences(game, again)
    print(f"read back: {differences} differences, {len(again['maps'])} maps, {time.time() - t0:.0f}s", flush=True)
    result = {"won": None, "leaves": None, "reason": "not played"}
    if not args.no_replay:
        def check(played):
            jobs = _jobs(played)
            with Pool(args.workers, _start, (again["maps"],)) as pool:
                won = True
                for done, ok in enumerate(pool.imap_unordered(_leaf, jobs), 1):
                    won = won and ok
                    if done % 500 == 0:
                        print(f"played {done}/{len(jobs)} leaves, {time.time() - t0:.0f}s", flush=True)
                return won

        result = replay(facts, root, again, N_LEVELS, CAP, CELLS, check=check)
        print(f"replay: {result}, {time.time() - t0:.0f}s", flush=True)
    save_evidence(args.out, {
        "started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec": "harness.floor_t3", "source_sha256": source["sha256"], "file": base64.b64encode(data).decode(),
        "bits": bits, "sha256": hashlib.sha256(data).hexdigest(), "parts": {k: round(v, 1) for k, v in parts.items()},
        "nodes": len(again["nodes"]), "maps": len(again["maps"]), "cells": sum(len(m) for m in again["maps"]),
        "leaves": len(leaf_states(facts.game, root, again["nodes"])["leaves"]), "differences": differences,
        "replay": result, "seconds": round(time.time() - t0, 1)})
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
