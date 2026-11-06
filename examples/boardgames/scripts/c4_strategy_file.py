"""§3.6 T1 — writes a whole game as one file, reads it back from the bytes alone and plays what it read against every
reply, on `--workers` processes. The game is Z2's, or G3's with `--game g3` (the dry run, into `--out`).

    PYTHONPATH=. .venv/bin/python scripts/c4_strategy_file.py --workers 7"""
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
from harness.evidence import save_evidence  # noqa: E402
from harness.floor_t1 import SPEC, mirror_cells, t1_differences, t1_game  # noqa: E402
from harness.steady_state import Facts  # noqa: E402
from harness.strategy_file import decode_strategy, encode_strategy, replay, replay_leaf  # noqa: E402
from tests.test_c49_t1_evidence import _g3_pieces, _opening, _z2_pieces  # noqa: E402

CELLS = mirror_cells(ROWS, COLS)
FACTS_LIMIT = 300_000
CAP = 10 ** 9
STATE: dict = {}


def _facts(limit: int = 2_000_000) -> Facts:
    return Facts(Connect4(), limit=limit, **{k: v for k, v in SPEC["rule"].items() if k != "n_levels"})


def _start(maps: list) -> None:
    STATE.update(facts=_facts(FACTS_LIMIT), maps=maps)


def _leaf(job: tuple) -> bool:
    board, to_move, node = job
    return replay_leaf(STATE["facts"], C4State(board, to_move, None, False), node, STATE["maps"],
                       SPEC["rule"]["n_levels"], CAP, CELLS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--game", choices=("z2", "g3"), default="z2")
    ap.add_argument("--out", default=str(ROOT / "evidence" / "c49_t1.json.gz"))
    ap.add_argument("--no-replay", action="store_true", help="the dry run's quick pass: write and read back only")
    args = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    root = Connect4().initial_state()
    t0 = time.time()
    facts = _facts()
    source = t1_game(facts, root, CELLS, _opening(), _z2_pieces() if args.game == "z2" else _g3_pieces())
    print(f"game: {len(source['nodes'])} nodes, {len(source['maps'])} maps, {time.time() - t0:.0f}s", flush=True)
    data, bits, parts = encode_strategy(facts, root, SPEC["rule"]["n_levels"], CELLS, source)
    print(f"file: {bits} bits, {len(data)} bytes, parts {sorted((k, round(v)) for k, v in parts.items())}", flush=True)
    read = decode_strategy(_facts(), root, SPEC["rule"]["n_levels"], CELLS, data)
    differences = t1_differences(source, read)
    print(f"read back: {differences} differences, {len(read['maps'])} maps, {time.time() - t0:.0f}s", flush=True)
    result = {"won": None, "leaves": None, "reason": "not played"}
    if not args.no_replay:
        def check(leaves):
            jobs = sorted(((s.board, s.to_move, n) for s, n in leaves), key=lambda j: sum(v == 0 for v in j[0]),
                          reverse=True)
            with Pool(args.workers, _start, (read["maps"],)) as pool:
                done = 0
                for ok in pool.imap_unordered(_leaf, jobs):
                    done += 1
                    if done % 200 == 0:
                        print(f"played {done}/{len(jobs)} leaves, {time.time() - t0:.0f}s", flush=True)
                    if not ok:
                        return False
            return True

        result = replay(facts, root, read, SPEC["rule"]["n_levels"], CAP, CELLS, check=check)
        print(f"replay: {result}, {time.time() - t0:.0f}s", flush=True)
    save_evidence(args.out, {
        "started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec": "harness.floor_t1", "game": args.game, "file": base64.b64encode(data).decode(), "bits": bits,
        "sha256": hashlib.sha256(data).hexdigest(), "parts": {k: round(v, 1) for k, v in parts.items()},
        "nodes": len(read["nodes"]), "maps": len(read["maps"]), "differences": differences, "replay": result,
        "seconds": round(time.time() - t0, 1)})
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
