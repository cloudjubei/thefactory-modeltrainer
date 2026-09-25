"""§C.48 T3 — a pre-solved bank of Connect-4 positions: each position with the exact weak value of every legal move.

Sources are the self-play positions the ten §C.47 Stage 0 nets recorded (checkpoints/c47_C0*/seed*.train.pt) — the
distribution a generic Connect-4 process actually plays — and, in equal number, one-move siblings of them: positions
one move off that play, where the Stage 0 nets were measurably weaker. Only plies MIN_PLY..MAX_PLY are taken (an
exact solve near the opening costs minutes). Positions are deduplicated by canonical key, so a position and its
mirror are ONE entry, stored in the orientation it was drawn in, and split train/test by a hash of that key so no
test position has a mirror image in training.

The bank serves both T3 (what accuracy each net size can reach from exact labels) and any later Stage 1, whose
probes would otherwise re-solve positions at ~10 h per five nets.

    PYTHONPATH=. .venv/bin/python scripts/build_bank.py --per-source 24000 --workers 10 --out evidence/c48_bank.json.gz
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

MIN_PLY, MAX_PLY = 14, 38
TEST_SHARE = 0.2
CHUNK = 400


def split_of(key) -> str:
    return "test" if int(hashlib.sha256(f"c48bank:{key}".encode()).hexdigest(), 16) % 100 < TEST_SHARE * 100 else "train"


def solve_chunk(chunk: list) -> list:
    """Exact weak move values for a chunk of (board, to_move, source) positions, in one process so the solver's
    transposition table is shared across related positions."""
    from games.connect4 import C4State
    from harness.solver import move_values

    out = []
    for board, to_move, source in chunk:
        s = C4State(tuple(board), to_move, None, False)
        vals = move_values(s, weak=True)
        out.append({"board": list(board), "to_move": to_move, "source": source,
                    "values": {str(a): v for a, v in sorted(vals.items())}})
    return out


def main() -> None:
    import torch

    from harness.evidence import save_evidence
    from harness.neural import one_ply_siblings
    from harness.registry import resolve_game

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-source", type=int, default=24000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    game = resolve_game("connect4")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    parts = sorted(glob.glob("checkpoints/c47_C0/seed*.train.pt") + glob.glob("checkpoints/c47_C0v2/seed*.train.pt"))
    selfplay: dict = {}
    for path in parts:
        for s in torch.load(path, weights_only=False)["states"]:
            if MIN_PLY <= game.ply(s) <= MAX_PLY and not game.is_terminal(s):
                selfplay.setdefault(game.canonical_key(s), s)
    sibs, _stats = one_ply_siblings(game, list(selfplay.values()), game.canonical_key)
    siblings = {game.canonical_key(s): s for s in sibs if MIN_PLY <= game.ply(s) <= MAX_PLY}
    rng = random.Random(args.seed)
    picked = []
    for name, pool in (("selfplay", selfplay), ("sibling", siblings)):
        keys = sorted(pool)
        rng.shuffle(keys)
        picked += [(list(pool[k].board), pool[k].to_move, name) for k in keys[: args.per_source]]
    picked.sort(key=lambda p: (p[2], sum(1 for v in p[0] if v)))
    chunks = [picked[i:i + CHUNK] for i in range(0, len(picked), CHUNK)]
    print(f"{len(selfplay)} self-play and {len(siblings)} sibling positions at plies {MIN_PLY}-{MAX_PLY}; solving "
          f"{len(picked)} in {len(chunks)} chunks", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, out in enumerate(ex.map(solve_chunk, chunks)):
            rows += out
            if i % 10 == 0:
                print(f"  {len(rows)}/{len(picked)} solved", flush=True)
    from games.connect4 import C4State

    for r in rows:
        r["split"] = split_of(game.canonical_key(C4State(tuple(r["board"]), r["to_move"], None, False)))
    save_evidence(args.out, {"game": "connect4", "started": started,
                             "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "config": {"sources": parts, "min_ply": MIN_PLY, "max_ply": MAX_PLY,
                                        "per_source": args.per_source, "seed": args.seed, "test_share": TEST_SHARE},
                             "positions": rows})
    print(f"bank written: {len(rows)} positions", flush=True)


if __name__ == "__main__":
    main()
