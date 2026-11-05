"""§3.6 X1 — mirror sharing priced on G3's certified whole game (harness.floor_x1), from evidence/c49_g3.json.gz, its
library's sources (c49_g2, c49_f2) and nothing else. The symmetry is found, not assumed: of the board grid's flips
and half turn, the one kept maps every move of 200 random plays to a move that fills the image cell, with the same end
and winner (Connect-4's gravity rules out the other two). Writes evidence/c49_x1.json.gz.

    PYTHONPATH=. .venv/bin/python scripts/c4_mirror_sharing.py"""
from __future__ import annotations

import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from games.connect4 import C4State, Connect4  # noqa: E402
from harness.evidence import load_evidence, save_evidence  # noqa: E402
from harness.floor_g1 import _coded  # noqa: E402
from harness.floor_g3 import SPEC as G3, g3_library, g3_size  # noqa: E402
from harness.steady_state import map_bits  # noqa: E402

GAME = Connect4()


def _grid_symmetries(rows: int, cols: int) -> dict:
    def perm(f):
        return [f(i // cols, i % cols) for i in range(rows * cols)]

    return {"flip_columns": perm(lambda r, c: r * cols + cols - 1 - c),
            "flip_rows": perm(lambda r, c: (rows - 1 - r) * cols + c),
            "half_turn": perm(lambda r, c: (rows - 1 - r) * cols + cols - 1 - c)}


def _image(state: C4State, perm: list) -> C4State:
    board = [0] * len(state.board)
    for i, v in enumerate(state.board):
        board[perm[i]] = v
    return C4State(tuple(board), state.to_move, state.winner, state.done)


def _filled(before: C4State, after: C4State) -> int:
    return next(i for i, (a, b) in enumerate(zip(before.board, after.board)) if a != b)


def _holds(perm: list, games: int, seed: int) -> int:
    """Positions checked if every move of `games` random plays has an image move filling the image cell, the end and
    winner unchanged; 0 at the first that does not."""
    rng = random.Random(seed)
    checked = 0
    for _ in range(games):
        s = GAME.initial_state()
        while not GAME.is_terminal(s):
            image = _image(s, perm)
            moves = {_filled(image, GAME.step(image, b)): b for b in GAME.legal_actions(image)}
            if len(moves) != len(GAME.legal_actions(s)):
                return 0
            for a in GAME.legal_actions(s):
                child = GAME.step(s, a)
                b = moves.get(perm[_filled(s, child)])
                if b is None:
                    return 0
                twin = GAME.step(image, b)
                if (twin.board != _image(child, perm).board or GAME.is_terminal(twin) != GAME.is_terminal(child)
                        or GAME.winner(twin) != GAME.winner(child)):
                    return 0
            checked += 1
            s = GAME.step(s, rng.choice(GAME.legal_actions(s)))
    return checked


def _key(levels: dict) -> frozenset:
    return frozenset((int(c), k) for c, k in levels.items())


def maps_price(e: dict, library: list, perm: list, g3_bits: int) -> dict:
    n_lib = len(library)
    played = sorted({n["leaf"] for r in e["roots"] for _, _, n in r["nodes"] if 1 <= n.get("leaf", 0) <= n_lib})
    entries = [(library[i - 1][0], library[i - 1][1]) for i in played]
    for r in e["roots"]:
        entries += list(zip(r["maps"][1 + r["shared"]:], r["map_empty"][1 + r["shared"]:]))
    width = math.ceil(math.log2(len(entries)))
    seen: set = set()
    twins, twin_bits = 0, 0
    for levels, empty in entries:
        key = _key(levels)
        image = frozenset((perm[c], k) for c, k in key)
        if image != key and image in seen:
            twins += 1
            twin_bits += map_bits(dict(key), empty, G3["builder"]["level_bits"])
        seen.add(key)
    price = g3_bits - twin_bits + twins * width + len(entries)
    return {"entries": len(entries), "twins": twins, "twin_bits": twin_bits, "index_bits": width, "price": price}


def roots_price(e: dict, library: list, perm: list) -> dict:
    by_board = {(tuple(r["board"]), r["to_move"]): r for r in e["roots"]}
    dropped = set()
    pairs = 0
    for key, r in by_board.items():
        image = (_image(C4State(key[0], key[1], None, False), perm).board, key[1])
        if image == key or image not in by_board or r["index"] in dropped:
            continue
        twin = by_board[image]
        if twin["index"] in dropped or (key, r["index"]) > (image, twin["index"]):
            continue
        pairs += 1
        dropped.add(max((_coded(r), r["index"]), (_coded(twin), twin["index"]))[1])
    kept = [r for r in e["roots"] if r["index"] not in dropped]
    n_lib = len(library)
    played = {n["leaf"] for r in kept for _, _, n in r["nodes"] if 1 <= n.get("leaf", 0) <= n_lib}
    shared = sum(map_bits({int(c): k for c, k in library[i - 1][0].items()}, library[i - 1][1],
                          G3["builder"]["level_bits"]) for i in played)
    frontier = len(e["roots"])
    opening = e["opening"]["moves"] * 4 + e["opening"]["trivial"]
    width = math.ceil(math.log2(frontier))
    price = opening + sum(_coded(r) for r in kept) + shared + frontier + len(dropped) * width
    return {"frontier": frontier, "pairs": pairs, "dropped": len(dropped),
            "dropped_bits": sum(_coded(r) for r in e["roots"] if r["index"] in dropped),
            "library_maps_kept": len(played), "index_bits": width, "price": price}


def main() -> None:
    out = ROOT / "evidence" / "c49_x1.json.gz"
    if out.exists():
        raise SystemExit(f"{out} exists")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    symmetries = {name: _holds(perm, 200, 7) for name, perm in _grid_symmetries(*GAME.board_shape).items()}
    held = [name for name, n in symmetries.items() if n]
    if held != ["flip_columns"]:
        raise SystemExit(f"symmetries held: {symmetries}")
    perm = _grid_symmetries(*GAME.board_shape)["flip_columns"]
    g3, g2, f2 = (load_evidence(ROOT / "evidence" / f"c49_{n}.json.gz") for n in ("g3", "g2", "f2"))
    library = g3_library(g2, f2)
    g3_bits = g3_size(g3, g2, f2, G3)["bits"]
    record = {"started": started, "g3_bits": g3_bits, "symmetries": symmetries,
              "symmetry_checked": symmetries["flip_columns"],
              "maps": maps_price(g3, library, perm, g3_bits), "roots": roots_price(g3, library, perm)}
    save_evidence(out, record)
    print(record)


if __name__ == "__main__":
    main()
