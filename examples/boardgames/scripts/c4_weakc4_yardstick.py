"""§3.6 Y1 — WeakC4 (2swap, https://github.com/2swap/WeakC4; GPL-3, read from a local checkout and never copied) as a
yardstick for G3's 280,640 bits: is the 6-8x gap the rule language, the shape of the tree, or the code? From the
checkout's solution/branches.json (a first-player move or a diagram index per position, one entry per mirror pair) and
solution/steady_states.json (a level 0-9 on every cell, row 0 the top), it walks WeakC4's strategy from the empty board
as a decoder would — every Yellow reply, mirrored entries played mirrored — and records:

  - the language: whether each leaf's diagram, translated cell for cell, wins at its position under our rule
    (harness.steady_state.Facts, the C verify) read three ways — "exact" with all ten levels (their reading but for
    our safe-move filter), "exact" with level 9 left unlevelled, and G3's rule ("lowest", unlevelled cells last) with
    level 9 left unlevelled;
  - the shape: table moves, leaves (positions Red wins at once included, as the builder makes them) and maps by plies,
    and the first-player positions each leaf's diagram plays;
  - the price in our code, as one strategy from the empty board: a bit per node, three per table move, references by
    first_use, and every map's bits (harness.steady_state.map_bits; a diagram used mirrored is a second map, our maps
    being cell-indexed) with each reading's levels renumbered in order.

Writes evidence/c49_weakc4.json.gz with the checkout's commit and the two files' hashes.

    PYTHONPATH=. .venv/bin/python scripts/c4_weakc4_yardstick.py --weakc4 <checkout>"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from games.connect4 import COLS, ROWS, C4State, Connect4  # noqa: E402
from harness.evidence import save_evidence  # noqa: E402
from harness.native_leaf import verify  # noqa: E402
from harness.reference_coding import code_bits  # noqa: E402
from harness.steady_state import Facts, map_bits  # noqa: E402

GAME = Connect4()
READINGS = {"exact_all": ({"ties": "exact"}, False), "exact_sparse": ({"ties": "exact"}, True),
            "g3_rule": ({"ties": "lowest", "unlevelled": "last"}, True)}


def _mirror(board: tuple) -> tuple:
    return tuple(board[r * COLS + COLS - 1 - c] for r in range(ROWS) for c in range(COLS))


def _replay(position: str) -> C4State:
    s = GAME.initial_state()
    for ch in position:
        s = GAME.step(s, int(ch) - 1)
    return s


def _levels(diagram: list, state: C4State, sparse: bool) -> dict:
    raw = {r * COLS + c: int(diagram[ROWS - 1 - r][c]) for r in range(ROWS) for c in range(COLS)
           if state.board[r * COLS + c] == 0}
    if sparse:
        raw = {cell: level for cell, level in raw.items() if level != 9}
    rank = {level: k for k, level in enumerate(sorted(set(raw.values())))}
    return {cell: rank[level] for cell, level in raw.items()}


def _wins_now(state: C4State) -> bool:
    return any(GAME.is_terminal(c) and GAME.winner(c) == 0 for a in GAME.legal_actions(state)
               for c in [GAME.step(state, a)])


def walk(branches: dict, diagrams: list) -> dict:
    entries = {}
    for position, value in branches.items():
        board = _replay(position).board
        entries[min(board, _mirror(board))] = (board, value)
    seen, moves, leaves, refs, maps = set(), [], [], [], {}
    stack = [GAME.initial_state()]
    while stack:
        s = stack.pop()
        if s.board in seen:
            continue
        seen.add(s.board)
        plies = sum(1 for v in s.board if v)
        if _wins_now(s):
            leaves.append({"plies": plies, "map": None})
            refs.append(0)
            continue
        stored, value = entries[min(s.board, _mirror(s.board))]
        mirrored = stored != s.board
        if isinstance(value, str):
            col = int(value) - 1
            moves.append({"plies": plies})
            child = GAME.step(s, COLS - 1 - col if mirrored else col)
            if GAME.is_terminal(child):
                continue
            replies = [GAME.step(child, a) for a in GAME.legal_actions(child)]
            if any(GAME.is_terminal(c) for c in replies):
                raise ValueError(f"a Yellow reply ends the game below {s.board}")
            stack.extend(reversed(replies))
            continue
        diagram = [row[::-1] for row in diagrams[value]] if mirrored else diagrams[value]
        key = (value, mirrored)
        if key not in maps:
            maps[key] = {"id": len(maps) + 1, "diagram": value, "mirrored": mirrored, "state": s}
        refs.append(maps[key]["id"])
        leaves.append({"plies": plies, "map": maps[key]["id"], "state": s, "diagram": diagram})
    return {"moves": moves, "leaves": leaves, "refs": refs, "maps": maps, "nodes": len(seen)}


def judge(tree: dict) -> dict:
    out = {}
    for name, (rule, sparse) in READINGS.items():
        facts = Facts(GAME, **rule)
        won, reasons, own, n_levels = 0, Counter(), [], 1
        for leaf in tree["leaves"]:
            if leaf["map"] is None:
                continue
            levels = _levels(leaf["diagram"], leaf["state"], sparse)
            n_levels = max(n_levels, len(set(levels.values())))
            r = verify(facts, leaf["state"], levels, 10, cap=50_000_000)
            won += r["won"]
            reasons[r["reason"] or "won"] += 1
            own.append(r["own_positions"])
        level_bits = math.ceil(math.log2(n_levels))
        map_cost = sum(map_bits(_levels(m["diagram"], m["state"], sparse), sum(1 for v in m["state"].board if v == 0),
                                level_bits) for m in _first_diagrams(tree))
        refs = code_bits("first_use", tree["refs"], len(tree["maps"]) + 1, 0)
        bits = tree["nodes"] + 3 * len(tree["moves"]) + refs + map_cost
        out[name] = {"won": won, "leaves": sum(1 for x in tree["leaves"] if x["map"] is not None),
                     "reasons": dict(reasons), "n_levels": n_levels, "level_bits": level_bits,
                     "bits": {"total": bits, "nodes": tree["nodes"], "moves": 3 * len(tree["moves"]),
                              "references": refs, "maps": map_cost},
                     "own_positions": own}
    return out


def _first_diagrams(tree: dict) -> list:
    first = {}
    for leaf in tree["leaves"]:
        if leaf["map"] is not None and leaf["map"] not in first:
            first[leaf["map"]] = {"diagram": leaf["diagram"], "state": leaf["state"]}
    return list(first.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weakc4", required=True, help="a local checkout of https://github.com/2swap/WeakC4")
    ap.add_argument("--out", default=str(ROOT / "evidence" / "c49_weakc4.json.gz"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    solution = Path(args.weakc4) / "solution"
    raw = {name: (solution / name).read_bytes() for name in ("branches.json", "steady_states.json")}
    commit = subprocess.run(["git", "-C", args.weakc4, "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    branches = json.loads(raw["branches.json"])
    diagrams = json.loads(raw["steady_states.json"])
    tree = walk(branches, diagrams)
    readings = judge(tree)
    save_evidence(out, {
        "started": started, "weakc4_commit": commit,
        "weakc4_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()},
        "entries": len(branches), "diagrams": len(diagrams), "nodes": tree["nodes"],
        "moves_by_plies": dict(sorted(Counter(m["plies"] for m in tree["moves"]).items())),
        "leaves_by_plies": dict(sorted(Counter(x["plies"] for x in tree["leaves"] if x["map"] is not None).items())),
        "won_at_once_by_plies": dict(sorted(Counter(x["plies"] for x in tree["leaves"] if x["map"] is None).items())),
        "maps": len(tree["maps"]), "mirrored_maps": sum(1 for m in tree["maps"].values() if m["mirrored"]),
        "readings": readings})
    for name, r in readings.items():
        print(name, r["won"], "/", r["leaves"], r["reasons"], r["bits"])


if __name__ == "__main__":
    main()
