"""§3.6 T1 — the whole game as one file. Every whole-game size so far is a count: each part priced by a formula, not
written down. T1 writes the game. `map_bits` pays one bit per cell empty at the position a map was FOUND, and the
assembled game never states that position; for 337 of the 801 maps Z2's game plays, no leaf has every levelled cell
empty. The count also leaves out how a decoder tells a piece from a trivial position and which list a piece's references
point into. T1 takes the opening's moves and trivial positions and the pieces the game plays (`t1_pieces`), reads them
as one strategy over positions, a position reached by two pieces taking the node of the piece reaching it first
(`t1_source`), drops from each map the cells full at every leaf playing it (harness.strategy_file.trim_maps), and
writes it with harness.strategy_file. The size is the file's length in bits. A decoder has the bytes, the game's rules,
the map rule's settings (`rule`) and the board's mirror; it reads the strategy back, which must equal what was written
(`t1_differences`), and the decoded strategy is played against every reply, every line ending in a win. Judged on that
and on the file's length against the 162,752 bits Z2's game was counted at: supported at 9/10 of it or less, refuted at
all of it or more. On G3's game, the dry run, the file took 225,678 bits against 280,640 counted."""
from __future__ import annotations

from harness.opening import frontier
from harness.steady_state import Facts
from harness.strategy_file import trim_maps

RULE_DEFAULTS = {"ties": "exact", "unlevelled": "none", "safe_depth": 1, "forcing": False, "order": "lowest"}
SPEC = {"rule": {"ties": "lowest", "unlevelled": "last", "safe_depth": 1, "forcing": False, "order": "lowest",
                 "n_levels": 8},
        "plies": 8, "control_bits": 162_752, "support": (9, 10), "refute": (1, 1), "file": "c49_t1.json.gz"}


def mirror_cells(rows: int, cols: int) -> list:
    return [r * cols + cols - 1 - c for r in range(rows) for c in range(cols)]


def t1_rule(builders: list) -> dict:
    """The one rule every builder in `builders` played by; refuses builders that differ."""
    rules = [{**{k: b.get(k, v) for k, v in RULE_DEFAULTS.items()}, "n_levels": b["n_levels"]} for b in builders]
    if any(r != rules[0] for r in rules):
        raise ValueError("the pieces were not built under one rule")
    return rules[0]


def t1_opening(game, positions: list, builder: dict, plies: int) -> dict:
    """{"moves", "trivial"} to `plies`, as harness.opening made them; `positions` are the label cache's rows."""
    book = {(tuple(r["board"]), r["to_move"]): {int(a) for a, v in r["values"].items() if v == 1} for r in positions}

    def winning(states):
        return [book[game.state_key(s)] for s in states]

    found = frontier(Facts(game), game.initial_state(), winning, plies, builder["n_levels"], builder["cap"])
    return {"moves": found["moves"], "trivial": set(found["trivial"])}


def t1_pieces(taken, mirrors: dict) -> dict:
    """{root position: (row, reads mirrored ids)}; `taken()` gives (run name, label, row) as harness.floor_z2._taken."""
    pieces: dict = {}
    for name, _, r in taken():
        key = (tuple(r["board"]), r["to_move"])
        if key in pieces:
            raise ValueError(f"two pieces at {key}")
        pieces[key] = (r, mirrors[name])
    return pieces


def _nodes(row: dict, mirror: bool, ids: dict, maps: list) -> dict:
    half = row["shared"] // 2 if mirror else 0
    out: dict = {}
    for board, to_move, node in row["nodes"]:
        if "move" in node:
            out[(tuple(board), to_move)] = {"move": node["move"]}
            continue
        levels = {int(c): k for c, k in row["maps"][node["leaf"]].items()}
        content = tuple(sorted(levels.items()))
        if content not in ids:
            ids[content] = len(maps)
            maps.append(levels)
        read = {"leaf": ids[content], **({"mirrored": True} if 1 <= node["leaf"] <= half else {})}
        if node.get("exceptions"):
            read["exceptions"] = {(tuple(b), t): m for b, t, m in node["exceptions"]}
        out[(tuple(board), to_move)] = read
    return out


def t1_source(game, root, opening: dict, pieces: dict) -> dict:
    """The strategy the opening and `pieces` make, each position reached taking the node of the first piece to reach
    it. A position nothing covers raises KeyError."""
    ids: dict = {(): 0}
    maps: list = [{}]
    read = {key: _nodes(row, mirror, ids, maps) for key, (row, mirror) in pieces.items()}
    nodes: dict = {}
    seen: set = set()
    stack = [(root, None)]
    while stack:
        s, piece = stack.pop()
        key = game.state_key(s)
        if key in seen or game.is_terminal(s):
            continue
        seen.add(key)
        if game.current_player(s) != game.current_player(root):
            stack.extend(reversed([(game.step(s, b), piece) for b in game.legal_actions(s)]))
            continue
        if piece is None and key in read:
            piece = read[key]
        if piece is not None:
            node = piece[key]
        elif key in opening["moves"]:
            node = {"move": opening["moves"][key]}
        elif key in opening["trivial"]:
            node = {"leaf": 0}
        else:
            raise KeyError(key)
        nodes[key] = node
        if "move" in node:
            stack.append((game.step(s, node["move"]), piece))
    return {"nodes": nodes, "maps": maps}


def t1_game(facts: Facts, root, cells: list, opening: dict, pieces: dict) -> dict:
    """The strategy to write: `t1_source`'s, with each map trimmed to the leaves that play it."""
    return trim_maps(facts, root, cells, t1_source(facts.game, root, opening, pieces))


def _content(node: dict, maps: list) -> tuple:
    if "move" in node:
        return ("move", node["move"])
    return ("leaf", tuple(sorted(maps[node["leaf"]].items())), bool(node.get("mirrored")),
            tuple(sorted(node.get("exceptions", {}).items())))


def t1_differences(written: dict, read: dict) -> int:
    """How many positions differ between two strategies, by what their nodes play, however their maps are numbered."""
    keys = set(written["nodes"]) | set(read["nodes"])
    return sum(k not in written["nodes"] or k not in read["nodes"]
               or _content(written["nodes"][k], written["maps"]) != _content(read["nodes"][k], read["maps"])
               for k in keys)


def t1_report(e: dict, read: dict, spec: dict) -> dict:
    """`read` is what a decoder made of the file now: its bits, its differences from the game, its leaves."""
    if e.get("spec") != "harness.floor_t1":
        return {"decoded": "not_run", "size": "not_run"}
    replay = e["replay"]
    if (read["bits"] != e["bits"] or read["differences"] or not replay["won"] or replay["leaves"] != read["leaves"]):
        return {"decoded": "refuted", "size": "not_run"}
    num, den = spec["support"]
    rnum, rden = spec["refute"]
    control, bits = spec["control_bits"], e["bits"]
    return {"decoded": "supported", "size": ("supported" if den * bits <= num * control else
                                             "refuted" if rden * bits >= rnum * control else "inconclusive")}
