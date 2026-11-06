"""§3.6 — a strategy as a file. A strategy here is `{"nodes": {position key: node}, "maps": [{cell: level}, ...]}`, a
node either {"move": a} or {"leaf": index into maps, "mirrored": True when it reads the map on the mirrored board,
"exceptions": {position key: move}}, and maps[0] the empty map. `code_strategy` is one walk from the root, in the order
harness.reference_coding reads a build, that writes such a strategy through harness.arithmetic_coding or reads it back:
each position once, whichever way it is reached. A decoder has the bytes, the game's rules, the map rule's settings
and the board's mirror, and nothing else: no list of maps, no position a map was found at.

  kind       — is the position a leaf, by how many stones are on the board;
  move       — which of the safe moves (a winning move is always safe);
  reference  — a move-to-front rank among the maps read so far, 0 for a map not read yet: its bit length, then its
               low bits;
  mirrored   — which way a leaf reads a map other than the empty one;
  cells      — a new map's cells, each levelled or not, by the cell and whether it is empty at the leaf that reads the
               map first (the only position the decoder can use);
  levels     — each levelled cell's level, by the cell;
  excepted   — does the leaf have exceptions, then harness.exception_coding's walk with its flags and moves coded here.

`trim_maps` drops from each map the cells full at every leaf that plays it: a leaf restricts a map to the cells empty
at its position and stones never leave, so no move changes. `replay` plays a strategy against every reply and accepts
only if every line ends in a win, without any solver."""
from __future__ import annotations

from harness.arithmetic_coding import Decoder, Encoder
from harness.mirror_builder import _mirrored_walk, board_image
from harness.steady_exceptions import verify_with
from harness.steady_state import Facts, choose, restrict

REFERENCE_WIDTHS = 16


def _reference(io, at):
    """`at` is the map's place in the move-to-front list, from 1, or 0 for a new map."""
    rank = None if at is None else at + 1
    width = io.symbol("reference", 0, REFERENCE_WIDTHS, None if rank is None else rank.bit_length() - 1)
    low = io.uniform("reference", 1 << width, None if rank is None else rank - (1 << width))
    return (1 << width) + low - 1


def _map(io, facts: Facts, frame: set, n_levels: int, levels) -> dict:
    out = {}
    for cell in range(facts.cells):
        if io.symbol("cells", (cell, cell in frame), 2, None if levels is None else int(cell in levels)):
            out[cell] = io.symbol("levels", cell, n_levels, None if levels is None else levels[cell])
    return out


def _exceptions(io, facts: Facts, root, levels: dict, n_levels: int, known) -> dict:
    game = facts.game
    player = game.current_player(root)
    out: dict = {}
    seen: set = set()
    stack = [root]
    while stack:
        s = stack.pop()
        key = game.state_key(s)
        if key in seen or game.is_terminal(s):
            continue
        seen.add(key)
        if game.current_player(s) != player:
            stack.extend(game.step(s, b) for b in game.legal_actions(s))
            continue
        safe = facts.safe_moves(s)
        move = choose(facts, s, levels, n_levels)
        fix = None if known is None else known.get(key)
        if move is None:
            picks = safe
        elif facts.of(s)[0] or len(safe) == 1:
            if fix is not None:
                raise ValueError(f"an exception at {key}, where the rule is forced: a decoder never reads one there")
            picks = None
        else:
            flagged = io.symbol("exception_flags", 0, 2, None if known is None else int(fix is not None))
            picks = [m for m in safe if m != move] if flagged else None
        if picks is not None:
            move = picks[io.uniform("exception_moves", len(picks), None if known is None else picks.index(fix))]
            out[key] = move
        stack.append(game.step(s, move))
    return out


def _leaf(io, facts: Facts, s, image, n_levels: int, table: dict, node, source):
    maps, order, known = table["maps"], table["order"], table["known"]
    at = None if node is None else order.index(known[node["leaf"]]) + 1 if node["leaf"] in known else 0
    at = _reference(io, at)
    flag = None if node is None else int(bool(node.get("mirrored")))
    if at == 0:
        mirrored = bool(io.symbol("mirrored", "new", 2, flag))
        frame = set(facts.empty_cells(image(s) if mirrored else s))
        index = len(maps)
        maps.append(_map(io, facts, frame, n_levels, None if node is None else source["maps"][node["leaf"]]))
        if node is not None:
            known[node["leaf"]] = index
    else:
        index = order.pop(at - 1)
        if index == 0 and flag:
            raise ValueError("the empty map is read one way")
        mirrored = index != 0 and bool(io.symbol("mirrored", "old", 2, flag))
    order.insert(0, index)
    out = {"leaf": index, **({"mirrored": True} if mirrored else {})}
    wanted = None if node is None else node.get("exceptions")
    if mirrored:
        if wanted:
            raise ValueError("a mirrored reading has no exceptions")
        return out
    if io.symbol("excepted", (at == 0, index == 0), 2, None if node is None else int(bool(wanted))):
        out["exceptions"] = _exceptions(io, facts, s, maps[index], n_levels, None if node is None else wanted)
    return out


def code_strategy(io, facts: Facts, root, n_levels: int, cells: list, source=None) -> dict:
    """Walks the strategy from `root` through `io`: an Encoder with the strategy as `source`, or a Decoder with none.
    Returns the strategy as read, its maps numbered in the order the walk first reads them."""
    game = facts.game
    image = board_image(cells)
    player = game.current_player(root)
    table: dict = {"maps": [{}], "order": [0], "known": {0: 0}}
    nodes: dict = {}
    seen: set = set()
    stack = [root]
    while stack:
        s = stack.pop()
        key = game.state_key(s)
        if key in seen or game.is_terminal(s):
            continue
        seen.add(key)
        if game.current_player(s) != player:
            stack.extend(reversed([game.step(s, b) for b in game.legal_actions(s)]))
            continue
        node = None if source is None else source["nodes"][key]
        stones = facts.cells - len(facts.empty_cells(s))
        if io.symbol("kind", stones, 2, None if node is None else int("leaf" in node)):
            nodes[key] = _leaf(io, facts, s, image, n_levels, table, node, source)
            continue
        safe = facts.safe_moves(s)
        move = safe[io.symbol("move", 0, len(safe), None if node is None else safe.index(node["move"]))]
        nodes[key] = {"move": move}
        stack.append(game.step(s, move))
    return {"nodes": nodes, "maps": table["maps"]}


def encode_strategy(facts: Facts, root, n_levels: int, cells: list, source: dict) -> tuple:
    """(the file's bytes, its length in bits, each part's ideal bits)."""
    io = Encoder()
    code_strategy(io, facts, root, n_levels, cells, source)
    data, bits = io.finish()
    return data, bits, io.parts


def decode_strategy(facts: Facts, root, n_levels: int, cells: list, data: bytes) -> dict:
    return code_strategy(Decoder(data), facts, root, n_levels, cells)


def leaf_states(game, root, nodes: dict) -> dict:
    """{"won": every line above the leaves ends in a win, "leaves": [(state, node)] in walk order}. A position the
    strategy reaches without a node raises KeyError."""
    player = game.current_player(root)
    won = True
    leaves: list = []
    seen: set = set()
    stack = [root]
    while stack:
        s = stack.pop()
        key = game.state_key(s)
        if key in seen:
            continue
        seen.add(key)
        if game.is_terminal(s):
            won = won and game.winner(s) == player
            continue
        if game.current_player(s) != player:
            stack.extend(reversed([game.step(s, b) for b in game.legal_actions(s)]))
            continue
        node = nodes[key]
        if "leaf" in node:
            leaves.append((s, node))
        else:
            stack.append(game.step(s, node["move"]))
    return {"won": won, "leaves": leaves}


def trim_maps(facts: Facts, root, cells: list, strategy: dict) -> dict:
    image = board_image(cells)
    live: dict = {}
    for s, node in leaf_states(facts.game, root, strategy["nodes"])["leaves"]:
        read = image(s) if node.get("mirrored") else s
        live.setdefault(node["leaf"], set()).update(facts.empty_cells(read))
    maps = [{c: k for c, k in m.items() if c in live.get(i, ())} for i, m in enumerate(strategy["maps"])]
    return {"nodes": strategy["nodes"], "maps": maps}


def replay_leaf(facts: Facts, state, node: dict, maps: list, n_levels: int, cap: int, cells: list) -> bool:
    if node.get("mirrored"):
        levels = restrict(maps[node["leaf"]], facts.empty_cells(board_image(cells)(state)))
        return _mirrored_walk(facts, state, levels, n_levels, cap, cells)["won"]
    levels = restrict(maps[node["leaf"]], facts.empty_cells(state))
    return verify_with(facts, state, levels, n_levels, node.get("exceptions", {}), cap)["won"]


def replay(facts: Facts, root, strategy: dict, n_levels: int, cap: int, cells: list, check=None) -> dict:
    """Every line from `root` played out. `check`, given the leaves, stands in for playing them here one by one."""
    found = leaf_states(facts.game, root, strategy["nodes"])
    leaves = found["leaves"]
    if not found["won"]:
        return {"won": False, "leaves": len(leaves), "reason": "lost line above the leaves"}
    won = (check(leaves) if check else
           all(replay_leaf(facts, s, n, strategy["maps"], n_levels, cap, cells) for s, n in leaves))
    return {"won": won, "leaves": len(leaves), "reason": None if won else "a leaf failed"}
