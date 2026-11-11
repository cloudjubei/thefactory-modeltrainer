"""§3.6 T8 — harness.strategy_file's walk with two things read from the position, which a wrapper of its coder never
sees. A leaf names a map it reads again by its place in the known maps ranked by `likeness`: how many cells of the
leaf's board agree with the best of the boards the map was read at before, either way round, ties in move-to-front
order; the decoder has every board before the reference, and the mirrored flag after it, so the ranking does not
depend on it. And each exception flag is coded under harness.exception_context's context and how many plies below
the leaf its position is, on a log scale. Everything else is strategy_file's walk, with its maps coded under
harness.level_context's and harness.cell_context's wrappers; strategy_file itself is pinned by T1–T5's claims and is
not changed."""
from __future__ import annotations

import numpy as np

from harness.arithmetic_coding import Decoder, Encoder
from harness.cell_context import CellContext
from harness.exception_context import exception_context
from harness.level_context import LevelContext
from harness.mirror_builder import board_image
from harness.steady_state import Facts, choose
from harness.strategy_file import _map, _reference


def depth_context(plies: int) -> int:
    return (plies + 1).bit_length() - 1


def likeness(boards: list, mirrored: list, used: list, board, n_maps: int) -> np.ndarray:
    """For each of `n_maps` maps, how many cells of `board` agree with the best of the boards it was read at
    (`boards[i]` read `used[i]`, `mirrored[i]` its mirror image), and -1 for a map not read yet."""
    best = np.full(n_maps, -1)
    if boards:
        agree = np.maximum((np.array(boards) == board).sum(1), (np.array(mirrored) == board).sum(1))
        np.maximum.at(best, np.array(used), agree)
    return best


def ranked(order: list, score) -> list:
    """`order`, the move-to-front list, best `score` first; a stable sort keeps move-to-front among ties."""
    return sorted(order, key=lambda k: -score[k])


def _board(facts: Facts, s) -> np.ndarray:
    game = facts.game
    return np.array(game.observation(s, game.current_player(s))[:facts.cells])


def _exceptions(io, facts: Facts, root, levels: dict, n_levels: int, known) -> dict:
    game = facts.game
    player = game.current_player(root)
    empty = len(facts.empty_cells(root))
    out: dict = {}
    seen: set = set()
    stack = [root]
    ones = gap = 0
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
            context = (*exception_context(ones, gap), depth_context(empty - len(facts.empty_cells(s))))
            flagged = io.symbol("exception_flags", context, 2, None if known is None else int(fix is not None))
            ones, gap = (ones + 1, 0) if flagged else (ones, gap + 1)
            picks = [m for m in safe if m != move] if flagged else None
        if picks is not None:
            move = picks[io.uniform("exception_moves", len(picks), None if known is None else picks.index(fix))]
            out[key] = move
        stack.append(game.step(s, move))
    return out


def _leaf(io, facts: Facts, s, image, n_levels: int, table: dict, node, source):
    maps, order, known, read = table["maps"], table["order"], table["known"], table["read"]
    board = _board(facts, s)
    choices = ranked(order, likeness(read["boards"], read["mirrored"], read["used"], board, len(maps)))
    at = None if node is None else choices.index(known[node["leaf"]]) + 1 if node["leaf"] in known else 0
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
        index = choices[at - 1]
        order.remove(index)
        if index == 0 and flag:
            raise ValueError("the empty map is read one way")
        mirrored = index != 0 and bool(io.symbol("mirrored", "old", 2, flag))
    order.insert(0, index)
    read["boards"].append(board)
    read["mirrored"].append(_board(facts, image(s)))
    read["used"].append(index)
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
    """Walks the strategy from `root` through `io` as harness.strategy_file.code_strategy does, with this module's
    references and exception flags."""
    game = facts.game
    image = board_image(cells)
    player = game.current_player(root)
    table: dict = {"maps": [{}], "order": [0], "known": {0: 0}, "read": {"boards": [], "mirrored": [], "used": []}}
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


def _coder(io, facts: Facts) -> LevelContext:
    cols = facts.game.board_shape[1]
    return LevelContext(CellContext(io, cols), cols)


def encode_strategy(facts: Facts, root, n_levels: int, cells: list, source: dict) -> tuple:
    """(the file's bytes, its length in bits, each part's ideal bits)."""
    io = Encoder()
    code_strategy(_coder(io, facts), facts, root, n_levels, cells, source)
    data, bits = io.finish()
    return data, bits, io.parts


def decode_strategy(facts: Facts, root, n_levels: int, cells: list, data: bytes) -> dict:
    return code_strategy(_coder(Decoder(data), facts), facts, root, n_levels, cells)
