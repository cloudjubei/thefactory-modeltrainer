"""§3.6 — harness.strategy_builder's strategies, with leaves that may read a library map on the mirrored board. A map
is cell-indexed and its rule breaks ties by the lowest move, so a map with its cells flipped is another map; read on the
board's mirror image, with the move mirrored back, it plays the same game as where it was found. The library comes
twice: each map first as its mirrored reading, then as its plain one, so `self.shared` is twice the maps and a leaf's
reference says which reading it plays; ids 1..`mirrored` read mirrored. Readings are tried like any known maps, most
recent first, so a build's own maps and every plain reading come before any mirrored one. `cells` is the symmetry, a
reflection of the board's cells, found by scripts/c4_mirror_sharing.py.

The builder trusts the symmetry to find a leaf (it verifies the map on the image), but `mirror_check` and
`mirrored_move` play on the real board: every reply is the real game's, so a strategy that passes does not rest on it."""
from __future__ import annotations

from dataclasses import replace

from harness.steady_exceptions import verify_with
from harness.steady_state import Facts, choose, restrict
from harness.strategy_builder import Builder, _own_under


def board_image(cells: list):
    def image(state):
        board = [0] * len(state.board)
        for i, v in enumerate(state.board):
            board[cells[i]] = v
        return replace(state, board=tuple(board))

    return image


def mirrored_move(facts: Facts, cells: list, state, levels: dict, n_levels: int):
    """The move at `state` of the map `levels` read on the mirrored board, or None where the rule has none."""
    image = board_image(cells)(state)
    move = choose(facts, image, levels, n_levels)
    if move is None:
        return None
    target = cells[facts.placed_cell(image, move)]
    return next(a for a in facts.game.legal_actions(state) if facts.placed_cell(state, a) == target)


class MirrorBuilder(Builder):
    def __init__(self, *args, cells: list, **kwargs):
        super().__init__(*args, **kwargs)
        self.image, self.mirrored = board_image(cells), self.shared // 2

    def _covering_map(self, state):
        recent = range(len(self.maps) - 1, max(0, len(self.maps) - 1 - self.reuse_window), -1)
        image = self.image(state)
        for i in [0, *recent]:
            read = image if 1 <= i <= self.mirrored else state
            levels = restrict(self.maps[i], self.facts.empty_cells(read))
            if self.verifier(self.facts, read, levels, self.n_levels, self.cap, deadline=self.deadline)["won"]:
                return i
        return None


def _mirrored_walk(facts: Facts, root, levels: dict, n_levels: int, cap: int, cells: list) -> dict:
    game = facts.game
    player = game.current_player(root)
    seen, own, stack = set(), set(), [root]
    while stack:
        s = stack.pop()
        key = game.state_key(s)
        if key in seen:
            continue
        seen.add(key)
        if game.is_terminal(s):
            if game.winner(s) != player:
                return {"won": False, "reason": "lost line", "own": own}
            continue
        if game.current_player(s) != player:
            stack.extend(game.step(s, b) for b in game.legal_actions(s))
            continue
        own.add(key)
        if len(own) > cap:
            return {"won": False, "reason": "cap", "own": own}
        move = mirrored_move(facts, cells, s, levels, n_levels)
        if move is None:
            return {"won": False, "reason": "undefined", "own": own}
        stack.append(game.step(s, move))
    return {"won": True, "reason": None, "own": own}


def mirror_check(facts: Facts, root, nodes: dict, maps: list, n_levels: int, cap: int, cells: list,
                 mirrored: int) -> dict:
    """harness.strategy_builder.check for a strategy whose leaves 1..`mirrored` read mirrored."""
    game = facts.game
    image = board_image(cells)
    player = game.current_player(root)
    seen: set = set()
    own: set = set()
    stack = [root]
    while stack:
        s = stack.pop()
        key = game.state_key(s)
        if key in seen:
            continue
        seen.add(key)
        if game.is_terminal(s):
            if game.winner(s) != player:
                return {"won": False, "reason": "lost line", "own_positions": len(own)}
            continue
        if game.current_player(s) != player:
            stack.extend(game.step(s, b) for b in game.legal_actions(s))
            continue
        node = nodes.get(key)
        if node is None:
            return {"won": False, "reason": "missing node", "own_positions": len(own)}
        own.add(key)
        if "move" in node:
            stack.append(game.step(s, node["move"]))
            continue
        if 1 <= node["leaf"] <= mirrored:
            levels = restrict(maps[node["leaf"]], facts.empty_cells(image(s)))
            walk = _mirrored_walk(facts, s, levels, n_levels, cap, cells)
            if not walk["won"]:
                return {"won": False, "reason": f"leaf {walk['reason']}", "own_positions": len(own)}
            own.update(walk["own"])
            continue
        levels = restrict(maps[node["leaf"]], facts.empty_cells(s))
        exc = node.get("exceptions", {})
        walk = verify_with(facts, s, levels, n_levels, exc, cap)
        if not walk["won"]:
            return {"won": False, "reason": f"leaf {walk['reason']}", "own_positions": len(own)}
        own.update(_own_under(facts, s, levels, n_levels, exc))
    return {"won": True, "reason": None, "own_positions": len(own)}
