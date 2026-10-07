"""Direct tests for harness/strategy_builder.py — a complete certified strategy as table moves plus steady-state
leaves, built depth first, and checked without any oracle. Tic-tac-toe fixtures checked by hand: in X{0,1} O{2,3},
X to move, only the centre wins (the map {4: 0}); after it, every O reply leaves X an immediate win."""
from __future__ import annotations

import pytest

from games.tictactoe import TicTacToe, TTTState
from harness.steady_search import find_steady_state
from harness.steady_state import Facts
from harness.strategy_builder import Builder, check

GAME = TicTacToe()


def _ttt(x, o):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=0 if len(x) == len(o) else 1, winner=None, done=False)


FORK = _ttt((0, 1), (2, 3))


def _winning(states):
    out = []
    for s in states:
        me = GAME.current_player(s)
        moves = set()
        for a in GAME.legal_actions(s):
            child = GAME.step(s, a)
            if (child.done and child.winner == me) or (not child.done and GAME.position_value(child) == -1):
                moves.add(a)
        out.append(moves)
    return out


def _sat(facts):
    def search(s, start):
        r = find_steady_state(facts, s, _winning, 2, max_constraints=500, conflicts=100_000, seconds=30.0)
        return {"levels": r["levels"], "exceptions": {}, "own_positions": None} if r["status"] == "found" else None
    return search


def _builder(min_leaf_depth=0, seconds=60.0, search=None, accept=None):
    facts = Facts(GAME)
    return Builder(facts, _winning, search or _sat(facts), n_levels=2, level_bits=3, cap=100_000,
                   min_leaf_depth=min_leaf_depth, reuse_window=50, seconds=seconds, accept=accept)


def test_a_searched_leaf_at_the_root_is_the_whole_strategy():
    b = _builder()
    assert b.build(FORK)
    assert b.nodes == {GAME.state_key(FORK): {"leaf": 1}} and list(b.maps[1]) == [4]
    assert b.bits() == {"nodes": 1 + 1 + (5 + 3), "moves": 0, "leaves": 1, "maps": 1, "map_bits": 8,
                        "exceptions": 0, "exception_bits": 0}
    assert check(b.facts, FORK, b.nodes, b.maps, 2, 100_000) == {"won": True, "reason": None, "own_positions": 5}


def test_without_searching_the_root_becomes_a_move_and_its_children_trivial_leaves():
    b = _builder(min_leaf_depth=99)
    assert b.build(FORK)
    root = b.nodes.pop(GAME.state_key(FORK))
    assert root == {"move": 4} and len(b.nodes) == 4 and all(n == {"leaf": 0} for n in b.nodes.values())
    b.nodes[GAME.state_key(FORK)] = root
    assert b.bits() == {"nodes": 5 + 4, "moves": 1, "leaves": 4, "maps": 0, "map_bits": 0, "exceptions": 0,
                        "exception_bits": 0}
    assert check(b.facts, FORK, b.nodes, b.maps, 2, 100_000)["won"]


def test_a_whole_game_from_an_early_win_builds_and_checks_and_every_move_wins():
    root = _ttt((0,), (1,))
    b = _builder()
    assert b.build(root)
    assert check(b.facts, root, b.nodes, b.maps, 2, 100_000)["won"]
    states = {}
    stack = [root]
    while stack:
        s = stack.pop()
        k = GAME.state_key(s)
        if k in states or s.done:
            continue
        states[k] = s
        if s.to_move == 0 and k in b.nodes and "move" in b.nodes[k]:
            stack.append(GAME.step(s, b.nodes[k]["move"]))
        elif s.to_move == 1:
            stack.extend(GAME.step(s, a) for a in GAME.legal_actions(s))
    moves = [(states[k], n["move"]) for k, n in b.nodes.items() if "move" in n]
    assert all(m in w for (s, m), w in zip(moves, _winning([s for s, _ in moves])))


def test_a_known_map_covers_a_position_without_searching():
    b = _builder(search=lambda s, start: pytest.fail("searched"))
    b.maps.append({4: 0, 8: 1})
    b.map_empty.append(5)
    assert b.build(FORK) and b.nodes[GAME.state_key(FORK)] == {"leaf": 1}


def test_the_deadline_stops_the_build():
    assert not _builder(seconds=0.0).build(FORK)


def test_a_root_its_side_cannot_win_is_refused():
    with pytest.raises(ValueError, match="cannot win"):
        _builder(min_leaf_depth=99).build(TicTacToe().initial_state())


def test_check_rejects_a_missing_node_and_a_losing_leaf():
    facts = Facts(GAME)
    assert check(facts, FORK, {}, [{}], 2, 100_000)["reason"] == "missing node"
    bad = {GAME.state_key(FORK): {"leaf": 1}}
    assert check(facts, FORK, bad, [{}, {6: 0}], 2, 100_000)["reason"] == "leaf draw"
    lost = {GAME.state_key(FORK): {"move": 5}}
    r = check(facts, FORK, lost, [{}], 2, 100_000)
    assert not r["won"] and r["reason"] in {"missing node", "lost line"}


def test_moves_are_chosen_to_make_the_most_children_ready():
    b = _builder(min_leaf_depth=99)
    root = _ttt((0,), (1,))
    assert b.build(root)
    first = b.nodes[GAME.state_key(root)]["move"]
    assert first in _winning([root])[0]


def test_a_found_map_is_simplified_before_it_is_kept():
    b = _builder(search=lambda s, start: {"levels": {4: 0, 5: 1, 6: 1}, "exceptions": {}, "own_positions": None})
    assert b.build(FORK) and b.maps[1] == {4: 0}


def test_a_failed_search_makes_a_move_node_not_an_empty_leaf():
    b = _builder(search=lambda s, start: None)
    assert b.build(FORK) and b.nodes[GAME.state_key(FORK)] == {"move": 4}


def test_the_move_whose_children_are_all_ready_is_preferred_over_a_lower_numbered_one():
    root = _ttt((0,), (5,))
    assert sorted(_winning([root])[0]) == [2, 4, 6]
    b = _builder(min_leaf_depth=99)
    assert b.build(root) and b.nodes[GAME.state_key(root)] == {"move": 4}


def test_check_walks_every_reply_not_just_the_first():
    after = GAME.step(FORK, 4)
    first_reply = GAME.step(after, GAME.legal_actions(after)[0])
    nodes = {GAME.state_key(FORK): {"move": 4}, GAME.state_key(first_reply): {"leaf": 0}}
    assert check(Facts(GAME), FORK, nodes, [{}], 2, 100_000)["reason"] == "missing node"


def test_check_rejects_a_line_the_opponent_wins():
    root = _ttt((0, 8), (4, 2))
    after = GAME.step(root, 3)
    nodes = {GAME.state_key(root): {"move": 3}}
    for b in GAME.legal_actions(after):
        child = GAME.step(after, b)
        if not child.done:
            nodes[GAME.state_key(child)] = {"leaf": 0}
    assert check(Facts(GAME), root, nodes, [{}], 2, 100_000)["reason"] == "lost line"


ROOT_FIX = {GAME.state_key(FORK): 4}


def _excepted(levels=None, exceptions=None, own=5):
    return lambda s, start: {"levels": levels or {}, "exceptions": dict(ROOT_FIX if exceptions is None else exceptions),
                             "own_positions": own}


def test_a_leaf_with_exceptions_is_kept_when_it_is_small_enough():
    b = _builder(search=_excepted(), accept=1)
    assert b.build(FORK)
    assert b.nodes == {GAME.state_key(FORK): {"leaf": 0, "exceptions": ROOT_FIX, "own": 5, "coded": 3}}
    assert b.searches == {"tried": 1, "found": 0, "excepted": 1}
    assert check(b.facts, FORK, b.nodes, b.maps, 2, 100_000) == {"won": True, "reason": None, "own_positions": 5}


def test_a_leaf_with_exceptions_too_large_for_the_threshold_becomes_a_move():
    """Its size: no map bits (the empty map) + the walk-order code — the root is undefined, so its exception needs no
    flag, only its move among 5 safe ones (3 bits): 15 table bits over 3, 5x."""
    b = _builder(search=_excepted(), accept=6)
    assert b.build(FORK) and b.nodes[GAME.state_key(FORK)] == {"move": 4}


def test_a_leaf_exactly_at_the_threshold_is_kept():
    """3 coded bits against a 15-bit table — exactly 5x."""
    b = _builder(search=_excepted(), accept=5)
    assert b.build(FORK) and b.nodes[GAME.state_key(FORK)]["exceptions"] == ROOT_FIX


def test_without_a_threshold_a_leaf_with_exceptions_is_never_kept():
    b = _builder(search=_excepted(), accept=None)
    assert b.build(FORK) and b.nodes[GAME.state_key(FORK)] == {"move": 4}


def test_a_leaf_s_map_is_kept_beside_its_exceptions_and_charged_in_its_size():
    b = _builder(search=_excepted(levels={5: 1}, own=1), accept=0.01)
    assert b.build(FORK)
    assert b.nodes[GAME.state_key(FORK)]["leaf"] == 1 and b.maps[1] == {5: 1}


def test_exceptions_are_charged_a_flag_per_leaf_and_their_walk_order_code():
    b = _builder(search=_excepted(), accept=1)
    assert b.build(FORK)
    assert b.bits() == {"nodes": 1 + 1 + 3, "moves": 0, "leaves": 1, "maps": 0, "map_bits": 0, "exceptions": 1,
                        "exception_bits": 1 + 3}


def test_the_search_starts_from_the_most_recent_map_on_the_cells_still_empty():
    starts = []

    def search(s, start):
        starts.append(start)
        return None

    b = _builder(search=search)
    b.maps.append({5: 1, 0: 0})
    b.map_empty.append(6)
    b.build(FORK)
    assert starts[0] == {5: 1}


def test_check_plays_a_leaf_s_exceptions():
    facts = Facts(GAME)
    with_fix = {GAME.state_key(FORK): {"leaf": 0, "exceptions": ROOT_FIX, "own": 5}}
    assert check(facts, FORK, with_fix, [{}], 2, 100_000)["won"]
    assert check(facts, FORK, {GAME.state_key(FORK): {"leaf": 0}}, [{}], 2, 100_000)["reason"] == "leaf undefined"


def test_check_rejects_a_losing_exception():
    bad = {GAME.state_key(FORK): {"leaf": 0, "exceptions": {GAME.state_key(FORK): 6}, "own": 5}}
    assert not check(Facts(GAME), FORK, bad, [{}], 2, 100_000)["won"]


def test_the_threshold_counts_the_leaf_s_map_bits():
    """own 7, map {5: 1}: 5 + 3 map bits + 3 coded bits (flag 1 of 1, move 2 of 4) against a 21-bit table — 1.9x,
    though 7x without the map."""
    b = _builder(search=_excepted(levels={5: 1}, own=7), accept=2)
    assert b.build(FORK) and b.nodes[GAME.state_key(FORK)] == {"move": 4}
