"""Direct tests for harness/mirror_builder.py — strategies whose leaves may read a library map on the mirrored board.
Tic-tac-toe fixtures checked by hand and by harness.steady_state.verify: in X{0,2} O{1,3}, X to move, the map {6: 0}
loses read plainly and wins read on the column-flipped board; in X{0,1} O{2,3} only the centre wins, on either board."""
from __future__ import annotations

import pytest

import harness.mirror_builder as floor
from games.tictactoe import TicTacToe, TTTState
from harness.mirror_builder import MirrorBuilder, board_image, mirror_check, mirrored_move
from harness.steady_state import Facts, choose, restrict, verify
from harness.strategy_builder import check

GAME = TicTacToe()
CELLS = [2, 1, 0, 5, 4, 3, 8, 7, 6]
ONE_SIDED = {6: 0}
CENTRE = {4: 0}
NEITHER = {7: 0}


def _ttt(x, o):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=0 if len(x) == len(o) else 1, winner=None, done=False)


MIRROR_ONLY = _ttt((0, 2), (1, 3))
FORK = _ttt((0, 1), (2, 3))
PLAIN_ONLY = board_image(CELLS)(MIRROR_ONLY)
FACTS = Facts(GAME)
LOWEST = Facts(GAME, ties="lowest", unlevelled="last")


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


def _builder(library, search=None, verifier=verify):
    pairs = [(m, 5) for m in library]
    return MirrorBuilder(FACTS, _winning, search or (lambda s, start, seconds: None), n_levels=2, level_bits=3,
                         cap=100_000, min_leaf_depth=0, reuse_window=50, seconds=60.0, library=[*pairs, *pairs],
                         verifier=verifier, cells=CELLS)


def _won(state, levels):
    return verify(FACTS, state, restrict(levels, FACTS.empty_cells(state)), 2, 100_000)["won"]


def _key(state):
    return GAME.state_key(state)


def test_the_fixtures_win_on_the_boards_they_are_said_to():
    assert not _won(MIRROR_ONLY, ONE_SIDED) and _won(PLAIN_ONLY, ONE_SIDED)
    assert _won(FORK, CENTRE) and _won(board_image(CELLS)(FORK), CENTRE)


def test_the_image_flips_the_board_and_keeps_the_rest():
    image = board_image(CELLS)(MIRROR_ONLY)
    assert image.board == (1, 2, 1, 0, 0, 2, 0, 0, 0)
    assert (image.to_move, image.winner, image.done) == (MIRROR_ONLY.to_move, None, False)
    assert board_image(CELLS)(image) == MIRROR_ONLY


def test_a_mirrored_move_is_the_image_of_the_move_the_rule_plays_on_the_image():
    assert choose(FACTS, PLAIN_ONLY, ONE_SIDED, 2) == 6
    assert mirrored_move(FACTS, CELLS, MIRROR_ONLY, ONE_SIDED, 2) == 8


def test_a_mirrored_move_where_the_rule_has_none_is_none():
    state = _ttt((), ())
    assert mirrored_move(FACTS, CELLS, state, {}, 2) is None


def test_the_first_half_of_the_library_reads_mirrored():
    b = _builder([ONE_SIDED])
    assert b.build(MIRROR_ONLY) and b.nodes[_key(MIRROR_ONLY)] == {"leaf": 1}
    assert b.searches["tried"] == 0


def test_the_second_half_reads_plainly():
    b = _builder([ONE_SIDED])
    assert b.build(PLAIN_ONLY) and b.nodes[_key(PLAIN_ONLY)] == {"leaf": 2}


def test_a_map_that_wins_both_ways_is_read_plainly():
    b = _builder([CENTRE])
    assert b.build(FORK) and b.nodes[_key(FORK)] == {"leaf": 2}


def test_every_map_read_plainly_comes_before_any_read_mirrored():
    b = _builder([CENTRE, ONE_SIDED])
    assert b._covering_map(FORK) == 3
    b = _builder([ONE_SIDED, CENTRE])
    assert b._covering_map(FORK) == 4


def test_the_empty_map_and_the_build_s_own_maps_read_plainly():
    seen = []

    def spy(facts, state, levels, n_levels, cap, deadline=None):
        seen.append((state, levels))
        return {"won": False}

    levels = {6: 0, 3: 1, 5: 1}
    b = _builder([levels], verifier=spy)
    b.maps.append(levels)
    assert b._covering_map(MIRROR_ONLY) is None
    image = board_image(CELLS)(MIRROR_ONLY)
    assert seen == [(MIRROR_ONLY, {}), (MIRROR_ONLY, {6: 0, 5: 1}), (MIRROR_ONLY, {6: 0, 5: 1}), (image, {6: 0, 3: 1})]


def test_the_scan_stops_at_the_reuse_window():
    b = _builder([ONE_SIDED, ONE_SIDED])
    b.reuse_window = 2
    assert b._covering_map(MIRROR_ONLY) is None
    b.reuse_window = 3
    assert b._covering_map(MIRROR_ONLY) == 2


def test_a_position_no_reading_covers_goes_to_the_search():
    found = {"levels": ONE_SIDED, "exceptions": {}, "own_positions": None}
    b = _builder([NEITHER], search=lambda s, start, seconds: found)
    assert not _won(PLAIN_ONLY, NEITHER) and not _won(MIRROR_ONLY, NEITHER)
    assert b.build(PLAIN_ONLY) and b.searches["found"] == 1
    assert b.nodes[_key(PLAIN_ONLY)]["leaf"] == 3


def test_a_mirrored_leaf_checks_on_the_real_board():
    nodes = {_key(MIRROR_ONLY): {"leaf": 1}}
    maps = [{}, ONE_SIDED, ONE_SIDED]
    r = mirror_check(FACTS, MIRROR_ONLY, nodes, maps, 2, 100_000, CELLS, 1)
    assert r["won"] and r["reason"] is None and r["own_positions"] > 1
    plain = check(FACTS, PLAIN_ONLY, {_key(PLAIN_ONLY): {"leaf": 2}}, maps, 2, 100_000)
    assert r["own_positions"] == plain["own_positions"]


def test_a_mirrored_leaf_that_does_not_win_fails_the_check():
    nodes = {_key(PLAIN_ONLY): {"leaf": 1}}
    r = mirror_check(FACTS, PLAIN_ONLY, nodes, [{}, ONE_SIDED, ONE_SIDED], 2, 100_000, CELLS, 1)
    assert not r["won"] and "leaf" in r["reason"]


def test_a_plain_leaf_is_checked_plainly():
    maps = [{}, ONE_SIDED, ONE_SIDED]
    assert mirror_check(FACTS, PLAIN_ONLY, {_key(PLAIN_ONLY): {"leaf": 2}}, maps, 2, 100_000, CELLS, 1)["won"]
    r = mirror_check(FACTS, MIRROR_ONLY, {_key(MIRROR_ONLY): {"leaf": 2}}, maps, 2, 100_000, CELLS, 1)
    assert not r["won"] and "leaf" in r["reason"]


def test_the_check_counts_a_plain_leaf_s_positions_as_the_plain_check_does():
    nodes = {_key(PLAIN_ONLY): {"leaf": 2}}
    maps = [{}, ONE_SIDED, ONE_SIDED]
    assert (mirror_check(FACTS, PLAIN_ONLY, nodes, maps, 2, 100_000, CELLS, 1)
            == check(FACTS, PLAIN_ONLY, nodes, maps, 2, 100_000))


def test_the_check_plays_table_moves_and_finds_missing_nodes():
    assert mirror_check(FACTS, FORK, {}, [{}], 2, 100_000, CELLS, 0) == {"won": False, "reason": "missing node",
                                                                         "own_positions": 0}
    nodes = {_key(FORK): {"move": 4}}
    assert mirror_check(FACTS, FORK, nodes, [{}], 2, 100_000, CELLS, 0)["reason"] == "missing node"
    child = GAME.step(FORK, 4)
    nodes.update({_key(GAME.step(child, a)): {"leaf": 0} for a in GAME.legal_actions(child)})
    assert mirror_check(FACTS, FORK, nodes, [{}], 2, 100_000, CELLS, 0) == check(FACTS, FORK, nodes, [{}], 2, 100_000)


def test_a_table_move_that_loses_fails_the_check():
    state = _ttt((0, 1, 3), (2, 4, 5))
    assert mirror_check(FACTS, state, {_key(state): {"move": 7}}, [{}], 2, 100_000, CELLS, 0) == {
        "won": False, "reason": "lost line", "own_positions": 1}


def test_the_empty_map_is_read_plainly_and_a_mirrored_reading_can_lose_a_line():
    maps = [{}, {}]
    assert mirror_check(LOWEST, FORK, {_key(FORK): {"leaf": 0}}, maps, 2, 100_000, CELLS, 1)["won"]
    r = mirror_check(LOWEST, FORK, {_key(FORK): {"leaf": 1}}, maps, 2, 100_000, CELLS, 1)
    assert r["reason"] == "leaf lost line"


def test_a_mirrored_leaf_is_walked_with_the_cells_empty_on_the_image(monkeypatch):
    seen = []

    def walk(facts, s, levels, n_levels, cap, cells):
        seen.append(levels)
        return {"won": True, "reason": None, "own": set()}

    monkeypatch.setattr(floor, "_mirrored_walk", walk)
    mirror_check(FACTS, MIRROR_ONLY, {_key(MIRROR_ONLY): {"leaf": 1}}, [{}, {6: 0, 3: 1, 5: 1}], 2, 100_000, CELLS, 1)
    assert seen == [{6: 0, 3: 1}]


def test_a_mirrored_leaf_past_the_cap_fails_the_check():
    nodes = {_key(MIRROR_ONLY): {"leaf": 1}}
    maps = [{}, ONE_SIDED, ONE_SIDED]
    own = mirror_check(FACTS, MIRROR_ONLY, nodes, maps, 2, 100_000, CELLS, 1)["own_positions"]
    assert mirror_check(FACTS, MIRROR_ONLY, nodes, maps, 2, own, CELLS, 1)["won"]
    r = mirror_check(FACTS, MIRROR_ONLY, nodes, maps, 2, own - 1, CELLS, 1)
    assert not r["won"] and r["reason"] == "leaf cap"


def test_a_mirrored_leaf_where_the_rule_has_no_move_fails_the_check():
    state = _ttt((), ())
    r = mirror_check(FACTS, state, {_key(state): {"leaf": 1}}, [{}, {}, {}], 2, 100_000, CELLS, 1)
    assert not r["won"] and r["reason"] == "leaf undefined"


def test_a_build_with_a_mirrored_leaf_passes_the_check():
    b = _builder([ONE_SIDED])
    assert b.build(MIRROR_ONLY)
    assert mirror_check(FACTS, MIRROR_ONLY, b.nodes, b.maps, 2, 100_000, CELLS, b.mirrored)["won"]
    assert b.mirrored == 1


@pytest.mark.parametrize("library,mirrored", [([], 0), ([ONE_SIDED], 1), ([ONE_SIDED, CENTRE], 2)])
def test_half_the_library_reads_mirrored(library, mirrored):
    assert _builder(library).mirrored == mirrored
