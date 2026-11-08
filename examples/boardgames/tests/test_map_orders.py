"""Direct tests for harness/map_orders.py — what a map must do in a strategy: the order its levels must keep at every
position its leaves play where the rule has a choice, the lowest levels that keep it, maps merged where their orders
agree, and the strategy rewritten with them. Fixture X{0,1} O{2,3}, X to move: five safe moves, only the centre (4)
wins, and after it every O reply leaves X an immediate win."""
from __future__ import annotations

import pytest

from games.tictactoe import TicTacToe, TTTState
from harness.map_orders import demands, lowest, merged, shrink
from harness.mirror_builder import board_image
from harness.steady_state import Facts, choose
from harness.strategy_file import replay

GAME = TicTacToe()
CELLS = [2, 1, 0, 5, 4, 3, 8, 7, 6]
LEVELS = 8


def _facts(**kwargs):
    return Facts(GAME, **{"ties": "lowest", "unlevelled": "last", **kwargs})


def _ttt(x, o):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=0 if len(x) == len(o) else 1, winner=None, done=False)


def _state(key):
    board, to_move = key
    return TTTState(board=tuple(board), to_move=to_move, winner=None, done=False)


FORK = _ttt((0, 1), (2, 3))
KEY = GAME.state_key(FORK)
CENTRE = (set(), {(4, 5), (4, 6), (4, 7), (4, 8)})


def test_a_choice_demands_the_move_s_cell_below_lower_moves_and_not_above_higher_ones():
    f = _facts()
    s = _ttt((0, 4), (1, 8))
    assert len(f.of(s)[1]) > 1 and not f.of(s)[0]
    move = choose(f, s, {6: 2, 3: 2}, LEVELS)
    cell = f.of(s)[2]
    got = demands(f, s, {6: 2, 3: 2}, LEVELS)
    first = {(cell[move], cell[a]) for a in f.of(s)[1] if a < move}
    after = {(cell[move], cell[a]) for a in f.of(s)[1] if a > move}
    assert first <= got["strict"] and after <= got["weak"] and first and after


def test_the_centre_is_demanded_level_with_the_moves_above_it():
    got = demands(_facts(), FORK, {4: 3}, LEVELS)
    assert (got["strict"], got["weak"]) == CENTRE and got["exceptions"] == {}


def test_forced_positions_demand_nothing():
    s = _ttt((0, 1, 4), (2, 3, 8))
    assert demands(_facts(), s, {}, LEVELS) == {"strict": set(), "weak": set(), "exceptions": {}}


def test_an_exception_s_position_demands_nothing_and_its_state_is_kept():
    got = demands(_facts(), FORK, {}, LEVELS, exceptions={KEY: 4})
    assert (got["strict"], got["weak"]) == (set(), set()) and got["exceptions"] == {KEY: FORK}


def test_the_walk_follows_the_exception_not_the_map():
    s = _ttt((0, 4), (1, 8))
    plain = demands(_facts(), s, {}, LEVELS)
    assert choose(_facts(), s, {}, LEVELS) == 2
    excepted = demands(_facts(), s, {}, LEVELS, exceptions={GAME.state_key(s): 7})
    assert excepted["weak"] == {(2, 3), (2, 5), (2, 6)} != plain["weak"], "the lines after the exception demand"


def test_a_mirrored_reading_demands_what_the_mirrored_board_does():
    image = board_image(CELLS)
    s = _ttt((0, 4), (1, 8))
    levels = {6: 2, 3: 1}
    read = demands(_facts(), s, levels, LEVELS, cells=CELLS, mirrored=True)
    on_image = demands(_facts(), image(s), levels, LEVELS)
    assert (read["strict"], read["weak"]) == (on_image["strict"], on_image["weak"]) and read["strict"] | read["weak"]
    assert read != demands(_facts(), s, levels, LEVELS)


def test_a_mirrored_reading_with_exceptions_is_refused():
    with pytest.raises(ValueError):
        demands(_facts(), FORK, {}, LEVELS, exceptions={KEY: 4}, cells=CELLS, mirrored=True)


def test_a_mirrored_reading_needs_the_mirror():
    with pytest.raises(ValueError):
        demands(_facts(), FORK, {}, LEVELS, mirrored=True)


@pytest.mark.parametrize("rule", [{"ties": "exact"}, {"unlevelled": "none"}, {"order": "tactics"}])
def test_only_the_lowest_tie_rule_with_unlevelled_cells_last_is_read(rule):
    with pytest.raises(ValueError):
        demands(_facts(**rule), FORK, {4: 0}, LEVELS)


def test_a_position_the_rule_has_no_move_for_is_refused():
    s = _ttt((0, 7, 8), (3, 4, 6))
    assert not _facts().of(s)[1]
    with pytest.raises(ValueError):
        demands(_facts(), s, {}, LEVELS)


def test_lowest_levels_a_strict_chain_upward_from_zero():
    assert lowest({(1, 2), (2, 3)}, set(), LEVELS) == {1: 0, 2: 1}


def test_lowest_levels_a_weak_demand_level_with_its_partner():
    assert lowest({(2, 3)}, {(1, 2)}, LEVELS) == {1: 0, 2: 0}


def test_lowest_leaves_cells_only_weakly_demanded_unlevelled():
    assert lowest(set(), {(1, 2), (2, 3)}, LEVELS) == {}
    assert lowest({(5, 6)}, {(1, 2)}, LEVELS) == {5: 0}


def test_lowest_refuses_a_strict_cycle_and_a_chain_longer_than_the_levels():
    assert lowest({(1, 2), (2, 1)}, set(), LEVELS) is None
    assert lowest({(1, 2)}, {(2, 1)}, LEVELS) is None
    chain = {(i, i + 1) for i in range(LEVELS + 1)}
    assert lowest(chain, set(), LEVELS) is None and lowest(chain, set(), LEVELS + 1) is not None
    assert lowest({(i, i + 1) for i in range(LEVELS)}, set(), LEVELS) == {i: i for i in range(LEVELS)}


def test_lowest_keeps_every_demand():
    strict, weak = {(3, 1), (5, 3), (5, 7)}, {(7, 3), (2, 5)}
    levels = lowest(strict, weak, LEVELS)
    last = LEVELS
    assert all(levels.get(m, last) < levels.get(o, last) for m, o in strict)
    assert all(levels.get(m, last) <= levels.get(o, last) for m, o in weak)


def test_maps_whose_orders_agree_merge():
    plan = merged({0: (set(), set()), 1: ({(1, 2)}, set()), 2: ({(1, 3)}, set())}, LEVELS)
    assert plan == {0: {}, 1: {1: 0}, 2: {1: 0}}


def test_maps_whose_orders_disagree_stay_apart():
    plan = merged({1: ({(1, 2)}, set()), 2: ({(2, 1)}, set())}, LEVELS)
    assert plan == {1: {1: 0}, 2: {2: 0}}


def test_a_map_joins_where_it_grows_the_fewest_cells():
    plan = merged({1: ({(1, 2), (3, 4)}, set()), 2: ({(5, 6)}, set()), 3: ({(1, 9)}, set())}, LEVELS)
    assert plan[3] == plan[1] == {1: 0, 3: 0} and plan[2] == {5: 0}


def test_a_group_keeps_the_orders_of_every_map_it_took():
    orders = {1: ({(1, 2), (3, 4)}, set()), 2: ({(5, 1), (1, 6)}, set()), 3: ({(1, 5)}, set())}
    plan = merged(orders, LEVELS)
    assert plan == {1: {5: 0, 1: 1, 3: 0}, 2: {5: 0, 1: 1, 3: 0}, 3: {1: 0}}
    for i, (strict, weak) in orders.items():
        assert all(plan[i].get(m, LEVELS) < plan[i].get(o, LEVELS) for m, o in strict)


def test_a_map_stays_apart_when_joining_would_cost_its_own_cells_or_more():
    plan = merged({1: ({(1, 2)}, set()), 2: ({(3, 4)}, set())}, LEVELS)
    assert plan == {1: {1: 0}, 2: {3: 0}}


def test_a_map_with_no_cells_joins_only_where_it_adds_none():
    plan = merged({1: ({(1, 2)}, set()), 2: (set(), {(2, 1)}), 3: (set(), {(1, 2)})}, LEVELS)
    assert plan == {1: {1: 0}, 2: {}, 3: {1: 0}}


def test_the_empty_map_is_never_merged_and_must_stay_empty():
    plan = merged({0: (set(), {(1, 2)}), 1: ({(1, 2)}, set())}, LEVELS)
    assert plan == {0: {}, 1: {1: 0}}
    with pytest.raises(ValueError):
        merged({0: ({(1, 2)}, set())}, LEVELS)


def test_unsatisfiable_demands_are_refused():
    with pytest.raises(ValueError):
        merged({1: ({(1, 2), (2, 1)}, set())}, LEVELS)


def _tree(leaf):
    nodes = {KEY: {"move": 4}}
    after = GAME.step(FORK, 4)
    for i, b in enumerate(GAME.legal_actions(after)):
        c = GAME.step(after, b)
        if not GAME.is_terminal(c):
            nodes[GAME.state_key(c)] = leaf(i, c)
    return nodes


def test_shrink_renumbers_the_maps_by_content_and_keeps_the_empty_map_first():
    strategy = {"nodes": _tree(lambda i, c: {"leaf": 1 + i % 3}), "maps": [{}, {5: 3}, {6: 1}, {7: 2}]}
    out = shrink(strategy, {0: {}, 1: {5: 0}, 2: {5: 0}, 3: {}}, {}, _facts(), LEVELS, _state)
    assert out["maps"][0] == {} and sorted(map(str, out["maps"][1:])) == ["{5: 0}"]
    reads = {k: out["maps"][n["leaf"]] for k, n in out["nodes"].items() if "leaf" in n}
    assert [reads[k] for k, n in strategy["nodes"].items() if "leaf" in n] == [
        {5: 0} if n["leaf"] < 3 else {} for k, n in strategy["nodes"].items() if "leaf" in n]
    assert all(n["leaf"] == 0 for k, n in out["nodes"].items() if "leaf" in n and not out["maps"][n["leaf"]])
    assert out["nodes"][KEY] == {"move": 4}


def test_shrink_gives_a_mirrored_empty_map_its_own_place():
    strategy = {"nodes": {KEY: {"leaf": 1, "mirrored": True}}, "maps": [{}, {3: 2}]}
    out = shrink(strategy, {0: {}, 1: {}}, {}, _facts(), LEVELS, _state)
    node = out["nodes"][KEY]
    assert node["mirrored"] and node["leaf"] != 0 and out["maps"][node["leaf"]] == {}


def test_shrink_drops_an_exception_the_new_map_plays_and_keeps_the_rest():
    strategy = {"nodes": {KEY: {"leaf": 1, "exceptions": {KEY: 4}}}, "maps": [{}, {5: 0}]}
    out = shrink(strategy, {0: {}, 1: {4: 0}}, {KEY: FORK}, _facts(), LEVELS, _state)
    assert out["nodes"][KEY] == {"leaf": 1}
    out = shrink(strategy, {0: {}, 1: {5: 0}}, {KEY: FORK}, _facts(), LEVELS, _state)
    assert out["nodes"][KEY] == {"leaf": 1, "exceptions": {KEY: 4}}


def test_shrink_reads_an_exception_s_state_from_its_key_when_none_was_kept():
    strategy = {"nodes": {KEY: {"leaf": 1, "exceptions": {KEY: 4}}}, "maps": [{}, {5: 0}]}
    assert shrink(strategy, {0: {}, 1: {4: 0}}, {}, _facts(), LEVELS, _state)["nodes"][KEY] == {"leaf": 1}


def test_a_map_without_a_plan_is_refused():
    with pytest.raises(KeyError):
        shrink({"nodes": {KEY: {"leaf": 1}}, "maps": [{}, {4: 0}]}, {0: {}}, {}, _facts(), LEVELS, _state)


@pytest.mark.parametrize("levels, read", [({4: 5, 8: 6}, {}), ({4: 3, 5: 6}, {})])
def test_a_shrunk_strategy_still_wins_every_line(levels, read):
    f = _facts()
    strategy = {"nodes": {KEY: {"leaf": 1}}, "maps": [{}, levels]}
    walked = demands(f, FORK, levels, LEVELS)
    plan = merged({0: (set(), set()), 1: (walked["strict"], walked["weak"])}, LEVELS)
    out = shrink(strategy, plan, walked["exceptions"], f, LEVELS, _state)
    assert out["maps"][out["nodes"][KEY]["leaf"]] == read
    assert replay(f, FORK, out, LEVELS, 10_000, CELLS)["won"]


def test_a_shrunk_map_that_still_needs_levels_wins_every_line():
    f = _facts()
    s = _ttt((0, 4), (1, 8))
    levels = {7: 1, 6: 0, 2: 3}
    assert replay(f, s, {"nodes": {GAME.state_key(s): {"leaf": 1}}, "maps": [{}, levels]}, LEVELS, 10_000, CELLS)["won"]
    walked = demands(f, s, levels, LEVELS)
    plan = merged({0: (set(), set()), 1: (walked["strict"], walked["weak"])}, LEVELS)
    out = shrink({"nodes": {GAME.state_key(s): {"leaf": 1}}, "maps": [{}, levels]}, plan, {}, f, LEVELS, _state)
    shrunk = out["maps"][out["nodes"][GAME.state_key(s)]["leaf"]]
    assert shrunk and max(shrunk.values()) <= max(levels.values())
    assert replay(f, s, out, LEVELS, 10_000, CELLS)["won"]
