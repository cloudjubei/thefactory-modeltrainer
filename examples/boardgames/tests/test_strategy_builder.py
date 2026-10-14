"""Direct tests for harness/strategy_builder.py — a complete certified strategy as table moves plus steady-state
leaves, built depth first, and checked without any oracle. Tic-tac-toe fixtures checked by hand: in X{0,1} O{2,3},
X to move, only the centre wins (the map {4: 0}); after it, every O reply leaves X an immediate win."""
from __future__ import annotations

import time

import pytest

from games.tictactoe import TicTacToe, TTTState
from harness.steady_search import find_steady_state
from harness.steady_state import Facts, verify
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
    def search(s, start, seconds):
        r = find_steady_state(facts, s, _winning, 2, max_constraints=500, conflicts=100_000, seconds=30.0)
        return {"levels": r["levels"], "exceptions": {}, "own_positions": None} if r["status"] == "found" else None
    return search


def _builder(min_leaf_depth=0, seconds=60.0, search=None, accept=None, budgets=None, verifier=verify,
             by_size=False):
    facts = Facts(GAME)
    return Builder(facts, _winning, search or _sat(facts), n_levels=2, level_bits=3, cap=100_000,
                   min_leaf_depth=min_leaf_depth, reuse_window=50, seconds=seconds, accept=accept, budgets=budgets,
                   verifier=verifier, choose_by_size=by_size)


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
    b = _builder(search=lambda s, start, seconds: pytest.fail("searched"))
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
    b = _builder(search=lambda s, start, seconds: {"levels": {4: 0, 5: 1, 6: 1}, "exceptions": {}, "own_positions": None})
    assert b.build(FORK) and b.maps[1] == {4: 0}


def test_a_failed_search_makes_a_move_node_not_an_empty_leaf():
    b = _builder(search=lambda s, start, seconds: None)
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
    return lambda s, start, seconds: {"levels": levels or {}, "exceptions": dict(ROOT_FIX if exceptions is None else exceptions),
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

    def search(s, start, seconds):
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


def _recording(calls):
    def search(s, start, seconds):
        calls.append((sum(1 for v in s.board if v), seconds))
        return None
    return search


def test_each_search_gets_the_budget_of_the_deepest_entry_at_or_above_its_depth():
    calls = []
    b = _builder(search=_recording(calls), budgets=[[0, 20.0], [2, 5.0]])
    assert b.build(EARLY_ROOT)
    pieces = {p: s for p, s in calls}
    assert pieces[2] == 20.0 and pieces[4] == 5.0 and pieces.get(6, 5.0) == 5.0


def test_without_budgets_the_search_uses_its_own_default():
    calls = []
    assert _builder(search=_recording(calls)).build(EARLY_ROOT)
    assert calls and all(seconds is None for _p, seconds in calls)


def test_a_search_shallower_than_every_budget_uses_its_own_default():
    calls = []
    assert _builder(search=_recording(calls), budgets=[[2, 5.0]]).build(EARLY_ROOT)
    assert dict(calls)[2] is None and dict(calls)[4] == 5.0


EARLY_ROOT = _ttt((0,), (1,))


def test_a_search_budget_never_runs_past_the_build_s_deadline():
    calls = []
    assert _builder(search=_recording(calls), budgets=[[0, 1000.0]], seconds=5.0).build(EARLY_ROOT)
    assert calls and all(0 <= seconds <= 5.0 for _p, seconds in calls)


def test_a_build_past_its_deadline_gives_a_search_no_time():
    b = _builder(search=lambda s, start, seconds: None, budgets=[[0, 1000.0]])
    b.deadline = 0.0
    assert b._budget(0) == 0.0


def _with_library(library, search=None, accept=None):
    facts = Facts(GAME)
    return Builder(facts, _winning, search or (lambda s, start, seconds: None), n_levels=2, level_bits=3, cap=100_000,
                   min_leaf_depth=0, reuse_window=50, seconds=60.0, accept=accept, library=library)


def test_a_library_map_covers_a_position_without_searching():
    b = _with_library([({4: 0, 8: 1}, 5)], search=lambda s, start, seconds: pytest.fail("searched"))
    assert b.build(FORK) and b.nodes[GAME.state_key(FORK)] == {"leaf": 1}


def test_library_maps_are_not_charged_to_the_build_but_their_references_are():
    b = _with_library([({4: 0, 8: 1}, 5), ({6: 0}, 4)])
    assert b.build(FORK)
    assert b.bits() == {"nodes": 1 + 2, "moves": 0, "leaves": 1, "maps": 0, "map_bits": 0, "exceptions": 0,
                        "exception_bits": 0}
    assert b.found_maps() == []


def test_maps_found_in_the_build_are_charged_and_reported_after_the_library():
    b = _with_library([({6: 0}, 4)], search=lambda s, start, seconds: {"levels": {4: 0}, "exceptions": {},
                                                                         "own_positions": None})
    assert b.build(FORK)
    assert b.nodes[GAME.state_key(FORK)] == {"leaf": 2}
    assert b.found_maps() == [({4: 0}, 5)]
    assert b.bits()["maps"] == 1 and b.bits()["map_bits"] == 5 + 3


def test_without_a_library_the_build_is_unchanged():
    b = _with_library(None, search=lambda s, start, seconds: {"levels": {4: 0}, "exceptions": {},
                                                              "own_positions": None})
    assert b.build(FORK) and b.nodes[GAME.state_key(FORK)] == {"leaf": 1} and b.found_maps() == [({4: 0}, 5)]


def test_the_builder_s_own_walks_get_its_deadline(monkeypatch):
    import harness.strategy_builder as sb

    seen = {"verify": [], "simplify": [], "code": []}
    real_verify, real_simplify, real_code = sb.verify, sb.simplify, sb.walk_order_cost

    def verify(*a, deadline=None, **k):
        seen["verify"].append(deadline)
        return real_verify(*a, deadline=deadline, **k)

    def simplify(*a, deadline=None, **k):
        seen["simplify"].append(deadline)
        return real_simplify(*a, deadline=deadline, **k)

    def code(*a, deadline=None, **k):
        seen["code"].append(deadline)
        return real_code(*a, deadline=deadline, **k)

    monkeypatch.setattr(sb, "simplify", simplify)
    monkeypatch.setattr(sb, "walk_order_cost", code)
    pure = _builder(search=lambda s, start, seconds: {"levels": {4: 0, 5: 1}, "exceptions": {}, "own_positions": None},
                    verifier=verify)
    assert pure.build(FORK)
    excepted = _builder(search=_excepted(), accept=1, verifier=verify)
    assert excepted.build(FORK)
    assert all(seen[k] and all(d == pure.deadline or d == excepted.deadline for d in seen[k]) for k in seen)


def test_a_leaf_whose_coding_runs_out_of_time_is_not_kept(monkeypatch):
    import harness.strategy_builder as sb

    def late(*a, **k):
        raise TimeoutError("past the deadline")

    monkeypatch.setattr(sb, "walk_order_cost", late)
    b = _builder(search=_excepted(), accept=1)
    assert b.build(FORK) and b.nodes[GAME.state_key(FORK)] == {"move": 4}


def test_the_builder_s_covering_and_simplify_use_its_verifier():
    calls = []

    def verifier(*a, **k):
        calls.append(a[2])
        return verify(*a, **k)

    facts = Facts(GAME)
    b = Builder(facts, _winning, lambda s, start, seconds: {"levels": {4: 0, 5: 1}, "exceptions": {},
                                                            "own_positions": None},
                n_levels=2, level_bits=3, cap=100_000, min_leaf_depth=0, reuse_window=50, seconds=60.0,
                verifier=verifier)
    assert b.build(FORK) and b.maps[1] == {4: 0}
    assert {} in calls and {4: 0} in calls


EARLY = _ttt((0,), (1,))


def test_by_size_a_leaf_with_exceptions_smaller_than_the_split_below_it_is_kept():
    """The leaf: 1 node + a flag + 3 coded bits = 5; the split: a move (1 + 4 bits) and 4 trivial leaves = 9."""
    b = _builder(search=_excepted(), accept=1, by_size=True)
    assert b.build(FORK)
    assert b.nodes == {GAME.state_key(FORK): {"leaf": 0, "exceptions": ROOT_FIX, "own": 5, "coded": 3}}
    assert b.maps == [{}] and b.bits()["nodes"] == 5


def test_by_size_a_split_smaller_than_the_leaf_replaces_it_and_the_leaf_s_map_is_not_kept():
    """The leaf with the map {5: 1} costs 14 bits (node, reference, 8 map bits, flag, 3 coded), the split 9 — and
    without the choice that leaf is kept (test_a_leaf_s_map_is_kept_beside_its_exceptions_and_charged_in_its_size)."""
    b = _builder(search=_excepted(levels={5: 1}, own=1), accept=0.01, by_size=True)
    assert b.build(FORK)
    assert b.nodes[GAME.state_key(FORK)] == {"move": 4} and len(b.nodes) == 5
    assert b.maps == [{}] and b.map_empty == [0] and b.bits()["nodes"] == 9
    assert check(b.facts, FORK, b.nodes, b.maps, 2, 100_000)["won"]


def test_by_size_a_tie_keeps_the_leaf():
    b = _builder(search=_excepted(), accept=1, by_size=True)
    b.bits = lambda: {"nodes": 7}
    assert b.build(FORK) and "exceptions" in b.nodes[GAME.state_key(FORK)]


def test_by_size_a_split_that_runs_out_of_time_loses_to_the_leaf_and_the_build_completes():
    b = _builder(search=_excepted(levels={5: 1}, own=1), accept=0.01, by_size=True)
    split = b._split

    def late(state, depth):
        b.deadline = time.monotonic() - 1.0
        return split(state, depth)

    b._split = late
    assert b.build(FORK)
    assert b.nodes == {GAME.state_key(FORK): {"leaf": 1, "exceptions": ROOT_FIX, "own": 1, "coded": 3}}
    assert b.maps == [{}, {5: 1}] and b.map_empty == [0, 5]


def test_by_size_a_winning_split_keeps_the_maps_it_found(monkeypatch):
    import harness.strategy_builder as sb

    monkeypatch.setattr(sb, "walk_order_cost", lambda *a, **k: {"bits": 10_000})
    facts = Facts(GAME)
    sat = _sat(facts)
    root = GAME.state_key(EARLY)
    fix = {root: min(_winning([EARLY])[0])}

    def search(s, start, seconds):
        if GAME.state_key(s) == root:
            return {"levels": {}, "exceptions": dict(fix), "own_positions": 10**9}
        return sat(s, start, seconds)

    b = Builder(facts, _winning, search, n_levels=2, level_bits=3, cap=100_000, min_leaf_depth=0, reuse_window=50,
                seconds=60.0, accept=1, choose_by_size=True)
    assert b.build(EARLY)
    assert "move" in b.nodes[root] and not any(n.get("exceptions") for n in b.nodes.values())
    assert len(b.maps) > 1 and all(n["leaf"] < len(b.maps) for n in b.nodes.values() if "leaf" in n)
    assert any(n.get("leaf", 0) > 0 for n in b.nodes.values())
    assert check(facts, EARLY, b.nodes, b.maps, 2, 100_000)["won"]


def test_by_size_the_choice_is_made_below_the_root_too(monkeypatch):
    import harness.strategy_builder as sb

    monkeypatch.setattr(sb, "walk_order_cost", lambda *a, **k: {"bits": 1})

    def search(s, start, seconds):
        return {"levels": {}, "exceptions": {GAME.state_key(s): min(_winning([s])[0])}, "own_positions": 10**9}

    b = _builder(search=search, accept=1, by_size=True)
    depths = []
    split = b._split

    def recorded(state, depth):
        depths.append(depth)
        return split(state, depth)

    b._split = recorded
    assert b.build(EARLY)
    assert 0 in depths and any(d >= 2 for d in depths)


def test_by_size_a_losing_split_leaves_none_of_its_maps_behind(monkeypatch):
    import harness.strategy_builder as sb

    monkeypatch.setattr(sb, "walk_order_cost", lambda *a, **k: {"bits": 1})
    facts = Facts(GAME)
    sat = _sat(facts)
    root = GAME.state_key(EARLY)
    found = []

    def search(s, start, seconds):
        if GAME.state_key(s) == root:
            return {"levels": {}, "exceptions": {root: min(_winning([EARLY])[0])}, "own_positions": 10**9}
        r = sat(s, start, seconds)
        found.append(r is not None)
        return r

    b = Builder(facts, _winning, search, n_levels=2, level_bits=3, cap=100_000, min_leaf_depth=0, reuse_window=50,
                seconds=60.0, accept=1, choose_by_size=True)
    assert b.build(EARLY)
    assert any(found) and "exceptions" in b.nodes[root] and len(b.nodes) == 1
    assert b.maps == [{}] and b.map_empty == [0]


def test_improving_replaces_a_leaf_with_exceptions_by_a_smaller_split_and_drops_its_map():
    b = _builder(search=_excepted(levels={5: 1}, own=1), accept=0.01)
    assert b.build(FORK) and b.bits()["nodes"] == 14
    assert b.improve() == {"tried": 1, "kept": 1}
    assert b.nodes[GAME.state_key(FORK)] == {"move": 4} and b.maps == [{}] and b.map_empty == [0]
    assert b.bits()["nodes"] == 9 and check(b.facts, FORK, b.nodes, b.maps, 2, 100_000)["won"]


def test_improving_keeps_a_leaf_smaller_than_its_split():
    b = _builder(search=_excepted(), accept=1)
    assert b.build(FORK)
    before = (dict(b.nodes), list(b.maps))
    assert b.improve() == {"tried": 1, "kept": 0}
    assert (b.nodes, b.maps) == before and b.bits()["nodes"] == 5


def test_improving_a_tie_keeps_the_leaf():
    b = _builder(search=_excepted(), accept=1)
    assert b.build(FORK)
    b.bits = lambda: {"nodes": 7}
    assert b.improve() == {"tried": 1, "kept": 0} and "exceptions" in b.nodes[GAME.state_key(FORK)]


def test_improving_stops_at_the_deadline():
    b = _builder(search=_excepted(levels={5: 1}, own=1), accept=0.01)
    assert b.build(FORK)
    b.deadline = time.monotonic() - 1.0
    assert b.improve() == {"tried": 0, "kept": 0} and "exceptions" in b.nodes[GAME.state_key(FORK)]


def test_improving_undoes_a_split_that_runs_out_of_time():
    b = _builder(search=_excepted(levels={5: 1}, own=1), accept=0.01)
    assert b.build(FORK)
    before = (dict(b.nodes), list(b.maps), list(b.map_empty))
    split = b._split

    def late(state, depth):
        b.deadline = time.monotonic() - 1.0
        return split(state, depth)

    b._split = late
    assert b.improve() == {"tried": 1, "kept": 0}
    assert (b.nodes, b.maps, b.map_empty) == before


def test_improving_tries_the_costliest_leaves_first_and_the_leaves_a_kept_split_makes(monkeypatch):
    import harness.strategy_builder as sb

    root = GAME.state_key(EARLY)

    def weight(s):
        return 1000 - 100 * sum(1 for v in s.board if v) + sum(i for i, v in enumerate(s.board) if v)

    monkeypatch.setattr(sb, "walk_order_cost",
                        lambda facts, s, *a, **k: {"bits": 10_000 if GAME.state_key(s) == root else weight(s)})

    def search(s, start, seconds):
        return {"levels": {}, "exceptions": {GAME.state_key(s): min(_winning([s])[0])}, "own_positions": 10**9}

    b = _builder(search=search, accept=1)
    assert b.build(EARLY) and "exceptions" in b.nodes[root]
    order = []
    split = b._split

    def recorded(state, depth):
        order.append((b.nodes.get(GAME.state_key(state)), GAME.state_key(state), weight(state)))
        return split(state, depth)

    b._split = recorded
    stats = b.improve()
    assert order[0][1] == root and stats["kept"] >= 1 and stats["tried"] > 1
    assert check(b.facts, EARLY, b.nodes, b.maps, 2, 100_000)["won"]


def _fabricated(coded):
    states = [_ttt((0,), (1,)), _ttt((4,), (1,)), _ttt((0, 4), (1, 8)), _ttt((2,), (5,))]
    b = _builder()
    for s, c in zip(states, coded):
        k = GAME.state_key(s)
        b.nodes[k] = {"leaf": 0, "exceptions": {k: 0}, "own": 1, "coded": c}
        b.leaf_states[k] = (s, 0)
    return b, [GAME.state_key(s) for s in states]


def test_improving_takes_the_costliest_leaf_first():
    b, keys = _fabricated([5, 9, 7])
    order = []
    b._split = lambda state, depth: order.append(GAME.state_key(state)) or False
    assert b.improve() == {"tried": 3, "kept": 0}
    assert order == [keys[1], keys[2], keys[0]]


def test_the_leaves_a_kept_split_makes_join_the_queue():
    b, keys = _fabricated([5])
    extra = _ttt((2,), (5,))
    order = []

    def split(state, depth):
        order.append(GAME.state_key(state))
        if GAME.state_key(state) == keys[0]:
            b.nodes[keys[0]] = {"move": 0}
            b.nodes[GAME.state_key(extra)] = {"leaf": 0, "exceptions": {GAME.state_key(extra): 0}, "own": 1, "coded": 1}
            b.leaf_states[GAME.state_key(extra)] = (extra, 2)
            return True
        return False

    sizes = iter([10, 5, 5])
    b.bits = lambda: {"nodes": next(sizes)}
    b._split = split
    assert b.improve() == {"tried": 2, "kept": 1}
    assert order == [keys[0], GAME.state_key(extra)]


def test_dropping_unused_maps_renumbers_leaves_without_touching_the_nodes_it_was_given():
    b = _builder()
    b.maps, b.map_empty = [{}, {1: 0}, {2: 0}], [0, 5, 5]
    node = {"leaf": 2}
    b.nodes = {"a": node, "b": {"move": 3}}
    b._compact()
    assert b.maps == [{}, {2: 0}] and b.map_empty == [0, 5]
    assert b.nodes == {"a": {"leaf": 1}, "b": {"move": 3}} and node == {"leaf": 2}


def test_dropping_unused_maps_keeps_the_library_s():
    b = _with_library([({6: 0}, 4)])
    b.maps += [{1: 0}, {2: 0}]
    b.map_empty += [5, 5]
    b.nodes = {"a": {"leaf": 3}}
    b._compact()
    assert b.maps == [{}, {6: 0}, {2: 0}] and b.map_empty == [0, 4, 5] and b.nodes == {"a": {"leaf": 2}}


_STATES = [_ttt((0,), (1,)), _ttt((4,), (1,)), _ttt((0, 4), (1, 8)), _ttt((2,), (5,)), _ttt((6,), (3,))]


def _planned(leaves: dict, plan: dict, order: list):
    """A builder over fabricated leaves {name: coded bits}; splitting a name in `plan` makes it a move node and creates
    the leaves listed for it; any other split fails. Size: the leaves' coded bits plus 3 per move node."""
    b = _builder()
    key = {name: GAME.state_key(s) for name, s in zip("abcde", _STATES)}
    state = {name: s for name, s in zip("abcde", _STATES)}

    def leaf(name, coded):
        b.nodes[key[name]] = {"leaf": 0, "exceptions": {key[name]: 0}, "own": 1, "coded": coded}
        b.leaf_states[key[name]] = (state[name], 0)

    for name, coded in leaves.items():
        leaf(name, coded)
    names = {v: k for k, v in key.items()}

    def split(s, depth):
        name = names[GAME.state_key(s)]
        order.append(name)
        if name not in plan:
            return False
        b.nodes[key[name]] = {"move": 0}
        for child, coded in plan[name]:
            leaf(child, coded)
        return True

    b._split = split
    b.bits = lambda: {"nodes": sum(n.get("coded", 0) for n in b.nodes.values())
                      + 3 * sum(1 for n in b.nodes.values() if "move" in n)}
    return b, key


def test_lookahead_keeps_a_split_that_is_smaller_once_its_own_leaves_are_improved():
    plan = {"a": [("b", 9)], "b": []}
    order = []
    b, key = _planned({"a": 10}, plan, order)
    assert b.improve(lookahead=True) == {"tried": 2, "kept": 2}
    assert b.nodes[key["a"]] == {"move": 0} and b.nodes[key["b"]] == {"move": 0} and b.bits()["nodes"] == 6
    plain, key = _planned({"a": 10}, plan, [])
    assert plain.improve() == {"tried": 1, "kept": 0} and "exceptions" in plain.nodes[key["a"]]


def test_lookahead_undoes_a_split_and_its_improvements_together_when_they_do_not_shrink_the_strategy():
    b, key = _planned({"a": 10}, {"a": [("b", 9)], "b": [("c", 8)]}, [])
    assert b.improve(lookahead=True) == {"tried": 3, "kept": 0}
    assert set(b.nodes) == {key["a"]} and "exceptions" in b.nodes[key["a"]] and b.bits()["nodes"] == 10


def test_lookahead_improves_only_the_leaves_the_split_made_before_judging_it():
    order = []
    b, key = _planned({"a": 10, "e": 9.5}, {"a": [("b", 9)], "b": []}, order)
    assert b.improve(lookahead=True)["kept"] == 2
    assert order == ["a", "b", "e"]


def test_lookahead_follows_leaves_made_by_splits_kept_inside_it():
    b, key = _planned({"a": 10}, {"a": [("b", 9)], "b": [("c", 4)], "c": []}, [])
    assert b.improve(lookahead=True) == {"tried": 3, "kept": 3} and b.bits()["nodes"] == 9


def test_lookahead_does_not_count_splits_undone_with_the_split_above_them():
    b, key = _planned({"a": 10}, {"a": [("b", 9)], "b": [("c", 5)]}, [])
    assert b.improve(lookahead=True) == {"tried": 3, "kept": 0}
    assert set(b.nodes) == {key["a"]}


def _found(levels):
    return lambda s, start, seconds: {"levels": dict(levels), "exceptions": {}, "own_positions": None}


def test_researching_replaces_a_leaf_with_exceptions_by_a_smaller_pure_map_and_drops_the_old_map():
    b = _builder(search=_excepted(levels={5: 1}, own=1), accept=0.01)
    assert b.build(FORK) and b.bits()["nodes"] == 14
    b.search = _found({4: 0, 5: 1})
    assert b.research(60.0) == {"tried": 1, "kept": 1}
    assert b.nodes[GAME.state_key(FORK)] == {"leaf": 1} and b.maps == [{}, {4: 0}] and b.bits()["nodes"] == 10
    assert check(b.facts, FORK, b.nodes, b.maps, 2, 100_000)["won"]


def test_researching_keeps_a_leaf_smaller_than_what_the_search_finds():
    b = _builder(search=_excepted(), accept=1)
    assert b.build(FORK)
    before = (dict(b.nodes), list(b.maps), list(b.map_empty))
    b.search = _found({4: 0})
    assert b.research(60.0) == {"tried": 1, "kept": 0}
    assert (b.nodes, b.maps, b.map_empty) == before


def test_researching_can_keep_a_smaller_leaf_with_exceptions():
    b = _builder(search=_excepted(levels={5: 1}, own=1), accept=0.01)
    assert b.build(FORK)
    b.search = _excepted()
    assert b.research(60.0) == {"tried": 1, "kept": 1}
    assert b.nodes[GAME.state_key(FORK)] == {"leaf": 0, "exceptions": ROOT_FIX, "own": 5, "coded": 3}
    assert b.maps == [{}] and b.bits()["nodes"] == 5


def test_researching_starts_from_the_leaf_s_map_with_the_budget_never_past_the_deadline():
    b = _builder(search=_excepted(levels={5: 1}, own=1), accept=0.01, seconds=600.0)
    assert b.build(FORK)
    calls = []
    b.search = lambda s, start, seconds: calls.append((start, seconds))
    assert b.research(60.0) == {"tried": 1, "kept": 0}
    assert calls == [({5: 1}, 60.0)]
    b2 = _builder(search=_excepted(levels={5: 1}, own=1), accept=0.01)
    assert b2.build(FORK)
    b2.deadline = time.monotonic() + 5.0
    b2.search = lambda s, start, seconds: calls.append((start, seconds))
    b2.research(60.0)
    assert 0 < calls[-1][1] <= 5.0


def test_researching_tries_the_costliest_leaves_first_and_each_once():
    order = []
    b, key = _planned({"a": 4, "b": 9, "c": 6}, {}, [])
    names = {v: k for k, v in key.items()}
    b.search = lambda s, start, seconds: order.append(names[GAME.state_key(s)])
    assert b.research(1.0) == {"tried": 3, "kept": 0} and order == ["b", "c", "a"]


def test_researching_stops_at_the_deadline():
    b = _builder(search=_excepted(levels={5: 1}, own=1), accept=0.01)
    assert b.build(FORK)
    b.deadline = time.monotonic() - 1.0
    b.search = lambda s, start, seconds: pytest.fail("searched past the deadline")
    assert b.research(60.0) == {"tried": 0, "kept": 0}


def test_researching_leaves_a_leaf_alone_when_coding_the_new_one_runs_out_of_time(monkeypatch):
    import harness.strategy_builder as sb

    b = _builder(search=_excepted(levels={5: 1}, own=1), accept=0.01)
    assert b.build(FORK)
    before = (dict(b.nodes), list(b.maps))

    def late(*a, **k):
        raise TimeoutError("past the deadline")

    monkeypatch.setattr(sb, "walk_order_cost", late)
    b.search = _excepted()
    assert b.research(60.0) == {"tried": 1, "kept": 0} and (b.nodes, b.maps) == before


def test_researching_a_tie_keeps_the_leaf():
    b = _builder(search=_excepted(levels={5: 1}, own=1), accept=0.01)
    assert b.build(FORK)
    before = (dict(b.nodes), list(b.maps))
    b.search = _found({4: 0})
    b.bits = lambda: {"nodes": 7}
    assert b.research(60.0) == {"tried": 1, "kept": 0} and (b.nodes, b.maps) == before
