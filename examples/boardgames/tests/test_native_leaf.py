"""Direct tests for harness/native_leaf.py (harness/c4leafwalk.c) — the C walk must agree with the Python reference
harness.steady_exceptions.needed exactly: the same status, the same exceptions (position and column) and the same
count of the side's positions, over Connect-4 positions reached by winning play and random priority maps of every
shape (empty, sparse, dense, all one level)."""
from __future__ import annotations

import random
import time

import pytest

from games.connect4 import Connect4
from harness.c4_oracle import solve
from harness.native_leaf import needed as native
from harness.steady_exceptions import needed as reference
from harness.steady_state import Facts

GAME = Connect4()


def _winning(states):
    out = []
    for s in states:
        me = GAME.current_player(s)
        moves = set()
        for a in GAME.legal_actions(s):
            child = GAME.step(s, a)
            kept = round(GAME.returns(child)[me]) if GAME.is_terminal(child) else -solve((child.board, child.to_move))
            if kept == 1:
                moves.add(a)
        out.append(moves)
    return out


def _positions(count: int, seed: int) -> list:
    """First-player positions 18-24 plies deep from random play that the first player can still win and whose leaf
    under the empty map plays 30-3,000 of the side's positions — big enough to need exceptions, small enough for the
    Python reference. Only late positions are solved, so no solve reaches the slow opening."""
    rng = random.Random(seed)
    found = []
    while len(found) < count:
        s = GAME.initial_state()
        plies = rng.choice((18, 20, 22, 24))
        while not GAME.is_terminal(s) and sum(1 for v in s.board if v) < plies:
            s = GAME.step(s, rng.choice(GAME.legal_actions(s)))
        if GAME.is_terminal(s) or s.to_move != 0 or not _winning([s])[0]:
            continue
        size = native(Facts(GAME), s, {}, 8, None, 1_000_000, deadline=time.monotonic() + 20)
        if size["status"] == "ok" and 30 <= size["own_positions"] <= 3_000:
            found.append(s)
    return found


def _maps(facts, s, rng):
    empty = facts.empty_cells(s)
    yield {}
    yield {c: rng.randrange(8) for c in rng.sample(empty, min(4, len(empty)))}
    yield {c: rng.randrange(8) for c in empty}
    yield {c: 0 for c in empty}


@pytest.fixture(scope="module")
def positions():
    return _positions(8, 5)


RULES = [("exact", 1, False, "none"), ("lowest", 1, False, "none"), ("exact", 2, False, "none"),
         ("lowest", 2, False, "none"), ("exact", 1, True, "none"), ("lowest", 1, True, "none"),
         ("lowest", 2, True, "none"), ("lowest", 1, False, "last"), ("exact", 1, False, "last"),
         ("lowest", 2, True, "last")]


@pytest.mark.parametrize("ties,depth,forcing,unlevelled", RULES)
def test_the_c_walk_matches_the_python_walk_on_every_position_and_map(positions, ties, depth, forcing,
                                                                      unlevelled):
    rng = random.Random(7)
    checked = 0
    for s in positions:
        facts = Facts(GAME, ties=ties, safe_depth=depth, forcing=forcing, unlevelled=unlevelled)
        for levels in _maps(facts, s, rng):
            assert native(facts, s, levels, 8, None, 1_000_000) == reference(facts, s, levels, 8, _winning, 1_000_000)
            checked += 1
    assert checked == 4 * len(positions)


def test_fewer_levels_read_the_same_map_differently_in_both(positions):
    facts = Facts(GAME)
    s = positions[0]
    levels = {c: k for k, c in enumerate(facts.empty_cells(s)[:6])}
    for n in (1, 3, 8):
        assert native(facts, s, levels, n, None, 1_000_000) == reference(facts, s, levels, n, _winning, 1_000_000)


def test_a_walk_past_the_cap_is_a_cap_in_both(positions):
    facts = Facts(GAME)
    s = positions[-1]
    assert native(facts, s, {}, 8, None, 50)["status"] == "cap" == reference(facts, s, {}, 8, _winning, 50)["status"]


def test_a_walk_past_its_deadline_is_a_timeout(positions):
    r = native(Facts(GAME), positions[-1], {}, 8, None, 1_000_000, deadline=time.monotonic() - 1)
    assert r["status"] == "timeout" and r["exceptions"] == {}


def test_a_root_its_side_cannot_win_is_refused():
    rng = random.Random(11)
    while True:
        s = GAME.initial_state()
        while not GAME.is_terminal(s) and sum(1 for v in s.board if v) < 24:
            s = GAME.step(s, rng.choice(GAME.legal_actions(s)))
        if not GAME.is_terminal(s) and s.to_move == 0 and not _winning([s])[0]:
            break
    with pytest.raises(ValueError, match="cannot win"):
        native(Facts(GAME), s, {}, 8, None, 1_000_000)


def test_exceptions_are_counted_without_building_their_keys(positions, monkeypatch):
    import harness.native_leaf as nl

    built = []
    real = nl._board
    monkeypatch.setattr(nl, "_board", lambda *a: built.append(1) or real(*a))
    r = native(Facts(GAME), positions[0], {}, 8, None, 1_000_000)
    assert len(r["exceptions"]) > 0 and not built
    assert dict(r["exceptions"]) == reference(Facts(GAME), positions[0], {}, 8, _winning, 1_000_000)["exceptions"]
    assert built


def test_one_walk_s_exceptions_survive_the_next_walk(positions):
    facts = Facts(GAME)
    first = native(facts, positions[0], {}, 8, None, 1_000_000)
    native(facts, positions[1], {}, 8, None, 1_000_000)
    assert dict(first["exceptions"]) == reference(facts, positions[0], {}, 8, _winning, 1_000_000)["exceptions"]


@pytest.mark.parametrize("ties,depth,forcing,unlevelled", RULES)
def test_the_c_verify_matches_the_python_verify_on_every_position_and_map(positions, ties, depth, forcing,
                                                                          unlevelled):
    from harness.native_leaf import verify as native_verify
    from harness.steady_state import verify as python_verify

    rng = random.Random(9)
    for s in positions:
        facts = Facts(GAME, ties=ties, safe_depth=depth, forcing=forcing, unlevelled=unlevelled)
        empty = facts.empty_cells(s)
        for levels in ({}, {c: rng.randrange(8) for c in empty}, {c: 0 for c in empty},
                       {c: k for k, c in enumerate(empty[:8])}):
            for n in (1, 8):
                assert native_verify(facts, s, levels, n) == python_verify(facts, s, levels, n)


def test_the_c_verify_reports_a_cap_and_a_timeout_as_the_python_verify(positions):
    from harness.native_leaf import verify as native_verify
    from harness.steady_state import verify as python_verify

    facts = Facts(GAME)
    s = positions[-1]
    for levels in ({c: k for k, c in enumerate(facts.empty_cells(s))}, {}):
        r = python_verify(facts, s, levels, 8, 5)
        assert native_verify(facts, s, levels, 8, 5) == r
    late = time.monotonic() - 1
    assert native_verify(facts, s, {}, 8, deadline=late) == python_verify(facts, s, {}, 8, deadline=late)
    assert native_verify(facts, s, {}, 8, deadline=late)["reason"] == "timeout"


def test_the_tie_rule_changes_what_both_walks_find(positions):
    from harness.native_leaf import verify as native_verify

    s = positions[0]
    exact, lowest = Facts(GAME), Facts(GAME, ties="lowest")
    level = {c: 0 for c in exact.empty_cells(s)}
    assert native(exact, s, level, 8, None, 1_000_000) != native(lowest, s, level, 8, None, 1_000_000)
    assert native_verify(exact, s, level, 8)["reason"] == "undefined"
    assert native_verify(lowest, s, level, 8) != native_verify(exact, s, level, 8)


def test_two_ply_safety_changes_what_both_walks_find(positions):
    from harness.native_leaf import verify as native_verify

    changed = 0
    for s in positions:
        one, two = Facts(GAME, ties="lowest"), Facts(GAME, ties="lowest", safe_depth=2)
        level = {c: 0 for c in one.empty_cells(s)}
        changed += native(one, s, level, 8, None, 1_000_000) != native(two, s, level, 8, None, 1_000_000)
        changed += native_verify(one, s, level, 8) != native_verify(two, s, level, 8)
    assert changed


def test_preferring_forcing_moves_changes_what_both_walks_find(positions):
    from harness.native_leaf import verify as native_verify

    changed = 0
    for s in positions:
        plain, forcing = Facts(GAME, ties="lowest"), Facts(GAME, ties="lowest", forcing=True)
        level = {c: 0 for c in plain.empty_cells(s)}
        changed += native(plain, s, level, 8, None, 1_000_000) != native(forcing, s, level, 8, None, 1_000_000)
        changed += native_verify(plain, s, level, 8) != native_verify(forcing, s, level, 8)
    assert changed


def test_unlevelled_cells_last_change_what_both_walks_find(positions):
    from harness.native_leaf import verify as native_verify

    changed = 0
    for s in positions:
        plain, last = Facts(GAME, ties="lowest"), Facts(GAME, ties="lowest", unlevelled="last")
        changed += native(plain, s, {}, 8, None, 1_000_000) != native(last, s, {}, 8, None, 1_000_000)
        changed += native_verify(plain, s, {}, 8) != native_verify(last, s, {}, 8)
    assert changed


def test_the_last_level_holds_only_unlevelled_cells_in_both_walks(positions):
    """Under exact ties, a map putting all but one safe move on level 0: level 0 holds several and does not decide, the
    one unlevelled safe move does — in both walks alike."""
    from harness.native_leaf import verify as native_verify
    from harness.steady_state import verify as python_verify

    facts = Facts(GAME, unlevelled="last")
    tried = 0
    for s in positions:
        _wins, safe, cell = facts.of(s)
        if len(safe) < 3:
            continue
        levels = {c: 0 for c in facts.empty_cells(s) if c != cell[safe[-1]]}
        assert native(facts, s, levels, 8, None, 1_000_000) == reference(facts, s, levels, 8, _winning, 1_000_000)
        assert native_verify(facts, s, levels, 8) == python_verify(facts, s, levels, 8)
        tried += 1
    assert tried >= 3
