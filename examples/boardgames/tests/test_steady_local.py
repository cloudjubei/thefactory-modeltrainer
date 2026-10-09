"""Direct tests for harness/steady_local.py — a local search over priority maps scored by the size of the complete
leaf each map makes (map bits plus the exceptions it needs over every line, harness.steady_exceptions.needed), the
proposal step WeakC4 used before exact checking. Fixture X{0,1} O{2,3}, X to move: only the centre (cell 4) wins;
the empty map gives no move at the root."""
from __future__ import annotations

import random

import pytest

from games.tictactoe import TicTacToe, TTTState
from harness.steady_exceptions import needed, verify_with
from harness.steady_local import leaf_bits, local_search
from harness.steady_state import Facts, verify

GAME = TicTacToe()


def _ttt(x, o):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=0 if len(x) == len(o) else 1, winner=None, done=False)


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


FORK = _ttt((0, 1), (2, 3))
EARLY = _ttt((0,), (1,))


def test_a_winning_map_costs_only_its_own_bits():
    assert leaf_bits(Facts(GAME), FORK, {4: 0}, 2, 1, _winning, 10_000) == {
        "bits": 5 + 1, "exceptions": {}, "own_positions": 5}


def test_a_map_with_no_root_move_pays_for_an_exception_there():
    r = leaf_bits(Facts(GAME), FORK, {}, 2, 1, _winning, 10_000)
    assert r == {"bits": 5 + 7, "exceptions": {GAME.state_key(FORK): 4}, "own_positions": 5}


def test_a_map_undefined_at_the_root_is_charged_for_the_whole_tree_below():
    facts = Facts(GAME)
    r = leaf_bits(facts, EARLY, {}, 2, 3, _winning, 100_000)
    assert r["exceptions"] == needed(facts, EARLY, {}, 2, _winning, 100_000)["exceptions"]
    assert len(r["exceptions"]) > 1 and r["own_positions"] > len(r["exceptions"])


def test_a_walk_past_the_cap_has_no_size():
    assert leaf_bits(Facts(GAME), FORK, {4: 0}, 2, 1, _winning, 3)["bits"] is None


def test_local_search_finds_the_fork_and_the_map_verifies():
    facts = Facts(GAME)
    r = local_search(facts, FORK, 2, 1, _winning, random.Random(0), seconds=10.0, cap=10_000)
    assert r["status"] == "found" and r["exceptions"] == {} and verify(facts, FORK, r["levels"], 2)["won"]
    assert r["bits"] == leaf_bits(facts, FORK, r["levels"], 2, 1, _winning, 10_000)["bits"]


def test_local_search_shrinks_a_whole_game_leaf_and_the_leaf_verifies():
    facts = Facts(GAME)
    start = leaf_bits(facts, EARLY, {}, 3, 2, _winning, 100_000)["bits"]
    r = local_search(facts, EARLY, 3, 2, _winning, random.Random(1), seconds=5.0, cap=100_000)
    assert r["bits"] < start and verify_with(facts, EARLY, r["levels"], 3, r["exceptions"], 100_000)["won"]
    assert r["bits"] == leaf_bits(facts, EARLY, r["levels"], 3, 2, _winning, 100_000)["bits"]


def test_a_search_out_of_time_mid_walk_has_no_leaf():
    r = local_search(Facts(GAME), EARLY, 2, 1, _winning, random.Random(0), seconds=0.0, cap=100_000)
    assert r["status"] == "budget" and r["evaluations"] == 1 and r["bits"] is None and r["exceptions"] == {}


def test_a_walk_past_the_deadline_has_no_size():
    import time

    assert leaf_bits(Facts(GAME), EARLY, {}, 2, 1, _winning, 100_000, deadline=time.monotonic() - 1)["bits"] is None


def test_local_search_reports_its_best_leaf_when_time_runs_out():
    facts = Facts(GAME)
    r = local_search(facts, EARLY, 2, 1, _winning, random.Random(0), seconds=0.5, cap=100_000)
    assert r["bits"] is not None and r["evaluations"] >= 1
    assert verify_with(facts, EARLY, r["levels"], 2, r["exceptions"], 100_000)["won"]


def test_local_search_refuses_a_root_its_side_cannot_win():
    with pytest.raises(ValueError, match="cannot win"):
        local_search(Facts(GAME), GAME.initial_state(), 2, 1, _winning, random.Random(0), 1.0, 100_000)


def test_local_search_starts_from_a_given_map():
    r = local_search(Facts(GAME), FORK, 2, 1, _winning, random.Random(0), seconds=10.0, cap=10_000, start={4: 0})
    assert r["status"] == "found" and r["evaluations"] == 1 and r["levels"] == {4: 0}


def test_a_capped_walk_is_never_read_as_a_win():
    r = local_search(Facts(GAME), FORK, 2, 1, _winning, random.Random(0), seconds=0.3, cap=3, start={4: 0})
    assert r["status"] == "budget" and r["bits"] is None


class _Script:
    """A scripted rng: picks cells and levels from fixed lists, repeating the last pair once they run out."""

    def __init__(self, cells, levels):
        self.cells, self.levels = list(cells), list(levels)

    def choice(self, seq):
        return self.cells.pop(0) if len(self.cells) > 1 else self.cells[0]

    def randrange(self, n):
        return self.levels.pop(0) if len(self.levels) > 1 else self.levels[0]


def _scripted(monkeypatch, costs):
    import harness.steady_local as sl

    def fake(facts, root, levels, n, level_bits, winning, cap, deadline=None, walker=None):
        bits = costs.get(tuple(sorted(levels.items())), 70)
        return {"bits": bits, "exceptions": {} if bits == 0 else {"x": 1}, "own_positions": 1}

    monkeypatch.setattr(sl, "leaf_bits", fake)


def test_a_change_that_grows_the_leaf_is_rejected(monkeypatch):
    _scripted(monkeypatch, {(): 50, ((7, 1),): 90, ((5, 0),): 0, ((5, 0), (7, 1)): 30})
    r = local_search(Facts(GAME), FORK, 2, 1, _winning, _Script([7, 5], [1, 0]), seconds=1.0, cap=10_000)
    assert r["status"] == "found" and r["levels"] == {5: 0} and r["evaluations"] == 3


def test_an_equal_sized_change_is_taken(monkeypatch):
    _scripted(monkeypatch, {(): 50, ((7, 1),): 50, ((5, 0), (7, 1)): 0})
    r = local_search(Facts(GAME), FORK, 2, 1, _winning, _Script([7, 5], [1, 0]), seconds=1.0, cap=10_000)
    assert r["status"] == "found" and r["levels"] == {5: 0, 7: 1}


def test_the_best_leaf_is_kept_when_later_changes_only_tie(monkeypatch):
    _scripted(monkeypatch, {(): 50, ((7, 1),): 40, ((5, 0), (7, 1)): 40})
    r = local_search(Facts(GAME), FORK, 2, 1, _winning, _Script([7, 5], [1, 0]), seconds=0.3, cap=10_000)
    assert r["status"] == "budget" and r["levels"] == {7: 1} and r["bits"] == 40


def test_a_capped_change_is_never_taken(monkeypatch):
    import harness.steady_local as sl

    def fake(facts, root, levels, n, level_bits, winning, cap, deadline=None, walker=None):
        key = tuple(sorted(levels.items()))
        bits = {(): 50, ((5, 0),): 0}.get(key, None if key == ((7, 1),) else 70)
        return {"bits": bits, "exceptions": {} if bits == 0 else {"x": 1}, "own_positions": 1}

    monkeypatch.setattr(sl, "leaf_bits", fake)
    r = local_search(Facts(GAME), FORK, 2, 1, _winning, _Script([7, 5], [1, 0]), seconds=1.0, cap=10_000)
    assert r["status"] == "found" and r["levels"] == {5: 0}


def test_a_level_that_blocks_the_win_can_be_cleared():
    facts = Facts(GAME)
    r = local_search(facts, FORK, 1, 1, _winning, random.Random(0), seconds=10.0, cap=10_000, start={5: 0})
    assert r["status"] == "found" and 5 not in r["levels"] and verify(facts, FORK, r["levels"], 1)["won"]


def test_every_walk_of_a_search_gets_its_deadline(monkeypatch):
    import harness.steady_local as sl

    seen = []

    def fake(facts, root, levels, n, level_bits, winning, cap, deadline=None, walker=None):
        seen.append(deadline)
        return {"bits": 50, "exceptions": {"x": 1}, "own_positions": 1}

    monkeypatch.setattr(sl, "leaf_bits", fake)
    local_search(Facts(GAME), FORK, 2, 1, _winning, _Script([7, 5], [1, 0]), seconds=0.2, cap=10_000)
    assert len(seen) > 1 and all(d is not None for d in seen)


def test_the_walk_can_be_replaced_and_receives_every_argument():
    calls = []

    def walker(facts, root, levels, n_levels, winning, cap, deadline=None):
        calls.append((root, dict(levels), n_levels, cap, deadline is not None))
        return needed(facts, root, levels, n_levels, winning, cap, deadline)

    facts = Facts(GAME)
    r = local_search(facts, FORK, 2, 1, _winning, random.Random(0), seconds=10.0, cap=10_000, walker=walker)
    assert r["status"] == "found" and calls and all(c[0] is FORK and c[2] == 2 and c[3] == 10_000 and c[4]
                                                     for c in calls)
    assert calls[0][1] == {} and len(calls) == r["evaluations"]
    assert leaf_bits(facts, FORK, {4: 0}, 2, 1, _winning, 10_000, walker=walker)["bits"] == 6


def test_the_default_walk_is_the_python_reference():
    import inspect

    import harness.steady_local as sl

    assert inspect.signature(sl.local_search).parameters["walker"].default is needed
    assert inspect.signature(sl.leaf_bits).parameters["walker"].default is needed
