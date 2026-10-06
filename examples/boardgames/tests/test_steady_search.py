"""Direct tests for harness/steady_search.py — SAT with counterexamples over priority maps. The decisive property is
exactness: on every tic-tac-toe position it is asked about, "found" must come with a map that verifies, and
"impossible" must mean no map in the language wins — checked against brute force over every map."""
from __future__ import annotations

import itertools

import pytest

from games.tictactoe import TicTacToe, TTTState
from harness.steady_search import _Encoding, _failures, find_steady_state
from harness.steady_state import Facts, choose, verify

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


def _find(s, n_levels=2, **kw):
    budget = {"max_constraints": 500, "conflicts": 100_000, "seconds": 60.0, **kw}
    return find_steady_state(Facts(GAME), s, _winning, n_levels, **budget)


def _brute_force(facts, s, n_levels):
    empty = facts.empty_cells(s)
    for assignment in itertools.product([None] + list(range(n_levels)), repeat=len(empty)):
        levels = {c: k for c, k in zip(empty, assignment) if k is not None}
        if verify(facts, s, levels, n_levels)["won"]:
            return True
    return False


def _cases(pieces):
    for x in itertools.combinations(range(9), pieces):
        for o in itertools.combinations([i for i in range(9) if i not in x], pieces):
            s = _ttt(x, o)
            if GAME.is_terminal(s) or any(c.done for c in [s]) or GAME.position_value(s) != 1:
                continue
            if verify(Facts(GAME), s, {}, 1)["won"]:
                continue
            yield s


def test_the_fork_position_is_found_and_the_map_verifies():
    s = _ttt((0, 1), (2, 3))
    r = _find(s)
    assert r["status"] == "found" and verify(Facts(GAME), s, r["levels"], 2)["won"]
    assert r["iterations"] >= 1 and r["constraints"] >= 1


@pytest.mark.parametrize("lines", [1, 8])
@pytest.mark.parametrize("pieces", [1, 2, 3])
def test_found_and_impossible_agree_with_brute_force_over_every_one_level_map(pieces, lines):
    facts = Facts(GAME)
    seen = {"found": 0, "impossible": 0}
    for s in _cases(pieces):
        r = find_steady_state(facts, s, _winning, 1, max_constraints=500, conflicts=100_000, seconds=60.0, lines=lines)
        assert r["status"] in seen
        assert (r["status"] == "found") == _brute_force(facts, s, 1), s
        if r["status"] == "found":
            assert verify(facts, s, r["levels"], 1)["won"]
        seen[r["status"]] += 1
    assert seen["found"] and (seen["impossible"] or pieces != 1)
    if pieces == 1:
        assert seen["impossible"]


def test_a_position_with_no_winning_move_is_impossible_at_once():
    drawn = TicTacToe().initial_state()
    r = _find(drawn)
    assert r["status"] == "impossible" and r["iterations"] == 1


@pytest.mark.parametrize("budget", [{"max_constraints": 0}, {"seconds": 0.0}, {"cap": 2}])
def test_an_exhausted_budget_is_reported_not_guessed(budget):
    assert _find(_ttt((0, 1), (2, 3)), **budget)["status"] == "budget"


def test_a_wrong_oracle_is_caught_instead_of_looping():
    with pytest.raises(RuntimeError, match="oracle"):
        find_steady_state(Facts(GAME), TicTacToe().initial_state(),
                          lambda states: [set(GAME.legal_actions(s)) for s in states],
                          2, max_constraints=5000, conflicts=100_000, seconds=60.0)


def _fired_level(enc, state, model):
    key = GAME.state_key(state)
    fired = [k for k in range(enc.n_levels) if enc.pool.id(("f", key, k)) in model]
    return fired


@pytest.mark.parametrize("seed", range(40))
def test_the_encoding_fires_exactly_the_level_the_rule_reads(seed):
    import random

    rng = random.Random(seed)
    s = _ttt((0, 1), (2, 3))
    facts = Facts(GAME)
    enc = _Encoding(facts, s, 3, "g4")
    enc.constrain(s, set(facts.of(s)[1]))
    levels = {c: rng.choice([0, 1, 2]) for c in enc.cells if rng.random() < 0.7}
    assumptions = [enc.x(c, k) if levels.get(c) == k else -enc.x(c, k) for c in enc.cells for k in range(3)]
    move = choose(facts, s, levels, 3)
    ok = enc.solver.solve(assumptions=assumptions)
    if move is None:
        assert not ok
    else:
        assert ok and _fired_level(enc, s, set(enc.solver.get_model())) == [levels[facts.of(s)[2][move]]]
    enc.solver.delete()


def test_every_model_is_a_map_with_at_most_one_level_per_cell():
    s = _ttt((0,), (4,))
    facts = Facts(GAME)
    enc = _Encoding(facts, s, 4, "g4")
    enc.constrain(s, set(facts.of(s)[1]))
    for _ in range(20):
        assert enc.solver.solve()
        model = set(lit for lit in enc.solver.get_model() if lit > 0)
        per_cell = [sum(enc.x(c, k) in model for k in range(4)) for c in enc.cells]
        assert max(per_cell) <= 1
        xs = [enc.x(c, k) for c in enc.cells for k in range(4)]
        chosen = [lit for lit in xs if lit in model]
        enc.solver.add_clause([-lit for lit in chosen] if chosen else xs)
    enc.solver.delete()


def test_a_forced_move_into_a_draw_is_impossible_at_once():
    drawn = _ttt((0, 2, 3, 7), (1, 4, 5, 6))
    r = _find(drawn)
    assert r["status"] == "impossible" and r["iterations"] == 1


def test_several_failing_lines_per_round_need_no_more_rounds_than_one():
    s = _ttt((0,), (1,))
    assert GAME.position_value(s) == 1
    one, many = _find(s, n_levels=3, lines=1), _find(s, n_levels=3, lines=16)
    assert one["status"] == many["status"] == "found" and many["iterations"] <= one["iterations"]
    assert many["constraints"] > one["constraints"]


def test_a_walk_collects_up_to_the_limit_of_failing_lines_each_from_the_root():
    s = _ttt((0, 1), (2, 3))
    facts = Facts(GAME)
    status, one = _failures(facts, s, {5: 0}, 2, 10_000, 1)
    status_many, many = _failures(facts, s, {5: 0}, 2, 10_000, 4)
    assert status == status_many == "failed" and len(one) == 1 and len(many) > 1
    assert all(line[0] == s for line in many)
    assert _failures(facts, s, {4: 0}, 2, 10_000, 4) == ("clean", [])


def test_a_walk_past_its_deadline_stops_as_a_timeout():
    s = _ttt((0, 1), (2, 3))
    assert _failures(Facts(GAME), s, {5: 0}, 2, 10_000, 4, deadline=0.0) == ("timeout", [])


class _StuckSolver:
    """A solver whose solve only returns when interrupted — a solve that would overrun any budget."""

    def __init__(self, name=None):
        import threading

        self._stop = threading.Event()

    def add_clause(self, clause):
        pass

    def conf_budget(self, n):
        pass

    def solve_limited(self, expect_interrupt=False):
        return None if self._stop.wait(30) else True

    def interrupt(self):
        self._stop.set()

    def clear_interrupt(self):
        self._stop.clear()

    def delete(self):
        pass


def test_a_solve_still_running_at_the_deadline_is_interrupted(monkeypatch):
    import time

    import pysat.solvers

    monkeypatch.setattr(pysat.solvers, "Solver", _StuckSolver)
    t = time.monotonic()
    r = find_steady_state(Facts(GAME), _ttt((0, 1), (2, 3)), _winning, 2, max_constraints=500, conflicts=10,
                          seconds=0.5)
    assert r["status"] == "budget" and time.monotonic() - t < 5


def test_a_walk_that_times_out_ends_the_search_as_budget(monkeypatch):
    import harness.steady_search as ss

    monkeypatch.setattr(ss, "_failures", lambda *a, **k: ("timeout", []))
    assert _find(_ttt((0, 1), (2, 3)))["status"] == "budget"


FORK = _ttt((0, 1), (2, 3))
ROUNDS = _ttt((3, 7), (0, 1))


def test_a_found_map_given_as_the_hint_is_found_in_one_round():
    cold = _find(ROUNDS)
    warm = _find(ROUNDS, hint=cold["levels"])
    assert cold["status"] == "found" and cold["iterations"] == 3
    assert warm["status"] == "found" and warm["iterations"] == 1 and warm["levels"] == cold["levels"]


def test_seeds_are_constrained_before_the_first_solve():
    facts = Facts(GAME)
    levels = _find(ROUNDS)["levels"]
    after = GAME.step(ROUNDS, choose(facts, ROUNDS, levels, 2))
    seed = next(c for c in (GAME.step(after, b) for b in GAME.legal_actions(after)) if not GAME.is_terminal(c))
    r = _find(ROUNDS, hint=levels, seeds=[seed])
    assert r["status"] == "found" and r["iterations"] == 1 and r["constraints"] == 2


def test_a_hint_outside_the_language_is_refused():
    with pytest.raises(ValueError, match="hint"):
        _find(FORK, hint={0: 0})
    with pytest.raises(ValueError, match="hint"):
        _find(FORK, hint={4: 2})


def test_a_hint_never_changes_the_answer():
    facts = Facts(GAME)
    checked = 0
    for s in _cases(2):
        empty = facts.empty_cells(s)
        wrong = {c: i % 2 for i, c in enumerate(empty)}
        r = _find(s, hint=wrong)
        assert (r["status"] == "found") == _brute_force(facts, s, 2)
        if r["status"] == "found":
            assert verify(facts, s, r["levels"], 2)["won"]
        checked += 1
    assert checked >= 5
