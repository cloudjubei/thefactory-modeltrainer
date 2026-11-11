"""Direct tests for harness/best_first.py — proof-number search with maps as proofs of whole subtrees, no solver. The
grader is a plain minimax over tic-tac-toe, which the search never sees. Fixtures: FORK, X{0,1} O{2,3}, which the
empty map wins by the centre; LONE, X{5} O{6}, where only 8 wins and the empty map plays 0 and draws."""
from __future__ import annotations

import time
from functools import lru_cache

import pytest

from games.tictactoe import TicTacToe, TTTState
from harness.best_first import BestFirst
from harness.composed import check
from harness.self_proof import Prover
from harness.steady_state import Facts
from harness.strategy_file import replay

GAME = TicTacToe()
CELLS = [2, 1, 0, 5, 4, 3, 8, 7, 6]
LEVELS = 8
CAP = 100_000


def _facts():
    return Facts(GAME, ties="lowest", unlevelled="last")


def _ttt(x, o):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=0 if len(x) == len(o) else 1, winner=None, done=False)


EMPTY, FORK, LONE = GAME.initial_state(), _ttt((0, 1), (2, 3)), _ttt((5,), (6,))
KEY, LONE_KEY = GAME.state_key(FORK), GAME.state_key(LONE)


@lru_cache(maxsize=None)
def _value(s) -> int:
    if GAME.is_terminal(s):
        w = GAME.winner(s)
        return 0 if w is None else 1 if w == GAME.current_player(s) else -1
    return max(-_value(GAME.step(s, a)) for a in GAME.legal_actions(s))


def _reachable():
    seen, stack = {}, [EMPTY]
    while stack:
        s = stack.pop()
        if s in seen or GAME.is_terminal(s):
            continue
        seen[s] = True
        stack.extend(GAME.step(s, a) for a in GAME.legal_actions(s))
    return list(seen)


def _search(**kwargs):
    return BestFirst(Prover(_facts(), LEVELS, CAP, **kwargs))


@pytest.mark.parametrize("compose", [False, True])
def test_every_position_is_proved_exactly_when_its_side_wins_and_each_proof_holds(compose):
    prover = Prover(_facts(), LEVELS, CAP, compose=compose)
    for s in _reachable():
        result = BestFirst(prover).prove(s)
        assert result is (_value(s) == 1)
        if result:
            strategy = prover.strategy(s)
            won = (check(_facts(), s, strategy, LEVELS, CAP)["won"] if compose
                   else replay(_facts(), s, strategy, LEVELS, CAP, CELLS)["won"])
            assert won


def test_the_proof_plays_the_one_winning_move():
    search = _search()
    assert search.prove(LONE) is True and search.prover.proved[LONE_KEY] == {"move": 8}
    assert search.expansions >= 1


def test_a_position_a_map_proves_takes_no_expansion():
    search = _search()
    assert search.prove(FORK) is True and search.prover.proved[KEY] == {"leaf": 0} and search.expansions == 0
    found = _search(search=lambda state, start: {"levels": {8: 0}, "exceptions": {}})
    assert found.prove(LONE) is True and found.prover.proved[LONE_KEY] == {"leaf": 1} and found.expansions == 0


def test_a_position_already_proved_is_taken_as_proved():
    prover = Prover(_facts(), LEVELS, CAP)
    prover.proved[LONE_KEY] = {"move": 8}
    search = BestFirst(prover)
    assert search.prove(LONE) is True and search.expansions == 0 and search.evaluations == 1


def test_a_position_with_no_safe_move_is_disproved_at_once():
    lost = _ttt((2, 6, 8), (4, 5, 7))
    assert not _facts().safe_moves(lost) and _value(lost) == -1
    search = _search()
    assert search.prove(lost) is False and search.expansions == 0


def test_a_drawn_position_is_disproved():
    search = _search()
    assert _value(EMPTY) == 0 and search.prove(EMPTY) is False and search.expansions > 1


DEEP = _ttt((6,), (3,))


def test_a_spent_budget_says_neither():
    full = _search()
    starts = []
    real = full._expand

    def watched(node):
        starts.append(full.evaluations)
        return real(node)

    full._expand = watched
    assert full.prove(DEEP) is True and full.expansions == 5
    for budget in (1, starts[-1]):
        cut = BestFirst(Prover(_facts(), LEVELS, CAP), max_evaluations=budget)
        assert cut.prove(DEEP) is None and cut.evaluations >= budget
    assert BestFirst(Prover(_facts(), LEVELS, CAP), max_evaluations=starts[-1] + 1).prove(DEEP) is True
    assert BestFirst(Prover(_facts(), LEVELS, CAP), deadline=time.monotonic() - 1).prove(DEEP) is None


def test_a_position_reached_twice_is_one_node():
    search = _search()
    search.prove(EMPTY)
    assert search.evaluations == len(search.nodes)
    shared = [n for n in search.nodes.values() if len(n.parents) > 1]
    assert shared


def test_proofs_by_moves_are_kept_with_the_prover_s_so_its_walks_stop_there():
    prover = Prover(_facts(), LEVELS, CAP, compose=True)
    assert BestFirst(prover).prove(LONE) is True
    moved = [k for k, n in prover.proved.items() if "move" in n]
    assert LONE_KEY in moved and all(_value(_state(k)) == 1 for k in moved)


def _state(key):
    board, to_move = key
    return TTTState(board=tuple(board), to_move=to_move, winner=None, done=False)


def test_a_large_position_is_still_proved_by_the_empty_map():
    prover = Prover(_facts(), LEVELS, CAP, search=lambda state, start: pytest.fail("searched"), search_below=0)
    search = BestFirst(prover)
    assert search.prove(FORK) is True and prover.proved[KEY] == {"leaf": 0} and search.expansions == 0


@pytest.mark.parametrize("compose", [False, True])
def test_with_no_maps_every_position_is_still_proved_exactly(compose):
    prover = Prover(_facts(), LEVELS, CAP, compose=compose)
    for s in _reachable():
        assert BestFirst(prover, maps=False).prove(s) is (_value(s) == 1)
    assert all("move" in n for n in prover.proved.values()) and prover.maps == [{}]


def test_a_move_that_wins_at_once_proves_its_position():
    win_now = _ttt((0, 1), (3, 4))
    search = BestFirst(Prover(_facts(), LEVELS, CAP), maps=False)
    assert search.prove(win_now) is True and search.prover.proved[GAME.state_key(win_now)] == {"move": 2}


def test_maps_cut_the_positions_a_proof_needs():
    with_maps, without = BestFirst(Prover(_facts(), LEVELS, CAP)), BestFirst(Prover(_facts(), LEVELS, CAP), maps=False)
    assert with_maps.prove(DEEP) is True and without.prove(DEEP) is True
    assert with_maps.expansions < without.expansions

