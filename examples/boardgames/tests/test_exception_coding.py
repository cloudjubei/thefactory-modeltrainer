"""Direct tests for harness/exception_coding.py — exceptions coded in walk order: a decoder that replays the leaf with
the game's rules flags only CONTESTED positions (no immediate win, two or more safe moves, the map gives a move),
knows every position where the map gives none must be an exception, and reads each exception's move among the safe
moves left — every safe move, whether or not the rule prefers forcing ones. Fixture X{0,1} O{2,3}, X to move: five
safe moves, only the centre (4) wins, and after it every O reply leaves X an immediate win."""
from __future__ import annotations

import pytest

from games.tictactoe import TicTacToe, TTTState
from harness.exception_coding import enumerative_bits, walk_order_cost
from harness.steady_state import Facts

GAME = TicTacToe()


def _ttt(x, o):
    board = [0] * 9
    for i in x:
        board[i] = 1
    for i in o:
        board[i] = 2
    return TTTState(board=tuple(board), to_move=0 if len(x) == len(o) else 1, winner=None, done=False)


FORK = _ttt((0, 1), (2, 3))
KEY = GAME.state_key(FORK)


def test_choosing_k_of_m_costs_the_count_and_the_choice():
    assert enumerative_bits(0, 0) == 0
    assert enumerative_bits(1, 0) == 1
    assert enumerative_bits(1, 1) == 1
    assert enumerative_bits(4, 2) == 3 + 3
    assert enumerative_bits(10, 0) == 4
    with pytest.raises(ValueError, match="cannot choose"):
        enumerative_bits(2, 3)


def test_a_position_the_map_leaves_undefined_needs_no_flag_only_its_move():
    r = walk_order_cost(Facts(GAME), FORK, {}, 2, {KEY: 4})
    assert r == {"contested": 0, "flagged": 0, "implicit": 1, "move_bits": 3, "bits": 3}


def test_an_exception_at_a_contested_position_is_flagged_and_its_move_read_among_the_rest():
    r = walk_order_cost(Facts(GAME), FORK, {5: 0}, 2, {KEY: 4})
    assert r == {"contested": 1, "flagged": 1, "implicit": 0, "move_bits": 2, "bits": 1 + 2}


def test_a_contested_position_without_an_exception_still_costs_its_share_of_the_flags():
    assert walk_order_cost(Facts(GAME), FORK, {4: 0}, 2, {}) == {
        "contested": 1, "flagged": 0, "implicit": 0, "move_bits": 0, "bits": 1}


def test_the_walk_follows_the_exception_not_the_map():
    early = _ttt((0,), (1,))
    facts = Facts(GAME)
    with_fix = walk_order_cost(facts, early, {}, 2, _needed(facts, early, {}))
    assert with_fix["implicit"] > 1


def test_an_exception_where_the_rule_is_forced_is_refused():
    after = GAME.step(GAME.step(FORK, 4), 5)
    with pytest.raises(ValueError, match="forced"):
        walk_order_cost(Facts(GAME), FORK, {4: 0}, 2, {GAME.state_key(after): 8})


def test_an_exception_move_that_is_not_safe_is_refused():
    """X{0,1,3} O{2,5,6}, X to move: O threatens 4 and 8, so no move is safe and the map is undefined."""
    lost = _ttt((0, 1, 3), (2, 5, 6))
    assert Facts(GAME).of(lost)[:2] == ([], [])
    with pytest.raises(ValueError, match="safe"):
        walk_order_cost(Facts(GAME), lost, {}, 2, {GAME.state_key(lost): 4})


def test_an_undefined_position_without_an_exception_is_refused():
    with pytest.raises(ValueError, match="undefined"):
        walk_order_cost(Facts(GAME), FORK, {}, 2, {})


def _needed(facts, root, levels):
    from harness.steady_exceptions import needed

    def winning(states):
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

    return needed(facts, root, levels, 2, winning, 100_000)["exceptions"]


def test_an_exception_where_only_one_move_is_safe_is_refused():
    """X{1,5} O{6,7}, X to move: O threatens 8, so 8 is the only safe move and the rule plays it."""
    blocked = _ttt((1, 5), (6, 7))
    assert Facts(GAME).of(blocked)[:2] == ([], [8])
    with pytest.raises(ValueError, match="forced"):
        walk_order_cost(Facts(GAME), blocked, {}, 2, {GAME.state_key(blocked): 8})


def test_an_exception_with_a_single_alternative_costs_no_move_bits():
    """X{0,1,5,6} O{2,3,4}, O to move: two safe moves; the map plays 7, so the exception's 8 is the only other."""
    last = _ttt((0, 1, 5, 6), (2, 3, 4))
    assert Facts(GAME).of(last)[:2] == ([], [7, 8])
    assert walk_order_cost(Facts(GAME), last, {7: 0}, 2, {GAME.state_key(last): 8}) == {
        "contested": 1, "flagged": 1, "implicit": 0, "move_bits": 0, "bits": 1}


def test_coding_past_its_deadline_is_refused():
    import time

    with pytest.raises(TimeoutError):
        walk_order_cost(Facts(GAME), FORK, {}, 2, {KEY: 4}, deadline=time.monotonic() - 1)
    assert walk_order_cost(Facts(GAME), FORK, {}, 2, {KEY: 4}, deadline=time.monotonic() + 60)["bits"] == 3


def test_with_forcing_moves_preferred_an_exception_is_read_among_every_safe_move():
    """A Connect-4 position from L4's first run: preferring forcing moves leaves 4 and 5, and the winning exception
    plays 0, a safe move that does not force; with no map every position the preference leaves open is undefined, so
    every exception is implicit, read among all the safe moves (3 bits at the root), and the five positions it narrows
    to one winning move among several safe ones are contested, unflagged."""
    from games.connect4 import C4State, Connect4
    from harness.native_leaf import needed

    game = Connect4()
    board = (1, 2, 1, 1, 0, 0, 2, 2, 1, 1, 2, 0, 0, 2, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 1, 0, 0, 1, 0, 2, 1, 1, 0, 0, 2, 0,
             0, 0, 2, 0, 0, 2)
    root = C4State(board, 0, None, False)
    facts = Facts(game, ties="lowest", forcing=True)
    assert facts.of(root)[1] == [4, 5] and facts.safe_moves(root) == [0, 1, 2, 4, 5]
    exceptions = dict(needed(facts, root, {}, 8, None, 1_000_000)["exceptions"])
    assert exceptions[game.state_key(root)] == 0
    r = walk_order_cost(facts, root, {}, 8, exceptions)
    assert (r["implicit"], r["contested"], r["flagged"]) == (len(exceptions), 5, 0) and r["move_bits"] >= 3


def test_with_forcing_moves_preferred_a_contested_exception_is_read_among_every_safe_move():
    """FORK with forcing moves preferred and the map on 7: three forcing moves, the map plays 7, which loses — so the
    exception (the centre) is flagged at a contested position and read among the four other safe moves."""
    facts = Facts(GAME, forcing=True)
    r = walk_order_cost(facts, FORK, {7: 0}, 2, {KEY: 4})
    assert (r["contested"], r["flagged"], r["implicit"], r["move_bits"]) == (1, 1, 0, 2)


def test_with_forcing_moves_preferred_a_position_left_one_forcing_move_is_contested_not_forced():
    """A Connect-4 position where preferring forcing moves leaves only 3, which does not win, among five safe moves (2
    wins): the rule plays 3 with no map, so the exception at the root is flagged at a contested position and read
    among the four other safe moves."""
    from games.connect4 import C4State, Connect4
    from harness.native_leaf import needed

    game = Connect4()
    board = (2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 0, 0, 2, 2, 0, 1, 1, 0, 0, 1, 1, 0, 2, 1, 0, 0, 1, 1, 0, 0,
             2, 0, 0, 2, 0, 0)
    root = C4State(board, 0, None, False)
    facts = Facts(game, ties="lowest", forcing=True)
    assert facts.of(root)[1] == [3] and facts.safe_moves(root) == [0, 2, 3, 5, 6]
    exceptions = dict(needed(facts, root, {}, 8, None, 1_000_000)["exceptions"])
    assert exceptions[game.state_key(root)] == 2
    r = walk_order_cost(facts, root, {}, 8, exceptions)
    assert r["contested"] >= 1 and r["flagged"] >= 1 and r["move_bits"] >= 2
