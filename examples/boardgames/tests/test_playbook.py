"""Direct tests for harness/playbook.py — §C.50 written rules: predicates of a position and a move, rules as
conjunctions of them printed as logical statements, playbooks as ordered decision lists, and their coverage and
precision against exact move values."""
from __future__ import annotations

import random

import pytest

from games.connect4 import COLS, ROWS, C4State, Connect4
from games.tictactoe import TicTacToe, TTTState
from harness.playbook import (PREDICATES, Rule, decide, evaluate, newell_simon, tactics, their_view)

TTT = TicTacToe()
C4 = Connect4()


def _ttt(cells: str, to_move: int) -> TTTState:
    """`cells` is 9 characters, row by row: x = player 0, o = player 1, . = empty."""
    return TTTState(tuple({"x": 1, "o": 2, ".": 0}[c] for c in cells), to_move, None, False)


def _c4(columns: list, to_move: int) -> C4State:
    """`columns[c]` is the stack in column c from the bottom: x = player 0, o = player 1."""
    board = [0] * (ROWS * COLS)
    for c, stack in enumerate(columns):
        for r, ch in enumerate(stack):
            board[r * COLS + c] = {"x": 1, "o": 2}[ch]
    return C4State(tuple(board), to_move, None, False)


def _holds(name, game, state):
    return sorted(m for m in game.legal_actions(state) if PREDICATES[name](game, state, m))


def test_their_view_is_the_same_board_with_the_other_side_to_move():
    s = _ttt("x...o....", 0)
    assert their_view(s).board == s.board and their_view(s).to_move == 1 and their_view(their_view(s)) == s


@pytest.mark.parametrize("cells,to_move,name,moves", [
    ("xx.oo....", 0, "wins", [2]),
    ("xx.oo....", 1, "wins", [5]),
    ("xx.o.....", 1, "blocks", [2]),
    ("xx.oo....", 0, "blocks", [5]),
    ("x...o...x", 1, "gives_win", []),
    ("xx..o....", 1, "gives_win", [3, 5, 6, 7, 8]),
    ("x...o...x", 0, "forks", [2, 6]),
    ("xo..o..x.", 0, "forks", [6]),
    ("xx.oo....", 1, "forks", [2]),
])
def test_the_tactical_predicates_on_tic_tac_toe(cells, to_move, name, moves):
    assert _holds(name, TTT, _ttt(cells, to_move)) == moves


@pytest.mark.parametrize("cells,to_move,name,moves", [
    ("x...o....", 0, "makes_threat", [1, 2, 3, 6]),
    ("x...o...x", 1, "opp_fork_at", [2, 6]),
    ("x...o...x", 1, "gives_fork", [2, 6]),
])
def test_the_lookahead_predicates_on_tic_tac_toe(cells, to_move, name, moves):
    assert _holds(name, TTT, _ttt(cells, to_move)) == moves


def test_the_tic_tac_toe_geometry_is_refused_on_another_game():
    with pytest.raises(ValueError, match="tic-tac-toe only"):
        PREDICATES["centre"](C4, C4.initial_state(random.Random(0)), 3)


def test_a_fork_needs_the_opponent_to_have_no_immediate_win():
    assert _holds("forks", TTT, _ttt("xo..o...x", 0)) == []


def test_the_tactical_predicates_on_connect_4_read_the_landing_square():
    s = _c4(["xxx", "ooo", "", "", "", "", ""], 0)
    assert _holds("wins", C4, s) == [0] and _holds("blocks", C4, s) == [1]
    lifted = _c4(["", "ox", "xx", "ox", "", "", "oo"], 1)
    assert _holds("gives_win", C4, lifted) == [0, 4]
    assert _holds("blocks", C4, _c4(["", "x", "x", "x", "o", "", "o"], 1)) == [0]


def test_the_geometry_predicates_of_tic_tac_toe():
    s = _ttt("o...x....", 0)
    assert _holds("centre", TTT, _ttt(".........", 0)) == [4]
    assert _holds("corner", TTT, s) == [2, 6, 8] and _holds("side", TTT, s) == [1, 3, 5, 7]
    assert _holds("opposite_corner", TTT, s) == [8]


def test_a_rule_recommends_the_moves_meeting_every_literal_and_prints_as_a_statement():
    rule = Rule("safe corner", (("corner", True), ("gives_win", False)))
    s = _ttt("xx..o....", 1)
    assert rule.moves(TTT, s) == [2]
    assert rule.text() == "safe corner: IF ∃m: corner(m) ∧ ¬gives_win(m) THEN play m"
    assert Rule("none", (("wins", True),)).moves(TTT, _ttt(".........", 0)) == []


def test_an_unknown_predicate_is_refused_when_the_rule_is_made():
    with pytest.raises(ValueError, match="unknown predicate"):
        Rule("bad", (("teleports", True),))


def test_the_first_rule_that_fires_decides():
    book = [Rule("win", (("wins", True),)), Rule("block", (("blocks", True),)), Rule("centre", (("centre", True),))]
    assert decide(book, TTT, _ttt("xx.oo....", 1)) == (0, [5])
    assert decide(book, TTT, _ttt("xx.o.....", 1)) == (1, [2])
    assert decide(book, TTT, _ttt(".........", 0)) == (2, [4])
    assert decide(book[:2], TTT, _ttt(".........", 0)) == (None, [])


def test_the_evaluation_counts_firings_and_optimal_recommendations_per_rule_and_for_the_book():
    book = [Rule("win", (("wins", True),)), Rule("centre", (("centre", True),))]
    positions = [_ttt("xx.oo....", 0), _ttt(".........", 0), _ttt("x...o....", 0), _ttt("xo.......", 0)]
    values = [{2: 1, 5: 0, 6: -1}, {4: 0, 0: 0, 1: 0}, {1: 0, 8: 0}, {4: 1, 3: 0}]
    r = evaluate(book, TTT, positions, values)
    assert r["positions"] == 4 and r["covered"] == 3 and r["correct"] == 3
    assert [(x["fires"], x["correct"], x["first_fires"], x["first_correct"]) for x in r["rules"]] == \
        [(1, 1, 1, 1), (2, 2, 2, 2)]
    assert r["rules"][0]["text"] == book[0].text()
    wrong = evaluate(book, TTT, positions, [{2: 0, 5: 1}, {4: -1, 0: 0}, {1: 0}, {4: 1}])
    assert wrong["correct"] == 1 and [x["first_correct"] for x in wrong["rules"]] == [0, 1]


def test_a_recommendation_counts_only_when_every_recommended_move_is_optimal():
    book = [Rule("corner", (("corner", True),))]
    r = evaluate(book, TTT, [_ttt(".........", 0)], [{0: 0, 2: 0, 6: 0, 8: -1, 4: 0}])
    assert r["covered"] == 1 and r["correct"] == 0


def test_move_values_read_from_json_with_text_keys_are_scored_as_moves():
    book = [Rule("win", (("wins", True),))]
    r = evaluate(book, TTT, [_ttt("xx.oo....", 0)], [{"2": 1, "5": 0}])
    assert r["correct"] == 1


def test_positions_and_values_must_pair_up():
    with pytest.raises(ValueError, match="same length"):
        evaluate(tactics(), TTT, [_ttt(".........", 0)], [])


def test_the_tactics_playbook_is_sound_at_every_tic_tac_toe_position():
    from harness.coverage import move_values, reachable_states

    raw, complete = reachable_states(TTT, exact=True, symmetry=False)
    assert complete
    raw = [s for s in raw if not TTT.is_terminal(s)]
    r = evaluate(tactics(), TTT, raw, [move_values(TTT, s) for s in raw])
    assert r["covered"] > 0 and r["correct"] == r["covered"]


def test_the_tactics_playbook_is_sound_on_solved_connect_4_positions():
    from harness.benchmark import sample_solvable_positions
    from harness.native_solver import move_values

    states = sample_solvable_positions(C4, 60, min_moves=24, seed=3)
    r = evaluate(tactics(), C4, states, [move_values(s) for s in states])
    assert r["covered"] > 0 and r["correct"] == r["covered"]


def test_newell_simon_is_the_eight_classic_rules_in_order():
    names = [r.name for r in newell_simon()]
    assert names == ["win", "block", "fork", "block fork", "force without a fork", "centre", "opposite corner",
                     "empty corner", "empty side"]
    rng = random.Random(0)
    assert decide(newell_simon(), TTT, TTT.initial_state(rng)) == (5, [4])


def _own_reference(book, game, root, player):
    seen, out, level = set(), [], [root]
    while level:
        nxt = []
        for s in level:
            k = game.state_key(s)
            if k in seen or game.is_terminal(s):
                continue
            seen.add(k)
            if game.current_player(s) == player:
                out.append(k)
                _i, moves = decide(book, game, s)
                moves = moves or game.legal_actions(s)
            else:
                moves = game.legal_actions(s)
            nxt.extend(game.step(s, a) for a in moves)
        level = nxt
    return sorted(out, key=str)


@pytest.mark.parametrize("player", [0, 1])
@pytest.mark.parametrize("book", [[Rule("centre", (("centre", True),)), Rule("corner", (("corner", True),))],
                                  [Rule("win", (("wins", True),))]], ids=["geometry", "tactics"])
def test_own_play_reaches_each_position_of_the_player_once_following_every_recommended_move(player, book):
    from harness.playbook import own_play_positions

    root = TTT.initial_state(random.Random(0))
    got = own_play_positions(book, TTT, root, player)
    assert sorted((TTT.state_key(s) for s in got), key=str) == _own_reference(book, TTT, root, player)
    assert len({TTT.state_key(s) for s in got}) == len(got)
    assert all(TTT.current_player(s) == player and not TTT.is_terminal(s) for s in got)


class _Cycle:
    """Four positions, players alternating by parity, every move leading back round the cycle — positions recur.
    It refuses to step more than 100 times, so a walk that never stops fails instead of hanging."""
    name = "cycle"

    def __init__(self):
        self.steps = 0

    def state_key(self, s):
        return s

    def is_terminal(self, s):
        return False

    def winner(self, s):
        return None

    def current_player(self, s):
        return s % 2

    def legal_actions(self, s):
        return [0, 1]

    def step(self, s, a, rng=None):
        self.steps += 1
        if self.steps > 100:
            raise RuntimeError("the walk did not stop")
        return (s + 1 + a) % 4


def test_own_play_does_not_walk_a_position_reached_again():
    from harness.playbook import own_play_positions

    assert own_play_positions([Rule("win", (("wins", True),))], _Cycle(), 0, 0) == [0, 2]


def test_a_forced_win_counts_every_reply_and_stops_at_its_ply_budget():
    from harness.playbook import forced_win

    edge = _ttt("xo.......", 0)
    assert forced_win(TTT, edge, 4, 5) and not forced_win(TTT, edge, 4, 3)
    assert forced_win(TTT, _ttt("xx.oo....", 0), 2, 1)
    assert not forced_win(TTT, _ttt("xoxxooox.", 0), 8, 5)
    assert not any(forced_win(TTT, _ttt(".........", 0), m, 5) for m in range(9))


def test_the_layered_predicates_on_tic_tac_toe():
    edge = _ttt("xo.......", 0)
    assert 4 in _holds("wins_in_5", TTT, edge)
    assert _holds("wins_in_5", TTT, _ttt(".........", 0)) == []
    after_corner = _ttt("x........", 1)
    assert 1 in _holds("gives_loss_in_6", TTT, after_corner) and 4 not in _holds("gives_loss_in_6", TTT, after_corner)
    assert _holds("gives_loss_in_4", TTT, after_corner) == []
    assert _holds("gives_loss_in_4", TTT, _ttt("x...o...x", 1)) == [2, 6]


def test_every_predicate_declares_how_many_plies_it_looks_ahead():
    from harness.playbook import LOOKAHEAD

    assert set(LOOKAHEAD) == set(PREDICATES)
    assert LOOKAHEAD["corner"] == 0 and LOOKAHEAD["wins"] == 1 and LOOKAHEAD["wins_in_5"] == 5
    assert LOOKAHEAD["gives_loss_in_6"] == 6
