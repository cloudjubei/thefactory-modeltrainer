"""Othello 8x8 — the first solver-free game, composed from harness/rules.py (§C.21 Increment 2)."""
import random

import pytest

from games.othello import CELLS, PASS, Othello, OthelloState
from harness.game import Game
from harness.registry import resolve_game


def _pos(cells: dict[int, int], to_move: int = 0) -> OthelloState:
    board = [0] * CELLS
    for c, v in cells.items():
        board[c] = v
    return OthelloState(board=tuple(board), to_move=to_move, done=False)


def test_registered_and_satisfies_the_protocol():
    g = resolve_game("othello")
    assert isinstance(g, Game)
    assert g.num_actions == 65 and g.board_shape == (8, 8) and g.num_players == 2


def test_opening_position_and_moves():
    g = Othello()
    s = g.initial_state(random.Random(0))
    assert sorted(i for i, v in enumerate(s.board) if v) == [27, 28, 35, 36]
    assert g.current_player(s) == 0
    assert g.legal_actions(s) == [19, 26, 37, 44]  # d3, c4, f5, e6
    assert PASS not in g.legal_actions(s)


def test_a_move_places_and_flips_and_hands_over():
    g = Othello()
    s = g.step(g.initial_state(random.Random(0)), 19)
    assert s.board[19] == 1 and s.board[27] == 1  # placed at d3, flipped d4
    assert s.to_move == 1 and not s.done
    assert g.winner(s) is None and g.returns(s) == [0.0, 0.0]


def test_illegal_moves_are_refused():
    g = Othello()
    s = g.initial_state(random.Random(0))
    with pytest.raises(ValueError):
        g.step(s, 0)  # flips nothing
    with pytest.raises(ValueError):
        g.step(s, 27)  # occupied
    with pytest.raises(ValueError):
        g.step(s, PASS)  # a flipping move exists
    with pytest.raises(ValueError):
        g.step(s, 99)


def test_pass_is_the_only_move_when_stuck_and_the_opponent_can_play():
    # White in the corner (uncapturable), black beside it: black cannot move, white can take (0,2).
    g = Othello()
    s = _pos({0: 2, 1: 1}, to_move=0)
    assert g.legal_actions(s) == [PASS]
    after = g.step(s, PASS)
    assert after.to_move == 1 and not after.done
    assert g.legal_actions(after) == [2]
    end = g.step(after, 2)
    assert end.done and end.board[:3] == (2, 2, 2)
    assert g.winner(end) == 1 and g.returns(end) == [-1.0, 1.0]


def test_game_ends_when_neither_side_can_move():
    g = Othello()
    s = _pos({0: 2, 63: 1}, to_move=0)  # two lone corners: nobody can capture anything
    assert g.is_terminal(s) is False  # `done` is set by step(); this hand-built state was never stepped
    stepped = g.step(_pos({0: 2, 1: 1, 2: 0}, to_move=1), 2)
    assert stepped.done


def test_draw_on_equal_discs():
    g = Othello()
    end = OthelloState(board=tuple([1] * 32 + [2] * 32), to_move=0, done=True)
    assert g.winner(end) is None and g.returns(end) == [0.0, 0.0]


def test_observation_is_mover_relative():
    g = Othello()
    s = g.initial_state(random.Random(0))
    o0, o1 = g.observation(s, 0), g.observation(s, 1)
    assert len(o0) == CELLS + 1
    assert o0[28] == 1.0 and o0[27] == -1.0 and o0[0] == 0.0
    assert o1[28] == -1.0 and o1[27] == 1.0
    assert o0[-1] == 1.0 and o1[-1] == 0.0


def test_random_playouts_terminate_with_a_majority_result():
    g = Othello()
    rng = random.Random(3)
    for _ in range(5):
        s = g.initial_state(rng)
        steps = 0
        while not g.is_terminal(s):
            s = g.step(s, rng.choice(g.legal_actions(s)), rng)
            steps += 1
            assert steps < 200
        x = sum(1 for v in s.board if v == 1)
        o = sum(1 for v in s.board if v == 2)
        w = g.winner(s)
        assert (w == 0) == (x > o) and (w == 1) == (o > x)
        assert g.legal_actions(s) == []


def test_heuristic_prefers_corners_then_flips_and_passes_when_stuck():
    g = Othello()
    rng = random.Random(0)
    s = g.initial_state(rng)
    assert g.heuristic_action(s, rng) in g.legal_actions(s)
    assert g.heuristic_action(_pos({0: 2, 1: 1}, to_move=0), rng) == PASS
    # a corner take beats a bigger flip elsewhere
    corner = _pos({1: 2, 2: 1, 3 * 8 + 3: 2, 3 * 8 + 4: 2, 3 * 8 + 5: 2, 3 * 8 + 6: 1, 3 * 8 + 2: 0}, to_move=0)
    assert g.heuristic_action(corner, rng) == 0


def test_render_and_labels():
    g = Othello()
    s = g.initial_state(random.Random(0))
    assert "X to move" in g.render(s)
    assert g.action_label(s, 19) == "d3" and g.action_label(s, PASS) == "pass"
    assert g.state_key(s) == (s.board, 0)


def test_no_symmetries_exposed_until_the_augmenter_can_use_them():
    # augment_examples permutes the LAST tensor axis (Connect-4 columns); 64-cell dihedral perms would corrupt.
    assert not hasattr(Othello(), "symmetries")


def test_generic_harness_agents_play_othello():
    # Othello's state has NO `winner` field. MctsAgent used to read `game.step(s, a).winner` — a Connect-4/TTT
    # state attribute, not the Protocol — and crashed here; the league uses MctsAgent as an opponent, so a full
    # solver-free run would have died mid-batch. Every generic agent must speak only the Game Protocol.
    from harness.agents import HeuristicAgent, MctsAgent, RandomAgent

    g = Othello()
    rng = random.Random(0)
    s = g.initial_state(rng)
    for agent in (RandomAgent(), HeuristicAgent(), MctsAgent(sims=16, solve_endgame=0, book=None)):
        a = agent.act(g, s, rng)
        assert a in g.legal_actions(s), type(agent).__name__
    # and a whole UCT-vs-UCT game runs to a result
    seats = {0: MctsAgent(sims=8, solve_endgame=0, book=None), 1: MctsAgent(sims=8, solve_endgame=0, book=None)}
    plies = 0
    while not g.is_terminal(s):
        s = g.step(s, seats[g.current_player(s)].act(g, s, rng), rng)
        plies += 1
        assert plies < 200
    assert g.returns(s) in ([1.0, -1.0], [-1.0, 1.0], [0.0, 0.0])
