"""Direct tests for games/checkers.py — the rules that no earlier game exercised: mandatory capture,
multi-jump sub-turns, promotion, and loss-by-no-move."""
from __future__ import annotations

import random

import pytest

from games.checkers import CELLS, DIRS4, IDLE_LIMIT, N, Checkers, CheckersState, _settle
from harness.game import Game

G = Checkers()
RNG = random.Random(0)


def _pos(pieces: dict[int, int], to_move: int = 0, jumping: int | None = None, idle: int = 0) -> CheckersState:
    board = [0] * CELLS
    for cell, v in pieces.items():
        board[cell] = v
    return _settle(tuple(board), to_move, jumping, idle)


def _act(cell: int, d: tuple[int, int]) -> int:
    return DIRS4.index(d) * CELLS + cell


def _cell(row: int, col: int) -> int:
    return row * N + col


def test_checkers_satisfies_the_game_protocol():
    assert isinstance(G, Game)
    assert G.num_actions == 4 * CELLS and G.board_shape == (N, N)


def test_the_opening_position_is_twelve_men_each_on_dark_squares_only():
    s = G.initial_state(RNG)
    assert sum(1 for v in s.board if v == 1) == 12 and sum(1 for v in s.board if v == 2) == 12
    light = [c for c in range(CELLS) if (c // N + c % N) % 2 == 0]
    assert all(s.board[c] == 0 for c in light)
    assert not s.done and G.current_player(s) == 0


def test_the_opening_has_seven_quiet_moves_and_no_captures():
    s = G.initial_state(RNG)
    actions = G.legal_actions(s)
    assert len(actions) == 7
    assert all(G.action_label(s, a).find("x") == -1 for a in actions)


def test_men_move_only_forward():
    s = _pos({_cell(4, 3): 1}, to_move=0)
    dests = {(a % CELLS, DIRS4[a // CELLS][0]) for a in G.legal_actions(s)}
    assert {d for _c, d in dests} == {-1}, "player 0's men move toward row 0 only"
    s1 = _pos({_cell(4, 3): 2}, to_move=1)
    assert {DIRS4[a // CELLS][0] for a in G.legal_actions(s1)} == {1}


def test_a_king_moves_in_all_four_diagonals():
    s = _pos({_cell(4, 3): 3}, to_move=0)
    assert {DIRS4[a // CELLS] for a in G.legal_actions(s)} == set(DIRS4)


def test_capture_is_mandatory_so_quiet_moves_disappear():
    s = _pos({_cell(5, 2): 1, _cell(4, 3): 2}, to_move=0)
    actions = G.legal_actions(s)
    assert actions == [_act(_cell(5, 2), (-1, 1))]
    assert "x" in G.action_label(s, actions[0])


def test_a_capture_removes_the_jumped_piece_and_lands_beyond_it():
    s = _pos({_cell(5, 2): 1, _cell(4, 3): 2}, to_move=0)
    nxt = G.step(s, _act(_cell(5, 2), (-1, 1)), RNG)
    assert nxt.board[_cell(3, 4)] == 1 and nxt.board[_cell(4, 3)] == 0 and nxt.board[_cell(5, 2)] == 0


def test_a_multi_jump_is_ONE_turn_so_the_mover_does_not_change():
    """The sub-turn property: after the first hop the same player must jump again from the landing cell."""
    s = _pos({_cell(5, 2): 1, _cell(4, 3): 2, _cell(2, 5): 2}, to_move=0)
    mid = G.step(s, _act(_cell(5, 2), (-1, 1)), RNG)
    assert G.current_player(mid) == 0, "a multi-jump does not hand over the turn"
    assert mid.jumping == _cell(3, 4)
    assert G.legal_actions(mid) == [_act(_cell(3, 4), (-1, 1))], "only the jumping piece may act"
    end = G.step(mid, _act(_cell(3, 4), (-1, 1)), RNG)
    assert G.current_player(end) == 1 and end.jumping is None
    assert sum(1 for v in end.board if v == 2) == 0


def test_a_man_reaching_the_far_row_is_promoted():
    s = _pos({_cell(1, 2): 1}, to_move=0)
    nxt = G.step(s, _act(_cell(1, 2), (-1, 1)), RNG)
    assert nxt.board[_cell(0, 3)] == 3, "player 0's man becomes a king on row 0"
    s1 = _pos({_cell(6, 2): 2}, to_move=1)
    nxt1 = G.step(s1, _act(_cell(6, 2), (1, 1)), RNG)
    assert nxt1.board[_cell(7, 3)] == 4


def test_promotion_ends_the_turn_even_when_another_jump_is_available():
    """English draughts: the new king does NOT continue capturing on the turn it was crowned."""
    s = _pos({_cell(2, 1): 1, _cell(1, 2): 2, _cell(1, 4): 2}, to_move=0)
    nxt = G.step(s, _act(_cell(2, 1), (-1, 1)), RNG)
    assert nxt.board[_cell(0, 3)] == 3 and nxt.jumping is None
    assert G.current_player(nxt) == 1, "the turn ends at promotion"


def test_a_player_with_no_pieces_has_lost():
    s = _pos({_cell(4, 3): 1}, to_move=1)
    assert G.is_terminal(s) and G.winner(s) == 0 and G.returns(s) == [1.0, -1.0]


def test_a_player_who_cannot_move_loses_even_with_pieces_left():
    s = _pos({_cell(0, 1): 1, _cell(1, 0): 2, _cell(1, 2): 2, _cell(2, 1): 2}, to_move=0)
    assert G.legal_actions(s) == [] and G.is_terminal(s)
    assert G.winner(s) == 1


def test_the_idle_rule_draws():
    s = _pos({_cell(4, 3): 3, _cell(0, 7): 4}, to_move=0, idle=IDLE_LIMIT - 1)
    nxt = G.step(s, _act(_cell(4, 3), (-1, -1)), RNG)
    assert nxt.idle == IDLE_LIMIT and nxt.done and G.winner(nxt) is None and G.returns(nxt) == [0.0, 0.0]


def test_a_man_move_or_a_capture_resets_the_idle_counter():
    quiet_king = _pos({_cell(4, 3): 3, _cell(0, 7): 4}, to_move=0, idle=5)
    assert G.step(quiet_king, _act(_cell(4, 3), (-1, -1)), RNG).idle == 6
    man = _pos({_cell(4, 3): 1, _cell(0, 7): 4}, to_move=0, idle=5)
    assert G.step(man, _act(_cell(4, 3), (-1, -1)), RNG).idle == 0


def test_step_refuses_an_illegal_action():
    s = G.initial_state(RNG)
    with pytest.raises(ValueError):
        G.step(s, _act(_cell(5, 0), (1, 1)), RNG)  # backwards
    with pytest.raises(ValueError):
        G.step(s, 9999, RNG)


def test_step_refuses_to_play_on_after_the_game_is_over():
    s = _pos({_cell(4, 3): 1}, to_move=1)
    with pytest.raises(ValueError, match="over"):
        G.step(s, 0, RNG)


def test_observation_is_four_planes_from_the_movers_point_of_view():
    s = _pos({_cell(4, 3): 1, _cell(2, 5): 4}, to_move=0)
    obs = G.observation(s, 0)
    assert len(obs) == 4 * CELLS + 2
    assert obs[_cell(4, 3)] == 1.0, "own man in plane 0"
    assert obs[3 * CELLS + _cell(2, 5)] == 1.0, "opponent king in plane 3"
    flipped = G.observation(s, 1)
    assert flipped[2 * CELLS + _cell(4, 3)] == 1.0, "the same man is an OPPONENT man for player 1"
    assert flipped[CELLS + _cell(2, 5)] == 1.0


def test_the_valid_mask_is_the_dark_squares():
    assert len(G.valid_mask) == CELLS and sum(G.valid_mask) == CELLS / 2
    assert G.valid_mask[_cell(0, 1)] == 1.0 and G.valid_mask[_cell(0, 0)] == 0.0


def test_state_key_separates_positions_that_differ_only_by_the_draw_clock():
    a = _pos({_cell(4, 3): 3, _cell(0, 7): 4}, to_move=0, idle=3)
    b = _pos({_cell(4, 3): 3, _cell(0, 7): 4}, to_move=0, idle=70)
    assert G.state_key(a) != G.state_key(b)


def test_heuristic_and_render_work_on_a_real_position():
    s = G.initial_state(RNG)
    assert G.heuristic_action(s, RNG) in G.legal_actions(s)
    assert "to move" in G.render(s) and len(G.render(s).splitlines()) == N


def test_a_random_playout_always_terminates_cleanly():
    rng = random.Random(7)
    for seed in range(6):
        rng.seed(seed)
        s = G.initial_state(rng)
        plies = 0
        while not G.is_terminal(s):
            s = G.step(s, rng.choice(G.legal_actions(s)), rng)
            plies += 1
            assert plies < 4000
        assert G.returns(s) in ([1.0, -1.0], [-1.0, 1.0], [0.0, 0.0])


def test_generic_harness_agents_play_checkers():
    """The Protocol-leak regression (§C.21): a generic agent must reach a verdict through `game.winner(...)`,
    never through a state field that happens to exist on another game."""
    from harness.agents import MctsAgent, RandomAgent

    rng = random.Random(3)
    s = G.initial_state(rng)
    agents = [MctsAgent(sims=8, solve_endgame=0, book=None), RandomAgent()]
    for _ in range(30):
        if G.is_terminal(s):
            break
        s = G.step(s, agents[G.current_player(s)].act(G, s, rng), rng)
