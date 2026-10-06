"""Direct tests for games/kalah.py — Kalah(m, n) under the rules of Irving, Donkers & Uiterwijk (2000), "Solving
Kalah", ICGA Journal 23(3): sow anticlockwise including the mover's store and skipping the opponent's; a last counter
in the mover's store moves again; a last counter in an empty hole of the mover captures it and the opposite hole (even
when that is empty); a move that leaves one side empty ends the game and the other player takes the rest. The paper's
tables are the outside oracle: its perfect games (Table 9) replay to their stated margins, its solved values (Tables
9 and 10) are what the solver returns."""
from __future__ import annotations

import random

import pytest

from games.kalah import Kalah, KalahState
from harness.kalah_paper import PERFECT_GAMES, TABLE_10


def _state(own, opp, stores=(0, 0), to_move=0):
    pits = tuple(own) + tuple(opp) if to_move == 0 else tuple(opp) + tuple(own)
    return KalahState(pits=pits, stores=tuple(stores), to_move=to_move, done=False)


def _replay(game, line):
    s = game.initial_state(random.Random(0))
    for turn in line.split("-"):
        mover = game.current_player(s)
        for i, hole in enumerate(turn):
            assert not game.is_terminal(s)
            assert game.current_player(s) == mover
            s = game.step(s, int(hole))
            last = i == len(turn) - 1
            assert game.is_terminal(s) or (game.current_player(s) == mover) != last
    return s


@pytest.mark.parametrize("shape", sorted(set(PERFECT_GAMES) - {(4, 6)}))
def test_the_paper_s_perfect_games_replay_to_their_stated_margins(shape):
    game = Kalah(*shape)
    margin, line = PERFECT_GAMES[shape]
    end = _replay(game, line)
    assert game.is_terminal(end) and end.stores[0] - end.stores[1] == margin
    assert sum(end.stores) == 2 * shape[0] * shape[1] and sum(end.pits) == 0


def test_the_paper_s_kalah_4_6_line_cannot_be_played_past_north_s_tenth_turn():
    game = Kalah(4, 6)
    line = PERFECT_GAMES[(4, 6)][1]
    turns = line.split("-")
    s = _replay(game, "-".join(turns[:9]))
    s = game.step(s, 2)
    assert s.to_move == 1 and s.pits[4] == 5
    s = game.step(s, 0)
    assert s.to_move == 0 and not game.is_terminal(s) and turns[9] == "201"


def test_sowing_includes_the_mover_s_store_and_skips_the_opponent_s():
    game = Kalah(3, 1)
    s = game.step(_state((1, 1, 6), (1, 1, 1)), 2)
    assert s.stores == (1, 0) and s.pits == (2, 2, 0, 2, 2, 2) and s.to_move == 1
    s = game.step(_state((1, 1, 6), (1, 1, 1), to_move=1), 2)
    assert s.stores == (0, 1) and s.pits == (2, 2, 2, 2, 2, 0) and s.to_move == 0


def test_a_last_counter_in_the_mover_s_store_moves_again():
    game = Kalah(3, 2)
    s = game.step(game.initial_state(), 1)
    assert s.pits == (2, 0, 3, 2, 2, 2) and s.stores == (1, 0) and s.to_move == 0


def test_a_last_counter_in_an_empty_own_hole_captures_it_and_the_opposite_hole():
    game = Kalah(3, 1)
    s = game.step(_state((1, 0, 1), (4, 2, 3), stores=(1, 1)), 0)
    assert s.pits == (0, 0, 1, 4, 0, 3) and s.stores == (4, 1) and s.to_move == 1


def test_the_capture_is_taken_even_when_the_opposite_hole_is_empty():
    game = Kalah(3, 1)
    s = game.step(_state((1, 0, 1), (4, 0, 3)), 0)
    assert s.pits == (0, 0, 1, 4, 0, 3) and s.stores == (1, 0) and s.to_move == 1


def test_a_last_counter_in_an_empty_hole_of_the_opponent_captures_nothing():
    game = Kalah(2, 1)
    s = game.step(_state((1, 2), (0, 5)), 1)
    assert s.pits == (1, 0, 1, 5) and s.stores == (1, 0) and s.to_move == 1


def test_a_lap_that_ends_in_the_emptied_start_hole_captures():
    game = Kalah(2, 1)
    s = game.step(_state((5, 1), (2, 3)), 0)
    assert s.pits == (0, 2, 3, 0) and s.stores == (6, 0) and s.to_move == 1


def test_a_move_that_empties_either_side_ends_the_game_and_the_other_player_takes_the_rest():
    game = Kalah(2, 1)
    s = game.step(_state((0, 1), (3, 2), stores=(3, 0)), 1)
    assert game.is_terminal(s) and s.stores == (4, 5) and s.pits == (0, 0, 0, 0)
    assert game.winner(s) == 1 and game.returns(s) == [-1.0, 1.0] and game.legal_actions(s) == []
    s = Kalah(3, 1).step(_state((1, 0, 2), (0, 3, 0), stores=(2, 4)), 0)
    assert s.stores == (8, 4) and s.pits == (0,) * 6 and Kalah(3, 1).winner(s) == 0


def test_an_equal_split_is_a_draw():
    game = Kalah(1, 1)
    s = game.step(game.initial_state(), 0)
    assert game.is_terminal(s) and s.stores == (1, 1) and game.winner(s) is None and game.returns(s) == [0.0, 0.0]


@pytest.mark.parametrize("hole", [-1, 3, 1])
def test_an_empty_or_missing_hole_is_not_a_move(hole):
    with pytest.raises(ValueError, match="hole"):
        Kalah(3, 1).step(_state((1, 0, 1), (1, 1, 1)), hole)


def test_only_non_empty_holes_are_legal_and_holes_are_numbered_from_the_mover_s_side():
    game = Kalah(4, 1)
    pits = (1, 0, 2, 0, 0, 5, 0, 0)
    assert game.legal_actions(KalahState(pits=pits, stores=(0, 0), to_move=0, done=False)) == [0, 2]
    assert game.legal_actions(KalahState(pits=pits, stores=(0, 0), to_move=1, done=False)) == [1]


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (3, 2), (4, 4), (6, 4), (5, 6)])
def test_every_move_raises_the_captured_count_or_brings_counters_nearer_their_owner_s_store(shape):
    game = Kalah(*shape)
    rng = random.Random(sum(shape))
    for _ in range(40):
        s = game.initial_state(rng)
        while not game.is_terminal(s):
            before = game.progress(s)
            s = game.step(s, rng.choice(game.legal_actions(s)))
            assert sum(s.pits) + sum(s.stores) == 2 * shape[0] * shape[1]
            assert game.progress(s) > before


def test_progress_is_the_captured_count_then_the_negated_distance_to_the_owners_stores():
    game = Kalah(3, 1)
    assert game.progress(_state((1, 0, 2), (0, 0, 4), stores=(2, 3))) == (5, -(3 + 2 + 4))


@pytest.mark.parametrize("m,n", [(m, n) for m in (1, 2, 3) for n in range(1, 7) if m * n <= 12] + [(4, 1), (5, 1)])
def test_the_solver_reproduces_the_paper_s_win_draw_loss_table(m, n):
    game = Kalah(m, n)
    v = game.position_value(game.initial_state())
    assert "LDW"[v + 1] == TABLE_10[m][n - 1]


@pytest.mark.parametrize("shape", [(4, 1), (4, 2), (5, 1), (6, 1)])
def test_the_solver_reproduces_the_paper_s_margins_and_its_perfect_games_are_optimal(shape):
    game = Kalah(*shape)
    margin, line = PERFECT_GAMES[shape]
    s = game.initial_state()
    assert game.margin(s) == margin
    for turn in line.split("-"):
        for hole in turn:
            assert int(hole) in game.optimal_actions(s)
            s = game.step(s, int(hole))


def test_the_margin_counts_what_is_already_stored_and_is_read_by_the_side_to_move():
    game = Kalah(2, 1)
    s = _state((0, 1), (3, 2), stores=(3, 0))
    assert game.margin(s) == -1 and game.position_value(s) == -1 and game.move_margin(s, 1) == -1
    flipped = _state((0, 1), (3, 2), stores=(0, 3), to_move=1)
    assert game.margin(flipped) == -1 and game.canonical_key(flipped) == game.canonical_key(s)
    assert game.canonical_key(_state((1, 0), (3, 2), stores=(3, 0))) != game.canonical_key(s)
    assert game.canonical_key(_state((0, 1), (3, 2), stores=(2, 1))) != game.canonical_key(s)


def test_a_finished_game_s_margin_is_its_store_difference_for_the_side_that_would_move():
    game = Kalah(2, 1)
    end = game.step(_state((0, 1), (3, 2), stores=(3, 0)), 1)
    assert game.margin(end) == -1 and game.position_value(end) == -1 and game.optimal_actions(end) == []


def test_optimal_actions_are_every_move_that_keeps_the_best_margin():
    game = Kalah(2, 1)
    s = _state((1, 1), (1, 1))
    values = {a: game.move_margin(s, a) for a in game.legal_actions(s)}
    assert game.optimal_actions(s) == sorted(a for a, v in values.items() if v == max(values.values()))
    assert game.move_margin(s, 1) == 2


def test_the_observation_is_one_plane_of_counter_shares_from_the_viewer_s_side_with_opposite_holes_aligned():
    game = Kalah(3, 2)
    s = _state((1, 0, 2), (4, 0, 3), stores=(1, 1))
    assert game.board_shape == (2, 4) and game.input_planes == 1 and game.num_actions == 3
    assert game.observation(s, 0) == [v / 12 for v in (1, 0, 2, 1, 3, 0, 4, 1)] + [1.0]
    assert game.observation(s, 1) == [v / 12 for v in (4, 0, 3, 1, 2, 0, 1, 1)] + [0.0]


def test_the_heuristic_takes_an_extra_move_then_the_largest_capture_then_the_hole_nearest_the_store():
    game = Kalah(3, 1)
    rng = random.Random(0)
    assert game.heuristic_action(_state((3, 1, 0), (1, 1, 1)), rng) == 0
    assert game.heuristic_action(_state((1, 0, 3), (1, 5, 1)), rng) == 0
    assert Kalah(4, 1).heuristic_action(_state((1, 0, 1, 0), (1, 1, 7, 1)), rng) == 0
    assert game.heuristic_action(_state((2, 3, 2), (1, 1, 1)), rng) == 2


def test_identity_is_the_only_declared_symmetry_and_ply_is_the_captured_count():
    game = Kalah(3, 2)
    assert game.symmetries() == [(list(range(8)), [0, 1, 2])]
    assert game.ply(_state((1, 0, 2), (4, 0, 3), stores=(1, 1))) == 2


def test_render_shows_both_rows_stores_and_status():
    game = Kalah(2, 1)
    text = game.render(game.initial_state())
    assert "S to move" in text and text.count("1") >= 4
    assert game.action_label(game.initial_state(), 1) == "hole 1"
    assert game.state_key(game.initial_state()) == ((1, 1, 1, 1), (0, 0), 0)


def test_a_variant_needs_at_least_one_hole_and_one_counter():
    with pytest.raises(ValueError, match="holes"):
        Kalah(0, 4)
    with pytest.raises(ValueError, match="counters"):
        Kalah(4, 0)


def test_the_standard_shape_is_named_kalah_and_every_other_shape_carries_its_size():
    assert Kalah().name == "kalah" and (Kalah().holes, Kalah().counters) == (6, 4)
    assert Kalah(4, 3).name == "kalah4x3" and Kalah(6, 5).name == "kalah6x5"
