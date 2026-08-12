"""Known-answer tests for the perfect-play Connect 4 oracle (harness/solver.py).

The game is solved, so these pin exact answers, not statistics. Two layers:
- INSTANT primitives — the bitboard win/threat/anticipation logic and the immediate-win reflex (no search).
- A brute-force CROSS-CHECK — on positions near the endgame (few plies left) a trivial game-tree minimax gives
  the exact value; the bitboard solver must agree move-for-move. This catches any solver bug without hand-built
  boards, and stays fast because solving is cheap once the board is nearly full.
The opening-theory tests (first player wins, centre is the unique optimal opening) are the slow, deep solves and
are marked `slow`.
"""
import os
import random

import pytest

import time

from games.connect4 import COLS, ROWS, C4State, Connect4
from harness import solver
from harness.solver import NearPerfectOracle, OracleAgent, move_values, optimal_columns

# Solving from the OPENING is a minutes-long deep search in pure Python (mid/late positions are instant — see
# the brute-force cross-check). The two headline-theory tests below are therefore opt-in: set
# BOARDGAMES_SOLVE_OPENING=1 to run them. Correctness is already guaranteed move-for-move by the cross-check.
_opening = pytest.mark.skipif(
    os.environ.get("BOARDGAMES_SOLVE_OPENING") != "1",
    reason="deep opening solve (minutes in pure Python); set BOARDGAMES_SOLVE_OPENING=1 to run",
)


def _play(cols):
    game = Connect4()
    state = game.initial_state(random.Random(0))
    for c in cols:
        state = game.step(state, c)
    return game, state


def _bits(cells):
    """A bitboard from (col, row) cells, in the solver's 7-bit-per-column layout."""
    b = 0
    for c, r in cells:
        b |= 1 << (c * solver._H1 + r)
    return b


# --- instant: the bitboard primitives -------------------------------------------------------------------

def test_alignment_detects_every_line_direction():
    assert solver._alignment(_bits([(0, 0), (1, 0), (2, 0), (3, 0)]))  # horizontal
    assert solver._alignment(_bits([(0, 0), (0, 1), (0, 2), (0, 3)]))  # vertical
    assert solver._alignment(_bits([(0, 0), (1, 1), (2, 2), (3, 3)]))  # diagonal /
    assert solver._alignment(_bits([(0, 3), (1, 2), (2, 1), (3, 0)]))  # diagonal \
    assert not solver._alignment(_bits([(0, 0), (1, 0), (2, 0)]))  # only three


def test_is_winning_move_agrees_with_the_game():
    game, state = _play([0, 0, 1, 1, 2, 2])  # P0 to move; col 3 completes 0-1-2-3
    position, mask, _ = solver.to_bitboard(state)
    for c in game.legal_actions(state):
        assert solver._is_winning_move(position, mask, c) == (game.step(state, c).winner == state.to_move)


def test_anticipation_forces_a_single_block():
    # P1 holds the bottom of columns 0,1,2 (threatens 3); P0 has no win → its only non-losing move is the block.
    _game, state = _play([6, 0, 5, 1, 6, 2])
    position, mask, _ = solver.to_bitboard(state)
    assert solver._possible_non_losing_moves(position, mask) == _bits([(3, 0)])


def test_anticipation_reports_a_double_threat_as_lost():
    # P1 holds the bottom of columns 1,2,3 → threatens BOTH 0 and 4; the defender cannot block both.
    _game, state = _play([6, 1, 5, 2, 6, 3])
    position, mask, _ = solver.to_bitboard(state)
    assert solver._possible_non_losing_moves(position, mask) == 0


# --- instant: the oracle reflex + terminal handling -----------------------------------------------------

def test_oracle_takes_the_immediate_win():
    game, state = _play([0, 0, 1, 1, 2, 2])
    assert game.step(state, 3).winner == 0
    assert OracleAgent().act(game, state, random.Random(0)) == 3  # fast path, no deep search


def test_move_values_empty_for_a_finished_game():
    _game, state = _play([0, 1, 0, 1, 0, 1, 0])  # P0 stacks column 0 to four → vertical win
    assert state.done and state.winner == 0
    assert move_values(state) == {}
    assert optimal_columns(state) == []


# --- fast + rigorous: cross-check the solver against brute force near the endgame -----------------------

def _bruteforce_value(game, state, cache):
    """Exact value from the mover's perspective with perfect play: +1 win / 0 draw / -1 loss."""
    key = (state.board, state.to_move)
    if key in cache:
        return cache[key]
    if state.done:
        v = 0 if state.winner is None else -1  # someone just moved and won → the mover to act has lost
    else:
        v = max(-_bruteforce_value(game, game.step(state, c), cache) for c in game.legal_actions(state))
    cache[key] = v
    return v


def _deep_position(game, plies_left, seed):
    """A random non-terminal position with ~`plies_left` empty cells (so brute force is cheap)."""
    target = ROWS * COLS - plies_left
    rng = random.Random(seed)
    for _ in range(500):
        s = game.initial_state(rng)
        for _ in range(target):
            legal = game.legal_actions(s)
            if not legal:
                break
            s = game.step(s, rng.choice(legal))
            if s.done:
                break
        if not s.done and len(game.legal_actions(s)) > 1:
            return s
    return None


@pytest.mark.parametrize("seed", range(8))
def test_solver_matches_bruteforce_move_for_move(seed):
    game = Connect4()
    state = _deep_position(game, plies_left=10, seed=seed)
    if state is None:
        pytest.skip("no deep non-terminal position sampled")
    solver._TT.clear()
    weak = move_values(state, weak=True)
    truth = {c: -_bruteforce_value(game, game.step(state, c), {}) for c in game.legal_actions(state)}
    assert weak == truth  # exact game-theoretic value of every legal move
    # the oracle plays a move that is optimal by brute force, and prefers to actually win (strong tie-break)
    best = max(truth.values())
    assert OracleAgent().act(game, state, random.Random(0)) in [c for c, v in truth.items() if v == best]


# --- the depth-limited near-perfect oracle (the fast opponent / spine rung) ------------------------------

def test_near_perfect_takes_the_win_and_blocks_the_loss():
    game, win = _play([0, 0, 1, 1, 2, 2])  # P0 can complete at 3
    assert NearPerfectOracle(depth=6).act(game, win, random.Random(0)) == 3
    game, block = _play([6, 0, 5, 1, 6, 2])  # P0 must block P1's threat at 3
    assert NearPerfectOracle(depth=6).act(game, block, random.Random(0)) == 3


@pytest.mark.parametrize("seed", range(8))
def test_near_perfect_is_exact_when_depth_covers_the_endgame(seed):
    # With ~8 plies left and depth 12, the heuristic leaf is never reached → play is provably OPTIMAL.
    game = Connect4()
    state = _deep_position(game, plies_left=8, seed=seed)
    if state is None:
        pytest.skip("no deep non-terminal position sampled")
    solver._TT.clear()
    assert NearPerfectOracle(depth=12).act(game, state, random.Random(0)) in optimal_columns(state)


def test_near_perfect_plays_the_opening_fast():
    game = Connect4()
    state = game.initial_state(random.Random(0))
    t0 = time.perf_counter()
    move = NearPerfectOracle(depth=9).act(game, state, random.Random(0))
    assert time.perf_counter() - t0 < 5.0  # milliseconds-to-seconds, unlike the exact solver's minutes
    assert move in range(COLS)


# --- slow: the headline theory (deep solves from the opening) -------------------------------------------

@pytest.mark.slow
@_opening
def test_first_player_wins_from_the_empty_board():
    game = Connect4()
    state = game.initial_state(random.Random(0))
    position, mask, moves = solver.to_bitboard(state)
    assert solver._solve(position, mask, moves, weak=True, tt=solver._TT) > 0


@pytest.mark.slow
@_opening
def test_centre_is_the_unique_optimal_opening():
    game = Connect4()
    state = game.initial_state(random.Random(0))
    assert optimal_columns(state) == [COLS // 2]  # column 3 only
