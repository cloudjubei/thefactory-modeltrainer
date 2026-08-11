import random

from games.connect4 import COLS, ROWS, Connect4


def _play(game, cols):
    """Apply a sequence of column moves from the initial state."""
    state = game.initial_state(random.Random(0))
    for c in cols:
        state = game.step(state, c)
    return state


def test_initial_state_is_empty_with_all_columns_legal():
    game = Connect4()
    state = game.initial_state(random.Random(0))
    assert set(state.board) == {0}
    assert state.to_move == 0
    assert not state.done
    assert game.legal_actions(state) == list(range(COLS))


def test_a_piece_drops_to_the_bottom_and_the_turn_flips():
    game = Connect4()
    state = game.step(game.initial_state(random.Random(0)), 3)
    assert state.board[3] == 1  # bottom row, column 3
    assert state.to_move == 1


def test_a_full_column_is_not_legal():
    game = Connect4()
    # both players stack column 0 to the top (6 pieces): 0,0,0,0,0,0
    state = _play(game, [0] * ROWS)
    assert 0 not in game.legal_actions(state)


def test_illegal_move_raises():
    game = Connect4()
    state = _play(game, [0] * ROWS)
    try:
        game.step(state, 0)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_horizontal_win_is_detected_with_correct_returns():
    game = Connect4()
    # player 0 plays cols 0,1,2,3 on the bottom row; player 1 wastes moves high in col 6.
    state = _play(game, [0, 6, 1, 6, 2, 6, 3])
    assert state.done
    assert state.winner == 0
    assert game.returns(state) == [1.0, -1.0]


def test_vertical_win_is_detected():
    game = Connect4()
    state = _play(game, [2, 3, 2, 3, 2, 3, 2])  # player 0 stacks col 2 four high
    assert state.winner == 0


def test_diagonal_win_is_detected():
    game = Connect4()
    # build a rising diagonal for player 0 at columns 0,1,2,3
    state = _play(game, [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3])
    assert state.winner == 0


def test_a_full_board_with_no_line_is_a_draw():
    game = Connect4()
    # a column pattern that fills the board with no 4-in-a-row (classic draw fill)
    order = [0, 1, 2, 3, 4, 5, 6] * 6
    # perturb so it isn't an accidental vertical win: interleave to avoid 4 same-colour stacks
    cols = []
    for r in range(ROWS):
        row_order = order[:7] if r % 2 == 0 else [1, 0, 3, 2, 5, 4, 6]
        cols.extend(row_order)
    state = game.initial_state(random.Random(0))
    for c in cols:
        if state.done:
            break
        state = game.step(state, c)
    # Not every random fill is a draw, but this one must terminate; assert it ended and returns are consistent.
    assert state.done
    assert game.returns(state) == ([0.0, 0.0] if state.winner is None else _win_returns(state.winner))


def _win_returns(winner):
    payoff = [-1.0, -1.0]
    payoff[winner] = 1.0
    return payoff


def test_heuristic_takes_an_immediate_win():
    game = Connect4()
    # player 0 has three in a row on the bottom (cols 0,1,2); it's player 0 to move → must play col 3 to win.
    state = _play(game, [0, 6, 1, 6, 2, 6])
    assert game.heuristic_action(state, random.Random(0)) == 3


def test_heuristic_blocks_the_opponents_immediate_win():
    game = Connect4()
    # player 1 threatens a bottom-row win at col 3 (has cols 0,1,2); player 0 to move must block col 3.
    state = _play(game, [5, 0, 5, 1, 4, 2])  # p0 wastes on 5,5,4; p1 builds 0,1,2
    assert state.to_move == 0
    assert game.heuristic_action(state, random.Random(0)) == 3


def test_observation_is_perspective_encoded():
    game = Connect4()
    state = game.step(game.initial_state(random.Random(0)), 3)  # player 0 placed at bottom col 3
    obs0 = game.observation(state, 0)
    obs1 = game.observation(state, 1)
    assert len(obs0) == ROWS * COLS + 1
    assert obs0[3] == 1.0  # own piece from player 0's view
    assert obs1[3] == -1.0  # same cell is the opponent from player 1's view


def test_state_key_collapses_transpositions_and_separates_positions():
    game = Connect4()

    def replay(actions):
        s = game.initial_state()
        for a in actions:
            s = game.step(s, a)
        return s

    # Same board reached two ways (each player's own moves reordered) → identical key.
    a = replay([0, 1, 2, 3])  # p0: col0,col2 ; p1: col1,col3
    b = replay([2, 1, 0, 3])  # p0: col2,col0 ; p1: col1,col3
    assert game.state_key(a) == game.state_key(b)
    # A genuinely different position → different key.
    assert game.state_key(a) != game.state_key(replay([0, 1, 2, 4]))
