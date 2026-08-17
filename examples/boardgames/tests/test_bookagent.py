import random

from games.connect4 import Connect4
from harness.book import build_book
from harness.bookagent import BookAgent
from harness.solver import optimal_columns
from harness.tablebase import Tablebase

game = Connect4()


def _deep_state(target_plies: int):
    s = game.initial_state(random.Random(0))
    plies = 0
    while plies < target_plies:
        placed = False
        for c in game.legal_actions(s):
            nxt = game.step(s, c)
            if not game.is_terminal(nxt):
                s, plies, placed = nxt, plies + 1, True
                break
        if not placed:
            break
    return s


def test_book_agent_takes_immediate_win():
    s = game.initial_state(random.Random(0))
    for c in [3, 0, 3, 1, 3, 2]:  # player 0 stacks col 3; back on move 0 with three in a column
        s = game.step(s, c)
    agent = BookAgent(Tablebase(cap=100), "connect4")
    assert agent.act(game, s, random.Random(0)) == 3


def test_book_agent_plays_book_move_when_covered():
    book = Tablebase(cap=100_000)
    s = _deep_state(30)
    build_book(game, book, roots=[s], max_plies=42, max_positions=100_000)
    agent = BookAgent(book, "connect4", solve_endgame=0)  # solver off → the move must come from the book
    assert agent.act(game, s, random.Random(0)) in optimal_columns(s)


def test_book_agent_uses_endgame_solver_when_book_thin():
    s = _deep_state(36)  # few empties → the exact endgame solver applies
    agent = BookAgent(Tablebase(cap=100), "connect4", solve_endgame=22)
    assert agent.act(game, s, random.Random(0)) in optimal_columns(s)


def test_book_agent_falls_back_and_stays_legal():
    s = game.initial_state(random.Random(0))  # empty book, opening → strong depth-limited fallback
    agent = BookAgent(Tablebase(cap=100), "connect4", solve_endgame=22, depth=4)
    assert agent.act(game, s, random.Random(0)) in game.legal_actions(s)
