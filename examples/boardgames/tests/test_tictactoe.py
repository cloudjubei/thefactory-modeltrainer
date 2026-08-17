import random

from games.tictactoe import TicTacToe
from harness.agents import RandomAgent
from harness.book import book_coverage, book_optimal_actions, build_book
from harness.bookagent import BookAgent
from harness.tablebase import Tablebase

game = TicTacToe()


def _play(seat_agents):
    s = game.initial_state(random.Random(0))
    while not game.is_terminal(s):
        s = game.step(s, seat_agents[game.current_player(s)].act(game, s, random.Random(0)))
    return s


def test_optimal_play_from_empty_board_is_a_draw():
    a = BookAgent(Tablebase(cap=1), "tictactoe", solve_endgame=9)  # exact solver from the empty board
    b = BookAgent(Tablebase(cap=1), "tictactoe", solve_endgame=9)
    assert game.winner(_play([a, b])) is None


def test_canonical_key_is_dihedral_invariant():
    corner0 = game.step(game.initial_state(), 0)  # X in one corner
    corner2 = game.step(game.initial_state(), 2)  # X in another corner (a reflection of the first)
    assert game.canonical_key(corner0) == game.canonical_key(corner2)


def test_symmetries_are_the_eight_dihedral_maps():
    perms = game.symmetries()
    assert len(perms) == 8
    assert all(sorted(p) == list(range(9)) for p in perms)  # each is a permutation of the 9 cells


def test_full_book_completes_and_plays_optimally():
    book = Tablebase(cap=10_000)
    stats = build_book(game, book, max_plies=9, max_positions=100_000)
    assert stats["solved"] > 0
    assert book_coverage(game, book, plies=9)["fraction"] == 1.0  # the ENTIRE reachable tree is exact

    s = game.initial_state(random.Random(0))
    agent = BookAgent(book, "tictactoe", solve_endgame=0)  # solver OFF → every move comes from the book
    while not game.is_terminal(s):
        assert book_optimal_actions(book, game, s) is not None  # the book covers every reachable position
        s = game.step(s, agent.act(game, s, random.Random(0)))
    assert game.winner(s) is None  # book-only optimal play draws


def test_book_agent_never_loses_from_either_seat():
    book = Tablebase(cap=10_000)
    build_book(game, book, max_plies=9, max_positions=100_000)
    agent = BookAgent(book, "tictactoe", solve_endgame=0)
    opp = RandomAgent()
    rng = random.Random(1)
    for seat in (0, 1):
        for _ in range(25):
            s = game.initial_state(rng)
            while not game.is_terminal(s):
                mover = game.current_player(s)
                s = game.step(s, agent.act(game, s, rng) if mover == seat else opp.act(game, s, rng))
            assert game.winner(s) != (1 - seat)  # the optimal agent never loses
