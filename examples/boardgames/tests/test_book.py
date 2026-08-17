import random

from games.connect4 import COLS, ROWS, C4State, Connect4
from harness.book import (
    book_coverage,
    book_optimal_actions,
    book_value,
    build_book,
    play_until_decided,
    position_value,
    run_build_book,
    sample_seeds,
)
from harness.bookagent import BookAgent
from harness.solver import move_values, optimal_columns
from harness.tablebase import Tablebase

game = Connect4()


def _deep_state(target_plies: int) -> C4State:
    """A non-terminal position with ~`target_plies` stones down (so exact solves are instant)."""
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


def _mirror_state(s: C4State) -> C4State:
    m = [0] * (ROWS * COLS)
    for r in range(ROWS):
        for c in range(COLS):
            m[r * COLS + (COLS - 1 - c)] = s.board[r * COLS + c]
    return C4State(board=tuple(m), to_move=s.to_move, winner=s.winner, done=s.done)


def test_position_value_matches_solver_sign():
    s = _deep_state(34)
    assert position_value(game, s) == max(move_values(s, weak=True).values())


def test_book_one_ply_lookahead_is_optimal():
    book = Tablebase(cap=100_000)
    s = _deep_state(32)
    stats = build_book(game, book, roots=[s], max_plies=42, max_positions=100_000)
    assert stats["solved"] > 0
    acts = book_optimal_actions(book, game, s)
    assert acts is not None
    assert set(acts) == set(optimal_columns(s))


def test_book_is_symmetry_canonical():
    book = Tablebase(cap=100_000)
    s = _deep_state(30)
    build_book(game, book, roots=[s], max_plies=42, max_positions=100_000)
    child = game.step(s, game.legal_actions(s)[0])
    if not game.is_terminal(child):
        assert book_value(book, game, child) == book_value(book, game, _mirror_state(child))


def test_build_book_is_resumable_and_bounded():
    book = Tablebase(cap=100_000)
    s = _deep_state(30)
    first = build_book(game, book, roots=[s], max_plies=42, max_positions=5)
    assert first["solved"] == 5
    n_after_first = len(book)
    second = build_book(game, book, roots=[s], max_plies=42, max_positions=100_000)
    assert second["skipped"] >= 5  # already-booked positions are skipped, not re-solved
    assert len(book) > n_after_first


def test_incomplete_book_returns_none_for_one_ply():
    book = Tablebase(cap=100_000)
    s = _deep_state(20)  # nothing booked below it → cannot guarantee optimal from the book alone
    assert book_optimal_actions(book, game, s) is None


def test_persist_roundtrip(tmp_path):
    book = Tablebase(cap=100_000)
    s = _deep_state(34)
    build_book(game, book, roots=[s], max_plies=42, max_positions=100_000)
    p = str(tmp_path / "connect4")
    book.save(p)
    reloaded = Tablebase.load(p)
    assert len(reloaded) == len(book)
    assert book_value(reloaded, game, s) == book_value(book, game, s)


def test_book_coverage_reports_booked_fraction():
    book = Tablebase(cap=100_000)
    cov = book_coverage(game, book, plies=2)
    assert cov["booked"] == 0 and cov["reachable"] >= 1 and cov["fraction"] == 0.0


def test_min_plies_bands_the_work():
    s = _deep_state(34)
    full = Tablebase(cap=100_000)
    n_full = build_book(game, full, roots=[s], max_plies=42, min_plies=0, max_positions=100_000)["solved"]
    banded = Tablebase(cap=100_000)
    n_band = build_book(game, banded, roots=[s], max_plies=42, min_plies=38, max_positions=100_000)["solved"]
    assert n_full > 0 and n_band < n_full  # the ply floor stores only the deep band, a strict subset


def test_sample_seeds_are_nonterminal_at_depth():
    seeds = sample_seeds(game, 5, 20, seed=1)
    assert len(seeds) == 5
    for s in seeds:
        assert not game.is_terminal(s)
        assert sum(1 for v in s.board if v != 0) == 20


def test_seed_mode_build_grows_coverage_fast(tmp_path):
    # midgame seeds have small, cheap subtrees → a real book materialises without any opening solve
    res = run_build_book(
        {"game": "connect4", "seed_games": 8, "seed_plies": 30, "max_plies": 42, "max_positions": 5000},
        log=None, book_dir=str(tmp_path),
    )
    assert res["added"] > 0 and res["total"] == res["added"]


def test_play_until_decided_ends_early_and_is_sound():
    book = Tablebase(cap=100_000)
    s = _deep_state(30)
    build_book(game, book, roots=[s], max_plies=42, max_positions=100_000)  # s and its subtree booked
    a = BookAgent(book, "connect4", solve_endgame=0)
    winner, plies, early = play_until_decided(game, [a, a], random.Random(0), book=book, start=s)
    assert early and plies == 0  # s itself is booked → decided before a move is made
    # soundness: playing it out fully with the same optimal agents yields the SAME winner
    full = BookAgent(book, "connect4", solve_endgame=22)
    w_full, _, early2 = play_until_decided(game, [full, full], random.Random(0), book=None, start=s)
    assert not early2 and w_full == winner


def test_run_build_book_plumbing(tmp_path):
    # deadline 0 → no expensive opening solves; exercises warm/build/persist/coverage glue into a temp dir.
    res = run_build_book(
        {"game": "connect4", "max_plies": 2, "deadline_seconds": 0}, log=None, book_dir=str(tmp_path)
    )
    assert res["game"] == "connect4"
    assert res["added"] == 0 and res["total"] == 0 and res["coverage"]["fraction"] == 0.0
    assert (tmp_path / "connect4.npz").is_file()
