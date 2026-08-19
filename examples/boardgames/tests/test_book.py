import random

from games.connect4 import COLS, ROWS, C4State, Connect4
from games.tictactoe import TicTacToe
from harness.agents import HeuristicAgent
from harness.book import (
    book_coverage,
    book_optimal_actions,
    book_value,
    build_book,
    estimate_position,
    evaluate,
    play_until_decided,
    position_value,
    run_build_book,
    sample_seeds,
)
from harness.bookagent import BookAgent
from harness.solver import move_values, optimal_columns
from harness.tablebase import ESTIMATE, PROVEN, Tablebase


def _c4_with_immediate_win():
    game = Connect4()
    s = game.initial_state(random.Random(0))
    for c in [0, 1, 0, 1, 0, 2]:  # p0 stacks three in column 0; it is p0's move with col 0 winning
        s = game.step(s, c)
    return game, s


def test_estimate_position_finds_the_immediate_win():
    game, s = _c4_with_immediate_win()
    value, mask, n = estimate_position(game, s, lambda: HeuristicAgent(), games=2)
    assert value == 1.0  # the winning move's child is TERMINAL → value +1 to the mover
    assert (mask >> 0) & 1  # column 0 (the win) is in the best-actions set


def test_evaluate_estimates_when_unprovable_and_proves_when_solvable():
    game = TicTacToe()
    est = lambda g, st: estimate_position(g, st, lambda: HeuristicAgent(), games=2)
    book = Tablebase(cap=10_000)
    s = game.initial_state(random.Random(0))
    belief = evaluate(game, s, book, est, max_exact_empty=0)  # empty book + no cheap solve → a BELIEF
    assert belief.status == ESTIMATE and belief.n > 0
    proof = evaluate(game, s, book, est, max_exact_empty=9)  # whole ttt tree is solvable → a PROOF
    assert proof.status == PROVEN and proof.value == 0.0 and proof.best_actions != 0  # optimal ttt = draw


def test_build_book_estimator_mode_proves_ttt_bottom_up_without_calling_the_estimator():
    # With an estimator wired but max_exact_empty=0, the WHOLE ttt tree still proves — bottom-up (deepest-first),
    # every position's children are already booked/terminal, so the FREE minimax proof fires and the estimator
    # is never reached. This is the eager upgrade end to end: proofs propagate up from the terminals.
    game = TicTacToe()

    def forbidden(g, st):
        raise AssertionError("estimator must not be reached — the tree proves bottom-up from terminals")

    book = Tablebase(cap=10_000)
    build_book(game, book, max_plies=9, max_positions=100_000, estimator=forbidden, max_exact_empty=0)
    assert book_coverage(game, book, plies=9)["fraction"] == 1.0
    e = book.entry(game.canonical_key(game.initial_state(random.Random(0))))
    assert e.status == PROVEN and e.value == 0.0 and e.best_actions != 0  # proven draw + stored optimal moves


def test_build_book_estimator_mode_estimates_the_unprovable_opening():
    # A shallow band whose children aren't booked and which is too deep to solve cheaply → ESTIMATE entries that
    # carry best_actions + a sample size, and stay INVISIBLE to exact consumers (proven_value None).
    game = Connect4()
    book = Tablebase(cap=100_000)
    est = lambda g, st: estimate_position(g, st, lambda: HeuristicAgent(), games=1)
    build_book(game, book, max_plies=3, min_plies=3, max_positions=20, estimator=est, max_exact_empty=0)
    entries = [book.entry(k) for k in book.keys()]
    assert entries and all(e.status == ESTIMATE and e.n > 0 for e in entries)
    assert all(book.proven_value(k) is None for k in book.keys())  # a belief is never trusted as a proof


def test_evaluate_proves_a_parent_from_its_booked_children_for_free():
    game = TicTacToe()
    never = lambda g, st: (0.0, 0, 0)  # the estimator must NOT be reached — the proof comes from children
    book = Tablebase(cap=10_000)
    s = game.initial_state(random.Random(0))
    for a in [0, 3, 1, 4]:  # X at 0,1 · O at 3,4 · X to move and can win by completing the top row (0,1,2)
        s = game.step(s, a)
    for a in game.legal_actions(s):  # book the non-terminal children as proven → parent proves by minimax
        child = game.step(s, a)
        if not game.is_terminal(child):
            book.put_proven(game.canonical_key(child), game.position_value(child))
    proof = evaluate(game, s, book, never, max_exact_empty=0)
    assert proof.status == PROVEN and proof.value == 1.0  # X has a forced win here
    assert (proof.best_actions >> 2) & 1  # the winning move (cell 2) is in the optimal set
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
