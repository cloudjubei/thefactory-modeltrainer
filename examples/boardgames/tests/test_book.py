import random

from games.connect4 import COLS, ROWS, C4State, Connect4
from games.tictactoe import TicTacToe
from harness.agents import HeuristicAgent
from harness.benchmark import sample_solvable_positions
from harness.book import (
    book_coverage,
    book_optimal_actions,
    book_value,
    build_book,
    estimate_position,
    evaluate,
    load_book,
    make_book_estimator,
    play_until_decided,
    position_value,
    principal_variation,
    run_build_book,
    sample_seeds,
)
from harness.bookagent import BookAgent
from harness.solver import move_values, optimal_columns


def test_run_bounded_caps_a_slow_call_and_passes_a_fast_one():
    import time as _t

    from harness.book import _run_bounded, _SolveTimeout

    timed_out = False
    try:
        _run_bounded(lambda: _t.sleep(2.0), 0.05)  # 2s work, 50ms cap
    except _SolveTimeout:
        timed_out = True
    assert timed_out
    assert _run_bounded(lambda: 42, 0.05) == 42  # a fast call returns normally, the timer is cleared


def test_solve_exact_bounded_defers_the_opening_wall():
    from harness.book import _solve_exact_bounded

    game = Connect4()
    empty = game.initial_state(random.Random(0))  # the empty board = the minutes-long opening solve
    assert _solve_exact_bounded(game, empty, Tablebase(cap=1), True, 0.1) is None  # 100ms cap → DEFERRED, not hung


def test_build_book_parallel_matches_sequential():
    # PROOF the parallel band solver is correct: solving a frontier across cores yields the byte-identical book to
    # the sequential build (exact solves are deterministic; booking stays in band order).
    game = Connect4()
    root = sample_solvable_positions(game, n=1, min_moves=20, seed=5)[0]
    ply = sum(1 for v in root.board if v != 0)
    seq = Tablebase(cap=1_000_000)
    build_book(game, seq, roots=[root], max_plies=ply + 2, min_plies=ply, max_positions=3000)
    par = Tablebase(cap=1_000_000)
    build_book(game, par, roots=[root], max_plies=ply + 2, min_plies=ply, max_positions=3000, workers=4)
    assert len(seq) > 0 and set(seq.keys()) == set(par.keys())
    for k in seq.keys():
        assert seq.proven_value(k) == par.proven_value(k)


def _state_at_exact_ply(game, ply, seed):
    r = random.Random(seed)
    s = game.initial_state(r)
    while sum(1 for v in s.board if v != 0) < ply:
        nxt = game.step(s, r.choice(game.legal_actions(s)))
        s = game.initial_state(r) if game.is_terminal(nxt) else nxt  # restart if a line ends before the target ply
    return s


def test_build_book_sequential_in_band_cap_matches_the_plain_build():
    # The in-app default routes exact builds through the banded path (a per-position cap is always set). With
    # workers=1 and a cap no solve reaches, that path must yield the byte-identical book to the original sequential
    # build — the mutation guard for the default in-app grind.
    game = Connect4()
    root = sample_solvable_positions(game, n=1, min_moves=20, seed=5)[0]
    ply = sum(1 for v in root.board if v != 0)
    plain = Tablebase(cap=1_000_000)
    build_book(game, plain, roots=[root], max_plies=ply + 2, min_plies=ply, max_positions=3000)
    capped = Tablebase(cap=1_000_000)
    build_book(game, capped, roots=[root], max_plies=ply + 2, min_plies=ply, max_positions=3000, max_position_seconds=5.0)
    assert len(plain) > 0 and set(plain.keys()) == set(capped.keys())
    for k in plain.keys():
        assert plain.proven_value(k) == capped.proven_value(k)


def test_build_book_respects_the_deadline_within_a_band():
    # gap (b): a time-capped accumulator must STOP near its deadline, not grind a whole band first. The ply-14 band
    # below a ply-12 root is ~40 cold solves (~120ms each ≈ 5s). A 0.4s deadline must book only a small prefix and
    # return promptly. Without a WITHIN-band deadline check the run solves every position before consulting the clock.
    import time as _t

    game = Connect4()
    root = _state_at_exact_ply(game, 12, seed=7)  # a deterministic ply-12 root → a single ply-14 frontier band
    book = Tablebase(cap=1_000_000)
    t0 = _t.perf_counter()
    st = build_book(game, book, roots=[root], max_plies=14, min_plies=14,
                    max_positions=10_000_000, workers=1, max_position_seconds=5.0, deadline=_t.perf_counter() + 0.4)
    elapsed = _t.perf_counter() - t0
    assert st["enumerated"] >= 30  # ply 12→14 frontier: a genuine multi-position band (~40 at ply 14)
    assert 1 <= st["solved"] <= 15  # the deadline cut the band to a small prefix (grinding it all books ~40 → >15)
    assert elapsed < 4.0  # returned promptly — did NOT solve the whole ~30s band before checking the clock


def test_connect4_solve_collapses_to_lookups_when_the_frontier_below_is_booked():
    # PROOF the amortization path makes opening solves FAST on-demand: once a position's children are booked, its
    # solve reads proven child values instead of searching below them — the search COLLAPSES. Measured by the
    # number of positions the solver had to search (transposition-table entries), a deterministic proxy for cost.
    # This is the no-dependency answer to the "158s wall": solve deep once (offline), then it's lookups.
    from harness import solver

    game = Connect4()
    # a ~19-stone position reached by FAST random play (no deep oracle line — that would dominate the test time);
    # it still searches enough nodes cold to make the collapse unmistakable.
    s = sample_solvable_positions(game, n=1, min_moves=19, seed=11)[0]
    ply = sum(1 for v in s.board if v != 0)

    solver._TT.clear()
    cold = move_values(s, weak=True)
    cold_searched = len(solver._TT)  # positions the cold solve had to search

    book = Tablebase(cap=1_000_000)  # book the children one ply below (the frontier)
    build_book(game, book, roots=[s], max_plies=ply + 1, min_plies=ply + 1, max_positions=50)

    solver._TT.clear()  # cold TT, so ONLY the book can shortcut the search
    warm = move_values(s, weak=True, book=book)
    warm_searched = len(solver._TT)

    assert warm == cold  # same exact answer
    assert cold_searched > 100 and warm_searched < cold_searched / 10  # booked frontier collapsed the search ≥10×


def test_connect4_bottom_up_book_is_correct_against_an_independent_solve():
    # PROOF the real-game book is CORRECT: build a midgame subtree bottom-up, then every booked position's
    # one-ply-lookahead optimal set must equal an INDEPENDENT from-scratch solve — the book plays exactly what a
    # fresh solver would, so a booked line is genuinely optimal, not merely self-consistent.
    game = Connect4()
    root = sample_solvable_positions(game, n=1, min_moves=18, seed=7)[0]  # a ~24-empty midgame position
    ply = sum(1 for v in root.board if v != 0)
    book = Tablebase(cap=1_000_000)
    build_book(game, book, roots=[root], max_plies=ply + 4, min_plies=ply, max_positions=8000)
    assert len(book) > 0

    acts = book_optimal_actions(book, game, root)  # the root's children were booked bottom-up
    assert acts is not None and set(acts) == set(optimal_columns(root))  # matches an independent full solve

    s = root  # every step of the book's own principal variation is optimal per an independent solve
    for a in principal_variation(book, game, root):
        bo = book_optimal_actions(book, game, s)
        if bo is not None:
            assert set(bo) == set(optimal_columns(s)) and a in optimal_columns(s)
        s = game.step(s, a)


def test_connect4_book_lifts_optimality_verification_of_a_real_line():
    # PROOF the loop works on the real game: booking a midgame subtree makes MORE of a strong agent's actual line
    # provably optimal — `optimality_verified_plies` climbs (0 with no book → the booked region afterwards).
    from harness.benchmark import exact_reference, optimality_trace
    from harness.solver import NearPerfectOracle

    game = Connect4()
    agent = NearPerfectOracle(depth=8, solve_endgame=22)
    act = lambda st: agent.act(game, st, random.Random(0))

    empty = Tablebase(cap=1_000_000)
    before = optimality_trace(game, act, exact_reference(game, book=empty, max_empty=0), max_plies=24)

    # book the subtree around the agent's OWN ply-16 midgame position, bottom-up
    s = game.initial_state(random.Random(0))
    for _ in range(16):
        s = game.step(s, act(s))
    book = Tablebase(cap=1_000_000)
    build_book(game, book, roots=[s], max_plies=sum(1 for v in s.board if v != 0) + 5, min_plies=16, max_positions=8000)

    after = optimality_trace(game, act, exact_reference(game, book=book, max_empty=0), max_plies=24)
    assert after["verified_plies"] > before["verified_plies"]  # coverage lifted how much of the line is verified
    assert after["first_blunder_ply"] is None  # and the near-perfect agent is optimal everywhere it's verified
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


def test_book_estimator_is_exact_when_its_search_reaches_ground_truth():
    # The DEFAULT estimator (design decision (a)) is book-aware MCTS-Solver self-play, NOT a blind rollout: where its
    # search reaches ground truth (here a solvable endgame) both seats play perfectly, so the bounded-search estimate
    # COLLAPSES onto the exact value + optimal set — the same grounding a booked child gives it in the deep opening.
    from harness.solver import optimal_columns

    game = Connect4()
    book = Tablebase(cap=10_000)
    late = sample_solvable_positions(game, n=1, min_moves=32, seed=3)[0]  # few empties → the solver plays it perfectly
    est = make_book_estimator(book, sims=1, solve_endgame=40, games=2)
    value, mask, n = est(game, late)
    assert value == float(position_value(game, late))  # the estimate is exactly the game-theoretic value
    assert mask == sum(1 << c for c in optimal_columns(late))  # and its best set is exactly the optimal set
    assert n > 0


def test_run_build_book_estimate_mode_grades_the_opening(tmp_path):
    from harness.tablebase import ESTIMATE as _ESTIMATE

    res = run_build_book(
        {"game": "connect4", "max_plies": 2, "min_plies": 0, "max_positions": 8,
         "estimate_games": 1, "estimate_sims": 2, "estimate_solve_endgame": 0, "max_exact_empty": 0},
        log=None, book_dir=str(tmp_path),
    )
    book = load_book("connect4", book_dir=str(tmp_path))
    assert res["added"] > 0
    assert any(book.entry(k).status == _ESTIMATE for k in book.keys())  # deep-opening positions got graded BELIEFS
    assert all(book.proven_value(k) is None for k in book.keys())  # a belief is never trusted as a proof


def test_run_build_book_honors_workers_and_the_per_position_cap(tmp_path):
    # The in-app grind's robustness knobs reach the solver through run_build_book: with a tiny per-position cap the
    # slow ply-10 solves are DEFERRED (booked later once cheap) instead of hanging the pass, and `workers` is accepted.
    res = run_build_book(
        {"game": "connect4", "seed_games": 4, "seed_plies": 10, "max_plies": 10, "min_plies": 10,
         "max_positions": 200, "workers": 2, "max_position_seconds": 0.05, "seed": 1},
        log=None, book_dir=str(tmp_path),
    )
    assert res["build"]["deferred"] >= 1  # ply-10 solves blow the 50ms cap → DEFERRED across the worker pool, no hang


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


def test_book_coverage_splits_proven_from_graded_estimates():
    # The honesty gauge counts graded ESTIMATEs as booked but reports `proven` apart, so the UI never labels a belief
    # "exact". A proof at the root + an estimate one ply down → booked 2, proven 1, provenFraction < fraction.
    book = Tablebase(cap=100_000)
    root = game.initial_state(random.Random(0))
    child = game.step(root, 3)
    book.put_proven(game.canonical_key(root), 0, best_actions=(1 << 3))
    book.put_estimate(game.canonical_key(child), 0.2, best_actions=(1 << 3), n=4)
    cov = book_coverage(game, book, plies=1)
    assert cov["booked"] == 2 and cov["proven"] == 1
    assert 0 < cov["provenFraction"] < cov["fraction"]


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
