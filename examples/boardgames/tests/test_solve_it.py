"""SOLVE-IT (docs/implementation-plan.md §C.5) — the honest measurement + winning-strategy grind + book/net
verification that turns a SOLVED GAME into a MODEL that plays perfectly. The truth spine: nothing here claims
"optimal" unless it is measured against the EXACT oracle. The full pipeline closes end-to-end on tic-tac-toe
(a game we fully solve) and on a Connect 4 forced-win subtree; the from-opening Connect 4 grind is the same
tool, bounded + resumable, run in-app.
"""
import random

from games.connect4 import Connect4
from games.tictactoe import TicTacToe
from harness.agents import ExactOptimalAgent, RandomAgent
from harness.benchmark import optimality_ladder, p1_conversion, verify_solved
from harness.book import (
    _key,
    book_optimal_actions,
    build_book,
    position_value,
    prove_winning_strategy,
    winning_strategy_coverage,
)
from harness.bookagent import BookAgent
from harness.neural import book_distill_examples
from harness.solver import OracleAgent, optimal_columns
from harness.tablebase import PROVEN, Tablebase


# --- fixtures -------------------------------------------------------------------------------------------------
def _c4_mover_win_root(empties: int, seed: int, require_deep: bool = False) -> "Connect4.C4State":
    """A non-terminal Connect 4 position with exactly `empties` empty cells where the side to move has a proven
    forced WIN (`position_value > 0`) — a small, tractable winning-strategy tree the grind can complete. With
    `require_deep`, exclude positions that already have an IMMEDIATE winning move, so the win is forced over
    several plies (real opponent branching) and the directed prover's recursion is genuinely exercised."""
    game = Connect4()
    target_stones = 42 - empties
    rng = random.Random(seed)
    for _ in range(40000):
        s = game.initial_state(rng)
        while sum(1 for v in s.board if v != 0) < target_stones:
            nxt = game.step(s, rng.choice(game.legal_actions(s)))
            if game.is_terminal(nxt):
                s = None
                break
            s = nxt
        if s is None or game.is_terminal(s):
            continue
        if sum(1 for v in s.board if v == 0) != empties or position_value(game, s) <= 0:
            continue
        if require_deep and any(game.step(s, a).winner == s.to_move for a in game.legal_actions(s)):
            continue  # an immediate win → a trivial one-node tree; keep looking for a genuinely forced win
        return s
    raise AssertionError("no mover-win root found — widen the search")


# --- M0: HONEST MEASUREMENT ----------------------------------------------------------------------------------
def test_p1_conversion_optimal_converts_a_forced_win_and_random_does_not():
    game = Connect4()
    root = _c4_mover_win_root(empties=14, seed=1, require_deep=True)
    won = p1_conversion(game, lambda: OracleAgent(), lambda: OracleAgent(), games=2, start=root)
    assert won["rate"] == 1.0  # perfect play converts the forced win vs perfect defence
    lost = p1_conversion(game, lambda: RandomAgent(), lambda: OracleAgent(), games=6, start=root, seed=3)
    assert lost["rate"] < 1.0  # a random first player throws the win away


def test_p1_conversion_counts_draws_as_not_lost_for_a_drawn_game():
    game = TicTacToe()
    res = p1_conversion(game, lambda: ExactOptimalAgent(), lambda: ExactOptimalAgent(), games=2)
    assert res["rate"] == 0.0 and res["not_lost_rate"] == 1.0  # optimal ttt is a draw: never wins, never loses


def test_verify_solved_is_gated_on_the_exact_oracle_not_a_proxy():
    game = Connect4()
    root = _c4_mover_win_root(empties=12, seed=2)
    solved = verify_solved(game, lambda: OracleAgent(), games=2, start=root)
    assert solved["solved"] is True  # perfect play IS solved from a forced-win root
    # a random model may beat a weak depth-limited proxy, but verify_solved plays it vs the EXACT oracle → not solved
    weak = verify_solved(game, lambda: RandomAgent(), games=6, start=root, seed=7)
    assert weak["solved"] is False


def test_optimality_ladder_reports_the_frontier_and_only_exact_proves_solved():
    game = Connect4()
    root = _c4_mover_win_root(empties=12, seed=4)
    lad = optimality_ladder(game, lambda: OracleAgent(), games=2, depths=(6, 8), start=root)
    labels = [r["label"] for r in lad["rungs"]]
    assert labels == ["depth-6", "depth-8", "exact"]  # weakest→strongest, exact on top
    assert lad["frontier"] == "exact" and lad["solved"] is True  # perfect play clears every rung incl. exact
    weak = optimality_ladder(game, lambda: RandomAgent(), games=4, depths=(6, 8), start=root, seed=5)
    assert weak["solved"] is False  # a random model never clears the exact rung


# --- M1: THE WINNING-STRATEGY GRIND --------------------------------------------------------------------------
def test_prove_winning_strategy_completes_tic_tac_toe():
    # The whole ttt winning-strategy tree (a drawn game: never lose) proves bottom-up → 100% coverage, and every
    # booked node's optimal set equals an independent minimax — a lookup-perfect player falls straight out.
    game = TicTacToe()
    book = Tablebase(cap=100_000)
    stats = prove_winning_strategy(game, book, max_plies=9, max_exact_empty=0)
    assert stats["proven"] > 0 and stats["deferred"] == 0
    cov = winning_strategy_coverage(game, book, max_plies=9, max_exact_empty=0)
    assert cov["complete"] is True and cov["provenFraction"] == 1.0 and cov["root_proven"] is True
    root = game.initial_state(random.Random(0))
    played = book_optimal_actions(book, game, root)  # a pruned strategy knows ONE drawing line, and it IS optimal
    assert played and set(played).issubset(set(game._optimal_actions(root)))


def test_prove_winning_strategy_completes_a_connect4_forced_win_subtree():
    # A genuinely FORCED win (no immediate winning move) → the directed prover must descend our optimal line and
    # answer every opponent reply, over several plies, before the win is proven — a real winning-strategy tree.
    game = Connect4()
    root = _c4_mover_win_root(empties=14, seed=1, require_deep=True)
    book = Tablebase(cap=1_000_000)
    stats = prove_winning_strategy(game, book, root=root, strategist=root.to_move,
                                   max_plies=42, max_exact_empty=8)
    assert stats["proven"] > 3 and stats["deferred"] == 0  # a multi-node tree, fully proven (not a 1-move win)
    cov = winning_strategy_coverage(game, book, root=root, strategist=root.to_move,
                                    max_plies=42, max_exact_empty=8)
    assert cov["root_proven"] is True and cov["root_value"] > 0  # the forced win is proven into the book
    assert cov["complete"] is True and cov["nodes"] > 3  # the whole (non-trivial) strategy tree is booked
    assert set(book_optimal_actions(book, game, root)) == set(optimal_columns(root))  # and it is genuinely optimal


def test_prove_winning_strategy_parallel_matches_sequential_and_proves_the_root():
    # The PARALLEL leaf-solver must reach the SAME proven result as the sequential prover (exact solves are
    # deterministic): the root is proven a win, BookAgent converts, and every booked value agrees.
    from harness.book import prove_winning_strategy_parallel

    game = Connect4()
    root = _c4_mover_win_root(empties=14, seed=1, require_deep=True)

    seq = Tablebase(cap=1_000_000)
    prove_winning_strategy(game, seq, root=root, strategist=root.to_move, max_plies=42, max_exact_empty=8)

    par = Tablebase(cap=1_000_000)
    stats = prove_winning_strategy_parallel(game, par, root=root, strategist=root.to_move, max_plies=42,
                                            max_exact_empty=8, workers=4, max_seconds=10)
    assert stats["root_proven"] is True and par.proven_value(_key(game, root)) > 0
    assert set(book_optimal_actions(par, game, root)) == set(optimal_columns(root))  # genuinely optimal
    # the parallel book converts the win vs the exact solver, exactly like the sequential one
    res = verify_solved(game, lambda: BookAgent(par, "connect4", solve_endgame=8), games=2, start=root,
                        strategist=root.to_move)
    assert res["solved"] is True


def test_prove_winning_strategy_is_bounded_and_resumable():
    # max_exact_empty=0 forces the WHOLE tree to be proven node-by-node (no solver shortcut) so bounding is real.
    game = TicTacToe()
    book = Tablebase(cap=100_000)
    first = prove_winning_strategy(game, book, max_plies=9, max_exact_empty=0, max_positions=3)
    assert 1 <= first["proven"] <= 8  # the position cap stops the pass early (a small overrun past a leaf is fine)
    n_after = len(book)
    prove_winning_strategy(game, book, max_plies=9, max_exact_empty=0)  # resume: proven skipped, coverage finishes
    assert len(book) > n_after
    assert winning_strategy_coverage(game, book, max_plies=9, max_exact_empty=0)["complete"] is True


def test_prove_winning_strategy_defers_a_hard_solve_instead_of_hanging():
    # A from-opening strategist node with a tiny per-solve cap must DEFER (not hang) — the endgame-back grind.
    game = Connect4()
    book = Tablebase(cap=1_000_000)
    stats = prove_winning_strategy(game, book, max_plies=6, max_exact_empty=0, max_seconds=0.05, max_positions=50)
    assert stats["deferred"] >= 1  # the cold opening solve blew the 50ms cap → deferred, no hang


# --- M2: THE BOOK-AWARE AGENT CONVERTS -----------------------------------------------------------------------
def test_book_agent_converts_a_proven_connect4_win_vs_the_exact_oracle():
    game = Connect4()
    root = _c4_mover_win_root(empties=14, seed=1, require_deep=True)
    book = Tablebase(cap=1_000_000)
    prove_winning_strategy(game, book, root=root, strategist=root.to_move, max_plies=42, max_exact_empty=8)
    # BookAgent (book for the deep proven line, exact solver for the ≤8-empty leaves) converts the win against
    # perfect defence — M2 met on this subtree. The book carries the 14→8-empty part the solver isn't asked for.
    res = verify_solved(game, lambda: BookAgent(book, "connect4", solve_endgame=8),
                        games=2, start=root, strategist=root.to_move)
    assert res["solved"] is True


def test_book_agent_never_loses_tic_tac_toe_from_a_full_book():
    # A FULLY solved game: build_book proves the WHOLE tree (both seats' nodes, every reply), so a lookup-only
    # BookAgent never loses a drawable position as EITHER player — the complete pipeline on a game we fully solve.
    game = TicTacToe()
    book = Tablebase(cap=100_000)
    build_book(game, book, max_plies=9, max_positions=100_000)
    assert winning_strategy_coverage(game, book, max_plies=9)["complete"] is True
    for seat in (0, 1):
        res = p1_conversion(game, lambda: BookAgent(book, "tictactoe", solve_endgame=0),
                            lambda: ExactOptimalAgent(), games=2, strategist=seat)
        assert res["not_lost_rate"] == 1.0  # optimal from a full book → never loses a drawable game either side


# --- M3: DISTILL THE PROVEN STRATEGY INTO THE NET (targets are exactly optimal) ------------------------------
def test_book_distill_targets_are_exactly_the_proven_optimal_play():
    # The distillation the net learns from is the PROVEN optimal move-set + exact value — so a net that fits the
    # anchor plays perfectly (the full train-to-SOLVED is the compute grind, not a unit test). Verified
    # structurally on the Connect-4 forced-win subtree's root (booked in its own orientation → no symmetry caveat):
    # the distill example's policy support = the exact optimal set, and its value = the exact game-theoretic value.
    game = Connect4()
    root = _c4_mover_win_root(empties=14, seed=1, require_deep=True)
    book = Tablebase(cap=1_000_000)
    prove_winning_strategy(game, book, root=root, strategist=root.to_move, max_plies=42, max_exact_empty=8)
    entry = book.entry(_key(game, root))  # the root's own canonical entry
    assert entry is not None and entry.status == PROVEN
    examples = book_distill_examples(game, book, [root], proof_copies=1, estimate_copies=0)
    assert len(examples) == 1
    _x, pi, value = examples[0]
    assert float(value) == float(position_value(game, root))  # exact value target (a proven win)
    support = {c for c, p in enumerate(pi) if p > 0}
    assert support == set(optimal_columns(root))  # policy target support = exactly the proven optimal set
