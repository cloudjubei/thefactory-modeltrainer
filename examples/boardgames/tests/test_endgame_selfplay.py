"""§C.7 #2 — the online endgame-tablebase-from-play loop: batched retrograde extension, train wiring, and the
generic-degradation gate. The write-through memo + exact-target override are covered in test_neural.py."""
import random

from games.connect4 import Connect4
from harness.book import _key
from harness.neural import (
    AlphaZeroAgent,
    Connect4Net,
    _endgame_enabled,
    extend_endgame_frontier,
    train_alphazero,
)
from harness.tablebase import Tablebase


class _NoHooksC4:
    """A Connect 4 with the exact-solve hooks HIDDEN — a stand-in for a game the online endgame loop cannot serve
    (chess opening / Go). Every other rule delegates, so training runs identically; only the gate should see no hooks."""

    def __init__(self):
        self._g = Connect4()

    def __getattr__(self, name):
        if name in ("canonical_key", "exact_optimal_actions"):
            raise AttributeError(name)
        return getattr(self._g, name)


def test_extend_frontier_climbs_via_free_minimax():
    # The climb mechanism: a parent too shallow to CHEAP-SOLVE (empties > max_empty) is still PROVEN once all its
    # children are booked — book._prove's free-minimax/winning-child branch, not a direct solve. This is what lets
    # the proven frontier march opening-ward across iterations, one ply per pass.
    game = Connect4()
    rng = random.Random(0)
    parent = game.initial_state(rng)
    parent = game.step(parent, 3)
    parent = game.step(parent, 3)  # a near-opening position: ~40 empties, far above any cheap-solve cutoff
    assert getattr(game, "exact_optimal_actions")(parent, 8) is None  # NOT cheap-solvable at max_empty=8

    tb = Tablebase(cap=1000)
    for a in game.legal_actions(parent):
        child = game.step(parent, a)
        assert not game.is_terminal(child)
        tb.put_proven(_key(game, child), -1)  # each child is a loss for the child-mover ⇒ a win for the parent

    proven = extend_endgame_frontier(game, tb, [parent], max_empty=8, max_positions=100, deadline_seconds=5.0)
    assert proven == 1
    assert tb.proven_value(_key(game, parent)) == 1  # proven a WIN via the booked children, no cheap solve
    assert tb.entry(_key(game, parent)).best_actions == 0  # VALUE-ONLY store (mirror-safe)


def test_extend_frontier_respects_budget_and_skips_booked():
    game = Connect4()
    rng = random.Random(1)
    parent = game.step(game.step(game.initial_state(rng), 2), 2)
    tb = Tablebase(cap=1000)
    for a in game.legal_actions(parent):
        tb.put_proven(_key(game, game.step(parent, a)), -1)

    assert extend_endgame_frontier(game, tb, [parent], max_empty=8, max_positions=0, deadline_seconds=5.0) == 0
    assert tb.proven_value(_key(game, parent)) is None  # zero budget ⇒ nothing proven
    tb.put_proven(_key(game, parent), 1)  # already booked ⇒ the extension skips it (no wasted re-prove)
    assert extend_endgame_frontier(game, tb, [parent], max_empty=8, max_positions=100, deadline_seconds=5.0) == 0


def test_endgame_enabled_gate():
    # The generic gate: the loop turns on ONLY for a game exposing BOTH hooks AND a run tablebase.
    game = Connect4()
    tb = Tablebase(cap=1000)
    assert _endgame_enabled(game, tb) is True
    assert _endgame_enabled(game, None) is False  # no store ⇒ off
    assert _endgame_enabled(_NoHooksC4(), tb) is False  # no exact hooks ⇒ off (degrades to pure #1)


def test_train_alphazero_grows_and_amortizes():
    # A short c4 run with the endgame loop ON must GROW the run tablebase (frontier proofs from self-play) and
    # surface the amortisation gauge in history — the observable evidence the online loop fired.
    game = Connect4()
    run_tb = Tablebase(cap=200000)
    _net, history = train_alphazero(
        game, iterations=2, selfplay_games=6, sims=8, epochs=1, seed=3,
        endgame_tb=run_tb, endgame_max_empty=14, endgame_exact_targets=1,
        endgame_extend_positions=500, endgame_extend_seconds=5.0,
    )
    assert len(run_tb) > 0  # the frontier grew from self-play
    assert history[-1]["endgame_total"] >= history[0]["endgame_total"]  # monotone non-decreasing store
    assert sum(h["endgame_booked"] for h in history) > 0  # the extension proved positions
    assert all("endgame_solves" in h and "endgame_hits" in h for h in history)


def test_generic_degradation_no_hooks():
    # Switch ON but the game lacks the hooks ⇒ byte-identical to pure #1: the store stays empty AND the training
    # trajectory (loss curve) matches the switch-off run exactly (same seed, same RNG consumption).
    off_net, off_hist = train_alphazero(Connect4(), iterations=2, selfplay_games=4, sims=8, epochs=1, seed=7)
    run_tb = Tablebase(cap=200000)
    on_net, on_hist = train_alphazero(
        _NoHooksC4(), iterations=2, selfplay_games=4, sims=8, epochs=1, seed=7,
        endgame_tb=run_tb, endgame_max_empty=14, endgame_exact_targets=1,
    )
    assert len(run_tb) == 0  # no hooks ⇒ nothing recorded
    assert [h["loss"] for h in on_hist] == [h["loss"] for h in off_hist]  # identical training path
