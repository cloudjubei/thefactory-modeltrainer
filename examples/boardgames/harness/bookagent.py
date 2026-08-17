"""BookAgent — the deployable, FAST, provably-optimal-where-covered agent (Levers 1 + 3 at play time).

Composes exact/strong sources in priority order, none of which pays a minutes-long opening solve at play time:
  1. an immediate winning move (take it outright);
  2. the OPENING BOOK — one-ply lookahead over stored optimal-play values (instant, exact wherever it reaches);
  3. the EXACT ENDGAME solver (perfect once ≤ `solve_endgame` cells are empty);
  4. a strong FALLBACK for the middle gap the book hasn't reached yet — a near-perfect depth-limited oracle by
     default, or a trained net if one is handed in.
As the book grows (harness/book.build_book), this agent's provably-optimal region grows with it; the fallback
only ever covers the shrinking middle band, and the endgame is always exact."""
from __future__ import annotations

import random

from harness.book import book_optimal_actions
from harness.game import Game, State
from harness.tablebase import Tablebase


def _center_first(actions: list[int], n_actions: int) -> int:
    mid = n_actions // 2
    return min(actions, key=lambda a: abs(a - mid))


class BookAgent:
    kind = "book"

    def __init__(self, book: Tablebase, game_name: str = "connect4", solve_endgame: int = 22,
                 fallback=None, depth: int = 12):
        self.book = book
        self.game_name = game_name
        self.solve_endgame = int(solve_endgame)
        self.depth = int(depth)
        self._fallback = fallback
        self.sims_used = 0

    def _fb(self):
        if self._fallback is None:
            from harness.solver import NearPerfectOracle

            self._fallback = NearPerfectOracle(depth=self.depth)
        return self._fallback

    def act(self, game: Game, state: State, rng: random.Random) -> int:
        self.sims_used += 1
        legal = game.legal_actions(state)
        if len(legal) == 1:
            return legal[0]
        me = game.current_player(state)
        wins = [a for a in legal if game.step(state, a).winner == me]
        if wins:
            return _center_first(wins, game.num_actions)
        acts = book_optimal_actions(self.book, game, state)  # exact opening (one-ply table lookahead)
        if acts:
            return _center_first(acts, game.num_actions)
        solve = getattr(game, "exact_optimal_actions", None)
        optimal = solve(state, self.solve_endgame) if solve is not None else None  # exact endgame
        if optimal:
            return _center_first(optimal, game.num_actions)
        return self._fb().act(game, state, rng)  # strong fallback for the unbooked middle
