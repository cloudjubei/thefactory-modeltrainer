"""Distance-to-optimal benchmark for Connect 4 — how often a policy plays a game-theoretically OPTIMAL move,
scored against the perfect-play oracle (harness/solver.py).

`oracle_optimality_rate` = the fraction of benchmark positions where the policy's chosen move lies in the
oracle's optimal SET (several moves can be equally optimal). It is the honest "how close to solved" gauge: the
oracle scores 1.0 by construction; a model approaches 1.0 as it learns perfect play.

SCOPE / HONESTY: solving from the OPENING is a minutes-long search in pure Python (mid/late positions are
instant), so the default corpus samples positions with enough stones on the board that the exact solve is fast.
A high rate here is NECESSARY but not SUFFICIENT for "solved" — verifying opening / principal-variation play
needs a precomputed PV fixture (a one-time offline solve), tracked as a follow-on.
"""
from __future__ import annotations

import random
from typing import Callable

from harness import solver
from harness.game import Game, State
from harness.solver import optimal_columns

ActFn = Callable[[State], int]


def sample_solvable_positions(game: Game, n: int, min_moves: int, seed: int) -> list[State]:
    """`n` random-play, non-terminal positions with at least `min_moves` stones already down (so the exact
    solve is cheap) and at least two legal replies (so 'optimal move' is a real choice)."""
    rng = random.Random(seed)
    out: list[State] = []
    attempts = 0
    while len(out) < n and attempts < n * 300:
        attempts += 1
        s = game.initial_state(rng)
        depth = min_moves + rng.randint(0, 6)
        for _ in range(depth):
            legal = game.legal_actions(s)
            if not legal:
                break
            s = game.step(s, rng.choice(legal))
            if s.done:
                break
        if not s.done and len(game.legal_actions(s)) >= 2:
            out.append(s)
    return out


def optimality_rate(states: list[State], act_fn: ActFn) -> tuple[int, int]:
    """(optimal moves, positions scored) — a move counts optimal iff it's in the oracle's optimal set."""
    optimal = 0
    for s in states:
        if act_fn(s) in optimal_columns(s, tt=solver._TT):
            optimal += 1
    return optimal, len(states)


def evaluate_optimality(
    game: Game,
    act_fn: ActFn,
    states: list[State] | None = None,
    n: int = 120,
    min_moves: int = 20,
    seed: int = 0,
) -> dict:
    """Score a policy's optimality against the oracle. Pass `states` to score the policy on positions IT
    reached (the meaningful corpus); omit to sample a fixed random-play corpus."""
    if states is None:
        states = sample_solvable_positions(game, n, min_moves, seed)
    optimal, total = optimality_rate(states, act_fn)
    return {
        "oracle_optimality_rate": (optimal / total) if total else 0.0,
        "oracle_positions": total,
        "oracle_min_moves": min_moves,
    }
