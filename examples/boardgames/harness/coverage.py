"""§C.41 OPTIMALITY AS STATE-SPACE COVERAGE — a measurable definition of "how close to perfect play".

The idea (user direction, 2026-09-22): a model plays optimally iff, for every reachable game state, it picks a
move the solver also calls optimal. So OPTIMALITY = the fraction of the reachable state space the model gets
right. Near 100% coverage ⇒ near-perfect: you could read the best move off the model at every state and reach the
proven result. For a small solvable game the state space is enumerable EXACTLY (collapsed by the verified
symmetry group — §C.36 — so a position and its mirror count once); for a large one it is SAMPLED, and pruned by
the fact that once a state's outcome is PROVEN, its subtree needs no exploration (the `decided` frontier below).

This needs a solver for ground truth, so it is exact only on `SolvableGame`s (connect4, tictactoe). For a
solver-free game (othello, chess) the same shape holds with a strong reference in place of the solver — that is
the open research path recorded in the north star, not something certifiable here yet.
"""
from __future__ import annotations

import random
from collections import deque


def optimal_actions(game, state) -> set:
    """The solver's optimal move SET at `state`: every action that preserves the game-theoretic value to the
    mover. Generic over any SolvableGame — the value to the mover of playing `a` is minus the child's
    position_value (which is stated from the child's mover, i.e. the opponent)."""
    me = game.current_player(state)
    vals = {}
    for a in game.legal_actions(state):
        child = game.step(state, a)
        if game.is_terminal(child):
            vals[a] = game.returns(child)[me]
        else:
            vals[a] = -game.position_value(child)
    if not vals:
        return set()
    best = max(vals.values())
    return {a for a, v in vals.items() if v == best}


def reachable_states(game, exact: bool = True, max_states: int = 300000, symmetry: bool = True,
                     sample_playouts: int = 2000, max_plies: int = 10**9, seed: int = 0):
    """Canonical, non-terminal reachable states. `exact`: full BFS from the initial state (small games) — returns
    `complete=True` only if it finished under `max_states`. Else: sampled by random playouts (any game). Symmetry
    collapses a state and its images to one via `canonical_key` when available."""
    def key(s):
        if symmetry and hasattr(game, "canonical_key"):
            return game.canonical_key(s)
        return game.state_key(s)

    seen, out = set(), []
    if exact:
        q = deque([game.initial_state()])
        complete = True
        while q:
            if len(seen) >= max_states:
                complete = False
                break
            s = q.popleft()
            k = key(s)
            if k in seen:
                continue
            seen.add(k)
            if game.is_terminal(s):
                continue
            out.append(s)
            for a in game.legal_actions(s):
                q.append(game.step(s, a))
        return out, complete
    rng = random.Random(seed)
    for _ in range(sample_playouts):
        s = game.initial_state(rng)
        plies = 0
        while not game.is_terminal(s) and plies < max_plies:
            k = key(s)
            if k not in seen:
                seen.add(k)
                out.append(s)
            s = game.step(s, rng.choice(game.legal_actions(s)))
            plies += 1
    return out, False


def _ply(game, state) -> int:
    return game.ply(state) if hasattr(game, "ply") else sum(1 for v in getattr(state, "board", ()) if v)


def state_coverage(game, act_fn, exact: bool = True, max_states: int = 300000, symmetry: bool = True,
                   sample_playouts: int = 2000, max_plies: int = 10**9, seed: int = 0,
                   states: list | None = None) -> dict:
    """What fraction of the reachable, canonical state space does `act_fn` play optimally? Returns the coverage
    rate, the state count, whether the space was fully enumerated (`complete`), and a by-ply breakdown so the
    part of the game the model gets wrong (usually the opening) is visible, not averaged away."""
    complete = None
    if states is None:
        states, complete = reachable_states(game, exact=exact, max_states=max_states, symmetry=symmetry,
                                            sample_playouts=sample_playouts, max_plies=max_plies, seed=seed)
    by_ply: dict = {}
    optimal = 0
    for s in states:
        ok = act_fn(s) in optimal_actions(game, s)
        optimal += 1 if ok else 0
        p = _ply(game, s)
        d = by_ply.setdefault(p, [0, 0])
        d[0] += 1 if ok else 0
        d[1] += 1
    n = len(states)
    return {"coverage": (optimal / n) if n else 0.0, "n_states": n, "optimal": optimal,
            "complete": complete, "symmetry_reduced": symmetry,
            "by_ply": {p: round(c[0] / c[1], 4) for p, c in sorted(by_ply.items())}}


def decided_frontier(game, act_fn, states: list) -> dict:
    """The 'provable-outcome' share: of the given states, the fraction where the mover has a PROVEN non-loss
    (a forced win or draw) AND `act_fn` preserves it. A high decided-frontier means large parts of the tree are
    settled — the subtree beyond a proven win needs no further search, the pruning the user asked to measure."""
    decided = kept = 0
    for s in states:
        me = game.current_player(s)
        best = max((game.returns(game.step(s, a))[me] if game.is_terminal(game.step(s, a))
                    else -game.position_value(game.step(s, a))) for a in game.legal_actions(s))
        if best >= 0:                       # the mover can force at least a draw (a decided, non-lost node)
            decided += 1
            if act_fn(s) in optimal_actions(game, s):
                kept += 1
    n = max(1, len(states))
    return {"decided_states": decided, "decided_frac": decided / n,
            "kept_when_decided": (kept / decided) if decided else 0.0}


# --- solver-FREE optimality: a strong REFERENCE, calibrated against the exact solver where one exists -------

def calibrate_reference(game, reference_fn, n: int = 200, min_moves: int = 12, seed: int = 0,
                        states: list | None = None) -> dict:
    """How trustworthy is a solver-FREE reference? On SOLVABLE positions, measure how often the reference's move
    is actually in the EXACT solver's optimal set. `exact_agreement` is the reference's own optimality; its
    complement is the FALSE-OPTIMAL rate — the rate at which crediting a model for matching this reference would
    credit a move the solver calls wrong. You must know this before trusting the reference off-solver."""
    if states is None:
        from harness.benchmark import sample_solvable_positions
        states = sample_solvable_positions(game, n, min_moves, seed)
    agree = sum(1 for st in states if reference_fn(st) in optimal_actions(game, st))
    m = len(states)
    return {"exact_agreement": (agree / m) if m else 0.0, "false_optimal": (1 - agree / m) if m else 1.0,
            "n": m, "min_moves": min_moves}


def coverage_vs_reference(game, act_fn, reference_fn, states: list) -> dict:
    """Coverage graded against a REFERENCE's move instead of the solver — the solver-free analogue of
    state_coverage. Trust it only alongside `calibrate_reference`: the reported coverage over-states true
    optimality by roughly the reference's false_optimal rate."""
    optimal = sum(1 for s in states if act_fn(s) == reference_fn(s))
    n = max(1, len(states))
    return {"coverage_vs_reference": optimal / n, "n_states": len(states)}
