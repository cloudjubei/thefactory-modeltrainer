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


def move_values(game, state) -> dict:
    """The exact value to the mover of every legal action: a terminal child scores its return, any other child
    minus its position_value (stated from the child's mover, i.e. the opponent). Generic over any SolvableGame."""
    me = game.current_player(state)
    vals = {}
    for a in game.legal_actions(state):
        child = game.step(state, a)
        vals[a] = game.returns(child)[me] if game.is_terminal(child) else -game.position_value(child)
    return vals


def optimal_actions(game, state) -> set:
    """The solver's optimal move SET at `state`: every action that preserves the game-theoretic value to the
    mover."""
    vals = move_values(game, state)
    if not vals:
        return set()
    best = max(vals.values())
    return {a for a, v in vals.items() if v == best}


def per_state_act(game, make_agent, seed: int = 0):
    """An act_fn that answers every state with a FRESH agent and a FRESH rng. Coverage is a property of a policy
    — a function of the state — but a search agent that keeps its tree across calls plays each state with the
    search spent on every state asked before it, so its score depends on the order states are visited (measured:
    0.9904 forward vs 0.9968 reversed on one net). Building the agent per state removes that dependence."""
    def act(state):
        return make_agent().act(game, state, random.Random(seed))
    return act


def _key(game, state):
    return game.canonical_key(state) if hasattr(game, "canonical_key") else game.state_key(state)


def _play_all(act_fn, states: list, purity_probe: int = 16) -> list:
    """Ask `act_fn` for a move at every state, then ask again at the first `purity_probe` states. A different
    answer the second time means the act_fn carries state between calls, so any coverage it earns depends on
    evaluation ORDER rather than on the policy — refused, with per_state_act as the remedy."""
    moves = [act_fn(s) for s in states]
    for i in range(min(purity_probe, len(states))):
        if act_fn(states[i]) != moves[i]:
            raise ValueError(f"act_fn answered state #{i} differently when asked again: it keeps state between "
                             f"calls, so coverage would depend on evaluation order — wrap the agent with "
                             f"per_state_act so every state is searched from scratch")
    return moves


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
                   states: list | None = None, purity_probe: int = 16) -> dict:
    """What fraction of the reachable, canonical state space does `act_fn` play optimally? Returns the coverage
    rate, the state count, whether the space was fully enumerated (`complete`), and a by-ply breakdown so the
    part of the game the model gets wrong (usually the opening) is visible, not averaged away."""
    complete = None
    if states is None:
        states, complete = reachable_states(game, exact=exact, max_states=max_states, symmetry=symmetry,
                                            sample_playouts=sample_playouts, max_plies=max_plies, seed=seed)
    by_ply: dict = {}
    optimal = 0
    for s, move in zip(states, _play_all(act_fn, states, purity_probe)):
        ok = move in optimal_actions(game, s)
        optimal += 1 if ok else 0
        p = _ply(game, s)
        d = by_ply.setdefault(p, [0, 0])
        d[0] += 1 if ok else 0
        d[1] += 1
    n = len(states)
    return {"coverage": (optimal / n) if n else 0.0, "n_states": n, "optimal": optimal,
            "complete": complete, "symmetry_reduced": symmetry,
            "by_ply": {p: round(c[0] / c[1], 4) for p, c in sorted(by_ply.items())}}


def decided_frontier(game, act_fn, states: list, purity_probe: int = 16) -> dict:
    """The 'provable-outcome' share: of the given states, the fraction where the mover has a PROVEN non-loss
    (a forced win or draw) AND `act_fn` preserves it. A high decided-frontier means large parts of the tree are
    settled — the subtree beyond a proven win needs no further search, the pruning the user asked to measure."""
    decided = kept = 0
    for s, move in zip(states, _play_all(act_fn, states, purity_probe)):
        vals = move_values(game, s)
        best = max(vals.values())
        if best >= 0:
            decided += 1
            kept += 1 if vals[move] == best else 0
    n = max(1, len(states))
    return {"decided_states": decided, "decided_frac": decided / n,
            "kept_when_decided": (kept / decided) if decided else 0.0}


_OUTCOME = {1: "win", 0: "draw", -1: "loss"}


def coverage_failures(game, act_fn, states: list, purity_probe: int = 16) -> list:
    """Every state where `act_fn` leaves the optimal set, with what it cost: the move played, the optimal set,
    the best value available to the mover, the value of the move played, and `severity` (e.g. win->draw). The
    coverage RATE says how often the model is wrong; this says WHERE, which is what a cause can be read from."""
    out = []
    for s, move in zip(states, _play_all(act_fn, states, purity_probe)):
        vals = move_values(game, s)
        best = max(vals.values())
        if vals[move] == best:
            continue
        played = vals[move]
        out.append({"key": _key(game, s), "ply": _ply(game, s), "move": move,
                    "optimal": sorted(a for a, v in vals.items() if v == best), "best": best, "played": played,
                    "lost": best - played,
                    "severity": f"{_OUTCOME.get(best, best)}->{_OUTCOME.get(played, played)}"})
    return out


def failable_keys(game, states: list) -> set:
    """Keys of the states where a NON-optimal move exists. Only these can ever be failed, so they — not the whole
    space — are the universe a failure pattern must be judged against: a state where every move is optimal is
    never missed by anyone, and counting it would make any failure set look concentrated."""
    out = set()
    for s in states:
        vals = move_values(game, s)
        if len(set(vals.values())) > 1:
            out.add(_key(game, s))
    return out


def optimal_play_keys(game, max_states: int = 300000) -> set:
    """Keys of the non-terminal states reachable when BOTH sides only ever play optimal moves — the equilibrium
    manifold. A state outside it can only be reached after somebody blundered, which is exactly the kind of
    position self-play between improving players stops visiting."""
    seen = set()
    q = deque([game.initial_state()])
    while q and len(seen) < max_states:
        s = q.popleft()
        if game.is_terminal(s):
            continue
        k = _key(game, s)
        if k in seen:
            continue
        seen.add(k)
        for a in optimal_actions(game, s):
            q.append(game.step(s, a))
    return seen


def training_visits(game, buffer: list, encode_fn, max_states: int = 300000) -> dict:
    """How often each canonical state appears in a training `buffer` of (x, ...) examples. Every reachable
    position (all symmetric images, not only canonical ones) is encoded with `encode_fn(game, state)` and mapped
    to its canonical key, so an augmented image counts toward its class. An encoding shared by two distinct
    classes would silently merge their counts, so it is refused; buffer entries matching no reachable position
    are counted as `unmatched`."""
    raw, _ = reachable_states(game, exact=True, max_states=max_states, symmetry=False)
    lookup: dict = {}
    for s in raw:
        code = _as_bytes(encode_fn(game, s))
        k = _key(game, s)
        if lookup.setdefault(code, k) != k:
            raise ValueError("encode_fn collides: two positions that are not symmetric images of each other share "
                             "an encoding, so their training visits cannot be told apart")
    visits: dict = {}
    unmatched = 0
    for entry in buffer:
        k = lookup.get(_as_bytes(entry[0]))
        if k is None:
            unmatched += 1
            continue
        visits[k] = visits.get(k, 0) + 1
    return {"visits": visits, "unmatched": unmatched}


def raw_encoding_lookup(game, encode_fn, max_states: int = 300000) -> dict:
    """{encoding bytes: the raw position it encodes} over every reachable position (every symmetric image, not only
    canonical ones). Refused when two different positions share an encoding — a label could not then be traced
    to the position it was written for."""
    raw, _ = reachable_states(game, exact=True, max_states=max_states, symmetry=False)
    lookup: dict = {}
    for s in raw:
        prev = lookup.setdefault(_as_bytes(encode_fn(game, s)), s)
        if game.state_key(prev) != game.state_key(s):
            raise ValueError("encode_fn collides: two different positions share an encoding, so a label cannot "
                             "be attributed to the position it was written for")
    return lookup


def label_dose(game, examples: list, encode_fn, target_keys, max_states: int = 300000,
               lookup: dict | None = None) -> dict:
    """The policy LABELS a net actually trained on at the target states. Each example `(x, pi, ...)` is mapped from
    its encoding back to the raw position it encodes — an augmented symmetric image keeps its own frame, with `pi`
    permuted to match — and its label is checked against the solver's optimal set for THAT position: `argmax_ok`
    (the label's top move is optimal) and `opt_mass` (the label's probability on optimal moves; the net trains on
    the soft label, so a wrong-argmax label can still push toward the right move). Counts, never ratios: a seed
    that trained on no target example has n = 0, which must read as "not delivered", not as a failure rate.
    Pass a prebuilt `lookup` (raw_encoding_lookup) when dosing many batches of the same game."""
    if lookup is None:
        lookup = raw_encoding_lookup(game, encode_fn, max_states)
    targets = set(target_keys)
    optimal: dict = {}
    per_key: dict = {}
    unmatched = 0
    for entry in examples:
        s = lookup.get(_as_bytes(entry[0]))
        if s is None:
            unmatched += 1
            continue
        k = _key(game, s)
        if k not in targets:
            continue
        sk = game.state_key(s)
        if sk not in optimal:
            optimal[sk] = optimal_actions(game, s)
        pi = list(entry[1])
        top = max(range(len(pi)), key=lambda a: pi[a])
        row = per_key.setdefault(k, {"n": 0, "argmax_ok": 0, "opt_mass": 0.0})
        row["n"] += 1
        row["argmax_ok"] += 1 if top in optimal[sk] else 0
        row["opt_mass"] += sum(pi[a] for a in optimal[sk])
    return {"per_key": per_key, "unmatched": unmatched, "n_target": sum(r["n"] for r in per_key.values()),
            "argmax_ok": sum(r["argmax_ok"] for r in per_key.values()),
            "opt_mass": sum(r["opt_mass"] for r in per_key.values())}


def _as_bytes(x) -> bytes:
    if isinstance(x, bytes):
        return x
    return x.detach().cpu().numpy().tobytes() if hasattr(x, "detach") else bytes(repr(x), "utf8")


def blind_spot_concentration(failure_sets: list, universe: set, trials: int = 20000, seed: int = 0) -> dict:
    """Are independently trained models wrong at the SAME states, or at different ones? The statistic is the
    number of (seed-pair, shared-failure) coincidences, sum over states of C(misses, 2). The null is that each
    seed's failures are a uniform random subset of the FAILABLE `universe` of the same size — seed noise with no
    systematic blind spot; p is its Monte-Carlo upper tail. `recurring` lists the states at least half the seeds
    miss. One seed cannot say anything about consistency, so fewer than two is refused."""
    sets = [set(f) for f in failure_sets]
    if len(sets) < 2:
        raise ValueError("concentration needs at least two independently trained seeds — one seed's failures say "
                         "nothing about whether another seed would fail at the same states")
    uni = sorted(universe)
    for i, f in enumerate(sets):
        if not f <= universe:
            raise ValueError(f"seed #{i} failed {len(f - universe)} state(s) outside the universe — the null must "
                             f"be drawn over every state a model could fail, or the p-value is meaningless")
    counts: dict = {}
    for f in sets:
        for k in f:
            counts[k] = counts.get(k, 0) + 1

    def pairs(cs):
        return sum(c * (c - 1) // 2 for c in cs)

    observed = pairs(counts.values())
    n = len(uni)
    sizes = [len(f) for f in sets]
    expected = sum(sizes[i] * sizes[j] for i in range(len(sizes)) for j in range(i + 1, len(sizes))) / n if n else 0.0
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        c: dict = {}
        for k in sizes:
            for x in rng.sample(uni, k):
                c[x] = c.get(x, 0) + 1
        hits += 1 if pairs(c.values()) >= observed else 0
    half = len(sets) / 2
    return {"n_seeds": len(sets), "universe": n, "sizes": sizes, "counts": counts,
            "observed_pairs": observed, "expected_pairs": expected,
            "ratio": (observed / expected) if expected else float("inf") if observed else 1.0,
            "p": (1 + hits) / (1 + trials),
            "recurring": sorted(k for k, c in counts.items() if c >= half)}


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
