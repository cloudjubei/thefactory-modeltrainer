"""§C.47 transfer probes — where a trained net goes wrong RELATIVE TO ITS OWN TRAINING DATA, on a game too large to
enumerate.

On tic-tac-toe one-ply siblings put ~89% of the failable positions into training (§C.46 h38), so their effect was
mostly in-sample. On Connect-4 they can only help where the net's errors sit one move from positions self-play
visited. Measuring that needs three things the tic-tac-toe tools do not provide, because every one of those
enumerates the state space:

  T1 A RECORDER of the positions self-play actually produced, keyed canonically, with no enumeration. It observes
     only — the run trains bit-identically with it on.
  T2 The NEIGHBOURHOOD of the visited set: its one-ply children (what siblings could reach) and two-ply
     grandchildren, each excluding what is nearer.
  T3 The FIRST ERROR of a policy converting a proven win against an exact defender: the first position where its
     move gives the proven win away. Where that position sits relative to T2 is the measurement."""
from __future__ import annotations

import random
from contextlib import contextmanager

from harness.coverage import _key, move_values

VISITED, ONE_MOVE, TWO_MOVES, FURTHER = "visited", "one_move_off", "two_moves_off", "further"


@contextmanager
def record_selfplay_states(game, probe: dict | None = None):
    """While active, every self-play game `harness.neural.self_play_game` plays appends one row to the yielded
    log's `games`: {"pass": the number of `train_net` calls made before the game, "states": the positions the
    game returned as training examples}. Random opening plies are not training examples and are not recorded.

    With a `probe` ({"x": encoded positions, "legal": bool mask, "optimal": one set of moves per position}, built
    before training so nothing here needs a solver), every pass appends the raw policy's accuracy on it to
    `log["probe"]` — whether the net is still learning when training stops. The forward runs in eval mode with
    no gradient and the net is put back in the mode the pass left it.

    It observes only: the wrapped game is asked for its states (which changes the shape of its return and nothing
    else), the caller receives exactly the shape it asked for, and nothing here draws from any RNG. Games played in
    worker processes are invisible to it, so a caller compares the recorded game count with what it launched. Both
    wrapped functions are restored however the block exits."""
    import harness.neural as neural

    log: dict = {"game": getattr(game, "name", type(game).__name__), "games": [], "passes": 0, "probe": []}
    original_play, original_train = neural.self_play_game, neural.train_net

    def play(*args, **kwargs):
        asked = kwargs.get("return_states", False)
        full = original_play(*args, **{**kwargs, "return_states": True})
        log["games"].append({"pass": log["passes"], "states": [e[0] for e in full]})
        return full if asked else [tuple(e[1:]) for e in full]

    def train(*args, **kwargs):
        result = original_train(*args, **kwargs)
        log["passes"] += 1
        if probe is not None:
            log["probe"].append(policy_accuracy(args[0] if args else kwargs["net"], probe))
        return result

    neural.self_play_game, neural.train_net = play, train
    try:
        yield log
    finally:
        neural.self_play_game, neural.train_net = original_play, original_train


def policy_accuracy(net, probe: dict) -> float:
    """Share of the probe positions where the raw policy's argmax over legal moves is an optimal move."""
    import torch

    mode = net.training
    net.eval()
    try:
        with torch.no_grad():
            logits, _v = net(probe["x"])
    finally:
        net.train(mode)
    moves = logits.masked_fill(~probe["legal"], float("-inf")).argmax(dim=1).tolist()
    return sum(1 for m, opt in zip(moves, probe["optimal"]) if m in opt) / len(moves)


def build_probe(game, states, encode_fn, values_fn=None) -> dict:
    """A fixed probe for `policy_accuracy`: the positions, their legal masks and their optimal move sets, all
    computed now — before training — so the recorder never needs a solver while the net trains."""
    import torch

    values_fn = values_fn or (lambda s: move_values(game, s))
    legal = torch.zeros(len(states), game.num_actions, dtype=torch.bool)
    optimal = []
    for i, s in enumerate(states):
        legal[i, game.legal_actions(s)] = True
        vals = values_fn(s)
        best = max(vals.values())
        optimal.append({a for a, v in vals.items() if v == best})
    return {"x": torch.stack([encode_fn(game, s) for s in states]), "legal": legal, "optimal": optimal}


@contextmanager
def forbid_solver():
    """While active, any exact solve raises — the proof, rather than a reading of the config, that a run which
    must be solver-free called no solver. Patches the solver's search entry points; restored however the block
    exits."""
    import harness.solver as solver

    saved = (solver._solve, solver.move_values)

    def refuse(*args, **kwargs):
        raise RuntimeError("the exact solver was called inside a block that must be solver-free")

    solver._solve, solver.move_values = refuse, refuse
    try:
        yield
    finally:
        solver._solve, solver.move_values = saved


def nontrivial(game, state, vals: dict) -> bool:
    """A position where a policy can go wrong in a way that matters: not terminal, the mover has no immediate win
    (win-in-one positions dominated §C.41's random roots, and they test nothing about the midgame), and the moves
    do not all share one game-theoretic value."""
    if game.is_terminal(state) or len(set(vals.values())) < 2:
        return False
    mover = game.current_player(state)
    for a in game.legal_actions(state):
        child = game.step(state, a)
        if game.is_terminal(child) and game.returns(child)[mover] > 0:
            return False
    return True


def rings_at(game, states, ply: int) -> dict:
    """Representatives, keyed canonically, of the positions at `ply` that are one move off the visited `states`
    (children of visited positions at ply-1, not themselves visited) and two moves off (grandchildren of visited
    positions at ply-2, in neither nearer set). Built only at the requested ply, so a large visited set does not
    materialise its whole two-ply neighbourhood."""
    visited = {_key(game, s) for s in states}

    def children(parents):
        out = {}
        for s in parents:
            for a in game.legal_actions(s):
                c = game.step(s, a)
                if not game.is_terminal(c):
                    out.setdefault(_key(game, c), c)
        return out

    one = {k: c for k, c in children([s for s in states if game.ply(s) == ply - 1]).items() if k not in visited}
    mid = children([s for s in states if game.ply(s) == ply - 2])
    two = {k: g for k, g in children(list(mid.values())).items() if k not in visited and k not in one}
    return {"one_move_off": one, "two_moves_off": two}


def neighbourhood(game, states) -> dict:
    """Canonical keys of the visited `states` (V), of their non-terminal one-ply children not in V (C1), and of
    those children's non-terminal children in neither (C2)."""
    visited = {_key(game, s) for s in states}
    frontier = {}
    for s in states:
        for a in game.legal_actions(s):
            c = game.step(s, a)
            if not game.is_terminal(c):
                frontier.setdefault(_key(game, c), c)
    one = {k: c for k, c in frontier.items() if k not in visited}
    two = set()
    for c in one.values():
        for a in game.legal_actions(c):
            g = game.step(c, a)
            if not game.is_terminal(g):
                two.add(_key(game, g))
    two -= visited | set(one)
    return {"visited": visited, "one_move_off": set(one), "two_moves_off": two}


def distance_class(game, state, hood: dict) -> str:
    k = _key(game, state)
    if k in hood["visited"]:
        return VISITED
    if k in hood["one_move_off"]:
        return ONE_MOVE
    if k in hood["two_moves_off"]:
        return TWO_MOVES
    return FURTHER


def first_error(game, act_fn, root, defender, seed: int = 0, values_fn=None) -> dict:
    """Play `act_fn` for the side to move at `root` — which must hold a PROVEN win — against `defender` (an
    agent with `.act(game, state, rng)`, meant to be exact). At each of its moves the exact move values decide
    whether the move keeps the win; the first move that does not is the FIRST ERROR and ends the probe. Returns
    {"converted", "plies", and on an error "state", "move", "winning"}. `values_fn(state)` gives the mover's exact
    value per legal move (default `coverage.move_values`)."""
    values_fn = values_fn or (lambda s: move_values(game, s))
    vals = values_fn(root)
    if not vals or max(vals.values()) <= 0:
        raise ValueError("the root holds no proven win for the side to move — there is nothing to convert")
    rng = random.Random(seed)
    mover = game.current_player(root)
    state, plies = root, 0
    while not game.is_terminal(state):
        if game.current_player(state) == mover:
            vals = values_fn(state)
            move = act_fn(state)
            if vals[move] <= 0:
                return {"converted": False, "plies": plies, "state": state, "move": move,
                        "winning": sorted(a for a, v in vals.items() if v > 0)}
        else:
            move = defender.act(game, state, rng)
        state = game.step(state, move)
        plies += 1
    if game.returns(state)[mover] <= 0:
        raise ValueError("the policy kept a proven win at every move yet did not win — the value oracle and the "
                         "game disagree")
    return {"converted": True, "plies": plies}
