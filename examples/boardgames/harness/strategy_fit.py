"""§C.49 — the ORACLE FRONTIER for a game too large to enumerate (Connect-4). The target cannot be every position,
and the positions that matter depend on the net: to play perfectly from the start, a net must be right exactly at the
positions its OWN strategy reaches against every reply. So the target is grown from the net's own strategy tree,
round by round:

  1. walk the tree — the net's single move at each of its positions, every legal reply at the other side's — to the
     horizon, labelling each of its positions with the exact best moves (cached across rounds);
  2. where its move is not among the best, record a FAILURE and do not follow the move;
  3. retrain a fresh net on every position labelled so far, until it holds them all;
  4. repeat until a round's walk finds no failure — the net is then certified through the horizon (harness.certify
     would agree) — or the rounds run out.

The solver is the TEACHER here, which is allowed: this measures what a net can REPRESENT (the smallest net that holds
a perfect strategy to depth D), not what a solver-free process finds."""
from __future__ import annotations

import random
from typing import Callable


def expand_round(game, root, player: int, choose: Callable[[list], list], move_values_fn: Callable, depth, known: dict,
                 move_values_many: Callable[[list], list] | None = None,
                 check_many: Callable[[list], list] | None = None) -> dict:
    """One walk of `player`'s strategy tree from `root` to `depth` plies (None: the whole game). `known` caches exact
    move values by state key across rounds; `move_values_many`, when given, receives each ply's unknown positions in
    one batch.

    `check_many` ((position, move) pairs → the value each move keeps) makes labelling CHECK-FIRST: the chosen move is
    checked with one solve, and when it keeps a WIN — nothing is better — that move alone is the label; only a move
    that does not win costs the full label (every move's value). `known` then holds partial entries ({move: 1}) that
    grow as later rounds check other moves. Returns {"labelled": {key: (state, best moves, value)} for every `player` position walked,
    "failures": keys where the chosen move was not best, "failures_by_depth", "complete" (nothing left beyond the
    horizon), "nodes": {"player", "opponent"}}."""
    rng = random.Random(0)
    level = {game.state_key(root): root}
    labelled: dict = {}
    failures, failures_by_depth = [], {}
    nodes = {"player": 0, "opponent": 0}
    ply = 0
    while level:
        live = {k: s for k, s in level.items() if not game.is_terminal(s)}
        if depth is not None and ply >= depth:
            return {"labelled": labelled, "failures": failures, "failures_by_depth": failures_by_depth,
                    "complete": not live, "nodes": nodes}
        mine = {k: s for k, s in live.items() if game.current_player(s) == player}
        theirs = [s for s in live.values() if game.current_player(s) != player]
        nodes["player"] += len(mine)
        nodes["opponent"] += len(theirs)
        keys = list(mine)
        chosen = choose([mine[k] for k in keys]) if keys else []
        if check_many is not None:
            ask = [(k, a) for k, a in zip(keys, chosen, strict=True) if a not in known.get(k, {})
                   and a in game.legal_actions(mine[k])]
            kept = check_many([(mine[k], a) for k, a in ask]) if ask else []
            for (k, a), v in zip(ask, kept, strict=True):
                if v == 1:
                    known.setdefault(k, {})[a] = 1
        unknown = [k for k, a in zip(keys, chosen, strict=True) if a not in known.get(k, {})
                   or (check_many is None and k not in known)]
        if unknown:
            solved = (move_values_many([mine[k] for k in unknown]) if move_values_many
                      else [move_values_fn(mine[k]) for k in unknown])
            known.update(zip(unknown, solved, strict=True))
        nxt: dict = {}
        for key, a in zip(keys, chosen, strict=True):
            vals = known[key]
            value = max(vals.values())
            best = sorted(m for m, v in vals.items() if v == value)
            labelled[key] = (mine[key], best, value)
            if a not in best:
                failures.append(key)
                failures_by_depth[ply] = failures_by_depth.get(ply, 0) + 1
                continue
            child = game.step(mine[key], a, rng)
            nxt.setdefault(game.state_key(child), child)
        for s in theirs:
            for b in game.legal_actions(s):
                child = game.step(s, b, rng)
                nxt.setdefault(game.state_key(child), child)
        level = nxt
        ply += 1
    return {"labelled": labelled, "failures": failures, "failures_by_depth": failures_by_depth, "complete": True,
            "nodes": nodes}


def strategy_target(game, labelled: dict) -> dict:
    """The labelled positions as a harness.frontier target: policy uniform over the best moves, value the exact value,
    graded by whether the argmax is a best move. Trained and graded on the same positions — the question is whether
    the net can HOLD them."""
    import torch

    from harness.neural import encode

    n = game.num_actions
    rows = list(labelled.values())
    legal = torch.zeros(len(rows), n, dtype=torch.bool)
    best = torch.zeros(len(rows), n, dtype=torch.bool)
    for i, (s, moves, _v) in enumerate(rows):
        legal[i, game.legal_actions(s)] = True
        best[i, moves] = True
    x = torch.stack([encode(game, s) for s, _m, _v in rows])
    value = torch.tensor([[float(v)] for _s, _m, v in rows])
    return {"allowed_failures": 0, "train": {"x": x, "legal": legal, "policy": best.float() / best.sum(1, keepdim=True),
                                             "value": value},
            "eval": {"x": x, "legal": legal, "optimal": best}}


def net_chooser(game, net) -> Callable[[list], list]:
    """The net's RAW move: argmax over the legal moves of one forward pass, batched."""
    import torch

    from harness.neural import encode

    def choose(states: list) -> list:
        out = []
        net.eval()
        for i in range(0, len(states), 4096):
            chunk = states[i:i + 4096]
            legal = torch.zeros(len(chunk), game.num_actions, dtype=torch.bool)
            for j, s in enumerate(chunk):
                legal[j, game.legal_actions(s)] = True
            with torch.no_grad():
                logits, _v = net(torch.stack([encode(game, s) for s in chunk]))
            out += logits.masked_fill(~legal, float("-inf")).argmax(dim=1).tolist()
        return out
    return choose


def fit_strategy(game, arch: dict, root, player: int, depth, seed: int, recipe: dict, rounds: int,
                 move_values_fn: Callable, known: dict, move_values_many: Callable[[list], list] | None = None,
                 check_many: Callable[[list], list] | None = None) -> dict:
    """Grow a net of `arch` to a perfect strategy through `depth`: alternate walks of its own tree and refits on
    everything labelled so far. A round that labels nothing new after a refit that could NOT hold its data would only
    repeat that refit exactly, so growth stops there as STALLED — the setup could not represent what it was taught.
    Returns {"certified", "stalled", "params", "positions" labelled in all, "rounds": [{"round",
    "failures", "failures_by_depth", "labelled" this walk, "positions" so far, "complete", "fit": the refit's
    readings or None}]}."""
    import torch

    from harness.frontier import fit
    from harness.neural import Connect4Net, arch_for_game

    torch.manual_seed(seed)
    net = Connect4Net(**arch_for_game(arch, game))
    data: dict = {}
    log = []
    certified = stalled = False
    last_fit_held = True
    for r in range(rounds):
        before = len(data)
        walk = expand_round(game, root, player, net_chooser(game, net), move_values_fn, depth, known, move_values_many,
                            check_many)
        data.update(walk["labelled"])
        entry = {"round": r, "failures": len(walk["failures"]), "failures_by_depth": walk["failures_by_depth"],
                 "labelled": len(walk["labelled"]), "positions": len(data), "complete": walk["complete"], "fit": None}
        log.append(entry)
        if not walk["failures"]:
            certified = True
            break
        if r > 0 and len(data) == before and not last_fit_held:
            stalled = True
            break
        if r == rounds - 1:
            break
        result = fit(game, arch, strategy_target(game, data), seed, recipe, return_net=True)
        net = result.pop("net")
        entry["fit"] = {k: result[k] for k in ("solved", "solved_at_epoch", "best_failures", "epochs_run")}
        last_fit_held = result["solved"]
    return {"certified": certified, "stalled": stalled, "params": sum(p.numel() for p in net.parameters()), "positions": len(data),
            "rounds": log}
