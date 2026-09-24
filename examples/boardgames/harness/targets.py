"""§C.45 WHERE A SELF-PLAY TARGET GOES WRONG — split one search into prior, search and target.

A self-play training target is the Gumbel completed-Q policy `softmax(log prior + (c_visit + maxN)·c_scale·q̂)`.
When that target names a losing move there are two different reasons, and they need different fixes:

  search_miss   the search never found the better move: no action it rates best is optimal. More search, or a
                better value head, is the lever.
  prior_anchor  the search DID find it (every action tied for the best completed Q is optimal) but the target
                still points elsewhere, because `c_scale` lets a confident prior outweigh the Q gap. More search
                only helps here through maxN; the direct lever is how far the target trusts Q.

§C.44 showed the tic-tac-toe blind spot survives the net SEEING the states, so the label is the next suspect;
this is the instrument that says which half of the label is wrong. It reads an AlphaZeroAgent's root node and is
not part of the training path.

§C.46 adds the solver-side pieces the training loop must never import itself: the exact label an oracle arm
trains on (`exact_policy_target`, named in a config through `POLICY_TARGETS`), and a per-pass recorder of what
each training pass was fed and what it produced (`record_training_passes`)."""
from __future__ import annotations

import hashlib
import math
import random
from contextlib import contextmanager

import torch

from harness.agents import state_key
from harness.coverage import _as_bytes, _key, label_dose, optimal_actions, raw_encoding_lookup
from harness.neural import completed_q_values


def search_decomposition(game, make_agent, state, seed: int = 0, temperature: float = 0.0) -> dict:
    """Run ONE search from a fresh agent and report what each stage chose. `temperature` > 0 turns on the root
    Gumbel noise self-play uses (tic-tac-toe self-play searches with it on nearly every move); `seed` fixes it."""
    agent = make_agent()
    agent.temperature = temperature
    pi = agent.run_search(game, state, random.Random(seed))
    root = agent._nodes[state_key(game, state)]
    q, _sum_n, _max_n = completed_q_values(root.prior, root.child_n, root.child_w, root.value, root.legal)
    best = max(q.values())
    return {"prior": max(root.legal, key=lambda a: root.prior[a]),
            "q_best": sorted(a for a in root.legal if q[a] == best),
            "label": max(pi, key=pi.get),
            "selected": agent._gumbel_selected if agent.gumbel else None,
            "q": dict(q), "visits": dict(root.child_n)}


def target_error(decomposition: dict, optimal) -> str:
    """`label_ok`, `prior_anchor` (search found an optimal move, the target did not follow it) or `search_miss`
    (the search's best-rated moves include a non-optimal one — a tie is only credited when EVERY tied move is
    optimal, so a search that cannot tell the win from the blunder is not said to have found it)."""
    opt = set(optimal)
    if decomposition["label"] in opt:
        return "label_ok"
    return "prior_anchor" if set(decomposition["q_best"]) <= opt else "search_miss"


def exact_policy_target(game, state) -> list[float]:
    """The label a PERFECT relabeler would write: uniform over the solver's optimal set (`optimal_actions`), zero on
    every other action, `game.num_actions` wide. Uniform because every optimal move is equally correct — putting
    the mass on one would teach an arbitrary tie-break as if it were knowledge. An arm trained on it reads the
    solver inside training, so it is a diagnostic ceiling on what label quality alone can buy, never a method. A
    terminal state has no move to label and an all-zero vector is not a distribution, so it is refused."""
    if game.is_terminal(state):
        raise ValueError("the exact policy target is only defined where the mover has a move — this state is "
                         "terminal, so there is nothing to label")
    opt = optimal_actions(game, state)
    return [1.0 / len(opt) if a in opt else 0.0 for a in range(game.num_actions)]


# A run's config names its policy target as a string, so the config stays JSON and still says which oracle trained it.
POLICY_TARGETS = {"exact_uniform_optimal": exact_policy_target}


@contextmanager
def record_training_labels(game, target_keys, encode_fn):
    """While active, every call to `harness.neural.train_net` first measures the policy labels it is about to train
    on at the target states (`coverage.label_dose`) and appends one row per training pass to the yielded list:
    `pass`, `examples`, `n_target`, `argmax_ok`, `opt_mass`, `per_key` (a list of [key, counts], so integer keys
    survive a JSON round trip). It observes only — the call it wraps is
    made with the same arguments — and the original is restored however the block exits. This is the §C.45
    delivered dose: the labels the net actually trained on, the same measure in every arm, where a fresh search
    from the final net would measure the failure itself."""
    import harness.neural as neural

    lookup = raw_encoding_lookup(game, encode_fn)
    targets = set(target_keys)
    log: list = []
    original = neural.train_net

    def watched(net, examples, *args, **kwargs):
        d = label_dose(game, examples, encode_fn, targets, lookup=lookup)
        log.append({"pass": len(log) + 1, "examples": len(examples), "n_target": d["n_target"],
                    "argmax_ok": d["argmax_ok"], "opt_mass": d["opt_mass"],
                    "per_key": [[k, row] for k, row in sorted(d["per_key"].items())]})
        return original(net, examples, *args, **kwargs)

    neural.train_net = watched
    try:
        yield log
    finally:
        neural.train_net = original


def _weights_sha(net) -> str:
    """sha256 over the bytes of every state_dict tensor, in the state_dict's own key order — so "these two arms
    share pass 1" is a checked fact about the weights rather than an assumption about the code path."""
    h = hashlib.sha256()
    for t in net.state_dict().values():
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def record_training_passes(game, states, encode_fn):
    """While active, every call to `harness.neural.train_net` appends one row to the yielded list:

      pass              1, 2, ... in call order.
      dose              BEFORE the pass: [key, sp_n, sp_ok, sib_n, sib_ok] for EVERY canonical key of `states`,
                        sorted, an untrained key reading zeros rather than going missing. An example whose value
                        target is NaN is a sibling row (§C.46: labelled by search, no game outcome behind it), any
                        other is self-play. `ok` is the label's argmax lying in the optimal set of the RAW position
                        its encoding maps to, so an augmented image is graded in its own frame.
      weights_sha       AFTER the pass: `_weights_sha(net)`.
      policy_fail_keys  AFTER the pass: the sorted keys of `states` whose raw-policy argmax, masked to legal moves,
                        is not optimal — one batched forward in eval mode (a train-mode forward would move BatchNorm
                        statistics the next pass trains from), the net then put back in the mode the pass left it.

    It observes only: the wrapped call gets the same arguments, its result is returned, and nothing here draws from
    any RNG, so a recorded run trains bit-identically to an unrecorded one. It nests with `record_training_labels`
    in either order — whichever is entered second wraps the first's wrapper — and the function it replaced is
    restored however the block exits. What cannot change between passes (the encoding lookup, the states'
    encodings, legal masks and optimal sets) is built once, on entry."""
    import harness.neural as neural

    lookup = raw_encoding_lookup(game, encode_fn)
    keys = [_key(game, s) for s in states]
    x = torch.stack([encode_fn(game, s) for s in states])
    legal = torch.zeros(len(states), game.num_actions, dtype=torch.bool)
    for i, s in enumerate(states):
        legal[i, game.legal_actions(s)] = True
    optimal: dict = {}

    def optimal_at(s):
        sk = game.state_key(s)
        if sk not in optimal:
            optimal[sk] = optimal_actions(game, s)
        return optimal[sk]

    state_optimal = [optimal_at(s) for s in states]
    log: list = []
    original = neural.train_net

    def dose(examples) -> list:
        rows = {k: [0, 0, 0, 0] for k in keys}
        for e in examples:
            s = lookup.get(_as_bytes(e[0]))
            if s is None:
                continue
            row = rows.get(_key(game, s))
            if row is None:
                continue
            pi = list(e[1])
            top = max(range(len(pi)), key=lambda a: pi[a])
            col = 2 if math.isnan(float(e[2])) else 0
            row[col] += 1
            row[col + 1] += 1 if top in optimal_at(s) else 0
        return [[k, *rows[k]] for k in sorted(rows)]

    def policy_fail_keys(net) -> list:
        mode = net.training
        net.eval()
        try:
            with torch.no_grad():
                logits, _v = net(x)
        finally:
            net.train(mode)
        moves = logits.masked_fill(~legal, float("-inf")).argmax(dim=1).tolist()
        return sorted(k for k, m, opt in zip(keys, moves, state_optimal) if m not in opt)

    def watched(net, examples, *args, **kwargs):
        fed = dose(examples)
        result = original(net, examples, *args, **kwargs)
        log.append({"pass": len(log) + 1, "dose": fed, "weights_sha": _weights_sha(net),
                    "policy_fail_keys": policy_fail_keys(net)})
        return result

    neural.train_net = watched
    try:
        yield log
    finally:
        neural.train_net = original
