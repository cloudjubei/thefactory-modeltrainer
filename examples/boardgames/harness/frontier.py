"""§C.49 — the ORACLE FRONTIER as a repeatable process: the smallest net of each family that can hold a target
exactly, when it is taught the exact answers. The first step of the smallest-setup programme (§C.48): it says what
is REPRESENTABLE, before any question of whether a self-play process finds it.

A FAMILY is a net shape with one width parameter (`arch_at`), optionally with a standardised-orientation input. A
TARGET is the positions a net must get right, with their exact optimal moves: every non-terminal position of an
enumerable game (`enumerated_target`), or any other set built the same way. A width SUCCEEDS under a RECIPE when
every seed's fit reaches the target (no more failures than it allows). The search doubles the width until it
succeeds, then bisects down to the smallest succeeding width, and records every probe — success need not be
monotone in width, and a failure above the frontier is flagged, never hidden."""
from __future__ import annotations

from typing import Callable

BODIES = ("mlp", "mlp2", "conv", "residual")
DEFAULT_RECIPE = {"lr": 2e-3, "batch": 256, "max_epochs": 6000, "check_every": 25, "patience": 1500}


def arch_at(family: dict, width: int) -> dict:
    """The net arch of `family` ({"body", "canonical"}) at `width`."""
    body, w = family["body"], int(width)
    if body not in BODIES or w < 1:
        raise ValueError(f"family body {body!r} at width {width}: bodies are {BODIES}, widths >= 1")
    arch = {"mlp": {"mlp_hidden": [w]}, "mlp2": {"mlp_hidden": [w, w]}, "conv": {"channels": w},
            "residual": {"channels": w, "blocks": 1, "head_hidden": w, "residual": True}}[body]
    return {**arch, "canonical_input": True} if family.get("canonical") else arch


def family_name(family: dict) -> str:
    return ("canon_" if family.get("canonical") else "") + family["body"]


def smallest_width(probe: Callable[[int], bool], start: int = 2, cap: int = 1024, confirm_above: int = 2) -> dict:
    """The smallest width `probe` accepts: double from `start` to the first success, then bisect between the last
    failure and it, then probe the `confirm_above` widths just above it — bisection alone can never see a failure
    above its answer. Returns {"frontier": width or None when nothing up to `cap` succeeds, "probes": {width: ok},
    "non_monotone": probed widths above the frontier that failed}."""
    if start < 1 or cap < start:
        raise ValueError(f"start {start} and cap {cap} must satisfy 1 <= start <= cap")
    probes: dict = {}
    width, last_fail = start, 0
    while True:
        probes[width] = bool(probe(width))
        if probes[width]:
            break
        last_fail = width
        if width >= cap:
            return {"frontier": None, "probes": probes, "non_monotone": []}
        width = min(cap, width * 2)
    lo, hi = last_fail, width
    while hi - lo > 1:
        mid = (lo + hi) // 2
        probes[mid] = bool(probe(mid))
        lo, hi = (lo, mid) if probes[mid] else (mid, hi)
    for w in range(hi + 1, hi + 1 + confirm_above):
        if w not in probes:
            probes[w] = bool(probe(w))
    return {"frontier": hi, "probes": probes, "non_monotone": sorted(w for w, ok in probes.items() if w > hi and not ok)}


def enumerated_target(game, canonical: bool) -> dict:
    """Every non-terminal reachable position with its exact optimal moves. The net is graded on EVERY raw position;
    it trains on one row per position it can tell apart — per symmetry class for a standardised-orientation net,
    which sees every orientation as the same input."""
    import torch

    from harness.coverage import optimal_actions, reachable_states
    from harness.neural import encode

    raw, complete = reachable_states(game, exact=True, symmetry=False)
    if not complete:
        raise ValueError(f"{game.name} could not be enumerated — use a target built from a pre-solved sample")
    raw = [s for s in raw if not game.is_terminal(s)]
    n = game.num_actions
    optimal = [sorted(optimal_actions(game, s)) for s in raw]
    key = game.canonical_key if canonical else game.state_key
    first: dict = {}
    for i, s in enumerate(raw):
        first.setdefault(key(s), i)
    train = sorted(first.values())

    def legal(s):
        m = torch.zeros(n, dtype=torch.bool)
        m[game.legal_actions(s)] = True
        return m

    x = torch.stack([encode(game, s) for s in raw])
    legal_mask = torch.stack([legal(s) for s in raw])
    best = torch.zeros(len(raw), n, dtype=torch.bool)
    for i, opt in enumerate(optimal):
        best[i, opt] = True
    policy = best.float() / best.sum(dim=1, keepdim=True)
    value = torch.tensor([[float(game.position_value(raw[i]))] for i in train])
    idx = torch.tensor(train)
    return {"game": game.name, "positions": len(raw), "allowed_failures": 0,
            "train": {"x": x[idx], "legal": legal_mask[idx], "policy": policy[idx], "value": value},
            "eval": {"x": x, "legal": legal_mask, "optimal": best}}


def failures(net, ev: dict) -> int:
    """Positions whose raw move — the argmax over legal moves — is not optimal."""
    import torch

    was = net.training
    net.eval()
    with torch.no_grad():
        logits, _v = net(ev["x"])
    net.train(was)
    moves = logits.masked_fill(~ev["legal"], float("-inf")).argmax(dim=1)
    return int((~ev["optimal"][torch.arange(len(moves)), moves]).sum())


def fit(game, arch: dict, target: dict, seed: int, recipe: dict) -> dict:
    """Train a fresh net of `arch` on the target's exact answers until it holds the target (no more failures than
    allowed) or has not improved for `patience` epochs. Policy: cross-entropy to uniform over the optimal moves;
    value: squared error to the exact value."""
    import torch

    from harness.neural import Connect4Net, arch_for_game

    torch.manual_seed(seed)
    net = Connect4Net(**arch_for_game(arch, game))
    opt = torch.optim.Adam(net.parameters(), lr=recipe["lr"])
    tr = target["train"]
    n = len(tr["x"])
    gen = torch.Generator().manual_seed(seed)
    best, best_epoch, solved_at, curve = None, 0, None, []
    epoch = 0
    for epoch in range(1, recipe["max_epochs"] + 1):
        order = torch.randperm(n, generator=gen)
        for i in range(0, n, recipe["batch"]):
            j = order[i:i + recipe["batch"]]
            logits, v = net(tr["x"][j])
            logp = torch.log_softmax(logits.masked_fill(~tr["legal"][j], -1e9), dim=1)
            loss = -(tr["policy"][j] * logp).sum(1).mean() + ((v - tr["value"][j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % recipe["check_every"]:
            continue
        f = failures(net, target["eval"])
        curve.append([epoch, f])
        if best is None or f < best:
            best, best_epoch = f, epoch
        if f <= target["allowed_failures"]:
            solved_at = epoch
            break
        if epoch - best_epoch >= recipe["patience"]:
            break
    return {"seed": seed, "params": sum(p.numel() for p in net.parameters()), "solved": solved_at is not None,
            "solved_at_epoch": solved_at, "best_failures": best, "epochs_run": epoch, "curve": curve}


def probe_width(game, family: dict, width: int, target: dict, seeds: list, recipe: dict) -> dict:
    """Fit every seed at `width`; the width succeeds only if every seed does, so the first failing seed ends it."""
    runs = []
    for seed in seeds:
        runs.append(fit(game, arch_at(family, width), target, seed, recipe))
        if not runs[-1]["solved"]:
            break
    return {"width": width, "params": runs[0]["params"], "ok": all(r["solved"] for r in runs) and len(runs) == len(seeds),
            "runs": runs}
