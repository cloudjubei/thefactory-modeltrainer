"""Scoring nets on ONE FIXED SET of exactly valued positions, and sizing a paired pilot from how much nets vary.

The 4-seed pilots read each net on its own first-player tree (~50 positions), so different nets were scored on
different positions and the same process scored 86% and 73% on two seed sets (h116). Here every net is scored on the
same positions — those the label cache records with every move's exact value — so arms are compared on equal ground
with thousands of readings per net. `calibration` reads how much nets of one arm vary and how much seed-matched
differences to base vary; `pairs_needed` turns that spread into the number of seed pairs a paired pilot needs."""
from __future__ import annotations

import math
from statistics import NormalDist, mean, stdev

from games.connect4 import C4State


def fixed_positions(rows: list, game, player: int, plies: list) -> list:
    """(state, ply, {move: exact value}) for every recorded position of `player` to move at one of `plies` whose
    every legal move has a recorded value and where some move is not optimal."""
    out = []
    for row in rows:
        s = C4State(tuple(row["board"]), row["to_move"], None, False)
        ply = sum(1 for c in s.board if c)
        values = {int(a): int(v) for a, v in row["values"].items()}
        if s.to_move != player or ply not in plies or game.is_terminal(s):
            continue
        if set(values) == set(game.legal_actions(s)) and len(set(values.values())) > 1:
            out.append((s, ply, values))
    return out


def score(moves: list, cases: list) -> dict:
    """How many of a net's raw moves (one per case) are optimal, overall and by ply."""
    if len(moves) != len(cases):
        raise ValueError(f"score needs one move per position, got {len(moves)} for {len(cases)}")
    by_ply: dict = {}
    optimal = 0
    for m, (_s, ply, values) in zip(moves, cases):
        right = values[m] == max(values.values())
        optimal += right
        slot = by_ply.setdefault(str(ply), {"positions": 0, "optimal": 0})
        slot["positions"] += 1
        slot["optimal"] += right
    return {"positions": len(cases), "optimal": optimal, "by_ply": by_ply}


def pairs_needed(sd: float, effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Seed pairs for a one-sided paired test at `alpha` to detect a mean difference `effect` with `power`, when
    seed-matched differences have standard deviation `sd` (normal approximation)."""
    if effect <= 0:
        raise ValueError(f"the effect to detect must be positive, got {effect}")
    z = NormalDist().inv_cdf(1 - alpha) + NormalDist().inv_cdf(power)
    return max(1, math.ceil((z * sd / effect) ** 2))


def calibration(nets: list) -> dict:
    """Each run/arm's shares, mean and spread; the seed-matched differences of every treatment arm to its run's base;
    and the pooled within-arm spread."""
    groups: dict = {}
    for n in nets:
        groups.setdefault(f"{n['run']}/{n['arm']}", {})[n["seed"]] = n["optimal"] / n["positions"]
    arms = {k: {"shares": [v[s] for s in sorted(v)], "mean": mean(v.values()),
                "sd": stdev(v.values()) if len(v) > 1 else 0.0} for k, v in groups.items()}
    paired = {}
    for key, shares in groups.items():
        run, arm = key.split("/")
        base = groups.get(f"{run}/base")
        if arm == "base" or base is None:
            continue
        diffs = [shares[s] - base[s] for s in sorted(shares) if s in base]
        paired[key] = {"differences": diffs, "mean": mean(diffs), "sd": stdev(diffs) if len(diffs) > 1 else 0.0}
    dof = sum(len(v) - 1 for v in groups.values())
    pooled = math.sqrt(sum((len(v) - 1) * arms[k]["sd"] ** 2 for k, v in groups.items()) / dof) if dof else 0.0
    return {"arms": arms, "paired": paired, "within_arm_sd": pooled}
