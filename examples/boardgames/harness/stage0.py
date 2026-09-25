"""§C.47 leg C, Stage 0 — does the DIRECT sibling channel have headroom on Connect-4? Pre-registered go/no-go.

On tic-tac-toe, one-ply siblings cut raw-policy failures ~10x (§C.46 E1), ~89% of it by putting positions into
training that self-play never produced (h38). On Connect-4 that direct channel can only matter where two things
hold at the positions siblings would add — exactly `neural.one_ply_siblings` of the final relabel buffer:

  E (exposure gap)   the net errs at those sibling positions more often than at the buffer positions it trained on;
  L (label headroom) the label siblings would train on there — the recipe's own 64-sim relabel search from the
                     final net — is right more often than the net's raw policy.

Both are measured per net on non-trivial, failable positions (no win-in-one, not every move of one value) at plies
16, 17, 19, 20, 22 and 23 — three for each mover, since siblings exist above all for the second player's positions after a
first-player deviation (h30) — per ply, then averaged over plies with equal weight so an uneven sample cannot make a
gap. The rule is an intersection-union test with the NET as the unit (five nets, t on 4 degrees of freedom):

  GO     both E and L have means >= C0_SESOI and one-sided 95% lower bounds above zero;
  STOP   either one-sided 95% upper bound falls below C0_SESOI;
  UNDECIDED otherwise — recorded, with no second look.

Calibration, computed before any Connect-4 data from §C.46's R200 arm (no siblings): the tic-tac-toe exposure gap
was 0.042 (0.0006 on trained keys, 0.042 one move off), per-seed sd ~0.019 — the channel that cut failures ~10x.
With C0_SAMPLING's sizes a simulation of the rule (binomial sampling per cell, that between-net spread) reads that
signature GO 0.93 of the time; a true zero gap reads GO 0.02 and STOP 0.47-0.69. A 0.05 bar would have called the
tic-tac-toe signature STOP.

A STOP says the direct channel has no headroom one move off the nets' own data. It says nothing about the
generalisation (spillover) route h38 measured at ~1 failure/seed on tic-tac-toe, which this stage does not test.
Any integrity failure, or evidence the report cannot read, is NOT_RUN — never a verdict."""
from __future__ import annotations

from harness.measurement import t_critical

C0_SEEDS = (201, 202, 203, 204, 205)
C0_ERA = "e974b3f9409d"
C0_MEASUREMENT_FP = "541fe66c6ad1"
C0_PLIES = (16, 17, 19, 20, 22, 23)
C0_DECISION = ("final_buffer", "sibling")
C0_CLASSES = ("final_buffer", "sibling", "evicted", "one_move_off", "two_moves_off", "random")
C0_SAMPLING = {"final_buffer": 250, "sibling": 300, "evicted": 60, "one_move_off": 60, "two_moves_off": 60,
               "random": 60, "probe_per_ply": 50, "probe_seed": 0, "conversion_roots": 16}
C0_MIN_PER_CELL = 80
C0_SESOI = 0.02
C0_RECIPE = {"game": "connect4", "iterations": 14, "selfplay_games": 48, "sims": 64,
             "arch": {"channels": 32, "blocks": 3, "head_hidden": 32, "residual": True}, "params": 60555,
             "opening_plies": 4, "opening_zero_frac": 0.3, "reanalyze_frac": 1.0, "reanalyze_sims": 64,
             "epochs": 6, "batch_size": 64, "lr": 0.001, "buffer_cap": 8000, "gumbel": True, "c_scale": 0.1,
             "augment": True, "label_sims": 64}


def _seed_problems(row: dict) -> list:
    where = f"net {row.get('seed')}"
    problems = []
    history = row.get("history")
    if not isinstance(history, list) or len(history) != C0_RECIPE["iterations"]:
        problems.append(f"{where}: not {C0_RECIPE['iterations']} training iterations")
    elif row.get("recorded_states") != sum(h.get("selfplay_states", -1) for h in history):
        problems.append(f"{where}: the recorder saw {row.get('recorded_states')} self-play states, training made "
                        f"{sum(h.get('selfplay_states', -1) for h in history)}")
    if row.get("games_per_pass") != [C0_RECIPE["selfplay_games"]] * C0_RECIPE["iterations"]:
        problems.append(f"{where}: the recorder did not see {C0_RECIPE['selfplay_games']} games on every iteration")
    if row.get("solver_forbidden") is not True:
        problems.append(f"{where}: training was not run with the solver forbidden")
    if row.get("params") != C0_RECIPE["params"]:
        problems.append(f"{where}: built {row.get('params')} parameters")
    if isinstance(history, list) and history and isinstance(row.get("sizes"), dict) \
            and history[-1].get("state_buffer") != row["sizes"].get("final_buffer"):
        problems.append(f"{where}: the final buffer the recorder rebuilt is not the one training relabelled")
    positions = row.get("positions")
    if not isinstance(positions, list):
        return problems + [f"{where}: no probed positions"]
    for ply in C0_PLIES:
        for cls in C0_DECISION:
            n = sum(1 for p in positions if isinstance(p, dict) and p.get("cls") == cls and p.get("ply") == ply)
            if n < C0_MIN_PER_CELL:
                problems.append(f"{where}: {n} {cls} positions at ply {ply}, fewer than {C0_MIN_PER_CELL}")
    return problems


def integrity(evidence: dict) -> list:
    """Every reason the evidence is not the registered Stage 0. Returns the problems; never raises."""
    if not isinstance(evidence, dict) or not isinstance(evidence.get("seeds"), list) \
            or not isinstance(evidence.get("config"), dict):
        return ["the evidence has no seeds or config"]
    problems = []
    if evidence.get("training_fingerprint") != C0_ERA:
        problems.append(f"training era {evidence.get('training_fingerprint')}, registered {C0_ERA}")
    if evidence.get("measurement_fingerprint") != C0_MEASUREMENT_FP:
        problems.append(f"measurement code {evidence.get('measurement_fingerprint')}, registered {C0_MEASUREMENT_FP}")
    for field, value in {**C0_RECIPE, "sampling": C0_SAMPLING, "plies": list(C0_PLIES)}.items():
        if evidence["config"].get(field) != value:
            problems.append(f"config {field} is {evidence['config'].get(field)!r}, registered {value!r}")
    seeds = sorted(r.get("seed") for r in evidence["seeds"] if isinstance(r, dict))
    if seeds != list(C0_SEEDS):
        problems.append(f"nets are seeds {seeds}, not the registered {list(C0_SEEDS)}")
    for row in evidence["seeds"]:
        problems.extend(_seed_problems(row) if isinstance(row, dict) else ["a net row is not a record"])
    return problems


def _rate(positions: list, cls: str, ply: int, field: str) -> float:
    rows = [p[field] for p in positions if p["cls"] == cls and p["ply"] == ply]
    return sum(1 for ok in rows if ok) / len(rows)


def _ply_gaps(positions: list, ply: int) -> dict:
    err_final = 1 - _rate(positions, "final_buffer", ply, "raw_ok")
    err_sib = 1 - _rate(positions, "sibling", ply, "raw_ok")
    label_sib = _rate(positions, "sibling", ply, "label_ok")
    return {"err_final_buffer": err_final, "err_sibling": err_sib, "label_ok_sibling": label_sib,
            "E": err_sib - err_final, "L": label_sib - (1 - err_sib)}


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def net_gaps(row: dict) -> dict:
    """One net's exposure gap and label headroom: per ply, then averaged over the plies with equal weight — and
    separately for each mover (even plies: the first player to move; odd: the second)."""
    per_ply = {ply: _ply_gaps(row["positions"], ply) for ply in C0_PLIES}
    movers = {"first": [p for p in C0_PLIES if p % 2 == 0], "second": [p for p in C0_PLIES if p % 2 == 1]}
    return {"seed": row["seed"], "E": _mean(g["E"] for g in per_ply.values()),
            "L": _mean(g["L"] for g in per_ply.values()),
            "by_mover": {m: {k: _mean(per_ply[p][k] for p in plies) for k in ("E", "L")} for m, plies in movers.items()},
            "per_ply": {str(p): g for p, g in per_ply.items()}}


def _across(values: list) -> dict:
    n = len(values)
    mean = sum(values) / n
    se = (sum((v - mean) ** 2 for v in values) / (n - 1) / n) ** 0.5
    t = t_critical(n - 1, two_sided_p=0.10)
    return {"mean": mean, "se": se, "lower": mean - t * se, "upper": mean + t * se, "per_net": values}


def _cells(rows: list) -> dict:
    out = {}
    for cls in C0_CLASSES:
        for ply in C0_PLIES:
            cell = [p for r in rows for p in r["positions"] if p["cls"] == cls and p["ply"] == ply]
            if cell:
                out[f"{cls}@{ply}"] = {"n": len(cell), **{f: sum(1 for p in cell if p[f]) / len(cell)
                                                          for f in ("raw_ok", "avg_ok", "label_ok")}}
    return out


def stage0_report(evidence: dict) -> dict:
    """The pre-registered Stage 0 verdict (go / stop / undecided / not_run) and its descriptives."""
    try:
        problems = integrity(evidence)
        if not problems:
            return _judged(evidence)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, "C0": {"verdict": "not_run"}}


def _judged(evidence: dict) -> dict:
    rows = sorted(evidence["seeds"], key=lambda r: r["seed"])
    gaps = [net_gaps(r) for r in rows]
    e, lab = (_across([g[k] for g in gaps]) for k in ("E", "L"))
    detail = (f"exposure gap {e['mean']:.3f} [{e['lower']:.3f}, {e['upper']:.3f}], label headroom {lab['mean']:.3f} "
              f"[{lab['lower']:.3f}, {lab['upper']:.3f}] (one-sided 95% bounds, net as unit)")
    if min(e["lower"], lab["lower"]) > 0 and min(e["mean"], lab["mean"]) >= C0_SESOI:
        verdict = "go"
        reading = (f"The direct sibling channel has headroom on Connect-4: one move off its own data the net errs more "
                   f"often than on its final buffer, and the 64-sim label siblings would train on is right more often "
                   f"than the net there — {detail}. Stage 1 is worth running; this is not evidence that siblings raise "
                   f"conversion.")
    elif min(e["upper"], lab["upper"]) < C0_SESOI:
        binding = "exposure gap" if e["upper"] <= lab["upper"] else "label headroom"
        verdict = "stop"
        reading = (f"No headroom for the direct sibling channel one move off the nets' own data (binding: the "
                   f"{binding}) — {detail}. The generalisation (spillover) route is untested by this stage.")
    else:
        verdict = "undecided"
        reading = f"Undecided — {detail}."
    return {"integrity": [],
            "C0": {"verdict": verdict, "E": e, "L": lab, "sesoi": C0_SESOI, "reading": reading},
            "descriptives": {"nets": gaps, "cells": _cells(rows),
                             "by_mover": {m: {k: _across([g["by_mover"][m][k] for g in gaps]) for k in ("E", "L")}
                                          for m in ("first", "second")},
                             "plateau": {r["seed"]: r.get("probe") for r in rows},
                             "containment": {r["seed"]: r.get("containment") for r in rows},
                             "conversion": {r["seed"]: r.get("conversion") for r in rows}}}
