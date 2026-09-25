"""§C.47 leg C, Stage 0 v2 — the pre-registered go/no-go re-run after v1 (h43) read NOT_RUN.

v1 (harness/stage0.py, pinned to h43) compared the net's errors at `one_ply_siblings(final buffer)` with its errors
on a census of the final relabel buffer. Its integrity gate refused the run: after the non-trivial filter the final
buffer held 44-78 positions at 16 of 30 (net, ply) cells, under the registered 80. The gate was right — a thin cell
is exactly how a gap is manufactured — and no E or L was computed. v2 changes only what the gate exposed, on FRESH
nets (seeds 206-210):

  TRAINED CELL  every position the net trained on at any iteration (the visited set), not only the final buffer:
                the reference the tic-tac-toe calibration itself used (trained on any pass), and ~1.5-2x the size.
                Whether a position is still in the final buffer is recorded on it, descriptively.
  MINIMUM       60 per decision cell (v1's counts, re-read with the visited set: 70-234 per cell).
  SAMPLING      siblings capped at 250 per ply; the descriptive rings and random positions at 30; conversion probes
                at 6 roots per kind per ply — v1's measurement took ~7.5 h, almost all of it exact solves.

The rule is v1's: E = error rate at the siblings minus at the trained positions, L = label accuracy minus raw
accuracy at the siblings, both per ply and averaged with equal weight; an intersection-union test with the net as
the unit — GO iff both means >= C0_SESOI and both one-sided 95% lower bounds > 0, STOP iff either upper bound is
below C0_SESOI, UNDECIDED otherwise. A simulation of the rule at v1's realised cell sizes (binomial sampling, raw
error ~0.45, the tic-tac-toe between-net spread) reads the tic-tac-toe signature GO 0.81 of the time and a true
zero gap GO 0.03-0.04, STOP 0.35-0.46. A STOP says the direct channel has no headroom one move off the nets' own
data; the generalisation route is untested. Any integrity failure, or unreadable evidence, is NOT_RUN."""
from __future__ import annotations

from harness.stage0 import C0_ERA, C0_PLIES, C0_RECIPE, C0_SESOI, _across

C0V2_SEEDS = (206, 207, 208, 209, 210)
C0V2_MEASUREMENT_FP = "1b39654595cb"
C0V2_DECISION = ("visited", "sibling")
C0V2_CLASSES = ("visited", "sibling", "one_move_off", "two_moves_off", "random")
C0V2_SAMPLING = {"visited": 300, "sibling": 250, "one_move_off": 30, "two_moves_off": 30, "random": 30,
                 "probe_per_ply": 50, "probe_seed": 0, "conversion_roots": 6}
C0V2_MIN_PER_CELL = 60


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
        for cls in C0V2_DECISION:
            n = sum(1 for p in positions if isinstance(p, dict) and p.get("cls") == cls and p.get("ply") == ply)
            if n < C0V2_MIN_PER_CELL:
                problems.append(f"{where}: {n} {cls} positions at ply {ply}, fewer than {C0V2_MIN_PER_CELL}")
    return problems


def integrity(evidence: dict) -> list:
    """Every reason the evidence is not the registered v2 Stage 0. Returns the problems; never raises."""
    if not isinstance(evidence, dict) or not isinstance(evidence.get("seeds"), list) \
            or not isinstance(evidence.get("config"), dict):
        return ["the evidence has no seeds or config"]
    problems = []
    if evidence.get("training_fingerprint") != C0_ERA:
        problems.append(f"training era {evidence.get('training_fingerprint')}, registered {C0_ERA}")
    if evidence.get("measurement_fingerprint") != C0V2_MEASUREMENT_FP:
        problems.append(f"measurement code {evidence.get('measurement_fingerprint')}, registered {C0V2_MEASUREMENT_FP}")
    for field, value in {**C0_RECIPE, "sampling": C0V2_SAMPLING, "plies": list(C0_PLIES), "design": "v2"}.items():
        if evidence["config"].get(field) != value:
            problems.append(f"config {field} is {evidence['config'].get(field)!r}, registered {value!r}")
    seeds = sorted(r.get("seed") for r in evidence["seeds"] if isinstance(r, dict))
    if seeds != list(C0V2_SEEDS):
        problems.append(f"nets are seeds {seeds}, not the registered {list(C0V2_SEEDS)}")
    for row in evidence["seeds"]:
        problems.extend(_seed_problems(row) if isinstance(row, dict) else ["a net row is not a record"])
    return problems


def _rate(positions: list, cls: str, ply: int, field: str) -> float:
    rows = [p[field] for p in positions if p["cls"] == cls and p["ply"] == ply]
    return sum(1 for ok in rows if ok) / len(rows)


def _ply_gaps(positions: list, ply: int) -> dict:
    err_trained = 1 - _rate(positions, "visited", ply, "raw_ok")
    err_sib = 1 - _rate(positions, "sibling", ply, "raw_ok")
    label_sib = _rate(positions, "sibling", ply, "label_ok")
    return {"err_visited": err_trained, "err_sibling": err_sib, "label_ok_sibling": label_sib,
            "E": err_sib - err_trained, "L": label_sib - (1 - err_sib)}


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def net_gaps(row: dict) -> dict:
    """One net's exposure gap and label headroom: per ply, averaged over the plies with equal weight, and
    separately for each mover (even plies: the first player to move; odd: the second)."""
    per_ply = {ply: _ply_gaps(row["positions"], ply) for ply in C0_PLIES}
    movers = {"first": [p for p in C0_PLIES if p % 2 == 0], "second": [p for p in C0_PLIES if p % 2 == 1]}
    return {"seed": row["seed"], "E": _mean(g["E"] for g in per_ply.values()),
            "L": _mean(g["L"] for g in per_ply.values()),
            "by_mover": {m: {k: _mean(per_ply[p][k] for p in plies) for k in ("E", "L")} for m, plies in movers.items()},
            "per_ply": {str(p): g for p, g in per_ply.items()}}


def _cells(rows: list) -> dict:
    out = {}
    for cls in C0V2_CLASSES:
        for ply in C0_PLIES:
            cell = [p for r in rows for p in r["positions"] if p["cls"] == cls and p["ply"] == ply]
            if cell:
                out[f"{cls}@{ply}"] = {"n": len(cell), **{f: sum(1 for p in cell if p[f]) / len(cell)
                                                          for f in ("raw_ok", "avg_ok", "label_ok")}}
    final = [p for r in rows for p in r["positions"] if p["cls"] == "visited" and p.get("in_final")]
    evicted = [p for r in rows for p in r["positions"] if p["cls"] == "visited" and not p.get("in_final")]
    for name, cell in (("visited_in_final_buffer", final), ("visited_evicted", evicted)):
        if cell:
            out[name] = {"n": len(cell), "raw_ok": sum(1 for p in cell if p["raw_ok"]) / len(cell)}
    return out


def stage0_v2_report(evidence: dict) -> dict:
    """The pre-registered v2 Stage 0 verdict (go / stop / undecided / not_run) and its descriptives."""
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
        reading = (f"The direct sibling channel has headroom on Connect-4: at the positions siblings would add the net "
                   f"errs more often than at positions it trained on, and the 64-sim label siblings would train on is "
                   f"right more often than the net there — {detail}. Stage 1 is worth running; this is not evidence "
                   f"that siblings raise conversion.")
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
