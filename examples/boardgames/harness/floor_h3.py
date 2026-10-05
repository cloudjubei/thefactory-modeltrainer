"""§C.49 H3 — train the plies the net must carry. Pre-registered.

H2 (h132, h133) showed that training the net beside the exact opening table — self-play from the table's frontier,
the strategy tree walked from there — raises its share of optimal moves one ply past the table from 75.0% to 81.6%.
H2's tree covered only that ply (6); the hybrid also needs the net at ply 8. H3 walks the tree from the frontier
deep enough to cover plies 6 and 8 (`strategy_tree` depth 4), on H2's seeds, with the per-pass curve on. H2's own
table-arm nets are the control: the training code is byte-identical (the same training fingerprint), so a rerun
would reproduce them.

Two readings on fixed position sets the table makes the same for every net (scripts/c4_h3_readout.py):
  s8     the ply-8 positions reached when the first player also plays optimally at ply 6 — H3's seed-matched gain
         over H2 by an exact one-sided sign-flip permutation test at `alpha`: SUPPORTED when it rejects, REFUTED
         when the mean gain is at most zero, INCONCLUSIVE otherwise;
  curve  H3's share at the ply-6 positions after every pass, `late` passes against `early` ones pooled over the
         seeds: SUPPORTED (still rising) at a gain of at least `rising_at`, REFUTED (flat) at or below `flat_at`.
`s8_positions` None means the set's size is whatever the readout solved, but every net must be scored on it. The
ply-6 shares and each net's description size (a certified hybrid's table plus exceptions) are reported, not
judged. A readout that is not the registered measurement is NOT_RUN for both."""
from __future__ import annotations

from statistics import mean

from harness.floor_c4_powered import permutation_p
from harness.floor_h2 import SPEC as H2_SPEC

_H2 = H2_SPEC["arms"]["table"]
SPEC = {"arms": {"h2": _H2, "h3": {**_H2, "strategy_tree": {"player": 0, "depth": 4}}},
        "runs": {"h2": "c49_H2/table", "h3": "c49_H3/deep"}, "seeds": H2_SPEC["seeds"], "era": H2_SPEC["era"],
        "measurement_fp": "65225dd2ce54", "s6_positions": 284, "s8_positions": None, "iterations": 20,
        "early": [11, 12, 13, 14, 15], "late": [16, 17, 18, 19, 20], "rising_at": 0.015, "flat_at": 0.005,
        "alpha": 0.05}


def _problems(readout, spec: dict) -> list:
    problems = []
    if readout.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement code {readout.get('measurement_fingerprint')}, registered "
                        f"{spec['measurement_fp']}")
    s8 = spec["s8_positions"] if spec["s8_positions"] is not None else readout.get("s8_positions")
    if readout.get("s6_positions") != spec["s6_positions"] or readout.get("s8_positions") != s8:
        problems.append("the fixed position sets are not the registered ones")
    runs = readout.get("runs", {})
    for arm, cfg in spec["arms"].items():
        run = runs.get(arm, {})
        config = {k: v for k, v in run.get("config", {}).items() if k != "certify_depth"}
        if config != {**cfg, "seeds": list(spec["seeds"])}:
            problems.append(f"{arm}: the nets were not trained on the registered recipe")
        if run.get("training_fingerprint") != spec["era"]:
            problems.append(f"{arm}: the nets were not trained by the registered code")
    if sorted((n["arm"], n["seed"]) for n in readout["nets"]) != sorted((a, s) for a in spec["arms"]
                                                                        for s in spec["seeds"]):
        problems.append("the readout does not cover exactly every registered arm and seed")
    for n in readout["nets"]:
        if n["s6"]["positions"] != spec["s6_positions"] or n["s8"]["positions"] != s8:
            problems.append(f"{n['arm']} seed {n['seed']}: not scored on the registered positions")
        if n["arm"] == "h3" and len(n.get("carried_curve") or []) != spec["iterations"] + 1:
            problems.append(f"h3 seed {n['seed']}: not one ply-6 reading per training pass")
    return problems


def _curve_reading(nets: list, spec: dict) -> dict:
    gains = []
    for n in nets:
        at = [p["optimal"] / p["positions"] for p in n["carried_curve"]]
        gains.append(mean(at[p - 1] for p in spec["late"]) - mean(at[p - 1] for p in spec["early"]))
    gain = round(mean(gains), 9)
    verdict = ("supported" if gain >= spec["rising_at"] else "refuted" if gain <= spec["flat_at"] else "inconclusive")
    return {"verdict": verdict, "gain": gain, "gains": gains}


def h3_report(readout, spec: dict = SPEC) -> dict:
    """Both pre-registered readings, the ply-6 shares and the description sizes; or NOT_RUN for both readings."""
    try:
        problems = _problems(readout, spec)
        if not problems:
            by = {(n["arm"], n["seed"]): n for n in readout["nets"]}
            share = {k: {a: [by[(a, s)][k]["optimal"] / by[(a, s)][k]["positions"] for s in spec["seeds"]]
                         for a in spec["arms"]} for k in ("s6", "s8")}
            diffs = [h3 - h2 for h3, h2 in zip(share["s8"]["h3"], share["s8"]["h2"])]
            p = permutation_p(diffs)
            s8 = {"verdict": "supported" if p <= spec["alpha"] else "refuted" if mean(diffs) <= 0 else "inconclusive",
                  "mean": mean(diffs), "p": p, "differences": diffs, "shares": share["s8"]}
            return {"integrity": [], "s8": s8,
                    "curve": _curve_reading([by[("h3", s)] for s in spec["seeds"]], spec), "s6": share["s6"],
                    "description": {a: [by[(a, s)]["description"] for s in spec["seeds"]] for a in spec["arms"]}}
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        problems = [f"the readout could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, "s8": {"verdict": "not_run"}, "curve": {"verdict": "not_run"}}
