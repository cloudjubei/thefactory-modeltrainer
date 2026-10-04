"""§C.49 T17 — the POWERED, pre-registered comparison of three value-signal fixes against base on Connect-4.

The 4-seed pilots (T14, T16) read each net on its own tree and could only see large effects (h116). Calibration on
one fixed set of 5,142 exactly valued positions (h117) showed seed-matched differences spread ~3.1 points. T17 trains
T14's pilot shape (a 6-ply tree, 20 iterations, no stop, then the settle) in four arms — base, n-step value targets,
tree-position value targets and the backward curriculum, each exactly as before — on seven fresh seeds, and scores
every final net on that fixed set (scripts/c4_fixed_set_readout.py).

Each treatment's seed-matched differences to base are judged by an exact one-sided sign-flip permutation test, with
Holm's correction across the treatments at `alpha`: SUPPORTED when Holm rejects, REFUTED when the mean difference is
at most zero, INCONCLUSIVE otherwise. Certification through `certify_depth` plies is reported per arm, not judged. A
readout that is not the registered measurement is NOT_RUN."""
from __future__ import annotations

from itertools import product
from statistics import mean

from harness.floor_c4_backplay import SPEC as T16_SPEC
from harness.floor_c4_value_signal import SPEC as T14_SPEC

SPEC = {"arms": {"base": T14_SPEC["arms"]["base"], "n_step": T14_SPEC["arms"]["n_step"],
                 "tree_value": T14_SPEC["arms"]["tree_value"], "backplay": T16_SPEC["arms"]["backplay"]},
        "treatments": ("n_step", "tree_value", "backplay"), "seeds": tuple(range(431, 438)), "prefix": "c49_T17",
        "era": "7b7e2b460262", "measurement_fp": "cfc0a014ec57", "plies": [0, 2, 4, 6, 8], "positions": 5142,
        "certify_depth": 6, "alpha": 0.05}


def permutation_p(differences: list) -> float:
    """The exact one-sided p of a sign-flip test: the share of the 2^n sign assignments whose mean is at least the
    observed mean."""
    observed = mean(differences)
    sizes = [abs(d) for d in differences]
    hits = sum(1 for signs in product((1, -1), repeat=len(sizes))
               if mean(s * d for s, d in zip(signs, sizes)) >= observed - 1e-12)
    return hits / 2 ** len(sizes)


def holm(ps: dict, alpha: float) -> set:
    """The names Holm's step-down procedure rejects at family-wise level `alpha`."""
    rejected = set()
    for k, (name, p) in enumerate(sorted(ps.items(), key=lambda kv: kv[1])):
        if p > alpha / (len(ps) - k):
            break
        rejected.add(name)
    return rejected


def _problems(readout, spec: dict) -> list:
    problems = []
    if readout.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement code {readout.get('measurement_fingerprint')}, registered "
                        f"{spec['measurement_fp']}")
    if readout.get("positions") != spec["positions"] or readout.get("plies") != spec["plies"]:
        problems.append("the fixed set is not the registered one")
    expected = {f"{spec['prefix']}/{a}": {**c, "seeds": list(spec["seeds"]), "certify_depth": spec["certify_depth"]}
                for a, c in spec["arms"].items()}
    runs = readout.get("runs", {})
    if {k: v.get("config") for k, v in runs.items()} != expected:
        problems.append("the nets were not trained on the registered recipes")
    if any(v.get("training_fingerprint") != spec["era"] for v in runs.values()):
        problems.append("the nets were not trained by the registered code")
    nets = sorted((n["run"], n["arm"], n["seed"]) for n in readout["nets"])
    if nets != sorted((spec["prefix"], a, s) for a in spec["arms"] for s in spec["seeds"]):
        problems.append("the readout does not cover exactly every registered arm and seed")
    return problems


def powered_report(readout, spec: dict = SPEC) -> dict:
    """Each treatment's pre-registered verdict, its seed-matched differences, mean and p, and certification per arm;
    or NOT_RUN for every treatment."""
    try:
        problems = _problems(readout, spec)
        if not problems:
            share = {(n["arm"], n["seed"]): n["optimal"] / n["positions"] for n in readout["nets"]}
            diffs = {t: [share[(t, s)] - share[("base", s)] for s in spec["seeds"]] for t in spec["treatments"]}
            ps = {t: permutation_p(d) for t, d in diffs.items()}
            rejected = holm(ps, spec["alpha"])
            treatments = {t: {"verdict": "supported" if t in rejected else
                              "refuted" if mean(d) <= 0 else "inconclusive",
                              "mean": mean(d), "p": ps[t], "differences": d} for t, d in diffs.items()}
            certified = {a: sum(1 for n in readout["nets"] if n["arm"] == a and n["certified"]) for a in spec["arms"]}
            return {"integrity": [], "treatments": treatments, "certified": certified,
                    "means": {a: mean(share[(a, s)] for s in spec["seeds"]) for a in spec["arms"]}}
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        problems = [f"the readout could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, "treatments": {t: {"verdict": "not_run"} for t in spec["treatments"]}}
