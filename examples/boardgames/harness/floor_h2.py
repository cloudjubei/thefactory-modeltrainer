"""§C.49 H2 — the hybrid's table IN THE TRAINING LOOP. Pre-registered.

H1 (h129-h131) showed a net trained without the table carries no ply past it: it is wrong at ~12% of the positions
one ply beyond. H2 trains the net beside the table instead: the process solves the opening exactly before training
(harness.opening_table.full_table — the first optimal move at every first-player position before `horizon` plies),
every self-play game starts at the table's frontier, and the strategy tree is walked from there, so all of the net's
capacity and all its real game outcomes go to the plies it must carry (scripts/c4_solver_free.py --opening-table).

Because the table fixes the first player's opening, every net faces the same positions one ply past it (the
`carried_positions`). The readout scores each final net's raw move there against exact values. The table arm's
seed-matched gain over the base arm (T17's base recipe, no table) is judged by an exact one-sided sign-flip
permutation test at `alpha`: SUPPORTED when it rejects, REFUTED when the mean gain is at most zero, INCONCLUSIVE
otherwise. The hybrid (table + net) certified through `horizon` + 4 plies — the net carrying two plies, one of them
never trained — is reported per arm, not judged. A readout that is not the registered measurement is NOT_RUN."""
from __future__ import annotations

from statistics import mean

from harness.floor_c4_powered import SPEC as T17_SPEC, permutation_p

_BASE = T17_SPEC["arms"]["base"]
SPEC = {"arms": {"base": _BASE, "table": {**_BASE, "strategy_tree": {"player": 0, "depth": 2},
                                          "opening_table": {"horizon": 5, "entries": 56, "frontier": 44}}},
        "seeds": tuple(range(481, 488)), "prefix": "c49_H2", "era": "33939d5e2d76", "measurement_fp": "05ef8262c544",
        "horizon": 5, "carried_positions": 284, "alpha": 0.05}


def _problems(readout, spec: dict) -> list:
    problems = []
    if readout.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement code {readout.get('measurement_fingerprint')}, registered "
                        f"{spec['measurement_fp']}")
    if readout.get("horizon") != spec["horizon"] or readout.get("carried_positions") != spec["carried_positions"]:
        problems.append("the table or the carried positions are not the registered ones")
    runs = readout.get("runs", {})
    expected = {f"{spec['prefix']}/{a}": {**c, "seeds": list(spec["seeds"])} for a, c in spec["arms"].items()}
    if {k: {kk: vv for kk, vv in v.get("config", {}).items() if kk != "certify_depth"} for k, v in runs.items()} \
            != expected:
        problems.append("the nets were not trained on the registered recipes")
    if any(v.get("training_fingerprint") != spec["era"] for v in runs.values()):
        problems.append("the nets were not trained by the registered code")
    if sorted((n["arm"], n["seed"]) for n in readout["nets"]) != sorted((a, s) for a in spec["arms"]
                                                                        for s in spec["seeds"]):
        problems.append("the readout does not cover exactly every registered arm and seed")
    for n in readout["nets"]:
        if n["carried"]["positions"] != spec["carried_positions"] or n["certificate"]["depth"] != spec["horizon"] + 4:
            problems.append(f"{n['arm']} seed {n['seed']}: not scored on the registered positions and depth")
    return problems


def h2_report(readout, spec: dict = SPEC) -> dict:
    """The pre-registered verdict, the seed-matched gains, each arm's shares, certificates and failures."""
    try:
        problems = _problems(readout, spec)
        if not problems:
            by = {(n["arm"], n["seed"]): n for n in readout["nets"]}
            share = {a: [by[(a, s)]["carried"]["optimal"] / by[(a, s)]["carried"]["positions"] for s in spec["seeds"]]
                     for a in spec["arms"]}
            diffs = [t - b for t, b in zip(share["table"], share["base"])]
            p = permutation_p(diffs)
            verdict = ("supported" if p <= spec["alpha"] else "refuted" if mean(diffs) <= 0 else "inconclusive")
            failures: dict = {a: {} for a in spec["arms"]}
            for n in readout["nets"]:
                for ply, count in n["certificate"]["failures_by_ply"].items():
                    failures[n["arm"]][ply] = failures[n["arm"]].get(ply, 0) + count
            return {"integrity": [], "verdict": verdict, "mean": mean(diffs), "p": p, "differences": diffs,
                    "shares": share, "failures_by_ply": failures,
                    "certified": {a: sum(1 for n in readout["nets"] if n["arm"] == a and n["certificate"]["certified"])
                                  for a in spec["arms"]}}
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        problems = [f"the readout could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, "verdict": "not_run"}
