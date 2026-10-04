"""§C.49 H1 — how far does a self-play net carry play beyond an exact opening table? Pre-registered.

The hybrid (harness.opening_table): an exception table, computed by the process with an exact "solve this position"
step, covers the first player's positions before `horizon` plies wherever the net's move is not optimal; the net plays
everywhere else. For each of the ten base-recipe nets (T19's base arm, the three 60-iteration T20 nets) and each
horizon h, the hybrid is certified through h + 2 plies (P-START, harness.certify) — so the net must carry one White
ply of its own, the one the table does not cover (a certificate the table alone covers would prove nothing).

For each `judged` horizon: SUPPORTED when at least `support_at` nets' hybrids are certified, REFUTED at `refute_at` or
fewer, INCONCLUSIVE between. The other horizons, every table's size (its entries are the hybrid's description cost)
and the net's failures by ply are reported, not judged. Evidence that is not the registered measurement is NOT_RUN."""
from __future__ import annotations

SPEC = {"runs": [["c49_T19_base.json.gz", list(range(451, 458))], ["c49_T20_s471.json.gz", [471]],
                 ["c49_T20_s472.json.gz", [472]], ["c49_T20_s473.json.gz", [473]]],
        "horizons": [3, 5, 7], "judged": [3, 5], "measurement_fp": "ae504f03a68d", "support_at": 8, "refute_at": 5}


def _problems(evidence, spec: dict) -> list:
    problems = []
    if evidence.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement code {evidence.get('measurement_fingerprint')}, registered "
                        f"{spec['measurement_fp']}")
    if evidence.get("horizons") != spec["horizons"]:
        problems.append("the horizons are not the registered ones")
    nets = sorted((n["run"], n["seed"]) for n in evidence["nets"])
    if nets != sorted((run, s) for run, seeds in spec["runs"] for s in seeds):
        problems.append("the nets are not exactly the registered ones")
    for n in evidence["nets"]:
        for h in spec["horizons"]:
            reading = n["horizons"].get(str(h))
            if not isinstance(reading, dict) or reading.get("depth") != h + 2 \
                    or not isinstance(reading.get("certified"), bool):
                problems.append(f"{n['run']} seed {n['seed']}: no certificate through {h + 2} plies at horizon {h}")
    return problems


def hybrid_report(evidence, spec: dict = SPEC) -> dict:
    """Per horizon: the certified count (with the verdict where judged), the table sizes and the net's failures."""
    try:
        problems = _problems(evidence, spec)
        if not problems:
            out = {}
            for h in spec["horizons"]:
                nets = sorted(evidence["nets"], key=lambda n: (n["run"], n["seed"]))
                readings = [n["horizons"][str(h)] for n in nets]
                certified = sum(1 for r in readings if r["certified"])
                failures: dict = {}
                for r in readings:
                    for ply, count in r["failures_by_ply"].items():
                        failures[ply] = failures.get(ply, 0) + count
                slot = {"certified": certified, "entries": [r["entries"] for r in readings],
                        "failures_by_ply": failures}
                if h in spec["judged"]:
                    slot["verdict"] = ("supported" if certified >= spec["support_at"] else
                                       "refuted" if certified <= spec["refute_at"] else "inconclusive")
                out[str(h)] = slot
            return {"integrity": [], "horizons": out}
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, "horizons": {str(h): {"verdict": "not_run"} for h in spec["horizons"]}}
