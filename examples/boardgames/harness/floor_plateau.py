"""§C.49 D1 — WHY the Connect-4 stop signal never fired (h88). Pre-registered.

The six T10 nets saved before the run hung (h88/h89) plateaued at 5-130 disagreements between their raw move and
their own 200-sim search on their own first-player tree. For each saved net the walk is recomputed exactly
(deterministic: the relabel search draws nothing) and the solver judges every disagreement: is the net's move optimal,
is the search's preferred move optimal? The solver also certifies each net through the horizon.

  BLOCKER    pooled over the nets, the share of disagreements where the NET is right (the search prefers a non-optimal
             move, or both moves are optimal). SUPPORTED — the search, not the net, blocks the signal — at
             >= `search_blocker_at`; REFUTED — the net is the blocker — at <= `net_blocker_at`; INCONCLUSIVE between
             or when there is no disagreement at all.
  CERTIFIED  how many nets the solver certifies through the horizon despite never stopping. SUPPORTED at
             >= `certified_support`, REFUTED at <= `certified_refute`, INCONCLUSIVE between.

Any integrity failure, or unreadable evidence, makes both readings NOT_RUN."""
from __future__ import annotations

from harness.c4_oracle import deepest_certified

CLASSES = ("both_optimal", "search_wrong", "net_wrong", "both_wrong")
SPEC = {"nets": {361: "3da8e1ab56554658815294e8ce678d7121cccf07966e9927efef3712cee34fbf",
                 362: "d1757c80fbc6bc1b7cd48aa98f3a7cf2d9917040683b8a93af73e774b6164411",
                 363: "b3adc13591455784f47b1c425b046920744b05038430f5eb1fab518e19ad0ea9",
                 364: "c3906cb07b1ed9bb69d9c0d3b3dfb89345614c6e12abfc29ff8f77e990e34c22",
                 365: "7c6f9e95201ff249141a0dc9e7267035422562c22b9ee6274fbe62bdbef9d298",
                 366: "631a3b406611337ea9a0842ffe5ca25d1c14d58e4f14be7a131fab21bf176cf8"},
        "config": {"sims": 200, "depth": 10, "agree_share": 0.5, "gumbel": True, "gumbel_m": 16, "c_scale": 0.1},
        "era": "ab26151f8373", "measurement_fp": "1130002a55c9", "search_blocker_at": 2 / 3, "net_blocker_at": 1 / 3,
        "certified_support": 4, "certified_refute": 2}


def _problems(evidence, spec: dict) -> list:
    if not isinstance(evidence, dict) or not isinstance(evidence.get("seeds"), list):
        return ["no seeds"]
    problems = []
    if evidence.get("training_fingerprint") != spec["era"]:
        problems.append(f"training era {evidence.get('training_fingerprint')}, registered {spec['era']}")
    if evidence.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement code {evidence.get('measurement_fingerprint')}, registered "
                        f"{spec['measurement_fp']}")
    if evidence.get("config") != spec["config"]:
        problems.append("config is not the registered diagnosis")
    rows = {r.get("seed"): r for r in evidence["seeds"] if isinstance(r, dict)}
    if sorted(rows) != sorted(spec["nets"]):
        problems.append(f"seeds are not the registered {sorted(spec['nets'])}")
    for seed, row in rows.items():
        if row.get("net_file_sha256") != spec["nets"].get(seed):
            problems.append(f"seed {seed}: not the saved net registered for it")
        if sum(row["classes"][c] for c in CLASSES) != row.get("disagreements"):
            problems.append(f"seed {seed}: the classified disagreements do not add up")
        cert = row["certificate"]
        if cert.get("horizon") != spec["config"]["depth"]:
            problems.append(f"seed {seed}: certified through {cert.get('horizon')}, not {spec['config']['depth']}")
        if not cert["certified"] and not cert["failures"]:
            problems.append(f"seed {seed}: an uncertified net with no failure — the walk was cut short")
    return problems


def _bars(value, high, low) -> str:
    return "supported" if value >= high else "refuted" if value <= low else "inconclusive"


def plateau_report(evidence, spec: dict = SPEC) -> dict:
    """Both pre-registered readings, or NOT_RUN for both when the evidence is not the registered diagnosis."""
    try:
        problems = _problems(evidence, spec)
        if not problems:
            rows = sorted(evidence["seeds"], key=lambda r: r["seed"])
            totals = {c: sum(r["classes"][c] for r in rows) for c in CLASSES}
            pooled = sum(totals.values())
            right = (totals["both_optimal"] + totals["search_wrong"]) / pooled if pooled else None
            blocker = ("inconclusive" if right is None
                       else _bars(right, spec["search_blocker_at"], spec["net_blocker_at"]))
            count = sum(1 for r in rows if r["certificate"]["certified"])
            return {"integrity": [],
                    "blocker": {"verdict": blocker, "net_right_share": right, "disagreements": pooled,
                                "totals": totals},
                    "certified": {"verdict": _bars(count, spec["certified_support"], spec["certified_refute"]),
                                  "count": count, "n": len(rows)},
                    "descriptives": {"deepest_certified": [deepest_certified(r["certificate"]) for r in rows],
                                     "per_seed": {r["seed"]: r["classes"] for r in rows}}}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    not_run = {"verdict": "not_run"}
    return {"integrity": problems, "blocker": not_run, "certified": not_run}
