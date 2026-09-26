"""§C.48 T5 — do solver-free COVERAGE levers take the fixed process to a raw-perfect tic-tac-toe net? Pre-registered.

T4 (harness/floor_settle.py, h49/h50) left 6/10 raw-perfect with augmentation and 0/10 without. At every one of the
13 positions the augmented nets still failed, a 200-sim search with the same net returned an optimal move, so those
labels were right and the positions were most likely never trained. Two levers put unvisited positions into the
buffer without a solver, each tried alone on T4's recipe (unique buffer + settle), with augmentation and without:

  sib2   siblings out to TWO moves off the recorded positions instead of one (`sibling_depth`);
  open6  up to six random opening moves in the games that take any (`selfplay_opening_plies`), instead of two.

A third change is a fix, not a lever: without augmentation the sibling key is now the RAW state key. T4 deduplicated
siblings by the canonical key even without augmentation, so a sibling whose mirror image was recorded never entered
training although the net must learn each orientation separately. `no_augment_rawkey` is that fix alone — the
control for the other two no-augmentation arms; T4's augment arm (bit-for-bit reproduced under this code, seed
T5_REPRO_SEED) is the control for the augmented ones. Every arm names `sibling_depth` explicitly.

Per arm, T4's bars: SUPPORTED at >= 8/10 seeds raw-perfect after the settle, REFUTED at <= 5, INCONCLUSIVE between.

The MECHANISM claim, pooled over all 50 final nets, with the net as the unit. A position is TRAINED when it is in
the final training set: every self-play position (the unique buffer never evicts here) and the last sibling set,
each standing for all its images under augmentation. Units are nets with at least one failure. SUPPORTED when at
least MECH_MIN_NETS nets are units, a one-sided sign test that untrained positions fail at a higher rate than trained
ones (a tie counts against) gives p < MECH_ALPHA, and the pooled rate ratio is >= MECH_SUPPORT_RATIO; REFUTED when
the pooled ratio is <= MECH_REFUTE_RATIO; INCONCLUSIVE otherwise; NOT_RUN with fewer units. Any integrity failure,
or unreadable evidence, makes every verdict NOT_RUN."""
from __future__ import annotations

import math

from harness.floor_settle import T4_ARMS, T4_POSITIONS, T4_REFUTE_AT, T4_SEEDS, T4_SUPPORT_AT, _seed_problems

T5_SEEDS = T4_SEEDS
T5_REPRO_SEED = 311
T5_ERA = "f1ee57d6152b"
T5_MEASUREMENT_FP = "fe2d36f5f9d3"
MECH_MIN_NETS = 10
MECH_ALPHA = 0.01
MECH_SUPPORT_RATIO = 5.0
MECH_REFUTE_RATIO = 2.0
T5_ARMS = {
    "augment_sib2": {**T4_ARMS["augment"], "sibling_depth": 2},
    "augment_open6": {**T4_ARMS["augment"], "sibling_depth": 1, "opening_plies": 6},
    "no_augment_rawkey": {**T4_ARMS["no_augment"], "sibling_depth": 1},
    "no_augment_sib2": {**T4_ARMS["no_augment"], "sibling_depth": 2},
    "no_augment_open6": {**T4_ARMS["no_augment"], "sibling_depth": 1, "opening_plies": 6},
}
_COVERAGE_FIELDS = ("trained", "untrained", "failures_trained", "failures_untrained")


def _coverage_problems(where: str, row: dict, depth: int) -> list:
    problems = []
    cov = row.get("coverage")
    if not isinstance(cov, dict) or not all(isinstance(cov.get(f), int) and cov[f] >= 0 for f in _COVERAGE_FIELDS):
        return [f"{where}: no coverage record"]
    if cov["trained"] + cov["untrained"] != T4_POSITIONS:
        problems.append(f"{where}: trained + untrained is not {T4_POSITIONS}")
    if cov["failures_trained"] + cov["failures_untrained"] != row["strict_failures_per_pass"][-1]:
        problems.append(f"{where}: coverage failures do not add up to the final pass's failures")
    iterations = row["history"][:-1]
    if iterations[0].get("sibling_rings") != [] or not all(
            len(h.get("sibling_rings") or []) == depth and sum(h["sibling_rings"]) == h["siblings"]
            for h in iterations[1:]):
        problems.append(f"{where}: the run did not build {depth} sibling ring(s) on every relabelled iteration")
    return problems


def _repro_problems(repro: dict, t4_augment: dict) -> list:
    if not isinstance(repro, dict) or not isinstance(repro.get("seeds"), list) or len(repro["seeds"]) != 1:
        return ["the T4 reproduction is missing"]
    problems = []
    if repro.get("training_fingerprint") != T5_ERA:
        problems.append(f"the T4 reproduction ran in era {repro.get('training_fingerprint')}, not {T5_ERA}")
    if repro.get("config") != {**T4_ARMS["augment"], "seeds": [T5_REPRO_SEED]}:
        problems.append("the T4 reproduction did not run T4's augment recipe on the registered seed")
    got = repro["seeds"][0]
    want = next((r for r in t4_augment["seeds"] if r.get("seed") == T5_REPRO_SEED), None)
    if got.get("seed") != T5_REPRO_SEED or want is None:
        problems.append(f"the T4 reproduction is not seed {T5_REPRO_SEED}")
    elif got.get("strict_failures_per_pass") != want["strict_failures_per_pass"] \
            or got.get("final_failing_keys") != want["final_failing_keys"]:
        problems.append("T4's augment run does not reproduce bit-for-bit under this code")
    return problems


def integrity(arms: dict, repro: dict, t4_augment: dict) -> list:
    """Every reason the evidence is not the registered T5. Returns the problems; never raises on a missing arm."""
    if not isinstance(arms, dict):
        return ["no arms"]
    problems = _repro_problems(repro, t4_augment)
    for name, config in T5_ARMS.items():
        arm = arms.get(name)
        if not isinstance(arm, dict) or not isinstance(arm.get("seeds"), list) \
                or not isinstance(arm.get("config"), dict):
            problems.append(f"arm {name} is missing or has no seeds/config")
            continue
        if arm.get("training_fingerprint") != T5_ERA:
            problems.append(f"{name}: training era {arm.get('training_fingerprint')}, registered {T5_ERA}")
        if arm.get("measurement_fingerprint") != T5_MEASUREMENT_FP:
            problems.append(f"{name}: measurement code {arm.get('measurement_fingerprint')}, registered "
                            f"{T5_MEASUREMENT_FP}")
        if arm["config"] != {**config, "seeds": list(T5_SEEDS)}:
            problems.append(f"{name}: config is not the registered recipe")
        seeds = sorted(r.get("seed") for r in arm["seeds"] if isinstance(r, dict))
        if seeds != list(T5_SEEDS):
            problems.append(f"{name}: seeds {seeds}, not the registered {list(T5_SEEDS)}")
        for row in arm["seeds"]:
            if not isinstance(row, dict):
                problems.append(f"{name}: a seed row is not a record")
                continue
            seed_problems = _seed_problems(name, row)
            problems.extend(seed_problems or _coverage_problems(f"{name} seed {row.get('seed')}", row,
                                                                config["sibling_depth"]))
    return problems


def _sign_p(higher: int, n: int) -> float:
    return sum(math.comb(n, k) for k in range(higher, n + 1)) / 2 ** n


def mechanism(coverages: list) -> dict:
    """Do the final failures concentrate on untrained positions? Net as the unit; see the module docstring."""
    units = [c for c in coverages if c["failures_trained"] + c["failures_untrained"] > 0]
    higher = sum(1 for c in units if c["untrained"] and c["failures_untrained"] * c["trained"]
                 > c["failures_trained"] * c["untrained"])
    fu, u = sum(c["failures_untrained"] for c in units), sum(c["untrained"] for c in units)
    ft, t = sum(c["failures_trained"] for c in units), sum(c["trained"] for c in units)
    rate_u, rate_t = (fu / u if u else 0.0), (ft / t if t else 0.0)
    ratio = rate_u / rate_t if rate_t else (float("inf") if rate_u else float("nan"))
    p = _sign_p(higher, len(units)) if units else float("nan")
    if len(units) < MECH_MIN_NETS:
        verdict = "not_run"
    elif ratio <= MECH_REFUTE_RATIO:
        verdict = "refuted"
    elif p < MECH_ALPHA and ratio >= MECH_SUPPORT_RATIO:
        verdict = "supported"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "nets": len(units), "untrained_higher": higher, "sign_p": p, "rate_ratio": ratio,
            "rate_untrained": rate_u, "rate_trained": rate_t}


def _judge(arm: dict) -> dict:
    rows = sorted(arm["seeds"], key=lambda r: r["seed"])
    finals = [r["strict_failures_per_pass"][-1] for r in rows]
    perfect = sum(1 for f in finals if f == 0)
    verdict = "supported" if perfect >= T4_SUPPORT_AT else "refuted" if perfect <= T4_REFUTE_AT else "inconclusive"
    failures = sum(finals)
    return {"verdict": verdict, "perfect": perfect, "n": len(rows),
            "descriptives": {"final_failures": finals,
                             "mean_trained_positions": sum(r["coverage"]["trained"] for r in rows) / len(rows),
                             "share_of_failures_untrained": (sum(r["coverage"]["failures_untrained"] for r in rows)
                                                             / failures if failures else None),
                             "final_siblings": [r["history"][-2]["siblings"] for r in rows]}}


def coverage_report(arms: dict, repro: dict, t4_augment: dict) -> dict:
    """The pre-registered T5 verdicts — one per arm and the pooled mechanism — or NOT_RUN for all when the evidence
    is not the registered run."""
    try:
        problems = integrity(arms, repro, t4_augment)
        if not problems:
            judged = {name: _judge(arms[name]) for name in T5_ARMS}
            pooled = mechanism([r["coverage"] for name in T5_ARMS for r in arms[name]["seeds"]])
            return {"integrity": [], **judged, "mechanism": pooled}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, **{name: {"verdict": "not_run"} for name in T5_ARMS},
            "mechanism": {"verdict": "not_run"}}
