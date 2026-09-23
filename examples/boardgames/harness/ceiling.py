"""§C.42 LOCALIZING THE CEILING — read a multi-seed dump of coverage failures into four verdicts.

§C.41 measured that the process plateaus short of certified-perfect play and that the misses sit at particular
plies. That alone cannot say whether the ceiling is a property of the RECIPE (fixable) or noise in one training
run. The evidence this module reads is, for each independently trained seed, every state its net misplays
(`coverage.coverage_failures`) plus per-failure diagnostics and the net's training-visit counts. Four claims are
judged from it, each at alpha / 4 so the family holds alpha:

  systematic     the seeds fail at the SAME states more than chance allows (`blind_spot_concentration` over the
                 failable universe). False means the ceiling is seed noise and there is no one cause to chase.
  net_not_budget more deployment search fixes FEWER than half the failures (one-sided Wilson upper bound < 0.5):
                 the error lives in the net, not in how long it was allowed to think. Read only when
                 `attributable` — a search-alone control (an untrained net at the deep budget) must itself fail
                 the same states, or the deep search is just brute-forcing a small tree.
  starved        failing states had FEWER training visits than non-failing states of the same seed and ply
                 (stratified permutation): the net was wrong where self-play stopped showing it the position.
  off_manifold   failing states lie off the optimal-play manifold more often than the base rate (exact
                 hypergeometric): the misses are positions reachable only after someone blundered.

Every verdict is a boolean computed here, from the stored evidence, never typed in; `decomposition` reports the
descriptive split (prior vs value error, severity, ply) that the verdicts do not test."""
from __future__ import annotations

import math
from statistics import NormalDist

from harness.coverage import blind_spot_concentration
from harness.measurement import hypergeom_sf, stratified_permutation_p, wilson_interval

CLAIMS = ("systematic", "net_not_budget", "starved", "off_manifold")


def _search_alone_control(pairs: list, z: float) -> dict:
    """Whether the deep-search check can attribute anything. Where the deep budget nearly exhausts the game tree,
    an UNTRAINED net at that budget also plays the failing states correctly, so "more search fixes it" says nothing
    about the net (measured on tictactoe: 400 sims with an untrained net fixed 15 of the 17 failing states).
    `confounded` is True when search alone fixes most failures, False when it fixes few, None when there is no
    control or it cannot tell; only False makes the net-vs-budget reading `attributable`."""
    if not pairs or any("control_deep_ok" not in f for f in pairs):
        return {"control_fix_rate": None, "confounded": None, "attributable": False}
    ctrl = sum(1 for f in pairs if f["control_deep_ok"])
    clo, chi = wilson_interval(ctrl, len(pairs), z)
    confounded = True if clo > 0.5 else False if chi < 0.5 else None
    return {"control_fix_rate": ctrl / len(pairs), "confounded": confounded, "attributable": confounded is False}


def localize_report(evidence: dict, alpha: float = 0.05, trials: int = 20000, seed: int = 0) -> dict:
    universe = set(evidence["universe"])
    on_path = set(evidence["optimal_play_keys"])
    ply_of = {k: p for k, p in evidence["plies"]}
    seeds = evidence["seeds"]
    a = alpha / len(CLAIMS)
    z = NormalDist().inv_cdf(1 - a)
    sets = [{f["key"] for f in s["failures"]} for s in seeds]

    conc = blind_spot_concentration(sets, universe, trials=trials, seed=seed)
    systematic = {"p": conc["p"], "ratio": conc["ratio"], "observed_pairs": conc["observed_pairs"],
                  "expected_pairs": conc["expected_pairs"], "verdict": conc["p"] < a}

    pairs = [f for s in seeds for f in s["failures"]]
    fixed = sum(1 for f in pairs if f["deep_ok"])
    lo, hi = wilson_interval(fixed, len(pairs), z) if pairs else (0.0, 1.0)
    net_not_budget = {"fixed": fixed, "failures": len(pairs), "fix_rate": fixed / len(pairs) if pairs else 0.0,
                      "wilson": [lo, hi], "verdict": hi < 0.5, "budget_fixes": lo > 0.5,
                      **_search_alone_control(pairs, z)}

    values, labels, strata = [], [], []
    for i, s in enumerate(seeds):
        visits = {k: v for k, v in s["visits"]}
        for k in sorted(universe):
            values.append(visits.get(k, 0))
            labels.append(k in sets[i])
            strata.append((i, ply_of[k]))
    p_starved = stratified_permutation_p(values, labels, strata, trials=trials, seed=seed, alternative="less")
    fail_visits = [v for v, lab in zip(values, labels) if lab]
    pass_visits = [v for v, lab in zip(values, labels) if not lab]
    starved = {"p": p_starved, "verdict": p_starved < a,
               "mean_visits_failing": sum(fail_visits) / max(1, len(fail_visits)),
               "mean_visits_other": sum(pass_visits) / max(1, len(pass_visits))}

    union = set().union(*sets)
    off = universe - on_path
    failing_off = len(union & off)
    p_off = hypergeom_sf(failing_off, len(universe), len(off), len(union))
    off_manifold = {"p": p_off, "failing_distinct": len(union), "failing_off": failing_off,
                    "base_rate": len(off) / len(universe) if universe else 0.0, "verdict": p_off < a}

    severity: dict = {}
    by_ply: dict = {}
    for f in pairs:
        severity[f["severity"]] = severity.get(f["severity"], 0) + 1
        by_ply[f["ply"]] = by_ply.get(f["ply"], 0) + 1
    decomposition = {"pairs": len(pairs),
                     "prior_prefers_played": sum(1 for f in pairs if f["prior_prefers_played"]),
                     "value_prefers_played": sum(1 for f in pairs if f["value_prefers_played"]),
                     "severity": severity, "by_ply": dict(sorted(by_ply.items()))}

    recurring = [{"key": k, "ply": ply_of[k], "seeds": conc["counts"][k], "on_path": k in on_path}
                 for k in conc["recurring"]]
    covs = [s["coverage"] for s in seeds]
    return {"alpha": alpha, "alpha_each": a, "n_seeds": len(seeds), "universe": len(universe),
            "coverage": {"per_seed": covs, "mean": sum(covs) / len(covs), "min": min(covs), "max": max(covs)},
            "systematic": systematic, "net_not_budget": net_not_budget, "starved": starved,
            "off_manifold": off_manifold, "decomposition": decomposition,
            "recurring": sorted(recurring, key=lambda r: (-r["seeds"], r["ply"]))}


AB_CLAIMS = ("replication", "target", "policy", "coverage")

# The mechanism a treatment is meant to move, read per seed from the arm evidence: the per-seed field it lives in
# and how to reduce it over the target set. `visits` = target-state self-play visits (§C.44); `labels` = the share
# of self-play targets at the target states that name an optimal move (§C.45, harness.targets).
MANIPULATIONS = {
    "visits": ("visits", lambda row, tkeys: sum(dict(row["visits"]).get(k, 0) for k in tkeys)),
    "labels": ("target_labels",
               lambda row, tkeys: row["target_labels"]["label_ok"] / max(1, sum(row["target_labels"].values()))),
    "dose": ("written_labels", lambda row, tkeys: sum(p["argmax_ok"] for p in row["written_labels"])),
}


# The sign-flip confidence bound sits ON a subset mean of the deltas, and with small-integer deltas (failure
# counts) that is often exactly a pre-registered threshold (0.5 = 3 of 6, 1 of 2, ...). The bisection lands a few
# 1e-14 past it, so every threshold comparison carries this slack; distinct subset means are ~1e-3 apart.
_TIE = 1e-9


def _bound_within(bound: float, margin: float) -> bool:
    """An upper bound at or below `margin`, reading an exact tie as within (the confidence set is closed)."""
    return bound <= margin + _TIE


def _bound_below(bound: float, threshold: float) -> bool:
    """An upper bound STRICTLY below `threshold`, reading an exact tie as not below."""
    return bound < threshold - _TIE


def _half_sums(mags: list) -> list:
    sums = [0.0]
    for m in mags:
        sums = [x + m for x in sums] + [x - m for x in sums]
    return sums


def _signflip_p(deltas: list) -> float:
    """Exact one-sided sign-flip p: the share of all 2^n sign patterns of |d_i| whose sum reaches sum(d). Counted
    meet-in-the-middle (two halves of 2^(n/2) sums, one sorted and bisected), so n = 20 costs ~10^4 steps, not
    10^6 — fast enough to invert into a confidence bound."""
    from bisect import bisect_left

    mags = [abs(d) for d in deltas]
    observed = sum(deltas) - 1e-12
    left, right = _half_sums(mags[: len(mags) // 2]), sorted(_half_sums(mags[len(mags) // 2:]))
    hits = sum(len(right) - bisect_left(right, observed - x) for x in left)
    return hits / 2 ** len(deltas)


def _lower_bound(deltas: list, conf_alpha: float) -> float:
    """The one-sided (1 - conf_alpha) LOWER confidence bound on the mean shift: the largest delta the sign-flip
    test still rejects as too small. p(d - δ) rises with δ, so it is found by bisection."""
    lo, hi = min(deltas) - 1.0, max(deltas)
    if _signflip_p([d - lo for d in deltas]) >= conf_alpha:
        return float("-inf")
    for _ in range(60):
        mid = (lo + hi) / 2
        if _signflip_p([d - mid for d in deltas]) < conf_alpha:
            lo = mid
        else:
            hi = mid
    return lo


def _paired_gain(deltas: list, a: float, conf_alpha: float = 0.05) -> dict:
    """Exact one-sided sign-flip permutation test that the treatment RAISES a per-seed metric: under the null each
    seed pair's delta is as likely negated, so p is the share of all 2^n sign patterns whose sum reaches the
    observed one. The unit is the training run. No normality is assumed — the per-seed metrics are discrete
    (k of 17 target states), where equal deltas are common and a t-test's zero spread would read as p = 0.

    `lower`/`upper` are one-sided (1 - conf_alpha) confidence bounds from inverting the same test, so a result
    that misses its bar still says something: "the effect is below `upper`" — never "there is no effect"."""
    n = len(deltas)
    if n < 2:
        raise ValueError("a paired comparison needs at least two seed pairs")
    if n > 30:
        raise ValueError("the exact sign-flip test enumerates 2^n patterns; more than 30 seed pairs needs sampling")
    mean = sum(deltas) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1))
    p = _signflip_p(deltas)
    return {"mean_delta": mean, "sd": sd, "n": n, "p": p, "verdict": p < a,
            "lower": _lower_bound(deltas, conf_alpha), "upper": -_lower_bound([-d for d in deltas], conf_alpha),
            "conf_alpha": conf_alpha}


def replication(arm: dict, tkeys: set, a: float) -> dict:
    """Do this arm's seeds misplay the target states beyond the target's share of the failable universe? Exact
    hypergeometric over (seed, state) pairs — the §C.43 blind spot, re-found (or not) on fresh seeds."""
    universe = set(arm["universe"])
    fails = [f["key"] for s in arm["seeds"] for f in s["failures"]]
    in_target = sum(1 for k in fails if k in tkeys)
    n = len(arm["seeds"])
    p = hypergeom_sf(in_target, n * len(universe), n * len(tkeys & universe), len(fails)) if fails else 1.0
    return {"in_target": in_target, "failures": len(fails), "target_share": len(tkeys & universe) / len(universe),
            "p": p, "verdict": p < a}


def ab_report(base: dict, treat: dict, target: dict, alpha: float = 0.05,
              treatment_keys: tuple = ("opening_plies", "opening_zero_frac"), manipulation: str = "visits",
              alpha_each: float | None = None, migration_margin: float = 0.5,
              manipulation_alpha: float | None = None) -> dict:
    """Does a recipe change fix the localized blind spot? `base` and `treat` are two localize_ceiling evidence
    files trained on the SAME seeds with the same training code, differing only in `treatment_keys`; `target`
    is the evidence the blind spot was localized from, whose failing states form the pre-declared target set.

    Four claims at alpha / 4: `replication` (fresh baseline seeds miss the target beyond its share of the
    failable universe — the §C.43 blind spot is real out of sample), and the treatment raising, per seed pair,
    `target` (target states played correctly at the eval budget), `policy` (raw-policy coverage) and `coverage`
    (eval-budget coverage). `manipulation` checks the treatment actually moved the MECHANISM it targets (one of
    MANIPULATIONS — target-state visits, or target-state label quality), and `mechanism_causal` combines it with
    `target`: a manipulation that succeeded while the target did NOT improve shows that mechanism is not the
    cause (§C.44: visits rose 4.5x, the blind spot stayed). A seed missing the manipulation's field is refused —
    an absent measurement is not a zero. `alpha_each`, when given, is the exact bar every verdict is judged at
    (a pre-registered family sets it); otherwise alpha is split over the four claims. `outside_target` guards
    against a fix that only MOVES the blind spot: eval failures outside the target per seed, and `holds` when the
    one-sided upper bound on their mean rise is within `migration_margin`. `manipulation_alpha` judges the
    manipulation check at a gate's own alpha (default: the claims' alpha). Refused too: a `target` whose states
    differ from the target_keys an arm recorded its label measurements on, arms measured by
    different measurement code, unpaired seeds, moved training code, arms differing in
    anything but the declared treatment (or in nothing), and a target chosen from the very seeds being evaluated —
    those seeds failed the target by selection, so any treatment would look like a fix by regression to the mean."""
    if manipulation not in MANIPULATIONS:
        raise ValueError(f"unknown manipulation {manipulation!r}; choose one of {sorted(MANIPULATIONS)}")
    if base.get("training_fingerprint") != treat.get("training_fingerprint"):
        raise ValueError("the arms were trained by different code (training fingerprint differs) — the "
                         "difference would not be the treatment's")
    if base.get("measurement_fingerprint") != treat.get("measurement_fingerprint"):
        raise ValueError("the arms were measured by different measurement code (measurement fingerprint "
                         "differs) — a difference could be the instrument's, not the treatment's")
    bcfg = {k: v for k, v in base["config"].items() if k != "seeds"}
    tcfg = {k: v for k, v in treat["config"].items() if k != "seeds"}
    diff = sorted(k for k in set(bcfg) | set(tcfg) if bcfg.get(k) != tcfg.get(k))
    if not diff:
        raise ValueError("the arms show no difference in config — there is no treatment to measure")
    if not set(diff) <= set(treatment_keys):
        raise ValueError(f"the arms differ in {sorted(set(diff) - set(treatment_keys))} beyond the declared "
                         f"treatment {list(treatment_keys)} — the effect could not be attributed")
    bseeds = {s["seed"]: s for s in base["seeds"]}
    tseeds = {s["seed"]: s for s in treat["seeds"]}
    if set(bseeds) != set(tseeds):
        raise ValueError(f"the arms' seeds do not pair: {sorted(bseeds)} vs {sorted(tseeds)}")
    tkeys = {f["key"] for s in target["seeds"] for f in s["failures"]}
    for arm_name, arm in (("base", base), ("treat", treat)):
        recorded = arm["config"].get("target_keys")
        if recorded is not None and set(recorded) != tkeys:
            raise ValueError(f"the {arm_name} arm recorded its label measurements on a different target set than "
                             f"the one this report judges ({len(recorded)} vs {len(tkeys)} states) — the gates and "
                             f"the claims would be read over different states")
    overlap = set(bseeds) & {s["seed"] for s in target["seeds"]}
    if overlap:
        raise ValueError(f"seeds {sorted(overlap)} both CHOSE the target set and are evaluated on it — they "
                         f"failed those states by selection, so any change looks like a fix by regression to "
                         f"the mean; evaluate on fresh seeds")
    a = alpha_each if alpha_each is not None else alpha / len(AB_CLAIMS)
    order = sorted(bseeds)

    def target_ok(keys):
        return 1 - len(set(keys) & tkeys) / len(tkeys)

    def per(arm, fn):
        return [fn(arm[s]) for s in order]

    metrics = {
        "target": lambda r: target_ok(f["key"] for f in r["failures"]),
        "target_policy": lambda r: target_ok(r.get("policy_fail_keys", [])),
        "policy": lambda r: r["policy_coverage"],
        "coverage": lambda r: r["coverage"],
    }
    out = {"alpha_each": a, "seeds": order, "treatment": {k: treat["config"].get(k) for k in diff},
           "target_size": len(tkeys)}
    for name, fn in metrics.items():
        b, t = per(bseeds, fn), per(tseeds, fn)
        out[name] = {"base": b, "treat": t, **_paired_gain([y - x for x, y in zip(b, t)], a)}

    field, read = MANIPULATIONS[manipulation]
    for arm_name, arm in (("base", bseeds), ("treat", tseeds)):
        for s in order:
            if field not in arm[s]:
                raise ValueError(f"seed {s} of the {arm_name} arm has no `{field}` — the {manipulation} "
                                 f"manipulation check cannot be read, and a missing measurement is not a zero")
    bt, tt = per(bseeds, lambda r: read(r, tkeys)), per(tseeds, lambda r: read(r, tkeys))
    ma = manipulation_alpha if manipulation_alpha is not None else a
    out["manipulation"] = {"kind": manipulation, "alpha": ma, "base": bt, "treat": tt, "base_mean": sum(bt) / len(bt),
                           "treat_mean": sum(tt) / len(tt), **_paired_gain([y - x for x, y in zip(bt, tt)], ma)}
    out["mechanism_causal"] = {"mechanism": manipulation, "manipulation_succeeded": out["manipulation"]["verdict"],
                               "target_improved": out["target"]["verdict"],
                               "is_causal": out["manipulation"]["verdict"] and out["target"]["verdict"]}

    def outside(r):
        return len({f["key"] for f in r["failures"]} - tkeys)
    bo, to = per(bseeds, outside), per(tseeds, outside)
    rise = _paired_gain([y - x for x, y in zip(bo, to)], a)
    out["outside_target"] = {"base": bo, "treat": to, "base_mean": sum(bo) / len(bo), "treat_mean": sum(to) / len(to),
                             "upper": rise["upper"], "margin": migration_margin,
                             "holds": _bound_within(rise["upper"], migration_margin)}

    out["replication"] = replication(base, tkeys, a)
    universe = set(base["universe"])
    on_path = set(base.get("optimal_play_keys", [])) & universe
    if on_path:
        out["on_path"] = {arm: [1 - len({f["key"] for f in r[s]["failures"]} & on_path) / len(on_path)
                                for s in order] for arm, r in (("base", bseeds), ("treat", tseeds))}
    return out


# §C.45 — the pre-registered claim family (A3, a transferable relabel budget S*, was dropped by its own
# pre-stated rule: no budget gave trained labels >= 0.85 while search alone stayed <= 0.65 on tic-tac-toe): (name, chain, control arm, treatment arm, metric, the config keys the two
# arms are DECLARED to differ in, the minimum share of the treatment's delivered target labels that must be right).
C45_CLAIMS = (
    ("A1", "A", "mixed_R32", "mixed_R200", "target", ("reanalyze_sims",), 0.85),
    ("A2", "A", "mixed_R32", "mixed_R200", "target_policy", ("reanalyze_sims",), 0.85),
    ("B1", "B", "mixed", "mixed_deep", "target", ("train_sims",), None),
)
C45_CHAIN_ALPHA = {"A": 0.04, "B": 0.01}


def _delivered(r: dict, control: dict, treat: dict, share: float | None, gate_alpha: float) -> dict:
    """Gate G2: the treatment must actually have delivered better LABELS to the target states. The count of correct
    target labels trained on must rise (paired, at the gate alpha), every seed of both arms must have trained on
    at least one target example, and — where the claim names a minimum — the treatment's delivered labels over
    the relabelled passes (pass >= 2) must mostly be right. A claim whose labels were not delivered says nothing
    about whether labels matter."""
    rises = r["manipulation"]["p"] < gate_alpha
    every = all(sum(p["n_target"] for p in s["written_labels"]) >= 1 for arm in (control, treat) for s in arm["seeds"])
    pooled, share_ok = None, True
    if share is not None:
        rows = [p for s in treat["seeds"] for p in s["written_labels"] if p["pass"] >= 2]
        n = sum(p["n_target"] for p in rows)
        pooled = sum(p["argmax_ok"] for p in rows) / n if n else 0.0
        share_ok = n > 0 and pooled >= share
    def seed_share(row):
        n = sum(p["n_target"] for p in row["written_labels"])
        return sum(p["argmax_ok"] for p in row["written_labels"]) / n if n else 0.0
    order = sorted(s["seed"] for s in control["seeds"])
    cs = {s["seed"]: seed_share(s) for s in control["seeds"]}
    ts = {s["seed"]: seed_share(s) for s in treat["seeds"]}
    shares = {"base_mean": sum(cs.values()) / len(cs), "treat_mean": sum(ts.values()) / len(ts),
              **_paired_gain([ts[k] - cs[k] for k in order], gate_alpha)}
    return {"dose_rises": rises, "every_seed_delivered": every, "written_share": pooled, "share": shares,
            "passed": rises and every and share_ok}


def c45_report(arms: dict, target: dict, gate_alpha: float = 0.05, chain_alpha: dict | None = None,
               sesoi_frac: float = 0.5, migration_margin: float = 0.5) -> dict:
    """Emit EXACTLY the §C.45 pre-registered claims, nothing else as a verdict.

    Gate G1 (replication on the `base` arm) must pass or every claim is `stopped`. Chain A (A1 -> A2) is a
    fixed sequence at its own alpha: a claim is tested only if every claim before it was `supported`, else it is
    `not_reached`. B1 is its own chain. Per claim: `not_delivered` when gate G2 fails; `supported` when p < alpha
    and — for a target-correctness claim — gate G3 (failures did not migrate outside the target) holds, `moved`
    when it does not; `refuted` only when the one-sided upper confidence bound excludes the smallest effect of
    interest (`sesoi_frac` of the control's shortfall); otherwise `inconclusive`. `not_run` when an arm is
    absent. Every ab_report behind a verdict is kept under `descriptive`."""
    alphas = chain_alpha or C45_CHAIN_ALPHA
    tkeys = {f["key"] for s in target["seeds"] for f in s["failures"]}
    g1 = replication(arms["base"], tkeys, gate_alpha)
    g1["passed"] = g1["verdict"]
    halted = dict.fromkeys(alphas, False)
    claims: dict = {}
    descriptive: dict = {}
    for name, chain, ctrl, trt, metric, keys, share in C45_CLAIMS:
        a = alphas[chain]
        if not g1["passed"]:
            claims[name] = {"verdict": "stopped", "alpha": a}
            continue
        if halted[chain]:
            claims[name] = {"verdict": "not_reached", "alpha": a}
            continue
        if ctrl not in arms or trt not in arms:
            claims[name] = {"verdict": "not_run", "alpha": a}
            halted[chain] = True
            continue
        r = ab_report(arms[ctrl], arms[trt], target, manipulation="dose", treatment_keys=keys, alpha_each=a,
                      migration_margin=migration_margin, manipulation_alpha=gate_alpha)
        descriptive[f"{name}: {trt} vs {ctrl}"] = r
        m = r[metric]
        sesoi = sesoi_frac * (1 - sum(m["base"]) / len(m["base"]))
        delivery = _delivered(r, arms[ctrl], arms[trt], share, gate_alpha)
        guard = r["outside_target"]["holds"] if metric == "target" else True
        if not delivery["passed"]:
            verdict = "not_delivered"
        elif m["p"] < a:
            verdict = "supported" if guard else "moved"
        elif _bound_below(m["upper"], sesoi):
            verdict = "refuted"
        else:
            verdict = "inconclusive"
        claims[name] = {"verdict": verdict, "alpha": a, "metric": metric, "treat": trt, "control": ctrl,
                        "p": m["p"], "mean_delta": m["mean_delta"], "lower": m["lower"], "upper": m["upper"],
                        "sesoi": sesoi, "delivery": delivery, "migration": r["outside_target"]}
        if verdict != "supported":
            halted[chain] = True
    return {"G1": g1, "claims": claims, "descriptive": descriptive}
