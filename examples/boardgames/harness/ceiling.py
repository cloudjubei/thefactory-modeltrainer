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

import hashlib
import math
from statistics import NormalDist

from harness.coverage import blind_spot_concentration, reachable_states
from harness.measurement import _betai, hypergeom_sf, stratified_permutation_p, t_critical, wilson_interval

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
    """The four §C.43 claims from one evidence file. An arm with NO failure in any seed has no blind spot to
    localize: every verdict is None (undefined), never a pass or a fail."""
    universe = set(evidence["universe"])
    on_path = set(evidence["optimal_play_keys"])
    ply_of = {k: p for k, p in evidence["plies"]}
    seeds = evidence["seeds"]
    a = alpha / len(CLAIMS)
    z = NormalDist().inv_cdf(1 - a)
    sets = [{f["key"] for f in s["failures"]} for s in seeds]

    if not any(sets):
        covs = [s["coverage"] for s in seeds]
        return {"alpha": alpha, "alpha_each": a, "n_seeds": len(seeds), "universe": len(universe),
                "coverage": {"per_seed": covs, "mean": sum(covs) / len(covs), "min": min(covs), "max": max(covs)},
                "no_failures": True, **{c: {"p": None, "verdict": None} for c in CLAIMS}, "recurring": []}
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


def _pair_arms(base: dict, treat: dict, target: dict, treatment_keys) -> dict:
    """The integrity every paired comparison of two arms rests on, shared by ab_report and c46_report so the two
    cannot drift apart. Refused: arms trained or measured by different code, arms differing in nothing or in
    anything beyond the declared `treatment_keys`, unpaired seeds, a `target` whose states differ from the
    target_keys an arm recorded its label measurements on, and a target chosen from the very seeds being evaluated —
    those seeds failed the target by selection, so any treatment would look like a fix by regression to the mean.
    Returns the config keys that differ, each arm's rows by seed, the target keys and the seed order."""
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
    return {"diff": diff, "base": bseeds, "treat": tseeds, "target_keys": tkeys, "order": sorted(bseeds)}


def _required(row: dict, field: str):
    """A per-seed field the report cannot do without — refused when absent, because reading a missing measurement as
    empty would score, e.g., a seed with no recorded raw-policy failures as a perfect prior."""
    if field not in row:
        raise ValueError(f"seed {row.get('seed')!r} has no `{field}` — a missing measurement is not a zero")
    return row[field]


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
    manipulation check at a gate's own alpha (default: the claims' alpha). Everything `_pair_arms` refuses is
    refused here too."""
    if manipulation not in MANIPULATIONS:
        raise ValueError(f"unknown manipulation {manipulation!r}; choose one of {sorted(MANIPULATIONS)}")
    pair = _pair_arms(base, treat, target, treatment_keys)
    diff, bseeds, tseeds, tkeys, order = pair["diff"], pair["base"], pair["treat"], pair["target_keys"], pair["order"]
    a = alpha_each if alpha_each is not None else alpha / len(AB_CLAIMS)

    def target_ok(keys):
        return 1 - len(set(keys) & tkeys) / len(tkeys)

    def per(arm, fn):
        return [fn(arm[s]) for s in order]

    metrics = {
        "target": lambda r: target_ok(f["key"] for f in r["failures"]),
        "target_policy": lambda r: target_ok(_required(r, "policy_fail_keys")),
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


def _judge(m: dict, a: float, sesoi: float, delivered: bool, guard: bool) -> str:
    """The verdict ladder below the stopping, chain and not-run checks, shared by §C.45 and §C.46 so a registered
    rule is written once: a claim whose manipulation was not delivered says nothing about it (`not_delivered`);
    p < alpha is `supported` only when its guard holds, else `moved`; a null is `refuted` only when the one-sided
    upper bound EXCLUDES the smallest effect of interest (a tie does not), otherwise `inconclusive`."""
    if not delivered:
        return "not_delivered"
    if m["p"] < a:
        return "supported" if guard else "moved"
    if _bound_below(m["upper"], sesoi):
        return "refuted"
    return "inconclusive"


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
        verdict = _judge(m, a, sesoi, delivery["passed"], guard)
        claims[name] = {"verdict": verdict, "alpha": a, "metric": metric, "treat": trt, "control": ctrl,
                        "p": m["p"], "mean_delta": m["mean_delta"], "lower": m["lower"], "upper": m["upper"],
                        "sesoi": sesoi, "delivery": delivery, "migration": r["outside_target"]}
        if verdict != "supported":
            halted[chain] = True
    return {"G1": g1, "claims": claims, "descriptive": descriptive}


# §C.46 — siblings × labels × architecture at the NET level. The nine registered arms, and every pair the report
# compares: the config keys the pair is DECLARED to differ in, and whether its two arms must share pass 1 byte for
# byte (nothing acts before the first relabel, so every residual arm trains the same pass 1, and so do the two
# legacy arms; legacy against residual cannot).
C46_ARMS = ("leg_R32", "leg_R200", "R32", "R200", "R32S", "R200S", "Rx", "RxS", "RxS_H")
C46_SIBLINGS = ("reanalyze_siblings", "steps_matched", "sibling_key")
C46_PAIRS = {
    ("R200", "R200S"): (C46_SIBLINGS, True),
    ("R32", "R32S"): (C46_SIBLINGS, True),
    ("leg_R200", "R200"): (("arch", "params_expected"), False),
    ("leg_R32", "leg_R200"): (("reanalyze_sims",), True),
    ("R32", "R200"): (("reanalyze_sims",), True),
    ("Rx", "RxS"): (C46_SIBLINGS, True),
    ("RxS", "RxS_H"): (("sibling_holdout",), True),
    ("R200S", "RxS"): (("reanalyze_sims", "policy_target", "oracle_fingerprint"), True),
}
# The registered claims: (name, chain, control arm, treatment arm, metric, the gates whose failure STOPS the claim,
# its delivery gate, the minimum label share that gate demands). Each chain is a fixed sequence at its own alpha
# (E 0.03 + A 0.01 + R 0.01 = the family's 0.05). E1 is deliberately not chained behind R1: R1 can legitimately
# fail on residual if 32-sim labels are already good there. The legacy replication heads the R chain because its
# power is ~1.00, so it costs almost nothing.
C46_CLAIMS = (
    ("E1", "E", "R200", "R200S", "policy_failures", (), "GE", 0.90),
    ("E2", "E", "R32", "R32S", "policy_failures", (), "GE", None),
    ("A1", "A", "leg_R200", "R200", "policy_failures", ("G0",), None, None),
    ("L1", "R", "leg_R32", "leg_R200", "target", ("G1-leg",), "G2", 0.85),
    ("L2", "R", "leg_R32", "leg_R200", "target_policy", ("G1-leg",), "G2", 0.85),
    ("R1", "R", "R32", "R200", "target", ("G1-leg", "G1-res"), "G2", 0.85),
    ("R2", "R", "R32", "R200", "target_policy", ("G1-leg", "G1-res"), "G2", 0.85),
)
C46_CHAIN_ALPHA = {"E": 0.03, "A": 0.01, "R": 0.01}
C46_G1_ARMS = {"G1-leg": "leg_R32", "G1-res": "R32"}
# The smallest effect of interest as a share of the control: of its mean failures per seed for a failure count,
# of its shortfall from 1 for a correctness share (as §C.45).
C46_SESOI = {"policy_failures": 0.25, "target": 0.5, "target_policy": 0.5}
C46_UNITS = {"policy_failures": "failures", "target": "target correctness",
             "target_policy": "target policy correctness"}
C46_STEP_RATIO = (0.75, 1.33)
C46_MIGRATION_MARGIN = 0.5
C46_CELL_SHARE = 0.85
# The evidence schema the report reads. A field missing from any of these records is refused, never read as zero.
C46_ARM_FIELDS = ("training_fingerprint", "measurement_fingerprint", "started", "config", "universe",
                  "optimal_play_keys", "plies", "seeds")
C46_CONFIG_FIELDS = ("arch", "params_expected", "train_sims", "reanalyze_frac", "reanalyze_sims", "reanalyze_siblings",
                     "sibling_key", "sibling_holdout", "steps_matched", "policy_target", "target_keys", "seeds")
C46_SEED_FIELDS = ("seed", "coverage", "policy_coverage", "failures", "policy_fail_keys", "written_labels", "visits",
                   "sibling_visits", "history", "passes", "params")
C46_HISTORY_FIELDS = ("siblings", "train_examples", "epoch_examples", "steps")
C46_PASS_FIELDS = ("pass", "weights_sha", "policy_fail_keys", "dose")


def _require(record: dict, fields: tuple, where: str) -> None:
    for field in fields:
        if field not in record:
            raise ValueError(f"{where} has no `{field}` — a missing field is refused, never read as zero")


def _pass_one(row: dict, where: str) -> dict:
    ones = [p for p in row["passes"] if p["pass"] == 1]
    if len(ones) != 1:
        raise ValueError(f"{where} did not record exactly one pass 1 ({len(ones)} found) — the first pass its pair "
                         f"shares cannot be checked")
    return ones[0]


def _c46_integrity(arms: dict, target: dict) -> None:
    """Refused before any claim is read: an unknown arm (a typo would otherwise read as `not_run`); any field of the
    evidence schema missing; a seed whose net has not the parameter count its config expects (§C.43-45 trained the
    legacy net while configured for 32/3/32, and nothing noticed); every declared pair through `_pair_arms`; and,
    within a pair declared to share pass 1, a seed whose pass-1 weights differ — the contrast would not start from
    the same net."""
    unknown = sorted(set(arms) - set(C46_ARMS))
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; the registered arms are {list(C46_ARMS)}")
    for name, arm in arms.items():
        _require(arm, C46_ARM_FIELDS, f"arm {name}")
        _require(arm["config"], C46_CONFIG_FIELDS, f"the config of arm {name}")
        for row in arm["seeds"]:
            where = f"seed {row.get('seed')} of arm {name}"
            _require(row, C46_SEED_FIELDS, where)
            for entry in row["history"]:
                _require(entry, C46_HISTORY_FIELDS, f"a history entry of {where}")
            for p in row["passes"]:
                _require(p, C46_PASS_FIELDS, f"a pass of {where}")
            if row["params"] != arm["config"]["params_expected"]:
                raise ValueError(f"{where} trained a net of {row['params']} parameters where its config expects "
                                 f"{arm['config']['params_expected']} — it is not the configured architecture")
    for (ctrl, trt), (keys, shares_pass1) in C46_PAIRS.items():
        if ctrl not in arms or trt not in arms:
            continue
        pair = _pair_arms(arms[ctrl], arms[trt], target, keys)
        if not shares_pass1:
            continue
        for s in pair["order"]:
            a = _pass_one(pair["base"][s], f"seed {s} of arm {ctrl}")["weights_sha"]
            b = _pass_one(pair["treat"][s], f"seed {s} of arm {trt}")["weights_sha"]
            if a != b:
                raise ValueError(f"seed {s}: arms {ctrl} and {trt} share their first pass by design, yet their "
                                 f"pass-1 weights differ ({a} vs {b}) — the pair is not the controlled contrast")


def one_ply_children(game) -> dict:
    """{key: keys of its non-terminal children} for every reachable non-terminal position, keyed as the trainer keys
    its siblings (symmetry-canonical when the game has a canonical key). Read from ONE representative per key: the
    children of symmetric images are images of each other, so any representative gives the same set."""
    key = getattr(game, "canonical_key", None) or game.state_key
    states, _ = reachable_states(game, exact=True, symmetry=True)
    out: dict = {}
    for s in states:
        kids = set()
        for a in game.legal_actions(s):
            child = game.step(s, a)
            if not game.is_terminal(child):
                kids.add(key(child))
        out[key(s)] = kids
    return out


def _held_out(k, holdout: dict | None) -> bool:
    """The §C.46 hold-out rule, re-derived from the recorded config so the report does not take the trainer's word
    for what it withheld: k is held out iff sha256(f"{salt}:{k!r}") % mod == 0."""
    if holdout is None:
        return False
    return int(hashlib.sha256(f"{holdout['salt']}:{k!r}".encode()).hexdigest(), 16) % int(holdout["mod"]) == 0


def sibling_ring(children: dict, keys, holdout: dict | None = None) -> set:
    """The one-ply closure of `keys` that §C.46 siblings must cover exactly: every non-terminal child of a key, minus
    the keys themselves and the held-out keys. A key the game cannot reach is refused — the evidence would be of
    another game or another keying, and its ring would silently read as empty."""
    own = set(keys)
    ring: set = set()
    for k in own:
        if k not in children:
            raise ValueError(f"key {k!r} is not a reachable non-terminal position of this game — the evidence and "
                             f"the game do not match")
        ring |= children[k]
    return {k for k in ring - own if not _held_out(k, holdout)}


def _closure_mismatches(row: dict, children: dict, holdout: dict | None) -> list:
    """GE(iii) for one treatment seed: on every relabelled pass the keys trained on as siblings must be EXACTLY the
    ring of the keys trained on from self-play. A missing key is exposure the arm claims but did not give; an extra
    one is a sibling the pinned definition does not allow."""
    out = []
    for p in row["passes"]:
        if p["pass"] < 2:
            continue
        own = {k for k, sp_n, _sp_ok, _sib_n, _sib_ok in p["dose"] if sp_n > 0}
        sib = {k for k, _sp_n, _sp_ok, sib_n, _sib_ok in p["dose"] if sib_n > 0}
        ring = sibling_ring(children, own, holdout)
        if sib != ring:
            out.append({"pass": p["pass"], "missing": sorted(ring - sib), "extra": sorted(sib - ring)})
    return out


def _by_seed(arm: dict) -> dict:
    return {s["seed"]: s for s in arm["seeds"]}


def _relabelled(row: dict) -> list:
    return [p for p in row["passes"] if p["pass"] >= 2]


def _sibling_rows(p: dict) -> int:
    return sum(sib_n for _k, _sp_n, _sp_ok, sib_n, _sib_ok in p["dose"])


def _step_ratio(treat_row: dict, control_row: dict) -> float:
    t = sum(h["steps"] for h in treat_row["history"])
    c = sum(h["steps"] for h in control_row["history"])
    return t / c if c else math.inf


def _exposure_gate(control: dict, treat: dict, children: dict, share: float | None, gate_alpha: float) -> dict:
    """Gate GE: the siblings must have been delivered as pinned, at matched optimisation steps, or a fall in failures
    says nothing about siblings. (i) every treatment seed trains on sibling rows on every relabelled pass (and has
    one), the control on none, and (vi) its first relabelled pass ends with DIFFERENT weights from the
    control's — everything before that pass is byte-identical by construction, so equal weights mean the siblings
    never reached training; (ii) the distinct failable keys trained on in the final pass rise, paired, at the
    gate alpha; (iii) on every treatment seed and relabelled pass the sibling keys are exactly the one-ply ring of
    that pass's self-play keys, re-derived here from the game's rules; (iv) every seed's realised treatment/control
    step ratio lies within C46_STEP_RATIO; (v) where the claim names a share, its pooled sibling labels on failable
    rows name an optimal move at least that often."""
    universe = set(control["universe"])
    holdout = treat["config"]["sibling_holdout"]
    cs, ts = _by_seed(control), _by_seed(treat)
    order = sorted(cs)
    every_pass = all(_relabelled(ts[s]) and all(_sibling_rows(p) > 0 for p in _relabelled(ts[s])) for s in order)
    control_none = all(_sibling_rows(p) == 0 for s in order for p in cs[s]["passes"])

    def exposure(row):
        final = max(row["passes"], key=lambda p: p["pass"])
        return len({k for k, sp_n, _sp_ok, sib_n, _sib_ok in final["dose"] if k in universe and sp_n + sib_n > 0})
    rise = _paired_gain([exposure(ts[s]) - exposure(cs[s]) for s in order], gate_alpha)
    rises = rise["p"] < gate_alpha
    mismatches = [{"seed": s, **m} for s in order for m in _closure_mismatches(ts[s], children, holdout)]
    ratios = [_step_ratio(ts[s], cs[s]) for s in order]
    lo, hi = C46_STEP_RATIO
    within = all(lo <= r <= hi for r in ratios)
    rows = [(sib_n, sib_ok) for s in order for p in _relabelled(ts[s])
            for k, _sp_n, _sp_ok, sib_n, sib_ok in p["dose"] if k in universe]
    n = sum(r[0] for r in rows)
    pooled = sum(r[1] for r in rows) / n if n else None
    share_holds = share is None or (pooled is not None and pooled >= share)

    def first_relabelled_sha(row):
        return min(_relabelled(row), key=lambda p: p["pass"])["weights_sha"] if _relabelled(row) else None
    changed = all(first_relabelled_sha(ts[s]) is not None and first_relabelled_sha(ts[s]) != first_relabelled_sha(cs[s])
                  for s in order)
    return {"siblings_every_pass": every_pass, "control_has_none": control_none,
            "siblings_changed_training": changed,
            "exposure": {"rises": rises, **{k: v for k, v in rise.items() if k != "verdict"}},
            "closure": {"holds": not mismatches, "mismatches": mismatches},
            "step_ratio": {"within": within, "bounds": list(C46_STEP_RATIO), "per_seed": ratios},
            "sibling_label_share": {"share": pooled, "min": share, "holds": share_holds},
            "passed": every_pass and control_none and changed and rises and not mismatches and within and share_holds}


def _bounds(m: dict) -> dict:
    return {k: m[k] for k in ("p", "mean_delta", "lower", "upper")}


def _failure_claim(control: dict, treat: dict, a: float, gate: str | None, share: float | None, children: dict,
                   gate_alpha: float) -> tuple[dict, dict]:
    """An E/A claim: raw-policy failures over every state, gain = control − treatment per seed pair — the net-level
    count the §C.42 floor test is stated in (failable coverage only rescales it, and is reported alongside)."""
    cs, ts = _by_seed(control), _by_seed(treat)
    order = sorted(cs)
    base = [len(cs[s]["policy_fail_keys"]) for s in order]
    trt = [len(ts[s]["policy_fail_keys"]) for s in order]
    m = _paired_gain([b - t for b, t in zip(base, trt)], a)
    sesoi = C46_SESOI["policy_failures"] * sum(base) / len(base)
    gates = {"GE": _exposure_gate(control, treat, children, share, gate_alpha)} if gate == "GE" else {}
    verdict = _judge(m, a, sesoi, all(g["passed"] for g in gates.values()), True)
    return ({"verdict": verdict, **_bounds(m), "sesoi": sesoi, "control_failures": base, "treat_failures": trt,
             "control_mean": sum(base) / len(base), "treat_mean": sum(trt) / len(trt),
             "gain_failable_coverage": m["mean_delta"] / len(control["universe"])}, gates)


def _target_claim(r: dict, control: dict, treat: dict, metric: str, a: float, share: float,
                  gate_alpha: float) -> tuple[dict, dict]:
    """An R-chain claim, read as §C.45 read it: G2 (the correct-label dose rises, every seed trained on a target
    example, the delivered share holds) and, for eval-budget target correctness, G3 (failures did not migrate
    outside the target)."""
    m = r[metric]
    sesoi = C46_SESOI[metric] * (1 - sum(m["base"]) / len(m["base"]))
    gates = {"G2": _delivered(r, control, treat, share, gate_alpha)}
    guard = True
    if metric == "target":
        gates["G3"] = r["outside_target"]
        guard = gates["G3"]["holds"]
    return {"verdict": _judge(m, a, sesoi, gates["G2"]["passed"], guard), **_bounds(m), "sesoi": sesoi}, gates


def recurrent_keys(reference: dict, min_seeds: int = 3) -> set:
    """P: the keys a reference arm's raw policy fails in at least `min_seeds` of its seeds (§C.46: 31 keys of
    c45_mixed_R200 at 3 of 20). Descriptive only; a seed counts a key once however often it lists it."""
    counts: dict = {}
    for s in reference["seeds"]:
        for k in set(s["policy_fail_keys"]):
            counts[k] = counts.get(k, 0) + 1
    return {k for k, c in counts.items() if c >= min_seeds}


def _beta_quantile(q: float, a: float, b: float) -> float:
    """The x with I_x(a, b) = q. I_x rises monotonically in x, so bisection on [0, 1] reaches float precision."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _betai(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """The exact two-sided interval on a binomial rate k/n: the beta quantiles B(α/2; k, n−k+1) and
    B(1−α/2; k+1, n−k). Exact rather than Wilson because the milestone counts sit at the edges (0 of 20, 20 of 20),
    where the approximations are worst."""
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"{k} of {n} is not a binomial count")
    tail = (1 - conf) / 2
    lo = 0.0 if k == 0 else _beta_quantile(tail, k, n - k + 1)
    hi = 1.0 if k == n else _beta_quantile(1 - tail, k + 1, n - k)
    return lo, hi


def _reading(k: int, n: int) -> str:
    """The registered reading of k of n seeds at raw-policy 1.0: `reaches` at three quarters or more (15 of 20),
    `does not` at a quarter or fewer (5 of 20), `partial` between."""
    if 4 * k >= 3 * n:
        return "reaches"
    if 4 * k <= n:
        return "does not"
    return "partial"


def _absent(arms: dict, names: tuple) -> dict | None:
    missing = [n for n in names if n not in arms]
    return {"missing": missing} if missing else None


def _describe_gain(deltas: list) -> dict:
    """A registered descriptive difference: per-seed deltas, their mean and the one-sided 95% bounds of the same
    sign-flip inversion the claims use — with no alpha, so no p and no verdict."""
    g = _paired_gain(deltas, 0.05)
    return {"deltas": deltas, **{k: g[k] for k in ("mean_delta", "sd", "n", "lower", "upper")}}


def _ring_ceiling(arm: dict) -> dict:
    perfect = sum(1 for s in arm["seeds"] if not s["policy_fail_keys"])
    n = len(arm["seeds"])
    return {"arm": "RxS", "perfect": perfect, "n": n, "reading": _reading(perfect, n)}


def _holdout_generalisation(arms: dict) -> dict:
    """D3: raw-policy failures inside H — the keys RxS_H withheld from its siblings — for Rx (no siblings), RxS_H
    and RxS (siblings everywhere), and the part of the sibling effect on H that survives withholding it,
    (Rx − RxS_H) / (Rx − RxS), None when there is no effect to divide. Read again over ONE key set per seed — the
    H keys that seed of RxS_H never trained on (no row of either kind on any pass) — counted for every arm, so the
    ratio compares failures on the same states rather than shrinking with an arm's own training set."""
    holdout = arms["RxS_H"]["config"]["sibling_holdout"]
    by = {a: _by_seed(arms[a]) for a in ("Rx", "RxS_H", "RxS")}
    order = sorted(by["RxS_H"])

    def trained(row):
        return {k for p in row["passes"] for k, sp_n, _sp_ok, sib_n, _sib_ok in p["dose"] if sp_n + sib_n > 0}

    def count(row, keys):
        return sum(1 for k in set(row["policy_fail_keys"]) if _held_out(k, holdout) and (keys is None or k in keys))
    ref = {s: {k for k in _dose_keys(by["RxS_H"][s]).union(*(by[a][s]["policy_fail_keys"] for a in by))
               if _held_out(k, holdout)} - trained(by["RxS_H"][s]) for s in order}
    out: dict = {"holdout": holdout}
    for label, keys_of in (("H", lambda s: None), ("H_never_trained", lambda s: ref[s])):
        means = {a: sum(count(by[a][s], keys_of(s)) for s in order) / len(order) for a in by}
        gap = means["Rx"] - means["RxS"]
        out[label] = {"mean_failures": means,
                      "generalisation_ratio": (means["Rx"] - means["RxS_H"]) / gap if gap else None}
    out["H_never_trained"].update(key_set="H keys the RxS_H seed never trained on, the same set for every arm",
                                  mean_keys=sum(len(ref[s]) for s in order) / len(order))
    return out


def _sum_counts(rows: list) -> dict:
    out: dict = {}
    for row in rows:
        for k, v in row.items():
            out[k] = _sum_counts([out.get(k, {}), v]) if isinstance(v, dict) else out.get(k, 0) + v
    return out


def _label_probes(arm: dict) -> dict:
    """D6: each label probe summed over seeds — the 32-sim `target_labels` beside the exact-leaf probe per c_scale —
    or None when any seed did not record it: an unrecorded probe is not a zero."""
    out = {}
    for field in ("target_labels", "target_labels_exact_leaf"):
        rows = [row.get(field) for row in arm["seeds"]]
        out[field] = None if any(r is None for r in rows) else _sum_counts(rows)
    return out


def _trajectory(arm: dict) -> list:
    by: dict = {}
    for row in arm["seeds"]:
        for p in row["passes"]:
            by.setdefault(p["pass"], []).append(len(p["policy_fail_keys"]))
    return [[p, sum(v) / len(v)] for p, v in sorted(by.items())]


def _cell_table(arm: dict) -> list:
    """Per seed: its raw-policy failures and, per key, the rows and right-argmax rows it trained on over the
    relabelled passes — the labels that could have repaired it."""
    table = []
    for row in arm["seeds"]:
        n: dict = {}
        ok: dict = {}
        for p in _relabelled(row):
            for k, sp_n, sp_ok, sib_n, sib_ok in p["dose"]:
                n[k] = n.get(k, 0) + sp_n + sib_n
                ok[k] = ok.get(k, 0) + sp_ok + sib_ok
        table.append((set(row["policy_fail_keys"]), n, ok))
    return table


def _count_cells(table: list, keys, share: float) -> dict:
    """D8 over (seed, key) cells: `never_delivered` got no label on a relabelled pass; `delivered_correct` got labels
    whose argmax was right at least `share` of the time — a cell that still fails there is failed by the net, not
    by its labels."""
    out = dict.fromkeys(("cells", "failed", "never_delivered", "never_delivered_failed", "delivered_correct",
                         "delivered_correct_failed"), 0)
    for fails, n, ok in table:
        for k in keys:
            failed = k in fails
            delivered = n.get(k, 0) > 0
            correct = delivered and ok[k] >= share * n[k]
            out["cells"] += 1
            out["failed"] += failed
            out["never_delivered"] += not delivered
            out["never_delivered_failed"] += failed and not delivered
            out["delivered_correct"] += correct
            out["delivered_correct_failed"] += failed and correct
    return out


def _delivery_cells(arm: dict, tkeys: set, recurrent: set | None) -> dict:
    table = _cell_table(arm)
    return {"T": _count_cells(table, tkeys, C46_CELL_SHARE),
            "P": None if recurrent is None else _count_cells(table, recurrent, C46_CELL_SHARE),
            "per_key": [[k, _count_cells(table, {k}, C46_CELL_SHARE)] for k in sorted(tkeys | (recurrent or set()))]}


def _mean_ci(values: list, tcrit: float) -> dict:
    m = sum(values) / len(values)
    if len(values) < 2:
        return {"mean": m, "ci": None}
    half = tcrit * math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1) / len(values))
    return {"mean": m, "ci": [m - half, m + half]}


def _failure_geography(arm: dict, children: dict, scopes: dict) -> dict:
    """D9: each raw-policy failure placed as `visited` (self-play trained on it — read from the dose over EVERY
    canonical key, since `visits` covers failable keys only and would drop the children of positions where every
    move is optimal), `ring` (one move off a visited key) or `far` (two or more plies away), within each scope, as a
    per-seed mean with a two-sided 95% t interval."""
    tcrit = t_critical(len(arm["seeds"]) - 1)
    per = {scope: {c: [] for c in ("visited", "ring", "far")} for scope in scopes}
    for row in arm["seeds"]:
        visited = {k for p in row["passes"] for k, sp_n, *_rest in p["dose"] if sp_n > 0}
        ring = sibling_ring(children, visited)
        where = {k: "visited" if k in visited else "ring" if k in ring else "far" for k in row["policy_fail_keys"]}
        for scope, inside in scopes.items():
            for c, counts in per[scope].items():
                counts.append(sum(1 for k, w in where.items() if w == c and inside(k)))
    return {scope: {c: _mean_ci(v, tcrit) for c, v in classes.items()} for scope, classes in per.items()}


def _milestone(arms: dict) -> dict:
    """M: per arm, the seeds at raw-policy 1.0 and at 48-sim 1.0 with exact 95% intervals, beside the arm's
    search-alone control. The floor test is met iff a GENERIC residual arm (no oracle labels) reaches raw-policy 1.0
    in three quarters of its seeds (15 of 20)."""
    rows = {}
    for name in C46_ARMS:
        if name not in arms:
            continue
        arm, cfg = arms[name], arms[name]["config"]
        n = len(arm["seeds"])
        raw = sum(1 for s in arm["seeds"] if not s["policy_fail_keys"])
        ev = sum(1 for s in arm["seeds"] if not s["failures"])
        rows[name] = {"n": n, "raw_policy_perfect": raw, "raw_policy_ci": list(clopper_pearson(raw, n)),
                      "eval_perfect": ev, "eval_ci": list(clopper_pearson(ev, n)),
                      "search_alone": arm.get("search_alone"),
                      "generic_residual": bool(cfg["arch"].get("residual")) and cfg["policy_target"] is None}
    met = any(r["generic_residual"] and _reading(r["raw_policy_perfect"], r["n"]) == "reaches" for r in rows.values())
    return {"arms": rows, "floor_test_met": met}


def _dose_keys(row: dict) -> set:
    return {k for p in row["passes"] for k, *_rest in p["dose"]}


def _g0_gate(g0: dict, arms: dict) -> dict:
    """The offline capacity gate, read from its OWN evidence (evidence/c46_G0_capacity.json.gz): `passed` is derived
    from its recorded verdict (not_run -> None), and it is refused unless it was trained by the same code as the arms — a
    gate run on other training code says nothing about the net these arms trained."""
    verdict = (g0.get("verdict") or {}).get("verdict") if isinstance(g0.get("verdict"), dict) else None
    if verdict is None or "training_fingerprint" not in g0:
        raise ValueError("G0 must be the capacity gate's evidence (a `verdict` dict and its `training_fingerprint`) — "
                         "a bare outcome cannot be tied to the code that produced it")
    fps = {a["training_fingerprint"] for a in arms.values()}
    if fps and g0["training_fingerprint"] not in fps:
        raise ValueError(f"G0 was trained by different code (training fingerprint {g0['training_fingerprint']}) than "
                         f"the arms ({sorted(fps)})")
    passed = True if verdict == "passed" else None if verdict == "not_run" else False
    return {"passed": passed, "verdict": verdict, "training_fingerprint": g0["training_fingerprint"],
            "summary": g0.get("summary")}


def c46_report(arms: dict, target: dict, game, g0: dict, reference: dict | None = None, notes: dict | None = None,
               gate_alpha: float = 0.05) -> dict:
    """Emit EXACTLY the §C.46 pre-registered claims; everything else goes under `descriptive`.

    `arms` maps registered arm names to their evidence, `target` is the evidence the 17-state target was localized
    from, `game` the game whose rules the sibling closure is re-derived from, `g0` the offline capacity gate (a
    boolean `passed` is required), `reference` the arm P is read from (c45_mixed_R200) and `notes` registered
    per-key annotations echoed into D8. Integrity is refused first (`_c46_integrity`). Per claim, in order:
    `stopped` when one of its G0/G1 gates failed; `not_reached` when an earlier claim of its chain was not
    supported; `not_run` when an arm is absent; then `_judge` — `not_delivered` (GE for the E chain, G2 for the R
    chain), `supported` at p < its chain alpha (a target claim needs G3, else `moved`), `refuted` when the one-sided
    95% upper bound on the gain lies below the SESOI (worded "effect bounded below X per seed"), else
    `inconclusive`. Any verdict but `supported` halts its chain."""
    _c46_integrity(arms, target)
    g0 = _g0_gate(g0, arms)
    tkeys = {f["key"] for s in target["seeds"] for f in s["failures"]}
    children = one_ply_children(game)
    gates: dict = {"G0": g0}
    for g, arm in C46_G1_ARMS.items():
        gates[g] = {"arm": arm, "passed": None}
        if arm in arms:
            gates[g].update(replication(arms[arm], tkeys, gate_alpha))
            gates[g]["passed"] = gates[g]["verdict"]
    halted = dict.fromkeys(C46_CHAIN_ALPHA, False)
    claims: dict = {}
    descriptive: dict = {}
    for name, chain, ctrl, trt, metric, stops, gate, share in C46_CLAIMS:
        a = C46_CHAIN_ALPHA[chain]
        claim = {"chain": chain, "alpha": a, "metric": metric, "control": ctrl, "treat": trt,
                 "gates": {g: gates[g] for g in stops}}
        if any(gates[g]["passed"] is False for g in stops):
            claim["verdict"] = "stopped"
        elif any(gates[g]["passed"] is None and g == "G0" for g in stops):
            claim["verdict"] = "not_run"
        elif halted[chain]:
            claim["verdict"] = "not_reached"
        elif ctrl not in arms or trt not in arms:
            claim["verdict"] = "not_run"
        elif metric == "policy_failures":
            result, own = _failure_claim(arms[ctrl], arms[trt], a, gate, share, children, gate_alpha)
            claim.update(result)
            claim["gates"].update(own)
        else:
            r = ab_report(arms[ctrl], arms[trt], target, manipulation="dose", treatment_keys=C46_PAIRS[(ctrl, trt)][0],
                          alpha_each=a, migration_margin=C46_MIGRATION_MARGIN, manipulation_alpha=gate_alpha)
            descriptive[f"{name}: {trt} vs {ctrl}"] = r
            result, own = _target_claim(r, arms[ctrl], arms[trt], metric, a, share, gate_alpha)
            claim.update(result)
            claim["gates"].update(own)
        if claim["verdict"] == "refuted":
            claim["statement"] = (f"effect bounded below {round(claim['upper'], 3) + 0.0:.3f} {C46_UNITS[metric]} "
                                  f"per seed")
        claims[name] = claim
        if claim["verdict"] != "supported":
            halted[chain] = True

    recurrent = recurrent_keys(reference) if reference is not None else None
    present = [a for a in C46_ARMS if a in arms]
    fails = {a: [len(_by_seed(arms[a])[s]["policy_fail_keys"]) for s in sorted(_by_seed(arms[a]))] for a in present}
    scopes = {"all": lambda k: True, "in_T": lambda k: k in tkeys, "out_T": lambda k: k not in tkeys}
    if recurrent is not None:
        scopes.update({"in_P": lambda k: k in recurrent, "out_P": lambda k: k not in recurrent})
    descriptive.update({
        "T": sorted(tkeys), "P": None if recurrent is None else sorted(recurrent),
        "D1": _absent(arms, ("RxS",)) or _ring_ceiling(arms["RxS"]),
        "D2": _absent(arms, ("Rx", "RxS")) or _describe_gain([x - y for x, y in zip(fails["Rx"], fails["RxS"])]),
        "D3": _absent(arms, ("Rx", "RxS", "RxS_H")) or _holdout_generalisation(arms),
        "D4": _absent(arms, ("R200S", "RxS")) or _describe_gain([x - y for x, y in zip(fails["R200S"], fails["RxS"])]),
        "D5": _absent(arms, ("R200", "R200S", "R32", "R32S")) or _describe_gain(
            [(w - x) - (y - z) for w, x, y, z in zip(fails["R200"], fails["R200S"], fails["R32"], fails["R32S"])]),
        "D6": {a: _label_probes(arms[a]) for a in present},
        "D7": {"g0": g0, "raw_policy_by_pass": {a: _trajectory(arms[a]) for a in present}},
        "D8": {"share": C46_CELL_SHARE, "notes": [[k, v] for k, v in (notes or {}).items()],
               "arms": {a: _delivery_cells(arms[a], tkeys, recurrent) for a in present}},
        "D9": {a: _failure_geography(arms[a], children, scopes) for a in present},
        "M": _milestone(arms),
    })
    return {"claims": claims, "descriptive": descriptive}
