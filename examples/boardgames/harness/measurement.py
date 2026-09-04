"""§C.9 MEASUREMENT INTEGRITY — the primitives every reported strength number must go through.

WHY THIS EXISTS (2026-09-03): a phantom result ("0.969 forced-win conversion, best ever") survived for days and
drove real decisions. Four compounding errors produced it, all of them ours:
  E1 the promotion gate SELECTED on the same roots the scorecard REPORTED (selection on the test set);
  E2 the headline was a max over 40 checkpoints on 32 roots (winner's curse: true rate ~0.85, E[max] ~0.96);
  E3 comparisons mixed provenance — one net's SELECTED-best against another's FINAL — manufacturing a difference
     that vanished (p=0.75) once both were compared final-vs-final on identical roots;
  E4 8 architectures were ranked on 16 roots, where ±1 root = 0.06 and the true spread was ~0.08.
The module is deliberately small and pure (no torch, no game imports) so it is trivially testable and cheap to
call from every reporting path. See tests/test_measurement.py — one test per error above.
"""
from __future__ import annotations

import math

# Seed ROLES are disjoint by construction. A seed used to CHOOSE (gates, sweeps, early stopping, arch picks) may
# never be used to REPORT, and vice versa — E1 is otherwise a one-line mistake away, and it is invisible in review.
SELECTION_SEEDS = frozenset({1013, 2027, 3041})
MEASUREMENT_SEEDS = frozenset({7, 99, 131, 257})


def assert_seed_roles_disjoint(selection_seed: int, measurement_seed: int) -> None:
    """Refuse the E1 configuration: choosing and reporting on the same root draw."""
    if selection_seed == measurement_seed:
        raise ValueError(
            f"selection and measurement seeds are identical ({selection_seed}) — this selects on the test set (E1)")
    if selection_seed in MEASUREMENT_SEEDS:
        raise ValueError(f"seed {selection_seed} is reserved for MEASUREMENT; do not select on it (E1)")
    if measurement_seed in SELECTION_SEEDS:
        raise ValueError(f"seed {measurement_seed} is reserved for SELECTION; do not report on it (E1)")


def wilson_interval(converted: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — the honest uncertainty on a conversion rate. Used instead of a bare point estimate
    because '0.969' and '0.85' are the same measurement at n=32 (E2/E4)."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = converted / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def expected_selected_max(true_rate: float, n_roots: int, k: int) -> float:
    """E[max over k independent Binomial(n_roots, true_rate)/n_roots] — how high a number you EXPECT to report
    purely by picking the best of k candidates on n roots. This is the winner's curse (E2) made numeric: with
    true_rate=0.85, n=32, k=40 it returns ~0.96, which is what we reported as a breakthrough."""
    if k < 1:
        raise ValueError("k (number of candidates selected from) must be >= 1")
    if n_roots <= 0:
        raise ValueError("n_roots must be positive")
    # exact: P(max <= x) = F(x)^k over the discrete binomial support
    log_c = [math.lgamma(n_roots + 1) - math.lgamma(i + 1) - math.lgamma(n_roots - i + 1) for i in range(n_roots + 1)]
    pmf = []
    for i in range(n_roots + 1):
        if true_rate <= 0.0:
            pmf.append(1.0 if i == 0 else 0.0)
        elif true_rate >= 1.0:
            pmf.append(1.0 if i == n_roots else 0.0)
        else:
            pmf.append(math.exp(log_c[i] + i * math.log(true_rate) + (n_roots - i) * math.log1p(-true_rate)))
    cdf, run = [], 0.0
    for v in pmf:
        run += v
        cdf.append(min(1.0, run))
    exp_max = 0.0
    prev = 0.0
    for i in range(n_roots + 1):
        p_max_i = cdf[i] ** k - prev
        prev = cdf[i] ** k
        exp_max += (i / n_roots) * p_max_i
    return exp_max


def mcnemar_exact(a: list[int], b: list[int]) -> dict:
    """Exact (binomial) McNemar on PAIRED per-root outcomes — the only valid way to compare two nets (E3).
    `a` and `b` must be 0/1 outcomes on the SAME roots in the same order."""
    if len(a) != len(b):
        raise ValueError("paired comparison requires equal-length outcome vectors on identical roots (E3)")
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n_d = only_a + only_b
    if n_d == 0:
        p = 1.0
    else:
        k = min(only_a, only_b)
        tail = sum(math.comb(n_d, i) for i in range(k + 1))
        p = min(1.0, 2.0 * tail / (2 ** n_d))
    return {"only_a": only_a, "only_b": only_b, "discordant": n_d, "both": sum(1 for x, y in zip(a, b) if x and y),
            "neither": sum(1 for x, y in zip(a, b) if not x and not y), "p": p}


def min_roots_for(delta: float, base: float, z_alpha: float = 1.96, z_beta: float = 0.84) -> int:
    """Roots needed to detect a `delta` difference around `base` at ~80% power — the sanity check that would have
    stopped us ranking 8 architectures on 16 roots (E4). Two-proportion normal approximation."""
    if not 0 < base < 1 or delta <= 0:
        raise ValueError("base must be in (0,1) and delta > 0")
    p1 = min(0.999, base)
    p2 = min(0.999, base + delta)
    pbar = (p1 + p2) / 2
    num = (z_alpha * math.sqrt(2 * pbar * (1 - pbar)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return max(1, math.ceil(num / (delta ** 2)))


def promotion_margin(rate: float, n: int) -> float:
    """How much a challenger must beat an incumbent by before the gain is believable: ONE standard error of the
    incumbent's own probe. Chosen deliberately over a full 95% half-width — on a small probe that would block even
    a perfect challenger (measured: at rate .833/n=12 the half-width is .20, so nothing could ever promote), while
    one SE blocks noise-level gains (+0.084 at n=12) and admits decisive ones. The real fix is a BIGGER probe: the
    margin shrinks as sqrt(n), so a 512-root probe resolves +0.03 that a 12-root probe never can."""
    if n <= 0:
        raise ValueError("n must be positive")
    p_ = min(max(rate, 0.0), 1.0)
    return math.sqrt(p_ * (1 - p_) / n)


def report_rate(converted: int, n: int, seed: int, selected_from: int = 1,
                true_rate_hint: float | None = None) -> dict:
    """The ONLY sanctioned way to report a conversion rate. Carries n, seed and a Wilson CI always, and when the
    number was SELECTED (best-of-k checkpoints/archs) it says so and attaches the expected winner's-curse
    inflation, so a selected maximum can never again be quoted as if it were an unbiased estimate (E2)."""
    if selected_from < 1:
        raise ValueError("selected_from (candidates) must be >= 1")
    rate = converted / n
    lo, hi = wilson_interval(converted, n)
    out = {"rate": rate, "converted": converted, "n": n, "seed": seed, "ci": (lo, hi),
           "selected": selected_from > 1, "candidates": selected_from, "caveat": ""}
    if selected_from > 1:
        hint = true_rate_hint if true_rate_hint is not None else lo
        out["expected_selected_max"] = expected_selected_max(hint, n, selected_from)
        out["caveat"] = (f"SELECTED best-of-{selected_from} on n={n}: a rate near "
                         f"{out['expected_selected_max']:.3f} is EXPECTED by selection alone even if the true rate "
                         f"is {hint:.3f}. Re-measure the chosen candidate on a held-out measurement seed.")
    return out


def rank_or_refuse(arms: dict[str, tuple[int, int]], null_arm_spread: float | None = None,
                   alpha: float = 0.20) -> dict:
    """The guard that turns "the sweep found nothing" from a POST-MORTEM into an automatic gate (§C.10 BUILD #2).

    `arms` maps name -> (converted, n). Emits a ranking ONLY when (a) a chi-square homogeneity test rejects "all
    arms equal" at `alpha`, and (b) the top-two gap exceeds the sweep's own measured noise floor (`null_arm_spread`
    — the observed spread between two architecturally-equivalent arms). Otherwise it returns the required n and
    refuses. Run against our real 8-arch/16-root screen this returns REFUSED (p=0.665): we ranked anyway, and the
    ranking reversed under a different root seed."""
    if not arms:
        raise ValueError("no arms")
    tot_k = sum(k for k, _ in arms.values())
    tot_n = sum(n for _, n in arms.values())
    if tot_n == 0:
        raise ValueError("no observations")
    pbar = tot_k / tot_n
    chi2 = 0.0
    for k, n in arms.values():
        exp = n * pbar
        if 0 < exp < n:
            chi2 += (k - exp) ** 2 / exp + ((n - k) - (n - exp)) ** 2 / (n - exp)
    df = max(1, len(arms) - 1)
    # survival function of chi-square via the regularised upper incomplete gamma (no scipy dependency)
    p = _chi2_sf(chi2, df)
    order = sorted(arms, key=lambda a: -(arms[a][0] / arms[a][1]))
    rates = [arms[a][0] / arms[a][1] for a in order]
    gap = (rates[0] - rates[1]) if len(rates) > 1 else 1.0
    required = min_roots_for(max(gap, 0.01), max(0.01, min(0.99, pbar)))
    out = {"homogeneity_p": p, "chi2": chi2, "df": df, "order": order,
           "top_gap": gap, "required_n": required, "ranked": False, "verdict": ""}
    if p > alpha:
        out["verdict"] = (f"REFUSED: no evidence the arms differ (chi2={chi2:.2f}, df={df}, p={p:.3f}). "
                          f"To resolve the observed top gap of {gap:.3f} you need ~{required} roots per arm.")
        return out
    if null_arm_spread is not None and gap <= null_arm_spread:
        out["verdict"] = (f"REFUSED: top gap {gap:.3f} is within the sweep's own null-arm noise floor "
                          f"({null_arm_spread:.3f}) — not a difference.")
        return out
    out["ranked"] = True
    out["verdict"] = f"RANKED: {order[0]} leads by {gap:.3f} (chi2 p={p:.3f})."
    return out


def _chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-square distribution (regularised incomplete gamma Q(df/2, x/2))."""
    if x <= 0:
        return 1.0
    a = df / 2.0
    z = x / 2.0
    if z < a + 1:  # series for P, then Q = 1 - P
        term = 1.0 / a
        total = term
        n = 1
        while n < 1000:
            term *= z / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
            n += 1
        return max(0.0, min(1.0, 1.0 - total * math.exp(-z + a * math.log(z) - math.lgamma(a))))
    # continued fraction for Q
    tiny = 1e-300
    b = z + 1 - a
    c = 1 / tiny
    d = 1 / b if b != 0 else 1 / tiny
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return max(0.0, min(1.0, h * math.exp(-z + a * math.log(z) - math.lgamma(a))))
