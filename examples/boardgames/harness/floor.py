"""§C.47 leg F — the tic-tac-toe FLOOR on fresh seeds, judged by an operator fixed before the data.

§C.42 set the floor: a generic process that cannot play the smallest game perfectly cannot be called near-optimal.
§C.46 showed the recipe that comes closest (R200S: residual net, 200-sim relabelling, one-ply siblings) and, in its
verification, that the raw net is not equivariant (h39: 0/20 seeds perfect in every orientation, 16/20 under the
policy averaged over the board's symmetries — a reading chosen AFTER the data). This module is the pre-registered
re-test: the same recipe on seeds 81-100, the averaged policy as the operator, and everything that could make the
run a different experiment checked before a verdict is drawn.

  F1 (primary) the floor under π̄ — the softmax over legal moves averaged across the 8 verified isometries, argmax,
     scored at every non-terminal canonical state (π̄ is equivariant, so one image per class covers all 4,520):
     SUPPORTED at >= 15 of 20 seeds perfect, REFUTED at <= 11 (the per-seed rate is then below 0.75 at 95%),
     INCONCLUSIVE between.
  F2 does the operator matter? An exact one-sided sign test over the seeds the two readings disagree on (perfect
     under π̄ but not in every raw orientation, against the reverse), SUPPORTED at p < 0.01, REFUTED if the reverse
     direction reaches it, INCONCLUSIVE otherwise.

Any failed integrity condition makes both NOT_RUN, and so does evidence the report cannot read: a crash must never
reach the register as a refutation. The integrity list below is the only route to NOT_RUN."""
from __future__ import annotations

import math
import random

from harness.ceiling import clopper_pearson

C47_SEEDS = tuple(range(81, 101))
C47_ERA = "27933b3a3bba"
C47_MEASUREMENT_FP = "e5a9cb7b706d"
C47_ISOMETRIES = ("identity", "rot180", "flip_h", "flip_v", "rot90", "rot270", "transpose", "anti_transpose")
C47_SUBGROUPS = {"1": ("identity",), "2": ("identity", "flip_h"), "4": ("identity", "rot180", "flip_h", "flip_v"),
                 "8": C47_ISOMETRIES}
C47_POSITIONS = 4520
C47_PASSES = 6
C47_PARAMS = 57453
C47_SUPPORT_AT = 15
C47_REFUTE_AT = 11
C47_SIGN_ALPHA = 0.01
C47_REPRO = {"seed": 41, "weights_sha": "78052b2a4caed52bedfe5fca4bfc3c12b54098a8ea1a3f4de70460b690fac57a",
             "policy_fail_keys": []}
C47_REFERENCE_CONFIG = {
    "game": "tictactoe", "arch": {"channels": 32, "blocks": 3, "head_hidden": 32, "residual": True},
    "params_expected": 57453, "iterations": 6, "selfplay": 48, "train_sims": 32, "eval_sims": 48, "deep_sims": 400,
    "threads": 1, "opening_plies": 2, "opening_zero_frac": 0.5, "reanalyze_frac": 1.0, "reanalyze_sims": 200,
    "reanalyze_siblings": True, "sibling_key": "canonical", "sibling_holdout": None, "steps_matched": True,
    "policy_target": None, "epochs": 6, "batch_size": 64, "lr": 0.001, "buffer_cap": 8000, "label_draws": 8,
    "label_sims": 32,
    "target_keys": [77, 90, 97, 385, 597, 601, 608, 768, 924, 1504, 1608, 1611, 2098, 2440, 4980, 4983, 5143]}
READINGS = {"symmetrized": "symmetrized_fail_keys", "strict": "image_fail_keys", "canonical": "canonical_fail_keys"}


def _seed_problems(row: dict) -> list:
    where = f"seed {row.get('seed')}"
    problems = []
    if row.get("params") != C47_PARAMS:
        problems.append(f"{where}: built {row.get('params')} parameters, not {C47_PARAMS}")
    o = row.get("orientation")
    if not isinstance(o, dict) or not all(isinstance(o.get(k), list) for k in READINGS.values()):
        return problems + [f"{where}: no complete orientation reading"]
    if o.get("positions") != C47_POSITIONS:
        problems.append(f"{where}: scored {o.get('positions')} positions, not {C47_POSITIONS}")
    if o.get("isometries") != list(C47_ISOMETRIES):
        problems.append(f"{where}: averaged over {o.get('isometries')}, not the verified {list(C47_ISOMETRIES)}")
    groups = row.get("subgroups")
    if not isinstance(groups, dict) or any((groups.get(k) or {}).get("isometries") != list(v)
                                           for k, v in C47_SUBGROUPS.items()):
        problems.append(f"{where}: subgroup readings missing or over other isometries")
    passes = row.get("passes")
    if not isinstance(passes, list) or len(passes) != C47_PASSES \
            or not all(isinstance(p, dict) and isinstance(p.get("dose"), list) for p in passes):
        problems.append(f"{where}: not {C47_PASSES} recorded training passes")
    history = row.get("history")
    if not isinstance(history, list) or len(history) != C47_PASSES \
            or not all(isinstance(h, dict) and (h.get("siblings") or 0) > 0 for h in history[1:]):
        problems.append(f"{where}: siblings were not added on every relabelled iteration")
    if not isinstance(row.get("failures"), list):
        problems.append(f"{where}: no search-agent failures recorded")
    return problems


def integrity(arm: dict, repro: dict) -> list:
    """Every reason the arm is not the registered experiment. Returns the problems; never raises."""
    if not isinstance(arm, dict) or not isinstance(arm.get("seeds"), list) or not isinstance(arm.get("config"), dict):
        return ["the arm evidence has no seeds or config"]
    problems = []
    if arm.get("training_fingerprint") != C47_ERA:
        problems.append(f"training era {arm.get('training_fingerprint')}, registered {C47_ERA}")
    if arm.get("measurement_fingerprint") != C47_MEASUREMENT_FP:
        problems.append(f"measurement code {arm.get('measurement_fingerprint')}, registered {C47_MEASUREMENT_FP}")
    for field, value in C47_REFERENCE_CONFIG.items():
        if arm["config"].get(field) != value:
            problems.append(f"config {field} is {arm['config'].get(field)!r}, the registered recipe's is {value!r}")
    seeds = sorted(s.get("seed") for s in arm["seeds"] if isinstance(s, dict))
    if seeds != list(C47_SEEDS) or arm["config"].get("seeds") != list(C47_SEEDS):
        problems.append(f"seeds are {seeds}, not the registered {C47_SEEDS[0]}-{C47_SEEDS[-1]}")
    for row in arm["seeds"]:
        problems.extend(_seed_problems(row) if isinstance(row, dict) else ["a seed row is not a record"])
    problems.extend(_repro_problems(arm, repro))
    return problems


def _repro_problems(arm: dict, repro: dict) -> list:
    """The driver's own knobs (augment, Gumbel, c_scale, the recorders) sit outside the training era, so the
    recipe is proved unchanged by rebuilding one §C.46 R200S seed bit-for-bit under the same code as the arm."""
    if not isinstance(repro, dict) or not isinstance(repro.get("seeds"), list) or len(repro["seeds"]) != 1:
        return ["no single-seed R200S reproduction"]
    row = repro["seeds"][0]
    if not isinstance(row, dict):
        return ["the reproduction's seed row is not a record"]
    passes = row.get("passes")
    problems = []
    if row.get("seed") != C47_REPRO["seed"] or not isinstance(passes, list) or not passes:
        return [f"the reproduction is not R200S seed {C47_REPRO['seed']} with its training passes"]
    if passes[-1].get("weights_sha") != C47_REPRO["weights_sha"]:
        problems.append(f"R200S seed {C47_REPRO['seed']} rebuilt to {str(passes[-1].get('weights_sha'))[:16]}, not "
                        f"the recorded {C47_REPRO['weights_sha'][:16]} — the recipe is not the one §C.46 ran")
    if sorted(row.get("policy_fail_keys") or []) != C47_REPRO["policy_fail_keys"]:
        problems.append("the rebuilt R200S net misplays other states than the recorded one")
    for field in ("training_fingerprint", "measurement_fingerprint"):
        if repro.get(field) != arm.get(field):
            problems.append(f"the reproduction ran under {field} {repro.get(field)}, the arm under {arm.get(field)}")
    for field, value in C47_REFERENCE_CONFIG.items():
        if (repro.get("config") or {}).get(field) != value:
            problems.append(f"the reproduction's config {field} differs from the registered recipe")
    return problems


def _sign_p(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2): the exact one-sided sign test."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def _bootstrap_mean(values: list, draws: int = 10000, seed: int = 0) -> dict:
    """Mean with a seeded percentile-bootstrap 95% interval — the counts are mostly zeros, where a t-interval
    runs below zero."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(draws))
    return {"mean": sum(values) / n, "ci": [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]}


def _trained_keys(row: dict) -> set:
    return {r[0] for p in row["passes"] for r in p["dose"] if r[1] + r[3] > 0}


def _exposure(arm: dict) -> dict:
    """Per seed, how much of the failable space got a training row, and where each reading's failures sit."""
    universe = set(arm["universe"])
    out = {"never_trained_share": [], "failures": {}}
    for name in ("canonical", "symmetrized"):
        out["failures"][name] = {"trained": 0, "never_trained": 0}
    for row in arm["seeds"]:
        never = universe - _trained_keys(row)
        out["never_trained_share"].append(len(never) / len(universe))
        for name in ("canonical", "symmetrized"):
            fails = set(row["orientation"][READINGS[name]])
            out["failures"][name]["never_trained"] += len(fails & never)
            out["failures"][name]["trained"] += len(fails - never)
    shares = out["never_trained_share"]
    out["mean_never_trained_share"] = sum(shares) / len(shares)
    return out


def _f1(perfect: int, n: int, canonical: int, strict: int, trained_share: float) -> dict:
    lower = clopper_pearson(perfect, n, conf=0.90)[0]
    upper = clopper_pearson(perfect, n, conf=0.90)[1]
    if perfect >= C47_SUPPORT_AT:
        verdict = "supported"
        reading = (f"Floor met under the symmetry-averaged policy — the net's softmax averaged over the 8 verified "
                   f"isometries, an operator adopted after §C.46 found raw nets are not equivariant: {perfect}/{n} fresh "
                   f"seeds perfect at every position; the per-seed rate is at least {lower:.2f} (one-sided 95%). The "
                   f"raw net scored canonically, the reading §C.46 registered, was perfect in {canonical}/{n}, and in "
                   f"every raw orientation in {strict}/{n}. Met by near-enumeration of 627 states with labels from a "
                   f"search near-exhaustive at this game size: {trained_share:.0%} of the failable positions received "
                   f"a training row. Necessary for the north star; not evidence the mechanism transfers.")
    elif perfect <= C47_REFUTE_AT:
        verdict = "refuted"
        reading = (f"Floor not met: {perfect}/{n} seeds perfect under the symmetry-averaged policy; the per-seed rate "
                   f"is below 0.75 (one-sided 95% upper bound {upper:.2f}).")
    else:
        verdict = "inconclusive"
        reading = (f"Not decided: {perfect}/{n} perfect under the symmetry-averaged policy lies between the "
                   f"refutation bar (<= {C47_REFUTE_AT}) and the floor (>= {C47_SUPPORT_AT}); the per-seed rate is "
                   f"between {lower:.2f} and {upper:.2f} (one-sided 95% each).")
    return {"verdict": verdict, "perfect": perfect, "n": n, "lower": lower, "upper": upper, "reading": reading}


def _f2(rows: list) -> dict:
    only_avg = sum(1 for r in rows if not r["orientation"]["symmetrized_fail_keys"] and r["orientation"]["image_fail_keys"])
    only_raw = sum(1 for r in rows if r["orientation"]["symmetrized_fail_keys"] and not r["orientation"]["image_fail_keys"])
    n = only_avg + only_raw
    p, reverse = _sign_p(only_avg, n), _sign_p(only_raw, n)
    verdict = "supported" if p < C47_SIGN_ALPHA else "refuted" if reverse < C47_SIGN_ALPHA else "inconclusive"
    return {"verdict": verdict, "only_averaged_perfect": only_avg, "only_strict_perfect": only_raw, "p": p,
            "reverse_p": reverse}


def floor_report(arm: dict, repro: dict) -> dict:
    """The pre-registered §C.47 leg-F verdicts, from the arm's evidence and the R200S reproduction."""
    try:
        problems = integrity(arm, repro)
        if not problems:
            return _judged(arm)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, "F1": {"verdict": "not_run"}, "F2": {"verdict": "not_run"}}


def _judged(arm: dict) -> dict:
    rows = arm["seeds"]
    n = len(rows)
    perfect = {name: sum(1 for r in rows if not r["orientation"][key]) for name, key in READINGS.items()}
    perfect["search"] = sum(1 for r in rows if not r["failures"])
    for order in C47_SUBGROUPS:
        perfect[f"subgroup_{order}"] = sum(1 for r in rows if not r["subgroups"][order]["fail_keys"])
    exposure = _exposure(arm)
    f1 = _f1(perfect["symmetrized"], n, perfect["canonical"], perfect["strict"],
             1 - exposure["mean_never_trained_share"])
    counts = {name: _bootstrap_mean([len(r["orientation"][key]) for r in rows]) for name, key in READINGS.items()}
    counts.update({f"subgroup_{o}_positions": _bootstrap_mean([r["subgroups"][o]["failing_positions"] for r in rows])
                   for o in C47_SUBGROUPS})
    disagree = [r["seed"] for r in rows if r["subgroups"]["8"]["fail_keys"] != r["orientation"]["symmetrized_fail_keys"]]
    return {"integrity": [], "F1": f1, "F2": _f2(rows),
            "descriptives": {"perfect": perfect, "failures": counts, "exposure": exposure,
                             "averaged_reading_disagrees_across_images": disagree,
                             "operator_sanity": arm.get("operator_sanity")}}
