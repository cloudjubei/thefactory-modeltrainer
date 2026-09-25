"""§C.48 T2 — does the generic self-play process, RUN TO CONVERGENCE, give a raw net that plays tic-tac-toe
perfectly? Pre-registered.

§C.47 met the floor only under the symmetry-averaged operator, on a deliberately tiny fixed budget (6 iterations,
288 self-play games). T1 (scripts/smallest_net.py) measures separately what size of net can hold perfect play at
all. T2 removes the budget: the §C.46 R200S recipe (residual 57,453-param net, 200-sim relabelling of the whole
buffer, one-ply siblings, steps-matched) runs 30 iterations, and the RAW net — argmax over legal moves, one
forward, no averaging, no canonical images — is scored at every one of the 4,520 non-terminal raw positions after
every pass. Two arms, judged separately:

  augment      the generic process as it is (training rows augmented by the verified symmetries);
  no_augment   no symmetry anywhere — the process must learn every orientation from its own data.

Per arm: SUPPORTED when at least T2_SUPPORT_AT of the 10 seeds end the final pass with zero failures (the per-seed
rate is then above 0.55 at one-sided 95%), REFUTED at T2_REFUTE_AT or fewer (below 0.78), INCONCLUSIVE between.
The FINAL pass decides: a net that was perfect earlier and relapsed is not perfect. Descriptively, a seed RELAPSED
if any pass after its first perfect one had failures again, whether or not it recovered. Any integrity failure, or
unreadable evidence, makes every verdict NOT_RUN."""
from __future__ import annotations

T2_SEEDS = tuple(range(301, 311))
T2_ITERATIONS = 30
T2_POSITIONS = 4520
T2_PARAMS = 57453
T2_ERA = "27933b3a3bba"
T2_MEASUREMENT_FP = "d1d15674c442"
T2_SUPPORT_AT = 8
T2_REFUTE_AT = 5
_RECIPE = {"game": "tictactoe", "arch": {"channels": 32, "blocks": 3, "head_hidden": 32, "residual": True},
           "params": T2_PARAMS, "iterations": T2_ITERATIONS, "selfplay": 48, "train_sims": 32, "opening_plies": 2,
           "opening_zero_frac": 0.5, "reanalyze_frac": 1.0, "reanalyze_sims": 200, "reanalyze_siblings": True,
           "steps_matched": True, "epochs": 6, "batch_size": 64, "lr": 0.001, "buffer_cap": 8000, "gumbel": True,
           "c_scale": 0.1}
T2_ARMS = {"augment": {**_RECIPE, "augment": True}, "no_augment": {**_RECIPE, "augment": False}}


def _seed_problems(arm: str, row: dict) -> list:
    where = f"{arm} seed {row.get('seed')}"
    problems = []
    curve = row.get("strict_failures_per_pass")
    if not isinstance(curve, list) or len(curve) != T2_ITERATIONS:
        problems.append(f"{where}: not {T2_ITERATIONS} scored passes")
    elif len(row.get("final_failing_positions") or []) != curve[-1]:
        problems.append(f"{where}: the final failure list does not match the final pass's count")
    if row.get("positions") != T2_POSITIONS:
        problems.append(f"{where}: scored {row.get('positions')} positions, not {T2_POSITIONS}")
    if row.get("params") != T2_PARAMS:
        problems.append(f"{where}: built {row.get('params')} parameters")
    if row.get("games_per_pass") != [_RECIPE["selfplay"]] * T2_ITERATIONS:
        problems.append(f"{where}: the recorder did not see {_RECIPE['selfplay']} games on every iteration")
    history = row.get("history")
    if not isinstance(history, list) or len(history) != T2_ITERATIONS \
            or not all(isinstance(h, dict) and (h.get("siblings") or 0) > 0 for h in history[1:]):
        problems.append(f"{where}: siblings were not added on every relabelled iteration")
    return problems


def integrity(arms: dict) -> list:
    """Every reason the evidence is not the registered T2. Returns the problems; never raises."""
    if not isinstance(arms, dict):
        return ["no arms"]
    problems = []
    for name, config in T2_ARMS.items():
        arm = arms.get(name)
        if not isinstance(arm, dict) or not isinstance(arm.get("seeds"), list) \
                or not isinstance(arm.get("config"), dict):
            problems.append(f"arm {name} is missing or has no seeds/config")
            continue
        if arm.get("training_fingerprint") != T2_ERA:
            problems.append(f"{name}: training era {arm.get('training_fingerprint')}, registered {T2_ERA}")
        if arm.get("measurement_fingerprint") != T2_MEASUREMENT_FP:
            problems.append(f"{name}: measurement code {arm.get('measurement_fingerprint')}, registered "
                            f"{T2_MEASUREMENT_FP}")
        for field, value in {**config, "seeds": list(T2_SEEDS)}.items():
            if arm["config"].get(field) != value:
                problems.append(f"{name}: config {field} is {arm['config'].get(field)!r}, registered {value!r}")
        seeds = sorted(r.get("seed") for r in arm["seeds"] if isinstance(r, dict))
        if seeds != list(T2_SEEDS):
            problems.append(f"{name}: seeds {seeds}, not the registered {list(T2_SEEDS)}")
        for row in arm["seeds"]:
            problems.extend(_seed_problems(name, row) if isinstance(row, dict) else [f"{name}: a seed row is not a record"])
    return problems


def _first_zero(curve: list):
    return next((i + 1 for i, f in enumerate(curve) if f == 0), None)


def _judge(arm: dict) -> dict:
    rows = sorted(arm["seeds"], key=lambda r: r["seed"])
    perfect = sum(1 for r in rows if r["strict_failures_per_pass"][-1] == 0)
    verdict = ("supported" if perfect >= T2_SUPPORT_AT else "refuted" if perfect <= T2_REFUTE_AT else "inconclusive")
    curves = [r["strict_failures_per_pass"] for r in rows]
    first = [_first_zero(c) for c in curves]
    relapsed = [r["seed"] for r, c, z in zip(rows, curves, first) if z is not None and any(f > 0 for f in c[z:])]
    return {"verdict": verdict, "perfect": perfect, "n": len(rows),
            "descriptives": {"final_failures": [c[-1] for c in curves],
                             "mean_failures_per_pass": [sum(c[i] for c in curves) / len(curves)
                                                        for i in range(T2_ITERATIONS)],
                             "first_zero_pass": first, "relapsed_seeds": relapsed}}


def converge_report(arms: dict) -> dict:
    """The pre-registered T2 verdict for each arm, or NOT_RUN for all when the evidence is not the registered run."""
    try:
        problems = integrity(arms)
        if not problems:
            return {"integrity": [], **{name: _judge(arms[name]) for name in T2_ARMS}}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, **{name: {"verdict": "not_run"} for name in T2_ARMS}}
