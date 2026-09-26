"""§C.48 T4 — with the process fixes T2 pointed at, does the generic self-play process give a RAW net that plays
tic-tac-toe perfectly? Pre-registered.

T2 (harness/floor_converge.py, h45/h46) was refuted in both arms. Its augment arm sat near 4 failures until the
8,000-row FIFO buffer filled with repeats of the same ~1,150 positions and began evicting, then got worse; the
learning rate never decayed, so the fit stayed on a noise floor. T4 runs the same R200S recipe on fresh seeds with
three knobs of harness.neural.train_alphazero switched on:

  buffer_unique          one row per position, refreshed to the newest end when seen again — no FIFO over repeats;
  settle_epochs          after the last iteration, SETTLE_EPOCHS more epochs on the last training set with the
                         learning rate decaying linearly to SETTLE_LR_FINAL;
  record_self_agreement  after every pass, the share of buffer positions where the raw policy's argmax is one of its
                         own label's best moves — observation only, read here against the truth.

The raw net (argmax over legal moves, one forward, no averaging) is scored at all 4,520 non-terminal raw positions
after every pass: 30 iterations and the settle, T4_PASSES in all. Arms and bars are T2's: SUPPORTED when at least
T4_SUPPORT_AT of the 10 seeds end the settle with zero failures, REFUTED at T4_REFUTE_AT or fewer, INCONCLUSIVE
between. Neither fix adds coverage, so the no_augment arm is expected to stay refuted (self-play reaches ~1,150 of
the 4,520 raw positions); it is run to measure what the fixes do there, not because they are expected to fix it.

Descriptively: how many seeds were already perfect before settling (the buffer fix alone), relapses, and whether
full self-agreement could serve as a solver-free stopping rule — a FALSE STOP is a pass where the net agreed with
all its own labels yet failed somewhere. Any integrity failure, or unreadable evidence, makes every verdict NOT_RUN."""
from __future__ import annotations

from harness.floor_converge import T2_ARMS

T4_SEEDS = tuple(range(311, 321))
T4_ITERATIONS = 30
T4_PASSES = T4_ITERATIONS + 1
T4_POSITIONS = 4520
T4_PARAMS = 57453
T4_ERA = "52ea07e577cb"
T4_MEASUREMENT_FP = "f8e92b3afd73"
T4_SETTLE_EPOCHS = 30
T4_SETTLE_LR_FINAL = 1e-5
T4_SUPPORT_AT = 8
T4_REFUTE_AT = 5
_FIXES = {"buffer_unique": True, "settle_epochs": T4_SETTLE_EPOCHS, "settle_lr_final": T4_SETTLE_LR_FINAL,
          "record_self_agreement": True}
T4_ARMS = {name: {**config, **_FIXES} for name, config in T2_ARMS.items()}


def _history_problems(where: str, history) -> list:
    if not isinstance(history, list) or len(history) != T4_PASSES or not all(isinstance(h, dict) for h in history):
        return [f"{where}: not {T4_PASSES} history entries"]
    problems = []
    iterations, settle = history[:-1], history[-1]
    if not all((h.get("siblings") or 0) > 0 for h in iterations[1:]):
        problems.append(f"{where}: siblings were not added on every relabelled iteration")
    if not all("merged" in h for h in iterations):
        problems.append(f"{where}: an iteration did not run the unique buffer")
    if settle.get("iteration") != "settle" or settle.get("epochs") != T4_SETTLE_EPOCHS \
            or settle.get("lr_final") != T4_SETTLE_LR_FINAL:
        problems.append(f"{where}: the last pass is not the registered settle")
    if not all(isinstance(h.get("self_agreement"), (int, float)) for h in history):
        problems.append(f"{where}: self-agreement was not recorded on every pass")
    return problems


def _seed_problems(arm: str, row: dict) -> list:
    where = f"{arm} seed {row.get('seed')}"
    problems = []
    curve = row.get("strict_failures_per_pass")
    if not isinstance(curve, list) or len(curve) != T4_PASSES:
        problems.append(f"{where}: not {T4_PASSES} scored passes")
    elif len(row.get("final_failing_positions") or []) != curve[-1]:
        problems.append(f"{where}: the final failure list does not match the final pass's count")
    if row.get("positions") != T4_POSITIONS:
        problems.append(f"{where}: scored {row.get('positions')} positions, not {T4_POSITIONS}")
    if row.get("params") != T4_PARAMS:
        problems.append(f"{where}: built {row.get('params')} parameters")
    if row.get("games_per_pass") != [T2_ARMS["augment"]["selfplay"]] * T4_ITERATIONS + [0]:
        problems.append(f"{where}: the recorder did not see the registered games on every iteration and none while "
                        "settling")
    return problems + _history_problems(where, row.get("history"))


def integrity(arms: dict) -> list:
    """Every reason the evidence is not the registered T4. Returns the problems; never raises."""
    if not isinstance(arms, dict):
        return ["no arms"]
    problems = []
    for name, config in T4_ARMS.items():
        arm = arms.get(name)
        if not isinstance(arm, dict) or not isinstance(arm.get("seeds"), list) \
                or not isinstance(arm.get("config"), dict):
            problems.append(f"arm {name} is missing or has no seeds/config")
            continue
        if arm.get("training_fingerprint") != T4_ERA:
            problems.append(f"{name}: training era {arm.get('training_fingerprint')}, registered {T4_ERA}")
        if arm.get("measurement_fingerprint") != T4_MEASUREMENT_FP:
            problems.append(f"{name}: measurement code {arm.get('measurement_fingerprint')}, registered "
                            f"{T4_MEASUREMENT_FP}")
        for field, value in {**config, "seeds": list(T4_SEEDS)}.items():
            if arm["config"].get(field) != value:
                problems.append(f"{name}: config {field} is {arm['config'].get(field)!r}, registered {value!r}")
        seeds = sorted(r.get("seed") for r in arm["seeds"] if isinstance(r, dict))
        if seeds != list(T4_SEEDS):
            problems.append(f"{name}: seeds {seeds}, not the registered {list(T4_SEEDS)}")
        for row in arm["seeds"]:
            problems.extend(_seed_problems(name, row) if isinstance(row, dict) else [f"{name}: a seed row is not a record"])
    return problems


def _first(values: list, pred):
    return next((i + 1 for i, v in enumerate(values) if pred(v)), None)


def _judge(arm: dict) -> dict:
    rows = sorted(arm["seeds"], key=lambda r: r["seed"])
    curves = [r["strict_failures_per_pass"] for r in rows]
    agreements = [[h["self_agreement"] for h in r["history"]] for r in rows]
    perfect = sum(1 for c in curves if c[-1] == 0)
    verdict = ("supported" if perfect >= T4_SUPPORT_AT else "refuted" if perfect <= T4_REFUTE_AT else "inconclusive")
    first = [_first(c, lambda f: f == 0) for c in curves]
    relapsed = [r["seed"] for r, c, z in zip(rows, curves, first) if z is not None and any(f > 0 for f in c[z:])]
    return {"verdict": verdict, "perfect": perfect, "n": len(rows),
            "descriptives": {"final_failures": [c[-1] for c in curves],
                             "perfect_before_settling": sum(1 for c in curves if c[-2] == 0),
                             "mean_failures_per_pass": [sum(c[i] for c in curves) / len(curves)
                                                        for i in range(T4_PASSES)],
                             "first_zero_pass": first, "relapsed_seeds": relapsed,
                             "self_agreement": {
                                 "final": [a[-1] for a in agreements],
                                 "first_full_pass": [_first(a, lambda v: v == 1.0) for a in agreements],
                                 "false_stops": sum(1 for c, a in zip(curves, agreements)
                                                    for f, v in zip(c, a, strict=True) if v == 1.0 and f > 0)}}}


def settle_report(arms: dict) -> dict:
    """The pre-registered T4 verdict for each arm, or NOT_RUN for all when the evidence is not the registered run."""
    try:
        problems = integrity(arms)
        if not problems:
            return {"integrity": [], **{name: _judge(arms[name]) for name in T4_ARMS}}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, **{name: {"verdict": "not_run"} for name in T4_ARMS}}
