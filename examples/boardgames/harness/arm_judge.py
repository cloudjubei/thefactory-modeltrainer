"""The generic pre-registered verdict for the arms of a tic-tac-toe PROCESS run (§C.49 onward): each arm is a recipe
run on the registered seeds, the raw net scored at every raw position after every pass, and judged on the pass after
the settle. A registration module fixes a SPEC — arms, their parameter counts, seeds, training era, measurement
fingerprint, iterations, positions, and the support/refute bars — and pins itself with this file.

Per arm: SUPPORTED when at least `support_at` seeds end raw-perfect, REFUTED at `refute_at` or fewer, INCONCLUSIVE
between. Any integrity failure, or unreadable evidence, makes every verdict NOT_RUN."""
from __future__ import annotations


def _seed_problems(spec: dict, name: str, row: dict) -> list:
    cfg, where, iterations = spec["arms"][name], f"{name} seed {row.get('seed')}", spec["iterations"]
    problems = []
    curve = row.get("strict_failures_per_pass")
    if not isinstance(curve, list) or len(curve) != iterations + 1:
        problems.append(f"{where}: not {iterations + 1} scored passes")
    elif len(row.get("final_failing_positions") or []) != curve[-1]:
        problems.append(f"{where}: the final failure list does not match the final pass's count")
    if row.get("positions") != spec["positions"]:
        problems.append(f"{where}: scored {row.get('positions')} positions, not {spec['positions']}")
    if row.get("params") != spec["params"][name]:
        problems.append(f"{where}: built {row.get('params')} parameters, not {spec['params'][name]}")
    if row.get("games_per_pass") != [cfg["selfplay"]] * iterations + [0]:
        problems.append(f"{where}: not the registered games on every iteration and none while settling")
    history = row.get("history")
    if not isinstance(history, list) or len(history) != iterations + 1:
        return problems + [f"{where}: not {iterations + 1} history entries"]
    settle = history[-1]
    if settle.get("iteration") != "settle" or settle.get("epochs") != cfg["settle_epochs"]:
        problems.append(f"{where}: the last pass is not the registered {cfg['settle_epochs']}-epoch settle")
    if not all(len(h.get("sibling_rings") or []) == cfg["sibling_depth"] and "merged" in h for h in history[1:-1]):
        problems.append(f"{where}: an iteration did not run the unique buffer with {cfg['sibling_depth']} rings")
    return problems


def integrity(arms: dict, spec: dict) -> list:
    """Every reason the evidence is not the registered run. Returns the problems; never raises on a missing arm."""
    if not isinstance(arms, dict):
        return ["no arms"]
    problems = []
    for name, cfg in spec["arms"].items():
        arm = arms.get(name)
        if not isinstance(arm, dict) or not isinstance(arm.get("seeds"), list) \
                or not isinstance(arm.get("config"), dict):
            problems.append(f"arm {name} is missing or has no seeds/config")
            continue
        if arm.get("training_fingerprint") != spec["era"]:
            problems.append(f"{name}: training era {arm.get('training_fingerprint')}, registered {spec['era']}")
        if arm.get("measurement_fingerprint") != spec["measurement_fp"]:
            problems.append(f"{name}: measurement code {arm.get('measurement_fingerprint')}, registered "
                            f"{spec['measurement_fp']}")
        if arm["config"] != {**cfg, "seeds": list(spec["seeds"])}:
            problems.append(f"{name}: config is not the registered recipe")
        if sorted(r.get("seed") for r in arm["seeds"] if isinstance(r, dict)) != list(spec["seeds"]):
            problems.append(f"{name}: seeds are not the registered {list(spec['seeds'])}")
        for row in arm["seeds"]:
            problems.extend(_seed_problems(spec, name, row) if isinstance(row, dict)
                            else [f"{name}: a seed row is not a record"])
    return problems


def _judge(arm: dict, spec: dict) -> dict:
    curves = [r["strict_failures_per_pass"] for r in sorted(arm["seeds"], key=lambda r: r["seed"])]
    perfect = sum(1 for c in curves if c[-1] == 0)
    verdict = ("supported" if perfect >= spec["support_at"] else "refuted" if perfect <= spec["refute_at"]
               else "inconclusive")
    return {"verdict": verdict, "perfect": perfect, "n": len(curves),
            "descriptives": {"final_failures": [c[-1] for c in curves],
                             "perfect_before_settling": sum(1 for c in curves if c[-2] == 0),
                             "failures_before_settling": [c[-2] for c in curves]}}


def report(arms: dict, spec: dict) -> dict:
    """The pre-registered verdict for each arm of `spec`, or NOT_RUN for all when the evidence is not the run."""
    try:
        problems = integrity(arms, spec)
        if not problems:
            return {"integrity": [], **{name: _judge(arms[name], spec) for name in spec["arms"]}}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, **{name: {"verdict": "not_run"} for name in spec["arms"]}}
