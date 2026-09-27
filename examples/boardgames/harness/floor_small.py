"""§C.49 T6 — step 4 of the smallest-setup programme on tic-tac-toe: does the working process reach perfection at
the ORACLE-FRONTIER setups? Pre-registered.

T5 (harness/floor_coverage.py, h52) made the 57,453-param net raw-perfect on 10/10 seeds with the unique buffer,
2-deep siblings and the settle. The §C.49 oracle frontier (evidence/c49_frontier_tictactoe.json.gz) says a net
taught the exact answers holds tic-tac-toe at ~1K params with a standardised-orientation input and ~5K without. T6
runs T5's process at those setups:

  canon_mlp32       standardised input, MLP 32 (938 params) — T5's recipe as it is;
  canon_mlp32_long  the same with a 1,500-epoch settle — the oracle fits needed ~1,000-1,500 epochs at this size,
                    and the process trains ~210, so a failure of the plain arm could be the budget, not the process;
  canon_conv6_long  standardised input, conv 6 (994 params) — the frontier point that held under every recipe;
  residual15        the raw frontier, residual width 15 (5,008 params), with augmentation;
  residual15_long   the same with the long settle.

The settle trains only on the process's own final labels: still solver-free. A standardised-input net takes no
augmentation (it would only duplicate rows). Per arm, T5's bars: SUPPORTED at >= 8/10 raw-perfect after the settle,
REFUTED at <= 5, INCONCLUSIVE between. Any integrity failure, or unreadable evidence, makes every verdict NOT_RUN."""
from __future__ import annotations

from harness.floor_coverage import T5_ARMS

T6_SEEDS = tuple(range(321, 331))
T6_ERA = "9122a962ef8d"
T6_MEASUREMENT_FP = "6c1258655a2f"
T6_LONG_SETTLE = 1500
T6_SUPPORT_AT = 8
T6_REFUTE_AT = 5
_BASE = T5_ARMS["augment_sib2"]
_CANON_MLP = {"mlp_hidden": [32], "canonical_input": True}
_CANON_CONV = {"channels": 6, "canonical_input": True}
_RESIDUAL = {"channels": 15, "blocks": 1, "head_hidden": 15, "residual": True}
T6_PARAMS = {"canon_mlp32": 938, "canon_mlp32_long": 938, "canon_conv6_long": 994, "residual15": 5008,
             "residual15_long": 5008}
T6_ARMS = {
    "canon_mlp32": {**_BASE, "arch": _CANON_MLP, "augment": False},
    "canon_mlp32_long": {**_BASE, "arch": _CANON_MLP, "augment": False, "settle_epochs": T6_LONG_SETTLE},
    "canon_conv6_long": {**_BASE, "arch": _CANON_CONV, "augment": False, "settle_epochs": T6_LONG_SETTLE},
    "residual15": {**_BASE, "arch": _RESIDUAL},
    "residual15_long": {**_BASE, "arch": _RESIDUAL, "settle_epochs": T6_LONG_SETTLE},
}
T6_ARMS = {name: {**cfg, "params": T6_PARAMS[name]} for name, cfg in T6_ARMS.items()}
T6_ITERATIONS = _BASE["iterations"]


def _seed_problems(name: str, row: dict) -> list:
    cfg, where = T6_ARMS[name], f"{name} seed {row.get('seed')}"
    problems = []
    curve = row.get("strict_failures_per_pass")
    if not isinstance(curve, list) or len(curve) != T6_ITERATIONS + 1:
        problems.append(f"{where}: not {T6_ITERATIONS + 1} scored passes")
    elif len(row.get("final_failing_positions") or []) != curve[-1]:
        problems.append(f"{where}: the final failure list does not match the final pass's count")
    if row.get("positions") != 4520:
        problems.append(f"{where}: scored {row.get('positions')} positions, not 4520")
    if row.get("params") != T6_PARAMS[name]:
        problems.append(f"{where}: built {row.get('params')} parameters, not {T6_PARAMS[name]}")
    if row.get("games_per_pass") != [cfg["selfplay"]] * T6_ITERATIONS + [0]:
        problems.append(f"{where}: not the registered games on every iteration and none while settling")
    history = row.get("history")
    if not isinstance(history, list) or len(history) != T6_ITERATIONS + 1:
        return problems + [f"{where}: not {T6_ITERATIONS + 1} history entries"]
    settle = history[-1]
    if settle.get("iteration") != "settle" or settle.get("epochs") != cfg["settle_epochs"]:
        problems.append(f"{where}: the last pass is not the registered {cfg['settle_epochs']}-epoch settle")
    if not all(len(h.get("sibling_rings") or []) == cfg["sibling_depth"] and "merged" in h for h in history[1:-1]):
        problems.append(f"{where}: an iteration did not run the unique buffer with {cfg['sibling_depth']} rings")
    return problems


def integrity(arms: dict) -> list:
    """Every reason the evidence is not the registered T6. Returns the problems; never raises on a missing arm."""
    if not isinstance(arms, dict):
        return ["no arms"]
    problems = []
    for name, cfg in T6_ARMS.items():
        arm = arms.get(name)
        if not isinstance(arm, dict) or not isinstance(arm.get("seeds"), list) \
                or not isinstance(arm.get("config"), dict):
            problems.append(f"arm {name} is missing or has no seeds/config")
            continue
        if arm.get("training_fingerprint") != T6_ERA:
            problems.append(f"{name}: training era {arm.get('training_fingerprint')}, registered {T6_ERA}")
        if arm.get("measurement_fingerprint") != T6_MEASUREMENT_FP:
            problems.append(f"{name}: measurement code {arm.get('measurement_fingerprint')}, registered "
                            f"{T6_MEASUREMENT_FP}")
        if arm["config"] != {**cfg, "seeds": list(T6_SEEDS)}:
            problems.append(f"{name}: config is not the registered recipe")
        if sorted(r.get("seed") for r in arm["seeds"] if isinstance(r, dict)) != list(T6_SEEDS):
            problems.append(f"{name}: seeds are not the registered {list(T6_SEEDS)}")
        for row in arm["seeds"]:
            problems.extend(_seed_problems(name, row) if isinstance(row, dict) else [f"{name}: a seed row is not a record"])
    return problems


def _judge(arm: dict) -> dict:
    rows = sorted(arm["seeds"], key=lambda r: r["seed"])
    curves = [r["strict_failures_per_pass"] for r in rows]
    perfect = sum(1 for c in curves if c[-1] == 0)
    verdict = "supported" if perfect >= T6_SUPPORT_AT else "refuted" if perfect <= T6_REFUTE_AT else "inconclusive"
    return {"verdict": verdict, "perfect": perfect, "n": len(rows),
            "descriptives": {"final_failures": [c[-1] for c in curves],
                             "perfect_before_settling": sum(1 for c in curves if c[-2] == 0),
                             "failures_before_settling": [c[-2] for c in curves]}}


def small_report(arms: dict) -> dict:
    """The pre-registered T6 verdict for each arm, or NOT_RUN for all when the evidence is not the registered run."""
    try:
        problems = integrity(arms)
        if not problems:
            return {"integrity": [], **{name: _judge(arms[name]) for name in T6_ARMS}}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, **{name: {"verdict": "not_run"} for name in T6_ARMS}}
