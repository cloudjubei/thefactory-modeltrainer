"""§C.49 T9 — the strategy tree INSTEAD of siblings: does the cheaper coverage lever still work? Pre-registered.

On Connect-4 the 2-deep siblings cost 94% of an iteration and grow with the buffer (~200 h per seed by iteration
30), while the strategy-tree walk relabels about as many positions as the buffer in a couple of minutes. The tree is
the TARGETED coverage — the positions the net's own first-player play reaches against every reply — and the siblings
the blanket. T9 runs T8's process with the siblings switched off and judges two readings of one run on tic-tac-toe:

  PSTART  the settled net plays perfectly from the start as the first player (the solver certifies its own tree).
          SUPPORTED at >= `support_at` seeds, REFUTED at <= `refute_at`, INCONCLUSIVE between.
  STOP    the h69 stop signal, read exactly as harness.floor_stop reads it (no false stop; first zero within
          `latency` of the first certified net on enough seeds).

All-position perfection is reported, not judged: the tree does not aim at positions the first player never meets.
Any integrity failure, or unreadable evidence, makes both readings NOT_RUN."""
from __future__ import annotations

from harness.floor_coverage import T5_ARMS
from harness.floor_stop import _seed_reading, _tree_problems, _verdict

T9_ARM = "tree_no_siblings"
_CONFIG = {**T5_ARMS["augment_sib2"], "reanalyze_siblings": False, "steps_matched": False, "sibling_depth": 1,
           "strategy_tree": {"player": 0, "depth": None}}
SPEC = {"arms": {T9_ARM: _CONFIG}, "params": {T9_ARM: _CONFIG["params"]}, "seeds": tuple(range(351, 361)),
        "era": "41605b4ce5d9", "measurement_fp": "90cabce936e2", "iterations": _CONFIG["iterations"],
        "positions": 4520, "latency": 3, "certify_at": 8, "support_at": 8, "refute_at": 5}


def _problems(arm, spec: dict) -> list:
    name = next(iter(spec["arms"]))
    cfg, n = spec["arms"][name], spec["iterations"]
    if not isinstance(arm, dict) or not isinstance(arm.get("seeds"), list) or not isinstance(arm.get("config"), dict):
        return ["the arm is missing or has no seeds/config"]
    problems = []
    if arm.get("training_fingerprint") != spec["era"]:
        problems.append(f"training era {arm.get('training_fingerprint')}, registered {spec['era']}")
    if arm.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement code {arm.get('measurement_fingerprint')}, registered {spec['measurement_fp']}")
    if arm["config"] != {**cfg, "seeds": list(spec["seeds"])}:
        problems.append("config is not the registered recipe")
    if sorted(r.get("seed") for r in arm["seeds"] if isinstance(r, dict)) != list(spec["seeds"]):
        problems.append(f"seeds are not the registered {list(spec['seeds'])}")
    for row in arm["seeds"]:
        where = f"seed {row.get('seed')}"
        if row.get("params") != spec["params"][name] or row.get("positions") != spec["positions"]:
            problems.append(f"{where}: not the registered net or position count")
        curve = row.get("strict_failures_per_pass")
        if not isinstance(curve, list) or len(curve) != n + 1:
            problems.append(f"{where}: not {n + 1} scored passes")
        if row.get("games_per_pass") != [cfg["selfplay"]] * n + [0]:
            problems.append(f"{where}: not the registered games on every iteration and none while settling")
        history = row.get("history")
        if not isinstance(history, list) or len(history) != n + 1:
            problems.append(f"{where}: not {n + 1} history entries")
            continue
        settle = history[-1]
        if settle.get("iteration") != "settle" or settle.get("epochs") != cfg["settle_epochs"]:
            problems.append(f"{where}: the last pass is not the registered {cfg['settle_epochs']}-epoch settle")
        if any((h.get("siblings") or 0) != 0 for h in history[:-1]):
            problems.append(f"{where}: siblings were added")
        if not all("merged" in h for h in history[:-1]):
            problems.append(f"{where}: an iteration did not run the unique buffer")
        problems.extend(_tree_problems(row, n))
    return problems


def _bars(count: int, spec: dict) -> str:
    return "supported" if count >= spec["support_at"] else "refuted" if count <= spec["refute_at"] else "inconclusive"


def tree_report(arm, spec: dict = SPEC) -> dict:
    """Both pre-registered readings of the one arm's evidence, or NOT_RUN for both when it is not the run."""
    try:
        problems = _problems(arm, spec)
        if not problems:
            rows = sorted(arm["seeds"], key=lambda r: r["seed"])
            certified = sum(1 for r in rows if r["tree_certification_per_pass"][-1]["certified"])
            seeds = [_seed_reading(r, spec) for r in rows]
            false_stops = sum(len(s["false_stops"]) for s in seeds)
            certifying = sum(1 for s in seeds if s["first_certified"] is not None)
            timely = sum(1 for s in seeds if s["timely"])
            return {"integrity": [],
                    "pstart": {"verdict": _bars(certified, spec), "certified": certified, "n": len(rows)},
                    "stop": {"verdict": _verdict(false_stops, certifying, timely, spec), "false_stops": false_stops,
                             "certifying": certifying, "timely": timely, "seeds": seeds},
                    "descriptives": {"raw_perfect": sum(1 for r in rows if r["strict_failures_per_pass"][-1] == 0),
                                     "train_seconds": [r.get("train_seconds") for r in rows]}}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    not_run = {"verdict": "not_run"}
    return {"integrity": problems, "pstart": not_run, "stop": not_run}
