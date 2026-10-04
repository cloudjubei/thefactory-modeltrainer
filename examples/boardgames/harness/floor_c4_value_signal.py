"""§C.49 T14 — fixes to the VALUE SIGNAL on Connect-4: a paired pilot. Pre-registered.

D3/D4 (h106-h109) found the Connect-4 opening's training target wrong at most of the nets' errors, because the
search reads a value head that learns only from whole-game outcomes of 32-sim self-play games — credit reaches the
opening through ~40 moves — and gets no target at the strategy-tree positions. T14 runs T10's process at a small
depth (a 6-ply tree, 20 iterations, no stop, then the settle) once per arm per seed:
  base        T10's process unchanged;
  tree_value  the strategy-tree positions get the relabel search's root value as their value target;
  n_step      value targets bootstrap from a lagged target net 8 moves ahead (refreshed every 2 iterations)
              instead of waiting for the game's result.
Each final net is read at the White positions of its own first-player tree through ply 4: is its raw move optimal
(exact values from the label cache or a solve)?

Each TREATMENT is judged against base, paired by seed: it WINS a seed when its net plays the optimal move at a
strictly higher share of its own positions. SUPPORTED with at least `support_wins` wins AND a pooled share at least
`min_gain` higher; REFUTED with at most `refute_wins` wins or no pooled gain; INCONCLUSIVE between. The value head's
rating and the 200-sim label at each error are reported per arm (D3/D4's readings on the nets' own trees). A
readout that is not the registered measurement is NOT_RUN."""
from __future__ import annotations

from harness.floor_c4 import CONFIG
from harness.value_diagnosis import position_reading

_BASE = {**CONFIG, "iterations": 20, "strategy_tree": {"player": 0, "depth": 6}, "relabel_workers": 4,
         "stop_on_agreement": False}
SPEC = {"arms": {"base": _BASE, "tree_value": {**_BASE, "tree_value_target": True},
                 "n_step": {**_BASE, "value_n_step": 8, "target_refresh": 2}},
        "treatments": ("tree_value", "n_step"), "seeds": (401, 402, 403, 404),
        "era": "74c9ad808be0", "measurement_fp": "796d1e3db0c8", "readout": {"plies": [0, 2, 4], "sims": 200},
        "support_wins": 3, "refute_wins": 1, "min_gain": 0.1}


def _problems(readout, spec: dict) -> list:
    problems = []
    if readout.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement code {readout.get('measurement_fingerprint')}, registered "
                        f"{spec['measurement_fp']}")
    if readout.get("eras") != {a: spec["era"] for a in spec["arms"]}:
        problems.append("the nets were not trained by the registered code")
    if readout.get("readout") != spec["readout"]:
        problems.append("the readout is not the registered one")
    if readout.get("configs") != {a: {**c, "seeds": list(spec["seeds"])} for a, c in spec["arms"].items()}:
        problems.append("the nets were not trained on the registered recipes")
    nets = {(p["arm"], p["seed"]) for p in readout["positions"]}
    if nets != {(a, s) for a in spec["arms"] for s in spec["seeds"]}:
        problems.append("the readout does not cover exactly every registered arm and seed")
    if any(p["ply"] not in spec["readout"]["plies"] for p in readout["positions"]):
        problems.append("a position is outside the registered plies")
    return problems


def _count(into: dict, key, optimal: bool) -> None:
    slot = into.setdefault(str(key), {"positions": 0, "optimal": 0})
    slot["positions"] += 1
    slot["optimal"] += int(optimal)


def _judge(treatment: str, nets: dict, spec: dict) -> dict:
    share = {a: {s: n["optimal"] / n["positions"] for s, n in nets[a].items()} for a in ("base", treatment)}
    wins = sum(1 for s in spec["seeds"] if share[treatment][str(s)] > share["base"][str(s)])
    pooled = {a: sum(n["optimal"] for n in nets[a].values()) / sum(n["positions"] for n in nets[a].values())
              for a in ("base", treatment)}
    gain = round(pooled[treatment] - pooled["base"], 9)
    verdict = ("supported" if wins >= spec["support_wins"] and gain >= spec["min_gain"] else
               "refuted" if wins <= spec["refute_wins"] or gain <= 0 else "inconclusive")
    return {"verdict": verdict, "wins": wins, "pooled": pooled, "gain": gain}


def pilot_report(readout, spec: dict = SPEC) -> dict:
    """Each treatment's pre-registered verdict against base, with per-net and per-ply shares and the readings at
    errors, or NOT_RUN for every treatment."""
    try:
        problems = _problems(readout, spec)
        if not problems:
            nets = {a: {} for a in spec["arms"]}
            by_ply = {a: {} for a in spec["arms"]}
            at_errors = {a: {"errors": 0, "value_backs": 0, "label_optimal": 0} for a in spec["arms"]}
            for p in readout["positions"]:
                values = {int(a): v for a, v in p["values"].items()}
                best = max(values.values())
                optimal = values[int(p["raw"])] == best
                _count(nets[p["arm"]], p["seed"], optimal)
                _count(by_ply[p["arm"]], p["ply"], optimal)
                reading = position_reading(int(p["raw"]), {int(a): q for a, q in p["q_hat"].items()}, values)
                if reading is not None and reading["error"]:
                    slot = at_errors[p["arm"]]
                    slot["errors"] += 1
                    slot["value_backs"] += int(reading["backs_raw"])
                    slot["label_optimal"] += int(values[int(p["label"])] == best)
            return {"integrity": [], "treatments": {t: _judge(t, nets, spec) for t in spec["treatments"]},
                    "nets": nets, "by_ply": by_ply, "at_errors": at_errors}
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        problems = [f"the readout could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, "treatments": {t: {"verdict": "not_run"} for t in spec["treatments"]}}
