"""§C.49 D4 — at the nets' POLICY-SIDE errors, does their own relabel search LABEL an optimal move? Pre-registered.

D3 (h106, h107) found the value head backs about 42% of the T10 nets' wrong moves; at the rest it already rates an
optimal move strictly above the net's raw move, yet the net plays the raw move. Training's policy target there is
the argmax of the net's own 200-sim relabel search. If that label is optimal, the search turns the value head's
knowledge into a right label and the failure is downstream — the net never trained there (off its own tree) or did
not absorb the label (on it). If the label is wrong, the search does not turn the value head's ranking into its
label.

Each position carries D3's inputs (raw move, value-head ratings, exact values), the label and whether it is on the
net's own first-player tree. SUPPORTED when the pooled share of policy-side errors with an optimal label is at least
`support_at`, REFUTED at or below `refute_at`, INCONCLUSIVE between or with fewer than `min_errors`. Value-side
errors (the value head backs the raw move) are reported apart. Evidence that is not the registered measurement is
NOT_RUN."""
from __future__ import annotations

from harness.value_diagnosis import D3_SPEC, position_reading

D4_SPEC = {"seeds": D3_SPEC["seeds"], "measurement_fp": "e69c66cc0a11",
           "config": {"plies": D3_SPEC["config"]["plies"], "sims": 200}, "min_errors": 100, "support_at": 0.6,
           "refute_at": 0.4}


def _add(into: dict, key, optimal_label: bool) -> None:
    slot = into.setdefault(str(key), {"policy_side": 0, "label_optimal": 0})
    slot["policy_side"] += 1
    slot["label_optimal"] += int(optimal_label)


def _problems(evidence, spec: dict) -> list:
    problems = []
    if evidence.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement code {evidence.get('measurement_fingerprint')}, registered "
                        f"{spec['measurement_fp']}")
    if evidence.get("config") != spec["config"]:
        problems.append("config is not the registered measurement")
    if evidence.get("seeds") != list(spec["seeds"]):
        problems.append(f"seeds are not the registered {list(spec['seeds'])}")
    for r in evidence["positions"]:
        if r["seed"] not in spec["seeds"] or r["ply"] not in spec["config"]["plies"]:
            problems.append("a position is from an unregistered net or ply")
            break
    return problems


def d4_report(evidence, spec: dict) -> dict:
    """The pre-registered verdict with its breakdown by tree, net and ply, or NOT_RUN."""
    try:
        problems = _problems(evidence, spec)
        if not problems:
            by_tree, by_seed, by_ply = {}, {}, {}
            value_side = {"errors": 0, "label_optimal": 0}
            for r in evidence["positions"]:
                values = {int(a): v for a, v in r["values"].items()}
                reading = position_reading(int(r["raw"]), {int(a): q for a, q in r["q_hat"].items()}, values)
                if reading is None or not reading["error"]:
                    continue
                optimal_label = values[int(r["label"])] == max(values.values())
                if reading["backs_raw"]:
                    value_side["errors"] += 1
                    value_side["label_optimal"] += int(optimal_label)
                    continue
                _add(by_tree, "on_tree" if r["on_tree"] else "off_tree", optimal_label)
                _add(by_seed, r["seed"], optimal_label)
                _add(by_ply, r["ply"], optimal_label)
            n = sum(s["policy_side"] for s in by_seed.values())
            right = sum(s["label_optimal"] for s in by_seed.values())
            share = right / n if n else None
            verdict = ("inconclusive" if n < spec["min_errors"] else "supported" if share >= spec["support_at"]
                       else "refuted" if share <= spec["refute_at"] else "inconclusive")
            return {"integrity": [], "verdict": verdict, "policy_side": n, "label_optimal": right, "share": share,
                    "value_side": value_side, "by_tree": by_tree, "by_seed": by_seed, "by_ply": by_ply}
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, "verdict": "not_run"}
