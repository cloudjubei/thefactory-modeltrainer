"""§C.49 D3 — does a net's VALUE HEAD back its own wrong moves? Pre-registered.

The self-play nets are wrong in the Connect-4 opening and their own search agrees (h92); more search does not fix it
(h96), and the net can represent the answer (h72). If the value head rates the position after a wrong move at least
as high as after the best move, the search inherits the error and the value head's training signal is the culprit;
if it rates the right move higher, the error comes from the policy or the search instead.

A position is read only when some move is not optimal. Mover's-view ratings `q_hat` (minus the value head at a
non-terminal child, the exact result at a finished one) are compared with exact move values:
  error       the raw move is not optimal;
  backs_raw   an error the value head rates at least as high as every optimal move;
  misranks    some non-optimal move is rated at least as high as every optimal move (whatever the raw move).
`misranks` at the positions the net gets RIGHT is the control: how often the value head errs where the policy does
not. SUPPORTED when the pooled share of backed errors is at least `support_at`, REFUTED at or below `refute_at`,
INCONCLUSIVE between or with fewer than `min_errors` errors. Evidence that is not the registered measurement is
NOT_RUN."""
from __future__ import annotations

_FIELDS = ("positions", "errors", "backed", "right", "misranked_when_right")


def position_reading(raw: int, q_hat: dict, values: dict) -> dict | None:
    """The reading of one position, or None when every move is optimal."""
    if set(q_hat) != set(values):
        raise ValueError("the value head must rate the same moves the exact values cover")
    best = max(values.values())
    optimal = [a for a, v in values.items() if v == best]
    others = [a for a in values if a not in optimal]
    if not others:
        return None
    top = max(q_hat[a] for a in optimal)
    error = raw not in optimal
    return {"error": error, "backs_raw": error and q_hat[raw] >= top,
            "misranks": max(q_hat[a] for a in others) >= top}


def _tally(into: dict, key, reading: dict) -> None:
    slot = into.setdefault(str(key), dict.fromkeys(_FIELDS, 0))
    slot["positions"] += 1
    slot["errors"] += int(reading["error"])
    slot["backed"] += int(reading["backs_raw"])
    slot["right"] += int(not reading["error"])
    slot["misranked_when_right"] += int(not reading["error"] and reading["misranks"])


def _problems(evidence, spec: dict) -> list:
    problems = []
    if evidence.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement code {evidence.get('measurement_fingerprint')}, registered "
                        f"{spec['measurement_fp']}")
    if evidence.get("config") != spec["config"]:
        problems.append("config is not the registered measurement")
    if evidence.get("seeds") != list(spec["seeds"]):
        problems.append(f"seeds are not the registered {list(spec['seeds'])}")
    if any(r["seed"] not in spec["seeds"] or r["ply"] not in spec["config"]["plies"] for r in evidence["positions"]):
        problems.append("a position is from an unregistered net or ply")
    return problems


def d3_report(evidence, spec: dict) -> dict:
    """The pre-registered verdict with its breakdown by net and ply, or NOT_RUN."""
    try:
        problems = _problems(evidence, spec)
        if not problems:
            by_seed, by_ply = {}, {}
            for r in evidence["positions"]:
                reading = position_reading(int(r["raw"]), {int(a): q for a, q in r["q_hat"].items()},
                                           {int(a): v for a, v in r["values"].items()})
                if reading is not None:
                    _tally(by_seed, r["seed"], reading)
                    _tally(by_ply, r["ply"], reading)
            errors = sum(s["errors"] for s in by_seed.values())
            backed = sum(s["backed"] for s in by_seed.values())
            share = backed / errors if errors else None
            verdict = ("inconclusive" if errors < spec["min_errors"] else "supported" if share >= spec["support_at"]
                       else "refuted" if share <= spec["refute_at"] else "inconclusive")
            return {"integrity": [], "verdict": verdict, "errors": errors, "backed": backed, "share": share,
                    "by_seed": by_seed, "by_ply": by_ply}
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, "verdict": "not_run"}


D3_SPEC = {"seeds": (361, 362, 363, 364, 365, 366), "measurement_fp": "08990db4de4d", "config": {"plies": [0, 2, 4, 6, 8]},
           "min_errors": 100, "support_at": 0.6, "refute_at": 0.4}
