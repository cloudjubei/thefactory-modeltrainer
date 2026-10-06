"""Plan §2.3 E2 — how small certified first-player play in Kalah is against the whole game, and how much the choice
among value-keeping moves matters (scripts/kalah_strategy_size.py, harness.strategy_size). Per shape: the game graph's
reachable positions, the canonical strategy's decisions and the tree-minimising strategy's decisions, both certified
whole-game by harness.certify. Judged on shapes whose graph holds at least `min_graph` positions:

  - fraction: every judged shape's minimising strategy decides at no more than 5% of its graph's positions —
    REFUTED if any exceeds 10%, INCONCLUSIVE between;
  - halving: on at least two thirds of judged shapes the minimising strategy needs at most half the canonical one's
    decisions — REFUTED at one third or fewer, INCONCLUSIVE between."""
from __future__ import annotations

SPEC = {
    "measurement_fp": "d59eb0fda1dd",
    "shapes": [[m, n] for m in (1, 2, 3) for n in range(1, 7)] + [[4, 1], [4, 2], [5, 1], [6, 1]],
    "min_graph": 1000,
    "fraction_support": 0.05,
    "fraction_refute": 0.10,
    "halving_support": 2 / 3,
    "halving_refute": 1 / 3,
}


def _integrity(e: dict, spec: dict) -> list[str]:
    problems = []
    if e.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement {e.get('measurement_fingerprint')} is not the registered {spec['measurement_fp']}")
    if sorted(r["shape"] for r in e["shapes"]) != sorted(spec["shapes"]):
        problems.append("the shapes measured are not the registered ones")
    if not all(r["canonical"]["certified"] and r["min_tree"]["certified"] for r in e["shapes"]):
        problems.append("a strategy did not certify")
    return problems


def e2_report(e: dict, spec: dict) -> dict:
    integrity = _integrity(e, spec)
    if integrity:
        return {"integrity": integrity, "fraction": {"verdict": "not_run"}, "halving": {"verdict": "not_run"}}
    judged = [r for r in sorted(e["shapes"], key=lambda r: r["shape"]) if r["graph"] >= spec["min_graph"]]
    fractions = [r["min_tree"]["decisions"] / r["graph"] for r in judged]
    fraction = ("supported" if max(fractions) <= spec["fraction_support"]
                else "refuted" if max(fractions) > spec["fraction_refute"] else "inconclusive")
    halved = sum(1 for r in judged if 2 * r["min_tree"]["decisions"] <= r["canonical"]["decisions"]) / len(judged)
    halving = ("supported" if halved >= spec["halving_support"]
               else "refuted" if halved <= spec["halving_refute"] else "inconclusive")
    return {"integrity": [], "judged": [r["shape"] for r in judged],
            "fraction": {"verdict": fraction, "values": fractions},
            "halving": {"verdict": halving, "share": halved}}
