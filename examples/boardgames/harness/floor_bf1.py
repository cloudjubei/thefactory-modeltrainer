"""§3.7 BF1 — best-first proving with maps, registered. Scratch pilots on 82 leaf positions of G3's game found
best-first proving with maps (harness.best_first) proving 54 at 4,000 positions evaluated each, against 17 for
depth-first proving with maps (harness.self_proof's `Prover`) and 8 for best-first proving with no maps, and every
proof passed harness.composed's check. BF1 tests that on a fresh sample: leaf positions of T8's game (Z2's, written in
c49_t8.json.gz), `per` for each count of stones in `stones` and each kind (the leaf reads a map with exceptions, or
one without; the empty map's leaves are left out, since the empty map proves them at once), drawn with `seed`. Each
position is given to the three arms with the same work (`evaluations` positions evaluated, `trials` trials a map
search, the bound `bound`, `reuse` recent maps, composition on) and `ceiling` seconds as a safeguard; a position any
arm cut short is left out of the counts and reported. Judged:

  - best_vs_depth, best_vs_nomaps — best-first with maps proves at least `support` times as many of the counted
    positions as the other arm: supported; at most `refute` times as many: refuted; inconclusive between, or when
    fewer than `min_positions` are counted;
  - sound — every proof any arm gives, counted or not, passes the check the proof test runs on it, and the solver,
    as a grader only, finds every proved position won.

No arm reads a solver; the solver's values are taken after every arm has finished."""
from __future__ import annotations

from harness.floor_t8 import SPEC as T8

ARMS = ("best", "depth", "nomaps")
SPEC = {"rule": T8["rule"], "source": T8["file"], "file": "c49_bf1.json.gz",
        "stones": (8, 10, 12, 14, 16, 18, 20, 22), "per": 8, "seed": 11, "evaluations": 4000, "trials": 50,
        "bound": 5000, "reuse": 8, "cap": 200_000, "ceiling": 3600, "support": 2.0, "refute": 1.2,
        "min_positions": 60}
SETTINGS = ("rule", "source", "stones", "per", "seed", "evaluations", "trials", "bound", "reuse", "cap", "ceiling")


def settings(spec: dict) -> dict:
    return {k: list(spec[k]) if isinstance(spec[k], tuple) else spec[k] for k in SETTINGS}


def verdict(treated: int, control: int, spec: dict) -> str:
    if treated > control and treated >= spec["support"] * control:
        return "supported"
    if treated <= spec["refute"] * control:
        return "refuted"
    return "inconclusive"


def bf1_report(e: dict, source: dict, spec: dict, check) -> dict:
    """`check(row, arm)` re-checks the proof `arm` gave at `row`, independently of the run's own word."""
    if (e.get("spec") != "harness.floor_bf1" or e.get("settings") != settings(spec)
            or e.get("source_sha256") != source.get("sha256")):
        return {"included": 0, "proved": {}, "best_vs_depth": "not_run", "best_vs_nomaps": "not_run",
                "sound": "not_run"}
    rows = e["positions"]
    counted = [r for r in rows if not any(r["arms"][arm]["ceiling"] for arm in ARMS)]
    proved = {arm: sum(r["arms"][arm]["result"] is True for r in counted) for arm in ARMS}
    enough = len(counted) >= spec["min_positions"]
    proofs = [(r, arm) for r in rows for arm in ARMS if r["arms"][arm]["result"] is True]
    sound = ("inconclusive" if not proofs
             else "supported" if all(check(r, arm) and r["value"] > 0 for r, arm in proofs) else "refuted")
    return {"included": len(counted), "proved": proved,
            "best_vs_depth": verdict(proved["best"], proved["depth"], spec) if enough else "inconclusive",
            "best_vs_nomaps": verdict(proved["best"], proved["nomaps"], spec) if enough else "inconclusive",
            "sound": sound}
