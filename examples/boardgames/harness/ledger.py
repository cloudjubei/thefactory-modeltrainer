"""§C.11 ANALYSIS LEDGER — measurements are RECORDED with provenance, and comparisons are DRAWN from the ledger.

WHY (2026-09-04): §C.9 gave us correct measurement primitives, and within hours I made three NEW errors anyway —
because the primitives were available but nothing forced measurements through them. Every one was a bookkeeping
failure, not a statistics failure:
  L1 I put a GATE-SELECTED checkpoint in a grid beside FINAL ones and compared them.
  L2 I labelled a comparison "matched budget" when the arms had 9.6k vs 16k games.
  L3 I ran ~6 paired tests on overlapping roots and then read p=0.039 as significant.
A ledger fixes all three by construction: a rate cannot be recorded without its provenance, budget and root
family, and a comparison cannot be drawn without those being checked and the multiplicity being counted.

The lesson generalises: when a guard exists but is optional, it will eventually be bypassed by whoever is in a
hurry — including me. Guards belong on the path, not beside it."""
from __future__ import annotations

import json
from pathlib import Path

from harness.measurement import mcnemar_exact, wilson_interval

PROVENANCE = ("final", "gate_selected", "best_of_n", "arbitrary")


class Ledger:
    """A persistent record of measurements + the comparisons drawn from them."""

    def __init__(self, path):
        self.path = Path(path)
        blob = {}
        if self.path.exists():
            try:
                blob = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                blob = {}
        self._entries = blob.get("entries", {})
        self._comparisons = blob.get("comparisons", [])

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"entries": self._entries, "comparisons": self._comparisons}, indent=1))
        tmp.replace(self.path)

    def entries(self) -> dict:
        return dict(self._entries)

    def record(self, name: str, outcomes: list[int], params: int, games: int, provenance: str,
               seed: int, roots_id: str) -> dict:
        """Record a measurement. `provenance` is MANDATORY — 'final' vs 'gate_selected' vs 'best_of_n' is the
        difference between a comparison that means something and one that does not (L1)."""
        if provenance not in PROVENANCE:
            raise ValueError(f"provenance must be one of {PROVENANCE}, got {provenance!r}")
        n = len(outcomes)
        if n == 0:
            raise ValueError("no outcomes")
        k = sum(1 for o in outcomes if o)
        e = {"name": name, "outcomes": [int(o) for o in outcomes], "n": n, "converted": k, "rate": k / n,
             "ci": list(wilson_interval(k, n)), "params": int(params), "games": int(games),
             "provenance": provenance, "seed": int(seed), "roots_id": roots_id}
        self._entries[name] = e
        self._save()
        return e

    def compare(self, a: str, b: str, allow_mixed_provenance: bool = False) -> dict:
        """Paired comparison drawn from the ledger, with all three bookkeeping checks enforced."""
        ea, eb = self._entries.get(a), self._entries.get(b)
        if ea is None or eb is None:
            raise ValueError(f"unknown entry: {a if ea is None else b}")
        if ea["roots_id"] != eb["roots_id"]:
            raise ValueError(f"different root families ({ea['roots_id']} vs {eb['roots_id']}) are NOT paired data")
        warning = ""
        if ea["provenance"] != eb["provenance"]:
            warning = (f"MIXED PROVENANCE: {a} is {ea['provenance']}, {b} is {eb['provenance']} — a selected "
                       f"checkpoint against a final one measures the SELECTOR, not the models")
            if not allow_mixed_provenance:
                raise ValueError(warning)
        res = mcnemar_exact(ea["outcomes"], eb["outcomes"])
        family = ea["roots_id"]
        self._comparisons.append({"a": a, "b": b, "roots_id": family, "p": res["p"]})
        self._save()
        n_fam = sum(1 for c in self._comparisons if c["roots_id"] == family)
        budget_matched = ea["games"] == eb["games"]
        return {**res,
                "a": a, "b": b, "rate_a": ea["rate"], "rate_b": eb["rate"],
                "budget_matched": budget_matched,
                "budget_note": ("" if budget_matched else
                                f"BUDGETS DIFFER: {a} trained on {ea['games']} games, {b} on {eb['games']} — "
                                f"this is not a like-for-like comparison"),
                "provenance_warning": warning,
                "comparisons_on_family": n_fam,
                "alpha_corrected": 0.05 / n_fam,
                "significant": res["p"] <= 0.05 / n_fam}

    def significant(self, p: float, roots_id: str, alpha: float = 0.05) -> bool:
        """Is `p` significant AFTER correcting for every comparison already drawn on this root family (L3)?"""
        n_fam = max(1, sum(1 for c in self._comparisons if c["roots_id"] == roots_id))
        return p <= alpha / n_fam
