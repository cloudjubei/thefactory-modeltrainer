"""§C.30 HYPOTHESIS REGISTER — a discovered claim becomes a falsifiable object, and the runs judge it.

WHY (2026-09-17): every finding in this track so far has lived as PROSE — a section in the plan, a memory file.
Prose cannot be wrong in a way the system notices. Nothing linked a claim to the run that would test it, nothing
stopped a claim being written up after the result was already in, and nothing forced a claim to say in advance
which outcome would kill it. That is the gap between "we keep learning things" and "the process proves things".

Three rules are enforced here rather than remembered:

  H1 STATUS IS DERIVED, NEVER ASSERTED. A hypothesis does not get to declare itself true. `status` is computed
     from ledger comparisons linked to it — supported / refuted / inconclusive / contested / untested — and the
     constructor refuses a hand-written status outright.
  H2 PRE-REGISTRATION IS CHECKED, NOT CLAIMED. A hypothesis is `pre_registered` only if it was registered before
     its evidence was drawn, verified against the ledger's own `drawn_at` timestamps. The difference between a
     prediction and a rationalisation is a timestamp, and this is where that gets checked. Note the status is
     unaffected — late-registered evidence still counts; only the claim to foresight is withdrawn.
  H3 THE COMPARISON AND THE UNIT ARE DECLARED UP FRONT. A hypothesis names the two arms, the DIRECTION that
     would support it, and the UNIT it is stated in. Without the direction, any significant result could be
     spun as confirmation; without the unit, the claim is not even well-formed — §C.29 measured the same
     recipe winning under simulations and losing under wall-clock.

Linking accumulates: the same declared comparison drawn twice attaches twice, so a replication is visible and a
contradiction becomes `contested` rather than quietly averaging away."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DIRECTIONS = ("a>b", "a<b")
STATUSES = ("untested", "supported", "refuted", "inconclusive", "contested")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Register:
    """A persistent register of hypotheses and the ledger evidence attached to each."""

    def __init__(self, path, now=_now):
        self.path = Path(path)
        self._now = now
        blob = {}
        if self.path.exists():
            try:
                blob = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                blob = {}
        self._h = blob.get("hypotheses", {})

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"hypotheses": self._h}, indent=1))
        tmp.replace(self.path)

    def register(self, id: str, claim: str, a: str, b: str, direction: str, unit: str,
                 note: str = "") -> dict:
        """Register a claim. `a`/`b`/`direction` name the comparison that would settle it; `unit` names what
        the claim is stated in (H3). There is deliberately no `status` parameter (H1)."""
        if id in self._h:
            raise ValueError(f"hypothesis {id!r} already exists — registering it again would overwrite the "
                             f"record of what was originally predicted")
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r} — an undeclared "
                             f"direction lets any significant result be read as confirmation")
        if not unit.strip():
            raise ValueError("unit is required: a claim about efficiency is not well-formed until it says "
                             "efficient IN WHAT (§C.29 — simulations and wall-clock gave opposite verdicts)")
        if not claim.strip():
            raise ValueError("claim is required")
        h = {"id": id, "claim": claim, "a": a, "b": b, "direction": direction, "unit": unit, "note": note,
             "registered_at": self._now(), "evidence": []}
        self._h[id] = h
        self._save()
        return self._view(h)

    def get(self, id: str) -> dict:
        if id not in self._h:
            raise KeyError(f"unknown hypothesis {id!r}")
        return self._view(self._h[id])

    def link(self, id: str, ledger) -> dict:
        """Attach every ledger comparison matching this hypothesis's DECLARED arms (H3). A comparison between
        other arms is refused — otherwise a claim could harvest whichever result happened to be significant."""
        h = self._h.get(id)
        if h is None:
            raise KeyError(f"unknown hypothesis {id!r}")
        found = [c for c in ledger.comparisons() if c["a"] == h["a"] and c["b"] == h["b"]]
        if not found:
            raise ValueError(f"no comparison of the DECLARED arms {h['a']} vs {h['b']} exists in this ledger — "
                             f"{id} predicts that pairing and may only be judged on it")
        usable = [c for c in found if c.get("diff") is not None and c.get("significant") is not None]
        if not usable:
            raise ValueError(f"{id}: the {len(found)} comparison(s) of {h['a']} vs {h['b']} have no recoverable "
                             f"direction, so they cannot support or refute a directional claim — re-draw them")
        found = usable
        # identity of a comparison RECORD, not merely a similar one: two distinct comparisons can share a
        # family and a timestamp, and collapsing them would hide a replication or a contradiction.
        known = {(e["roots_id"], e["drawn_at"], e["p"], e["diff"]) for e in h["evidence"]}
        for c in found:
            if (c["roots_id"], c.get("drawn_at"), c["p"], c["diff"]) in known:
                continue
            h["evidence"].append({"a": c["a"], "b": c["b"], "roots_id": c["roots_id"], "p": c["p"],
                                  "diff": c["diff"], "significant": c["significant"],
                                  "drawn_at": c.get("drawn_at")})
        self._save()
        return self._view(h)

    def report(self) -> list[dict]:
        return [self._view(h) for h in self._h.values()]

    def _view(self, h: dict) -> dict:
        return {**h, "status": self._status(h), "pre_registered": self._pre_registered(h)}

    def _status(self, h: dict) -> str:
        """H1: derived from the evidence, never stored."""
        sig = [e for e in h["evidence"] if e["significant"]]
        if not h["evidence"]:
            return "untested"
        wants_positive = h["direction"] == "a>b"
        for_it = [e for e in sig if (e["diff"] > 0) == wants_positive]
        against = [e for e in sig if (e["diff"] > 0) != wants_positive]
        if for_it and against:
            return "contested"
        if for_it:
            return "supported"
        if against:
            return "refuted"
        return "inconclusive"

    def _pre_registered(self, h: dict) -> bool:
        """H2: true only when EVERY piece of evidence was drawn after the claim was registered."""
        stamps = [e.get("drawn_at") for e in h["evidence"]]
        if not stamps or any(s is None for s in stamps):
            return not h["evidence"]
        return all(s > h["registered_at"] for s in stamps)
