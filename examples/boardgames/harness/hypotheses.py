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
STATUSES = ("untested", "supported", "refuted", "null", "inconclusive", "contested")


def _run_pytest(nodeid: str) -> dict:
    """Default proof runner. Exit 5 is pytest's "no tests collected", which must never read as a pass."""
    import subprocess

    r = subprocess.run([".venv/bin/python", "-m", "pytest", nodeid, "-q", "-p", "no:randomly"],
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    collected = 0 if (r.returncode == 5 or "no tests ran" in out) else 1
    return {"ok": r.returncode == 0 and collected == 1, "collected": collected, "detail": out.strip()[-300:]}


def _mode(h: dict) -> str:
    """Records written before `mode` existed carry none; a claim with declared arms is a comparison, one with a
    proof is a test. Derived on read rather than migrated, so stored records are never rewritten."""
    return h.get("mode") or ("test" if h.get("proof") else "comparison")


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

    def register(self, id: str, claim: str, a: str | None = None, b: str | None = None,
                 direction: str | None = None, unit: str = "", proof: str = "", note: str = "",
                 null_below: float = 0.03) -> dict:
        """Register a claim, in exactly ONE of two modes.

        COMPARISON-backed: `a`/`b`/`direction` name the ledger comparison that would settle it and `unit` says
        what it is stated in (H3). TEST-backed: `proof` is a pytest node id, for the many findings that are not
        A/B comparisons at all — "a resumed run keeps its optimizer state", "encode was hardcoded to two
        planes". Those are proved by a regression test, and a claim without one is just prose.

        There is deliberately no `status` parameter in either mode (H1)."""
        if id in self._h:
            raise ValueError(f"hypothesis {id!r} already exists — registering it again would overwrite the "
                             f"record of what was originally predicted")
        has_cmp = bool(a or b or direction)
        if has_cmp and proof:
            raise ValueError(f"{id}: a claim is settled by a comparison OR by a test, exactly one — a claim "
                             f"with two kinds of proof has no single thing that would refute it")
        if not has_cmp and not proof:
            raise ValueError(f"{id}: a claim needs either a comparison (a/b/direction) or a test `proof` — "
                             f"without one it is prose, which is what this register exists to replace")
        if proof:
            h = {"id": id, "claim": claim, "mode": "test", "proof": proof, "note": note,
                 "registered_at": self._now(), "evidence": []}
            self._h[id] = h
            self._save()
            return self._view(h)
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r} — an undeclared "
                             f"direction lets any significant result be read as confirmation")
        if not unit.strip():
            raise ValueError("unit is required: a claim about efficiency is not well-formed until it says "
                             "efficient IN WHAT (§C.29 — simulations and wall-clock gave opposite verdicts)")
        if not claim.strip():
            raise ValueError("claim is required")
        h = {"id": id, "claim": claim, "mode": "comparison", "a": a, "b": b, "direction": direction,
             "unit": unit, "note": note, "null_below": float(null_below),
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
        if _mode(h) != "comparison":
            raise ValueError(f"{id} is test-backed; only a comparison-backed claim takes ledger evidence "
                             f"(use verify())")
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

    def verify(self, id: str, run_test=None) -> dict:
        """Run a test-backed claim's proof and attach the outcome.

        A proof that COLLECTS NOTHING is refused rather than recorded: a typo'd node id makes pytest exit 0
        having run nothing, and a green-by-vacuum proof is worse than no proof, because it looks like evidence.
        This is the §C.21 vacuous-test trap in its purest form."""
        h = self._h.get(id)
        if h is None:
            raise KeyError(f"unknown hypothesis {id!r}")
        if _mode(h) != "test":
            raise ValueError(f"verify() is for test-backed claims; {id} is comparison-backed and is settled "
                             f"by ledger evidence (use link())")
        res = (run_test or _run_pytest)(h["proof"])
        if not res.get("collected"):
            raise ValueError(f"{id}: proof {h['proof']!r} collected no tests — a proof that runs nothing "
                             f"proves nothing, and would otherwise read as a pass")
        h["evidence"].append({"proof": h["proof"], "ok": bool(res["ok"]), "detail": res.get("detail", ""),
                              "drawn_at": self._now()})
        self._save()
        return self._view(h)

    def report(self) -> list[dict]:
        return [self._view(h) for h in self._h.values()]

    def _view(self, h: dict) -> dict:
        return {**h, "mode": _mode(h), "status": self._status(h), "pre_registered": self._pre_registered(h)}

    def _status(self, h: dict) -> str:
        """H1: derived from the evidence, never stored."""
        if not h["evidence"]:
            return "untested"
        if _mode(h) == "test":
            oks = [e["ok"] for e in h["evidence"]]
            if all(oks):
                return "supported"
            if not any(oks):
                return "refuted"
            return "contested"
        sig = [e for e in h["evidence"] if e["significant"]]
        wants_positive = h["direction"] == "a>b"
        for_it = [e for e in sig if (e["diff"] > 0) == wants_positive]
        against = [e for e in sig if (e["diff"] > 0) != wants_positive]
        if for_it and against:
            return "contested"
        if for_it:
            return "supported"
        if against:
            return "refuted"
        # NULL says the effect is ABSENT; INCONCLUSIVE says the measurement could not see it. Collapsing them
        # loses the distinction between "we looked and it is not there" and "we lacked the power to look".
        threshold = float(h.get("null_below", 0.03))
        if all(abs(e["diff"]) < threshold for e in h["evidence"]):
            return "null"
        return "inconclusive"

    def _pre_registered(self, h: dict) -> bool:
        """H2: true only when EVERY piece of evidence was drawn after the claim was registered."""
        stamps = [e.get("drawn_at") for e in h["evidence"]]
        if not stamps or any(s is None for s in stamps):
            return not h["evidence"]
        return all(s > h["registered_at"] for s in stamps)
