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
STATUSES = ("untested", "supported", "negligible", "refuted", "null", "inconclusive", "contested")


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


def _ts(text: str) -> datetime:
    t = datetime.fromisoformat(text)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _data_files(h: dict) -> list:
    """The stored-evidence files a claim declared, as a list whatever the record's shape: one file is stored as a
    dict (every record before multi-file claims), several as a list. A claim is pre-registered only if it predates
    the EARLIEST of them — the first moment any of its data existed."""
    data = h.get("data")
    if not data:
        return []
    return [data] if isinstance(data, dict) else list(data)


def _data_started(path: str) -> str:
    """The `started` stamp a stored-evidence file carries — when its data began to be produced."""
    try:
        started = json.loads(Path(path).read_text()).get("started")
    except (OSError, json.JSONDecodeError, AttributeError) as e:
        raise ValueError(f"cannot read a `started` timestamp from {path!r}: {e}") from e
    if not started:
        raise ValueError(f"{path!r} carries no `started` timestamp — data of unknown age cannot time a claim")
    _ts(started)
    return started


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
                 null_below: float = 0.03, data: str | list = "", inconclusive_proof: str = "",
                 reads_no_data: bool = False) -> dict:
        """Register a claim, in exactly ONE of two modes.

        COMPARISON-backed: `a`/`b`/`direction` name the ledger comparison that would settle it and `unit` says
        what it is stated in (H3). TEST-backed: `proof` is a pytest node id, for the many findings that are not
        A/B comparisons at all — "a resumed run keeps its optimizer state", "encode was hardcoded to two
        planes". Those are proved by a regression test, and a claim without one is just prose.

        A test-backed claim MUST declare what its proof reads: `data` — the stored-evidence file, carrying its own
        `started` timestamp, so the claim is pre-registered only if it was written before the data was produced —
        or `reads_no_data` for a proof over code alone. Without the declaration the only clock is the verify call,
        which always comes after registration, so a claim written after reading the results would still read as a
        prediction (§C.43 t27) — and remembering to attach the data before verifying nearly failed (§C.45). A
        PRE-registered claim names data that does not exist yet; it is timed when verify first finds it.

        A test-backed claim may also declare an `inconclusive_proof`: a test that PASSES when the claim cannot be
        judged (a pre-registered gate failed, or the data only bound the effect). A failed proof then reads
        INCONCLUSIVE instead of REFUTED — without it the only outcomes are pass and refuted, and an underpowered
        null reads as a refutation (§C.44 h19-h22 did).

        There is deliberately no `status` parameter in either mode (H1)."""
        if id in self._h:
            raise ValueError(f"hypothesis {id!r} already exists — registering it again would overwrite the "
                             f"record of what was originally predicted")
        has_cmp = bool(a or b or direction)
        if data and not proof:
            raise ValueError(f"{id}: only a test-backed claim takes a `data` file — a comparison is timed by the "
                             f"ledger's own drawn_at")
        if inconclusive_proof and not proof:
            raise ValueError(f"{id}: only a test-backed claim takes an `inconclusive_proof` — a comparison reads "
                             f"inconclusive from its own significance and null band")
        if inconclusive_proof and inconclusive_proof == proof:
            raise ValueError(f"{id}: the inconclusive proof must differ from the proof — one test cannot both "
                             f"settle a claim and say it cannot be settled")
        if has_cmp and proof:
            raise ValueError(f"{id}: a claim is settled by a comparison OR by a test, exactly one — a claim "
                             f"with two kinds of proof has no single thing that would refute it")
        if not has_cmp and not proof:
            raise ValueError(f"{id}: a claim needs either a comparison (a/b/direction) or a test `proof` — "
                             f"without one it is prose, which is what this register exists to replace")
        if proof and data and reads_no_data:
            raise ValueError(f"{id}: a claim cannot both name a data file and declare it reads no data")
        if proof and not data and not reads_no_data:
            raise ValueError(f"{id}: declare what the proof reads — `data` (the evidence file, even one not produced "
                             f"yet) or `reads_no_data` — or the claim can only be timed by its verify call")
        if proof:
            h = {"id": id, "claim": claim, "mode": "test", "proof": proof, "note": note,
                 "registered_at": self._now(), "evidence": []}
            if data:
                paths = [data] if isinstance(data, str) else list(data)
                files = [{"path": d, "started": _data_started(d) if Path(d).exists() else None} for d in paths]
                h["data"] = files[0] if len(files) == 1 else files
            if reads_no_data:
                h["reads_no_data"] = True
            if inconclusive_proof:
                h["inconclusive_proof"] = inconclusive_proof
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
                                  "caveats": c.get("caveats"), "drawn_at": c.get("drawn_at")})
        self._save()
        return self._view(h)

    def attach_data(self, id: str, path: str) -> dict:
        """Declare the stored evidence a test-backed claim's proof reads, after registration. The pre-registration
        check compares the claim's ORIGINAL registration time with the data's `started`, so a late declaration
        can only withdraw a claim to foresight, never grant one. Declared once: swapping in another file would
        let a claim be re-timed against whichever data suits it."""
        h = self._h.get(id)
        if h is None:
            raise KeyError(f"unknown hypothesis {id!r}")
        if _mode(h) != "test":
            raise ValueError(f"{id} is comparison-backed; only a test-backed claim takes a `data` file")
        if h.get("data"):
            raise ValueError(f"{id} already declares {h['data']['path']!r} — its data cannot be swapped")
        h["data"] = {"path": path, "started": _data_started(path)}
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
        for f in _data_files(h):
            if f["started"] is None:
                if not Path(f["path"]).exists():
                    raise ValueError(f"{id}: its data {f['path']!r} has not been produced yet — nothing to verify")
                f["started"] = _data_started(f["path"])
            elif _data_started(f["path"]) != f["started"]:
                raise ValueError(f"{id}: {f['path']!r} changed its `started` since it was declared — the data was "
                                 f"regenerated under the claim, so register a new claim against the new data")
        runner = run_test or _run_pytest
        res = runner(h["proof"])
        if not res.get("collected"):
            raise ValueError(f"{id}: proof {h['proof']!r} collected no tests — a proof that runs nothing "
                             f"proves nothing, and would otherwise read as a pass")
        entry = {"proof": h["proof"], "ok": bool(res["ok"]), "detail": res.get("detail", ""), "drawn_at": self._now()}
        if not res["ok"] and h.get("inconclusive_proof"):
            inc = runner(h["inconclusive_proof"])
            if not inc.get("collected"):
                raise ValueError(f"{id}: inconclusive proof {h['inconclusive_proof']!r} collected no tests — it "
                                 f"would otherwise read as 'decidable', turning an undecidable claim into a refutation")
            entry["inconclusive"] = bool(inc["ok"])
        h["evidence"].append(entry)
        self._save()
        return self._view(h)

    def supersede(self, id: str, by: str, reason: str) -> dict:
        """Mark `id` as superseded by `by` — "do not build on this claim any more".

        This is a POINTER, not a status rewrite. h3 ("at equal simulations LOWER search wins") was genuinely
        supported by its own comparison at 460,800 simulations and genuinely reversed at 1,843,200; rewriting
        it to `refuted` would misreport what its evidence found, and deleting it would erase the fact that a
        budget-scoped claim looked general for three days. Both stay readable, with the later one named.

        A claim is retired by EVIDENCE, so a superseder with none is refused — otherwise supersession becomes
        a way to overrule a measurement by assertion, which is the whole failure this register exists to end."""
        h, sup = self._h.get(id), self._h.get(by)
        if h is None:
            raise KeyError(f"unknown hypothesis {id!r}")
        if sup is None:
            raise KeyError(f"unknown superseding hypothesis {by!r}")
        if not reason.strip():
            raise ValueError("a reason is required: supersession without one leaves the next reader unable to "
                             "tell a retired finding from a contradicted one")
        if not sup["evidence"]:
            raise ValueError(f"{by} has no evidence, so it cannot retire {id} — a finding is superseded by a "
                             f"measurement, never by assertion")
        seen, cursor = {id}, by   # seeded with `id`, so self-supersession is the degenerate cycle
        while cursor is not None:
            if cursor in seen:
                raise ValueError(f"supersession cycle: {by} already leads back to {id}, and a mutual "
                                 f"supersession leaves both claims unreadable")
            seen.add(cursor)
            cursor = (self._h.get(cursor) or {}).get("supersession", {}).get("by")
        h["supersession"] = {"by": by, "reason": reason.strip(), "at": self._now()}
        self._save()
        return self._view(h)

    def report(self) -> list[dict]:
        return [self._view(h) for h in self._h.values()]

    def _view(self, h: dict) -> dict:
        return {**h, "mode": _mode(h), "status": self._status(h), "pre_registered": self._pre_registered(h),
                "superseded_by": h.get("supersession", {}).get("by"),
                "caveats": sorted({k for e in h["evidence"] for k in (e.get("caveats") or {})})}

    def _status(self, h: dict) -> str:
        """H1: derived from the evidence, never stored."""
        if not h["evidence"]:
            return "untested"
        if _mode(h) == "test":
            passed = any(e["ok"] for e in h["evidence"])
            failed = any(not e["ok"] and not e.get("inconclusive") for e in h["evidence"])
            if passed and failed:
                return "contested"
            if passed:
                return "supported"
            if failed:
                return "refuted"
            return "inconclusive"
        sig = [e for e in h["evidence"] if e["significant"]]
        wants_positive = h["direction"] == "a>b"
        for_it = [e for e in sig if (e["diff"] > 0) == wants_positive]
        against = [e for e in sig if (e["diff"] > 0) != wants_positive]
        threshold = float(h.get("null_below", 0.03))
        if for_it and against:
            return "contested"
        if for_it:
            # A claim must clear the bar ITS OWN registrant set. Significance says the effect is not zero;
            # `null_below` says how big it must be to matter, and a claim that asserts an effect cannot be
            # supported by one it would itself have called absent. Testing significance first let power, not
            # effect size, decide the verdict.
            if all(abs(e["diff"]) < threshold for e in for_it):
                return "negligible"
            return "supported"
        if against:
            # Deliberately asymmetric: magnitude gates a claim being ASSERTED, never one being contradicted.
            # "a > b" is wrong if b wins significantly, however narrowly.
            return "refuted"
        # NULL says the effect is ABSENT; INCONCLUSIVE says the measurement could not see it. Collapsing them
        # loses the distinction between "we looked and it is not there" and "we lacked the power to look".
        if all(abs(e["diff"]) < threshold for e in h["evidence"]):
            return "null"
        return "inconclusive"

    def _pre_registered(self, h: dict) -> bool:
        """H2: true only when EVERY piece of evidence was drawn after the claim was registered."""
        stamps = [e.get("drawn_at") for e in h["evidence"]]
        if not stamps or any(s is None for s in stamps):
            return not h["evidence"]
        files = _data_files(h)
        if files and (any(f["started"] is None for f in files)
                      or not _ts(h["registered_at"]) < min(_ts(f["started"]) for f in files)):
            return False
        return all(s > h["registered_at"] for s in stamps)
