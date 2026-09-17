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
import re
from datetime import datetime, timezone
from pathlib import Path

from harness.measurement import mcnemar_exact, wilson_interval

PROVENANCE = ("final", "budget_matched", "gate_selected", "best_of_n", "arbitrary")

# What L1 is actually about is SELECTION, not the label: a checkpoint chosen by looking at a score carries the
# winner's curse and one that isn't does not, so a comparison is only meaningful when both arms sit on the same
# side of this line. "budget_matched" (pinned to the other arm's batch index, never scored) is selection-free;
# "arbitrary" gets the pessimistic reading, since an unexplained checkpoint cannot be shown not to have been picked.
SCORE_SELECTED = frozenset({"gate_selected", "best_of_n", "arbitrary"})


def run_budget(ckpt) -> dict:
    """Training budget and provenance of `ckpt`, DERIVED from its own run's artefacts rather than asserted.

    The two facts a comparison needs about a checkpoint — how much training bought it, and whether anyone chose
    it by looking at a score — are both readable off the run: the batch index is in the filename, the games are
    in metrics.jsonl (or, for runs that predate per-batch cost accounting, in `iterations_done` times the config's
    games-per-iteration), and a checkpoint that is not the run's last batch is a budget-matched index. Deriving
    them is what stops a cherry-picked checkpoint from wearing an honest label: pick an earlier index and the
    budget shrinks with it, so `compare` refuses the pairing."""
    ckpt = Path(ckpt)
    m = re.fullmatch(r"ckpt_(\d+)", ckpt.stem)
    if not m:
        raise ValueError(f"{ckpt.name} is not a ckpt_<batch> checkpoint — provenance cannot be derived from it")
    idx = int(m.group(1))
    metrics = ckpt.parent / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{metrics} is empty — there is no budget to read")
    upto = [r for r in rows if int(r["batch"]) <= idx]
    if len(upto) != idx + 1:
        raise ValueError(f"{ckpt.name} claims batch {idx} but its run records {len(upto)} batches up to it")

    if all("games" in r for r in upto):
        games = sum(int(r["games"]) for r in upto)
    else:
        cfg_path = ckpt.parent.parent / f"{ckpt.parent.name}.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        per_iter, iters = cfg.get("games"), upto[-1].get("iterations_done")
        if per_iter is None or iters is None:
            raise ValueError(f"cannot establish the training budget of {ckpt}: metrics.jsonl records no `games` "
                             f"and {cfg_path.name} plus `iterations_done` do not supply one either")
        games = int(per_iter) * int(iters)

    prov = _read_json(ckpt.parent / "provenance.json") or _read_json(ckpt.parent / "summary.json") or {}
    last = max(int(r["batch"]) for r in rows)

    # "final" is the most trusted label in the system, so it has to mean the run reached the end it DECLARED,
    # not merely the last row present when someone happened to look. On a live run the old rule made every
    # freshly-written checkpoint "final" for as long as it was the newest; on a run stopped early at a
    # nice-looking batch it would have laundered that choice — L1 (selection) re-entering through a derived
    # label. An index that is not the declared end is exactly what "budget_matched" already describes.
    declared = (prov.get("request") or {}).get("batches", prov.get("batches"))
    run_complete = None if declared is None else (last == int(declared) - 1)
    is_final = idx == last and run_complete is not False

    # §C.26: `games` is only a PROXY for what a search-based learner spends. Two arms at different search
    # budgets are comparable on SIMULATIONS, not on games — 96 sims x N games is the same compute as 32 x 3N.
    sims = (prov.get("request") or {}).get("sims")
    if sims is None:
        cfg_path = ckpt.parent.parent / f"{ckpt.parent.name}.json"
        sims = (_read_json(cfg_path) or {}).get("sims") if cfg_path.exists() else None
    return {"batch": idx, "games": games, "provenance": "final" if is_final else "budget_matched",
            "code": prov.get("training_fingerprint"), "run_complete": run_complete,
            "sims": int(sims) if sims is not None else None,
            "simulations": games * int(sims) if sims is not None else None}


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


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

    def comparisons(self) -> list[dict]:
        """Stored comparisons, with `diff`/`significant` DERIVED for records drawn before those fields existed.

        The rates were always in `entries`, so nothing is invented — and deriving on read beats re-drawing the
        comparison, which would add a row to the family and silently tighten alpha for every other comparison
        on it. `drawn_at` is genuinely unrecoverable and stays None, which is what forfeits a pre-registration
        claim downstream (harness/hypotheses.py H2)."""
        out = []
        for c in self._comparisons:
            c = dict(c)
            if "diff" not in c or "significant" not in c:
                ea, eb = self._entries.get(c["a"]), self._entries.get(c["b"])
                n_fam = sum(1 for x in self._comparisons if x["roots_id"] == c["roots_id"])
                c["diff"] = (ea["rate"] - eb["rate"]) if ea and eb else None
                c["significant"] = (c["p"] <= 0.05 / max(1, n_fam)) if ea and eb else None
            c.setdefault("drawn_at", None)
            out.append(c)
        return out

    def record(self, name: str, outcomes: list[int], params: int, games: int, provenance: str,
               seed: int, roots_id: str, code: str | None = None, config: str | None = None,
               run_complete: bool | None = None, compute: int | None = None) -> dict:
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
             "provenance": provenance, "seed": int(seed), "roots_id": roots_id, "code": code, "config": config,
             "run_complete": run_complete, "compute": compute}
        self._entries[name] = e
        self._save()
        return e

    def compare(self, a: str, b: str, allow_mixed_provenance: bool = False,
                treatment: str = "config") -> dict:
        """Paired comparison drawn from the ledger, with all the bookkeeping checks enforced.

        `treatment` names WHICH dimension is under test, because the answer inverts the code check. Normally the
        config is the treatment and the training code must be held fixed (§C.17). But measuring a harness change
        itself — the §C.14 optimizer regime, say — is a real experiment in which the code IS the variable and the
        CONFIG is what must be held fixed. Declaring it is not an opt-out: each setting still demands that the
        other dimension match, and that the declared one actually differ, so a mislabelled experiment is refused
        rather than waved through."""
        if treatment not in ("config", "code", "budget"):
            raise ValueError(f"treatment must be 'config', 'code' or 'budget', got {treatment!r}")
        ea, eb = self._entries.get(a), self._entries.get(b)
        if ea is None or eb is None:
            raise ValueError(f"unknown entry: {a if ea is None else b}")
        if ea["roots_id"] != eb["roots_id"]:
            raise ValueError(f"different root families ({ea['roots_id']} vs {eb['roots_id']}) are NOT paired data")
        budget_matched = ea["games"] == eb["games"]
        # §C.26 COMPUTE MATCHING: the efficiency arms differ in `games` BY DESIGN, because equal compute at
        # different search budgets means unequal game counts. Equal simulations is therefore just as good a
        # claim to "like for like" as equal games — but only when both arms actually recorded their compute;
        # an unknown compute must never be read as a match.
        ca_, cb_ = ea.get("compute"), eb.get("compute")
        compute_matched = bool(ca_ and cb_ and ca_ == cb_)
        matched = budget_matched or compute_matched
        if treatment == "budget":
            # A LEARNING CURVE: two checkpoints of one recipe under one code, and the training budget IS the
            # variable. Everything else must be held fixed, and the budgets must actually differ.
            if ea["games"] == eb["games"]:
                raise ValueError(f"{a} and {b} have the SAME budget ({ea['games']} games) — with budget declared "
                                 f"as the treatment there is nothing under test")
        else:
            for e, name in ((ea, a), (eb, b)):
                if e["provenance"] == "budget_matched" and not matched:
                    raise ValueError(f"{name} is recorded as budget_matched but the budgets differ "
                                     f"({ea['games']} vs {eb['games']} games, compute {ca_} vs {cb_}) — "
                                     f"the label is false")
        ca, cb = ea.get("code"), eb.get("code")
        ga, gb = ea.get("config"), eb.get("config")
        if treatment == "config":
            if ca and cb and ca != cb:
                raise ValueError(f"DIFFERENT TRAINING CODE: {a} was trained by {ca}, {b} by {cb} — the arms differ "
                                 f"by more than the flag under test, so the comparison attributes to the wrong cause")
            if ga and gb and ga == gb:
                raise ValueError(f"{a} and {b} have the SAME CONFIG ({ga}) — with the config declared as the "
                                 f"treatment there is nothing under test here")
        elif treatment == "code":
            if ga and gb and ga != gb:
                raise ValueError(f"DIFFERENT CONFIG: {a} is {ga}, {b} is {gb} — with the training code declared "
                                 f"as the treatment the config is what must be held fixed")
            if ca and cb and ca == cb:
                raise ValueError(f"{a} and {b} were trained by the SAME TRAINING CODE ({ca}) — declaring code as "
                                 f"the treatment describes an experiment that is not being run")
        else:  # budget: code AND config held fixed
            if ca and cb and ca != cb:
                raise ValueError(f"DIFFERENT TRAINING CODE: {a} {ca} vs {b} {cb} — a learning curve must be one code")
            if ga and gb and ga != gb:
                raise ValueError(f"DIFFERENT CONFIG: {a} {ga} vs {b} {gb} — a learning curve must be one recipe")
        code_warning = ("" if ca and cb else
                        f"TRAINING CODE UNKNOWN for {a if not ca else b} — it cannot be shown that both arms were "
                        f"trained by the same harness, so a second, unrecorded difference may be in play")
        warning = ""
        if ea["provenance"] != eb["provenance"]:
            warning = f"MIXED PROVENANCE: {a} is {ea['provenance']}, {b} is {eb['provenance']}"
            if (ea["provenance"] in SCORE_SELECTED) != (eb["provenance"] in SCORE_SELECTED):
                warning += " — a selected checkpoint against an unselected one measures the SELECTOR, not the models"
                if not allow_mixed_provenance:
                    raise ValueError(warning)
        incomplete = [n for e, n in ((ea, a), (eb, b)) if e.get("run_complete") is False]
        completeness_warning = ("" if not incomplete else
                                f"RUN NOT FINISHED: {', '.join(incomplete)} did not reach its declared end — the "
                                f"reading is a progress report, and if the run is later stopped at a batch chosen "
                                f"by looking at scores, that checkpoint is selected rather than final")
        res = mcnemar_exact(ea["outcomes"], eb["outcomes"])
        family = ea["roots_id"]
        n_fam = sum(1 for c in self._comparisons if c["roots_id"] == family) + 1
        # The stored record carries WHEN it was drawn and WHICH WAY it went, because a hypothesis register can
        # only tell a prediction from a rationalisation by comparing those timestamps (harness/hypotheses.py).
        self._comparisons.append({"a": a, "b": b, "roots_id": family, "p": res["p"],
                                  "diff": ea["rate"] - eb["rate"],
                                  "significant": res["p"] <= 0.05 / n_fam,
                                  "drawn_at": datetime.now(timezone.utc).isoformat(timespec="microseconds")})
        self._save()
        return {**res,
                "a": a, "b": b, "rate_a": ea["rate"], "rate_b": eb["rate"],
                "budget_matched": budget_matched,
                "compute_matched": compute_matched,
                "budget_note": ("" if budget_matched else
                                f"COMPUTE-MATCHED: {a} played {ea['games']} games and {b} {eb['games']}, which "
                                f"differ BY DESIGN — both spent {ca_} simulations" if compute_matched else
                                f"BUDGETS DIFFER: {a} trained on {ea['games']} games, {b} on {eb['games']} — "
                                f"this is not a like-for-like comparison"),
                "provenance_warning": warning, "code_warning": code_warning,
                "completeness_warning": completeness_warning,
                "comparisons_on_family": n_fam,
                "alpha_corrected": 0.05 / n_fam,
                "significant": res["p"] <= 0.05 / n_fam}

    def significant(self, p: float, roots_id: str, alpha: float = 0.05) -> bool:
        """Is `p` significant AFTER correcting for every comparison already drawn on this root family (L3)?"""
        n_fam = max(1, sum(1 for c in self._comparisons if c["roots_id"] == roots_id))
        return p <= alpha / n_fam
