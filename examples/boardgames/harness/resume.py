"""§C.17a RESUME PREFLIGHT — an interrupted long run must resume HONESTLY, not merely successfully.

WHY (2026-09-10): the Othello transfer run is ~2-3 days over 24 batches on a machine that will not stay
uninterrupted, so resume stopped being a convenience and became the load-bearing path. Reading it found three
ways a resume corrupts the RECORD while looking like it worked:

  R1 CODE DRIFT. `scaled_run` resumes against whatever source is on disk, and provenance.json is stamped ONCE
     at launch. Edit a training module during the run, let the run be interrupted, and the later batches are
     trained by a different era than the run's label claims — the §C.17 control-reuse confound, reintroduced
     invisibly through the very mechanism built to detect it. This is the LIKELY one: days of a live run is
     exactly when training code gets edited.
  R2 ORPHAN CHECKPOINT. `ckpt_N.pt` is written BEFORE batch N's metrics row. Killed in between, that row is
     lost for good (resume restarts at N+1 and never revisits N), and `run_budget` — which establishes a
     budget by COUNTING rows — then refuses ckpt_N and every checkpoint above it. Training continues perfectly
     while the run quietly becomes unreadable by the ledger: the comparison it exists to draw can never be drawn.
  R3 TRUNCATED BUFFER. `torch.save(buf, buffer_path)` is not atomic (unlike the `_write_json_atomic` written
     for this exact failure class a few lines above it), so a kill mid-write leaves a buffer.pt that raises on
     EVERY later resume. The run is wedged until a human deletes the file.

R2 and R3 are repairable from outside the training path: drop the checkpoints the ledger cannot read (that
batch is retrained, costing what the interruption cost anyway) and drop an unreadable buffer. R1 is NOT
repairable — no file deletion makes two eras the same — so the preflight refuses and hands the judgement back.

This module lives deliberately OUTSIDE `fingerprint.TRAINING_MODULES` and is imported by nothing in the
training path, so consulting it can never move the era it is checking (§C.17a rule 2)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from harness.fingerprint import training_fingerprint


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def completed_checkpoints(run_dir) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in Path(run_dir).glob("ckpt_*.pt"))


def metrics_batches(run_dir) -> list[int]:
    rows = _metrics_rows(run_dir)
    return sorted(int(r["batch"]) for r in rows if "batch" in r)


def _metrics_rows(run_dir) -> list[dict]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # a kill mid-append leaves a partial last line; everything before it is still sound
    return rows


def ledger_readable_prefix(run_dir) -> int:
    """The number of leading batches for which a checkpoint AND its metrics row both exist. `run_budget`
    counts rows up to a checkpoint's index, so only this contiguous prefix can ever be read by the ledger."""
    ck, mb = set(completed_checkpoints(run_dir)), set(metrics_batches(run_dir))
    k = 0
    while k in ck and k in mb:
        k += 1
    return k


def orphan_checkpoints(run_dir) -> list[int]:
    """Checkpoints the ledger cannot read: the one whose metrics row was lost, plus every one above it."""
    keep = ledger_readable_prefix(run_dir)
    return [b for b in completed_checkpoints(run_dir) if b >= keep]


def buffer_status(run_dir) -> str:
    """'absent' | 'ok' | 'corrupt' — whether the replay buffer would survive the next `torch.load`."""
    path = Path(run_dir) / "buffer.pt"
    if not path.exists():
        return "absent"
    import torch

    try:
        torch.load(path)
    except Exception:  # truncation surfaces as OSError / UnpicklingError / BadZipFile / EOFError by cut point
        return "corrupt"
    return "ok"


def era_status(run_dir) -> dict:
    """Does the training code on disk still match the era this run was launched under (R1)?"""
    run_dir = Path(run_dir)
    prov = _read_json(run_dir / "provenance.json")
    if prov is None:
        if completed_checkpoints(run_dir):
            return {"ok": False, "stamped": None, "now": None,
                    "reason": f"{run_dir.name} holds checkpoints but no provenance.json — the era that trained "
                              f"them cannot be established, so nothing may be concluded from them"}
        return {"ok": True, "stamped": None, "now": None, "reason": "fresh run dir — nothing trained yet"}
    stamped = prov.get("training_fingerprint")
    now = training_fingerprint(prov.get("game", "connect4"))
    if stamped == now:
        return {"ok": True, "stamped": stamped, "now": now, "reason": ""}
    return {"ok": False, "stamped": stamped, "now": now,
            "reason": f"TRAINING CODE MOVED UNDER A LIVE RUN: launched as {stamped}, on disk now {now} — "
                      f"resuming would train the remaining batches with code the run's own label denies"}


def _truncate_metrics(run_dir, keep: int) -> None:
    path = Path(run_dir) / "metrics.jsonl"
    rows = [r for r in _metrics_rows(run_dir) if int(r.get("batch", -1)) < keep]
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, path)


def _bump_resumes(run_dir, resumes: int) -> None:
    path = Path(run_dir) / "provenance.json"
    prov = _read_json(path) or {}
    prov["resumes"] = resumes
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(prov, indent=1))
    os.replace(tmp, path)


def preflight(run_dir, repair: bool = False) -> dict:
    """Inspect a run dir before resuming it; with `repair`, restore the invariants that CAN be restored.

    Repair is deliberately conservative: it only ever deletes artefacts the ledger has already been shown to
    be unable to read, and it refuses to touch anything at all while the era check is failing — a code-drift
    run must not be quietly tidied into looking resumable."""
    run_dir = Path(run_dir)
    era = era_status(run_dir)
    orphans = orphan_checkpoints(run_dir)
    buf = buffer_status(run_dir)
    keep = ledger_readable_prefix(run_dir)
    repaired: list[str] = []
    resumes = int((_read_json(run_dir / "provenance.json") or {}).get("resumes", 0))

    if repair and era["ok"]:
        for b in orphans:
            (run_dir / f"ckpt_{b}.pt").unlink()
            repaired.append(f"ckpt_{b}.pt")
        if orphans:
            _truncate_metrics(run_dir, keep)
        if buf == "corrupt":
            (run_dir / "buffer.pt").unlink()
            repaired.append("buffer.pt")
            buf = "absent"
        orphans = []
        if completed_checkpoints(run_dir):
            # §C.14: `scaled_run` holds ONE Adam per PROCESS, and no optimizer state is persisted — so every
            # resume restarts the moment estimates. That is a real (if small) trajectory difference, and an
            # undeclared one is exactly what §C.17 is about, so the run carries how often it happened.
            resumes += 1
            _bump_resumes(run_dir, resumes)

    return {"ok": era["ok"] and not orphans and buf != "corrupt",
            "era": era, "orphans": orphans, "buffer": buf, "repaired": repaired,
            "resume_at": ledger_readable_prefix(run_dir), "resumes": resumes}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="harness.resume", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--repair", action="store_true", help="delete artefacts the ledger cannot read, so the "
                                                          "affected batch is retrained instead of lost")
    args = ap.parse_args()
    r = preflight(args.run_dir, repair=args.repair)
    print(json.dumps(r, indent=1))
    if not r["ok"]:
        why = r["era"]["reason"] or (f"checkpoints {r['orphans']} have no metrics row, so the ledger cannot "
                                     f"read them — rerun with --repair to retrain those batches")
        if r["buffer"] == "corrupt":
            why += " | buffer.pt is truncated and would crash the resume"
        raise SystemExit(f"REFUSING TO RESUME {args.run_dir}: {why}")
    print(f"resume at batch {r['resume_at']} (resumes so far: {r['resumes']})")


if __name__ == "__main__":
    main()
