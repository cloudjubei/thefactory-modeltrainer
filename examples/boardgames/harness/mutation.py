"""§C.32 MUTATION RUNNER — a guard is only as good as the mutation that kills it, and a mutation run by hand
lies in at least four ways. All four were hit for real on 2026-09-20 while guarding harness.ledger (L4):

  M-A STALE BYTECODE. Rewriting a module in a tight loop can leave a .pyc whose (mtime, size) still validate,
      so pytest imports the PREVIOUS source. A restored file scored 2 failures; the same bytes, cache cleared,
      scored 46 passed. Either direction of that error is fatal: a killed mutation may have been killed by the
      previous mutation's code, and a survivor may have survived on stale bytes.
  M-B UNAPPLIED MUTATION. If the text to replace is not found, nothing is mutated and the suite passes — which
      reads as SURVIVED, the most alarming possible verdict, produced by a typo.
  M-C RED BASELINE. Mutating a suite that is already failing makes every mutation look killed.
  M-D KILLED IS A PROXY. "Some test failed" is not "the test that asserts the property failed". The L4 guard's
      first attempt at the after-the-write mutation went inert instead (it re-read the row it had just
      written), so BOTH tests died on 'DID NOT RAISE' and the data-preservation test was never exercised.
      Naming the dead tests is what separates a guard from a guard-shaped object.

This is the recurring lesson of the whole track — the guard must check the PROPERTY, not a proxy — turned on
mutation testing itself."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def clear_bytecode(root: Path) -> int:
    """Remove every __pycache__ under `root`. Returns how many were removed (M-A)."""
    n = 0
    for d in sorted(Path(root).rglob("__pycache__"), reverse=True):
        shutil.rmtree(d, ignore_errors=True)
        n += 1
    return n


def run_suite(tests: list[str], cwd: Path | None = None, runner=None, timeout: float | None = None) -> dict:
    """Run pytest with bytecode writing OFF, returning which test ids failed.

    Exit 5 is "no tests collected" — a green-by-vacuum run, which must never read as a pass (the same trap
    harness.hypotheses.verify refuses). A suite still running after `timeout` seconds is stopped and read as
    failed: a mutation that makes the code under test never return (a game that no longer ends) is caught, not
    waited on forever."""
    cwd = Path(cwd or ".")
    clear_bytecode(cwd)
    if runner is not None:
        return runner(tests)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", *tests, "-q", "-p", "no:randomly"],
                           capture_output=True, text=True, cwd=str(cwd), env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"green": False, "collected": True, "failed": [f"timeout after {timeout:g}s"],
                "detail": f"stopped after {timeout:g}s"}
    out = (r.stdout or "") + (r.stderr or "")
    failed = sorted({ln.split(" ")[1] for ln in out.splitlines()
                     if ln.startswith("FAILED ") and len(ln.split(" ")) > 1})
    return {"green": r.returncode == 0, "collected": r.returncode != 5 and "no tests ran" not in out,
            "failed": failed, "detail": out.strip()[-400:]}


def _edits(m: dict) -> list[dict]:
    """A mutation is one or more text edits. The single `old`/`new` pair is the shorthand; `edits` is the full
    form, needed because the most valuable mutation class — RELOCATING a guard so it still runs but too late —
    deletes the check in one place and re-inserts it in another, and cannot be written as one substitution."""
    if "edits" in m:
        if not m["edits"]:
            raise ValueError(f"mutation {m.get('name')!r}: `edits` is empty, so it would change nothing and "
                             f"report SURVIVED")
        return list(m["edits"])
    return [{"old": m["old"], "new": m["new"]}]


def _apply(original: str, m: dict) -> str:
    """Apply every edit against the PRISTINE text. Offsets are resolved first and spliced right through, so
    edit 2 can never match text that edit 1 just created — which would silently make the mutation something
    other than the one named in the report."""
    spans = []
    for ed in _edits(m):
        hits = original.count(ed["old"])
        if hits != 1:
            raise ValueError(f"mutation {m.get('name')!r}: the text {ed['old'][:40]!r} occurs {hits} times "
                             f"and must occur exactly once — an unapplied edit runs the ORIGINAL code and "
                             f"reports SURVIVED (M-B)")
        i = original.index(ed["old"])
        spans.append((i, i + len(ed["old"]), ed["new"]))
    spans.sort()
    for (_, a_end, _x), (b_start, _y, _z) in zip(spans, spans[1:]):
        if b_start < a_end:
            raise ValueError(f"mutation {m.get('name')!r}: two of its edits overlap, so the result depends on "
                             f"which is applied first — split them into separate mutations")
    out, prev = [], 0
    for start, end, replacement in spans:
        out.append(original[prev:start])
        out.append(replacement)
        prev = end
    out.append(original[prev:])
    return "".join(out)


def mutate(path, mutations: list[dict], tests: list[str], cwd: Path | None = None, runner=None,
           timeout: float | None = None) -> dict:
    """Apply each mutation to `path` in turn, run `tests`, restore, and report what each one killed.

    `mutations` are {"name", "old", "new"} text substitutions. Each is applied to the PRISTINE file, never on
    top of the previous one. The original is restored in a finally, so an exploding suite cannot leave the
    tree mutated."""
    path = Path(path)
    original = path.read_text()
    base = run_suite(tests, cwd=cwd, runner=runner, timeout=timeout)
    if not base["collected"]:
        raise ValueError(f"baseline collected no tests from {tests} — a mutation run against nothing would "
                         f"report every mutation as killed")
    if not base["green"]:
        raise ValueError(f"baseline is RED ({len(base['failed'])} failing: {base['failed'][:3]}) — against a "
                         f"failing suite every mutation looks killed (M-C). Fix the suite, then mutate.")
    results = []
    try:
        for m in mutations:
            path.write_text(_apply(original, m))
            res = run_suite(tests, cwd=cwd, runner=runner, timeout=timeout)
            results.append({"name": m["name"], "killed": not res["green"], "killed_by": res["failed"]})
    finally:
        path.write_text(original)
        clear_bytecode(Path(cwd or "."))
    survived = [r["name"] for r in results if not r["killed"]]
    return {"baseline_green": True, "results": results, "survived": survived,
            "all_killed": not survived}
