"""§C.17 TRAINING-CODE FINGERPRINT — which code produced a checkpoint, checked as a property rather than a date.

WHY (2026-09-08): the ledger enforces provenance, budget and root family, and every one of those held while two
experiments were quietly invalid. The controls were reused across THREE different optimizer regimes — Adam reset
per iteration, then per batch, then once per run — because a control is a file on disk and files do not record
what code wrote them. Both the global-pool A/B and the categorical-head A/B therefore varied two things at once,
and nothing in the system could have said so.

A commit hash would be the obvious fingerprint and is the wrong one: it changes when the docs change, so it fires
on comparisons that are perfectly valid, and a guard that cries wolf gets switched off (the same failure as the
over-broad solve-depth guard and the provenance-label check, §C.16). The property that actually matters is whether
the code that COMPUTES WEIGHTS is the same, so that is what gets hashed: the normalized syntax trees of the
training-path modules, with docstrings and comments stripped, and nothing else in the repository.
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path

# Every module whose behaviour can change the weights a run produces. Measurement, benchmarking and ledger code
# is deliberately EXCLUDED: it reads checkpoints and never feeds back into the training trajectory, so including
# it would make the guard fire on measurement work — the exact false positive that teaches people to bypass it.
TRAINING_MODULES = (
    "harness/neural.py",
    "harness/selfplay.py",
    "harness/scaled_run.py",
    "harness/league.py",
    "harness/refutation.py",
    "harness/solver.py",
    "harness/book.py",
    "harness/tablebase.py",
    "harness/config.py",
)

HARNESS_ROOT = Path(__file__).resolve().parent.parent

ABSENT = "<absent>"


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Drop docstring statements so prose edits do not read as behaviour changes."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return tree


def normalize(source: str) -> str:
    """Source reduced to what it DOES: comments gone with the parse, docstrings stripped, layout irrelevant."""
    if source == ABSENT:
        return ABSENT
    return ast.dump(_strip_docstrings(ast.parse(source)))


def _source_at(path: str, revision: str | None, root: Path) -> str:
    """Source of `path`, or ABSENT if the module did not exist yet.

    A module that has not been written is a real state of the training path, not an error: `refutation.py` post-dates
    the earliest runs we still compare against. Raising here would make the guard useless for exactly the historical
    question it exists to answer, and a guard that crashes gets routed around."""
    if revision is None:
        p = root / path
        return p.read_text() if p.exists() else ABSENT
    rel = (root / path).resolve().relative_to(_git_root(root))
    r = subprocess.run(["git", "show", f"{revision}:{rel}"], cwd=root, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ABSENT


def _git_root(root: Path) -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root, check=True,
                         capture_output=True, text=True).stdout.strip()
    return Path(out)


def training_fingerprint(game: str | None = None, revision: str | None = None, root: Path | None = None) -> str:
    """A short hash of the training path's normalized source, optionally as of a git `revision`.

    `game` folds in that game's own module, so a Connect-4 run is not invalidated by an edit to Tic-tac-toe."""
    root = HARNESS_ROOT if root is None else Path(root)
    paths = list(TRAINING_MODULES)
    if game:
        paths.append(f"games/{game}.py")
    h = hashlib.sha256()
    for path in paths:
        h.update(path.encode())
        h.update(normalize(_source_at(path, revision, root)).encode())
    return h.hexdigest()[:12]
