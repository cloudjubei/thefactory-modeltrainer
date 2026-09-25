"""Suite-wide safety rails.

The exact Connect-4 solver is milliseconds on late positions and MINUTES-TO-HOURS near the opening. A test that
accidentally triggers the latter hangs the whole suite, and a hung suite looks exactly like a slow one — I hit
this three separate times (the LBR screen test, then the randomized-oracle tests). Rather than rely on reviewer
attention, the solver is given a depth limit for the duration of the test session so the wall fails FAST and
names the fix. A test that genuinely needs a deep solve opts in with @pytest.mark.allow_deep_solve.
"""
import json
import os
from pathlib import Path

import pytest

from harness import solver

TEST_MAX_SOLVE_EMPTIES = 30  # mid-game solves are fine; the measured wall in this suite is 34+ empties


def pytest_configure(config):
    config.addinivalue_line("markers", "allow_deep_solve: permit an exact solve near the opening (slow)")


REGISTER = Path(__file__).resolve().parent.parent / "hypotheses.json"


def _recorded_outcomes(root: Path) -> dict:
    """What the register last recorded for every test-backed claim's proof nodes, keyed (resolved file, test name):
    ("skip", why) for the undecidable proof of a SUPPORTED claim, which the register never consulted; ("xfail", why)
    for a proof the register recorded FAILING — an inconclusive claim's proof, a refuted claim's proof and its
    undecidable proof. Nodes it recorded passing are absent: they run as written."""
    try:
        claims = json.loads(Path(os.environ.get("HYPOTHESES_REGISTER") or REGISTER).read_text())["hypotheses"]
    except (OSError, ValueError, KeyError):
        return {}

    def node(nodeid: str):
        path, _sep, name = nodeid.partition("::")
        return (root / path).resolve(), name

    out = {}
    for h in claims.values():
        runs = [e for e in h.get("evidence", []) if "proof" in e]
        if not h.get("proof") or not runs:
            continue
        last, inc = runs[-1], h.get("inconclusive_proof")
        if last["ok"]:
            if inc:
                out[node(inc)] = ("skip", f"{h['id']} is supported; the register never consults its undecidable proof")
        else:
            out[node(h["proof"])] = ("xfail", f"{h['id']} recorded this proof FAILING")
            if inc and not last.get("inconclusive"):
                out[node(inc)] = ("xfail", f"{h['id']} recorded its undecidable proof failing too")
    return out


def pytest_collection_modifyitems(config, items):
    """The suite follows the REGISTER. A registered claim's proof file is pinned, so a proof that the register
    recorded as failing can be neither edited nor deleted — and it would fail every suite run. So each proof node is
    expected to do what the register last recorded: a failing proof is a STRICT xfail (if it starts passing, the
    evidence or the judge changed under the claim, and the suite goes red), and a supported claim's undecidable
    proof, which the register never consults, is skipped. The register's own runner sets REGISTER_PROOF and every
    proof runs exactly as written."""
    if os.environ.get("REGISTER_PROOF") == "1":
        return
    recorded = _recorded_outcomes(Path(str(config.rootpath)))
    if not recorded:
        return
    for item in items:
        outcome = recorded.get((Path(str(item.path)).resolve(), getattr(item, "originalname", item.name)))
        if outcome is None:
            continue
        kind, why = outcome
        item.add_marker(pytest.mark.skip(reason=why) if kind == "skip" else pytest.mark.xfail(strict=True, reason=why))


@pytest.fixture(autouse=True)
def _guard_opening_wall(request):
    previous = solver.MAX_SOLVE_EMPTIES
    solver.MAX_SOLVE_EMPTIES = 0 if request.node.get_closest_marker("allow_deep_solve") else TEST_MAX_SOLVE_EMPTIES
    yield
    solver.MAX_SOLVE_EMPTIES = previous
