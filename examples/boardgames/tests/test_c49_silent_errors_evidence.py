"""§C.49 D1, read after the data — SILENT errors: at the first ply where the solver rejects a saved T10 net's play, the
net and its own 200-sim search mostly AGREE on the losing move, so no agreement-based stop signal could see them.
Read from evidence/c49_D1_plateau.json.gz (registered after the data, as a description)."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_D1_plateau.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def _silent_at_first_failure(row: dict) -> int:
    """Failures at the net's first failing ply beyond those its search flagged as the net's error there."""
    failures = {int(p): n for p, n in row["certificate"]["failures_by_ply"].items()}
    first = min(failures)
    flagged = row["by_ply"].get(str(first), {})
    return failures[first] - flagged.get("net_wrong", 0) - flagged.get("both_wrong", 0)


def test_c49_at_its_first_failing_ply_the_net_and_its_search_agree_on_a_losing_move_in_5_of_6_nets():
    rows = load_evidence(EVIDENCE / FILE)["seeds"]
    assert len(rows) == 6 and not any(r["certificate"]["certified"] for r in rows)
    silent = {r["seed"]: _silent_at_first_failure(r) for r in rows}
    assert sum(1 for n in silent.values() if n > 0) == 5
    first_moves_wrong = [r["seed"] for r in rows if "0" in {str(p) for p in r["certificate"]["failures_by_ply"]}]
    assert sorted(first_moves_wrong) == [361, 362] and silent[361] == 1
