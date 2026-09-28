"""§C.49 P-START baseline — descriptive findings pinned to their evidence (registered after the data; not predictions).

evidence/c49_certify_ab302_d12.json.gz: the strongest existing Connect-4 net (ab302_gpool_s0 ckpt_16, 335K params)
certified as the first player from the empty board through 12 plies — its RAW move at each of its positions, every
reply at the other side's, each move checked by the native exact solver to keep the proven win."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_certify_ab302_d12.json.gz",)
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_the_best_existing_net_gives_away_the_proven_win_at_its_second_move_and_50_times_within_12_plies():
    d = load_evidence(EVIDENCE / FILES[0])
    r = d["result"]
    assert d["config"]["depth"] == 12 and d["params"] == 335377 and r["complete"] and not r["certified"]
    assert r["failures"] == 50 and {int(k): v for k, v in r["failures_by_ply"].items()} == {2: 2, 6: 4, 8: 9, 10: 35}
    assert r["by_ply"]["2"] == [7, 0]
