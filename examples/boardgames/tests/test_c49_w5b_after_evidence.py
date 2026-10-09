"""§3.6 W5b — finding registered AFTER the data, from evidence/c49_w5b.json.gz against the W2 and W3 builds of the
same frontier positions: on the seven positions W2 or W3 also completed, W5b's strategies are 44% smaller in total
(7,766 bits against 13,842; six of seven smaller), and the whole-game projection rose past W3's lower bound only
because #230 — unfinished in W1-W4 — now completes, at 21,054 bits: 73% of the sample's bits on one position."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w5b.json.gz"
FILES = (FILE, "c49_w2.json.gz", "c49_w3.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _complete(file: str) -> dict:
    return {r["index"]: r["bits"]["nodes"] for r in load_evidence(EVIDENCE / file)["roots"] if r["complete"]}


def test_c49_w5b_is_smaller_where_earlier_builds_finished_and_one_position_holds_most_of_its_bits():
    w5b = _complete(FILE)
    earlier = {**_complete("c49_w2.json.gz"), **_complete("c49_w3.json.gz")}
    both = sorted(set(w5b) & set(earlier))
    assert len(both) == 7 and 230 not in earlier
    assert sum(w5b[i] for i in both) == 7766 and sum(earlier[i] for i in both) == 13842
    assert sum(w5b[i] < earlier[i] for i in both) == 6
    assert len(w5b) == 8 and w5b[230] == 21054 and w5b[230] / sum(w5b.values()) > 0.73
