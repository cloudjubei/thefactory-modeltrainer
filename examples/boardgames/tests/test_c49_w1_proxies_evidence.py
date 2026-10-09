"""§3.6 W1 — finding registered AFTER the data, from evidence/c49_w1_proxies.json.gz: neither cheap stand-in orders
W1's frontier positions by their strategy size — the fewest unwon positions two plies below is 1 for both the
smallest (429 bits) and the largest (29,935) strategy, and the largest has the slowest forced win while one of the
fastest is among the largest — so an opening cannot yet be chosen for small frontier strategies by either."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w1_proxies.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def _discordant(rows, key) -> bool:
    """Some pair is ordered one way by size and the other way by the stand-in."""
    return any(a["bits"] < b["bits"] and a[key] > b[key] for a in rows for b in rows)


def test_c49_w1_neither_cheap_stand_in_orders_frontier_strategy_sizes():
    rows = load_evidence(EVIDENCE / FILE)["rows"]
    assert len(rows) == 8
    small, large = min(rows, key=lambda r: r["bits"]), max(rows, key=lambda r: r["bits"])
    assert small["left_below"] == large["left_below"]
    assert _discordant(rows, "left_below") and _discordant(rows, "strong_score")
    assert large["strong_score"] == min(r["strong_score"] for r in rows)
