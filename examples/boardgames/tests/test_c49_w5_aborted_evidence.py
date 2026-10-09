"""§3.6 W5 — finding registered AFTER the data, from evidence/c49_w5_aborted.json.gz (W5 stopped by hand): with the C
walk a frontier root can get a pure steady state at once — #362 in under a minute — but three of the four first-wave
builds were still running past an hour beyond the 2-hour cap: the builder checks its deadline only between positions,
and its own Python walks (covering checks, simplify, walk-order coding) ignore it."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_w5_aborted.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def _seconds(elapsed: str) -> int:
    parts = [int(x) for x in elapsed.replace("-", ":").split(":")]
    return sum(v * m for v, m in zip(reversed(parts), (1, 60, 3600, 86400)))


def test_c49_w5_one_root_is_one_pure_map_but_three_builds_overran_the_cap_by_an_hour():
    e = load_evidence(EVIDENCE / FILE)
    done = {r["index"]: r for r in e["finished_roots"]}
    r = done[362]
    assert r["complete"] and r["checked"] and r["moves_win"] and r["leaves_certified"]
    assert r["bits"]["leaves"] == 1 and r["bits"]["exceptions"] == 0 and r["build_seconds"] < 60
    busy = [w for w in e["workers_running_at_stop"] if w["cpu_percent"] > 50]
    assert len(busy) == 3 and all(_seconds(w["elapsed"]) > e["cap_seconds"] + 3600 for w in busy)
