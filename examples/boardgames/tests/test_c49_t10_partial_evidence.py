"""§C.49 T10, recovered — the six seeds the solver-free Connect-4 run finished before it hung, read from
evidence/c49_T10_partial.json.gz. Registered AFTER the data, as a description: the stop signal never fired."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_T10_partial.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


def test_c49_on_connect4_the_walk_never_reads_full_agreement_in_60_iterations_on_any_of_6_seeds():
    e = load_evidence(EVIDENCE / FILE)
    seeds = e["seeds"]
    assert sorted(s["seed"] for s in seeds) == list(range(361, 367))
    assert e["registered_config"]["iterations"] == 60 and e["registered_config"]["strategy_tree"]["depth"] == 10
    for s in seeds:
        walks = s["disagreements"][1:]
        assert not s["stopped"] and len(walks) == 59 and min(walks) > 0
    late = sorted(min(s["disagreements"][16:]) for s in seeds)
    assert late[0] == 5 and late[1] >= 27
