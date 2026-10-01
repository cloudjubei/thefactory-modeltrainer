"""§C.49 — the Connect-4 strategy frontier through 10 plies with refits trained until they hold their data (the
"hold" recipe: rate decaying to 1e-5 over 4,000 epochs, no early stop), read from
evidence/c49_strategy_d10_hold.json.gz. Registered AFTER the data, as a description of what the run found."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_strategy_d10_hold.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _runs():
    e = load_evidence(EVIDENCE / FILE)
    assert e["config"]["recipe_name"] == "hold" and e["config"]["depth"] == 10 and e["config"]["rounds"] == 20
    assert e["training_fingerprint"] == "a9e2872d0e37" and e["measurement_fingerprint"] == "92bac541fa1b"
    return {r["width"]: r for r in e["runs"] if r["family"] == "canon_conv"}


def test_c49_with_refits_that_hold_a_20_6K_net_is_certified_through_10_plies_and_an_8K_net_is_not():
    wide, narrow = _runs()[32], _runs()[16]
    assert wide["params"] == 20616 and wide["certified"] and len(wide["rounds"]) == 18
    assert wide["rounds"][-1]["failures"] == 0
    assert narrow["params"] == 8008 and not narrow["certified"] and not narrow["stalled"]
    assert len(narrow["rounds"]) == 20 and narrow["rounds"][-1]["failures"] > 0
    held = [r["fit"]["solved"] for r in narrow["rounds"] if r["fit"] is not None]
    misses = [r["fit"]["best_failures"] for r in narrow["rounds"] if r["fit"] is not None]
    assert held[:3] == [True] * 3 and not any(held[3:]) and max(misses) >= 40
    wide_misses = [r["fit"]["best_failures"] for r in wide["rounds"] if r["fit"] is not None]
    assert max(wide_misses) <= 4
