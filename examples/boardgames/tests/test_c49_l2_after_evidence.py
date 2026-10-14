"""§3.6 L2 — finding registered AFTER the data, from evidence/c49_l2.json.gz against evidence/c49_w8.json.gz: with the
first level holding any safe move deciding, the 32 strategies need 90 maps instead of W8's 586 and take 14,666 bits
— and the whole game projects to 308,603 bits, its 95% bootstrap interval (177K-469K) entirely under the trained net's
~660K: the first projection below the net. On positions seen before; W9 tests it on fresh ones."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_l2 import SPEC
from harness.floor_w6 import SPEC as W6, w6_projection

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_l2.json.gz"
FILES = (FILE, "c49_w8.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_l2_ninety_maps_and_the_whole_game_projects_under_the_net():
    l2 = load_evidence(EVIDENCE / FILE)
    w8 = load_evidence(EVIDENCE / "c49_w8.json.gz")
    assert sum(r["bits"]["maps"] for r in l2["roots"]) == 90 and sum(r["bits"]["maps"] for r in w8["roots"]) == 586
    assert sum(r["bits"]["nodes"] for r in l2["roots"]) == 14_666
    judge = {**SPEC, **{k: W6[k] for k in ("net_bits", "bootstrap", "bootstrap_seed")}}
    p = w6_projection(l2, judge)
    assert round(p["bits"]) == 308_603 and not p["lower_bound"] and p["high"] < W6["net_bits"]
