"""§3.6 F2 — finding registered AFTER the data, from evidence/c49_f2.json.gz against evidence/c49_f1.json.gz: L5's rule
carries its pilot gain to the whole game. Every one of the 671 positions completes; the whole strategy takes 455,959 bits
against F1's 548,424 (0.83) — the positions' own bits 446,266 against 538,731. Map bits fall from 292,054 to 222,410 and
levelled cells from 65,608 to 46,314; exception bits from 136,253 to 107,049. F2 is smaller on 420 positions, larger on
189, equal on 62. The gain is largest in the heavy tail: F1's 67 heaviest positions take 223,703 bits against 311,823
(0.72), though the tail stays heavy — F2's own heaviest 67 hold 57% of its bits."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_f1 import SPEC as F1, f1_bits
from harness.floor_f2 import SPEC

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_f2.json.gz"
FILES = (FILE, "c49_f1.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_f2_l5_s_rule_cuts_the_whole_game_by_a_sixth_most_in_the_heavy_tail():
    f2_e, f1_e = load_evidence(EVIDENCE / FILE), load_evidence(EVIDENCE / "c49_f1.json.gz")
    f2 = {r["index"]: r for r in f2_e["roots"]}
    f1 = {r["index"]: r for r in f1_e["roots"]}
    assert len(f2) == 671 and all(r["complete"] for r in f2.values())
    assert (f1_bits(f2_e, SPEC), f1_bits(f1_e, F1)) == (455_959, 548_424)

    def total(roots, key):
        return sum(r["bits"][key] for r in roots.values())

    assert (total(f2, "map_bits"), total(f1, "map_bits")) == (222_410, 292_054)
    assert (total(f2, "exception_bits"), total(f1, "exception_bits")) == (107_049, 136_253)
    assert (sum(len(m) for r in f2.values() for m in r["maps"][r["shared"] + 1:]),
            sum(len(m) for r in f1.values() for m in r["maps"][r["shared"] + 1:])) == (46_314, 65_608)
    assert (sum(1 for i in f2 if f2[i]["bits"]["nodes"] < f1[i]["bits"]["nodes"]),
            sum(1 for i in f2 if f2[i]["bits"]["nodes"] > f1[i]["bits"]["nodes"])) == (420, 189)
    heaviest = sorted(f1, key=lambda i: -f1[i]["bits"]["nodes"])[:67]
    assert (sum(f2[i]["bits"]["nodes"] for i in heaviest), sum(f1[i]["bits"]["nodes"] for i in heaviest)) == (
        223_703, 311_823)
    own = sorted((r["bits"]["nodes"] for r in f2.values()), reverse=True)
    assert 0.56 < sum(own[:67]) / sum(own) < 0.58
