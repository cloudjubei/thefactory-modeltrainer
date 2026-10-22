"""§3.6 K1 — finding registered AFTER the data, from evidence/c49_k1.json.gz against evidence/c49_l6.json.gz: the builds
barely use two-phase maps — 7 of the 759 maps their leaves use have a second phase — and the strategies end where
L6's did, within run-to-run swing: smaller on 13 positions, larger on 18, 1 tied (sign test p = 0.47). Exceptions fall
2,656 -> 2,171 (32,407 -> 28,168 bits) while maps rise 699 -> 759 (52,122 -> 56,487 bits, a flag bit each included)
and table moves 444 -> 520."""
from __future__ import annotations

from math import comb
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.steady_state import SWITCH

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_k1.json.gz"
FILES = (FILE, "c49_l6.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_k1_the_builds_barely_use_two_phase_maps_and_end_where_l6_did():
    arm = {r["index"]: r for r in load_evidence(EVIDENCE / FILE)["roots"]}
    control = {r["index"]: r for r in load_evidence(EVIDENCE / "c49_l6.json.gz")["roots"]}
    used = [r["maps"][i] for r in arm.values() for i in {n["leaf"] for _b, _t, n in r["nodes"] if "leaf" in n} if i]
    assert len(used) == 759 and sum(str(SWITCH) in m for m in used) == 7

    def total(roots, key):
        return sum(r["bits"][key] for r in roots.values())

    assert (total(arm, "exceptions"), total(control, "exceptions")) == (2_171, 2_656)
    assert (total(arm, "exception_bits"), total(control, "exception_bits")) == (28_168, 32_407)
    assert (total(arm, "maps"), total(control, "maps")) == (759, 699)
    assert (total(arm, "map_bits"), total(control, "map_bits")) == (56_487, 52_122)
    assert (total(arm, "moves"), total(control, "moves")) == (520, 444)
    sizes = [(arm[i]["bits"]["nodes"], control[i]["bits"]["nodes"]) for i in arm]
    smaller, larger = sum(a < c for a, c in sizes), sum(a > c for a, c in sizes)
    assert (smaller, larger) == (13, 18)
    n = smaller + larger
    assert round(2 * sum(comb(n, k) for k in range(min(smaller, larger) + 1)) / 2 ** n, 2) == 0.47
