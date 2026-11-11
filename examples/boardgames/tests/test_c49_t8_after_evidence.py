"""§3.6 T8 — finding registered AFTER the data, from evidence/c49_t8.json.gz and T5's: what reading the position saved.
The file takes 106,044 bits (13,256 bytes), 0.9726 of T5's 109,034. Every part but the references and the exception
flags is T5's to a tenth of a bit; the references take 13,331.0 bits against 15,775.1, 15.5% fewer, and the exception
flags 13,728.2 against 14,273.0, 3.8% fewer, near the 14.1% and 3.6% the same walk saved on G3's file in the dry run
(28,842 to 24,779 and 27,285 to 26,306)."""
from __future__ import annotations

import base64

import pytest

from harness.floor_t8 import SPEC
from tests.test_c49_t8_evidence import FILES
from tests.test_c49_z2_evidence import EVIDENCE, _evidence

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")

CHANGED = ("reference", "exception_flags")


def test_c49_t8_the_file_takes_106_044_bits_and_only_the_references_and_exception_flags_changed():
    e, t5 = _evidence(SPEC["file"]), _evidence(SPEC["source"])
    assert (e["bits"], len(base64.b64decode(e["file"]))) == (106_044, 13_256)
    assert round(e["bits"] / SPEC["control_bits"], 4) == 0.9726
    assert {k: v for k, v in e["parts"].items() if k not in CHANGED} == {
        k: v for k, v in t5["parts"].items() if k not in CHANGED}
    assert [(e["parts"][k], t5["parts"][k]) for k in CHANGED] == [(13_331.0, 15_775.1), (13_728.2, 14_273.0)]
    assert (e["cells"], e["maps"], e["leaves"]) == (18_444, 1_009, 3_372)
    assert [round(1 - e["parts"][k] / t5["parts"][k], 3) for k in CHANGED] == [0.155, 0.038]
    assert (round(1 - 24_779 / 28_842, 3), round(1 - 26_306 / 27_285, 3)) == (0.141, 0.036)
