"""§3.6 T5 — finding registered AFTER the data, from evidence/c49_t5.json.gz and T4's: what the exception flags'
context saved. The file takes 109,034 bits (13,630 bytes), 0.9680 of T4's 112,644. Every part but the exception flags
is T4's to a tenth of a bit; the exception flags take 14,273.0 bits against 17,883.9, 20.2% fewer, near the 19.0% the
same context saved on G3's file in the dry run (33,698 to 27,285)."""
from __future__ import annotations

import base64

import pytest

from harness.floor_t5 import SPEC
from tests.test_c49_t5_evidence import FILES
from tests.test_c49_z2_evidence import EVIDENCE, _evidence

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_t5_the_file_takes_109_034_bits_and_only_the_exception_flags_changed():
    e, t4 = _evidence(SPEC["file"]), _evidence(SPEC["source"])
    assert (e["bits"], len(base64.b64decode(e["file"]))) == (109_034, 13_630)
    assert round(e["bits"] / SPEC["control_bits"], 4) == 0.9680
    assert {k: v for k, v in e["parts"].items() if k != "exception_flags"} == {
        k: v for k, v in t4["parts"].items() if k != "exception_flags"}
    assert (e["parts"]["exception_flags"], t4["parts"]["exception_flags"]) == (14_273.0, 17_883.9)
    assert (e["cells"], e["maps"], e["leaves"]) == (18_444, 1_009, 3_372)
    assert round(1 - e["parts"]["exception_flags"] / t4["parts"]["exception_flags"], 3) == 0.202
    assert round(1 - 27_285 / 33_698, 3) == 0.190
