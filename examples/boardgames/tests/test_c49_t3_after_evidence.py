"""§3.6 T3 — finding registered AFTER the data, from evidence/c49_t3.json.gz and T2's: what the levels' context saved.
The file takes 115,599 bits (14,450 bytes), 0.9780 of T2's 118,197. Every part but the levels is T2's to a tenth of a
bit; the levels take 43,495.5 bits against 46,095.6, 2.36 a cell of the 18,444 against 2.50, 5.6% fewer, the share the
same context saved on G3's T2 file in the dry run (76,981 to 72,644)."""
from __future__ import annotations

import base64

import pytest

from harness.floor_t3 import SPEC
from tests.test_c49_t3_evidence import FILES
from tests.test_c49_z2_evidence import EVIDENCE, _evidence

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_t3_the_file_takes_115_599_bits_and_only_the_levels_changed():
    e, t2 = _evidence(SPEC["file"]), _evidence(SPEC["source"])
    assert (e["bits"], len(base64.b64decode(e["file"]))) == (115_599, 14_450)
    assert round(e["bits"] / SPEC["control_bits"], 4) == 0.9780
    assert {k: v for k, v in e["parts"].items() if k != "levels"} == {
        k: v for k, v in t2["parts"].items() if k != "levels"}
    assert (e["parts"]["levels"], t2["parts"]["levels"]) == (43_495.5, 46_095.6)
    assert (e["cells"], e["maps"], e["leaves"]) == (18_444, 1_009, 3_372)
    assert (round(e["parts"]["levels"] / e["cells"], 2), round(t2["parts"]["levels"] / e["cells"], 2)) == (2.36, 2.50)
    assert round(1 - e["parts"]["levels"] / t2["parts"]["levels"], 3) == round(1 - 72_644 / 76_981, 3) == 0.056
