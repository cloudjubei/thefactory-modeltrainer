"""§3.6 T4 — finding registered AFTER the data, from evidence/c49_t4.json.gz and T3's: what the cell flags' context
saved. The file takes 112,644 bits (14,081 bytes), 0.9744 of T3's 115,599. Every part but the flags is T3's to a tenth
of a bit; the flags take 27,995.3 bits against 30,949.4, 9.5% fewer, near the 9.9% the same context saved on G3's file
in the dry run (51,994 to 46,828)."""
from __future__ import annotations

import base64

import pytest

from harness.floor_t4 import SPEC
from tests.test_c49_t4_evidence import FILES
from tests.test_c49_z2_evidence import EVIDENCE, _evidence

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def test_c49_t4_the_file_takes_112_644_bits_and_only_the_flags_changed():
    e, t3 = _evidence(SPEC["file"]), _evidence(SPEC["source"])
    assert (e["bits"], len(base64.b64decode(e["file"]))) == (112_644, 14_081)
    assert round(e["bits"] / SPEC["control_bits"], 4) == 0.9744
    assert {k: v for k, v in e["parts"].items() if k != "cells"} == {
        k: v for k, v in t3["parts"].items() if k != "cells"}
    assert (e["parts"]["cells"], t3["parts"]["cells"]) == (27_995.3, 30_949.4)
    assert (e["cells"], e["maps"], e["leaves"]) == (18_444, 1_009, 3_372)
    assert round(1 - e["parts"]["cells"] / t3["parts"]["cells"], 3) == 0.095
    assert round(1 - 46_828 / 51_994, 3) == 0.099
