"""§3.6 U2 — PRE-REGISTERED claim on choosing the whole game again from every certified piece the runs hold, X4's
mirror-off rebuilds among them: judged by harness.floor_u2 from the X4 evidence and the runs before it. No new builds;
the search had not been run when this was registered. The whole game takes 199/200 of 166,623 or less."""
from __future__ import annotations

from functools import lru_cache

import pytest

from harness.floor_g4 import SPEC as G4
from harness.floor_u2 import SPEC, u2_size
from harness.floor_x2 import SPEC as X2
from harness.floor_x3 import SPEC as X3
from harness.floor_x4 import SPEC as X4
from harness.floor_z1 import SPEC as Z1
from tests.test_c49_v1_evidence import _assembly
from tests.test_c49_x3_evidence import EARLIER
from tests.test_c49_x4_evidence import EVIDENCE, FILES, LABELS, _evidence

pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES) or not LABELS.exists(),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _size() -> dict:
    earlier = [(_evidence(name), spec) for name, spec in EARLIER.items()]
    return u2_size((_evidence(), X4), (_evidence("c49_x3.json.gz"), X3), (_evidence("c49_x2.json.gz"), X2),
                   (_evidence("c49_z1.json.gz"), Z1), earlier, (_evidence("c49_g4.json.gz"), G4),
                   _evidence("c49_g3.json.gz"), _assembly(), SPEC)


def test_c49_u2_the_whole_game_chosen_again_takes_199_200_of_166_623_or_less():
    assert _size()["verdict"] == "supported"


def test_c49_u2_the_size_is_undecidable():
    assert _size()["verdict"] in {"inconclusive", "not_run"}
