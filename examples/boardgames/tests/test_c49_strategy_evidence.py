"""§C.49 Connect-4 strategy frontier — descriptive findings pinned to their evidence (registered after the data; not
predictions). evidence/c49_strategy_d8.json.gz, c49_strategy_d10.json.gz and c49_strategy_d10_wide.json.gz: nets grown
on their own first-player strategy trees with exact solver labels (harness.strategy_fit), certified or not through
the horizon."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILES = ("c49_strategy_d8.json.gz", "c49_strategy_d10.json.gz", "c49_strategy_d10_wide.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


def _runs(name):
    return {(r["family"], r["width"]): r for r in load_evidence(EVIDENCE / name)["runs"]}


def test_c49_a_1_6K_standardised_net_holds_a_certified_first_player_strategy_through_8_plies():
    r = _runs(FILES[0])[("canon_conv", 4)]
    assert r["certified"] and r["params"] == 1576 and r["rounds"][-1]["failures"] == 0


def test_c49_no_setup_up_to_23_5K_is_certified_through_10_plies_because_the_refits_do_not_hold_their_data():
    small = _runs(FILES[1])
    wide = _runs(FILES[2])
    assert not any(r["certified"] for r in [*small.values(), *wide.values()])
    for key in (("canon_conv", 16), ("canon_conv", 32), ("residual", 32)):
        r = wide[key]
        assert not r["stalled"] and len(r["rounds"]) == 20
        late = [e["fit"] for e in r["rounds"][10:-1]]
        assert all(f is not None and not f["solved"] and f["best_failures"] > 0 for f in late)
    assert max(e["fit"]["best_failures"] for e in wide[("canon_conv", 32)]["rounds"][12:-1]) <= 5
