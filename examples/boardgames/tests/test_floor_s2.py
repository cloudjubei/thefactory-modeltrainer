"""Direct tests for harness/floor_s2.py — S2, complete certified strategies (table moves + steady-state leaves) for the
rest of the game from sampled ply-10 positions of the canonical exact Connect-4 table."""
from __future__ import annotations

import pytest

from harness.floor_s2 import SPEC, s2_report

TEST_SPEC = {"measurement_fp": "m" * 12, "positions_fp": "p" * 12, "roots": 4, "builder": {"min_leaf_depth": 2},
             "complete_support": 3, "complete_refute": 1, "compression_support": 10, "compression_refute": 1}


def _root(complete=True, bits=100, own=1000, checked=True, moves_win=True, leaves_certified=True):
    return {"complete": complete, "bits": {"nodes": bits}, "own_positions": own if complete else None,
            "checked": checked if complete else None, "moves_win": moves_win if complete else None,
            "leaves_certified": leaves_certified if complete else None}


def _readout(roots=None):
    roots = roots if roots is not None else [_root(own=1000), _root(own=500), _root(own=2000), _root(complete=False)]
    return {"measurement_fingerprint": "m" * 12, "positions_fingerprint": "p" * 12,
            "builder": {"min_leaf_depth": 2}, "roots": roots}


def test_completed_roots_are_compared_with_the_same_strategy_as_a_3_bit_table():
    r = s2_report(_readout(), TEST_SPEC)
    assert r["integrity"] == [] and r["complete"] == {"verdict": "supported", "count": 3}
    assert r["compression"]["values"] == [pytest.approx(30.0), pytest.approx(15.0), pytest.approx(60.0)]
    assert r["compression"]["median"] == pytest.approx(30.0) and r["compression"]["verdict"] == "supported"


@pytest.mark.parametrize("n,verdict", [(4, "supported"), (3, "supported"), (2, "inconclusive"), (1, "refuted"),
                                       (0, "refuted")])
def test_the_completion_bars_are_registered(n, verdict):
    roots = [_root()] * n + [_root(complete=False)] * (4 - n)
    assert s2_report(_readout(roots), TEST_SPEC)["complete"]["verdict"] == verdict


@pytest.mark.parametrize("own,verdict", [(400, "supported"), (300, "inconclusive"), (34, "inconclusive"),
                                         (33, "refuted")])
def test_the_compression_bars_are_registered(own, verdict):
    roots = [_root(own=own)] * 3 + [_root(complete=False)]
    assert s2_report(_readout(roots), TEST_SPEC)["compression"]["verdict"] == verdict


def test_no_completed_root_leaves_compression_not_run():
    roots = [_root(complete=False)] * 4
    assert s2_report(_readout(roots), TEST_SPEC)["compression"] == {"verdict": "not_run"}


@pytest.mark.parametrize("breakage", ["fingerprint", "positions", "builder", "roots", "unchecked", "losing move",
                                      "uncertified leaf"])
def test_a_readout_that_is_not_the_registered_measurement_is_not_run(breakage):
    e = _readout()
    if breakage == "fingerprint":
        e["measurement_fingerprint"] = "0" * 12
    elif breakage == "positions":
        e["positions_fingerprint"] = "0" * 12
    elif breakage == "builder":
        e["builder"] = {"min_leaf_depth": 0}
    elif breakage == "roots":
        e["roots"] = e["roots"][:3]
    elif breakage == "unchecked":
        e["roots"][0] = _root(checked=False)
    elif breakage == "losing move":
        e["roots"][1] = _root(moves_win=False)
    else:
        e["roots"][2] = _root(leaves_certified=False)
    r = s2_report(e, TEST_SPEC)
    assert r["integrity"] and r["complete"] == {"verdict": "not_run"} and r["compression"] == {"verdict": "not_run"}


def test_the_registered_study_builds_8_roots_with_registered_builder_settings():
    assert SPEC["roots"] == 8 and SPEC["seed"] == 2 and SPEC["ply"] == 10
    assert SPEC["builder"] == {"n_levels": 8, "level_bits": 3, "cap": 1_000_000, "min_leaf_depth": 2,
                               "reuse_window": 200, "seconds": 7200.0}
    assert SPEC["search"] == {"max_constraints": 20_000, "conflicts": 1_000_000, "seconds": 30.0, "cap": 1_000_000,
                              "lines": 64}
    assert (SPEC["complete_support"], SPEC["complete_refute"]) == (6, 3)
    assert (SPEC["compression_support"], SPEC["compression_refute"]) == (10, 1)


@pytest.mark.parametrize("own,verdict", [(100, "supported"), (10, "inconclusive")])
def test_a_median_exactly_on_a_bar_reads_as_registered(own, verdict):
    roots = [_root(own=own, bits=30)] * 3 + [_root(complete=False)]
    assert s2_report(_readout(roots), TEST_SPEC)["compression"]["verdict"] == verdict
