"""Direct tests for harness/floor_e2.py — E2, how small a certified first-player Kalah strategy is against the whole
game graph, and how much choosing among value-keeping moves shrinks it. Judged on the shapes whose graph holds at
least 1,000 positions."""
from __future__ import annotations

import pytest

from harness.floor_e2 import SPEC, e2_report

TEST_SPEC = {"measurement_fp": "m" * 12, "shapes": [[2, 2], [3, 3], [4, 2]], "min_graph": 1000,
             "fraction_support": 0.05, "fraction_refute": 0.10, "halving_support": 2 / 3, "halving_refute": 1 / 3}


def _row(shape, graph, canonical, best, certified=True):
    return {"shape": shape, "graph": graph, "canonical": {"decisions": canonical, "certified": certified},
            "min_tree": {"decisions": best, "tree": best, "certified": certified}}


def _readout(rows=None):
    rows = rows if rows is not None else [_row([2, 2], 22, 6, 4), _row([3, 3], 23118, 400, 150),
                                          _row([4, 2], 52000, 2000, 900)]
    return {"measurement_fingerprint": "m" * 12, "shapes": rows}


def test_small_graphs_are_reported_but_not_judged():
    r = e2_report(_readout(), TEST_SPEC)
    assert r["integrity"] == [] and r["judged"] == [[3, 3], [4, 2]]
    assert r["fraction"]["values"] == [pytest.approx(150 / 23118), pytest.approx(900 / 52000)]


@pytest.mark.parametrize("best,verdict", [(900, "supported"), (2600, "supported"), (2601, "inconclusive"),
                                          (5200, "inconclusive"), (5201, "refuted")])
def test_the_fraction_bars_are_registered(best, verdict):
    rows = [_row([2, 2], 22, 6, 4), _row([3, 3], 23118, 400, 150), _row([4, 2], 52000, 6000, best)]
    assert e2_report(_readout(rows), TEST_SPEC)["fraction"]["verdict"] == verdict


@pytest.mark.parametrize("pairs,verdict", [([(400, 150), (2000, 900), (100, 40)], "supported"),
                                           ([(400, 150), (2000, 900), (100, 60)], "supported"),
                                           ([(400, 150), (2000, 1100), (100, 60)], "refuted"),
                                           ([(400, 200), (2000, 1100), (100, 60)], "refuted")])
def test_halving_needs_two_thirds_of_judged_shapes(pairs, verdict):
    rows = [_row([s, 9], 5000, c, b) for s, (c, b) in zip((3, 4, 5), pairs)]
    spec = {**TEST_SPEC, "shapes": [[3, 9], [4, 9], [5, 9]]}
    assert e2_report(_readout(rows), spec)["halving"]["verdict"] == verdict


def test_between_the_halving_bars_is_inconclusive():
    pairs = [(400, 150), (2000, 1100), (100, 60), (100, 40), (100, 60), (100, 50)]
    rows = [_row([s, 9], 5000, c, b) for s, (c, b) in zip(range(3, 9), pairs)]
    spec = {**TEST_SPEC, "shapes": [[s, 9] for s in range(3, 9)]}
    assert e2_report(_readout(rows), spec)["halving"]["verdict"] == "inconclusive"


@pytest.mark.parametrize("breakage", ["fingerprint", "missing shape", "uncertified canonical", "uncertified best"])
def test_a_readout_that_is_not_the_registered_measurement_is_not_run(breakage):
    e = _readout()
    if breakage == "fingerprint":
        e["measurement_fingerprint"] = "0" * 12
    elif breakage == "missing shape":
        e["shapes"] = e["shapes"][1:]
    elif breakage == "uncertified canonical":
        e["shapes"][1]["canonical"]["certified"] = False
    else:
        e["shapes"][2]["min_tree"]["certified"] = False
    r = e2_report(e, TEST_SPEC)
    assert r["integrity"] and r["fraction"] == {"verdict": "not_run"} and r["halving"] == {"verdict": "not_run"}


def test_the_registered_study_measures_every_shape_the_solver_handles_in_memory():
    assert SPEC["shapes"] == [[m, n] for m in (1, 2, 3) for n in range(1, 7)] + [[4, 1], [4, 2], [5, 1], [6, 1]]
    assert SPEC["min_graph"] == 1000 and (SPEC["fraction_support"], SPEC["fraction_refute"]) == (0.05, 0.10)
    assert (SPEC["halving_support"], SPEC["halving_refute"]) == (2 / 3, 1 / 3)
