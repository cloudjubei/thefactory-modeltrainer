"""Direct tests for harness/floor_hybrid.py — H1: how far does a self-play net carry play beyond an exact opening
exception table? For each net and table horizon h, the hybrid (table before h, net after) is certified through h + 2
plies, so the net must carry one ply of its own."""
from __future__ import annotations

import pytest

from harness.floor_hybrid import SPEC, hybrid_report

TEST_SPEC = {"runs": [["A.json.gz", [1, 2, 3, 4, 5]], ["B.json.gz", [9]]], "horizons": [3, 5, 7], "judged": [3, 5],
             "measurement_fp": "m" * 12, "support_at": 5, "refute_at": 3}


def _net(run, seed, certified=(True, True, False), entries=(2, 6, 20)):
    return {"run": run, "seed": seed, "net_file_sha256": "f",
            "horizons": {str(h): {"depth": h + 2, "certified": c, "entries": e, "positions": 3 * e, "failures": 0 if c
                                  else 2, "failures_by_ply": {} if c else {str(h + 1): 2}}
                         for h, c, e in zip(TEST_SPEC["horizons"], certified, entries)}}


def _evidence(overrides=None):
    overrides = overrides or {}
    nets = [_net(run, s, **overrides.get(s, {})) for run, seeds in TEST_SPEC["runs"] for s in seeds]
    return {"measurement_fingerprint": TEST_SPEC["measurement_fp"], "horizons": TEST_SPEC["horizons"],
            "runs": TEST_SPEC["runs"], "nets": nets}


def test_each_judged_horizon_counts_the_nets_whose_hybrid_is_certified_one_ply_past_the_table():
    r = hybrid_report(_evidence(), TEST_SPEC)
    assert r["integrity"] == []
    assert r["horizons"]["3"]["certified"] == 6 and r["horizons"]["3"]["verdict"] == "supported"
    assert r["horizons"]["7"]["certified"] == 0 and "verdict" not in r["horizons"]["7"]


@pytest.mark.parametrize("failing,verdict", [(0, "supported"), (1, "supported"), (2, "inconclusive"),
                                             (3, "refuted"), (6, "refuted")])
def test_the_bars_apply_to_the_count_of_certified_nets(failing, verdict):
    seeds = [1, 2, 3, 4, 5, 9]
    r = hybrid_report(_evidence({s: {"certified": (True, False, False)} for s in seeds[:failing]}), TEST_SPEC)
    assert r["horizons"]["5"]["certified"] == 6 - failing and r["horizons"]["5"]["verdict"] == verdict


def test_table_sizes_and_the_net_s_failures_are_reported_per_horizon():
    r = hybrid_report(_evidence({9: {"entries": (1, 4, 30)}}), TEST_SPEC)
    assert r["horizons"]["7"]["entries"] == [20, 20, 20, 20, 20, 30]
    assert r["horizons"]["7"]["failures_by_ply"] == {"8": 12}


@pytest.mark.parametrize("breakage", ["fingerprint", "horizons", "missing net", "extra net", "missing horizon",
                                      "wrong depth"])
def test_evidence_that_is_not_the_registered_measurement_is_not_run(breakage):
    e = _evidence()
    if breakage == "fingerprint":
        e["measurement_fingerprint"] = "0" * 12
    elif breakage == "horizons":
        e["horizons"] = [3, 5]
    elif breakage == "missing net":
        e["nets"] = e["nets"][1:]
    elif breakage == "extra net":
        e["nets"].append(_net("B.json.gz", 10))
    elif breakage == "missing horizon":
        del e["nets"][2]["horizons"]["5"]
    else:
        e["nets"][0]["horizons"]["3"]["depth"] = 4
    r = hybrid_report(e, TEST_SPEC)
    assert r["integrity"] and all(h == {"verdict": "not_run"} for h in r["horizons"].values())


def test_the_registered_measurement_reads_the_ten_base_recipe_nets_at_three_horizons():
    assert SPEC["runs"] == [["c49_T19_base.json.gz", list(range(451, 458))], ["c49_T20_s471.json.gz", [471]],
                            ["c49_T20_s472.json.gz", [472]], ["c49_T20_s473.json.gz", [473]]]
    assert SPEC["horizons"] == [3, 5, 7] and SPEC["judged"] == [3, 5]
    assert (SPEC["support_at"], SPEC["refute_at"]) == (8, 5)
