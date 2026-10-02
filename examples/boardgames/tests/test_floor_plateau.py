"""Direct tests for harness/floor_plateau.py — the pre-registered diagnosis of the Connect-4 stop-signal plateau:
pooled over the saved T10 nets, is the net's raw move or its search's preferred move the one the solver rejects where
they disagree, and are the nets certified through the horizon anyway."""
from __future__ import annotations

import pytest

from harness.floor_plateau import plateau_report

SPEC = {"nets": {1: "a" * 64, 2: "b" * 64, 3: "c" * 64}, "config": {"sims": 200, "depth": 10, "agree_share": 0.5},
        "era": "e" * 12, "measurement_fp": "m" * 12, "search_blocker_at": 2 / 3, "net_blocker_at": 1 / 3,
        "certified_support": 2, "certified_refute": 1}


def _row(seed, classes=(6, 2, 1, 1), certified=True, failures_by_ply=None):
    both_optimal, search_wrong, net_wrong, both_wrong = classes
    failures_by_ply = failures_by_ply if failures_by_ply is not None else ({} if certified else {"5": 2})
    return {"seed": seed, "net_file_sha256": SPEC["nets"][seed], "walked": 900,
            "disagreements": sum(classes),
            "classes": {"both_optimal": both_optimal, "search_wrong": search_wrong, "net_wrong": net_wrong,
                        "both_wrong": both_wrong},
            "certificate": {"certified": certified, "horizon": 10, "failures": sum(failures_by_ply.values()),
                            "failures_by_ply": failures_by_ply, "complete": True}}


def _evidence(rows=None):
    rows = rows or {}
    return {"training_fingerprint": SPEC["era"], "measurement_fingerprint": SPEC["measurement_fp"],
            "config": dict(SPEC["config"]), "seeds": [_row(s, **rows.get(s, {})) for s in SPEC["nets"]]}


def test_mostly_right_nets_blocked_by_their_search_support_the_search_reading():
    r = plateau_report(_evidence(), SPEC)
    assert r["integrity"] == [] and r["blocker"]["verdict"] == "supported"
    assert r["blocker"]["net_right_share"] == pytest.approx(24 / 30)
    assert r["certified"]["verdict"] == "supported" and r["certified"]["count"] == 3
    assert r["descriptives"]["deepest_certified"] == [10, 10, 10]


@pytest.mark.parametrize("classes,verdict", [((2, 0, 1, 0), "supported"), ((1, 1, 1, 0), "supported"),
                                             ((1, 0, 2, 0), "refuted"), ((0, 1, 1, 1), "refuted"),
                                             ((1, 0, 1, 0), "inconclusive")])
def test_the_blocker_reading_pools_the_share_of_disagreements_where_the_net_is_right(classes, verdict):
    r = plateau_report(_evidence({s: {"classes": classes} for s in SPEC["nets"]}), SPEC)
    assert r["blocker"]["verdict"] == verdict


def test_no_disagreement_at_all_cannot_name_a_blocker():
    r = plateau_report(_evidence({s: {"classes": (0, 0, 0, 0)} for s in SPEC["nets"]}), SPEC)
    assert r["blocker"]["verdict"] == "inconclusive"


@pytest.mark.parametrize("uncertified,verdict", [(0, "supported"), (1, "supported"), (2, "refuted"),
                                                 (3, "refuted")])
def test_the_certified_reading_counts_nets_certified_through_the_horizon(uncertified, verdict):
    seeds = list(SPEC["nets"])[:uncertified]
    r = plateau_report(_evidence({s: {"certified": False} for s in seeds}), SPEC)
    assert r["certified"]["count"] == 3 - uncertified and r["certified"]["verdict"] == verdict
    if uncertified:
        assert r["descriptives"]["deepest_certified"][0] == 5


def _set(path, value):
    def apply(e):
        target = e
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = value
    return apply


BREAKAGES = {
    "era": _set(("training_fingerprint",), "0" * 12),
    "measurement": _set(("measurement_fingerprint",), "0" * 12),
    "config": _set(("config", "sims"), 32),
    "another net": _set(("seeds", 0, "net_file_sha256"), "z" * 64),
    "missing seed": lambda e: e["seeds"].pop(),
    "classes do not add up": _set(("seeds", 1, "disagreements"), 99),
    "shallower certificate": _set(("seeds", 2, "certificate", "horizon"), 8),
    "uncertified without failure or cut": _set(("seeds", 2, "certificate", "certified"), False),
}


@pytest.mark.parametrize("breakage", sorted(BREAKAGES))
def test_evidence_that_is_not_the_registered_run_is_not_run(breakage):
    e = _evidence()
    BREAKAGES[breakage](e)
    r = plateau_report(e, SPEC)
    assert r["integrity"] and r["blocker"]["verdict"] == "not_run" and r["certified"]["verdict"] == "not_run"


@pytest.mark.parametrize("e", [None, {}, {"seeds": "x"}, []])
def test_missing_or_unreadable_evidence_is_not_run(e):
    assert plateau_report(e, SPEC)["blocker"]["verdict"] == "not_run"


def test_a_certificate_cut_short_is_named_as_such():
    e = _evidence()
    e["seeds"][0]["certificate"]["certified"] = False
    assert any("cut short" in p for p in plateau_report(e, SPEC)["integrity"])
