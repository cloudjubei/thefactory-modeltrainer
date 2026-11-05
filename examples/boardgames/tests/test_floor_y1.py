"""Direct tests for harness/floor_y1.py — Y1, WeakC4 as a yardstick for G3: judged on whether our rule reads its
diagrams (the share of its leaves won under "exact" with all ten levels) and on its whole solution's bits in our code
(the cheapest reading that wins at every leaf) against G3's 280,640."""
from __future__ import annotations

from harness.floor_y1 import SPEC, y1_language, y1_price


def _reading(won, leaves=100, total=50_000):
    return {"won": won, "leaves": leaves, "bits": {"total": total}}


def _e(exact_all=_reading(100), exact_sparse=_reading(100, total=40_000), g3_rule=_reading(60, total=30_000), **over):
    return {"weakc4_commit": "51dc838", "readings": {"exact_all": exact_all, "exact_sparse": exact_sparse,
                                                     "g3_rule": g3_rule}, **over}


def test_y1_measures_weakc4_against_g3_s_whole_game():
    assert SPEC == {"g3_bits": 280_640, "language_support": (9, 10), "language_refute": (1, 2),
                    "price_support": (1, 3), "price_refute": (1, 2)}


def test_the_language_is_the_share_of_leaves_won_under_the_exact_rule_with_all_ten_levels():
    for won, verdict in ((100, "supported"), (90, "supported"), (89, "inconclusive"), (51, "inconclusive"),
                         (50, "refuted"), (0, "refuted")):
        r = y1_language(_e(exact_all=_reading(won)), SPEC)
        assert r == {"verdict": verdict, "won": won, "leaves": 100}
    assert y1_language(_e(exact_all=_reading(9, leaves=10), exact_sparse=_reading(0)), SPEC)["verdict"] == "supported"


def test_the_price_is_the_cheapest_reading_that_wins_every_leaf():
    r = y1_price(_e(), SPEC)
    assert r == {"verdict": "supported", "bits": 40_000, "reading": "exact_sparse", "ratio": 40_000 / 280_640}
    r = y1_price(_e(exact_sparse=_reading(99, total=10_000)), SPEC)
    assert (r["bits"], r["reading"]) == (50_000, "exact_all")
    r = y1_price(_e(g3_rule=_reading(100, total=30_000)), SPEC)
    assert (r["bits"], r["reading"]) == (30_000, "g3_rule")


def test_the_price_is_judged_against_g3_s_bits():
    for total, verdict in ((280_640 // 3, "supported"), (280_640 // 3 + 1, "inconclusive"),
                           (280_640 // 2 - 1, "inconclusive"), (280_640 // 2, "refuted")):
        r = y1_price(_e(exact_all=_reading(100, total=total), exact_sparse=_reading(0)), SPEC)
        assert (r["verdict"], r["bits"]) == (verdict, total)


def test_a_price_of_exactly_a_third_is_supported_and_exactly_a_half_refuted():
    even = {**SPEC, "g3_bits": 300_000}
    for total, verdict in ((100_000, "supported"), (100_001, "inconclusive"), (149_999, "inconclusive"),
                           (150_000, "refuted")):
        assert y1_price(_e(exact_all=_reading(100, total=total), exact_sparse=_reading(0)), even)["verdict"] == verdict


def test_with_no_reading_winning_every_leaf_the_price_is_not_judged():
    assert y1_price(_e(exact_all=_reading(99), exact_sparse=_reading(98), g3_rule=_reading(97)), SPEC) == {
        "verdict": "not_run"}


def test_a_run_without_weakc4_s_commit_or_leaves_is_not_judged():
    assert y1_language(_e(weakc4_commit=""), SPEC) == {"verdict": "not_run"}
    assert y1_price(_e(weakc4_commit=""), SPEC) == {"verdict": "not_run"}
    assert y1_language(_e(exact_all=_reading(0, leaves=0)), SPEC) == {"verdict": "not_run"}
