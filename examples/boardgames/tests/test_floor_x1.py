"""Direct tests for harness/floor_x1.py — X1, mirror sharing priced on G3's certified whole game: maps stored once per
mirror pair, and frontier positions played as their mirror image's strategy, each judged on the share of G3's bits it
saves."""
from __future__ import annotations

from harness.floor_x1 import SPEC, x1_maps, x1_roots


def _e(maps_price=260_000, roots_price=250_000, **over):
    return {"g3_bits": 280_640, "symmetry_checked": 1_000, "maps": {"price": maps_price},
            "roots": {"price": roots_price}, **over}


def test_x1_prices_mirror_sharing_against_g3():
    assert SPEC == {"g3_bits": 280_640, "maps_support": (1, 20), "maps_refute": (1, 100), "roots_support": (1, 10),
                    "roots_refute": (1, 50)}


def test_maps_shared_by_mirror_are_judged_on_the_share_saved():
    even = {**SPEC, "g3_bits": 300_000}
    for price, verdict in ((285_000, "supported"), (285_001, "inconclusive"), (296_999, "inconclusive"),
                           (297_000, "refuted"), (310_000, "refuted")):
        r = x1_maps(_e(maps_price=price, g3_bits=300_000), even)
        assert r == {"verdict": verdict, "bits": price, "saved": 300_000 - price, "share": (300_000 - price) / 300_000}


def test_frontier_positions_played_mirrored_are_judged_on_the_share_saved():
    even = {**SPEC, "g3_bits": 300_000}
    for price, verdict in ((270_000, "supported"), (270_001, "inconclusive"), (293_999, "inconclusive"),
                           (294_000, "refuted")):
        r = x1_roots(_e(roots_price=price, g3_bits=300_000), even)
        assert r == {"verdict": verdict, "bits": price, "saved": 300_000 - price, "share": (300_000 - price) / 300_000}


def test_the_bars_are_g3_s_registered_bits():
    assert x1_maps(_e(maps_price=266_608), SPEC)["verdict"] == "supported"
    assert x1_roots(_e(roots_price=252_576), SPEC)["verdict"] == "supported"
    assert x1_roots(_e(roots_price=252_577), SPEC)["verdict"] == "inconclusive"


def test_a_measurement_off_g3_or_without_the_symmetry_checked_is_not_judged():
    for over in ({"g3_bits": 280_641}, {"symmetry_checked": 0}):
        assert x1_maps(_e(**over), SPEC) == {"verdict": "not_run"}
        assert x1_roots(_e(**over), SPEC) == {"verdict": "not_run"}
