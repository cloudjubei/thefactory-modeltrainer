"""Direct tests for harness/floor_c4_backplay.py — T16: the backward curriculum in T14's Connect-4 pilot shape, base
retrained on fresh seeds, judged by T14's judge."""
from __future__ import annotations

from harness.floor_c4_backplay import SPEC
from harness.floor_c4_value_signal import SPEC as T14_SPEC


def test_the_pilot_is_t14_s_base_against_the_curriculum_on_fresh_seeds_with_t14_s_bars():
    base = T14_SPEC["arms"]["base"]
    assert SPEC["arms"] == {"base": base, "backplay": {**base, "backplay": {"frac": 0.5, "ramp": 10}}}
    assert SPEC["treatments"] == ("backplay",)
    assert SPEC["seeds"] == (411, 412, 413, 414) and not set(SPEC["seeds"]) & set(T14_SPEC["seeds"])
    same = ("readout", "support_wins", "refute_wins", "min_gain")
    assert {k: SPEC[k] for k in same} == {k: T14_SPEC[k] for k in same}
    assert SPEC["era"] != T14_SPEC["era"]
