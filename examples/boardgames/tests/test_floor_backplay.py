"""Direct tests for harness/floor_backplay.py — T15: T9's tic-tac-toe process with the backward curriculum, judged by
T9's own judge on its own registered recipe."""
from __future__ import annotations

from harness.floor_backplay import SPEC, T15_ARM
from harness.floor_tree import SPEC as T9_SPEC, T9_ARM


def test_the_recipe_is_t9_s_with_half_the_games_started_late_reaching_whole_games_at_iteration_15():
    assert SPEC["arms"][T15_ARM] == {**T9_SPEC["arms"][T9_ARM], "backplay": {"frac": 0.5, "ramp": 15}}
    assert SPEC["seeds"] == tuple(range(421, 431)) and not set(SPEC["seeds"]) & set(T9_SPEC["seeds"])
    same = ("iterations", "positions", "latency", "certify_at", "support_at", "refute_at")
    assert {k: SPEC[k] for k in same} == {k: T9_SPEC[k] for k in same}
    assert SPEC["params"] == {T15_ARM: T9_SPEC["params"][T9_ARM]}
