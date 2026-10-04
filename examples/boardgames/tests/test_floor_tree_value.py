"""Direct tests for harness/floor_tree_value.py — T13: T9's tic-tac-toe process with value targets for the
strategy-tree positions, judged by T9's own judge on its own registered recipe."""
from __future__ import annotations

from harness.floor_tree import SPEC as T9_SPEC, T9_ARM
from harness.floor_tree_value import SPEC, T13_ARM


def test_the_recipe_is_t9_s_with_tree_value_targets_and_fresh_seeds():
    assert SPEC["arms"][T13_ARM] == {**T9_SPEC["arms"][T9_ARM], "tree_value_target": True}
    assert SPEC["seeds"] == tuple(range(391, 401)) and not set(SPEC["seeds"]) & set(T9_SPEC["seeds"])
    same = ("iterations", "positions", "latency", "certify_at", "support_at", "refute_at")
    assert {k: SPEC[k] for k in same} == {k: T9_SPEC[k] for k in same}
    assert SPEC["params"] == {T13_ARM: T9_SPEC["params"][T9_ARM]}
