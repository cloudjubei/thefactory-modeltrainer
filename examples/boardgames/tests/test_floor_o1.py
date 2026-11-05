"""Direct tests for harness/floor_o1.py — O1, the builder's move at ply 6 across the opening: the 31 ply-6 positions
beyond E1's 8 where the builder's move rule, given G3's library, picks another move than the opening, rebuilt from ply
6 as E1 built its 8 and judged as E1 judged them, once the scope evidence names the same 31."""
from __future__ import annotations

import harness.floor_o1 as floor_o1
from harness.floor_e1 import SPEC as E1
from harness.floor_o1 import SPEC, o1_report, o1_selected, o1_size
from harness.floor_w6 import w6_report

INDICES = [3, 13, 14, 21, 28, 37, 41, 42, 43, 46, 48, 52, 54, 73, 77, 78, 80, 81, 89, 92, 97, 103, 105, 113, 118, 124,
           125, 126, 127, 135, 140]


def _scope(picks, **over):
    """`picks` is [(index, opening's move, builder's pick), ...]."""
    return {"positions_fp": SPEC["positions_fp"], "library_from": SPEC["library_from"],
            "positions": [{"index": i, "opening": o, "library_pick": p} for i, o, p in picks], **over}


def test_o1_rebuilds_the_positions_the_builder_moves_differently_beyond_e1_s_eight():
    assert SPEC["indices"] == INDICES and SPEC["roots"] == 31 and SPEC["excluded"] == E1["indices"]
    assert (SPEC["complete_support"], SPEC["complete_refute"], SPEC["support"]) == (29, 25, (3, 4))
    assert SPEC["g3_subtree_bits"] == 51_312 and SPEC["scope"] == "c49_o1_scope.json.gz"
    assert {k: v for k, v in SPEC.items() if k in E1 and k not in {
        "indices", "roots", "complete_support", "complete_refute", "support", "g3_subtree_bits"}} == {
        k: v for k, v in E1.items() if k not in {
            "indices", "roots", "complete_support", "complete_refute", "support", "g3_subtree_bits"}}
    assert o1_report is w6_report


def test_the_selection_is_every_position_picked_differently_but_e1_s():
    picks = [(0, 2, 2), (1, 0, 3), (18, 2, 4), (2, 1, 0), (5, 3, 3), (115, 2, 4)]
    assert o1_selected(_scope(picks), {**SPEC, "excluded": [18, 115]}) == [1, 2]
    assert o1_selected(_scope(picks), {**SPEC, "excluded": [115]}) == [1, 2, 18]


def test_a_scope_of_other_positions_or_another_library_selects_nothing():
    picks = [(1, 0, 3)]
    assert o1_selected(_scope(picks, positions_fp="x"), SPEC) is None
    assert o1_selected(_scope(picks, library_from="other.json.gz"), SPEC) is None


def test_o1_is_judged_as_e1_only_on_the_registered_selection(monkeypatch):
    calls = []
    monkeypatch.setattr(floor_o1, "e1_size", lambda *args: calls.append(args) or {"verdict": "supported"})
    spec = {**SPEC, "indices": [1, 2], "excluded": [18]}
    scope = _scope([(1, 0, 3), (2, 1, 0), (18, 2, 4)])
    assert o1_size("e", "g3", "library", "moves", scope, spec) == {"verdict": "supported"}
    assert calls == [("e", "g3", "library", "moves", spec)]
    for other in (_scope([(1, 0, 3)]), _scope([(1, 0, 3), (2, 1, 0), (3, 0, 1)]),
                  _scope([(1, 0, 3), (2, 1, 0)], positions_fp="x")):
        assert o1_size("e", "g3", "library", "moves", other, spec) == {"verdict": "not_run"}
    assert len(calls) == 1
