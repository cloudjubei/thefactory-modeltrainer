"""Direct tests for harness/floor_r2.py — R2, G1 rebuilt with the builder counting its references by first_use (the
judges' code) instead of the index's full width, judged on completion out of 16 and on its total against G1's, both
coded by first_use with each library map a leaf plays paid once."""
from __future__ import annotations

from harness.floor_b1 import SPEC as B1
from harness.floor_b1_control import SPEC as CONTROL
from harness.floor_g1 import SPEC as G1, g1_size
from harness.floor_r2 import SPEC, r2_coded, r2_report, r2_size
from harness.floor_w6 import w6_report
from harness.reference_coding import code_bits
from harness.steady_state import map_bits

BOARD = [0] * 42
OWN = [{"3": 0}, {"3": 0, "10": 1}]
LIBRARY = [OWN[k % 2] for k in range(16)]


def _root(index, nodes, leaf, maps, shared, complete=True, empty=42):
    return {"index": index, "complete": complete, "bits": {"nodes": nodes, "leaves": 1}, "board": BOARD,
            "to_move": 0, "nodes": [[BOARD, 0, {"leaf": leaf}]], "maps": [{}, *maps],
            "map_empty": [0] + [empty] * len(maps), "shared": shared, "found_maps": [[m, 42] for m in maps[shared:]],
            "build_seconds": 600.0, "checked": True if complete else None,
            "moves_win": True if complete else None, "leaves_certified": True if complete else None}


def _e(spec, roots, **over):
    return {"measurement_fingerprint": spec["measurement_fp"], "positions_fingerprint": spec["positions_fp"],
            "builder": spec["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def _control():
    return _e(CONTROL, [_root(i, 100, 1, [OWN[k % 2]], 0) for k, i in enumerate(B1["indices"])])


def _built(spec, sizes, leaves=None, incomplete=(), library=None, empty=42):
    library = library or LIBRARY
    leaves = leaves or [1] * 16
    return _e(spec, [_root(i, s, leaf, library, 16, complete=i not in incomplete, empty=empty)
                     for i, s, leaf in zip(B1["indices"], sizes, leaves)])


def _coded(nodes, refs, n_maps, shared):
    return nodes - code_bits("fixed", refs, n_maps, shared) + code_bits("first_use", refs, n_maps, shared)


G1_RUN = _built(G1, [60 - g1_size(_built(G1, [60] * 16), _control())["bits"] % 10] + [60] * 15)
G1_BITS = g1_size(G1_RUN, _control())["bits"]
JUDGED = {**SPEC, "g1_bits": G1_BITS}


def test_r2_is_g1_with_the_builder_counting_references_by_first_use():
    assert SPEC == {**G1, "measurement_fp": "77fc4515e75e", "builder": {**G1["builder"], "refs": "first_use"},
                    "g1_bits": 74_556}
    assert "refs" not in G1["builder"] and SPEC["support"] == 0.9 and SPEC["roots"] == 16
    assert (SPEC["complete_support"], SPEC["complete_refute"]) == (14, 12)
    assert r2_report is w6_report


def test_a_finished_build_s_count_is_already_by_first_use_and_is_not_recoded():
    root = _root(1, 30, 1, LIBRARY, 16)
    assert r2_coded(root) == 30 != _coded(30, [1], 17, 16)


def test_the_size_is_the_strategies_by_first_use_and_each_played_library_map_once():
    leaves = [1] * 10 + [2] * 6
    r = r2_size(_built(SPEC, [30] * 16, leaves=leaves), G1_RUN, _control(), JUDGED)
    bits = 16 * 30 + map_bits({3: 0}, 42, 3) + map_bits({3: 0, 10: 1}, 42, 3)
    assert r == {"verdict": "supported", "bits": bits, "control_bits": G1_BITS, "ratio": bits / G1_BITS,
                 "lower_bound": False, "library_maps": 2}


def test_the_verdict_is_judged_against_g1_s_bits():
    assert G1_BITS % 10 == 0
    fixed = map_bits({3: 0}, 42, 3)

    def sizes(total):
        return [total - fixed - 15 * ((total - fixed) // 16)] + [(total - fixed) // 16] * 15

    for total, verdict in ((G1_BITS * 9 // 10, "supported"), (G1_BITS * 9 // 10 + 1, "inconclusive"),
                           (G1_BITS - 1, "inconclusive"), (G1_BITS, "refuted")):
        r = r2_size(_built(SPEC, sizes(total)), G1_RUN, _control(), JUDGED)
        assert (r["bits"], r["verdict"]) == (total, verdict)


def test_the_empty_map_and_a_build_s_own_maps_are_not_charged_as_library():
    r = r2_size(_built(SPEC, [30] * 16, leaves=[0] * 16), G1_RUN, _control(), JUDGED)
    assert r["library_maps"] == 0 and r["bits"] == 16 * 30
    arm = _built(SPEC, [30] * 16)
    own = arm["roots"][0]
    own["maps"].append({"9": 0})
    own["map_empty"].append(42)
    own["nodes"] = [[BOARD, 0, {"leaf": 17}]]
    r = r2_size(arm, G1_RUN, _control(), JUDGED)
    assert r["library_maps"] == 1
    assert r["bits"] == 16 * 30 + map_bits({3: 0}, 42, 3)


def test_an_unfinished_root_counts_without_its_references_and_can_only_refute():
    first = B1["indices"][0]
    r = r2_size(_built(SPEC, [30] * 16, incomplete=(first,)), G1_RUN, _control(), JUDGED)
    assert r["bits"] == 15 * 30 + 30 - code_bits("first_use", [1], 17, 16) + map_bits({3: 0}, 42, 3)
    assert r["lower_bound"] and r["verdict"] == "inconclusive"
    assert r2_size(_built(SPEC, [200] * 16, incomplete=(first,)), G1_RUN, _control(), JUDGED)["verdict"] == "refuted"


def test_a_build_that_did_not_start_from_g1_s_library_is_not_judged():
    assert r2_size(_built(SPEC, [30] * 16, library=LIBRARY[::-1]), G1_RUN, _control(), JUDGED)["verdict"] == "not_run"
    assert r2_size(_built(SPEC, [30] * 16, empty=41), G1_RUN, _control(), JUDGED)["verdict"] == "not_run"
    arm = _built(SPEC, [30] * 16)
    arm["roots"][3]["shared"] = 15
    assert r2_size(arm, G1_RUN, _control(), JUDGED)["verdict"] == "not_run"


def test_a_run_off_its_registration_is_not_judged():
    arm = _built(SPEC, [30] * 16)
    assert r2_size(_e(SPEC, arm["roots"], builder=G1["builder"]), G1_RUN, _control(), JUDGED)["verdict"] == "not_run"
    assert r2_size(_e(SPEC, arm["roots"][1:]), G1_RUN, _control(), JUDGED)["verdict"] == "not_run"
    assert r2_size(_e(SPEC, arm["roots"], measurement_fingerprint=G1["measurement_fp"]), G1_RUN, _control(),
                   JUDGED)["verdict"] == "not_run"


def test_g1_must_be_judged_complete_at_its_registered_bits():
    arm = _built(SPEC, [30] * 16)
    assert r2_size(arm, G1_RUN, _control(), {**JUDGED, "g1_bits": G1_BITS + 1})["verdict"] == "not_run"
    unfinished = _built(G1, [60] * 16, incomplete=(B1["indices"][5],))
    assert g1_size(unfinished, _control())["lower_bound"]
    assert r2_size(arm, unfinished, _control(), {**JUDGED, "g1_bits": g1_size(unfinished, _control())["bits"]})[
        "verdict"] == "not_run"
    assert r2_size(arm, _built(SPEC, [60] * 16), _control(), JUDGED)["verdict"] == "not_run"
