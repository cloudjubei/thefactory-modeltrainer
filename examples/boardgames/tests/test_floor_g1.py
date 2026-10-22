"""Direct tests for harness/floor_g1.py — G1, L5's rule with every map of B1's control within reach, on the same 16
positions, judged on completion out of 16 and on its total against the control with both arms' references coded by
first_use and each library map a leaf plays paid once."""
from __future__ import annotations

from harness.floor_b1 import SPEC as B1
from harness.floor_b1_control import SPEC as CONTROL
from harness.floor_g1 import SPEC, g1_library, g1_report, g1_size
from harness.floor_k1 import SPEC as K1
from harness.floor_l5 import SPEC as L5
from harness.floor_w6 import w6_report
from harness.reference_coding import code_bits
from harness.steady_state import map_bits

BOARD = [0] * 42
OWN = [{"3": 0}, {"3": 0, "10": 1}]


def _root(index, nodes, leaf, maps, shared, complete=True, leaves=1):
    return {"index": index, "complete": complete, "bits": {"nodes": nodes, "leaves": leaves}, "board": BOARD,
            "to_move": 0, "nodes": [[BOARD, 0, {"leaf": leaf}]], "maps": [{}, *maps],
            "map_empty": [0] + [42] * len(maps),
            "shared": shared, "found_maps": [[m, 42] for m in maps[shared:]], "build_seconds": 600.0,
            "checked": True if complete else None, "moves_win": True if complete else None,
            "leaves_certified": True if complete else None}


def _e(spec, roots, **over):
    return {"measurement_fingerprint": spec["measurement_fp"], "positions_fingerprint": spec["positions_fp"],
            "builder": spec["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def _control(sizes=(100,) * 16):
    return _e(CONTROL, [_root(i, s, 1, [OWN[k % 2]], 0) for k, (i, s) in enumerate(zip(B1["indices"], sizes))])


def _arm(sizes, leaves=None, incomplete=(), library=None):
    library = library or [OWN[k % 2] for k in range(16)]
    leaves = leaves or [1] * 16
    return _e(SPEC, [_root(i, s, leaf, library, 16, complete=i not in incomplete)
                     for i, s, leaf in zip(B1["indices"], sizes, leaves)])


def test_g1_is_l5_with_b1_s_control_s_maps_all_in_reach_on_b1_s_positions():
    assert SPEC["indices"] == B1["indices"] and SPEC["roots"] == 16 and SPEC["library_from"] == "c49_b1_control.json.gz"
    assert (SPEC["complete_support"], SPEC["complete_refute"], SPEC["support"]) == (14, 12, 0.9)
    assert SPEC["builder"] == {**L5["builder"], "reuse_window": 100_000} and L5["builder"]["reuse_window"] == 200
    assert SPEC["search"] == L5["search"] and SPEC["measurement_fp"] == K1["measurement_fp"]
    assert g1_report is w6_report


def test_the_library_is_every_map_the_control_found_in_its_order():
    assert g1_library(_control()) == [(m, 42) for m in OWN * 8]


def _coded(nodes, refs, n_maps, shared):
    return nodes - code_bits("fixed", refs, n_maps, shared) + code_bits("first_use", refs, n_maps, shared)


def test_both_arms_are_coded_by_first_use_and_a_played_library_map_paid_once():
    r = g1_size(_arm([50] * 16, leaves=[1] * 8 + [2] * 8), _control())
    arm = 16 * _coded(50, [1], 17, 16) + map_bits({3: 0}, 42, 3) + map_bits({3: 0, 10: 1}, 42, 3)
    control = 16 * _coded(100, [1], 2, 0)
    assert r == {"verdict": "supported", "bits": arm, "control_bits": control, "ratio": arm / control,
                 "lower_bound": False, "library_maps": 2}


def test_the_verdict_is_judged_on_the_ratio():
    control = 16 * _coded(99, [1], 2, 0)
    assert control == 1600
    fixed = 16 * _coded(0, [1], 17, 16) + map_bits({3: 0}, 42, 3)

    def sizes(total):
        return [total - fixed - 15 * ((total - fixed) // 16)] + [(total - fixed) // 16] * 15

    for total, verdict in ((1440, "supported"), (1441, "inconclusive"), (1599, "inconclusive"), (1600, "refuted")):
        r = g1_size(_arm(sizes(total)), _control((99,) * 16))
        assert (r["bits"], r["verdict"]) == (total, verdict)


def test_the_empty_map_and_a_build_s_own_maps_are_not_charged_as_library():
    arm = _arm([50] * 16, leaves=[0] * 16)
    r = g1_size(arm, _control())
    assert r["library_maps"] == 0 and r["bits"] == 16 * _coded(50, [0], 17, 16)
    arm = _arm([50] * 16)
    own = arm["roots"][0]
    own["maps"].append({"5": 0})
    own["map_empty"].append(42)
    own["nodes"] = [[BOARD, 0, {"leaf": 17}]]
    r = g1_size(arm, _control())
    assert r["library_maps"] == 1
    assert r["bits"] == 15 * _coded(50, [1], 17, 16) + _coded(50, [17], 18, 16) + map_bits({3: 0}, 42, 3)


def test_an_unfinished_root_counts_without_its_references_and_can_only_refute():
    r = g1_size(_arm([50] * 16, incomplete=(207,)), _control())
    arm = 15 * _coded(50, [1], 17, 16) + 50 - code_bits("fixed", [1], 17, 16) + map_bits({3: 0}, 42, 3)
    assert r["bits"] == arm and r["lower_bound"] and r["verdict"] == "inconclusive"
    assert g1_size(_arm([200] * 16, incomplete=(207,)), _control())["verdict"] == "refuted"


def test_a_build_that_did_not_start_from_the_control_s_maps_is_not_judged():
    assert g1_size(_arm([50] * 16, library=[OWN[(k + 1) % 2] for k in range(16)]), _control())["verdict"] == "not_run"
    arm = _arm([50] * 16)
    arm["roots"][3]["map_empty"][2] = 41
    assert g1_size(arm, _control())["verdict"] == "not_run"
    arm = _arm([50] * 16)
    arm["roots"][3]["shared"] = 15
    assert g1_size(arm, _control())["verdict"] == "not_run"


def test_a_run_off_either_registration_is_not_judged():
    assert g1_size(_arm([50] * 16), _e(CONTROL, _control()["roots"], builder=SPEC["builder"]))["verdict"] == "not_run"
    assert g1_size(_e(SPEC, _arm([50] * 16)["roots"], builder=CONTROL["builder"]), _control())["verdict"] == "not_run"
    assert g1_size(_e(SPEC, _arm([50] * 16)["roots"], measurement_fingerprint=L5["measurement_fp"]),
                   _control())["verdict"] == "not_run"


def test_a_control_with_an_unfinished_root_is_not_judged():
    control = _control()
    control["roots"][0]["complete"] = False
    assert g1_size(_arm([50] * 16), control)["verdict"] == "not_run"
