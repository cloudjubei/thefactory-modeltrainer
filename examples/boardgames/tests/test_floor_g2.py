"""Direct tests for harness/floor_g2.py — G2, the whole game built again with L5's rule from every map F2 found, all
within reach, judged on completion as F1 and on the whole game's size against F2's, both coded by first_use with each
library map a leaf plays paid once."""
from __future__ import annotations

from harness.floor_f1 import f1_complete
from harness.floor_f2 import SPEC as F2
from harness.floor_g1 import g1_library
from harness.floor_g2 import SPEC, g2_complete, g2_library, g2_size
from harness.floor_k1 import SPEC as K1
from harness.floor_l5 import SPEC as L5
from harness.reference_coding import code_bits
from harness.steady_state import map_bits

BOARD = [0] * 42
OWN = [{"3": 0}, {"3": 0, "10": 1}]
OPENING = 204 * 4 + 259


def _root(index, nodes, leaf, maps, shared, found=(), complete=True):
    return {"index": index, "complete": complete, "bits": {"nodes": nodes, "leaves": 1}, "board": BOARD,
            "to_move": 0, "nodes": [[BOARD, 0, {"leaf": leaf}]], "maps": [{}, *maps],
            "map_empty": [0] + [42] * len(maps), "shared": shared, "found_maps": [[m, 42] for m in found],
            "build_seconds": 600.0, "checked": True if complete else None,
            "moves_win": True if complete else None, "leaves_certified": True if complete else None}


def _e(spec, roots, **over):
    return {"measurement_fingerprint": spec["measurement_fp"], "positions_fingerprint": spec["positions_fp"],
            "builder": spec["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def _coded(nodes, refs, n_maps, shared):
    return nodes - code_bits("fixed", refs, n_maps, shared) + code_bits("first_use", refs, n_maps, shared)


def _f2(first=100):
    return _e(F2, [_root(i, first if k == 0 else 100, 1, [OWN[0]], 0, found=[OWN[k]] if k < 2 else [])
                   for k, i in enumerate(F2["indices"])])


CONTROL = OPENING + F2["library_bits"] + 671 * _coded(100, [1], 2, 0)
JUDGED = {**SPEC, "f2_bits": CONTROL}


def _arm(sizes, leaves=None, incomplete=(), library=None):
    library = library or OWN
    leaves = leaves or [1] * 671
    return _e(SPEC, [_root(i, s, leaf, library, 2, complete=i not in incomplete)
                     for i, s, leaf in zip(F2["indices"], sizes, leaves)])


def test_g2_is_f2_built_once_more_from_every_map_f2_found_all_in_reach():
    assert SPEC["indices"] == F2["indices"] and SPEC["roots"] == 671 and SPEC["library_from"] == "c49_f2.json.gz"
    assert "wave_size" not in SPEC and "library_bits" not in SPEC and "f1_bits" not in SPEC
    assert SPEC["builder"] == {**L5["builder"], "reuse_window": 100_000} and F2["builder"] == L5["builder"]
    assert SPEC["search"] == F2["search"] and SPEC["measurement_fp"] == K1["measurement_fp"]
    assert (SPEC["f2_bits"], SPEC["support"], SPEC["unfinished_refute"]) == (398_195, (9, 10), 7)
    assert g2_complete is f1_complete and g2_library is g1_library


def test_the_whole_game_is_the_opening_the_strategies_by_first_use_and_each_played_library_map_once():
    leaves = [1] * 300 + [2] * 371
    r = g2_size(_arm([50] * 671, leaves=leaves), _f2(), JUDGED)
    bits = (OPENING + 300 * _coded(50, [1], 3, 2) + 371 * _coded(50, [2], 3, 2)
            + map_bits({3: 0}, 42, 3) + map_bits({3: 0, 10: 1}, 42, 3))
    assert r == {"verdict": "supported", "bits": bits, "control_bits": CONTROL, "ratio": bits / CONTROL,
                 "lower_bound": False, "library_maps": 2}


def test_the_verdict_is_judged_against_f2_s_bits():
    first = 100 + (-CONTROL) % 10
    control = CONTROL + first - 100
    assert control % 10 == 0
    judged = {**SPEC, "f2_bits": control}
    fixed = OPENING + 671 * _coded(0, [1], 3, 2) + map_bits({3: 0}, 42, 3)

    def sizes(total):
        return [total - fixed - 670 * ((total - fixed) // 671)] + [(total - fixed) // 671] * 670

    nine_tenths = control * 9 // 10
    for total, verdict in ((nine_tenths, "supported"), (nine_tenths + 1, "inconclusive"),
                           (control - 1, "inconclusive"), (control, "refuted")):
        r = g2_size(_arm(sizes(total)), _f2(first), judged)
        assert (r["bits"], r["verdict"]) == (total, verdict)


def test_the_empty_map_and_a_build_s_own_maps_are_not_charged_as_library():
    r = g2_size(_arm([50] * 671, leaves=[0] * 671), _f2(), JUDGED)
    assert r["library_maps"] == 0 and r["bits"] == OPENING + 671 * _coded(50, [0], 3, 2)
    arm = _arm([50] * 671)
    own = arm["roots"][0]
    own["maps"].append({"5": 0})
    own["map_empty"].append(42)
    own["nodes"] = [[BOARD, 0, {"leaf": 3}]]
    r = g2_size(arm, _f2(), JUDGED)
    assert r["library_maps"] == 1
    assert r["bits"] == OPENING + 670 * _coded(50, [1], 3, 2) + _coded(50, [3], 4, 2) + map_bits({3: 0}, 42, 3)


def test_an_unfinished_root_counts_without_its_references_and_can_only_refute():
    first = F2["indices"][0]
    r = g2_size(_arm([50] * 671, incomplete=(first,)), _f2(), JUDGED)
    bits = OPENING + 670 * _coded(50, [1], 3, 2) + 50 - code_bits("fixed", [1], 3, 2) + map_bits({3: 0}, 42, 3)
    assert r["bits"] == bits and r["lower_bound"] and r["verdict"] == "inconclusive"
    assert g2_size(_arm([200] * 671, incomplete=(first,)), _f2(), JUDGED)["verdict"] == "refuted"


def test_a_build_that_did_not_start_from_f2_s_maps_is_not_judged():
    assert g2_size(_arm([50] * 671, library=OWN[::-1]), _f2(), JUDGED)["verdict"] == "not_run"
    arm = _arm([50] * 671)
    arm["roots"][3]["map_empty"][2] = 41
    assert g2_size(arm, _f2(), JUDGED)["verdict"] == "not_run"
    arm = _arm([50] * 671)
    arm["roots"][3]["shared"] = 1
    assert g2_size(arm, _f2(), JUDGED)["verdict"] == "not_run"


def test_a_run_off_either_registration_is_not_judged():
    arm = _arm([50] * 671)
    assert g2_size(arm, _e(F2, _f2()["roots"], builder=SPEC["builder"]), JUDGED)["verdict"] == "not_run"
    assert g2_size(_e(SPEC, arm["roots"], builder=F2["builder"]), _f2(), JUDGED)["verdict"] == "not_run"
    assert g2_size(_e(SPEC, arm["roots"][1:]), _f2(), JUDGED)["verdict"] == "not_run"
    assert g2_size(_e(SPEC, arm["roots"], measurement_fingerprint=F2["measurement_fp"]),
                   _f2(), JUDGED)["verdict"] == "not_run"


def test_f2_must_be_complete_and_come_to_its_registered_bits():
    f2 = _f2()
    f2["roots"][0]["complete"] = False
    unfinished = CONTROL - _coded(100, [1], 2, 0) + 100 - 1
    assert g2_size(_arm([50] * 671), f2, {**JUDGED, "f2_bits": unfinished})["verdict"] == "not_run"
    f2["roots"][0]["complete"] = True
    assert g2_size(_arm([50] * 671), f2, JUDGED)["verdict"] == "supported"
    assert g2_size(_arm([50] * 671), _f2(), {**JUDGED, "f2_bits": CONTROL + 1})["verdict"] == "not_run"
