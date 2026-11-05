"""Direct tests for harness/floor_n2.py — N2, L5's rule breaking a deciding level's ties by N1's net, on L4's 32
positions (held out of the net's training), judged on completion as W6 and on its total against L6 (tactics, already
built, under its own registration)."""
from __future__ import annotations

from harness.floor_l4 import SPEC as L4
from harness.floor_l5 import SPEC as L5
from harness.floor_l6 import SPEC as L6
from harness.floor_n1 import SPEC as N1
from harness.floor_n2 import SPEC, n2_report, n2_size
from harness.floor_w6 import w6_report


def _e(spec, sizes, incomplete=(), **over):
    roots = [{"index": i, "complete": i not in incomplete, "bits": {"nodes": s}, "build_seconds": 600.0,
              "checked": True, "moves_win": True, "leaves_certified": True} for i, s in zip(spec["indices"], sizes)]
    return {"measurement_fingerprint": spec["measurement_fp"], "positions_fingerprint": spec["positions_fp"],
            "builder": spec["builder"], "opening": {"moves": 204, "trivial": 259, "frontier": 671}, "roots": roots,
            **over}


def test_n2_is_l5_breaking_ties_by_n1_s_net_on_positions_the_net_never_saw():
    assert SPEC["indices"] == L4["indices"] == N1["train"]["skip"] and SPEC["roots"] == 32
    assert SPEC["builder"] == {**L5["builder"], "order": "net", "net": "c49_n1.json.gz"}
    assert SPEC["search"] == L5["search"] and "library_from" not in SPEC
    assert SPEC["measurement_fp"] == "af5ea127269c" and SPEC["support"] == 0.9 and SPEC["net_bits"] == 19_996
    assert (SPEC["complete_support"], SPEC["complete_refute"]) == (28, 24) and n2_report is w6_report


def test_n2_is_judged_against_l6_under_l6_s_own_registration():
    control = _e(L6, [100] + [0] * 31)
    assert n2_size(_e(SPEC, [90] + [0] * 31), control, SPEC)["verdict"] == "supported"
    assert n2_size(_e(SPEC, [91] + [0] * 31), control, SPEC)["verdict"] == "inconclusive"
    assert n2_size(_e(SPEC, [100] + [0] * 31), control, SPEC)["verdict"] == "refuted"
    assert n2_size(_e(SPEC, [90] + [0] * 31), _e(L6, [100] + [0] * 31, builder=L5["builder"]),
                   SPEC)["verdict"] == "not_run"
    assert n2_size(_e(SPEC, [90] + [0] * 31, builder=L6["builder"]), control, SPEC)["verdict"] == "not_run"
    assert n2_size(_e(SPEC, [90] + [0] * 31), _e(L5, [100] + [0] * 31), SPEC)["verdict"] == "not_run"
