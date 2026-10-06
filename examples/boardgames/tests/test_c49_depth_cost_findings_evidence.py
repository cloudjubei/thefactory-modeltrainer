"""§C.49 net vs table by depth — findings registered AFTER the data, from evidence/c49_depth_cost.json.gz (the run
judged by h142 and h143, both inconclusive). Read with harness.depth_cost's accounting: a complete table is one 3-bit
move per first-player position in walk order, a hybrid's exceptions the cheaper of a sparse index or a position mask."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from harness.depth_cost import SPEC, depth_report, exception_bits, walk_table_bits
from harness.evidence import load_evidence

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_depth_cost.json.gz"
FILES = (FILE,)
pytestmark = pytest.mark.skipif(
    not (EVIDENCE / FILE).exists(),
    reason="evidence/ is gitignored and this file is not on this machine — restore it to run the proofs")


@lru_cache(maxsize=1)
def _evidence() -> dict:
    return load_evidence(EVIDENCE / FILE)


def _rows(depth):
    return [r for r in depth_report(_evidence(), SPEC)["rows"] if r["depth"] == depth]


def _pooled_share(ply):
    nets = [n for n in _evidence()["nets"] if str(ply) in n["positions_by_ply"]]
    return (sum(n["exceptions_by_ply"].get(str(ply), 0) for n in nets)
            / sum(n["positions_by_ply"][str(ply)] for n in nets))


def test_c49_depth_break_even_roughly_quadruples_every_two_plies_as_the_table_does():
    assert [_rows(d)[0]["table_entries"] for d in (9, 11, 13)] == [1862, 8845, 36431]
    for depth, low, high in ((9, 0.17, 0.22), (11, 0.9, 1.05), (13, 3.7, 4.36)):
        values = [r["break_even"] for r in _rows(depth)]
        assert low <= min(values) and max(values) <= high, (depth, values)
    assert len(_rows(9)) == len(_rows(11)) == 7 and len(_rows(13)) == 3


def test_c49_depth_the_net_s_exception_share_falls_about_a_point_a_ply_then_holds_near_9_percent():
    shares = [_pooled_share(p) for p in (6, 8, 10, 12)]
    assert shares[0] > shares[1] > shares[2] and abs(shares[3] - shares[2]) < 0.002
    assert 0.105 < shares[0] < 0.11 and 0.09 < shares[2] < 0.092


def test_c49_depth_the_canonical_table_is_not_the_smallest_a_net_s_own_tree_is_up_to_57_percent_smaller():
    deep = [n for n in _evidence()["nets"] if n["horizon"] == 13]
    canonical = sum(_evidence()["table"]["by_ply"].values())
    sizes = sorted(sum(n["positions_by_ply"].values()) for n in deep)
    assert canonical == 36431 and sizes == [15782, 19269, 24707]
    smallest = min(deep, key=lambda n: sum(n["positions_by_ply"].values()))
    positions = sum(smallest["positions_by_ply"].values())
    exceptions = sum(smallest["exceptions_by_ply"].values())
    own_table = walk_table_bits(positions, SPEC["actions"])
    assert own_table < walk_table_bits(canonical, SPEC["actions"]) / 2
    break_even = (own_table - exception_bits(exceptions, positions, SPEC["actions"])) / smallest["params"]
    assert 1.3 < break_even < 1.4
