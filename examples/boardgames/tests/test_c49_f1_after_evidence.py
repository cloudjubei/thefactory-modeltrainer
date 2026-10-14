"""§3.6 F1 — findings registered AFTER the data, from evidence/c49_f1.json.gz (every one of the 671 frontier positions
built and certified; 548,424 bits with the opening and the seed library, twice W9's projection of 271,740 and above its
95% high of 476,290). The miss is sampling in a heavy tail, not drift: the 32 positions W9 projected from are light in
F1 too (10,932 bits, a mean of 342 against the frontier's 803 — a random 32 is that light 273 times in 20,000); the
heaviest 67 positions hold 311,823 bits (58%) while the median position takes 154; and the two halves of the seeded
order cost the same (269,927 and 268,804). The library carried little: 880 of 8,904 leaves reference it, and the
positions it was found on (L2's 32) cost 8,432 bits in F1 against 14,666 in L2. As a method, a 32-position bootstrap's
95% high falls below this frontier's true mean 113 times in 1,000, not 25: on a heavy-tailed frontier a projection
from 32 positions is a guess, not a bound."""
from __future__ import annotations

import random
import statistics
from functools import lru_cache
from pathlib import Path

import pytest

from harness.evidence import load_evidence
from harness.floor_f1 import SPEC

EVIDENCE = Path(__file__).resolve().parent.parent / "evidence"
FILE = "c49_f1.json.gz"
FILES = (FILE, "c49_l2.json.gz")
pytestmark = pytest.mark.skipif(
    not all((EVIDENCE / f).exists() for f in FILES),
    reason="evidence/ is gitignored and these files are not on this machine — restore them to run the proofs")


@lru_cache(maxsize=1)
def _roots() -> dict:
    return {r["index"]: r for r in load_evidence(EVIDENCE / FILE)["roots"]}


def _bits_in_seeded_order() -> list[int]:
    roots = _roots()
    return [roots[i]["bits"]["nodes"] for i in SPEC["indices"]]


def test_c49_f1_the_projection_missed_by_sampling_a_heavy_tail_not_by_drift():
    bits = _bits_in_seeded_order()
    assert len(bits) == 671 and sum(bits) == 538_731
    w9 = sum(bits[40:72])
    assert w9 == 10_932
    rng = random.Random(0)
    assert sum(1 for _ in range(20_000) if sum(rng.sample(bits, 32)) <= w9) == 273
    assert sum(sorted(bits, reverse=True)[:67]) == 311_823 and statistics.median(bits) == 154
    assert (sum(bits[:335]), sum(bits[335:])) == (269_927, 268_804)


def test_c49_f1_the_library_carried_little_beyond_the_positions_it_was_found_on():
    roots = _roots()
    assert sum(r["library_leaves"] for r in roots.values()) == 880
    assert sum(r["bits"]["leaves"] for r in roots.values()) == 8_904
    l2 = [r for r in load_evidence(EVIDENCE / "c49_l2.json.gz")["roots"] if r["complete"]]
    assert sorted(r["index"] for r in l2) == sorted(SPEC["indices"][8:40])
    assert sum(r["bits"]["nodes"] for r in l2) == 14_666
    assert sum(roots[r["index"]]["bits"]["nodes"] for r in l2) == 8_432


def test_c49_f1_a_32_position_bootstrap_high_is_not_a_bound_on_this_frontier():
    bits = _bits_in_seeded_order()
    true = sum(bits) / len(bits)
    rng = random.Random(0)

    def high(sample: list[int]) -> float:
        return sorted(sum(rng.choices(sample, k=32)) / 32 for _ in range(400))[390]

    assert sum(1 for _ in range(1_000) if high(rng.sample(bits, 32)) < true) == 113
