"""§3.6 T2 — the maps of T1's file shrunk to what its leaves play. Every map was found by a search that keeps whatever
levels it lands on, and `simplify` then drops only the levelled cells the map can lose one at a time: the levels left
are spread evenly over all 8 (T1's file pays 2.92 bits a cell for them, h382). Under T1's rule a map matters only by the
order it puts the safe moves' cells in where the rule has a choice (harness.map_orders). T2 reads T1's file back
(`source`), walks every line each leaf plays and collects its map's orders (`t2_orders`), gives each map the lowest
levels that keep them, merges the maps whose orders agree (harness.map_orders.merged) and writes the game again with
harness.strategy_file. Every move the game makes is the move T1's file makes. The decoded file must equal what was
written and win every line when played against every reply. Judged on that and on the file's length against T1's
130,005 bits: supported at 19/20 of it or less, refuted at 99/100 or more. On G3's game, the dry run, the same steps
took T1's 225,678 bits to 204,454."""
from __future__ import annotations

from harness.floor_t1 import SPEC as T1

SPEC = {"rule": T1["rule"], "control_bits": 130_005, "support": (19, 20), "refute": (99, 100),
        "source": T1["file"], "file": "c49_t2.json.gz"}


def t2_orders(walks) -> dict:
    """{map id: (strict, weak)} from (map id, strict, weak) for each leaf walked, the pairs of every leaf on a map
    together."""
    orders: dict = {}
    for i, strict, weak in walks:
        both = orders.setdefault(i, (set(), set()))
        both[0].update(map(tuple, strict))
        both[1].update(map(tuple, weak))
    return orders


def t2_report(e: dict, read: dict, source: dict, spec: dict) -> dict:
    """`read` is what a decoder made of the file now: its bits, its differences from the game, its leaves; `source` is
    T1's evidence, the file this one was made from."""
    if (e.get("spec") != "harness.floor_t2" or e.get("source_sha256") != source.get("sha256")
            or source.get("bits") != spec["control_bits"]):
        return {"decoded": "not_run", "size": "not_run"}
    replay = e["replay"]
    if read["bits"] != e["bits"] or read["differences"] or not replay["won"] or replay["leaves"] != read["leaves"]:
        return {"decoded": "refuted", "size": "not_run"}
    num, den = spec["support"]
    rnum, rden = spec["refute"]
    control, bits = spec["control_bits"], e["bits"]
    return {"decoded": "supported", "size": ("supported" if den * bits <= num * control else
                                             "refuted" if rden * bits >= rnum * control else "inconclusive")}
