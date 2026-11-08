"""§3.6 T3 — T2's game written again with each map's levels coded in context. After T2 the levels are the largest part
of the file, 46,096 of its 118,197 bits (h385), and harness.strategy_file codes each by its cell alone. T3 reads T2's
file back (`source`) and writes the same game with harness.level_context, which codes a level under the highest level
read so far in its map, its row and whether the cell below holds a level. Nothing but the levels' code changes. The
decoded file must equal T2's game and win every line when played against every reply. Judged on that and on the
file's length against T2's 118,197 bits: supported at 99/100 of it or less, refuted at 999/1000 or more. On G3's
game, the dry run, the same context took the levels of T2's G3 file from 76,981 bits to 72,644 and the file from
204,454 to 200,115."""
from __future__ import annotations

from harness.floor_t2 import SPEC as T2

SPEC = {"rule": T2["rule"], "control_bits": 118_197, "support": (99, 100), "refute": (999, 1000),
        "source": T2["file"], "file": "c49_t3.json.gz"}


def t3_report(e: dict, read: dict, source: dict, spec: dict) -> dict:
    """`read` is what a decoder made of the file now: its bits, its differences from T2's game, its leaves; `source`
    is T2's evidence, the file this one was made from."""
    if (e.get("spec") != "harness.floor_t3" or e.get("source_sha256") != source.get("sha256")
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
