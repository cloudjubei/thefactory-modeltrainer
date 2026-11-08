"""§3.6 T4 — T3's game written again with each map's cell flags coded in context too. After T3 the cell flags are the
second part of the file, 30,949 of its 115,599 bits (h388), coded by the cell and whether it is empty at the leaf
that first reads the map. T4 reads T3's file back (`source`) and writes the same game with harness.cell_context,
which also codes each flag under whether each of the two cells below is empty and holds a level. Nothing but the
flags' code changes. The decoded file must equal T3's game and win every line when played against every reply. Judged
on that and on the file's length against T3's 115,599 bits: supported at 99/100 of it or less, refuted at 999/1000 or
more. On G3's game, the dry run, the same context took the flags of G3's file from 51,994 bits to 46,828 and the file
from 200,115 to 194,950."""
from __future__ import annotations

from harness.floor_t3 import SPEC as T3

SPEC = {"rule": T3["rule"], "control_bits": 115_599, "support": (99, 100), "refute": (999, 1000),
        "source": T3["file"], "file": "c49_t4.json.gz"}


def t4_report(e: dict, read: dict, source: dict, spec: dict) -> dict:
    """`read` is what a decoder made of the file now: its bits, its differences from T3's game, its leaves; `source`
    is T3's evidence, the file this one was made from."""
    if (e.get("spec") != "harness.floor_t4" or e.get("source_sha256") != source.get("sha256")
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
