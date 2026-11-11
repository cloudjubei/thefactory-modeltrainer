"""§3.6 T8 — T5's game written again by harness.position_walk, which reads two things from the position that a
wrapper of harness.strategy_file's coder never sees: a leaf names a map it reads again by its place among the known
maps ranked by how alike the leaf's board is to the boards each was read at, and each exception flag is coded under
the plies below its leaf too. T5's references take 15,775 of its 109,034 bits and its exception flags 14,273 (h394).
Nothing else in the file changes. The decoded file must equal T5's game and win every line when played against every
reply. Judged on that and on the file's length against T5's 109,034 bits: supported at 99/100 of it or less, refuted
at 999/1000 or more. On G3's game, the dry run, the same walk took G3's references from 28,842 bits to 24,779, its
exception flags from 27,285 to 26,306 and the file from 188,537 to 183,497."""
from __future__ import annotations

from harness.floor_t5 import SPEC as T5

SPEC = {"rule": T5["rule"], "control_bits": 109_034, "support": (99, 100), "refute": (999, 1000),
        "source": T5["file"], "file": "c49_t8.json.gz"}


def t8_report(e: dict, read: dict, source: dict, spec: dict) -> dict:
    """`read` is what a decoder made of the file now: its bits, its differences from T5's game, its leaves; `source`
    is T5's evidence, the file this one was made from."""
    if (e.get("spec") != "harness.floor_t8" or e.get("source_sha256") != source.get("sha256")
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
