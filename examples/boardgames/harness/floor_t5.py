"""§3.6 T5 — T4's game written again with each exception flag coded in context too. After T4 the exception flags
are the third part of the file, 17,884 of its 112,644 bits (h391), all coded under one context. T5 reads T4's file
back (`source`) and writes the same game with harness.exception_context, which codes each flag under how many
exceptions the leaf's walk has read so far and how many flags since the last one. Nothing but the flags' code changes.
The decoded file must equal T4's game and win every line when played against every reply. Judged on that and on the
file's length against T4's 112,644 bits: supported at 99/100 of it or less, refuted at 999/1000 or more. On G3's
game, the dry run, the same context took the exception flags of G3's file from 33,698 bits to 27,285 and the file
from 194,950 to 188,537."""
from __future__ import annotations

from harness.floor_t4 import SPEC as T4

SPEC = {"rule": T4["rule"], "control_bits": 112_644, "support": (99, 100), "refute": (999, 1000),
        "source": T4["file"], "file": "c49_t5.json.gz"}


def t5_report(e: dict, read: dict, source: dict, spec: dict) -> dict:
    """`read` is what a decoder made of the file now: its bits, its differences from T4's game, its leaves; `source`
    is T4's evidence, the file this one was made from."""
    if (e.get("spec") != "harness.floor_t5" or e.get("source_sha256") != source.get("sha256")
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
