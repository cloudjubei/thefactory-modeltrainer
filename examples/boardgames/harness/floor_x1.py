"""§3.6 X1 — mirror sharing, priced on G3's certified 280,640-bit whole game (h324) with no rebuild. WeakC4 stores
one diagram per mirror pair (h334); our maps are cell-indexed, so a map and its mirror image are two maps. With a
symmetry of the game found generically (scripts/c4_mirror_sharing.py: a permutation of cells and moves under which the
rules are unchanged on sampled play), two codes are priced from G3's evidence:

  - maps — every map G3 stores (the library maps its leaves play, then each position's own) gets a flag bit, and one
    that is the mirror image of an earlier map is stored as that map's index instead of its levels;
  - roots — every frontier position gets a flag bit, and one whose mirror image is also a frontier position plays that
    position's strategy mirrored, at the partner's index; of each pair the cheaper is kept, and library maps only the
    dropped strategies played are no longer paid.

Each is judged on the share of G3's bits it saves: supported at a twentieth (maps) or a tenth (roots) or more, refuted
at a hundredth (maps) or a fiftieth (roots) or less; not judged off G3's registered bits or without the symmetry
checked."""
from __future__ import annotations

SPEC = {"g3_bits": 280_640, "maps_support": (1, 20), "maps_refute": (1, 100), "roots_support": (1, 10),
        "roots_refute": (1, 50)}


def _judged(e: dict, spec: dict, part: str) -> dict:
    if e["g3_bits"] != spec["g3_bits"] or not e["symmetry_checked"]:
        return {"verdict": "not_run"}
    price, g3 = e[part]["price"], e["g3_bits"]
    saved = g3 - price
    (sn, sd), (rn, rd) = spec[f"{part}_support"], spec[f"{part}_refute"]
    verdict = "supported" if sd * saved >= sn * g3 else "refuted" if rd * saved <= rn * g3 else "inconclusive"
    return {"verdict": verdict, "bits": price, "saved": saved, "share": saved / g3}


def x1_maps(e: dict, spec: dict) -> dict:
    return _judged(e, spec, "maps")


def x1_roots(e: dict, spec: dict) -> dict:
    return _judged(e, spec, "roots")
