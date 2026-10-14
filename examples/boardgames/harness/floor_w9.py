"""§3.6 W9 — L2's finding on fresh positions. With the first level holding any safe move deciding, W6's 32 positions
took a fifth of the exact rule's bits and the whole game projected under the trained net (h260) — but those positions
had been built five times. W9 builds the next 32 of W1's seeded frontier order, never built before, with L2's rule
and pipeline exactly. Judged on completion as W6, and on whether the whole game projects below the net's ~660K
bits: supported when the whole 95% bootstrap interval of W6's projection is below it and every root finished;
refuted when the whole interval is above (a partial size is a lower bound, so above can be read while unfinished)."""
from __future__ import annotations

from harness.floor_l2 import SPEC as L2
from harness.floor_w6 import w6_projection, w6_report

SPEC = {**{k: v for k, v in L2.items() if k not in ("eight_level_bits", "support")},
        "indices": [615, 180, 504, 384, 135, 301, 669, 308, 49, 120, 638, 70, 564, 499, 631, 327, 236, 528, 104, 576,
                    54, 19, 375, 536, 379, 46, 614, 572, 112, 426, 353, 208]}

w9_report = w6_report


def w9_below(e: dict, spec: dict) -> dict:
    r = w6_projection(e, spec)
    return {**r, "verdict": {"supported": "refuted", "refuted": "supported"}.get(r["verdict"], r["verdict"])}
