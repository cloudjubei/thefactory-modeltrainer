"""§C.49 T12 — the solver-free Connect-4 run again, stopped by the VALUE-AWARE signal. Pre-registered.

T10 never stopped: on all six seeds that finished, the share rule kept reading 20-120 disagreements to the cap (h88),
59% of them between two optimal moves (h90). T12 is T10's process unchanged except that a raw move also agrees when
the search's Q for it is within `stop_value_delta` of the Q of the label's top move — the reading T11 recalibrated on
tic-tac-toe under a weakened search (h101). Judged exactly as T10 (harness.floor_c4.c4_report): a seed succeeds
when its run stopped on full agreement AND the solver certifies the net it stopped at through `certify_depth`
plies; SUPPORTED at >= `support_at`, REFUTED at <= `refute_at`, INCONCLUSIVE between. A stop on a net the solver
does not certify is a FALSE STOP, reported beside the verdict."""
from __future__ import annotations

from harness.floor_c4 import SPEC as T10_SPEC, T10_CONFIG

T12_CONFIG = {**T10_CONFIG, "stop_value_delta": 0.1}
SPEC = {**T10_SPEC, "config": T12_CONFIG, "seeds": tuple(range(381, 391)), "era": "72aa3c065c8c",
        "measurement_fp": "613b4f7dc021"}
