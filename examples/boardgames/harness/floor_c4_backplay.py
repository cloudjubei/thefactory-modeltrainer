"""§C.49 T16 — the backward curriculum on Connect-4: a paired pilot. Pre-registered.

T14's pilot shape (T10's process at a 6-ply tree, 20 iterations, no stop, then the settle), with two arms on fresh
seeds — base, and half the self-play games started near the end of the previous iteration's games, reaching whole
games at iteration 10 — judged by harness.floor_c4_value_signal.pilot_report: the curriculum WINS a seed when its net
plays the optimal move at a strictly higher share of its own first-player positions through ply 4; SUPPORTED with at
least `support_wins` wins AND a pooled share at least `min_gain` higher, REFUTED with at most `refute_wins` wins or
no pooled gain, INCONCLUSIVE between. Base is retrained under this code, not reused from T14."""
from __future__ import annotations

from harness.floor_c4_value_signal import SPEC as T14_SPEC

_BASE = T14_SPEC["arms"]["base"]
SPEC = {**T14_SPEC, "arms": {"base": _BASE, "backplay": {**_BASE, "backplay": {"frac": 0.5, "ramp": 10}}},
        "treatments": ("backplay",), "seeds": (411, 412, 413, 414), "era": "7b7e2b460262",
        "measurement_fp": "c0d85a95888e"}
