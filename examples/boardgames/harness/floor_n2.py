"""§3.6 N2 — a learned default in the build (user's choice). N1 (h317) trained a 19,996-bit net that keeps the win at
4,461 of D1's 4,523 positions — 62 misses against tactics' 200 — on positions from strategies for every root but
L4's 32; D1 warned that fewer misses need not make a smaller strategy. N2 builds L4's 32 positions — the ones the
net never saw — with L5's rule and pipeline, no library, a deciding level holding several safe moves playing the one
N1's quantized net scores highest (harness.steady_state.Facts order "net", harness.integer_net; the builder's "net"
names N1's evidence), and nothing else changed. Judged on completion as W6 and on its total against L6 — the same
positions with ties broken by tactics, the best order adding no bits (harness.floor_l6, already built, under its own
registration): supported at nine tenths of it or less, refuted at L6's or more. The net's own bits (`net_bits`), paid
once for a whole game, are not in the 32 positions' total."""
from __future__ import annotations

from harness.floor_l4 import l4_size
from harness.floor_l5 import SPEC as L5
from harness.floor_l6 import SPEC as L6
from harness.floor_w6 import w6_report

SPEC = {**L5, "measurement_fp": "af5ea127269c", "builder": {**L5["builder"], "order": "net", "net": "c49_n1.json.gz"},
        "support": 0.9, "net_bits": 19_996}

n2_report = w6_report


def n2_size(e: dict, control: dict, spec: dict) -> dict:
    return l4_size(e, control, spec, L6)
