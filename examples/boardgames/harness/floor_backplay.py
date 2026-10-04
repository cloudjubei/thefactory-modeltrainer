"""§C.49 T15 — the BACKWARD CURRICULUM: does it do no harm where the process already works? Pre-registered.

T14 (h111-h113) showed that a value target the net computes cannot add information it lacks: it has to come from real
game outcomes. The backward curriculum (the user's idea, 2026-10-02; Backplay, reverse curriculum) starts a share of
each iteration's self-play games near the end of the previous iteration's games, where the real result is a few moves
away, and moves the start back to whole games over `ramp` iterations (`train_alphazero(backplay=...)`). Before trying
it on Connect-4 it must not break tic-tac-toe, where T9's process plays perfectly from the start on 10/10 seeds (h73).

T15 is T9's process with half the games started late, reaching whole games at iteration 15 of 30, judged exactly as
T9 (harness.floor_tree.tree_report): P-START SUPPORTED at >= `support_at` certified seeds, REFUTED at <= `refute_at`;
the stop reading is reported beside it."""
from __future__ import annotations

from harness.floor_tree import _CONFIG as T9_CONFIG, SPEC as T9_SPEC

T15_ARM = "backplay"
_CONFIG = {**T9_CONFIG, "backplay": {"frac": 0.5, "ramp": 15}}
SPEC = {**T9_SPEC, "arms": {T15_ARM: _CONFIG}, "params": {T15_ARM: _CONFIG["params"]}, "seeds": tuple(range(421, 431)),
        "era": "9ea894591459", "measurement_fp": "3758500de40d"}
