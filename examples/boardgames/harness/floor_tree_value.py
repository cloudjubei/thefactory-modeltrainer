"""§C.49 T13 — VALUE TARGETS for the strategy-tree positions: does the change do no harm where the process already
works? Pre-registered.

D3/D4 (h106-h109) found that in the Connect-4 opening the training target itself is wrong: the search reads the
value head, and the value head is trained only on outcomes of 32-sim self-play games — the strategy-tree positions,
the ones training cares about most, get no value target at all. T13 gives them one: the root value of the same
200-sim relabel search that labels their policy (`train_alphazero(tree_value_target=True)`). Before trying it on
Connect-4 it must not break tic-tac-toe, where T9's process plays perfectly from the start on 10/10 seeds (h73).

T13 is T9's process with the option on, judged exactly as T9 (harness.floor_tree.tree_report): P-START SUPPORTED
at >= `support_at` certified seeds, REFUTED at <= `refute_at`; the stop reading is reported beside it."""
from __future__ import annotations

from harness.floor_tree import _CONFIG as T9_CONFIG, SPEC as T9_SPEC

T13_ARM = "tree_value_target"
_CONFIG = {**T9_CONFIG, "tree_value_target": True}
SPEC = {**T9_SPEC, "arms": {T13_ARM: _CONFIG}, "params": {T13_ARM: _CONFIG["params"]}, "seeds": tuple(range(391, 401)),
        "era": "607d6dccb102", "measurement_fp": "851a36746117"}
