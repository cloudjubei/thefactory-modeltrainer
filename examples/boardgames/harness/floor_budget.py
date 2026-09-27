"""§C.49 T7 — is the ~1K standardised-input setups' T6 failure the OPTIMISATION BUDGET? Pre-registered.

T6 (harness/floor_small.py, h58-h60) left the ~1K standardised-input nets 0/10 raw-perfect, with every failure on a
TRAINED position (not coverage). They train on ~625 rows (one per symmetry class) where the augmented raw net trains
on ~11,500, so the same epochs give them ~20x fewer gradient steps, and their settle starts at lr 1e-3 where the
oracle fits used 2e-3-5e-3. T7 runs T6's process at the standardised setups with the settle matched to the oracle
fit's budget: 3,000 epochs starting at lr 5e-3 and decaying to 1e-5 (~24K steps). Still solver-free: the settle trains
only on the process's own final labels.

  canon_mlp32_budget  MLP 32 (938 params) — T6's h59 point, for the direct comparison;
  canon_mlp48_budget  MLP 48 (1,402 params) — a width inside the oracle frontier's robust range;
  canon_conv6_budget  conv 6 (994 params) — the frontier point that held under every recipe.

Judged by harness.arm_judge.report with T5's bars: SUPPORTED at >= 8/10 raw-perfect after the settle, REFUTED at
<= 5, INCONCLUSIVE between."""
from __future__ import annotations

from harness.arm_judge import report
from harness.floor_coverage import T5_ARMS

_BASE = {**T5_ARMS["augment_sib2"], "augment": False, "settle_epochs": 3000, "settle_lr": 5e-3}
_ARCHES = {"canon_mlp32_budget": ({"mlp_hidden": [32], "canonical_input": True}, 938),
           "canon_mlp48_budget": ({"mlp_hidden": [48], "canonical_input": True}, 1402),
           "canon_conv6_budget": ({"channels": 6, "canonical_input": True}, 994)}
SPEC = {"arms": {name: {**_BASE, "arch": arch, "params": params} for name, (arch, params) in _ARCHES.items()},
        "params": {name: params for name, (_arch, params) in _ARCHES.items()},
        "seeds": tuple(range(331, 341)), "era": "1458537ff834", "measurement_fp": "ead16a9bdaab",
        "iterations": _BASE["iterations"], "positions": 4520, "support_at": 8, "refute_at": 5}


def budget_report(arms: dict) -> dict:
    return report(arms, SPEC)
