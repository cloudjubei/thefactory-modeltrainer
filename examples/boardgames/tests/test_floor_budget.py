"""Direct tests for harness/floor_budget.py — the §C.49 T7 registration: T6's process at the standardised-input
setups with the settle matched to the oracle fit's budget. The verdict logic is harness/arm_judge.py's (tested there);
these pin the registered arms."""
from __future__ import annotations

from harness.floor_budget import SPEC


def test_the_arms_are_t6_s_process_with_only_the_settle_budget_changed():
    from harness.floor_small import T6_ARMS

    t6 = T6_ARMS["canon_mlp32_long"]
    for name, cfg in SPEC["arms"].items():
        differs = {k for k in set(cfg) | set(t6) if cfg.get(k) != t6.get(k)}
        assert differs <= {"arch", "params", "settle_epochs", "settle_lr"}
        assert cfg["settle_epochs"] == 3000 and cfg["settle_lr"] == 5e-3 and not cfg["augment"]
        assert cfg["arch"]["canonical_input"] is True
    assert SPEC["arms"]["canon_mlp32_budget"]["arch"] == t6["arch"]
    assert not set(SPEC["seeds"]) & set(range(301, 331)) and len(SPEC["seeds"]) == 10


def test_the_registered_arches_build_nets_of_the_registered_sizes():
    from games.tictactoe import TicTacToe
    from harness.neural import Connect4Net, arch_for_game

    for name, cfg in SPEC["arms"].items():
        net = Connect4Net(**arch_for_game(cfg["arch"], TicTacToe()))
        assert sum(p.numel() for p in net.parameters()) == SPEC["params"][name] == cfg["params"]
