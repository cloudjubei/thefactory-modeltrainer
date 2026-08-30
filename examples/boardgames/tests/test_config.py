import json

import pytest

from harness.config import DEFAULT_AZ_SOLVE_ENDGAME, TrainerConfig, load_config, validate_config


def test_trained_net_defaults_to_the_exact_endgame_cutoff():
    # A DEPLOYED/eval net must play the endgame exactly (provably perfect + cheap), not approximate it with the
    # value head — so the default is ON. (Self-play exploration keeps it off via the AlphaZeroAgent class default.)
    assert TrainerConfig().az_solve_endgame == DEFAULT_AZ_SOLVE_ENDGAME
    assert DEFAULT_AZ_SOLVE_ENDGAME > 0


def _write(tmp_path, obj):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(obj))
    return p


def test_load_config_ignores_engine_injected_keys(tmp_path):
    p = _write(tmp_path, {
        "game": "connect4", "model_name": "mcts", "mcts_sims": 80, "opponent": "random",
        "eval_games": 10, "seed": 1, "device": "cpu", "keep_checkpoint": True,
    })
    cfg = load_config(p)
    assert cfg.model_name == "mcts" and cfg.mcts_sims == 80 and cfg.seed == 1


def test_load_config_coerces_numeric_strings(tmp_path):
    p = _write(tmp_path, {"model_name": "mcts", "mcts_sims": "120", "eval_games": "20", "seed": "2"})
    cfg = load_config(p)
    assert cfg.mcts_sims == 120 and cfg.eval_games == 20 and cfg.seed == 2


def test_validate_rejects_bad_choice_and_out_of_range():
    with pytest.raises(ValueError):
        validate_config(TrainerConfig(model_name="nope"))
    with pytest.raises(ValueError):
        validate_config(TrainerConfig(opponent="ghost"))
    with pytest.raises(ValueError):
        validate_config(TrainerConfig(mcts_sims=0))
    with pytest.raises(ValueError):
        validate_config(TrainerConfig(eval_games=1))
    with pytest.raises(ValueError):
        validate_config(TrainerConfig(mcts_solve_endgame=-1))
    with pytest.raises(ValueError):
        validate_config(TrainerConfig(mcts_solve_endgame=99))


def test_mcts_solve_endgame_defaults_off_and_coerces(tmp_path):
    assert TrainerConfig().mcts_solve_endgame == 0  # pure mcts by default
    cfg = load_config(_write(tmp_path, {"model_name": "mcts", "mcts_solve_endgame": "12"}))
    assert cfg.mcts_solve_endgame == 12


def test_load_config_accepts_alphazero_and_coerces_az_levers(tmp_path):
    p = _write(tmp_path, {
        "model_name": "alphazero", "az_iterations": "2", "az_selfplay_games": "8",
        "az_sims": "30", "az_epochs": "3", "az_warm_start": "0",
    })
    cfg = load_config(p)
    assert cfg.model_name == "alphazero"
    assert cfg.az_iterations == 2 and cfg.az_selfplay_games == 8 and cfg.az_sims == 30 and cfg.az_epochs == 3
    assert cfg.az_warm_start == 0  # coerced from the string "0"


def test_az_warm_start_defaults_on_and_rejects_bad_value():
    assert TrainerConfig(model_name="alphazero").az_warm_start == 1
    with pytest.raises(ValueError):
        validate_config(TrainerConfig(model_name="alphazero", az_warm_start=2))


def test_validate_rejects_out_of_range_az_levers():
    with pytest.raises(ValueError):
        validate_config(TrainerConfig(model_name="alphazero", az_sims=0))
    with pytest.raises(ValueError):
        validate_config(TrainerConfig(model_name="alphazero", az_iterations=0))
    with pytest.raises(ValueError):
        validate_config(TrainerConfig(model_name="alphazero", az_distill_games=-1))


def test_az_distill_games_defaults_off_and_coerces(tmp_path):
    assert TrainerConfig().az_distill_games == 0
    cfg = load_config(_write(tmp_path, {"model_name": "alphazero", "az_distill_games": "80"}))
    assert cfg.az_distill_games == 80


def test_az_endgame_defaults_keep_the_loop_off():
    # The online endgame-tablebase-from-play loop is OPT-IN — defaults must leave it fully OFF so a run is
    # byte-identical to pure #1 self-play (the honest baseline). Only the accuracy knobs carry non-zero defaults.
    c = TrainerConfig(model_name="alphazero")
    assert c.az_endgame_tablebase == 0  # master switch OFF
    assert c.az_endgame_max_empty == 14  # matches the estimate_solve_endgame default (ms-cheap solves)
    assert c.az_endgame_exact_targets == 1  # inert while the store is empty; only bites once the switch is on
    assert c.az_endgame_extend_positions == 2000
    assert c.az_endgame_extend_seconds == 5.0
    assert c.az_endgame_cap == 200000  # conservative RAM bound given the OOM history
    assert c.az_endgame_warm_start == 0
    assert c.az_endgame_persist == 0


def test_az_endgame_knobs_coerce_from_strings(tmp_path):
    cfg = load_config(_write(tmp_path, {
        "model_name": "alphazero", "az_endgame_tablebase": "1", "az_endgame_max_empty": "16",
        "az_endgame_exact_targets": "0", "az_endgame_extend_positions": "500",
        "az_endgame_extend_seconds": "8", "az_endgame_cap": "50000",
        "az_endgame_warm_start": "1", "az_endgame_persist": "1",
    }))
    assert cfg.az_endgame_tablebase == 1 and cfg.az_endgame_max_empty == 16
    assert cfg.az_endgame_exact_targets == 0 and cfg.az_endgame_extend_positions == 500
    assert cfg.az_endgame_extend_seconds == 8.0 and cfg.az_endgame_cap == 50000
    assert cfg.az_endgame_warm_start == 1 and cfg.az_endgame_persist == 1


def test_az_net_capacity_levers_default_to_legacy_and_coerce(tmp_path):
    c = TrainerConfig(model_name="alphazero")
    assert c.az_channels == 32 and c.az_blocks == 0 and c.az_residual == 0
    assert c.az_batchnorm == 0 and c.az_head_hidden == 0  # default = today's 20K-param legacy net
    cfg = load_config(_write(tmp_path, {
        "model_name": "alphazero", "az_channels": "128", "az_blocks": "6",
        "az_residual": "1", "az_batchnorm": "1", "az_head_hidden": "64",
    }))
    assert cfg.az_channels == 128 and cfg.az_blocks == 6 and cfg.az_residual == 1
    assert cfg.az_batchnorm == 1 and cfg.az_head_hidden == 64


def test_validate_rejects_out_of_range_capacity_levers():
    for bad in (
        TrainerConfig(model_name="alphazero", az_channels=4),
        TrainerConfig(model_name="alphazero", az_channels=9999),
        TrainerConfig(model_name="alphazero", az_blocks=-1),
        TrainerConfig(model_name="alphazero", az_blocks=99),
        TrainerConfig(model_name="alphazero", az_residual=2),
        TrainerConfig(model_name="alphazero", az_batchnorm=2),
        TrainerConfig(model_name="alphazero", az_head_hidden=-1),
        TrainerConfig(model_name="alphazero", az_head_hidden=99999),
    ):
        with pytest.raises(ValueError):
            validate_config(bad)


def test_validate_rejects_out_of_range_endgame_knobs():
    for bad in (
        TrainerConfig(model_name="alphazero", az_endgame_tablebase=2),
        TrainerConfig(model_name="alphazero", az_endgame_exact_targets=2),
        TrainerConfig(model_name="alphazero", az_endgame_warm_start=2),
        TrainerConfig(model_name="alphazero", az_endgame_persist=2),
        TrainerConfig(model_name="alphazero", az_endgame_max_empty=-1),
        TrainerConfig(model_name="alphazero", az_endgame_max_empty=99),
        TrainerConfig(model_name="alphazero", az_endgame_extend_positions=-1),
        TrainerConfig(model_name="alphazero", az_endgame_extend_positions=10**9),
        TrainerConfig(model_name="alphazero", az_endgame_extend_seconds=-1.0),
        TrainerConfig(model_name="alphazero", az_endgame_extend_seconds=999.0),
        TrainerConfig(model_name="alphazero", az_endgame_cap=100),  # below the floor
        TrainerConfig(model_name="alphazero", az_endgame_cap=10**12),
    ):
        with pytest.raises(ValueError):
            validate_config(bad)


def test_az_league_levers_and_solver_free_assertion(tmp_path):
    c = TrainerConfig(model_name="alphazero")
    assert c.az_league == 0 and c.az_league_p1_frac == 0.7 and c.az_league_snapshots == 4
    cfg = load_config(_write(tmp_path, {"model_name": "alphazero", "az_league": "1", "az_pool_frac": "0.4",
                                        "az_league_p1_frac": "0.8", "az_league_snapshots": "6",
                                        "az_league_anchor_frac": "0.3"}))
    assert cfg.az_league == 1 and cfg.az_league_p1_frac == 0.8 and cfg.az_league_snapshots == 6
    # league replaces oracle OPENING distillation (rejected), but the grow-as-you-go ENDGAME table is complementary (allowed)
    for bad in (
        TrainerConfig(model_name="alphazero", az_league=1, az_distill_games=80),
        TrainerConfig(model_name="alphazero", az_league=2),
        TrainerConfig(model_name="alphazero", az_league_p1_frac=1.5),
    ):
        with pytest.raises(ValueError):
            validate_config(bad)
    validate_config(TrainerConfig(model_name="alphazero", az_league=1, az_endgame_tablebase=1,
                                  az_distill_positions=0))  # league + grow-as-you-go endgame table = OK
