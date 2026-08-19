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
