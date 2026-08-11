import json

import pytest

from harness.config import TrainerConfig, load_config, validate_config


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
