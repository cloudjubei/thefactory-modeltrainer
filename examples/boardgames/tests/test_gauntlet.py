import random

from games.connect4 import Connect4
from harness.agents import HeuristicAgent, RandomAgent
from harness.gauntlet import _model_factory, _rung_factory, climb_spine, run_gauntlet

SPINE = [
    {"id": "random", "kind": "random", "rating": 0},
    {"id": "heuristic", "kind": "heuristic", "rating": 400},
    {"id": "mcts60", "kind": "mcts", "sims": 60, "rating": 700},
]


def test_rung_factory_builds_the_near_perfect_oracle_rung():
    game = Connect4()
    factory = _rung_factory({"id": "oracle", "kind": "oracle", "depth": 8, "rating": 2000}, game)
    agent = factory()
    assert agent.kind == "oracle_depth"
    assert agent.act(game, game.initial_state(random.Random(0)), random.Random(0)) in range(7)


def test_the_oracle_rung_beats_a_random_model():
    game = Connect4()
    spine = [{"id": "oracle", "kind": "oracle", "depth": 6, "rating": 2000}]
    pairings = climb_spine(game, lambda: RandomAgent(), spine, n=4, base_seed=0, opening_plies=2, model_idx=0)
    assert pairings[0]["opponent"] == "oracle"
    assert pairings[0]["score"] < 0.5  # random cannot beat near-perfect play


def test_model_factory_constructs_a_fresh_agent_per_game():
    game = Connect4()
    mf = _model_factory({"model_name": "mcts", "mcts_sims": 30}, game)
    a, b = mf(), mf()
    assert a is not b  # fresh instance per game → no transposition-table bleed across the gauntlet's games


def test_climb_stops_after_the_first_rung_it_fails_to_clear():
    game = Connect4()
    pairings = climb_spine(game, lambda: HeuristicAgent(), SPINE, n=6, base_seed=0, opening_plies=4, model_idx=0)
    assert pairings[0]["opponent"] == "random"  # climbs the weakest rung first
    assert len(pairings) <= len(SPINE)
    # either it climbed the whole spine, or the last rung it played is the one it failed to clear (<0.55)
    assert len(pairings) == len(SPINE) or pairings[-1]["score"] < 0.55


def test_run_gauntlet_climbs_a_strong_model_further_than_a_weak_one():
    request = {
        "game": "connect4",
        "games_per_rung": 6,
        "base_seed": 1,
        "opening_plies": 4,
        "rungs": SPINE,
        "models": [
            {"id": "weak", "model_name": "random"},
            {"id": "strong", "model_name": "mcts", "mcts_sims": 120},
        ],
    }
    res = run_gauntlet(request)
    by = {r["model_id"]: r for r in res["ratings"]}
    assert "error" not in by["strong"] and "error" not in by["weak"]
    assert by["strong"]["rungs_played"] >= by["weak"]["rungs_played"]
    # every pairing carries the fixed anchor rating the TS rating math fits against
    assert all("opponentRating" in p and "score" in p and "games" in p for p in by["strong"]["pairings"])


def test_run_gauntlet_reports_a_bad_rung_as_a_per_model_error():
    request = {"game": "connect4", "rungs": [{"id": "x", "kind": "nope", "rating": 0}], "models": [{"id": "m", "model_name": "random"}]}
    res = run_gauntlet(request)
    assert "error" in res["ratings"][0]
