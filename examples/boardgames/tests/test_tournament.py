import random

from harness.registry import resolve_game
from harness.agents import HeuristicAgent, RandomAgent
from harness.tournament import play_match, run_tournament, ORACLE_ID


def _game():
    return resolve_game("connect4")


def test_play_match_rates_are_normalised_and_track_the_first_player():
    game = _game()
    res = play_match(game, RandomAgent, RandomAgent, n=8, rng=random.Random(0), opening_plies=2)
    assert abs(res["a_win"] + res["draw"] + res["b_win"] - 1.0) < 1e-9
    # first-player accounting partitions every decisive game by who moved first
    assert abs(res["first_player_win_rate"] + res["draw"] + res["first_player_loss_rate"] - 1.0) < 1e-9
    assert res["games"] == 8


def test_stronger_side_wins_the_match():
    game = _game()
    # heuristic (win/block/centre) crushes random; A/B result is seat-cancelled so strength shows through.
    res = play_match(game, HeuristicAgent, RandomAgent, n=12, rng=random.Random(1), opening_plies=0)
    assert res["a_win"] > res["b_win"]


def test_run_tournament_round_robins_ranks_and_scores_optimality():
    req = {
        "game": "connect4",
        "competitors": [
            {"id": "rand", "label": "random", "model_name": "random"},
            {"id": "heur", "label": "heuristic", "model_name": "heuristic"},
        ],
        "include_oracle": True,
        "oracle_depth": 4,
        "games_per_pair": 6,
        "opening_plies": 2,
    }
    res = run_tournament(req)
    # oracle was added as a competitor
    ids = [c["id"] for c in res["competitors"]]
    assert ORACLE_ID in ids and "rand" in ids and "heur" in ids
    # round-robin over 3 competitors → 3 pairs
    assert len(res["matrix"]) == 3
    # standings are ranked + the oracle (perfect) tops them, random is bottom
    assert [s["rank"] for s in res["standings"]] == [1, 2, 3]
    assert res["standings"][0]["id"] == ORACLE_ID
    assert res["standings"][-1]["id"] == "rand"
    # optimality measured for the non-oracle competitors; random can't beat the oracle as P1
    assert set(res["optimality"].keys()) == {"rand", "heur"}
    assert res["optimality"]["rand"]["wins_as_p1_vs_oracle"] < 0.5
    assert res["optimality"]["rand"]["verdict"] in ("suboptimal", "near-optimal")
    # self-play ran for the default (first) competitor and reports a first-player win rate
    assert "rand" in res["selfPlay"]
    assert 0.0 <= res["selfPlay"]["rand"]["first_player_win_rate"] <= 1.0
    assert 0.0 <= res["firstPlayerWinRate"] <= 1.0


def test_self_play_ids_are_honoured():
    req = {
        "game": "connect4",
        "competitors": [{"id": "heur", "model_name": "heuristic"}],
        "games_per_pair": 4,
        "self_play_ids": ["heur"],
    }
    res = run_tournament(req)
    assert "heur" in res["selfPlay"] and res["selfPlay"]["heur"]["games"] == 4


def test_run_tournament_streams_per_pairing_progress_markers():
    events = []
    req = {
        "game": "connect4",
        "competitors": [
            {"id": "rand", "label": "random", "model_name": "random"},
            {"id": "heur", "label": "heuristic", "model_name": "heuristic"},
        ],
        "include_oracle": True,
        "oracle_depth": 4,
        "games_per_pair": 4,
        "self_play_ids": ["heur"],
    }
    res = run_tournament(req, on_progress=events.append)
    phases = [e["phase"] for e in events]
    assert phases[0] == "start"
    start = events[0]
    assert start["pairings"] == 3  # 3 competitors → 3 pairs
    assert [c["id"] for c in start["competitors"]] == [c["id"] for c in res["competitors"]]
    # one round-robin marker per pairing, with a growing done count + partial standings + the pair result
    rr = [e for e in events if e["phase"] == "roundrobin"]
    assert [e["done"] for e in rr] == [1, 2, 3]
    assert all(e["total"] == 3 and "standings" in e and "pair" in e for e in rr)
    # the LAST partial standings match the final standings (same live accumulators)
    assert [s["id"] for s in rr[-1]["standings"]] == [s["id"] for s in res["standings"]]
    # one self-play marker + one optimality marker per non-oracle competitor
    assert [e["id"] for e in events if e["phase"] == "selfplay"] == ["heur"]
    assert {e["id"] for e in events if e["phase"] == "optimality"} == {"rand", "heur"}


def test_progress_callback_does_not_change_the_result():
    req = {
        "game": "connect4",
        "competitors": [{"id": "heur", "model_name": "heuristic"}, {"id": "rand", "model_name": "random"}],
        "games_per_pair": 4,
    }
    base = run_tournament(req)
    assert run_tournament(req, on_progress=lambda _p: None) == base  # byte-identical with a callback


def test_parallel_round_robin_is_wellformed_and_deterministic():
    # 4 fast competitors → 6 pairings ≥ the parallel threshold → exercises the process-pool path end to end.
    req = {
        "game": "connect4",
        "competitors": [
            {"id": "heur", "label": "heuristic", "model_name": "heuristic"},
            {"id": "rand", "label": "random", "model_name": "random"},
            {"id": "m10", "label": "mcts@10", "model_name": "mcts", "mcts_sims": 10},
            {"id": "m5", "label": "mcts@5", "model_name": "mcts", "mcts_sims": 5},
        ],
        "games_per_pair": 4,
        "opening_plies": 2,
    }
    a = run_tournament(req)
    b = run_tournament(req)
    assert a == b  # each pairing carries its own seed → deterministic despite parallel completion order
    assert len(a["matrix"]) == 6  # every pairing present, re-sorted into (i,j) order
    assert [s["rank"] for s in a["standings"]] == [1, 2, 3, 4]
    assert a["standings"][0]["id"] != "rand"  # random shouldn't top the tournament
