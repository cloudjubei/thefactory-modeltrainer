import random
import time
from concurrent.futures import Future

import harness.tournament as tour
from harness.registry import resolve_game
from harness.agents import HeuristicAgent, RandomAgent
from harness.tournament import play_match, run_tournament, ORACLE_ID


class _WedgedPool:
    """A pool that delivers ONLY the pre-chosen submit-indices and leaves the rest pending forever — a stand-in
    for a worker OOM-killed under memory pressure, whose futures never complete (the real-world play-off stall)."""

    def __init__(self, resolved: set[int]):
        self._resolved = set(resolved)
        self._n = 0

    def submit(self, fn, task):
        i, self._n = self._n, self._n + 1
        fut: Future = Future()
        if i in self._resolved:
            fut.set_result(("POOL", task))
        return fut  # unresolved futures stay pending → the pool "wedges" on them

    def shutdown(self, wait=False, cancel_futures=False):
        pass


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


def test_oracle_label_shows_its_search_depth():
    # A depth-limited oracle is EXACT only in the endgame; the label must state its depth so a depth-6 yardstick
    # isn't mistaken for perfect play (deeper vs shallower is 50/50 head-to-head — the seat decides, not depth).
    res = run_tournament({
        "game": "connect4",
        "competitors": [{"id": "heur", "model_name": "heuristic"}],
        "include_oracle": True,
        "oracle_depth": 6,
        "games_per_pair": 2,
    })
    oracle = next(c for c in res["competitors"] if c["id"] == ORACLE_ID)
    assert oracle["label"] == "oracle (depth 6)"


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


def test_imap_indexed_finishes_stragglers_in_process_when_the_pool_wedges(monkeypatch):
    # A wedged pool (some futures never complete) must NOT hang the play-off: after a stall window with zero
    # progress, the undelivered tasks are finished IN-PROCESS so every task index is yielded exactly once.
    tasks = list(range(6))
    resolved = {0, 2, 4}  # the pool delivers these; 1, 3, 5 wedge forever
    monkeypatch.setattr(tour, "_make_pool", lambda request, tasks, min_tasks: _WedgedPool(resolved))
    out = dict(tour._imap_indexed({}, lambda t: ("SEQ", t), tasks, min_tasks=1, stall_s=0.05))
    assert set(out) == set(tasks)  # nothing lost, nothing hung
    for i in tasks:
        tag, value = out[i]
        assert value == i
        assert tag == ("POOL" if i in resolved else "SEQ")  # wedged ones were recovered in-process


def test_imap_indexed_runs_sequentially_below_the_parallel_threshold():
    # Below the parallel threshold `_make_pool` returns None; every task still runs, in index order.
    got = list(tour._imap_indexed({}, lambda t: t * t, [3, 4, 5], min_tasks=99, stall_s=0.05))
    assert got == [(0, 9), (1, 16), (2, 25)]


def test_play_off_with_pathological_high_sim_models_completes_and_is_wellformed():
    # THE REAL failure: a leaderboard's top models are high-sim mcts (mcts@2000 is ~15s/game — a 4-game
    # pairing alone would blow a play-off's budget). With max_sims the whole round-robin + optimality must
    # FINISH with every pairing present, quickly. Uncapped, two such models pairing is minutes.
    req = {
        "game": "connect4",
        "competitors": [
            {"id": "big1", "label": "mcts·big1", "model_name": "mcts", "mcts_sims": 2000},
            {"id": "big2", "label": "mcts·big2", "model_name": "mcts", "mcts_sims": 2000},
            {"id": "heur", "label": "heuristic", "model_name": "heuristic"},
            {"id": "rand", "label": "random", "model_name": "random"},
        ],
        "include_oracle": True,
        "oracle_depth": 6,
        "games_per_pair": 4,
        "opening_plies": 2,
        "max_sims": 80,
    }
    t = time.time()
    res = run_tournament(req)
    elapsed = time.time() - t
    # 5 competitors (incl. oracle) → C(5,2)=10 pairings, ALL present, everyone played their full round-robin
    assert len(res["matrix"]) == 10
    assert all(s["matches"] == 4 for s in res["standings"])
    assert [s["rank"] for s in res["standings"]] == [1, 2, 3, 4, 5]
    assert res["standings"][-1]["id"] == "rand"  # random is bottom (sanity)
    # the cap makes it tractable — uncapped this is minutes; capped it is seconds even sequentially
    assert elapsed < 120, f"play-off took {elapsed:.0f}s — the max_sims cap is not bounding search cost"
