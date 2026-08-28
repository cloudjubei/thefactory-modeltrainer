"""§C.7 #3 — the solver-free league pool: no oracle/book anywhere, arch-matched own snapshots, weak→recent spread."""
from games.connect4 import Connect4
from harness.agents import HeuristicAgent, MctsAgent, RandomAgent
from harness.league import build_solver_free_pool, select_snapshot_spread
from harness.neural import AlphaZeroAgent, Connect4Net, save_net
from harness.solver import NearPerfectOracle


def test_select_snapshot_spread_keeps_ends_and_bounds():
    assert select_snapshot_spread([], 4) == []
    assert select_snapshot_spread([0, 1, 2], 4) == [0, 1, 2]  # fewer than k ⇒ all
    sp = select_snapshot_spread(list(range(10)), 4)
    assert sp[0] == 0 and sp[-1] == 9 and len(sp) == 4  # weak (0) → recent (9) kept
    assert sp == sorted(sp) and len(set(sp)) == len(sp)
    assert select_snapshot_spread(list(range(10)), 1) == [9]  # k=1 ⇒ the strongest


def test_build_solver_free_pool_has_no_solver(tmp_path):
    game = Connect4()
    # two arch-matched own snapshots on disk (weak→recent)
    arch = {"channels": 16, "blocks": 1, "residual": True, "batchnorm": True, "head_hidden": 8}
    for b in (0, 1):
        save_net(Connect4Net(**arch), str(tmp_path / f"ckpt_{b}.pt"))
    cfg = {"include_random": True, "include_heuristic": True, "mcts_sims": [30, 120],
           "snapshots": 2, "snapshot_sims": 16, "gumbel": True, "c_scale": 0.1}
    pool = build_solver_free_pool(game, tmp_path, [0, 1], arch, cfg)
    agents = [f() for f in pool]
    # NONE is a solver/oracle; every MCTS is pure (no book, no endgame cutoff); the AZ opponents are net-guided
    assert not any(isinstance(a, NearPerfectOracle) for a in agents)
    assert any(isinstance(a, RandomAgent) for a in agents)
    assert any(isinstance(a, HeuristicAgent) for a in agents)
    mcts = [a for a in agents if isinstance(a, MctsAgent)]
    assert len(mcts) == 2 and all(a.solve_endgame == 0 and a.book is None for a in mcts)
    az = [a for a in agents if isinstance(a, AlphaZeroAgent)]
    assert len(az) == 2 and all(a.solve_endgame == 0 and a.book is None for a in az)  # own snapshots, no proof leaves


def test_pool_skips_heuristic_when_game_lacks_hook(tmp_path):
    # genericity: a game with no heuristic_action must not get a HeuristicAgent (it would crash)
    class _NoHeuristic:
        name = "stub"
        def __getattr__(self, n):
            if n == "heuristic_action":
                raise AttributeError(n)
            raise AttributeError(n)
    cfg = {"include_random": True, "include_heuristic": True, "mcts_sims": [30], "snapshots": 0}
    pool = build_solver_free_pool(_NoHeuristic(), tmp_path, [], {}, cfg)
    assert not any(isinstance(f(), HeuristicAgent) for f in pool)
