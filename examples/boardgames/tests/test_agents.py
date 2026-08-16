import random

import pytest

from games.connect4 import Connect4
from harness.agents import HeuristicAgent, MctsAgent, RandomAgent, resolve_agent


def test_random_agent_only_plays_legal_moves():
    game, agent, rng = Connect4(), RandomAgent(), random.Random(1)
    state = game.initial_state(rng)
    for _ in range(10):
        if game.is_terminal(state):
            break
        a = agent.act(game, state, rng)
        assert a in game.legal_actions(state)
        state = game.step(state, a)


def test_heuristic_agent_takes_an_immediate_win():
    game = Connect4()
    state = game.initial_state(random.Random(0))
    for c in [0, 6, 1, 6, 2, 6]:  # player 0 has 0,1,2 on the bottom, to move
        state = game.step(state, c)
    assert HeuristicAgent().act(game, state, random.Random(0)) == 3


def test_mcts_beats_random_convincingly():
    game = Connect4()
    rng = random.Random(7)
    wins = 0
    n = 20
    for i in range(n):
        model = MctsAgent(sims=60)
        opp = RandomAgent()
        seats = [model, opp] if i % 2 == 0 else [opp, model]
        model_seat = 0 if i % 2 == 0 else 1
        state = game.initial_state(rng)
        while not game.is_terminal(state):
            state = game.step(state, seats[game.current_player(state)].act(game, state, rng))
        if game.winner(state) == model_seat:
            wins += 1
    assert wins / n >= 0.7  # a 60-sim search should crush random in Connect 4


def test_mcts_counts_the_simulations_it_spends():
    game = Connect4()
    agent = MctsAgent(sims=50)
    agent.act(game, game.initial_state(random.Random(0)), random.Random(0))
    assert agent.sims_used == 50


def _replay(game, actions):
    state = game.initial_state(random.Random(0))
    for a in actions:
        state = game.step(state, a)
    return state


def test_mcts_takes_an_immediate_win():
    game = Connect4()
    # player 0 has three in column 3 (rows 0-2) and is to move: col 3 completes a vertical four.
    state = _replay(game, [3, 0, 3, 1, 3, 2])
    assert game.current_player(state) == 0
    assert MctsAgent(sims=200).act(game, state, random.Random(0)) == 3


def test_mcts_blocks_an_immediate_threat():
    game = Connect4()
    # player 1 has three in column 3; player 0 is to move and MUST block col 3 or lose next turn. A real
    # tree search sees the opponent's winning reply and blocks; flat Monte-Carlo often misses it.
    state = _replay(game, [0, 3, 1, 3, 2, 3])
    assert game.current_player(state) == 0
    assert MctsAgent(sims=400).act(game, state, random.Random(0)) == 3


def test_mcts_takes_the_win_without_needing_search():
    game = Connect4()
    state = _replay(game, [3, 0, 3, 1, 3, 2])  # P0 completes column 3 vertically
    assert MctsAgent(sims=1).act(game, state, random.Random(0)) == 3  # the tactical guard, not search depth


def test_mcts_never_hands_the_opponent_a_mate_in_one():
    game = Connect4()
    state = _replay(game, [0, 3, 1, 3, 2, 3])  # P1 threatens column 3; every other move loses on the spot
    assert MctsAgent(sims=1).act(game, state, random.Random(0)) == 3  # blocks even at a single simulation


def test_mcts_builds_a_tree_beyond_the_root():
    game = Connect4()
    agent = MctsAgent(sims=200)
    agent.act(game, game.initial_state(random.Random(0)), random.Random(0))
    # A real UCT tree expands past the root + its 7 children (8 nodes); flat MC would keep only the root.
    assert len(agent._tree) > 8


def test_mcts_reuses_its_tree_across_moves():
    game = Connect4()
    agent = MctsAgent(sims=150)
    rng = random.Random(3)
    state = game.initial_state(rng)
    agent.act(game, state, rng)
    n1 = len(agent._tree)
    state = game.step(state, 3)  # our move
    state = game.step(state, 2)  # opponent reply
    agent.act(game, state, rng)
    n2 = len(agent._tree)
    assert n2 >= n1  # the transposition table persists across moves (memory), it is not reset each turn


def test_exact_optimal_actions_gates_on_solve_cost():
    from harness.benchmark import sample_solvable_positions
    from harness.solver import optimal_columns

    game = Connect4()
    # The opening (42 empty cells) is a minutes-long exact solve → refused under a cheap threshold.
    assert game.exact_optimal_actions(game.initial_state(), max_empty=14) is None
    late = sample_solvable_positions(game, n=1, min_moves=32, seed=1)[0]  # ≤ ~10 empty → instant solve
    assert game.exact_optimal_actions(late, max_empty=40) == optimal_columns(late)
    assert game.exact_optimal_actions(late, max_empty=0) is None  # explicitly disabled


def test_mcts_solve_endgame_plays_perfectly_on_late_positions():
    from harness.benchmark import sample_solvable_positions
    from harness.solver import optimal_columns

    game = Connect4()
    # With the exact-endgame cutoff on, even a ONE-simulation mcts is game-theoretically optimal in the endgame:
    # the solver, not the tree, chooses. This is the mechanism that drives its endgame loss rate to zero.
    solver_mcts = MctsAgent(sims=1, solve_endgame=40)
    states = sample_solvable_positions(game, n=20, min_moves=30, seed=7)
    assert states
    for s in states:
        assert solver_mcts.act(game, s, random.Random(0)) in optimal_columns(s)


def test_mcts_solve_endgame_is_off_by_default_so_reference_rungs_stay_pure():
    # The fixed-strength reference rungs (opponent mcts) must NOT gain solver assistance, or the rating anchors
    # would shift — the cutoff is strictly opt-in.
    assert MctsAgent(sims=10).solve_endgame == 0
    assert resolve_agent("mcts", {}).solve_endgame == 0
    assert resolve_agent("mcts", {"mcts_sims": 25, "mcts_solve_endgame": 12}).solve_endgame == 12


def test_resolve_agent_builds_baselines_and_rejects_unknown():
    assert isinstance(resolve_agent("random", {}), RandomAgent)
    assert isinstance(resolve_agent("heuristic", {}), HeuristicAgent)
    m = resolve_agent("mcts", {"mcts_sims": 25})
    assert isinstance(m, MctsAgent) and m.sims == 25
    with pytest.raises(ValueError):
        resolve_agent("nope", {})
    with pytest.raises(NotImplementedError):
        resolve_agent("champion:abc", {})
