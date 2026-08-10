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


def test_resolve_agent_builds_baselines_and_rejects_unknown():
    assert isinstance(resolve_agent("random", {}), RandomAgent)
    assert isinstance(resolve_agent("heuristic", {}), HeuristicAgent)
    m = resolve_agent("mcts", {"mcts_sims": 25})
    assert isinstance(m, MctsAgent) and m.sims == 25
    with pytest.raises(ValueError):
        resolve_agent("nope", {})
    with pytest.raises(NotImplementedError):
        resolve_agent("champion:abc", {})
