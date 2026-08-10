"""Self-play evaluation: play the model under test against one opponent rung, fairly (alternating seats),
and fold the outcomes into the §C metric battery + a cost accounting + a sampled replay.

A "run" evaluates ONE (model_name, opponent) pairing so the engine can treat `opponent` as the split axis
(the ladder) with the top rung marked the locked TEST. Win-rate is the truth metric; a shaped per-game
reward is the PROXY (§C.8) the gates check win-rate against.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

from harness.agents import Agent, resolve_agent
from harness.config import OPPONENT_MCTS_SIMS
from harness.game import Game, State


@dataclass
class MatchResult:
    winner: int | None
    model_seat: int
    model_won: bool
    draw: bool
    moves: int
    model_reward: float
    sims_used: int
    replay: dict | None = None


@dataclass
class EvalResult:
    opponent: str
    games: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    first_player_seat_winrate: float
    episode_reward: float
    total_moves: int
    sims_used: int
    win_rate_series: list[float] = field(default_factory=list)
    reward_series: list[float] = field(default_factory=list)
    sample_game: dict | None = None
    flags: list[str] = field(default_factory=list)


def _sims(agent: Agent) -> int:
    return int(getattr(agent, "sims_used", 0))


def play_match(
    game: Game,
    model: Agent,
    opponent: Agent,
    model_seat: int,
    rng: random.Random,
    capture: bool = False,
) -> MatchResult:
    seats: list[Agent] = [model, opponent] if model_seat == 0 else [opponent, model]
    state: State = game.initial_state(rng)
    move_log: list[dict] = []
    frames: list[str] = [game.render(state)] if capture else []
    moves = 0
    while not game.is_terminal(state):
        player = game.current_player(state)
        action = seats[player].act(game, state, rng)
        if capture:
            move_log.append({"player": player, "action": action, "label": game.action_label(state, action)})
        state = game.step(state, action, rng)
        moves += 1
        if capture:
            frames.append(game.render(state))
    winner = game.winner(state)
    draw = winner is None
    model_won = winner == model_seat
    payoff = game.returns(state)[model_seat]
    # Reward SHAPING (the proxy §C.8 tests against win-rate): the win/loss payoff plus a small bonus for
    # winning quickly. A shaping that lifts this reward but not win-rate is exactly the proxy divergence.
    fast_bonus = 0.0 if (draw or not model_won) else max(0.0, 1.0 - moves / (game.num_actions * 6)) * 0.5
    shaped = payoff + fast_bonus
    replay = None
    if capture:
        replay = {"model_seat": model_seat, "winner": winner, "moves": move_log, "frames": frames}
    return MatchResult(winner, model_seat, model_won, draw, moves, shaped, _sims(model) + _sims(opponent), replay)


def evaluate_vs_opponent(
    game: Game,
    model_name: str,
    opponent: str,
    cfg: dict,
    n_games: int,
    rng: random.Random,
    personas: dict[str, Callable[[dict], Agent]] | None = None,
    capture_sample: bool = True,
) -> EvalResult:
    wins = draws = losses = 0
    seat0_wins = 0
    total_moves = 0
    total_reward = 0.0
    sims_used = 0
    win_series: list[float] = []
    reward_series: list[float] = []
    sample: dict | None = None
    # The opponent rung is a FIXED reference — its mcts strength is the ladder's, not the model's knob.
    opp_cfg = {**cfg, "mcts_sims": OPPONENT_MCTS_SIMS}
    for i in range(n_games):
        model_seat = i % 2  # alternate seats so the first-move advantage is shared, not attributed to the model
        model = resolve_agent(model_name, cfg, personas)
        opp = resolve_agent(opponent, opp_cfg, personas)
        capture = capture_sample and i == 0
        res = play_match(game, model, opp, model_seat, rng, capture=capture)
        wins += res.model_won
        draws += res.draw
        losses += not res.model_won and not res.draw
        if res.winner == 0:
            seat0_wins += 1
        total_moves += res.moves
        total_reward += res.model_reward
        sims_used += res.sims_used
        win_series.append(wins / (i + 1))
        reward_series.append(total_reward / (i + 1))
        if capture:
            sample = {"opponent": opponent, **(res.replay or {})}
    games = max(1, n_games)
    flags: list[str] = []
    if draws == n_games:
        flags.append("all_draws")
    return EvalResult(
        opponent=opponent,
        games=n_games,
        win_rate=wins / games,
        draw_rate=draws / games,
        loss_rate=losses / games,
        first_player_seat_winrate=seat0_wins / games,
        episode_reward=total_reward / games,
        total_moves=total_moves,
        sims_used=sims_used,
        win_rate_series=win_series,
        reward_series=reward_series,
        sample_game=sample,
        flags=flags,
    )
