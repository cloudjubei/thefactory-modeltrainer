"""Watch, replay, and play against a trained model — the human-facing surface over a checkpoint.

Every run already writes a `sample_game` replay into its summary and `harness.run.load_policy` already turns a
checkpoint back into a playable `act(game, state, rng)`. This module is the thin surface that makes those
usable by a person:

  * `--from-summary S.json`  render the game a run sampled, move by move.
  * `--checkpoint C.json --vs mcts`  WATCH the saved model play an opponent rung.
  * `--checkpoint C.json --vs human`  PLAY against it in the terminal (you drop columns; it answers).

The formatters and the interactive loop take injected `prompt`/`emit` callables so the loop is testable with a
scripted human and no real stdin. NOTE: the current cores (`random`/`heuristic`/`mcts`) are search/rules, not
learned weights, so "the model" is the search at the checkpoint's strength; a neural core saves weights beside
the same checkpoint spec and slots in here unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

from harness.agents import Agent, resolve_agent
from harness.config import MODEL_NAME_CHOICES, OPPONENT_MCTS_SIMS
from harness.game import Game, State
from harness.registry import personas_for, resolve_game
from harness.run import load_policy
from harness.selfplay import play_match

ActFn = Callable[[Game, State, random.Random], int]


def _result_line(winner: int | None, labels: dict[int, str]) -> str:
    if winner is None:
        return "Result: draw"
    return f"Result: {labels.get(winner, f'seat {winner}')} wins"


def render_replay(game: Game, replay: dict, labels: dict[int, str] | None = None) -> str:
    """Format a replay dict (`{model_seat, winner, moves, frames}`, as captured by `play_match` and stored as a
    summary's `sample_game`) into a move-by-move transcript. `frames[0]` is the initial board and `frames[i+1]`
    is the board after `moves[i]`."""
    model_seat = int(replay.get("model_seat", 0))
    opponent = replay.get("opponent")
    if labels is None:
        labels = {model_seat: "model", 1 - model_seat: opponent or "opponent"}
    frames = list(replay.get("frames") or [])
    moves = list(replay.get("moves") or [])

    header = f"Replay — model is seat {model_seat}" + (f", opponent '{opponent}'" if opponent else "")
    lines = [header]
    if frames:
        lines.append(frames[0])
    for i, mv in enumerate(moves):
        who = labels.get(int(mv.get("player", -1)), f"seat {mv.get('player')}")
        label = mv.get("label", mv.get("action"))
        lines.append(f"Move {i + 1}: {who} plays {label}")
        if i + 1 < len(frames):
            lines.append(frames[i + 1])
    lines.append(_result_line(replay.get("winner"), labels))
    return "\n".join(lines)


def parse_human_action(game: Game, state: State, raw: str) -> int:
    """Parse a human's typed move into a legal action, raising ValueError (with a helpful message) otherwise."""
    text = (raw or "").strip()
    try:
        action = int(text)
    except (TypeError, ValueError):
        raise ValueError(f"{text!r} is not a column number")
    legal = game.legal_actions(state)
    if action not in legal:
        raise ValueError(f"column {action} is not a legal move (legal: {legal})")
    return action


def _read_human_action(game: Game, state: State, prompt: Callable[[], str], emit: Callable[[str], None]) -> int:
    while True:
        emit(f"Your move — legal columns {game.legal_actions(state)}:")
        try:
            return parse_human_action(game, state, prompt())
        except ValueError as e:
            emit(f"Invalid: {e}. Try again.")


def play_human_vs_model(
    game: Game,
    model_act: ActFn,
    human_seat: int,
    rng: random.Random | None,
    prompt: Callable[[], str],
    emit: Callable[[str], None],
) -> dict:
    """Interactive loop: you (`human_seat`) and the model alternate until the game ends. Renders each resulting
    board through `emit` and returns a replay dict in the same shape as a summary's `sample_game`."""
    rng = rng or random.Random(0)
    model_seat = 1 - human_seat
    state = game.initial_state(rng)
    frames = [game.render(state)]
    moves: list[dict] = []
    emit(frames[0])
    while not game.is_terminal(state):
        player = game.current_player(state)
        if player == human_seat:
            action = _read_human_action(game, state, prompt, emit)
        else:
            action = model_act(game, state, rng)
            emit(f"Model plays {game.action_label(state, action)}.")
        moves.append({"player": player, "action": action, "label": game.action_label(state, action)})
        state = game.step(state, action, rng)
        frames.append(game.render(state))
        emit(frames[-1])
    winner = game.winner(state)
    labels = {human_seat: "you", model_seat: "model"}
    emit(_result_line(winner, labels))
    return {"model_seat": model_seat, "human_seat": human_seat, "winner": winner, "moves": moves, "frames": frames}


class _FnAgent:
    """Adapt a bare `act` function into the Agent object `play_match` expects."""

    def __init__(self, fn: ActFn):
        self._fn = fn

    def act(self, game: Game, state: State, rng: random.Random) -> int:
        return self._fn(game, state, rng)


def _model_act(game: Game, args: argparse.Namespace) -> tuple[ActFn, str]:
    """Build the model's `act` from either a saved checkpoint or an inline `--model` core, plus a label."""
    if args.checkpoint:
        return load_policy(args.checkpoint), Path(args.checkpoint).stem
    cfg = {"model_name": args.model, "mcts_sims": args.mcts_sims, "game": args.game}
    agent: Agent = resolve_agent(args.model, cfg, personas_for(args.game))
    return (lambda g, s, r: agent.act(g, s, r)), args.model


def main() -> None:
    parser = argparse.ArgumentParser(prog="harness.play", description="Watch, replay, or play against a model.")
    parser.add_argument("--from-summary", type=Path, help="render the sample_game a run summary captured, then exit")
    parser.add_argument("--checkpoint", type=Path, help="a saved checkpoint to load as the model")
    parser.add_argument("--model", choices=MODEL_NAME_CHOICES, default="mcts", help="inline core if no checkpoint")
    parser.add_argument("--mcts-sims", type=int, default=200, help="strength of an inline mcts model")
    parser.add_argument("--game", default="connect4")
    parser.add_argument("--vs", default="human", help="human | random | heuristic | mcts | <persona>")
    parser.add_argument("--seat", type=int, default=0, help="the model's (and in a human game, YOUR) seat is the other; 0 moves first")
    parser.add_argument("--opponent-sims", type=int, default=OPPONENT_MCTS_SIMS, help="strength of an mcts opponent to watch against")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    game = resolve_game(args.game)
    rng = random.Random(args.seed)

    if args.from_summary:
        summary = json.loads(Path(args.from_summary).read_text())
        sample = summary.get("sample_game")
        if not sample:
            raise SystemExit(f"no sample_game in {args.from_summary} — run a full (non-calibrate) run to capture one")
        print(render_replay(game, sample))
        return

    model_act, model_label = _model_act(game, args)

    if args.vs == "human":
        # In a human game the model takes `--seat`; you take the other seat.
        human_seat = 1 - args.seat
        print(f"You are seat {human_seat}; the model ('{model_label}') is seat {args.seat}. Enter a column number to drop a piece.")
        play_human_vs_model(game, model_act, human_seat, rng, prompt=lambda: input("> "), emit=print)
        return

    opp = resolve_agent(args.vs, {"model_name": args.vs, "mcts_sims": args.opponent_sims, "game": args.game}, personas_for(args.game))
    result = play_match(game, _FnAgent(model_act), opp, args.seat, rng, capture=True)
    replay = {**(result.replay or {}), "opponent": args.vs}
    print(render_replay(game, replay, labels={args.seat: model_label, 1 - args.seat: args.vs}))
    print(f"(model seat {args.seat} vs {args.vs}; win={'model' if result.model_won else ('draw' if result.draw else args.vs)})")


if __name__ == "__main__":
    main()
