"""Gauntlet — play each model against a FIXED reference SPINE (+ champions) to produce COMPARABLE pairings.

The rating math lives in the engine (TS `fitRatingFromPairings`); this only plays the games and emits the
pairings, so every model — search cores AND learned nets AND past champions — is measured against the SAME
frozen opponents on one scale. It's adaptive: a model climbs the spine (weakest rung first) and STOPS after
its first clear loss, so weak models never pay to play the expensive top rungs. Randomized openings make
deterministic (greedy) net-vs-net games distinct so a rung's win-rate is real, not one game repeated.

Contract: `.venv/bin/python -m harness.gauntlet --config-json {configPath} --summary-out {summaryOut}`.
Request: { game, models:[{id, checkpoint?|model_name?,mcts_sims?}], rungs:[{id,kind,sims?,rating,weights_path?}],
           games_per_rung?, base_seed?, opening_plies? }.
Result:  { ratings:[{model_id, pairings:[{opponent,opponent_rating,score,games}], rungs_played}] }.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

from harness.agents import Agent, HeuristicAgent, MctsAgent, RandomAgent, resolve_agent
from harness.game import Game
from harness.neural import head_to_head  # the seat-alternated, random-opening match primitive (needs torch)
from harness.registry import personas_for, resolve_game

WIN_THRESHOLD = 0.55  # a rung is "cleared" at ≥ this score; the first rung below it stops the climb


def _rung_factory(rung: dict, game: Game) -> Callable[[], Agent]:
    """A fresh-per-game agent factory for one reference rung."""
    kind = rung.get("kind")
    if kind == "random":
        return lambda: RandomAgent()
    if kind == "heuristic":
        return lambda: HeuristicAgent()
    if kind == "mcts":
        sims = int(rung.get("sims", 120))
        return lambda: MctsAgent(sims=sims)
    if kind == "oracle":
        from harness.solver import NearPerfectOracle

        depth = int(rung.get("depth", 12))
        return lambda: NearPerfectOracle(depth=depth, solve_endgame=22)
    if kind == "book":
        from harness.book import load_book
        from harness.bookagent import BookAgent

        gname = game.name
        se = int(rung.get("book_solve_endgame", 22))
        depth = int(rung.get("depth", 12))
        book = load_book(gname)  # load ONCE; a fresh agent per game shares the book
        return lambda: BookAgent(book, gname, solve_endgame=se, depth=depth)
    if kind in ("champion", "alphazero"):
        from harness.neural import AlphaZeroAgent, load_net

        net = load_net(rung["weights_path"])  # load ONCE; a fresh agent per game shares the net
        sims = int(rung.get("sims", 100))
        return lambda: AlphaZeroAgent(net, sims=sims)
    raise ValueError(f"unknown rung kind {kind!r}")


def _model_factory(model: dict, game: Game) -> Callable[[], Agent]:
    """A fresh-per-game agent factory for a model under test — parse the checkpoint spec ONCE, construct a
    NEW agent per game (so a search/net transposition table can't bleed across games)."""
    checkpoint = model.get("checkpoint")
    if checkpoint:
        spec = json.loads(Path(checkpoint).read_text())
        model_name = spec.get("model_name", "mcts")
        if model_name == "alphazero" and spec.get("az_weights"):
            from harness.neural import AlphaZeroAgent, load_net

            net = load_net(spec["az_weights"])
            sims = int(spec.get("az_sims", 100))
            se = int(spec.get("az_solve_endgame", 0))
            return lambda: AlphaZeroAgent(net, sims=sims, solve_endgame=se)
        cfg = dict(spec)
        return lambda: resolve_agent(model_name, cfg, personas_for(game.name))
    if model.get("az_weights"):  # a learned net passed by weights directly (e.g. a champion .pt, no spec file)
        from harness.neural import AlphaZeroAgent, load_net

        net = load_net(model["az_weights"])
        sims = int(model.get("az_sims", 100))
        se = int(model.get("az_solve_endgame", 0))
        return lambda: AlphaZeroAgent(net, sims=sims, solve_endgame=se)
    model_name = model.get("model_name", "mcts")
    cfg = {
        "model_name": model_name,
        "mcts_sims": int(model.get("mcts_sims", 120)),
        "mcts_solve_endgame": int(model.get("mcts_solve_endgame", 0)),
        "game": game.name,
    }
    # Thread the agent-strength knobs a `book` / `oracle_depth` competitor carries, or it silently falls back to
    # the DEFAULT depth (12) — a ~30× slower opening search that makes the whole play-off time out.
    for k in ("oracle_depth", "book_solve_endgame"):
        if k in model:
            cfg[k] = int(model[k])
    return lambda: resolve_agent(model_name, cfg, personas_for(game.name))


def climb_spine(
    game: Game,
    model_factory: Callable[[], Agent],
    rungs: list[dict],
    n: int,
    base_seed: int,
    opening_plies: int,
    model_idx: int,
) -> list[dict]:
    """Climb the (rating-sorted, weakest-first) spine; stop after the first rung the model fails to clear."""
    pairings: list[dict] = []
    for i, rung in enumerate(rungs):
        rng = random.Random(base_seed * 100003 + model_idx * 101 + i)
        res = head_to_head(game, model_factory, _rung_factory(rung, game), n, rng, opening_plies=opening_plies)
        score = res["win_rate"] + 0.5 * res["draw_rate"]
        # camelCase keys: this output IS the engine's RatingPairing[] contract (fed to fitRatingFromPairings).
        pairings.append(
            {
                "opponent": rung.get("id", rung.get("kind")),
                "opponentRating": rung["rating"],
                "score": score,
                "games": n,
            }
        )
        if score < WIN_THRESHOLD:
            break
    return pairings


def run_gauntlet(request: dict) -> dict:
    """Play every model up the spine and emit its pairings (a failed model gets an {error})."""
    game = resolve_game(request.get("game", "connect4"))
    rungs = sorted(request.get("rungs", []), key=lambda r: r["rating"])
    n = int(request.get("games_per_rung", 40))
    base_seed = int(request.get("base_seed", 0))
    opening_plies = int(request.get("opening_plies", 4))
    ratings: list[dict] = []
    for idx, model in enumerate(request.get("models", [])):
        model_id = str(model.get("id", idx))
        try:
            pairings = climb_spine(game, _model_factory(model, game), rungs, n, base_seed, opening_plies, idx)
            ratings.append({"model_id": model_id, "pairings": pairings, "rungs_played": len(pairings)})
        except Exception as e:
            ratings.append({"model_id": model_id, "error": str(e)})
    return {"ratings": ratings}


def main() -> None:
    parser = argparse.ArgumentParser(prog="harness.gauntlet", description="Rate models against a fixed spine.")
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    result = run_gauntlet(json.loads(Path(args.config_json).read_text()))
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
