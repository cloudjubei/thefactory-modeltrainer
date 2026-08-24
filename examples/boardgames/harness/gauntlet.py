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


def _rung_factory(rung: dict, game: Game, max_sims: int | None = None) -> Callable[[], Agent]:
    """A fresh-per-game agent factory for one reference rung. `max_sims` caps the search rungs (mcts) so a rating
    pass stays fast — the MODEL under test is never capped (it is rated at full strength)."""
    kind = rung.get("kind")
    if kind == "random":
        return lambda: RandomAgent()
    if kind == "heuristic":
        return lambda: HeuristicAgent()
    if kind == "mcts":
        sims = _cap_sims(int(rung.get("sims", 120)), max_sims)
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


def _cap_sims(sims: int, max_sims: int | None) -> int:
    """Bound a search agent's per-move sims. A play-off round-robin is O(n²) games; a leaderboard's top model
    can be a several-thousand-sim mcts (~30s/game) that would blow the whole budget, so callers that must stay
    tractable pass `max_sims`. `None` = no cap (the gauntlet path, where each model climbs at its own strength)."""
    return min(sims, max_sims) if max_sims else sims


def _az_factory(spec: dict, weights: str, max_sims: int | None) -> Callable[[], Agent]:
    """Build a per-game AlphaZero agent factory from a checkpoint spec, threading the DEPLOYMENT operator knobs
    (`az_gumbel` / `az_gumbel_m` / `az_c_scale`) so a net trained under the validated recipe is DEPLOYED under the
    Gumbel/Sequential-Halving completed-Q search it was measured to be strongest with — not silently downgraded to
    plain PUCT. Defaults keep prior behaviour (gumbel off) for old checkpoints."""
    from harness.config import DEFAULT_AZ_SOLVE_ENDGAME
    from harness.neural import AlphaZeroAgent, load_net

    net = load_net(weights)
    sims = _cap_sims(int(spec.get("az_sims", 100)), max_sims)
    se = int(spec.get("az_solve_endgame", DEFAULT_AZ_SOLVE_ENDGAME))
    gumbel = bool(spec.get("az_gumbel", False))
    gumbel_m = int(spec.get("az_gumbel_m", 16))
    c_scale = float(spec.get("az_c_scale", 0.1))
    return lambda: AlphaZeroAgent(net, sims=sims, solve_endgame=se, gumbel=gumbel, gumbel_m=gumbel_m, c_scale=c_scale)


def _model_factory(model: dict, game: Game, max_sims: int | None = None) -> Callable[[], Agent]:
    """A fresh-per-game agent factory for a model under test — parse the checkpoint spec ONCE, construct a
    NEW agent per game (so a search/net transposition table can't bleed across games). `max_sims` bounds the
    per-move search cost of EVERY competitor (mcts + net) so a round-robin can't time out on a huge-sim model."""
    checkpoint = model.get("checkpoint")
    if checkpoint:
        spec = json.loads(Path(checkpoint).read_text())
        model_name = spec.get("model_name", "mcts")
        if model_name == "alphazero" and spec.get("az_weights"):
            return _az_factory(spec, spec["az_weights"], max_sims)
        cfg = dict(spec)
        if "mcts_sims" in cfg:
            cfg["mcts_sims"] = _cap_sims(int(cfg["mcts_sims"]), max_sims)
        return lambda: resolve_agent(model_name, cfg, personas_for(game.name))
    if model.get("az_weights"):  # a learned net passed by weights directly (e.g. a champion .pt, no spec file)
        return _az_factory(model, model["az_weights"], max_sims)
    model_name = model.get("model_name", "mcts")
    cfg = {
        "model_name": model_name,
        "mcts_sims": _cap_sims(int(model.get("mcts_sims", 120)), max_sims),
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
    max_sims: int | None = None,
) -> list[dict]:
    """Climb the (rating-sorted, weakest-first) spine; stop after the first rung the model fails to clear."""
    pairings: list[dict] = []
    for i, rung in enumerate(rungs):
        rng = random.Random(base_seed * 100003 + model_idx * 101 + i)
        res = head_to_head(game, model_factory, _rung_factory(rung, game, max_sims), n, rng, opening_plies=opening_plies)
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
    max_sims = int(request["max_sims"]) if request.get("max_sims") else None  # caps the mcts RUNGS, not the model
    ratings: list[dict] = []
    for idx, model in enumerate(request.get("models", [])):
        model_id = str(model.get("id", idx))
        try:
            pairings = climb_spine(game, _model_factory(model, game), rungs, n, base_seed, opening_plies, idx, max_sims)
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
