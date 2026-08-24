"""PROVE-IT-GOOD scorecard — the reusable, chat-reachable evaluation capability (plan §C.6, directive #3).

One call turns a trained model into an HONEST scorecard: how close to optimal is it, and is it *proven* so? It
composes the measurement primitives (`benchmark.py`) into a single verdict, surfacing the two that were defined but
wired nowhere — `verify_solved` (the EXACT-oracle gate) and `sim_scaling_curve` (strength-per-compute). It is the
diagnostic behind the Connect-4-to-SOLVED driving case AND the generic scoring engine the meta-selector will rank
processes with. Solver-dependent parts run where a `SolvableGame` oracle exists; everything degrades honestly where
it does not (fields become null, never fabricated).

CLI contract (mirrors `harness.tournament`): `--config-json` in, `--summary-out` out, `@@PROGRESS` lines streamed.
Request: { game, model:{checkpoint|az_weights, az_sims, az_gumbel, az_c_scale, ...}, games, ladder_depths:[6,8,10],
sims_curve:[2,8,32], reference_depth:8, corpus:{n,min_moves,seed}, exact:false }.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

from harness.benchmark import (
    evaluate_optimality,
    optimality_ladder,
    p1_conversion,
    sample_solvable_positions,
    sim_scaling_curve,
    verify_solved,
)
from harness.game import Game
from harness.gauntlet import _model_factory
from harness.registry import resolve_game


def _neural_model_at(spec: dict, game: Game) -> Callable[[int], object] | None:
    """A sims-parameterised builder `sims -> agent` for the sim-scaling curve (`sim_scaling_curve` calls it as
    `model_factory(sims)` and treats the result as the agent, a FRESH one per game). Loads a neural checkpoint's net
    ONCE and rebuilds the agent at each budget under its DEPLOYMENT operator (gumbel knobs from the spec). None for
    non-neural specs (the curve is skipped)."""
    weights = spec.get("az_weights") or (
        json.loads(Path(spec["checkpoint"]).read_text()).get("az_weights") if spec.get("checkpoint") else None
    )
    if not weights:
        return None
    resolved = spec
    if spec.get("checkpoint"):
        resolved = {**json.loads(Path(spec["checkpoint"]).read_text()), **spec}
    from harness.config import DEFAULT_AZ_SOLVE_ENDGAME
    from harness.neural import AlphaZeroAgent, load_net

    net = load_net(weights)
    se = int(resolved.get("az_solve_endgame", DEFAULT_AZ_SOLVE_ENDGAME))
    gumbel = bool(resolved.get("az_gumbel", False))
    gumbel_m = int(resolved.get("az_gumbel_m", 16))
    c_scale = float(resolved.get("az_c_scale", 0.1))
    return lambda sims: AlphaZeroAgent(net, sims=int(sims), solve_endgame=se, gumbel=gumbel,
                                       gumbel_m=gumbel_m, c_scale=c_scale, temperature=0.0)


def scorecard_verdict(card: dict) -> tuple[str, str]:
    """The honest headline (verdict, one-sentence rationale) from an assembled scorecard. `solved` is the ONLY
    verdict that may claim perfection and requires the EXACT oracle (verify_solved.solved, or the ladder's exact
    rung cleared). Below that: near-optimal (clears the deepest proxy rung + high distance-to-optimal + never loses),
    strong (high distance-to-optimal), developing (otherwise). Never overclaims — a depth-proxy win is progress, not proof."""
    vs = card.get("verify_solved")
    frontier = (card.get("ladder") or {}).get("frontier", "none")
    if (vs and vs.get("solved")) or frontier == "exact":
        return "solved", "converts the first-player win vs the EXACT oracle — proven perfect."
    om = card.get("oracle_match")
    not_lost = (card.get("p1") or {}).get("not_lost_rate", 0.0)
    deep = frontier.startswith("depth-") and int(frontier.split("-")[1]) >= 10
    if om is not None and om >= 0.95 and not_lost >= 0.95 and deep:
        return "near-optimal", f"clears {frontier}, never loses (not-lost {not_lost:.2f}), distance-to-optimal {om:.2f} — near-optimal but unproven."
    if om is not None and om >= 0.80:
        return "strong", f"distance-to-optimal {om:.2f}, frontier {frontier} — strong but not near-optimal."
    return "developing", f"distance-to-optimal {om if om is None else round(om, 2)}, frontier {frontier} — still developing."


def process_scorecard(request: dict, on_progress: Callable[[dict], None] | None = None) -> dict:
    """Assemble the PROVE-IT-GOOD scorecard for one model. Fast by default (depth-proxy ladder + sim-scaling +
    mid-game distance-to-optimal + P1 conversion/not-lost vs a fast reference); `exact:true` adds the slow
    verify_solved EXACT gate. `start` (a serialised position) scopes the audit to a subtree (e.g. a forced-win root)."""
    game = resolve_game(request.get("game", "connect4"))
    games = int(request.get("games", 12))
    depths = tuple(request.get("ladder_depths", [6, 8, 10]))
    sims_curve = tuple(request.get("sims_curve", [2, 8, 32]))
    ref_depth = int(request.get("reference_depth", 8))
    exact = bool(request.get("exact", False))
    strategist = int(request.get("strategist", 0))
    model_spec = request.get("model", {})

    def emit(payload: dict) -> None:
        if on_progress is not None:
            on_progress(payload)

    mf = _model_factory(model_spec, game, max_sims=request.get("max_sims"))
    label = model_spec.get("label", "model")
    emit({"phase": "start", "game": game.name, "model": label, "exact": exact})

    # Fast depth-proxy ladder (the champion's real frontier); exact rung opt-in.
    ladder = optimality_ladder(game, mf, games=games, depths=depths, include_exact=exact)  # audits P1 (strategist 0)
    emit({"phase": "ladder", "frontier": ladder["frontier"], "rungs": ladder["rungs"]})

    # Strength-per-compute vs a fast reference, and P1 conversion/not-lost (the draw-vs-loss diagnostic).
    card: dict = {"game": game.name, "model": label, "ladder": ladder}
    from harness.solver import NearPerfectOracle

    ref = lambda: NearPerfectOracle(depth=ref_depth, solve_endgame=22) if game.name == "connect4" else None
    model_at = _neural_model_at(model_spec, game)
    if model_at is not None and game.name == "connect4":
        card["sim_scaling"] = {
            k: v for k, v in sim_scaling_curve(game, model_at, ref, sims_list=sims_curve, games=games,
                                               strategist=strategist).items() if k in ("points", "auc", "flatness")
        }
        emit({"phase": "sim_scaling", **card["sim_scaling"]})
    if game.name == "connect4":
        p1 = p1_conversion(game, mf, ref, games=games, strategist=strategist)
        card["p1"] = {k: p1[k] for k in ("rate", "not_lost_rate", "wins", "draws", "losses", "games")}
        emit({"phase": "p1", **card["p1"]})

    # Mid-game distance-to-optimal on a shared solver corpus (Connect-4; a generic-game oracle is a §C.6 gap).
    if game.name == "connect4":
        cspec = request.get("corpus", {})
        corpus = sample_solvable_positions(game, int(cspec.get("n", 120)), int(cspec.get("min_moves", 16)),
                                           int(cspec.get("seed", 999)))
        agent = mf()
        card["oracle_match"] = round(
            evaluate_optimality(game, lambda s: agent.act(game, s, random.Random(0)), states=corpus)["oracle_optimality_rate"], 3
        )
        emit({"phase": "oracle_match", "oracle_match": card["oracle_match"]})

    # The EXACT gate (opt-in, slow) — the ONLY claim allowed to say "solved".
    if exact:
        card["verify_solved"] = verify_solved(game, mf, games=games, strategist=strategist)
        emit({"phase": "verify_solved", **card["verify_solved"]})

    verdict, headline = scorecard_verdict(card)
    card["verdict"], card["headline"] = verdict, headline
    emit({"phase": "done", "verdict": verdict, "headline": headline})
    return card


def main() -> None:
    parser = argparse.ArgumentParser(prog="harness.process_eval", description="PROVE-IT-GOOD model scorecard.")
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    def _emit(payload: dict) -> None:
        print("@@PROGRESS " + json.dumps(payload), flush=True)  # flush is load-bearing: the exact tail streams

    result = process_scorecard(json.loads(Path(args.config_json).read_text()), on_progress=_emit)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
