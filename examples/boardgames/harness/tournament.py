"""Tournament — pit models DIRECTLY against each other (a round-robin) so the true winner comes from ACTUAL
games, not a rating fitted from anchor pairings. Also answers "is a model optimal?" for a solved game two ways:

  1. FIRST-PLAYER win rate. Connect 4 is a first-player win under perfect play, so a model played against
     ITSELF from the standard opening should let whoever moves first win ~100% of the time — the sharpest
     "does it convert the move advantage" signal (the user's test). We report it per match + for self-play.
  2. vs the ORACLE. A model that, as the FIRST player, beats the (near-)perfect oracle has converted the
     first-player win against perfect defence — direct evidence of optimal play (`wins_as_p1_vs_oracle`).

Contract: `.venv/bin/python -m harness.tournament --config-json {configPath} --summary-out {summaryOut}`.
Request: { game, competitors:[{id,label?,checkpoint?|model_name?,mcts_sims?}], include_oracle?, oracle_depth?,
           games_per_pair?, base_seed?, opening_plies?, self_play_ids?[] }.
Result:  { competitors, matrix, standings, firstPlayerWinRate, selfPlay, optimality }.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable
import random

from harness.game import Game
from harness.gauntlet import _model_factory
from harness.registry import resolve_game

ORACLE_ID = "oracle"


def _oracle_factory(depth: int) -> Callable[[], object]:
    from harness.solver import NearPerfectOracle

    return lambda: NearPerfectOracle(depth=depth)


def play_match(
    game: Game,
    a_factory: Callable[[], object],
    b_factory: Callable[[], object],
    n: int,
    rng: random.Random,
    opening_plies: int = 0,
) -> dict:
    """Play `n` seat-alternated games A vs B and tally BOTH the A/B result AND the first-player result (by who
    moved first, regardless of which competitor that was). A fresh agent per game (factories) so no search /
    net transposition table bleeds across games. `opening_plies` random opening moves diversify otherwise
    deterministic greedy games; 0 = the standard opening (a single canonical line per seat assignment)."""
    a_win = b_win = draw = fp_win = fp_loss = 0
    for i in range(n):
        a_seat = i % 2  # A moves first on even games, B on odd — so seat bias cancels across the match
        seats = [a_factory(), b_factory()] if a_seat == 0 else [b_factory(), a_factory()]
        state = game.initial_state(rng)
        ply = 0
        while not game.is_terminal(state):
            if ply < opening_plies:
                action = rng.choice(game.legal_actions(state))
            else:
                action = seats[game.current_player(state)].act(game, state, rng)
            state = game.step(state, action)
            ply += 1
        winner = game.winner(state)  # seat index (0 = first player) or None
        if winner is None:
            draw += 1
        elif winner == 0:
            fp_win += 1
        else:
            fp_loss += 1
        if winner is not None:
            if winner == a_seat:
                a_win += 1
            else:
                b_win += 1
    g = max(1, n)
    return {
        "a_win": a_win / g,
        "draw": draw / g,
        "b_win": b_win / g,
        "first_player_win_rate": fp_win / g,
        "first_player_loss_rate": fp_loss / g,
        "games": n,
    }


def _factories(request: dict, game: Game) -> tuple[list[dict], dict]:
    """Resolve the competitor list (+ the oracle when asked) to (competitor-meta, id→factory)."""
    competitors: list[dict] = []
    factories: dict = {}
    for idx, c in enumerate(request.get("competitors", [])):
        cid = str(c.get("id", idx))
        competitors.append({"id": cid, "label": c.get("label", cid)})
        factories[cid] = _model_factory(c, game)
    if request.get("include_oracle") and game.name == "connect4":
        competitors.append({"id": ORACLE_ID, "label": "oracle (near-perfect)"})
        factories[ORACLE_ID] = _oracle_factory(int(request.get("oracle_depth", 14)))
    return competitors, factories


def run_tournament(request: dict) -> dict:
    """Round-robin every competitor pair, rank by match score, and measure optimality (self-play first-player
    win rate + wins-as-P1 vs the oracle)."""
    game = resolve_game(request.get("game", "connect4"))
    n = int(request.get("games_per_pair", 20))
    base_seed = int(request.get("base_seed", 0))
    opening_plies = int(request.get("opening_plies", 2))
    competitors, factories = _factories(request, game)
    ids = [c["id"] for c in competitors]

    matrix: list[dict] = []
    wins = {cid: 0.0 for cid in ids}
    played = {cid: 0 for cid in ids}
    fp_games = fp_wins = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            rng = random.Random(base_seed * 100003 + i * 1009 + j)
            res = play_match(game, factories[a], factories[b], n, rng, opening_plies)
            matrix.append({"a": a, "b": b, "a_win": res["a_win"], "draw": res["draw"], "b_win": res["b_win"],
                           "first_player_win_rate": res["first_player_win_rate"], "games": n})
            wins[a] += res["a_win"] + 0.5 * res["draw"]
            wins[b] += res["b_win"] + 0.5 * res["draw"]
            played[a] += 1
            played[b] += 1
            fp_games += n
            fp_wins += round(res["first_player_win_rate"] * n)

    standings = sorted(
        ({"id": cid, "label": next(c["label"] for c in competitors if c["id"] == cid),
          "score": wins[cid], "matches": played[cid],
          "scorePerMatch": (wins[cid] / played[cid]) if played[cid] else 0.0} for cid in ids),
        key=lambda s: s["scorePerMatch"], reverse=True,
    )
    for rank, s in enumerate(standings, 1):
        s["rank"] = rank

    self_play = _self_play(request, game, factories, n, base_seed)
    optimality = _optimality(game, competitors, factories, n, base_seed) if factories.get(ORACLE_ID) else {}

    return {
        "game": game.name,
        "competitors": competitors,
        "matrix": matrix,
        "standings": standings,
        "firstPlayerWinRate": (fp_wins / fp_games) if fp_games else 0.0,
        "gamesPerPair": n,
        "selfPlay": self_play,
        "optimality": optimality,
    }


def _self_play(request: dict, game: Game, factories: dict, n: int, base_seed: int) -> dict:
    """Each requested competitor plays ITSELF: from the STANDARD opening a perfect player lets the first mover
    win every game, so first_player_win_rate → 1.0 is the move-advantage / optimality signal."""
    ids = request.get("self_play_ids")
    if not ids:
        ids = [cid for cid in factories if cid != ORACLE_ID][:1]  # default: the first (top) competitor
    out: dict = {}
    for cid in ids:
        f = factories.get(cid)
        if not f:
            continue
        rng = random.Random(base_seed * 7 + hash(cid) % 1000)
        res = play_match(game, f, f, n, rng, opening_plies=0)  # standard opening → pure move advantage
        out[cid] = {
            "first_player_win_rate": res["first_player_win_rate"],
            "draw_rate": res["draw"],
            "first_player_loss_rate": res["first_player_loss_rate"],
            "games": n,
        }
    return out


def _optimality(game: Game, competitors: list[dict], factories: dict, n: int, base_seed: int) -> dict:
    """Per competitor (excluding the oracle): as the FIRST player vs the oracle from the standard opening, an
    optimal model wins (Connect 4 is a first-player win) → wins_as_p1_vs_oracle == 1.0 is the optimality proof."""
    oracle = factories[ORACLE_ID]
    out: dict = {}
    m = max(4, min(n, 12))
    for c in competitors:
        cid = c["id"]
        if cid == ORACLE_ID:
            continue
        rng = random.Random(base_seed * 31 + hash(cid) % 1000)
        # model ALWAYS first (a_seat forced): play_match alternates, so run model-first games explicitly.
        wins = 0
        for k in range(m):
            model, opp = factories[cid](), oracle()
            state = game.initial_state(rng)
            while not game.is_terminal(state):
                mover = game.current_player(state)
                action = (model if mover == 0 else opp).act(game, state, rng)
                state = game.step(state, action)
            if game.winner(state) == 0:
                wins += 1
        rate = wins / m
        out[cid] = {
            "wins_as_p1_vs_oracle": rate,
            "games": m,
            "verdict": "optimal" if rate >= 0.999 else ("near-optimal" if rate >= 0.5 else "suboptimal"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(prog="harness.tournament", description="Round-robin models + optimality vs oracle.")
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    result = run_tournament(json.loads(Path(args.config_json).read_text()))
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
