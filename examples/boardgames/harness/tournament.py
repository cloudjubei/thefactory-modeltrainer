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
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from harness.game import Game
from harness.gauntlet import _model_factory
from harness.registry import resolve_game

ORACLE_ID = "oracle"

# --- PARALLEL round-robin: each pairing / optimality check is independent, and the near-perfect agents are slow
# (an oracle move from the opening is ~seconds), so a sequential O(n²) round-robin can blow past the job budget.
# We fan the matches across CPU cores. A worker builds the game + agent factories ONCE (nets loaded once per
# worker), then plays whichever matches it's handed. Each task carries its OWN seed, so results are identical to
# the sequential path regardless of completion order — the final matrix is re-sorted into (i,j) order. Any pool
# failure (or a small run) falls back to in-process sequential, so this never breaks a play-off, only speeds it.
_WORKER: dict = {}


def _init_worker(request: dict) -> None:
    try:
        import torch

        torch.set_num_threads(1)  # each worker is single-threaded so N workers don't oversubscribe the cores
    except Exception:
        pass
    game = resolve_game(request.get("game", "connect4"))
    _WORKER["game"] = game
    _WORKER["factories"] = _factories(request, game)[1]


def _pair_task(args: tuple) -> tuple:
    a, b, n, seed_key, opening_plies = args
    res = play_match(_WORKER["game"], _WORKER["factories"][a], _WORKER["factories"][b], n, random.Random(seed_key), opening_plies)
    return a, b, res


def _opt_task(args: tuple) -> tuple:
    cid, m, seed_key = args
    game, factories = _WORKER["game"], _WORKER["factories"]
    oracle = factories[ORACLE_ID]
    rng = random.Random(seed_key)
    wins = 0
    for _ in range(m):  # model ALWAYS first vs the oracle (Connect 4 is a first-player win under perfect play)
        model, opp = factories[cid](), oracle()
        state = game.initial_state(rng)
        while not game.is_terminal(state):
            mover = game.current_player(state)
            state = game.step(state, (model if mover == 0 else opp).act(game, state, rng))
        if game.winner(state) == 0:
            wins += 1
    return cid, wins


def _imap_indexed(request: dict, fn: Callable, tasks: list, min_tasks: int = 4):
    """Yield (original_index, result) as each task finishes — parallel across a spawned process pool when there's
    enough work + cores, else in-process sequential. A pool that can't even be CREATED (sandbox forbids it) falls
    back cleanly with no double-execution; only after tasks are running do errors propagate (they surface as a
    failed play-off, and the already-streamed partial standings are preserved)."""
    ex = None
    if len(tasks) >= min_tasks and (os.cpu_count() or 1) >= 2:
        try:
            ex = ProcessPoolExecutor(
                max_workers=min(os.cpu_count() or 2, len(tasks), 8),
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
                initargs=(request,),
            )
        except Exception:
            ex = None
    if ex is not None:
        try:
            futures = {ex.submit(fn, t): i for i, t in enumerate(tasks)}
            for fut in as_completed(futures):
                yield futures[fut], fut.result()
        finally:
            ex.shutdown(wait=True)
        return
    for i, t in enumerate(tasks):
        yield i, fn(t)


def _oracle_factory(depth: int) -> Callable[[], object]:
    from harness.solver import NearPerfectOracle

    # Exact-endgame cutoff makes the oracle EXACT once few cells remain AND far faster than depth-searching the
    # endgame — the difference between a play-off that finishes in minutes and one that times out.
    return lambda: NearPerfectOracle(depth=depth, solve_endgame=22)


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


def _standings(competitors: list[dict], wins: dict, played: dict) -> list[dict]:
    """Rank competitors by match score-per-match (win=1, draw=½), highest first. Works on PARTIAL accumulators
    too (fewer matches played), so the same rows feed both the live progress stream and the final result."""
    label_of = {c["id"]: c["label"] for c in competitors}
    rows = sorted(
        ({"id": cid, "label": label_of.get(cid, cid), "score": wins[cid], "matches": played[cid],
          "scorePerMatch": (wins[cid] / played[cid]) if played[cid] else 0.0} for cid in [c["id"] for c in competitors]),
        key=lambda s: s["scorePerMatch"], reverse=True,
    )
    for rank, s in enumerate(rows, 1):
        s["rank"] = rank
    return rows


def run_tournament(request: dict, on_progress: Callable[[dict], None] | None = None) -> dict:
    """Round-robin every competitor pair, rank by match score, and measure optimality (self-play first-player
    win rate + wins-as-P1 vs the oracle). `on_progress`, when given, is called with a `start` marker, then one
    `roundrobin` marker per pairing (carrying the growing partial standings), then one `selfplay`/`optimality`
    marker per competitor — the live stream the viewer renders. It does NOT affect the returned result."""
    game = resolve_game(request.get("game", "connect4"))
    n = int(request.get("games_per_pair", 20))
    base_seed = int(request.get("base_seed", 0))
    opening_plies = int(request.get("opening_plies", 2))
    competitors, factories = _factories(request, game)
    ids = [c["id"] for c in competitors]

    def emit(payload: dict) -> None:
        if on_progress is not None:
            on_progress(payload)

    pairings_total = len(ids) * (len(ids) - 1) // 2
    self_play_ids = [cid for cid in (request.get("self_play_ids") or [c for c in ids if c != ORACLE_ID][:1]) if factories.get(cid)]
    has_oracle = bool(factories.get(ORACLE_ID))
    optimality_total = sum(1 for c in competitors if c["id"] != ORACLE_ID) if has_oracle else 0
    emit({
        "phase": "start", "game": game.name,
        "competitors": [{"id": c["id"], "label": c["label"]} for c in competitors],
        "pairings": pairings_total, "selfPlayCount": len(self_play_ids),
        "optimalityCount": optimality_total, "gamesPerPair": n,
    })

    # Round-robin, fanned across cores (each pairing is independent). Keep the matrix in (i,j) order for a
    # deterministic result; accumulate standings + stream a marker as each pairing finishes (completion order).
    _WORKER["game"], _WORKER["factories"] = game, factories  # used by the in-process sequential fallback
    pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
    pair_seeds = [base_seed * 100003 + i * 1009 + j for i in range(len(ids)) for j in range(i + 1, len(ids))]
    pair_tasks = [(a, b, n, pair_seeds[k], opening_plies) for k, (a, b) in enumerate(pairs)]
    matrix_by_idx: list[dict | None] = [None] * len(pairs)
    wins = {cid: 0.0 for cid in ids}
    played = {cid: 0 for cid in ids}
    fp_games = fp_wins = 0
    done = 0
    for idx, (a, b, res) in _imap_indexed(request, _pair_task, pair_tasks):
        pair = {"a": a, "b": b, "a_win": res["a_win"], "draw": res["draw"], "b_win": res["b_win"],
                "first_player_win_rate": res["first_player_win_rate"], "games": n}
        matrix_by_idx[idx] = pair
        wins[a] += res["a_win"] + 0.5 * res["draw"]
        wins[b] += res["b_win"] + 0.5 * res["draw"]
        played[a] += 1
        played[b] += 1
        fp_games += n
        fp_wins += round(res["first_player_win_rate"] * n)
        done += 1
        emit({
            "phase": "roundrobin", "done": done, "total": pairings_total, "pair": pair,
            "standings": _standings(competitors, wins, played),
            "firstPlayerWinRate": (fp_wins / fp_games) if fp_games else 0.0,
        })
    matrix = [m for m in matrix_by_idx if m is not None]

    standings = _standings(competitors, wins, played)

    self_play = _self_play(request, game, factories, n, base_seed, on_progress=on_progress, ids=self_play_ids)

    # Optimality vs the oracle — also fanned across cores (each competitor's model-first games are independent).
    optimality: dict = {}
    if has_oracle:
        m = max(4, min(n, 12))
        opt_tasks = [(c["id"], m, base_seed * 31 + hash(c["id"]) % 1000) for c in competitors if c["id"] != ORACLE_ID]
        opt_done = 0
        for _idx, (cid, w) in _imap_indexed(request, _opt_task, opt_tasks, min_tasks=3):
            rate = w / m
            entry = {
                "wins_as_p1_vs_oracle": rate, "games": m,
                "verdict": "optimal" if rate >= 0.999 else ("near-optimal" if rate >= 0.5 else "suboptimal"),
            }
            optimality[cid] = entry
            opt_done += 1
            emit({"phase": "optimality", "id": cid, "done": opt_done, "total": len(opt_tasks), **entry})

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


def _self_play(
    request: dict, game: Game, factories: dict, n: int, base_seed: int,
    on_progress: Callable[[dict], None] | None = None, ids: list[str] | None = None,
) -> dict:
    """Each requested competitor plays ITSELF: from the STANDARD opening a perfect player lets the first mover
    win every game, so first_player_win_rate → 1.0 is the move-advantage / optimality signal."""
    if ids is None:
        ids = request.get("self_play_ids") or [cid for cid in factories if cid != ORACLE_ID][:1]
    targets = [cid for cid in ids if factories.get(cid)]
    out: dict = {}
    done = 0
    for cid in targets:
        rng = random.Random(base_seed * 7 + hash(cid) % 1000)
        res = play_match(game, factories[cid], factories[cid], n, rng, opening_plies=0)  # standard opening
        entry = {
            "first_player_win_rate": res["first_player_win_rate"],
            "draw_rate": res["draw"],
            "first_player_loss_rate": res["first_player_loss_rate"],
            "games": n,
        }
        out[cid] = entry
        done += 1
        if on_progress is not None:
            on_progress({"phase": "selfplay", "id": cid, "done": done, "total": len(targets), **entry})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(prog="harness.tournament", description="Round-robin models + optimality vs oracle.")
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    def _emit(payload: dict) -> None:
        # `flush=True` is load-bearing: without it stdout buffers until exit and the slow oracle tail never streams.
        print("@@PROGRESS " + json.dumps(payload), flush=True)

    result = run_tournament(json.loads(Path(args.config_json).read_text()), on_progress=_emit)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
