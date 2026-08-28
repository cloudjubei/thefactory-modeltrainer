"""§C.7 batched, RESUMABLE scaled-run driver — trains a (large) net in BATCHES, checkpointing after each so a long
run is never done in one shot and survives interruption (resume picks up from the last checkpoint). PURE self-play
by default (no league / no promotion), so it is a clean falsifiable capacity experiment. Per-batch it records the
pre-registered metrics: opening_value on the TRUE (canonical) empty board, net-only oracle_optimality_rate, and the
OFF-LINE P1 loss-rate vs a depth-8 oracle from a FIXED diverse-opening corpus (the frontier that loses ~54% today).

Reusable + game-agnostic (only needs the trainer plumbing). CLI mirrors harness.book / harness.process_eval."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

from harness.benchmark import evaluate_optimality
from harness.config import BENCHMARK_MIN_MOVES, BENCHMARK_SEED
from harness.neural import AlphaZeroAgent, load_net, net_value, save_net, train_alphazero
from harness.registry import resolve_game
from harness.tablebase import Tablebase


def _deploy_agent(net, arch: dict, sims: int, gumbel: bool, c_scale: float):
    return AlphaZeroAgent(net, sims=sims, solve_endgame=0, gumbel=gumbel, c_scale=c_scale, temperature=0.0)


def offline_p1_loss(game, net, sims: int, n_openings: int, opening_plies: int, seed: int,
                    gumbel: bool = True, c_scale: float = 0.1, oracle_depth: int = 8) -> dict:
    """Champion plays P1 from `n_openings` FIXED diverse random openings vs a depth-`oracle_depth` oracle; return
    the loss/draw/win rates. The pre-registered robustness metric — a near-optimal P1 rarely loses a won opening."""
    from harness.solver import NearPerfectOracle

    w = d = l = 0
    for i in range(n_openings):
        rng = random.Random(seed * 100003 + i)
        state = game.initial_state(rng)
        for _ in range(opening_plies):
            if game.is_terminal(state):
                break
            state = game.step(state, rng.choice(game.legal_actions(state)))
        champ = _deploy_agent(net, net.arch, sims, gumbel, c_scale)
        opp = NearPerfectOracle(depth=oracle_depth, solve_endgame=22)
        seats = {0: champ, 1: opp}
        while not game.is_terminal(state):
            state = game.step(state, seats[game.current_player(state)].act(game, state, rng))
        wn = game.winner(state)
        if wn == 0:
            w += 1
        elif wn is None:
            d += 1
        else:
            l += 1
    n = max(1, n_openings)
    return {"p1_win": round(w / n, 4), "p1_draw": round(d / n, 4), "p1_loss": round(l / n, 4), "openings": n_openings}


def batch_metrics(game, net, seed: int, sims: int, benchmark_positions: int, offline_openings: int,
                  opening_plies: int, gumbel: bool, c_scale: float) -> dict:
    """The pre-registered per-batch scorecard (all net-only, deployed under the trained operator)."""
    m: dict = {}
    m["opening_value"] = round(net_value(net, game, game.initial_state(random.Random(0))), 4)  # TRUE empty board
    if benchmark_positions > 0:
        ag = _deploy_agent(net, net.arch, sims, gumbel, c_scale)
        r = random.Random(seed + 4242)
        m["oracle_optimality_rate"] = round(
            evaluate_optimality(game, lambda s: ag.act(game, s, r), n=benchmark_positions,
                                min_moves=BENCHMARK_MIN_MOVES, seed=BENCHMARK_SEED)["oracle_optimality_rate"], 4)
    if offline_openings > 0:
        m["offline"] = offline_p1_loss(game, net, sims, offline_openings, opening_plies, seed, gumbel, c_scale)
    return m


def _completed_batches(run_dir: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("ckpt_*.pt"))


def run_scaled_experiment(request: dict, on_progress: Callable[[dict], None] | None = None) -> dict:
    """Train in BATCHES with a checkpoint after each; RESUMES from the latest checkpoint in run_dir. Returns the
    accumulated per-batch metrics. Pure self-play (endgame loop optional via `endgame`)."""
    game = resolve_game(request.get("game", "connect4"))
    run_dir = Path(request["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    net_arch = dict(request["net_arch"])
    seed = int(request.get("seed", 0))
    sims = int(request.get("sims", 400))
    iters_per_batch = int(request.get("iters_per_batch", 10))
    games = int(request.get("games", 1000))
    batches = int(request.get("batches", 6))
    epochs = int(request.get("epochs", 2))
    buffer_cap = int(request.get("buffer_cap", 400000))
    opening_plies = int(request.get("opening_plies", 4))
    value_n_step = int(request.get("value_n_step", 0))  # honest default: pure-MC (see §C.7)
    gumbel = bool(request.get("gumbel", True))
    c_scale = float(request.get("c_scale", 0.1))
    endgame = bool(request.get("endgame", False))
    benchmark_positions = int(request.get("benchmark_positions", 60))
    offline_openings = int(request.get("offline_openings", 50))
    # §C.7 #3 SOLVER-FREE LEAGUE — break opening value-collapse WITHOUT an oracle (see harness/league.py).
    league = bool(request.get("league", False))
    league_frac = float(request.get("league_frac", 0.4))  # forwarded as pool_frac; kept < 1 so a self-play majority anchors honest values
    league_p1_frac = float(request.get("league_p1_frac", 0.7))
    league_anchor_frac = float(request.get("league_anchor_frac", 0.25))
    league_anchor_cap = int(request.get("league_anchor_cap", 4000))
    league_frozen_self = bool(request.get("league_frozen_self", True))
    league_cfg = {
        "include_random": bool(request.get("league_include_random", True)),
        "include_heuristic": bool(request.get("league_include_heuristic", True)),
        "mcts_sims": request.get("league_mcts_sims", [30, 120]),
        "snapshots": int(request.get("league_snapshots", 4)),
        "snapshot_sims": int(request.get("league_snapshot_sims", sims)),
        "gumbel": gumbel, "c_scale": c_scale,
    }

    def emit(p: dict) -> None:
        if on_progress:
            on_progress(p)

    done = _completed_batches(run_dir)
    start = (done[-1] + 1) if done else 0
    init_net = load_net(str(run_dir / f"ckpt_{done[-1]}.pt")) if done else None
    buffer_path = run_dir / "buffer.pt"
    init_buffer = None
    if done and buffer_path.exists():
        import torch  # local: keep the module torch-free for its CLI contract until training actually runs
        init_buffer = torch.load(buffer_path)  # carry the replay history across batches (continuous-run equivalence)
    tb_path = run_dir / "endgame.npz"
    run_tb = None
    if endgame:
        run_tb = Tablebase.load(str(tb_path), cap=int(request.get("endgame_cap", 200000))) if tb_path.exists() else Tablebase(cap=int(request.get("endgame_cap", 200000)))
    metrics_path = run_dir / "metrics.jsonl"
    emit({"phase": "resume", "completed_batches": done, "start_batch": start, "arch": net_arch})

    # EXACT-TARGET distillation (§C.7 #1 — the direct fix for opening value-collapse): a broad oracle-labelled
    # opening→endgame corpus imprinted every training pass, so the OPENING carries its true value instead of the
    # ~0 self-play collapses to. Built ONCE and cached under run_dir. distill_games=0 ⇒ off (pure self-play).
    distill_games = int(request.get("distill_games", 0))
    distill_positions = int(request.get("distill_positions", 0))
    distill_corpus = None
    if distill_games > 0 or distill_positions > 0:
        from harness.neural import build_distill_corpus

        spec = {"games": distill_games, "seed": seed, "oracle_depth": int(request.get("distill_oracle_depth", 8)),
                "exact_max_empty": 22, "opening_plies": opening_plies,
                "late": {"n": distill_positions, "min_moves": 20}}
        distill_corpus = build_distill_corpus(game, spec, cache_dir=str(run_dir), log=None)
        emit({"phase": "distill_corpus", "examples": len(distill_corpus)})

    all_metrics: list[dict] = []
    if metrics_path.exists():
        all_metrics = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]

    import torch

    for b in range(start, batches):
        opp_pool = None
        if league:  # rebuild each batch so the ladder grows with the run's own fresh snapshots
            from harness.league import build_solver_free_pool

            opp_pool = build_solver_free_pool(game, run_dir, list(range(b)), net_arch, league_cfg)
        net, hist, buf = train_alphazero(
            game, iterations=iters_per_batch, selfplay_games=games, sims=sims, epochs=epochs,
            seed=seed * 1000 + b, init_net=init_net, net_arch=net_arch, buffer_cap=buffer_cap,
            gumbel=gumbel, c_scale=c_scale, value_n_step=value_n_step, selfplay_opening_plies=opening_plies,
            endgame_tb=run_tb, endgame_max_empty=int(request.get("endgame_max_empty", 14)),
            endgame_exact_targets=int(request.get("endgame_exact_targets", 1)),
            init_buffer=init_buffer, return_buffer=True,
            selfplay_workers=int(request.get("selfplay_workers", 1)),
            distill_corpus=distill_corpus,
            distill_fraction=float(request.get("distill_fraction", 0.34)),
            opponent_pool=opp_pool, pool_frac=(league_frac if league else 0.0),
            league_p1_frac=league_p1_frac, opening_anchor_cap=(league_anchor_cap if league else 0),
            league_anchor_frac=league_anchor_frac, league_frozen_self=league_frozen_self,
        )
        save_net(net, str(run_dir / f"ckpt_{b}.pt"))
        torch.save(buf, buffer_path)  # persist the replay buffer so the NEXT batch continues, not restarts
        init_buffer = buf
        if run_tb is not None:
            run_tb.save(str(tb_path))
        m = batch_metrics(game, net, seed, sims, benchmark_positions, offline_openings, opening_plies, gumbel, c_scale)
        rec = {"batch": b, "iterations_done": (b + 1) * iters_per_batch, "final_loss": round(hist[-1]["loss"], 4),
               "endgame_total": (len(run_tb) if run_tb is not None else 0),
               "league_vs_pool": sum(h["vs_pool_games"] for h in hist), **m}
        with metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        all_metrics.append(rec)
        init_net = net
        emit({"phase": "batch", **rec})

    return {"game": game.name, "run_dir": str(run_dir), "arch": net_arch, "batches": batches,
            "metrics": all_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(prog="harness.scaled_run", description="§C.7 batched resumable scaled self-play run.")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    def _emit(p: dict) -> None:
        print("@@PROGRESS " + json.dumps(p), flush=True)

    request = json.loads(Path(args.config_json).read_text())
    result = run_scaled_experiment(request, on_progress=_emit)
    Path(args.summary_out).write_text(json.dumps(result, indent=1))
    last = result["metrics"][-1] if result["metrics"] else {}
    print(f"DONE batches={result['batches']} last={json.dumps({k: last.get(k) for k in ('batch','opening_value','oracle_optimality_rate','offline')})}")


if __name__ == "__main__":
    main()
