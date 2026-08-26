"""CLI entry: honors --config-json / --summary-out / --calibrate / --evaluate (the model-training standard).

A run EVALUATES `model_name` against one `opponent` rung over `eval_games` games and writes a §C RunSummary.
The saved checkpoint is a small JSON spec a `load_policy` seam turns back into a playable agent — the artifact
you take to a live test (e.g. a BoardGameArena bridge).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

from harness.agents import Agent, resolve_agent
from harness.config import (
    OPPONENT_MCTS_SIMS,
    STRONG_MCTS_SIMS,
    TrainerConfig,
    config_hash,
    load_config,
    load_eval_config,
)
from harness.game import Game, State
from harness.registry import personas_for, resolve_game
from harness.selfplay import evaluate_vs_opponent
from harness.summary import build_eval_summary, build_summary, write_summary

CALIBRATION_GAMES = 6
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(prog="harness.run", description="Board-game trainer (model-training standard).")
    parser.add_argument("--config-json", type=Path, help="path to a fully resolved run config")
    parser.add_argument("--summary-out", type=Path, required=True, help="where to write the RunSummary JSON")
    parser.add_argument("--calibrate", action="store_true", help="tiny end-to-end pass reporting throughput")
    parser.add_argument("--evaluate", action="store_true", help="re-test a saved checkpoint")
    args = parser.parse_args()

    if args.evaluate and args.calibrate:
        parser.error("--evaluate and --calibrate are mutually exclusive")
    if args.config_json is None and not args.calibrate:
        parser.error("--config-json is required unless --calibrate is set")

    if args.evaluate:
        _run_evaluation(args.config_json, args.summary_out)
    else:
        _run_training(args.config_json, args.summary_out, args.calibrate)


def _run_training(config_path: Path | None, summary_out: Path, calibrate: bool) -> None:
    if calibrate:
        config = replace(TrainerConfig(), model_name="mcts", mcts_sims=40, opponent="random", eval_games=CALIBRATION_GAMES)
    else:
        config = load_config(config_path)

    game = resolve_game(config.game)
    rng = random.Random(config.seed)

    wall0, cpu0 = time.perf_counter(), time.process_time()
    steps: dict[str, float] = {}
    eval_cfg = asdict(config)
    extra_spec: dict = {}
    az_report: dict | None = None
    az_metrics: dict = {}
    if config.model_name == "alphazero":
        # A LEARNED run: WARM-START from the champion, train against a strong LEAGUE, save weights, measure
        # meaningful yardsticks (vs strong mcts + the champion), and gate promotion.
        weights_path, extra_spec, az_report, az_metrics, train_seconds = _run_alphazero_training(game, config)
        steps["train_seconds"] = round(train_seconds, 3)
        eval_cfg = {**eval_cfg, "az_weights": weights_path}

    ev = evaluate_vs_opponent(
        game, config.model_name, config.opponent, eval_cfg, config.eval_games, rng, personas_for(config.game)
    )
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    steps["selfplay_seconds"] = round(wall - steps.get("train_seconds", 0.0), 3)

    checkpoint = _save_checkpoint(config, extra_spec)
    calibration = None
    if calibrate:
        calibration = {"unitsPerSecond": (config.eval_games / wall) if wall > 0 else 0.0, "secondsObserved": wall, "units": config.eval_games}

    summary = build_summary(config, ev, checkpoint, wall, cpu, steps, calibration)
    if az_metrics:
        summary["metrics"].update(az_metrics)
    if az_report:
        summary["alphazero"] = az_report
    if not calibrate:
        summary["metrics"].update(_benchmark_metrics(game, config, eval_cfg))
    write_summary(summary, summary_out)
    print(f"win_rate={ev.win_rate:.3f} vs {config.opponent} cost=${summary['cost']['estCostUsd']:.6f} summary={summary_out}")


def _benchmark_metrics(game: Game, config: TrainerConfig, eval_cfg: dict) -> dict:
    """`oracle_optimality_rate` for this run's policy — the honest 'distance to solved' gauge (how often the
    model plays a game-theoretically optimal move vs the perfect oracle, on a fixed late-game corpus). Only
    Connect 4 has a solver; off when `benchmark_positions == 0`."""
    if config.game != "connect4" or config.benchmark_positions <= 0:
        return {}
    from harness.benchmark import evaluate_optimality, exact_reference, optimality_trace
    from harness.book import load_book
    from harness.config import BENCHMARK_MIN_MOVES, BENCHMARK_SEED, DEFAULT_AZ_SOLVE_ENDGAME

    agent = resolve_agent(config.model_name, eval_cfg)
    brng = random.Random(config.seed + 4242)
    result = evaluate_optimality(
        game,
        lambda s: agent.act(game, s, brng),
        n=config.benchmark_positions,
        min_moves=BENCHMARK_MIN_MOVES,
        seed=BENCHMARK_SEED,
    )
    # Opening-INCLUSIVE trace: how deep the agent's ACTUAL line is provably optimal (book + cheap endgame solve)
    # and the first ply it deviates (-1 = optimal throughout the verified region). Unlike `oracle_optimality_rate`
    # (a late-game positional sample) this follows the real game from the start, so `optimality_verified_plies`
    # exposes exactly how much of the opening is still unverified — it climbs as the opening book fills.
    ref = exact_reference(game, book=load_book(config.game), max_empty=DEFAULT_AZ_SOLVE_ENDGAME)
    trace = optimality_trace(game, lambda s: agent.act(game, s, random.Random(config.seed + 99)), ref)
    result["optimality_verified_plies"] = trace["verified_plies"]
    result["first_blunder_ply"] = -1 if trace["first_blunder_ply"] is None else trace["first_blunder_ply"]
    return result


def _run_evaluation(config_path: Path, summary_out: Path) -> None:
    eval_config = load_eval_config(config_path)
    checkpoint = Path(eval_config.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    spec = json.loads(checkpoint.read_text())
    config = replace(load_checkpoint_config(spec), opponent=eval_config.config.opponent, eval_games=eval_config.eval_games, seed=eval_config.config.seed)

    game = resolve_game(config.game)
    rng = random.Random(config.seed)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    ev = evaluate_vs_opponent(game, config.model_name, config.opponent, asdict(config), config.eval_games, rng, personas_for(config.game))
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0

    summary = build_eval_summary(config, ev, str(checkpoint), wall, cpu, {"eval_seconds": wall})
    write_summary(summary, summary_out)
    print(f"win_rate={ev.win_rate:.3f} vs {config.opponent} (eval of {checkpoint.name}) summary={summary_out}")


def _save_checkpoint(config: TrainerConfig, extra: dict | None = None) -> str:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{config_hash(config)}.json"
    # The playable spec: which game + core + strength. A neural core adds `az_weights` (a `.pt` beside this)
    # so `load_policy` reconstructs the trained net.
    spec = {"game": config.game, "model_name": config.model_name, "mcts_sims": config.mcts_sims}
    if extra:
        spec.update(extra)
    path.write_text(json.dumps(spec, indent=2) + "\n")
    return str(path)


def _save_weights(config: TrainerConfig, net) -> str:
    from harness.neural import save_net

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{config_hash(config)}.pt"
    save_net(net, str(path))
    return str(path)


PROMOTION_THRESHOLD = 0.55  # a new net must win a majority vs the incumbent champion to be crowned


def _run_alphazero_training(game: Game, config: TrainerConfig):
    """Warm-start from the champion, train against a strong league (mcts + heuristic + past champions), save
    the weights, measure yardsticks vs a strong mcts + the champion, and PROMOTE the net if it beats the
    incumbent. Returns (weights_path, extra_spec, az_report, az_metrics, train_seconds)."""
    from harness import champions
    from harness.agents import HeuristicAgent, MctsAgent
    from harness.neural import AlphaZeroAgent, head_to_head, load_net, net_value, save_net, train_alphazero
    from harness.solver import NearPerfectOracle

    device = "cpu"
    az_sims = int(config.az_sims)
    az_solve_endgame = int(config.az_solve_endgame)  # the DEPLOYED agent's exact-endgame cutoff — eval with it too
    strong_sims = STRONG_MCTS_SIMS  # a FIXED reference so win_rate_vs_strong_mcts is a comparable anchor

    # WARM-START from the strongest saved net (cumulative), unless the run asks to start fresh. BUT: a
    # distillation run must NOT warm-start from a PRE-distillation champion — that net has a broken edge-first
    # opening the imprint can't fully overwrite, so it stays sub-optimal. Start FRESH in that case; once a
    # distilled champion exists, warm-starting from it compounds safely.
    distilling = config.game == "connect4" and int(config.az_distill_games) > 0
    # §C.7 capacity levers → the net architecture this run trains (legacy default unless residual/blocks set).
    net_arch = {"channels": int(config.az_channels), "blocks": int(config.az_blocks),
                "residual": bool(config.az_residual), "batchnorm": bool(config.az_batchnorm),
                "head_hidden": int(config.az_head_hidden)}
    init_net = None
    parent_gen = 0
    if config.az_warm_start and (not distilling or champions.champion_is_distilled(config.game)):
        cp = champions.best_champion_path(config.game)
        if cp:
            cand = load_net(cp, device)
            # Only warm-start when the champion's ARCHITECTURE matches this run's — else the levers would be
            # silently ignored (a legacy champion would pin the 20K-param shape). Mismatch ⇒ start fresh.
            if all(cand.arch.get(k) == net_arch[k] for k in net_arch):
                init_net = cand
                parent_gen = champions.champion_generation(config.game)
            else:
                print(f"warm-start SKIPPED: champion arch {cand.arch} != configured {net_arch} → training fresh")

    # LEAGUE opponent pool: a strong search opponent, the heuristic, a NEAR-PERFECT oracle (a perfect-play
    # reality check that punishes any residual blunder), and recent past champions. Only Connect 4 has a solver.
    pool = [lambda: MctsAgent(sims=OPPONENT_MCTS_SIMS), lambda: HeuristicAgent()]
    if config.game == "connect4":
        pool.append(lambda: NearPerfectOracle(depth=6))  # depth 6 → fast league games, still tactically perfect
    for cpath in champions.champion_pool_paths(config.game, k=3):
        cnet = load_net(cpath, device)
        pool.append(lambda cnet=cnet: AlphaZeroAgent(cnet, sims=az_sims, device=device))

    # BROAD OPENING→endgame oracle distillation — the lever that teaches the net to OPEN CENTRE and hold the
    # first-player win (a warm-started net otherwise keeps its broken edge-first opening and loses to the oracle).
    # Built ONCE and cached (FIXED seed → one anchor reused by every generation), so it costs nothing after gen 1.
    distill_corpus = None
    if config.game == "connect4" and int(config.az_distill_games) > 0:
        from harness.book import load_book
        from harness.neural import build_distill_corpus

        book = load_book(config.game)  # exact opening labels wherever the committed book reaches
        spec = {
            "games": int(config.az_distill_games),
            "seed": 0,
            "oracle_depth": 8,
            "exact_max_empty": 14,
            "book_entries": len(book),  # a grown book re-keys the cache so opening labels get more exact
            "late": {"n": int(config.az_distill_positions), "min_moves": 20},
        }
        distill_corpus = build_distill_corpus(
            game, spec, cache_dir=str(CHECKPOINT_DIR / "distill_cache"), log=print, book=book,
        )

    # §C.7 #2 ONLINE ENDGAME TABLEBASE-FROM-PLAY: a RUN-OWNED value-only store the self-play loop records/solves
    # into and reads back — off unless the switch is on AND the game exposes the exact hooks (else pure #1).
    from harness.neural import _endgame_enabled
    from harness.tablebase import Tablebase

    run_tb = None
    if int(config.az_endgame_tablebase) and _endgame_enabled(game, Tablebase(cap=int(config.az_endgame_cap))):
        run_tb = Tablebase(cap=int(config.az_endgame_cap))
        if int(config.az_endgame_warm_start):  # seed from the COMMITTED opening book (values only; never written back)
            from harness.book import load_book

            seed_book = load_book(config.game)
            for k in list(seed_book.keys()):
                pv = seed_book.proven_value(k)
                if pv is not None:
                    run_tb.put_proven(k, int(pv))

    t0 = time.perf_counter()
    net, _hist = train_alphazero(
        game,
        iterations=config.az_iterations,
        selfplay_games=config.az_selfplay_games,
        sims=az_sims,
        epochs=config.az_epochs,
        seed=config.seed,
        device=device,
        init_net=init_net,
        # PURITY: pool_frac=0 ⇒ no league games at all (net vs net only), so a solver-derived oracle opponent never
        # touches training — the honest generic self-play test. The pool is still built (cheap) but never sampled.
        opponent_pool=(pool if config.az_pool_frac > 0 else None),
        # Self-play (balanced ~50% games) is the main strength engine; the league is a MINORITY of games —
        # facing a far-stronger fixed opponent yields mostly-losing games, a weak learning signal on its own.
        pool_frac=config.az_pool_frac,
        # ORACLE DISTILLATION — supervised optimal-move targets, the biggest lever for reaching perfect play. A
        # broad cached corpus (opening→endgame) when `az_distill_games>0`; else the late-only sampled positions.
        distill_positions=config.az_distill_positions,
        distill_corpus=distill_corpus,
        # The VALIDATED GENERIC recipe (plan §C.6) — enabled via the manifest; defaults preserve the classic loop.
        gumbel=bool(config.az_gumbel),
        c_scale=config.az_c_scale,
        value_n_step=config.az_value_n_step,
        target_refresh=config.az_target_refresh,
        selfplay_opening_plies=config.az_selfplay_opening_plies,
        buffer_cap=config.az_buffer_cap,
        # §C.7 #2 online endgame loop (inert when run_tb is None / the game lacks the exact hooks).
        endgame_tb=run_tb,
        endgame_max_empty=int(config.az_endgame_max_empty),
        endgame_exact_targets=int(config.az_endgame_exact_targets),
        endgame_extend_positions=int(config.az_endgame_extend_positions),
        endgame_extend_seconds=float(config.az_endgame_extend_seconds),
        net_arch=net_arch,
        selfplay_workers=int(config.az_selfplay_workers),
        log=print,
    )
    train_seconds = time.perf_counter() - t0
    weights_path = _save_weights(config, net)

    # Meaningful yardsticks (always measured, regardless of the `opponent` lever).
    eval_rng = random.Random(config.seed + 1)
    n_eval = max(10, config.eval_games)

    # DEPLOY UNDER THE TRAINED OPERATOR (§C.7 method-bug fix): a Gumbel-trained net is measurably weaker under
    # plain PUCT, so eval + the promotion gate must build the agent with the SAME gumbel operator it was trained with.
    def _az_deploy(net) -> Agent:
        return AlphaZeroAgent(net, sims=az_sims, device=device, solve_endgame=az_solve_endgame,
                              gumbel=bool(config.az_gumbel), c_scale=float(config.az_c_scale))

    def model_factory() -> Agent:
        return _az_deploy(load_net(weights_path, device))

    vs_strong = head_to_head(game, model_factory, lambda: MctsAgent(sims=strong_sims), n_eval, eval_rng)
    az_metrics = {"win_rate_vs_strong_mcts": round(vs_strong["win_rate"], 4)}
    # Opening value-belief: the net's value on the standard opening. Connect 4 is a first-player WIN, so a
    # correctly-trained net should read ~+1 here; ~0 or negative flags the value-label contamination (weak
    # self-play teaching the opening is a draw/loss) that makes a champion forfeit the forced win.
    az_metrics["opening_value"] = round(net_value(net, game, game.initial_state(random.Random(config.seed)), device), 4)

    # PROMOTION gate: beat the incumbent champion (or be the first) to be crowned.
    incumbent = champions.best_champion_path(config.game)
    promoted = False
    if incumbent:
        inc_net = load_net(incumbent, device)
        vs_champ = head_to_head(
            game,
            model_factory,
            lambda: _az_deploy(inc_net),
            n_eval,
            eval_rng,
        )
        az_metrics["win_rate_vs_champion"] = round(vs_champ["win_rate"], 4)
        if vs_champ["win_rate"] >= PROMOTION_THRESHOLD:
            champions.promote_champion(lambda p: save_net(net, p), config.game, distilled=distilling)
            promoted = True
    else:
        champions.promote_champion(lambda p: save_net(net, p), config.game, distilled=distilling)
        promoted = True

    az_report = {
        "parent_generation": parent_gen,
        "promoted": promoted,
        "champion_generation": champions.champion_generation(config.game),
        "league_pool_size": len(pool),
        # Per-iteration curve (loss + cheap opening_value quality probe + endgame growth) — the #2-vs-#1 A/B reads this.
        "history": _hist,
    }
    # §C.7 #2 evidence: aggregate the per-iteration endgame gauges. `endgame_hit_rate` = solves AVOIDED by the memo
    # (reuse of the loop's OWN solves — a boundedness gauge, NOT a vs-#1 speedup number); `frontier_empties` = how
    # far the proven frontier climbed opening-ward (priority == empties by construction).
    eg_hist = [h for h in _hist if "endgame_total" in h]
    if run_tb is not None and eg_hist:
        solves = sum(h["endgame_solves"] for h in eg_hist)
        hits = sum(h["endgame_hits"] for h in eg_hist)
        frontier_empties = max(run_tb._p.values(), default=0)
        az_report["endgame"] = {
            "booked": len(run_tb),
            "proven_this_run": sum(h["endgame_booked"] for h in eg_hist),
            "frontier_empties": int(frontier_empties),
            "solves": solves,
            "hits": hits,
            "targets_overridden": int(config.az_endgame_exact_targets),
        }
        az_metrics["endgame_booked"] = len(run_tb)
        az_metrics["endgame_frontier_empties"] = int(frontier_empties)
        az_metrics["endgame_hit_rate"] = round(hits / (hits + solves), 4) if (hits + solves) else 0.0
        if int(config.az_endgame_persist):  # RUN-OWNED path — never the committed book artifact
            run_tb.save(str(CHECKPOINT_DIR / "endgame_runs" / f"{config.game}-{config_hash(config)}.npz"))
            az_report["endgame"]["persisted"] = f"endgame_runs/{config.game}-{config_hash(config)}.npz"
    extra_spec = {
        "az_weights": weights_path,
        "az_channels": int(config.az_channels),
        "az_sims": az_sims,
        "az_solve_endgame": int(config.az_solve_endgame),
        # Carry the DEPLOYMENT operator so the checkpoint plays under the search it was trained/measured strongest
        # with (a gumbel-trained net deployed under plain PUCT is measurably weaker — see _az_factory).
        "az_gumbel": bool(config.az_gumbel),
        "az_c_scale": config.az_c_scale,
    }
    return weights_path, extra_spec, az_report, az_metrics, train_seconds


def load_checkpoint_config(spec: dict) -> TrainerConfig:
    return TrainerConfig(game=spec["game"], model_name=spec["model_name"], mcts_sims=int(spec.get("mcts_sims", 80)))


def load_policy(checkpoint_path: str | Path) -> Callable[[Game, State, random.Random], int]:
    """Reconstruct a playable agent from a checkpoint — the seam a live test / BGA bridge drives.

    Returns `act(game, state, rng) -> action`. The agent is built ONCE (a learned core loads its weights here,
    not every move), so the same trained net plays the whole game.
    """
    spec = json.loads(Path(checkpoint_path).read_text())
    config = load_checkpoint_config(spec)
    cfg = asdict(config)
    for k in ("az_weights", "az_channels", "az_sims", "az_solve_endgame"):  # learned-core artifacts the spec carries
        if k in spec:
            cfg[k] = spec[k]
    agent: Agent = resolve_agent(config.model_name, cfg, personas_for(config.game))

    def act(game: Game, state: State, rng: random.Random) -> int:
        return agent.act(game, state, rng)

    return act


if __name__ == "__main__":
    main()
