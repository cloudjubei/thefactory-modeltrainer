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
from harness.config import OPPONENT_MCTS_SIMS, TrainerConfig, config_hash, load_config, load_eval_config
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
    write_summary(summary, summary_out)
    print(f"win_rate={ev.win_rate:.3f} vs {config.opponent} cost=${summary['cost']['estCostUsd']:.6f} summary={summary_out}")


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
    from harness.neural import AlphaZeroAgent, head_to_head, load_net, save_net, train_alphazero

    device = "cpu"
    az_sims = int(config.az_sims)
    strong_sims = max(az_sims, OPPONENT_MCTS_SIMS)

    # WARM-START from the strongest saved net (cumulative), unless the run asks to start fresh.
    init_net = None
    parent_gen = 0
    if config.az_warm_start:
        cp = champions.best_champion_path(config.game)
        if cp:
            init_net = load_net(cp, device)
            parent_gen = champions.champion_generation(config.game)

    # LEAGUE opponent pool: a strong search opponent, the heuristic, and recent past champions.
    pool = [lambda: MctsAgent(sims=OPPONENT_MCTS_SIMS), lambda: HeuristicAgent()]
    for cpath in champions.champion_pool_paths(config.game, k=3):
        cnet = load_net(cpath, device)
        pool.append(lambda cnet=cnet: AlphaZeroAgent(cnet, sims=az_sims, device=device))

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
        opponent_pool=pool,
        # Self-play (balanced ~50% games) is the main strength engine; the league is a MINORITY of games —
        # facing a far-stronger fixed mcts yields mostly-losing games, a weak learning signal on its own.
        pool_frac=0.35,
        log=print,
    )
    train_seconds = time.perf_counter() - t0
    weights_path = _save_weights(config, net)

    # Meaningful yardsticks (always measured, regardless of the `opponent` lever).
    eval_rng = random.Random(config.seed + 1)
    n_eval = max(10, config.eval_games)

    def model_factory() -> Agent:
        return AlphaZeroAgent(load_net(weights_path, device), sims=az_sims, device=device)

    vs_strong = head_to_head(game, model_factory, lambda: MctsAgent(sims=strong_sims), n_eval, eval_rng)
    az_metrics = {"win_rate_vs_strong_mcts": round(vs_strong["win_rate"], 4)}

    # PROMOTION gate: beat the incumbent champion (or be the first) to be crowned.
    incumbent = champions.best_champion_path(config.game)
    promoted = False
    if incumbent:
        inc_net = load_net(incumbent, device)
        vs_champ = head_to_head(
            game, model_factory, lambda: AlphaZeroAgent(inc_net, sims=az_sims, device=device), n_eval, eval_rng
        )
        az_metrics["win_rate_vs_champion"] = round(vs_champ["win_rate"], 4)
        if vs_champ["win_rate"] >= PROMOTION_THRESHOLD:
            champions.promote_champion(lambda p: save_net(net, p), config.game)
            promoted = True
    else:
        champions.promote_champion(lambda p: save_net(net, p), config.game)
        promoted = True

    az_report = {
        "parent_generation": parent_gen,
        "promoted": promoted,
        "champion_generation": champions.champion_generation(config.game),
        "league_pool_size": len(pool),
    }
    extra_spec = {"az_weights": weights_path, "az_channels": 32, "az_sims": az_sims}
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
    for k in ("az_weights", "az_channels", "az_sims"):  # learned-core artifacts the spec carries
        if k in spec:
            cfg[k] = spec[k]
    agent: Agent = resolve_agent(config.model_name, cfg, personas_for(config.game))

    def act(game: Game, state: State, rng: random.Random) -> int:
        return agent.act(game, state, rng)

    return act


if __name__ == "__main__":
    main()
