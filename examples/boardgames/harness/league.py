"""§C.7 #3 — SOLVER-FREE league opponents for breaking opening value-collapse WITHOUT an oracle. The learner plays
weak→strong solver-free opponents (random, heuristic, pure-UCT MCTS rungs, and its OWN past arch-matched snapshots)
from the TRUE opening, so game OUTCOMES label the opening +1 — distillation's effect reproduced from play, no
oracle/book/solve_endgame anywhere. The legacy champion store (arch-mismatched 20k nets) is NEVER used here."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from harness.agents import HeuristicAgent, MctsAgent, RandomAgent


def select_snapshot_spread(completed_batches: list[int], k: int) -> list[int]:
    """≤k batch indices spread WEAK→RECENT, always keeping the earliest and latest completed batch (the ladder spans
    a weak early net to the strongest recent one). Fewer than k available ⇒ all of them."""
    xs = sorted({int(b) for b in completed_batches})
    if k <= 0 or not xs:
        return []
    if len(xs) <= k:
        return xs
    if k == 1:
        return [xs[-1]]
    idxs = sorted({round(i * (len(xs) - 1) / (k - 1)) for i in range(k)})  # evenly spaced, both ends inclusive
    return [xs[i] for i in idxs]


def build_solver_free_pool(game, run_dir, completed_batches, arch, cfg: dict) -> list[Callable[[], object]]:
    """League opponent factories — ALL solver-free (never NearPerfectOracle/book/solve_endgame>0). Snapshot nets are
    the RUN'S OWN arch-matched ckpt_*.pt, loaded ONCE and closed over so a factory never re-reads disk per game."""
    pool: list[Callable[[], object]] = []
    if cfg.get("include_random", True):
        pool.append(lambda: RandomAgent())
    if cfg.get("include_heuristic", True) and getattr(game, "heuristic_action", None) is not None:
        pool.append(lambda: HeuristicAgent())  # genericity gate: skip where the game exposes no heuristic
    for s in cfg.get("mcts_sims", [30, 120]):
        si = int(s)
        pool.append(lambda si=si: MctsAgent(sims=si, solve_endgame=0, book=None))  # pure UCT, NO solver leaves
    snaps = select_snapshot_spread(list(completed_batches), int(cfg.get("snapshots", 0)))
    if snaps:
        from harness.neural import AlphaZeroAgent, load_net

        ssims = int(cfg.get("snapshot_sims", 96))
        gumbel = bool(cfg.get("gumbel", True))
        cs = float(cfg.get("c_scale", 0.1))
        rd = Path(run_dir)
        for b in snaps:
            p = rd / f"ckpt_{b}.pt"
            if p.is_file():
                net = load_net(str(p))  # arch-matched (save_net/load_net round-trip the arch); loaded ONCE
                pool.append(
                    lambda net=net, ssims=ssims, gumbel=gumbel, cs=cs: AlphaZeroAgent(
                        net, sims=ssims, gumbel=gumbel, c_scale=cs, solve_endgame=0
                    )
                )
    return pool
