"""SCALE the VALIDATED generic recipe (Gumbel/completed-Q search + calibrated completed-Q target c_scale=0.1 +
n-step value n=8/k=2, distillation OFF) and measure it on the REAL near-optimal audit — the depth-6→8→10
conversion LADDER as first player vs NearPerfectOracle — not just mid-game oracle-match. This is the test of
whether the generic loop, scaled, pushes past the champion's depth-6 plateau toward a near-optimal MODEL.

Deploys the net under its best search (Gumbel, greedy) with the exact-endgame cutoff on (a legitimate generic
deployment optimisation). Prints the ladder frontier + each rung's P1-conversion + the mid-game oracle-match.

    PYTHONPATH=. .venv/bin/python scripts/scale_up.py [iterations] [selfplay_games] [sims] [n_step]
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

from games.connect4 import Connect4
from harness.benchmark import evaluate_optimality, optimality_ladder, sample_solvable_positions
from harness.neural import AlphaZeroAgent, load_net, save_net, train_alphazero

CACHE = os.path.join(os.path.dirname(__file__), "..", "gumbel_ab_nets")


def main() -> None:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    selfplay_games = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    n_step = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    game = Connect4()
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"scaleup-it{iterations}-sp{selfplay_games}-s{sims}-ns{n_step}.pt")

    if os.path.isfile(path):
        print(f"[scale] cache hit {os.path.basename(path)}", flush=True)
        net = load_net(path)
    else:
        print(f"[scale] TRAIN validated recipe: it={iterations} sp={selfplay_games} sims={sims} "
              f"n_step={n_step} (gumbel + c_scale=0.1 + n-step, distill OFF)", flush=True)
        t0 = time.perf_counter()
        net, hist = train_alphazero(
            game, iterations=iterations, selfplay_games=selfplay_games, sims=sims, epochs=4, channels=32,
            buffer_cap=16000, seed=0, augment=True, gumbel=True, c_scale=0.1, value_n_step=n_step, target_refresh=2,
            distill_positions=0, distill_corpus=None, book=None,
            log=lambda m: print(f"[scale] {m}", flush=True),
        )
        save_net(net, path)
        print(f"[scale] trained in {time.perf_counter() - t0:.0f}s, final loss {hist[-1]['loss']:.3f}", flush=True)

    # Deploy under the Gumbel search + exact-endgame cutoff (the model's best generic deployment).
    def model():
        return AlphaZeroAgent(net, sims=32, gumbel=True, temperature=0.0, solve_endgame=22)

    print("[scale] --- depth-conversion LADDER (P1 vs NearPerfectOracle) ---", flush=True)
    lad = optimality_ladder(game, model, games=20, seed=123, depths=(6, 8, 10), include_exact=False)
    for r in lad["rungs"]:
        print(f"[scale]   {r['label']}: P1-conversion {r['rate']:.2f} ({'CLEARED' if r['cleared'] else 'no'})", flush=True)
    print(f"[scale] FRONTIER = {lad['frontier']}", flush=True)

    corpus = sample_solvable_positions(game, n=150, min_moves=16, seed=999)
    om = evaluate_optimality(game, lambda s: AlphaZeroAgent(net, sims=16, gumbel=False, temperature=0.0)
                             .act(game, s, random.Random(0)), states=corpus)["oracle_optimality_rate"]
    print(f"[scale] mid-game oracle-match (PUCT n16) = {om:.3f}", flush=True)
    print("[scale] " + json.dumps({"frontier": lad["frontier"], "rungs": lad["rungs"], "oracle_match": round(om, 3)}),
          flush=True)


if __name__ == "__main__":
    main()
