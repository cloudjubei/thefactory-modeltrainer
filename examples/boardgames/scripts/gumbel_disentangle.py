"""DISENTANGLE net-quality from the deployment SEARCH, and find WHERE (if anywhere) the Gumbel+completed-Q arm
wins. The fair-eval A/B showed the gumbel-trained net LOSING under PUCT deployment despite a much lower training
loss and a higher oracle-match under its OWN search — so net quality and the deploy-time operator are confounded.

This scores the 2×2 {net} × {deploy search} on a SHARED oracle-match corpus, and sweeps the sim budget
∈ {2,4,8,16,32} — the completed-Q operator's guaranteed improvement is a LOW-sim effect, so if it ever helps it
should show at 2–4 sims, not 16. Plus head-to-head under a SINGLE shared search (both PUCT, both gumbel).

Reads the cached nets from `gumbel_ab_nets/` (run gumbel_ab.py first).

    PYTHONPATH=. .venv/bin/python scripts/gumbel_disentangle.py [it] [sp] [s]   # cache key: defaults 12 32 16
"""
from __future__ import annotations

import json
import os
import random
import sys

from games.connect4 import Connect4
from harness.benchmark import evaluate_optimality, sample_solvable_positions
from harness.neural import AlphaZeroAgent, head_to_head, load_net

CACHE = os.path.join(os.path.dirname(__file__), "..", "gumbel_ab_nets")


def _greedy(game, net, use_gumbel, sims):
    agent = AlphaZeroAgent(net, sims=sims, gumbel=use_gumbel, temperature=0.0)
    return lambda s: agent.act(game, s, random.Random(0))


def main() -> None:
    it = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    sp = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    s = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    gsuffix = sys.argv[4] if len(sys.argv) > 4 else ""  # e.g. "-cs0.1-ns8" to load the n-step gumbel net
    game = Connect4()
    nets = {
        "gumbel": load_net(os.path.join(CACHE, f"gumbel-it{it}-sp{sp}-s{s}-seed0{gsuffix}.pt")),
        "visit": load_net(os.path.join(CACHE, f"visitcount-it{it}-sp{sp}-s{s}-seed0.pt")),
    }
    print(f"[dis] gumbel net = gumbel-it{it}-sp{sp}-s{s}-seed0{gsuffix}.pt", flush=True)
    corpus = sample_solvable_positions(game, n=150, min_moves=16, seed=999)
    print(f"[dis] nets loaded; corpus={len(corpus)} positions (min_moves=16)", flush=True)

    # 2×2 × sim-sweep oracle-match (net closeness to the exact optimal SET), same corpus throughout.
    grid = {}
    for net_tag in ("gumbel", "visit"):
        for search in ("puct", "gumbel"):
            row = {}
            for sims in (2, 4, 8, 16, 32):
                om = evaluate_optimality(
                    game, _greedy(game, nets[net_tag], search == "gumbel", sims), states=corpus
                )["oracle_optimality_rate"]
                row[sims] = round(om, 3)
            grid[f"{net_tag}_net/{search}_search"] = row
            print(f"[dis] {net_tag} net, {search} search: " +
                  " ".join(f"n{k}={v:.3f}" for k, v in row.items()), flush=True)

    # Head-to-head under a SINGLE shared search (isolates net quality per operator).
    h2h = {}
    for search in ("puct", "gumbel"):
        use_g = search == "gumbel"
        r = head_to_head(
            game,
            lambda ug=use_g: AlphaZeroAgent(nets["gumbel"], sims=s, gumbel=ug, temperature=0.0),
            lambda ug=use_g: AlphaZeroAgent(nets["visit"], sims=s, gumbel=ug, temperature=0.0),
            n=40, rng=random.Random(7), opening_plies=2,
        )
        h2h[search] = {"gumbel_win": r["win_rate"], "draw": r["draw_rate"], "gumbel_loss": r["loss_rate"]}
        print(f"[dis] H2H gumbel-net vs visit-net under {search}: "
              f"win {r['win_rate']:.2f} / draw {r['draw_rate']:.2f} / loss {r['loss_rate']:.2f}", flush=True)

    print("\n[dis] ===== GRID =====", flush=True)
    print(json.dumps({"oracle_match_grid": grid, "head_to_head_same_search": h2h}, indent=2), flush=True)


if __name__ == "__main__":
    main()
