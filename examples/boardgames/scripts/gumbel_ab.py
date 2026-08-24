"""A/B the GENERIC learning loop: Gumbel + completed-Q policy target vs the raw visit-count target, at EQUAL
compute and with DISTILLATION OFF, so any gain is attributable to the operator (not the Connect-4-only oracle
crutch). This is the empirical gate behind plan §C.5 priority #1 — the unit test proved the operator improves a
policy at n=8 sims; this proves it trains a STRONGER net per compute.

Both arms start from a random net (no book, no distill corpus) and run the identical budget; only `gumbel`
differs. Metrics are chosen to DISCRIMINATE weak-but-different nets (a near-perfect oracle reference floors both
at zero and teaches nothing):
  - HEAD-TO-HEAD (the decisive A/B, solver-free): the gumbel net vs the visit-count net directly, seat-alternated
    with random openings. >0.5 for gumbel = it trained stronger at equal compute.
  - ORACLE-MATCH (graded distance-to-optimal): fraction of moves in the exact optimal set over a SHARED corpus.
  - SIM-SCALING vs a frozen PEER net (strength-per-compute, non-zero): conversion at sims ∈ {4,16} vs a fixed
    random-init net.

Trained nets are cached under `gumbel_ab_nets/` so the eval can be re-run without retraining.

    PYTHONPATH=. .venv/bin/python scripts/gumbel_ab.py [iterations] [selfplay_games] [sims]
"""
from __future__ import annotations

import json
import os
import sys
import time

from games.connect4 import Connect4
from harness.benchmark import evaluate_optimality, sample_solvable_positions, sim_scaling_curve
from harness.neural import AlphaZeroAgent, Connect4Net, head_to_head, load_net, save_net, train_alphazero

CACHE = os.path.join(os.path.dirname(__file__), "..", "gumbel_ab_nets")


def _train_or_load(game, gumbel, iterations, selfplay_games, sims, seed, c_scale, n_step, reanalyze, log):
    os.makedirs(CACHE, exist_ok=True)
    tag = "gumbel" if gumbel else "visitcount"
    suffix = ((f"-cs{c_scale}" if (gumbel and c_scale != 1.0) else "") + (f"-ns{n_step}" if n_step else "")
              + (f"-rz{reanalyze}" if reanalyze else ""))
    path = os.path.join(CACHE, f"{tag}-it{iterations}-sp{selfplay_games}-s{sims}-seed{seed}{suffix}.pt")
    if os.path.isfile(path):
        log(f"{tag}: cache hit {os.path.basename(path)}")
        return load_net(path), 0.0, None
    t0 = time.perf_counter()
    net, hist = train_alphazero(
        game, iterations=iterations, selfplay_games=selfplay_games, sims=sims, epochs=4,
        channels=32, buffer_cap=8000, seed=seed, augment=True, gumbel=gumbel, c_scale=c_scale,
        value_n_step=n_step, target_refresh=2, reanalyze_frac=reanalyze,
        distill_positions=0, distill_corpus=None, book=None,  # DISTILLATION OFF — the generic loop only
        log=log,
    )
    save_net(net, path)
    return net, round(time.perf_counter() - t0, 1), hist[-1]["loss"]


def main() -> None:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    selfplay_games = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    c_scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    n_step = int(sys.argv[5]) if len(sys.argv) > 5 else 0  # n-step value target for the GUMBEL arm (0 = raw MC)
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0    # replication seed (both arms + eval share it)
    reanalyze = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0  # reanalyze_frac for the GUMBEL arm (0 = off)
    game = Connect4()
    print(f"[ab] iterations={iterations} selfplay_games={selfplay_games} sims={sims} c_scale={c_scale} "
          f"(distillation OFF)", flush=True)

    nets, meta = {}, {}
    for gumbel in (False, True):
        tag = "gumbel" if gumbel else "visitcount"
        cs = c_scale if gumbel else 1.0
        ns = n_step if gumbel else 0  # n-step applies to the gumbel arm; visit-count stays the MC baseline
        rz = reanalyze if gumbel else 0.0  # reanalyze applies to the gumbel arm
        print(f"\n[ab] === train arm: {tag} (c_scale={cs}, n_step={ns}, reanalyze={rz}, seed={seed}) ===", flush=True)
        nets[tag], secs, loss = _train_or_load(game, gumbel, iterations, selfplay_games, sims, seed, cs, ns, rz,
                                               log=lambda m, t=tag: print(f"[{t}] {m}", flush=True))
        meta[tag] = {"train_seconds": secs, "final_loss": loss}

    print("\n[ab] === eval ===", flush=True)
    # DEPLOY BOTH NETS IDENTICALLY: greedy PUCT+argmax (gumbel=False), so the comparison isolates NET quality,
    # not eval-time search noise. (Gumbel exploration is a training device; at deployment the trained net is what
    # matters, played under the same standard search for both arms.) Shared corpus + frozen peer reference.
    corpus = sample_solvable_positions(game, n=150, min_moves=16, seed=999)
    ref_net = Connect4Net()  # a fixed random-init peer (seed-fixed by torch default state at import)

    def greedy(net):
        agent = AlphaZeroAgent(net, sims=sims, gumbel=False, temperature=0.0)
        return lambda s: agent.act(game, s, __import__("random").Random(0))

    results = {}
    for gumbel in (False, True):
        tag = "gumbel" if gumbel else "visitcount"
        om = evaluate_optimality(game, greedy(nets[tag]), states=corpus)["oracle_optimality_rate"]
        curve = sim_scaling_curve(
            game, lambda s, t=tag: AlphaZeroAgent(nets[t], sims=s, gumbel=False, temperature=0.0),
            lambda: AlphaZeroAgent(ref_net, sims=8, gumbel=False, temperature=0.0),
            sims_list=(4, 16), games=16, seed=123,
        )
        results[tag] = {**meta[tag], "oracle_match": round(om, 3), "peer_auc": round(curve["auc"], 3),
                        "peer_curve": curve["points"]}
        print(f"[ab] {tag}: oracle_match={om:.3f} peer_auc={curve['auc']:.3f} loss={meta[tag]['final_loss']}", flush=True)

    import random as _r
    h2h = head_to_head(game, lambda: AlphaZeroAgent(nets["gumbel"], sims=sims, gumbel=False, temperature=0.0),
                       lambda: AlphaZeroAgent(nets["visitcount"], sims=sims, gumbel=False, temperature=0.0),
                       n=40, rng=_r.Random(7), opening_plies=2)

    print("\n[ab] ===== SUMMARY (distillation OFF, equal compute) =====", flush=True)
    print(json.dumps({"config": {"iterations": iterations, "selfplay_games": selfplay_games, "sims": sims},
                      "arms": results, "head_to_head_gumbel_vs_visitcount": h2h}, indent=2), flush=True)
    gm, vm = results["gumbel"]["oracle_match"], results["visitcount"]["oracle_match"]
    print(f"\n[ab] HEAD-TO-HEAD gumbel vs visit-count: win {h2h['win_rate']:.2f} / draw {h2h['draw_rate']:.2f} / "
          f"loss {h2h['loss_rate']:.2f}", flush=True)
    print(f"[ab] ORACLE-MATCH gumbel {gm:.3f} vs visit-count {vm:.3f} (Δ {gm - vm:+.3f})", flush=True)
    verdict = "GUMBEL WINS" if (h2h["win_rate"] > h2h["loss_rate"] and gm >= vm) else (
        "MIXED" if h2h["win_rate"] > h2h["loss_rate"] or gm > vm else "no gain")
    print(f"[ab] VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
