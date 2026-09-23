"""§C.20 CROSS-GAME REGRESSION MATRIX — run the SAME change across every encoded game and read whether it
GENERALISES. Rows = a change, columns = games, cells = a paired score delta; the verdict (generalises / local /
contested / null / inconclusive / single_game) is derived with the right bar per question (harness.matrix): the
UNION question "did any game move" takes Bonferroni alpha/N; the CONJUNCTION "did every game move" is an
intersection-union test at alpha.

The unit of replication is the TRAINING RUN, not the opening (§C.19): a cell is built from the per-SEED paired
deltas across several training seeds, tested between seeds — so its significance answers "does the lever move
NETS", not "did these two particular nets differ on the openings" (the single-net confound §C.20 exists to kill).
One seed is allowed but returns a non-significant cell with a standing caveat.

  sweep    — train BASELINE and VARIANT nets (one lever apart) for each game and each training seed, measure the
             paired exploitability delta vs the game-agnostic UCT refuter on a FIXED opening set, and assemble.
      PYTHONPATH=. .venv/bin/python scripts/cross_game_matrix.py sweep --games connect4,tictactoe \\
          --lever augment --seeds 7,99,131 --iterations 1 --selfplay-games 8 --sims 8 --n-openings 16

  assemble — build the matrix from comparisons ALREADY in a ledger (rigorous A/Bs). --spec is JSON:
             {change: {game: [arm_a, arm_b, n]}}. A pure read: it does not mutate the ledger.
"""
from __future__ import annotations

import argparse
import json
import sys

from harness.matrix import build_matrix, cell_from_compare, cell_from_seed_deltas, is_saturated, render_matrix
from harness.measurement import MEASUREMENT_SEEDS
from harness.registry import resolve_game

# a lever is a pair of kwarg-overrides to train_alphazero: (baseline, variant), plus the eval-time operator each
# arm should be DEPLOYED under (so a train-time operator change is also deployed that way, not forced constant).
LEVERS = {
    "augment": (({"augment": False}, {"augment": True}), (True, True)),
    "width": (({}, {}), (True, True)),            # widths come from --channels (baseline) vs 2x (variant)
    "gumbel": (({"gumbel": False}, {"gumbel": True}), (False, True)),
}


def _train(game, base_kwargs, override, seed):
    from harness.neural import train_alphazero

    net, _ = train_alphazero(game, seed=seed, **{**base_kwargs, **override})
    return net


def _arm_factory(net, sims, gumbel):
    from harness.neural import AlphaZeroAgent

    return lambda: AlphaZeroAgent(net, sims=sims, solve_endgame=0, gumbel=gumbel, c_scale=0.1)


def _measure_delta(game, base_net, var_net, eval_gumbel, args):
    """One training seed's paired (variant - baseline) held-rate delta on the FIXED opening set, plus the two
    held rates (for a saturation check)."""
    from harness.agents import MctsAgent
    from harness.benchmark import paired_exploitability

    gb, gv = eval_gumbel
    res = paired_exploitability(
        game, {"variant": _arm_factory(var_net, args.eval_sims, gv),
               "baseline": _arm_factory(base_net, args.eval_sims, gb)},
        depths=[args.depth], n_openings=args.n_openings, opening_plies=args.opening_plies,
        seed=args.measure_seed,
        refuter_factory=lambda d: MctsAgent(sims=d, solve_endgame=0, book=None))
    row = res["by_depth"][0]
    hv = [1 - o for o in row["arms"]["variant"]["outcomes"]]
    hb = [1 - o for o in row["arms"]["baseline"]["outcomes"]]
    rate_v, rate_b = sum(hv) / len(hv), sum(hb) / len(hb)
    return rate_v - rate_b, rate_v, rate_b


def _sweep_caveats(args, rates_v, rates_b, n_seeds) -> dict:
    cav = {}
    if n_seeds < 2:
        cav["provenance"] = "single training draw — training-seed variance uncancelled"
    if args.iterations < 2 or args.n_openings < 32:
        cav["budget"] = (f"SMOKE SCALE (iterations={args.iterations}, n_openings={args.n_openings}, "
                         f"sims={args.sims}) — underpowered; not a rigorous read")
    if is_saturated(rates_v + rates_b):
        cav["completeness"] = ("SATURATED — both arms at the floor/ceiling vs the refuter, so the cell cannot "
                               "discriminate; raise training or lower the refuter depth")
    return cav


def sweep(args) -> dict:
    games = [(name, resolve_game(name)) for name in args.games.split(",")]   # resolve ALL up front (fail fast)
    (base_over, var_over), eval_gumbel = LEVERS[args.lever]
    if args.lever == "width":
        base_over, var_over = {"channels": args.channels}, {"channels": 2 * args.channels}
    base_kwargs = dict(iterations=args.iterations, selfplay_games=args.selfplay_games, sims=args.sims,
                       channels=args.channels, epochs=args.epochs)
    seeds = [int(s) for s in args.seeds.split(",")]
    cells = []
    for name, game in games:
        try:
            deltas, rates_v, rates_b = [], [], []
            for s in seeds:
                bn = _train(game, base_kwargs, base_over, s)
                vn = _train(game, base_kwargs, var_over, s)
                d, rv, rb = _measure_delta(game, bn, vn, eval_gumbel, args)
                deltas.append(d)
                rates_v.append(rv)
                rates_b.append(rb)
                print(f"[{name}] seed {s}: variant {rv:.3f} - baseline {rb:.3f} = {d:+.3f}", flush=True)
            cell = cell_from_seed_deltas(name, deltas)
            cell["caveats"] = {**cell.get("caveats", {}), **_sweep_caveats(args, rates_v, rates_b, len(seeds))}
            cell["per_seed_deltas"] = deltas
            print(f"[{name}] pooled {cell['diff']:+.3f}  p={cell['p']:.4f}  ({len(seeds)} seeds)", flush=True)
        except Exception as exc:                        # one game must not waste the others
            print(f"[{name}] ERROR: {exc}", flush=True)
            cell = {"game": name, "diff": 0.0, "p": 1.0, "n": 0, "ci": (-1.0, 1.0), "null_below": 0.03,
                    "caveats": {"error": f"{type(exc).__name__}: {exc}"}}
        cells.append(cell)
        if args.out:                       # persist the partial matrix after EACH game (survives a crash/sleep)
            partial = build_matrix({f"{args.lever} (seeds={len(seeds)}, n_openings={args.n_openings})": cells})
            json.dump({"games_done": [c["game"] for c in cells], "cells": cells, "matrix": partial},
                      open(args.out, "w"), indent=1, default=list)
    return build_matrix({f"{args.lever} (seeds={len(seeds)}, n_openings={args.n_openings})": cells})


def assemble(args) -> dict:
    from harness.ledger import Ledger

    led = Ledger(args.ledger)
    spec = json.loads(open(args.spec).read())
    rows = {}
    for change, games in spec.items():
        cells = []
        for game, triple in games.items():
            arm_a, arm_b, n = triple
            try:
                res = led.compare(arm_a, arm_b, treatment=args.treatment, record=False)   # PURE READ
                cells.append(cell_from_compare(res, game, int(n)))
            except (ValueError, KeyError) as exc:
                cells.append({"game": game, "diff": 0.0, "p": 1.0, "n": int(n), "ci": (-1.0, 1.0),
                              "null_below": 0.03, "caveats": {"error": str(exc)}})
        rows[change] = cells
    return build_matrix(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("sweep")
    sw.add_argument("--games", required=True, help="comma-separated game names")
    sw.add_argument("--lever", required=True, choices=sorted(LEVERS))
    sw.add_argument("--seeds", default="7", help="comma-separated TRAINING seeds; >=2 for a real between-run read")
    sw.add_argument("--iterations", type=int, default=1)
    sw.add_argument("--selfplay-games", type=int, default=8)
    sw.add_argument("--sims", type=int, default=8, help="training search budget")
    sw.add_argument("--eval-sims", type=int, default=8, help="the arms' search budget at measurement")
    sw.add_argument("--epochs", type=int, default=4)
    sw.add_argument("--channels", type=int, default=8)
    sw.add_argument("--n-openings", type=int, default=16)
    sw.add_argument("--opening-plies", type=int, default=2)
    sw.add_argument("--depth", type=int, default=20, help="the single UCT refuter depth")
    sw.add_argument("--measure-seed", type=int, default=131,
                    help=f"a MEASUREMENT seed {sorted(MEASUREMENT_SEEDS)}, FIXED across training seeds")
    sw.add_argument("--out", default="", help="write the partial matrix JSON after each game (crash/sleep safe)")
    asm = sub.add_parser("assemble")
    asm.add_argument("--spec", required=True)
    asm.add_argument("--ledger", default="checkpoints/scaled_runs/analysis_ledger.json")
    asm.add_argument("--treatment", default="config", choices=("config", "code", "budget"))
    args = ap.parse_args()

    if args.cmd == "sweep" and args.measure_seed not in MEASUREMENT_SEEDS:
        raise SystemExit(f"--measure-seed {args.measure_seed} is not a MEASUREMENT seed {sorted(MEASUREMENT_SEEDS)}")
    matrix = sweep(args) if args.cmd == "sweep" else assemble(args)
    print("\n" + render_matrix(matrix))
    for name, row in matrix["rows"].items():
        extra = f" [{row['sub_status']}]" if row.get("sub_status") else ""
        print(f"\n{name}: {row['status'].upper()}{row['direction'] if row['direction'] in '+-' else ''}{extra} "
              f"({row['n_significant']}/{row['n_games']} games clear the union bar alpha/{row['n_games']})")
        if row["caveated_games"]:
            print(f"  ⚠ caveated: {', '.join(row['caveated_games'])}")


if __name__ == "__main__":
    main()
