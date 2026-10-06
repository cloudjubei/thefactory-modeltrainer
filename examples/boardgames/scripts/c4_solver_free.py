"""§C.49 T10 — the SOLVER-FREE Connect-4 run, then the solver's verdict on the net it stopped at.

Training: T9's process (h73/h74: unique buffer, whole-buffer 200-sim relabel and the strategy-tree walk instead of
siblings — first player, through --tree-depth plies), at the depth-10 oracle setup (standardised-input conv-32,
20,616 params, h72). It runs inside `forbid_solver`, so any exact solve raises, with relabelling spread over --workers
processes (bit-identical to serial), and STOPS at the first walk that reads zero disagreements (the h69/h74 signal),
returning the net that made that walk. A seed that never reads zero runs to the --iterations cap and settles.

Certification, after all seeds have trained: the solver walks each seed's final net's own first-player tree
through --certify-depth plies against every reply (harness.certify, P-START). Values come from the exact solves
already recorded in the label cache where it has them, and from the native solver otherwise; the evidence counts
both.

    PYTHONPATH=. .venv/bin/python scripts/c4_solver_free.py --seeds 361 362 --iterations 60 --workers 8 \\
        --out evidence/c49_T10_solver_free.json.gz

`--value-delta` makes the stop value-aware (T12, harness.floor_c4_value): a raw move also agrees when the search's
Q for it is within that much of the label's top move. `--tree-value-target` gives the strategy-tree positions a value
target, `--value-n-step` / `--target-refresh` bootstrap the value targets from a lagged target net, and `--no-stop`
trains to the cap whatever the walk reads (T14, harness.floor_c4_value_signal). `--backplay FRAC RAMP` starts that
share of the games near the end of the last iteration's games (T16, harness.floor_c4_backplay). `--exploiter FRAC
FACTOR` plays that share of the games against the current net with FACTOR times the search (T19,
harness.floor_c4_exploiter). `--curve` scores the net on the fixed position set (harness.fixed_set) after every
training pass and records the readings as `curve` (T20, harness.curve_diagnosis). `--opening-table HORIZON` builds
the hybrid's exact opening table (harness.opening_table.full_table) before training — outside the solver guard, it is
the process's declared exact step — then starts every self-play game at its frontier and walks the strategy tree from
there, so the net trains only where the table stops (H2, harness.floor_h2). With `--curve` it also scores the net
after every pass at the positions one ply past the table, against exact values solved before training
(`carried_curve`, H3, harness.floor_h3).
"""
from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NETS = ROOT / "checkpoints" / "c49_sf"
LABELS = ROOT / "books" / "c4_labels.json.gz"
MEASUREMENT_MODULES = ("scripts/c4_solver_free.py", "harness/certify.py", "harness/native_solver.py",
                       "harness/transfer.py", "harness/fixed_set.py", "harness/opening_table.py",
                       "harness/exact_values.py")
EMPTY_BOARD_VALUE = 1
EMPTY_BOARD_VALUE_SOURCE = "Connect-4 is a first-player win (Allis 1988; Allen 1988; Tromp's database)"
_BOOK = None


def _solve(job: tuple) -> int:
    global _BOOK
    from games.connect4 import C4State
    from harness import native_solver
    from harness.book import load_book

    if _BOOK is None:
        _BOOK = load_book("connect4")
    board, to_move = job
    return native_solver.solve_position(C4State(tuple(board), to_move, None, False), book=_BOOK)


def _recorded_values(game) -> dict:
    """{position key: exact value to the side to move} for every position one move after a label-cache position:
    the cache stores, per position, the value each move keeps for the mover."""
    from games.connect4 import C4State
    from harness.evidence import load_evidence

    out = {}
    if not LABELS.exists():
        return out
    for row in load_evidence(LABELS)["positions"]:
        parent = C4State(tuple(row["board"]), row["to_move"], None, False)
        for a, v in row["values"].items():
            child = game.step(parent, int(a))
            if not game.is_terminal(child):
                out[game.state_key(child)] = -int(v)
    return out


def _train(cfg: dict, seed: int, threads: int, nets: Path, curve_cases: list | None = None,
           starts: list | None = None, carried_cases: list | None = None) -> dict:
    import torch

    from harness.neural import save_net, train_alphazero
    from harness.registry import resolve_game
    from harness.targets import _weights_sha
    from harness.transfer import forbid_solver, record_selfplay_states

    torch.set_num_threads(threads)
    game = resolve_game(cfg["game"])
    t0 = time.time()
    def on_pass(net):
        seconds = round(time.time() - t0, 1)
        if curve_cases is None:
            return seconds
        from harness.fixed_set import score
        from harness.strategy_tree import raw_chooser

        choose = raw_chooser(game, net)
        reading = {"seconds": seconds, **score(choose([s for s, _p, _v in curve_cases]), curve_cases)}
        if carried_cases:
            reading["carried"] = score(choose([s for s, _p, _v in carried_cases]), carried_cases)
        return reading

    with forbid_solver(), record_selfplay_states(game, on_pass=on_pass) as log:
        net, history = train_alphazero(
            game, iterations=cfg["iterations"], selfplay_games=cfg["selfplay"], sims=cfg["train_sims"],
            channels=cfg["arch"]["channels"], net_arch=cfg["arch"], augment=cfg["augment"], gumbel=cfg["gumbel"],
            c_scale=cfg["c_scale"], seed=seed, selfplay_opening_plies=cfg["opening_plies"],
            opening_plies_zero_frac=cfg["opening_zero_frac"], epochs=cfg["epochs"], batch_size=cfg["batch_size"],
            lr=cfg["lr"], buffer_cap=cfg["buffer_cap"], reanalyze_frac=cfg["reanalyze_frac"],
            reanalyze_sims=cfg["reanalyze_sims"], reanalyze_siblings=cfg["reanalyze_siblings"],
            steps_matched=cfg["steps_matched"], buffer_unique=cfg["buffer_unique"],
            settle_epochs=cfg["settle_epochs"], settle_lr_final=cfg["settle_lr_final"],
            strategy_tree=cfg["strategy_tree"], relabel_workers=cfg["relabel_workers"],
            stop_on_agreement=cfg["stop_on_agreement"], stop_value_delta=cfg.get("stop_value_delta"),
            tree_value_target=cfg.get("tree_value_target", False), value_n_step=cfg.get("value_n_step", 0),
            target_refresh=cfg.get("target_refresh", 4), backplay=cfg.get("backplay"),
            exploiter=cfg.get("exploiter"), selfplay_starts=starts, tree_roots=starts)
    path = nets / f"seed_{seed}.pt"
    save_net(net, str(path))
    stopped = bool(history and history[-1].get("stopped"))
    print(f"seed {seed}: {'STOPPED at iteration ' + str(history[-1]['iteration']) if stopped else 'not stopped'}; "
          f"disagreements {[h.get('tree_disagreements') for h in history if 'tree_walked' in h]} "
          f"[{time.time() - t0:.0f}s]", flush=True)
    return {"seed": seed, "stopped": stopped,
            "net": {"path": str(path.relative_to(ROOT)), "weights_sha": _weights_sha(net),
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            "params": sum(p.numel() for p in net.parameters()),
            "pass_seconds": [r["seconds"] for r in log["on_pass"]] if curve_cases else log["on_pass"],
            **({"curve": [{k: v for k, v in r.items() if k not in ("seconds", "carried")} for r in log["on_pass"]]}
               if curve_cases else {}),
            **({"carried_curve": [r["carried"] for r in log["on_pass"]]} if carried_cases else {}),
            "games_per_pass": [sum(1 for g in log["games"] if g["pass"] == p) for p in range(log["passes"])],
            "train_seconds": round(time.time() - t0, 1), "history": history}


def _certify(net_path: Path, depth: int, recorded: dict, pool, game) -> dict:
    import random

    from harness.certify import certify
    from harness.neural import load_net
    from harness.strategy_tree import raw_chooser

    net = load_net(str(net_path))
    counts = {"recorded": 0, "solved": 0}
    t0 = time.time()

    def many(states: list) -> list:
        out = [recorded.get(game.state_key(s)) for s in states]
        todo = [i for i, v in enumerate(out) if v is None]
        for i, v in zip(todo, pool.map(_solve, [(states[i].board, states[i].to_move) for i in todo], chunksize=4)):
            out[i] = v
        counts["recorded"] += len(states) - len(todo)
        counts["solved"] += len(todo)
        return out

    result = certify(game, game.initial_state(random.Random(0)), 0, raw_chooser(game, net),
                     lambda s: many([s])[0], max_depth=depth, value_many=many, root_value=EMPTY_BOARD_VALUE)
    return {"certified": result["certified"], "horizon": result["horizon"], "complete": result["complete"],
            "failures": result["failures"], "failures_by_ply": result["failures_by_ply"], "nodes": result["nodes"],
            "values": counts, "seconds": round(time.time() - t0, 1),
            "net_file_sha256": hashlib.sha256(net_path.read_bytes()).hexdigest()}


def main() -> None:
    import platform

    import torch

    from games.connect4 import Connect4
    from harness import native_solver
    from harness.evidence import save_evidence
    from harness.fingerprint import training_fingerprint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", default=None, metavar="MODULE:ARM",
                    help="train exactly a registered recipe (harness.recipe.load_recipe): its config and seeds, with "
                         "every other training flag refused (h138)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--iterations", type=int, default=None, help="the cap; a run that stops on agreement ends sooner")
    ap.add_argument("--tree-depth", type=int, default=None, help="default 10")
    ap.add_argument("--certify-depth", type=int, default=10)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--value-delta", type=float, default=None,
                    help="stop on the value-aware reading: a move also agrees within this Q of the top move (T12)")
    ap.add_argument("--tree-value-target", action="store_true",
                    help="give the strategy-tree positions the relabel search's root value as their value target (T14)")
    ap.add_argument("--no-stop", action="store_true", help="train every iteration and settle, whatever the walk reads")
    ap.add_argument("--value-n-step", type=int, default=0,
                    help="bootstrap value targets from a lagged target net this many moves ahead (T14)")
    ap.add_argument("--target-refresh", type=int, default=None,
                    help="refresh that target net every this many iterations")
    ap.add_argument("--backplay", type=float, nargs=2, metavar=("FRAC", "RAMP"), default=None,
                    help="start FRAC of the games near the end of last iteration's games, reaching whole games after "
                         "RAMP iterations (T16)")
    ap.add_argument("--exploiter", type=float, nargs=2, metavar=("FRAC", "FACTOR"), default=None,
                    help="play FRAC of the games against the current net with FACTOR times the search, greedily (T19)")
    ap.add_argument("--opening-table", type=int, default=None, metavar="HORIZON",
                    help="the hybrid's exact opening table before HORIZON plies: self-play starts at its frontier and "
                         "the strategy tree is walked from there (H2)")
    ap.add_argument("--curve", action="store_true",
                    help="score the net on the fixed position set after every training pass (T20)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from harness.floor_c4 import CONFIG

    if args.recipe:
        from harness.recipe import load_recipe

        hand_set = [flag for flag, value in (("--seeds", args.seeds), ("--iterations", args.iterations),
                                             ("--tree-depth", args.tree_depth), ("--value-delta", args.value_delta),
                                             ("--tree-value-target", args.tree_value_target),
                                             ("--no-stop", args.no_stop), ("--value-n-step", args.value_n_step),
                                             ("--target-refresh", args.target_refresh), ("--backplay", args.backplay),
                                             ("--exploiter", args.exploiter),
                                             ("--opening-table", args.opening_table)) if value]
        if hand_set:
            ap.error(f"--recipe trains the registered recipe as it stands; drop {', '.join(hand_set)}")
        cfg, args.seeds = load_recipe(args.recipe)
        args.opening_table = cfg.get("opening_table", {}).get("horizon")
    elif args.seeds is None or args.iterations is None:
        ap.error("give --recipe, or --seeds and --iterations")
    else:
        cfg = {**CONFIG, "iterations": args.iterations,
               "strategy_tree": {"player": 0, "depth": 10 if args.tree_depth is None else args.tree_depth},
               "relabel_workers": args.workers,
               **({"stop_value_delta": args.value_delta} if args.value_delta is not None else {}),
               **({"tree_value_target": True} if args.tree_value_target else {}),
               **({"stop_on_agreement": False} if args.no_stop else {}),
               **({"value_n_step": args.value_n_step} if args.value_n_step else {}),
               **({"target_refresh": args.target_refresh} if args.target_refresh is not None else {}),
               **({"backplay": {"frac": args.backplay[0], "ramp": int(args.backplay[1])}} if args.backplay else {}),
               **({"exploiter": {"frac": args.exploiter[0], "sims_factor": int(args.exploiter[1])}}
                  if args.exploiter else {})}
    stamps = (training_fingerprint("connect4"), training_fingerprint(modules=MEASUREMENT_MODULES))
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    nets = NETS / Path(args.out).name.replace(".json.gz", "")
    nets.mkdir(parents=True, exist_ok=True)
    curve_cases = None
    if args.curve:
        from harness.evidence import load_evidence
        from harness.fixed_set import fixed_positions

        curve_cases = fixed_positions(load_evidence(LABELS)["positions"], Connect4(), player=0, plies=[0, 2, 4, 6, 8])
    starts, table_rows, carried_cases = None, None, None
    if args.opening_table:
        import random

        from harness.evidence import load_evidence
        from harness.exact_values import ExactValues
        from harness.opening_table import full_table

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            exact = ExactValues(Connect4(), pool, load_evidence(LABELS)["positions"])
            table, starts = full_table(Connect4(), Connect4().initial_state(random.Random(0)), 0, exact.move_values,
                                       args.opening_table)
            if args.curve:
                game = Connect4()
                carried = list({game.state_key(c): c for s in starts for b in game.legal_actions(s)
                                for c in [game.step(s, b)] if not game.is_terminal(c)}.values())
                carried_cases = [(s, args.opening_table + 1, v) for s, v in zip(carried, exact.move_values(carried))]
        built = {"horizon": args.opening_table, "entries": len(table), "frontier": len(starts)}
        if args.recipe and cfg["opening_table"] != built:
            raise SystemExit(f"the opening table built here {built} is not the recipe's {cfg['opening_table']}")
        cfg = {**cfg, "opening_table": built}
        table_rows = [[list(k[0]), k[1], m] for k, m in table.items()]
    rows = [_train(cfg, seed, args.threads, nets, curve_cases, starts, carried_cases) for seed in args.seeds]
    game = Connect4()
    recorded = _recorded_values(game)
    with ProcessPoolExecutor(max_workers=args.workers + 2) as pool:
        for row in rows:
            row["certificate"] = _certify(ROOT / row["net"]["path"], args.certify_depth, recorded, pool, game)
            c = row["certificate"]
            print(f"seed {row['seed']}: certified through {args.certify_depth} = {c['certified']} "
                  f"(failures {c['failures']} {c['failures_by_ply']}; values {c['values']}) [{c['seconds']:.0f}s]",
                  flush=True)
    if (training_fingerprint("connect4"), training_fingerprint(modules=MEASUREMENT_MODULES)) != stamps:
        raise SystemExit("training or measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "training_fingerprint": stamps[0], "measurement_fingerprint": stamps[1],
                             "versions": {"python": platform.python_version(), "torch": torch.__version__},
                             "solver_c_sha256": hashlib.sha256(native_solver.SOURCE.read_bytes()).hexdigest(),
                             "root_value_source": EMPTY_BOARD_VALUE_SOURCE,
                             "config": {**cfg, "seeds": list(args.seeds), "certify_depth": args.certify_depth},
                             **({"curve_positions": len(curve_cases)} if curve_cases else {}),
                             **({"opening_table": table_rows} if table_rows else {}),
                             "seeds": rows})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
