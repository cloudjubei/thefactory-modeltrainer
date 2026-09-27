"""§C.49 T6 — T5's working process at the oracle-frontier setups, scoring the RAW net at every raw tic-tac-toe
position after every pass. Writes one evidence file per arm, judged by harness.floor_small.small_report.

    PYTHONPATH=. .venv/bin/python scripts/small_floor.py --spec floor_small --arm canon_mlp32 --workers 2 \\
        --out evidence/c49_T6_canon_mlp32.json.gz

`--spec` names the registration module whose arms and seeds to run: floor_small (T6) or floor_budget (T7).

A position counts as TRAINED when it is in the final training set (every recorded self-play position and the last
sibling set) under the key the net itself tells positions apart by: every orientation is one input for a
standardised-orientation net or with augmentation, and a separate one otherwise."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

MEASUREMENT_MODULES = ("scripts/small_floor.py", "harness/transfer.py", "harness/coverage.py")


def run(job: dict) -> dict:
    import time

    import torch

    torch.set_num_threads(1)
    from harness.coverage import move_values, reachable_states
    from harness.neural import encode, train_alphazero
    from harness.registry import resolve_game
    from harness.transfer import build_probe, record_selfplay_states

    cfg, seed = job["config"], job["seed"]
    game = resolve_game(cfg["game"])
    raw, complete = reachable_states(game, exact=True, symmetry=False)
    assert complete
    raw = [s for s in raw if not game.is_terminal(s)]
    probe = build_probe(game, raw, encode, lambda s: move_values(game, s))
    t0 = time.time()
    with record_selfplay_states(game, probe) as log:
        net, history = train_alphazero(
            game, iterations=cfg["iterations"], selfplay_games=cfg["selfplay"], sims=cfg["train_sims"],
            channels=cfg["arch"].get("channels", 32), net_arch=cfg["arch"], augment=cfg["augment"],
            gumbel=cfg["gumbel"], c_scale=cfg["c_scale"], seed=seed, selfplay_opening_plies=cfg["opening_plies"],
            opening_plies_zero_frac=cfg["opening_zero_frac"], epochs=cfg["epochs"], batch_size=cfg["batch_size"],
            lr=cfg["lr"], buffer_cap=cfg["buffer_cap"], reanalyze_frac=cfg["reanalyze_frac"],
            reanalyze_sims=cfg["reanalyze_sims"], reanalyze_siblings=cfg["reanalyze_siblings"],
            steps_matched=cfg["steps_matched"], buffer_unique=cfg["buffer_unique"],
            settle_epochs=cfg["settle_epochs"], settle_lr_final=cfg["settle_lr_final"],
            record_self_agreement=cfg["record_self_agreement"], sibling_depth=cfg["sibling_depth"],
            settle_lr=cfg.get("settle_lr"))
    if any(h.get("evicted", 0) for h in history):
        raise RuntimeError(f"seed {seed}: the buffer evicted, so the recorded positions are not the trained set")
    net.eval()
    with torch.no_grad():
        logits, _v = net(probe["x"])
    moves = logits.masked_fill(~probe["legal"], float("-inf")).argmax(dim=1).tolist()
    failing = {i for i, (m, opt) in enumerate(zip(moves, probe["optimal"])) if m not in opt}
    canonical = cfg["augment"] or cfg["arch"].get("canonical_input", False)
    key = game.canonical_key if canonical else game.state_key
    trained_keys = {key(s) for g in log["games"] for s in g["states"]}
    trained_keys |= {key(s) for s in log["siblings"][-1]["states"]}
    trained = [key(s) in trained_keys for s in raw]
    per_pass = [0] * log["passes"]
    for g in log["games"]:
        per_pass[g["pass"]] += 1
    return {"seed": seed, "params": sum(p.numel() for p in net.parameters()), "positions": len(raw),
            "strict_failures_per_pass": [round((1 - a) * len(raw)) for a in log["probe"]],
            "final_failing_positions": [[list(raw[i].board), game.current_player(raw[i]), trained[i]]
                                        for i in sorted(failing)],
            "final_failing_keys": sorted({game.canonical_key(raw[i]) for i in failing}),
            "coverage": {"trained": sum(trained), "untrained": len(raw) - sum(trained),
                         "failures_trained": sum(1 for i in failing if trained[i]),
                         "failures_untrained": sum(1 for i in failing if not trained[i])},
            "games_per_pass": per_pass, "history": history, "train_seconds": round(time.time() - t0, 1)}


def main() -> None:
    import platform

    import torch

    from harness.evidence import save_evidence
    from harness.fingerprint import training_fingerprint
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, choices=["floor_small", "floor_budget"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.spec == "floor_small":
        from harness.floor_small import T6_ARMS as arms, T6_SEEDS as seeds
    else:
        from harness.floor_budget import SPEC

        arms, seeds = SPEC["arms"], SPEC["seeds"]
    if args.arm not in arms:
        raise SystemExit(f"{args.spec} has no arm {args.arm}: {sorted(arms)}")
    config = arms[args.arm]
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamps = (training_fingerprint(config["game"]), training_fingerprint(modules=MEASUREMENT_MODULES))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run, [{"config": config, "seed": s} for s in seeds]):
            print(f"{args.arm} seed {r['seed']}: strict failures per pass {r['strict_failures_per_pass']} "
                  f"coverage {r['coverage']} [{r['train_seconds']:.0f}s]", flush=True)
            rows.append(r)
    if (training_fingerprint(config["game"]), training_fingerprint(modules=MEASUREMENT_MODULES)) != stamps:
        raise SystemExit("training or measurement code changed while the seeds ran — evidence not written")
    save_evidence(args.out, {"arm": args.arm, "started": started,
                             "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "training_fingerprint": stamps[0], "measurement_fingerprint": stamps[1],
                             "versions": {"python": platform.python_version(), "torch": torch.__version__},
                             "config": {**config, "seeds": list(seeds)}, "seeds": rows})


if __name__ == "__main__":
    main()
