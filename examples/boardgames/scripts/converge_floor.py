"""§C.48 T2 — run the generic tic-tac-toe process to convergence and score the RAW net at every raw position after
every pass. Writes one evidence file per arm, judged by harness.floor_converge.converge_report.

    PYTHONPATH=. .venv/bin/python scripts/converge_floor.py --arm augment --workers 10 \\
        --out evidence/c48_T2_augment.json.gz --save-nets checkpoints/c48_T2/augment

The per-pass score comes from harness.transfer.record_selfplay_states' probe: every one of the 4,520 non-terminal
raw positions with its exact optimal set, built before training, graded after each train_net call by one batched
eval-mode forward. It observes only — the run trains exactly as it would without it."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

MEASUREMENT_MODULES = ("scripts/converge_floor.py", "harness/transfer.py", "harness/coverage.py")


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
            channels=cfg["arch"]["channels"], net_arch=cfg["arch"], augment=cfg["augment"], gumbel=cfg["gumbel"],
            c_scale=cfg["c_scale"], seed=seed, selfplay_opening_plies=cfg["opening_plies"],
            opening_plies_zero_frac=cfg["opening_zero_frac"], epochs=cfg["epochs"], batch_size=cfg["batch_size"],
            lr=cfg["lr"], buffer_cap=cfg["buffer_cap"], reanalyze_frac=cfg["reanalyze_frac"],
            reanalyze_sims=cfg["reanalyze_sims"], reanalyze_siblings=cfg["reanalyze_siblings"],
            steps_matched=cfg["steps_matched"])
    net.eval()
    with torch.no_grad():
        logits, _v = net(probe["x"])
    moves = logits.masked_fill(~probe["legal"], float("-inf")).argmax(dim=1).tolist()
    failing = [i for i, (m, opt) in enumerate(zip(moves, probe["optimal"])) if m not in opt]
    per_pass = [0] * cfg["iterations"]
    for g in log["games"]:
        per_pass[g["pass"]] += 1
    visited = {game.state_key(s) for g in log["games"] for s in g["states"]}
    if job.get("save_nets"):
        from pathlib import Path

        out = Path(job["save_nets"])
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "arch": cfg["arch"]}, out / f"seed{seed}.pt")
    return {"seed": seed, "params": sum(p.numel() for p in net.parameters()), "positions": len(raw),
            "strict_failures_per_pass": [round((1 - a) * len(raw)) for a in log["probe"]],
            "final_failing_positions": [[list(raw[i].board), game.current_player(raw[i])] for i in failing],
            "final_failing_keys": sorted({game.canonical_key(raw[i]) for i in failing}),
            "games_per_pass": per_pass, "selfplay_raw_positions_visited": len(visited), "history": history,
            "train_seconds": round(time.time() - t0, 1)}


def main() -> None:
    import platform

    from harness.evidence import save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.floor_converge import T2_ARMS, T2_SEEDS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=sorted(T2_ARMS))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--save-nets", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    config = T2_ARMS[args.arm]
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamps = (training_fingerprint(config["game"]), training_fingerprint(modules=MEASUREMENT_MODULES))
    jobs = [{"config": config, "seed": s, "save_nets": args.save_nets} for s in T2_SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run, jobs):
            print(f"{args.arm} seed {r['seed']}: strict failures per pass {r['strict_failures_per_pass']} "
                  f"[{r['train_seconds']:.0f}s]", flush=True)
            rows.append(r)
    if (training_fingerprint(config["game"]), training_fingerprint(modules=MEASUREMENT_MODULES)) != stamps:
        raise SystemExit("training or measurement code changed while the seeds ran — evidence not written")
    import torch

    save_evidence(args.out, {"arm": args.arm, "started": started,
                             "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "training_fingerprint": stamps[0], "measurement_fingerprint": stamps[1],
                             "versions": {"python": platform.python_version(), "torch": torch.__version__},
                             "config": {**config, "seeds": list(T2_SEEDS)}, "seeds": rows})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
