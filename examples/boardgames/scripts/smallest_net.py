"""§C.48 T1 — the SMALLEST net that plays tic-tac-toe perfectly, learned from the exact answers with no symmetry.

Every reachable non-terminal raw position (4,520) gets one row: the policy uniform over the solver's optimal set,
the value the exact game value. No augmentation, no canonical keys, no averaging — each orientation is its own
position, so a net that passes has learned all of them. Every candidate is trained until its raw policy (argmax
over legal moves, one forward, no search) is optimal at EVERY position, or until it stops improving; that is the
difference from §C.46's G0, which trained two sizes at a fixed step budget and scored one image per state.

Families: the harness's own nets (the legacy two-conv net at several widths; the residual tower at several
widths) and, as the floor for "how small can a net be", a plain one-hidden-layer MLP over the 18 input cells.

    PYTHONPATH=. .venv/bin/python scripts/smallest_net.py --seeds 1-5 --workers 10 --out evidence/c48_T1_smallest.json.gz
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

GAME = "tictactoe"
CANDIDATES = {
    **{f"mlp{h}": {"family": "mlp", "hidden": h} for h in (4, 8, 12, 16, 24, 32, 64)},
    **{f"legacy{c}": {"family": "harness", "arch": {"channels": c}} for c in (1, 2, 4, 8, 16, 32)},
    **{f"residual{c}": {"family": "harness", "arch": {"channels": c, "blocks": 1, "head_hidden": c, "residual": True}}
       for c in (4, 8, 16, 32)},
}
MEASUREMENT_MODULES = ("scripts/smallest_net.py", "harness/coverage.py")
BATCH = 256
LR = 2e-3
CHECK_EVERY = 25
MAX_EPOCHS = 6000
PATIENCE = 1500


def _parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def dataset(game):
    """All non-terminal raw positions with exact labels, and the tensors the trainer and the grader need."""
    import torch

    from harness.coverage import optimal_actions, reachable_states
    from harness.neural import encode

    raw, complete = reachable_states(game, exact=True, symmetry=False)
    assert complete
    raw = [s for s in raw if not game.is_terminal(s)]
    n = game.num_actions
    x = torch.stack([encode(game, s) for s in raw])
    legal = torch.zeros(len(raw), n, dtype=torch.bool)
    target = torch.zeros(len(raw), n)
    optimal = torch.zeros(len(raw), n, dtype=torch.bool)
    value = torch.zeros(len(raw), 1)
    for i, s in enumerate(raw):
        legal[i, game.legal_actions(s)] = True
        opt = sorted(optimal_actions(game, s))
        optimal[i, opt] = True
        target[i, opt] = 1.0 / len(opt)
        value[i, 0] = float(game.position_value(s))
    return {"x": x, "legal": legal, "target": target, "optimal": optimal, "value": value, "positions": len(raw)}


def build(spec: dict, game):
    import torch
    from torch import nn

    from harness.neural import Connect4Net, arch_for_game

    if spec["family"] == "harness":
        return Connect4Net(**arch_for_game(spec["arch"], game))

    class MLP(nn.Module):
        def __init__(self, inputs: int, hidden: int, actions: int):
            super().__init__()
            self.body = nn.Sequential(nn.Flatten(), nn.Linear(inputs, hidden), nn.ReLU())
            self.policy = nn.Linear(hidden, actions)
            self.value = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.body(x)
            return self.policy(h), torch.tanh(self.value(h))

    return MLP(18, spec["hidden"], game.num_actions)


def failures(net, data) -> int:
    """Raw positions whose policy argmax over the LEGAL moves is not optimal — every orientation counted."""
    import torch

    net.eval()
    with torch.no_grad():
        logits, _v = net(data["x"])
    net.train()
    moves = logits.masked_fill(~data["legal"], float("-inf")).argmax(dim=1)
    return int((~data["optimal"][torch.arange(len(moves)), moves]).sum())


def run(job: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    from harness.registry import resolve_game

    game = resolve_game(GAME)
    data = dataset(game)
    torch.manual_seed(job["seed"])
    net = build(CANDIDATES[job["name"]], game)
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = data["positions"]
    best, best_epoch, curve, solved_at = n, 0, [], None
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        order = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            logits, v = net(data["x"][idx])
            logp = torch.log_softmax(logits.masked_fill(~data["legal"][idx], -1e9), dim=1)
            loss = -(data["target"][idx] * logp).sum(dim=1).mean() + ((v - data["value"][idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % CHECK_EVERY == 0:
            f = failures(net, data)
            curve.append([epoch, f])
            if f < best:
                best, best_epoch = f, epoch
            if f == 0:
                solved_at = epoch
                break
            if epoch - best_epoch >= PATIENCE:
                break
    return {"name": job["name"], "spec": CANDIDATES[job["name"]], "seed": job["seed"], "params": params,
            "solved": solved_at is not None, "solved_at_epoch": solved_at, "best_failures": best,
            "final_failures": curve[-1][1] if curve else None, "epochs_run": curve[-1][0] if curve else 0,
            "steps_per_epoch": -(-n // BATCH), "curve": curve, "seconds": round(time.time() - t0, 1)}


def main() -> None:
    import platform

    import torch

    from harness.evidence import save_evidence
    from harness.fingerprint import training_fingerprint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="1-5")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only", default="", help="comma-separated candidate names (default: all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    names = [n for n in CANDIDATES if not args.only or n in args.only.split(",")]
    seeds = _parse_seeds(args.seeds)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    jobs = [{"name": n, "seed": s} for n in names for s in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run, jobs):
            print(f"{r['name']:>12} ({r['params']:>6} params) seed {r['seed']}: "
                  f"{'SOLVED at epoch ' + str(r['solved_at_epoch']) if r['solved'] else 'best ' + str(r['best_failures']) + ' failures'}"
                  f"  [{r['seconds']:.0f}s]", flush=True)
            rows.append(r)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("the measurement code changed while the runs ran — evidence not written")
    summary = []
    for n in names:
        rs = [r for r in rows if r["name"] == n]
        summary.append({"name": n, "params": rs[0]["params"], "solved_seeds": sum(r["solved"] for r in rs),
                        "seeds": len(rs), "best_failures": [r["best_failures"] for r in rs],
                        "solved_at_epoch": [r["solved_at_epoch"] for r in rs]})
    save_evidence(args.out, {"game": GAME, "started": started,
                             "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp,
                             "versions": {"python": platform.python_version(), "torch": torch.__version__},
                             "config": {"candidates": CANDIDATES, "seeds": seeds, "batch": BATCH, "lr": LR,
                                        "check_every": CHECK_EVERY, "max_epochs": MAX_EPOCHS, "patience": PATIENCE},
                             "runs": rows, "summary": summary})
    for s in sorted(summary, key=lambda s: s["params"]):
        print(f"{s['name']:>12} {s['params']:>7} params: {s['solved_seeds']}/{s['seeds']} solved; best {s['best_failures']}")


if __name__ == "__main__":
    main()
