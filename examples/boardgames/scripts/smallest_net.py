"""§C.48 T1 — the SMALLEST setup that plays tic-tac-toe perfectly, learned from the exact answers.

Every candidate is scored the same way: at EVERY one of the 4,520 reachable non-terminal raw positions, one forward
pass, the argmax over legal moves must be an optimal move. Every candidate is trained on exact labels (the policy
uniform over the solver's optimal set, the value the exact game value) until it is perfect or stops improving.

A setup is more than a size: it is the net AND how the position reaches it. Input modes:

  raw         the position as it is (no symmetry anywhere) — the first T1 sweep;
  canon       the position mapped to one canonical orientation by the game's verified isometries (the one whose
              state key is smallest), the net's move mapped back; the net then has 627 cases to learn, not 4,520.
              Cost at play time: one transform per isometry;
  feat        the raw position plus two rule-derived features per move — does it win now, does it let the opponent
              win next move — computed only through the game's own `step`. Cost: 9 + 9x8 rule calls;
  canon_feat  both.

Families: the harness's own nets (legacy two-conv, residual tower) and plain MLPs (one or two hidden layers).

    PYTHONPATH=. .venv/bin/python scripts/smallest_net.py --seeds 1-5 --workers 10 --out evidence/c48_T1b_setups.json.gz
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

GAME = "tictactoe"


def _mlp(hidden, mode="raw"):
    return {"family": "mlp", "hidden": list(hidden), "input": mode}


def _harness(arch, mode="raw"):
    return {"family": "harness", "arch": arch, "input": mode}


def _res(c):
    return {"channels": c, "blocks": 1, "head_hidden": c, "residual": True}


CANDIDATES = {
    **{f"mlp{h}": _mlp([h]) for h in (4, 8, 12, 16, 24, 32, 64, 128, 256)},
    **{f"mlp{h}x{h}": _mlp([h, h]) for h in (16, 32, 64)},
    **{f"legacy{c}": _harness({"channels": c}) for c in (1, 2, 4, 8, 16, 32)},
    **{f"residual{c}": _harness(_res(c)) for c in (4, 8, 16, 32)},
    **{f"canon_mlp{h}": _mlp([h], "canon") for h in (4, 8, 16, 32, 64, 128)},
    **{f"canon_mlp{h}x{h}": _mlp([h, h], "canon") for h in (16, 32)},
    **{f"canon_legacy{c}": _harness({"channels": c}, "canon") for c in (2, 4, 8, 16)},
    **{f"canon_residual{c}": _harness(_res(c), "canon") for c in (4, 8, 16)},
    **{f"feat_mlp{h}": _mlp([h], "feat") for h in (2, 4, 8, 16, 32)},
    **{f"canon_feat_mlp{h}": _mlp([h], "canon_feat") for h in (2, 4, 8, 16)},
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


def rule_features(game, s) -> list[float]:
    """Per move: 1 if it wins at once; 1 if it hands the opponent a win on the very next move. Only `step`."""
    n = game.num_actions
    wins, gives = [0.0] * n, [0.0] * n
    me = game.current_player(s)
    for a in game.legal_actions(s):
        c = game.step(s, a)
        if game.is_terminal(c):
            wins[a] = 1.0 if game.returns(c)[me] > 0 else 0.0
            continue
        for b in game.legal_actions(c):
            g = game.step(c, b)
            if game.is_terminal(g) and game.returns(g)[me] < 0:
                gives[a] = 1.0
                break
    return wins + gives


def canonicalise(game, s, isos):
    """The image of `s` with the smallest state key under the verified isometries, and the forward action image
    that maps a move in `s` to the same move in that image."""
    best = None
    for iso, perm in isos:
        t = game.transform_state(s, iso)
        k = repr(game.state_key(t))
        if best is None or k < best[0]:
            best = (k, t, perm)
    return best[1], best[2]


def dataset(game, mode: str):
    """Training rows in the net's frame, and evaluation rows for EVERY raw position with the map from the net's move
    back to the raw position's move."""
    import torch

    from harness.coverage import optimal_actions, reachable_states
    from harness.neural import encode
    from harness.symmetry import verified_isometries

    raw, complete = reachable_states(game, exact=True, symmetry=False)
    assert complete
    raw = [s for s in raw if not game.is_terminal(s)]
    n = game.num_actions
    isos = verified_isometries(game)
    canon = mode.startswith("canon")
    feats = mode.endswith("feat")

    def row(s):
        x = encode(game, s).flatten()
        if feats:
            x = torch.cat([x, torch.tensor(rule_features(game, s))])
        legal = torch.zeros(n, dtype=torch.bool)
        legal[game.legal_actions(s)] = True
        opt = sorted(optimal_actions(game, s))
        target = torch.zeros(n)
        target[opt] = 1.0 / len(opt)
        return x, legal, target, float(game.position_value(s))

    net_states, back, optimal_raw = [], [], []
    for s in raw:
        t, perm = canonicalise(game, s, isos) if canon else (s, list(range(n)))
        inverse = [0] * n
        for a, b in enumerate(perm):
            inverse[b] = a
        net_states.append(t)
        back.append(inverse)
        opt = torch.zeros(n, dtype=torch.bool)
        opt[sorted(optimal_actions(game, s))] = True
        optimal_raw.append(opt)
    eval_rows = [row(t) for t in net_states]
    train_states = list({repr(game.state_key(t)): t for t in net_states}.values())
    train_rows = [row(t) for t in train_states]

    def stack(rows):
        return {"x": torch.stack([r[0] for r in rows]), "legal": torch.stack([r[1] for r in rows]),
                "target": torch.stack([r[2] for r in rows]), "value": torch.tensor([[r[3]] for r in rows])}

    return {"train": stack(train_rows), "eval": {**stack(eval_rows), "back": torch.tensor(back),
                                                  "optimal": torch.stack(optimal_raw)},
            "positions": len(raw), "train_rows": len(train_rows)}


def build(spec: dict, game, inputs: int):
    import torch
    from torch import nn

    from harness.neural import Connect4Net, arch_for_game

    if spec["family"] == "harness":
        net = Connect4Net(**arch_for_game(spec["arch"], game))

        class Flat(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = net

            def forward(self, x):
                return self.net(x.reshape(-1, 2, 3, 3))

        return Flat()

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers, width = [], inputs
            for h in spec["hidden"]:
                layers += [nn.Linear(width, h), nn.ReLU()]
                width = h
            self.body = nn.Sequential(*layers)
            self.policy = nn.Linear(width, game.num_actions)
            self.value = nn.Linear(width, 1)

        def forward(self, x):
            h = self.body(x)
            return self.policy(h), torch.tanh(self.value(h))

    return MLP()


def failures(net, ev) -> int:
    """Raw positions whose move — the net's argmax over the legal moves in its own frame, mapped back — is not
    optimal. Every raw position counted."""
    import torch

    net.eval()
    with torch.no_grad():
        logits, _v = net(ev["x"])
    net.train()
    moves = logits.masked_fill(~ev["legal"], float("-inf")).argmax(dim=1)
    rows = torch.arange(len(moves))
    raw_moves = ev["back"][rows, moves]
    return int((~ev["optimal"][rows, raw_moves]).sum())


def run(job: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    from harness.registry import resolve_game

    game = resolve_game(GAME)
    spec = CANDIDATES[job["name"]]
    data = dataset(game, spec["input"])
    tr, ev = data["train"], data["eval"]
    torch.manual_seed(job["seed"])
    net = build(spec, game, tr["x"].shape[1])
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    n = len(tr["x"])
    best, best_epoch, curve, solved_at = data["positions"], 0, [], None
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        order = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            logits, v = net(tr["x"][idx])
            logp = torch.log_softmax(logits.masked_fill(~tr["legal"][idx], -1e9), dim=1)
            loss = -(tr["target"][idx] * logp).sum(dim=1).mean() + ((v - tr["value"][idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % CHECK_EVERY == 0:
            f = failures(net, ev)
            curve.append([epoch, f])
            if f < best:
                best, best_epoch = f, epoch
            if f == 0:
                solved_at = epoch
                break
            if epoch - best_epoch >= PATIENCE:
                break
    return {"name": job["name"], "spec": spec, "seed": job["seed"], "params": params, "train_rows": data["train_rows"],
            "positions": data["positions"], "solved": solved_at is not None, "solved_at_epoch": solved_at,
            "best_failures": best, "final_failures": curve[-1][1] if curve else None,
            "epochs_run": curve[-1][0] if curve else 0, "steps_per_epoch": -(-n // BATCH), "curve": curve,
            "seconds": round(time.time() - t0, 1)}


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
            print(f"{r['name']:>18} ({r['params']:>6} params) seed {r['seed']}: "
                  f"{'SOLVED at epoch ' + str(r['solved_at_epoch']) if r['solved'] else 'best ' + str(r['best_failures']) + ' failures'}"
                  f"  [{r['seconds']:.0f}s]", flush=True)
            rows.append(r)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("the measurement code changed while the runs ran — evidence not written")
    summary = []
    for n in names:
        rs = [r for r in rows if r["name"] == n]
        summary.append({"name": n, "input": CANDIDATES[n]["input"], "params": rs[0]["params"],
                        "train_rows": rs[0]["train_rows"], "solved_seeds": sum(r["solved"] for r in rs),
                        "seeds": len(rs), "best_failures": [r["best_failures"] for r in rs],
                        "solved_at_epoch": [r["solved_at_epoch"] for r in rs]})
    save_evidence(args.out, {"game": GAME, "started": started,
                             "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp,
                             "versions": {"python": platform.python_version(), "torch": torch.__version__},
                             "config": {"candidates": {n: CANDIDATES[n] for n in names}, "seeds": seeds,
                                        "batch": BATCH, "lr": LR, "check_every": CHECK_EVERY,
                                        "max_epochs": MAX_EPOCHS, "patience": PATIENCE},
                             "runs": rows, "summary": summary})
    for s in sorted(summary, key=lambda s: s["params"]):
        print(f"{s['name']:>18} {s['input']:>10} {s['params']:>7} params: {s['solved_seeds']}/{s['seeds']} solved; "
              f"best {s['best_failures']}")


if __name__ == "__main__":
    main()
