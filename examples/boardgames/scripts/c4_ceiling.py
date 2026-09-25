"""§C.48 T3 — the Connect-4 CEILING by net size: how often can a net of each size pick an optimal move on positions
it never saw, when it is trained on EXACT labels? This separates "the net cannot represent near-perfect play" from
"our solver-free loop does not find it" — the question T1 answers for tic-tac-toe, asked where enumeration is
impossible and a net must generalise.

Data: the pre-solved bank (scripts/build_bank.py): self-play positions of the §C.47 nets and one-move siblings of
them, plies 14-38, split train/test by a hash of the canonical key. Labels: the policy uniform over the exact
optimal set, the value the exact game value. Each (net, training size, seed) trains until its accuracy on a held-out
tenth of the training positions has not improved for PATIENCE epochs; the test set is read once, at that best
checkpoint. Reported: raw-policy accuracy (argmax over legal moves) on all test positions and on the NON-TRIVIAL
ones (no win-in-one, not every move of one value), by source and ply band, and training-set accuracy (capacity).

    PYTHONPATH=. .venv/bin/python scripts/c4_ceiling.py --bank evidence/c48_bank.json.gz --workers 10 \\
        --out evidence/c48_T3_ceiling.json.gz
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

NETS = {
    "legacy32": {"channels": 32},
    "residual32": {"channels": 32, "blocks": 3, "head_hidden": 32, "residual": True},
    "residual64": {"channels": 64, "blocks": 4, "head_hidden": 32, "residual": True, "batchnorm": True},
    "residual128": {"channels": 128, "blocks": 6, "head_hidden": 64, "residual": True, "batchnorm": True},
}
TRAIN_SIZES = (4000, 12000, None)
SEEDS = (1, 2)
BATCH = 256
LR = 1e-3
MAX_EPOCHS = 300
PATIENCE = 20
MEASUREMENT_MODULES = ("scripts/c4_ceiling.py", "scripts/build_bank.py")


def tensors(bank: dict):
    import torch

    from games.connect4 import C4State
    from harness.neural import encode
    from harness.registry import resolve_game
    from harness.transfer import nontrivial

    game = resolve_game("connect4")
    rows = bank["positions"]
    n, a = len(rows), game.num_actions
    x = torch.zeros(n, 2, 6, 7)
    legal = torch.zeros(n, a, dtype=torch.bool)
    optimal = torch.zeros(n, a, dtype=torch.bool)
    target = torch.zeros(n, a)
    value = torch.zeros(n, 1)
    meta = []
    for i, r in enumerate(rows):
        s = C4State(tuple(r["board"]), r["to_move"], None, False)
        vals = {int(k): v for k, v in r["values"].items()}
        best = max(vals.values())
        opt = [m for m, v in vals.items() if v == best]
        x[i] = encode(game, s)
        legal[i, list(vals)] = True
        optimal[i, opt] = True
        target[i, opt] = 1.0 / len(opt)
        value[i, 0] = float(best)
        meta.append({"split": r["split"], "source": r["source"], "ply": game.ply(s),
                     "nontrivial": nontrivial(game, s, vals)})
    return {"x": x, "legal": legal, "optimal": optimal, "target": target, "value": value, "meta": meta}


def _accuracy(net, data, idx) -> float:
    import torch

    net.eval()
    with torch.no_grad():
        hits = 0
        for i in range(0, len(idx), 2048):
            j = idx[i:i + 2048]
            logits, _v = net(data["x"][j])
            moves = logits.masked_fill(~data["legal"][j], float("-inf")).argmax(dim=1)
            hits += int(data["optimal"][j, :][torch.arange(len(j)), moves].sum())
    net.train()
    return hits / len(idx) if len(idx) else float("nan")


def run(job: dict) -> dict:
    import copy

    import torch

    torch.set_num_threads(job["threads"])
    from harness.evidence import load_evidence
    from harness.neural import Connect4Net, arch_for_game
    from harness.registry import resolve_game

    game = resolve_game("connect4")
    data = tensors(load_evidence(job["bank"]))
    meta = data["meta"]
    train_all = [i for i, m in enumerate(meta) if m["split"] == "train"]
    test = torch.tensor([i for i, m in enumerate(meta) if m["split"] == "test"])
    g = torch.Generator().manual_seed(1000 + job["seed"])
    perm = [train_all[i] for i in torch.randperm(len(train_all), generator=g).tolist()]
    n_val = len(perm) // 10
    val, pool = torch.tensor(perm[:n_val]), perm[n_val:]
    train = torch.tensor(pool[: job["train_size"]] if job["train_size"] else pool)
    torch.manual_seed(job["seed"])
    net = Connect4Net(**arch_for_game(NETS[job["net"]], game))
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    best, best_epoch, best_state, curve = -1.0, 0, None, []
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        order = train[torch.randperm(len(train))]
        for i in range(0, len(order), BATCH):
            j = order[i:i + BATCH]
            logits, v = net(data["x"][j])
            logp = torch.log_softmax(logits.masked_fill(~data["legal"][j], -1e9), dim=1)
            loss = -(data["target"][j] * logp).sum(dim=1).mean() + ((v - data["value"][j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = _accuracy(net, data, val)
        curve.append([epoch, round(acc, 4)])
        if acc > best:
            best, best_epoch, best_state = acc, epoch, copy.deepcopy(net.state_dict())
        if epoch - best_epoch >= PATIENCE:
            break
    net.load_state_dict(best_state)

    def subset(pred):
        return torch.tensor([int(i) for i in test.tolist() if pred(meta[i])], dtype=torch.long)

    readings = {"test_all": _accuracy(net, data, test),
                "test_nontrivial": _accuracy(net, data, subset(lambda m: m["nontrivial"])),
                "train_fit": _accuracy(net, data, train), "val": best}
    for src in ("selfplay", "sibling"):
        readings[f"test_nontrivial_{src}"] = _accuracy(net, data, subset(lambda m, s=src: m["nontrivial"] and m["source"] == s))
    for lo, hi in ((14, 21), (22, 29), (30, 38)):
        readings[f"test_nontrivial_ply{lo}_{hi}"] = _accuracy(
            net, data, subset(lambda m, a=lo, b=hi: m["nontrivial"] and a <= m["ply"] <= b))
    return {"net": job["net"], "params": params, "train_size": len(train), "seed": job["seed"],
            "best_epoch": best_epoch, "epochs_run": curve[-1][0], "readings": readings, "val_curve": curve,
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    import platform

    import torch

    from harness.evidence import load_evidence, manifest_entry, save_evidence
    from harness.fingerprint import training_fingerprint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bank = load_evidence(args.bank)
    nets = [n for n in NETS if not args.only or n in args.only.split(",")]
    jobs = [{"net": n, "train_size": t, "seed": s, "bank": args.bank, "threads": 2 if n == "residual128" else 1}
            for n in reversed(nets) for t in TRAIN_SIZES for s in SEEDS]
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run, jobs):
            rd = r["readings"]
            print(f"{r['net']:>11} ({r['params']:>7}) n={r['train_size']:>6} seed {r['seed']}: test {rd['test_all']:.3f} "
                  f"non-trivial {rd['test_nontrivial']:.3f} train-fit {rd['train_fit']:.3f} "
                  f"[best epoch {r['best_epoch']}, {r['seconds']:.0f}s]", flush=True)
            rows.append(r)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("the measurement code changed while the runs ran — evidence not written")
    save_evidence(args.out, {"game": "connect4", "started": started,
                             "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "bank": args.bank,
                             "bank_sha256": (manifest_entry(args.bank) or {}).get("sha256"),
                             "bank_positions": len(bank["positions"]),
                             "versions": {"python": platform.python_version(), "torch": torch.__version__},
                             "config": {"nets": NETS, "train_sizes": list(TRAIN_SIZES), "seeds": list(SEEDS),
                                        "batch": BATCH, "lr": LR, "max_epochs": MAX_EPOCHS, "patience": PATIENCE},
                             "runs": rows})


if __name__ == "__main__":
    main()
