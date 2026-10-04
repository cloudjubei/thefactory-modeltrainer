"""§C.49 D4 — at each T10 net's wrong moves on D3's positions (the label cache's White-to-move positions at plies
0-8 with every move's exact value), the net's own relabel LABEL: the argmax of its 200-sim search's improved policy,
with T10's search settings, exactly what training uses as the policy target. Each row also carries D3's inputs (raw
move, value-head ratings, exact values) and whether the position is on the net's own first-player tree. Judged by
harness.label_diagnosis.d4_report on D4_SPEC. No solver is used.

    PYTHONPATH=. .venv/bin/python scripts/c4_label_diagnosis.py --workers 8 --out evidence/c49_D4_labels.json.gz
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NETS = ROOT / "checkpoints" / "c49_sf" / "c49_T10_solver_free"
LABELS = ROOT / "books" / "c4_labels.json.gz"
MEASUREMENT_MODULES = ("scripts/c4_label_diagnosis.py",)
SEARCH = {"gumbel": True, "gumbel_m": 16, "c_scale": 0.1, "add_noise": False}


def _value_head(game, net, states: list) -> list:
    import torch

    from harness.neural import encode

    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(states), 4096):
            _logits, v = net(torch.stack([encode(game, s) for s in states[i:i + 4096]]))
            out.extend(float(x) for x in v[:, 0])
    return out


def main() -> None:
    import hashlib
    import multiprocessing as mp
    import os
    import platform
    import random
    import shutil
    import tempfile

    import torch

    import harness.neural as neural
    from games.connect4 import C4State, Connect4
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.label_diagnosis import D4_SPEC
    from harness.strategy_tree import raw_chooser, strategy_tree_positions

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    stamps = (training_fingerprint("connect4"), training_fingerprint(modules=MEASUREMENT_MODULES))
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    game = Connect4()
    root = game.initial_state(random.Random(0))
    plies = D4_SPEC["config"]["plies"]
    cases = []
    for row in load_evidence(LABELS)["positions"]:
        s = C4State(tuple(row["board"]), row["to_move"], None, False)
        ply = sum(1 for c in s.board if c)
        values = {int(a): int(v) for a, v in row["values"].items()}
        if s.to_move == 0 and ply in plies and not game.is_terminal(s) and set(values) == set(game.legal_actions(s)):
            cases.append((s, ply, values))
    children = [(i, a, game.step(s, a)) for i, (s, _p, values) in enumerate(cases) for a in values]
    open_children = [c for _i, _a, c in children if not game.is_terminal(c)]

    tmpdir = tempfile.mkdtemp(prefix="d4_rl_")
    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}
    for k in saved:
        os.environ[k] = "1"
    pool = mp.get_context("spawn").Pool(args.workers, initializer=neural._relabel_worker_init, initargs=("connect4",))
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    positions, nets = [], {}
    for version, seed in enumerate(D4_SPEC["seeds"], start=1):
        path = NETS / f"seed_{seed}.pt"
        net = neural.load_net(str(path))
        nets[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
        choose = raw_chooser(game, net)
        tree = {game.state_key(s) for s in strategy_tree_positions(game, root, 0, choose, max(plies) + 1)}
        raw = choose([s for s, _p, _v in cases])
        heads = iter(_value_head(game, net, open_children))
        q_hat = [{} for _ in cases]
        for i, a, child in children:
            q_hat[i][a] = game.returns(child)[0] if game.is_terminal(child) else -next(heads)
        wrong = [i for i, ((_s, _p, values), m) in enumerate(zip(cases, raw)) if values[m] != max(values.values())]
        labels = neural._parallel_relabel(pool, args.workers, tmpdir, net, version,
                                          {"sims": D4_SPEC["config"]["sims"], **SEARCH}, [cases[i][0] for i in wrong])
        for i, (_x, pi, _v) in zip(wrong, labels):
            s, p, values = cases[i]
            positions.append({"seed": seed, "ply": p, "raw": raw[i], "label": pi.index(max(pi)),
                              "on_tree": game.state_key(s) in tree,
                              "q_hat": {str(a): round(q, 6) for a, q in q_hat[i].items()},
                              "values": {str(a): v for a, v in values.items()}})
        print(f"seed {seed}: {len(wrong)} wrong moves relabelled; own tree {len(tree)} positions", flush=True)
    pool.close()
    pool.join()
    shutil.rmtree(tmpdir, ignore_errors=True)
    if (training_fingerprint("connect4"), training_fingerprint(modules=MEASUREMENT_MODULES)) != stamps:
        raise SystemExit("training or measurement code changed while the measurement ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "training_fingerprint": stamps[0], "measurement_fingerprint": stamps[1],
                             "versions": {"python": platform.python_version(), "torch": torch.__version__},
                             "config": D4_SPEC["config"], "search": SEARCH, "seeds": list(D4_SPEC["seeds"]),
                             "nets": nets, "positions": positions})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
