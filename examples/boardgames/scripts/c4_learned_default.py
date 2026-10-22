"""§3.6 N1 — a learned default, piloted: does a small net, paid once, keep the win where the zero-bit orders miss?
The test positions are D1's exactly (scripts/c4_default_accuracy.py: the own positions with two or more safe moves and
no winning move on the certified leaves of a SPEC's run, a seeded sample a root), the training positions drawn the
same way from another run's roots — those the test's run built left out, and any position the test holds dropped.
Every position is solved exactly; a net (harness.learned_default) is trained on the training positions with the
SPEC's settings, its weights quantized, and its pick among all the safe moves — float and quantized — scored on the
test positions with the lowest and tactics orders beside it, which must reproduce D1's hits.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_learned_default.py --spec harness.floor_n1 --workers 2 \\
        --out evidence/c49_n1.json.gz
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import platform
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_MODULES = ("scripts/c4_learned_default.py", "harness/learned_default.py", "harness/steady_state.py",
                       "harness/native_solver.py", "harness/native_build.py", "scripts/c4_default_accuracy.py")
C_SOURCES = ("harness/c4solver.c",)
_FACTS: dict = {}


def _facts(builder: dict):
    from games.connect4 import Connect4
    from harness.steady_state import Facts

    rule = {"ties": builder.get("ties", "exact"), "safe_depth": builder.get("safe_depth", 1),
            "forcing": builder.get("forcing", False), "unlevelled": builder.get("unlevelled", "none"),
            "order": builder.get("order", "lowest")}
    key = tuple(rule.values())
    if key not in _FACTS:
        _FACTS[key] = Facts(Connect4(), **rule)
    return _FACTS[key]


def pooled(run: dict, builder: dict, per_leaf: int, per_root: int, seed: int, skip=()) -> list:
    """D1's sample (scripts/c4_default_accuracy.py), as (board, to_move) keys in its order."""
    from games.connect4 import C4State, Connect4
    from harness.steady_state import choose

    game, facts, n_levels = Connect4(), _facts(builder), builder["n_levels"]
    out = []
    for r in run["roots"]:
        if not r["complete"] or r["index"] in skip:
            continue
        pool: set = set()
        for board, to_move, node in r["nodes"]:
            if "leaf" not in node:
                continue
            leaf = C4State(tuple(board), to_move, None, False)
            player = game.current_player(leaf)
            empty = set(facts.empty_cells(leaf))
            levels = {int(c): k for c, k in r["maps"][node["leaf"]].items() if int(c) in empty}
            exceptions = {(tuple(eb), et): move for eb, et, move in node.get("exceptions", [])}
            stack, seen, own = [leaf], set(), 0
            while stack and own < per_leaf:
                s = stack.pop()
                key = game.state_key(s)
                if key in seen or game.is_terminal(s):
                    continue
                seen.add(key)
                if game.current_player(s) != player:
                    stack.extend(game.step(s, a) for a in game.legal_actions(s))
                    continue
                own += 1
                wins, safe, _cell = facts.of(s)
                if not wins and len(safe) >= 2:
                    pool.add(key)
                move = exceptions.get(key)
                if move is None:
                    move = choose(facts, s, levels, n_levels)
                if move is not None:
                    stack.append(game.step(s, move))
        keys = sorted(pool)
        out += random.Random(seed * 1_000_003 + r["index"]).sample(keys, min(per_root, len(keys)))
    return out


def labelled(job: tuple) -> dict:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from c4_default_accuracy import pick

    from games.connect4 import C4State, Connect4
    from harness.learned_default import features
    from harness.native_solver import move_values

    (board, to_move), builder = job
    game, facts = Connect4(), _facts(builder)
    s = C4State(tuple(board), to_move, None, False)
    safe = facts.of(s)[1]
    values = move_values(s)
    return {"features": features(game, facts, s), "safe": [int(a in safe) for a in range(game.num_actions)],
            "won": [int(a in safe and values.get(a) == 1) for a in range(game.num_actions)],
            "lowest": pick("lowest", safe, game, facts, s), "tactics": pick("tactics", safe, game, facts, s)}


def main() -> None:
    import torch

    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.learned_default import net_bits, picks, quantized, train

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    spec = importlib.import_module(args.spec).SPEC
    torch.set_num_threads(1)
    sets = {}
    for part in ("test", "train"):
        p = spec[part]
        builder = importlib.import_module(p["spec"]).SPEC["builder"]
        keys = pooled(load_evidence(ROOT / "evidence" / p["targets"]), builder, p["per_leaf"], p["per_root"],
                      p["seed"], set(p.get("skip", ())))
        sets[part] = (keys, builder)
    test_keys = set(sets["test"][0])
    train_keys = [k for k in sets["train"][0] if k not in test_keys]
    dropped = len(sets["train"][0]) - len(train_keys)
    print(f"test {len(test_keys)}, train {len(train_keys)} ({dropped} held by the test dropped)", flush=True)
    rows = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for part, keys in (("test", sets["test"][0]), ("train", train_keys)):
            rows[part] = list(pool.map(labelled, [(k, sets[part][1]) for k in keys], chunksize=64))
            print(f"{part} labelled", flush=True)

    def tensors(part):
        r = rows[part]
        return (torch.tensor([x["features"] for x in r], dtype=torch.float32),
                torch.tensor([x["safe"] for x in r], dtype=torch.bool),
                torch.tensor([x["won"] for x in r], dtype=torch.bool))

    x, safe, won = tensors("train")
    n = spec["net"]
    net = train(x, safe, won, hidden=n["hidden"], epochs=n["epochs"], batch=n["batch"], lr=n["lr"], seed=n["seed"])
    q = quantized(net, n["weight_bits"])
    tx, tsafe, twon = tensors("test")

    def hits(chosen):
        return int(twon[torch.arange(len(chosen)), chosen].sum())

    totals = {"solved": len(rows["test"]), "lowest": sum(r["won"][r["lowest"]] for r in rows["test"]),
              "tactics": sum(r["won"][r["tactics"]] for r in rows["test"]), "net_float": hits(picks(net, tx, tsafe)),
              "net": hits(picks(q, tx, tsafe)), "train_hits": int(won[torch.arange(len(x)), picks(q, x, safe)].sum()),
              "trained_on": len(rows["train"]), "dropped": dropped, "bits": net_bits(q, n["weight_bits"])}
    print(totals, flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version(),
                                                                            "torch": torch.__version__},
                             "c_sources": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest()[:12]
                                           for s in C_SOURCES},
                             "spec": args.spec, "test": spec["test"], "train": spec["train"], "net": n,
                             "totals": totals, "weights": {k: v.tolist() for k, v in q.state_dict().items()}})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
