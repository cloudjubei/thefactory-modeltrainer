"""§3.7 BF1 — reads T8's file back, draws harness.floor_bf1's sample of its leaf positions, gives each to the three
arms (best-first proving with maps, depth-first proving with maps, best-first proving with no maps) on `--workers`
processes, stores every proof, and only then asks the solver, as a grader, for each position's value.

    PYTHONPATH=. .venv/bin/python scripts/c4_best_first.py --workers 4"""
from __future__ import annotations

import argparse
import base64
import random
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from games.connect4 import COLS, ROWS, C4State, Connect4  # noqa: E402
from harness.best_first import BestFirst  # noqa: E402
from harness.c4_oracle import solve  # noqa: E402
from harness.composed import check  # noqa: E402
from harness.evidence import load_evidence, save_evidence  # noqa: E402
from harness.floor_bf1 import ARMS, SPEC, settings  # noqa: E402
from harness.floor_t1 import mirror_cells  # noqa: E402
from harness.position_walk import decode_strategy  # noqa: E402
from harness.self_proof import Prover, bounded_search  # noqa: E402
from harness.steady_state import Facts  # noqa: E402
from harness.strategy_file import leaf_states  # noqa: E402

CHECK_CAP = 10_000_000


def _facts() -> Facts:
    return Facts(Connect4(), **{k: v for k, v in SPEC["rule"].items() if k != "n_levels"})


def sample(nodes: dict) -> list:
    """[(stones, has exceptions, board, to_move)]: `per` leaf positions for each count of stones and kind."""
    game = Connect4()
    leaves = leaf_states(game, game.initial_state(), nodes)["leaves"]
    found: dict = {}
    for s, node in leaves:
        if node.get("mirrored") or node["leaf"] == 0:
            continue
        stones = sum(v != 0 for v in s.board)
        found.setdefault((stones, bool(node.get("exceptions"))), {})[s.board] = s.to_move
    rng = random.Random(SPEC["seed"])
    out = []
    for stones in SPEC["stones"]:
        for kind in (False, True):
            pool = sorted(found.get((stones, kind), {}).items())
            rng.shuffle(pool)
            out.extend((stones, kind, board, to_move) for board, to_move in pool[:SPEC["per"]])
    return out


def _proof(strategy: dict) -> dict:
    return {"nodes": [[list(k[0]), k[1], n] for k, n in strategy["nodes"].items()],
            "maps": [[[c, v] for c, v in sorted(m.items())] for m in strategy["maps"]]}


def run(job: tuple) -> dict:
    board, to_move, index, arm = job
    facts = _facts()
    state = C4State(board, to_move, None, False)
    levels = SPEC["rule"]["n_levels"]
    t0 = time.monotonic()
    deadline = t0 + SPEC["ceiling"]
    prover = Prover(facts, levels, SPEC["cap"], reuse_window=SPEC["reuse"], deadline=deadline,
                    search_below=SPEC["bound"], compose=True,
                    search=bounded_search(facts, levels, random.Random(index), SPEC["cap"], evaluations=SPEC["trials"]),
                    max_calls=SPEC["evaluations"] if arm == "depth" else None)
    if arm == "depth":
        result, work = prover.prove(state), None
    else:
        search = BestFirst(prover, max_evaluations=SPEC["evaluations"], deadline=deadline, maps=arm == "best")
        result, work = search.prove(state), search.evaluations
    seconds = time.monotonic() - t0
    out = {"result": result, "evaluations": prover.calls if work is None else work, "seconds": round(seconds, 1),
           "ceiling": seconds >= SPEC["ceiling"]}
    if result:
        strategy = prover.strategy(state)
        out.update(proof=_proof(strategy), checked=check(facts, state, strategy, levels, CHECK_CAP)["won"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--source", default=str(ROOT / "evidence" / SPEC["source"]))
    ap.add_argument("--out", default=str(ROOT / "evidence" / SPEC["file"]))
    args = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()
    game = Connect4()
    source = load_evidence(args.source)
    strategy = decode_strategy(_facts(), game.initial_state(), SPEC["rule"]["n_levels"], mirror_cells(ROWS, COLS),
                               base64.b64decode(source["file"]))
    picked = sample(strategy["nodes"])
    print(f"source read, {len(picked)} positions sampled, {time.time() - t0:.0f}s", flush=True)
    jobs = [(board, to_move, i, arm) for i, (_, _, board, to_move) in enumerate(picked) for arm in ARMS]
    with Pool(args.workers) as pool:
        results = pool.map(run, jobs, chunksize=1)
    print(f"arms done, {time.time() - t0:.0f}s", flush=True)
    with Pool(args.workers) as pool:
        values = pool.map(solve, [(board, to_move) for _, _, board, to_move in picked], chunksize=1)
    rows = [{"stones": stones, "excepted": kind, "board": list(board), "to_move": to_move, "value": value,
             "arms": dict(zip(ARMS, results[len(ARMS) * i: len(ARMS) * (i + 1)]))}
            for i, ((stones, kind, board, to_move), value) in enumerate(zip(picked, values))]
    save_evidence(args.out, {
        "started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec": "harness.floor_bf1", "settings": settings(SPEC), "source_sha256": source["sha256"],
        "positions": rows, "seconds": round(time.time() - t0, 1)})
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
