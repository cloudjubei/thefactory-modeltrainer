"""§3.6 K0 — do two-phase maps make smaller leaves where one-phase ones run out? From a SPEC's run, every position its
strategies for the heaviest roots split at (a move node: no leaf the builder would keep was found there), a seeded
sample is searched twice with the same budget and seed — harness.steady_local.local_search without and with `phases`,
under the run's rule, in C — and each search's smallest leaf (map bits plus exception bits, None when every walk
outgrew the cap) is saved with whether its map has a second phase.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_phase_pilot.py --spec harness.floor_k0 --out evidence/c49_k0.json.gz
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
MEASUREMENT_MODULES = ("scripts/c4_phase_pilot.py", "harness/steady_local.py", "harness/steady_state.py",
                       "harness/steady_exceptions.py", "harness/native_leaf.py", "harness/native_build.py",
                       "harness/winning_cache.py", "scripts/c4_steady_states.py")
C_SOURCES = ("harness/c4leafwalk.c", "harness/c4solver.c")


def search_one(job: dict) -> dict:
    import sys
    import time

    sys.path.insert(0, str(ROOT / "scripts"))
    from c4_steady_states import _winning

    from games.connect4 import C4State, Connect4
    from harness.native_leaf import needed
    from harness.steady_local import local_search
    from harness.steady_state import SWITCH, Facts
    from harness.winning_cache import CachedWinning

    game = Connect4()
    b = job["builder"]
    facts = Facts(game, ties=b.get("ties", "exact"), safe_depth=b.get("safe_depth", 1),
                  forcing=b.get("forcing", False), unlevelled=b.get("unlevelled", "none"),
                  order=b.get("order", "lowest"))
    winning = CachedWinning(lambda states: _winning(game, states), game.state_key, job["cache_limit"])
    state = C4State(tuple(job["board"]), job["to_move"], None, False)
    out = {"root": job["root"], "board": job["board"], "to_move": job["to_move"]}
    for name, phases in (("one", False), ("two", True)):
        t0 = time.time()
        r = local_search(facts, state, b["n_levels"], b["level_bits"], winning, random.Random(job["seed"]),
                         job["seconds"], job["cap"], walker=needed, phases=phases)
        out[name] = {"bits": r["bits"], "exceptions": len(r["exceptions"]), "switched": SWITCH in r["levels"],
                     "evaluations": r["evaluations"], "seconds": round(time.time() - t0, 1)}
    return out


def main() -> None:
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    spec = importlib.import_module(args.spec).SPEC
    run = load_evidence(ROOT / "evidence" / spec["source"])
    builder = run["builder"]
    by_index = {r["index"]: r for r in run["roots"]}
    jobs = []
    for index in spec["roots"]:
        splits = [(board, to_move) for board, to_move, node in by_index[index]["nodes"] if "move" in node]
        rng = random.Random(spec["seed"] * 1_000_003 + index)
        for board, to_move in rng.sample(splits, min(spec["per_root"], len(splits))):
            jobs.append({"root": index, "board": board, "to_move": to_move, "builder": builder,
                         "seed": spec["seed"] + len(jobs), "seconds": spec["seconds"], "cap": spec["cap"],
                         "cache_limit": spec["cache_limit"]})
    with ProcessPoolExecutor(args.workers) as pool:
        rows = []
        for row in pool.map(search_one, jobs):
            rows.append(row)
            print(row, flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "c_sources": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest()[:12]
                                           for s in C_SOURCES},
                             "spec": args.spec, "source": spec["source"], "builder": builder, "rows": rows})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
