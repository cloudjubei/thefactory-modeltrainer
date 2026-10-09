"""§3.6 — the C leaf walk (harness.native_leaf) against the Python reference (harness.steady_exceptions.needed) on
frontier positions of the W1/W2 opening, under the empty map: the same result required, and the time of a first
(cold) walk and of a repeated (warm) walk for each, as local search repeats walks over mostly the same positions.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_native_leaf_bench.py --out evidence/c49_native_leaf_bench.json.gz
    (re-run as evidence/c49_native_leaf_bench_lazy.json.gz after the walk solved only the moves it needs, h210)
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_MODULES = ("scripts/c4_native_leaf_bench.py", "harness/native_leaf.py", "harness/steady_exceptions.py",
                       "harness/steady_state.py", "harness/winning_cache.py", "scripts/c4_steady_states.py")
C_SOURCES = ("harness/c4leafwalk.c", "harness/c4solver.c")
INDICES = (393, 346, 87, 263)


def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from c4_steady_states import _winning
    from c4_strategy_s2 import _opening

    from games.connect4 import C4State, Connect4
    from harness.evidence import save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.floor_w2 import SPEC
    from harness.native_leaf import needed as native
    from harness.steady_exceptions import needed as reference
    from harness.steady_state import Facts
    from harness.winning_cache import CachedWinning

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    game = Connect4()
    _stats, rows = _opening(SPEC)
    out = []
    for index in INDICES:
        r = rows[index]
        root = C4State(tuple(r["board"]), r["to_move"], None, False)
        timings = {}
        for name, walk, oracle in (("native", native, None),
                                   ("python", reference,
                                    CachedWinning(lambda st: _winning(game, st), game.state_key, 1_500_000))):
            facts = Facts(game)
            t0 = time.perf_counter()
            first = walk(facts, root, {}, 8, oracle, 1_000_000)
            t1 = time.perf_counter()
            walk(facts, root, {}, 8, oracle, 1_000_000)
            timings[name] = {"cold": round(t1 - t0, 3), "warm": round(time.perf_counter() - t1, 3), "result": first}
        same = timings["native"]["result"] == timings["python"]["result"]
        row = {"index": index, "same": same, "own_positions": timings["native"]["result"]["own_positions"],
               "exceptions": len(timings["native"]["result"]["exceptions"]),
               **{f"{n}_{k}": timings[n][k] for n in ("native", "python") for k in ("cold", "warm")}}
        out.append(row)
        print(row, flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "c_sources": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest()[:12]
                                           for s in C_SOURCES},
                             "rows": out})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
