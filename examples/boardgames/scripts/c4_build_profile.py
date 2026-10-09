"""§3.6 — where a frontier build's time goes near the root: one of W1's frontier positions built as a registered spec
builds it (default harness.floor_p2, root #316; `--budgets` overrides its search budgets; the spec's walker and
library are used) for a short cap under cProfile, and the functions with the most
cumulative time saved, with the wall time spent inside the leaf searches (and how far each search ran past the budget
the builder gave it) measured directly around the search calls.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_build_profile.py --spec harness.floor_p2 --index 316 --seconds 900 \\
        --out evidence/c49_build_profile.json.gz
"""
from __future__ import annotations

import argparse
import cProfile
import importlib
import platform
import pstats
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_MODULES = ("scripts/c4_build_profile.py", "scripts/c4_strategy_s2.py", "harness/strategy_builder.py",
                       "harness/native_leaf.py",
                       "harness/steady_local.py", "harness/steady_exceptions.py", "harness/steady_state.py",
                       "harness/winning_cache.py", "harness/exception_coding.py", "harness/opening.py",
                       "scripts/c4_steady_states.py")


def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from c4_steady_states import _winning
    from c4_strategy_s2 import _opening

    from games.connect4 import C4State, Connect4
    from harness.evidence import save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.native_leaf import needed as native_walk
    from harness.steady_exceptions import needed
    from harness.evidence import load_evidence
    from harness.steady_local import local_search
    from harness.steady_state import Facts
    from harness.strategy_builder import Builder
    from harness.winning_cache import CachedWinning

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="harness.floor_p2")
    ap.add_argument("--index", type=int, default=316)
    ap.add_argument("--seconds", type=float, default=900.0)
    ap.add_argument("--budgets", default=None, help="override the builder's budgets, e.g. 0:60 (depth:seconds,...)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = importlib.import_module(args.spec).SPEC
    if args.budgets:
        pairs = [[int(d), float(s)] for d, s in (x.split(":") for x in args.budgets.split(","))]
        spec = {**spec, "builder": {**spec["builder"], "budgets": pairs}}
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _stats, rows = _opening(spec)
    row = rows[args.index]
    game = Connect4()
    root = C4State(tuple(row["board"]), row["to_move"], None, False)
    facts = Facts(game)
    b, s = spec["builder"], spec["search"]
    winning = CachedWinning(lambda states: _winning(game, states), game.state_key, s["cache_limit"])
    rng = random.Random(args.index)
    walker = native_walk if s.get("walker") == "native" else needed
    earlier = load_evidence(ROOT / "evidence" / spec["library_from"])["roots"] if spec.get("library_from") else []
    library = [({int(c): k for c, k in levels.items()}, empty) for r in earlier for levels, empty in r["found_maps"]]
    searches = []

    def search(st, start, seconds):
        budget = s["seconds"] if seconds is None else seconds
        t0 = time.monotonic()
        r = local_search(facts, st, b["n_levels"], b["level_bits"], winning, rng, budget, s["cap"], start, walker)
        searches.append({"pieces": sum(1 for v in st.board if v), "budget": budget,
                         "seconds": round(time.monotonic() - t0, 2), "evaluations": r["evaluations"],
                         "own_positions": r["own_positions"], "pure": r["status"] == "found"})
        return None if r["bits"] is None else {k: r[k] for k in ("levels", "exceptions", "own_positions")}

    builder = Builder(facts, winning, search, b["n_levels"], b["level_bits"], b["cap"], b["min_leaf_depth"],
                      b["reuse_window"], args.seconds, b.get("accept"), b.get("budgets"), library,
                      choose_by_size=b.get("choose_by_size", False))
    profile = cProfile.Profile()
    t0 = time.monotonic()
    profile.enable()
    complete = builder.build(root)
    profile.disable()
    wall = time.monotonic() - t0
    stats = pstats.Stats(profile)
    top = sorted(((f"{Path(f).name}:{line}:{name}", v[3]) for (f, line, name), v in stats.stats.items()),
                 key=lambda kv: -kv[1])[:40]
    in_search = sum(x["seconds"] for x in searches)
    print(f"wall {wall:.0f}s, in searches {in_search:.0f}s over {len(searches)} searches, complete {complete}",
          flush=True)
    for name, cum in top[:15]:
        print(f"  {cum:8.1f}s  {name}", flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "spec": args.spec, "index": args.index, "cap_seconds": args.seconds,
                             "budgets": spec["builder"].get("budgets"),
                             "wall_seconds": round(wall, 1), "complete": complete, "searches": searches,
                             "search_seconds": round(in_search, 1),
                             "top_cumulative": [{"function": n, "seconds": round(c, 1)} for n, c in top]})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
