"""§3.6 — do maps carry across the frontier? Under a SPEC's rule (its builder's `ties`, `safe_depth`, `forcing`,
`unlevelled`), for every root and every leaf of the strategies in one run's evidence, whether any map found by
another run's builds wins every line from there (the C verify, each map restricted to the cells empty at that
position, as the builder applies it). Saved: per root, whether a map covers the root and how many of its leaves (with
exceptions or without) one covers.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_map_reuse.py --spec harness.floor_l2 --maps c49_l2.json.gz \\
        --targets c49_w9.json.gz --out evidence/c49_map_reuse.json.gz
"""
from __future__ import annotations

import argparse
import importlib
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_MODULES = ("scripts/c4_map_reuse.py", "harness/native_leaf.py",
                       "harness/native_build.py", "harness/steady_state.py")
C_SOURCES = ("harness/c4leafwalk.c", "harness/c4solver.c")
SECONDS = 30.0


def main() -> None:
    import hashlib

    from games.connect4 import C4State, Connect4
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.native_leaf import verify
    from harness.steady_state import Facts

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--maps", required=True, help="evidence file whose builds' maps are offered")
    ap.add_argument("--targets", required=True, help="evidence file whose roots and leaves are tried")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    b = importlib.import_module(args.spec).SPEC["builder"]
    game = Connect4()
    facts = Facts(game, ties=b.get("ties", "exact"), safe_depth=b.get("safe_depth", 1),
                  forcing=b.get("forcing", False), unlevelled=b.get("unlevelled", "none"))
    offered = [{int(c): k for c, k in m.items()}
               for r in load_evidence(ROOT / "evidence" / args.maps)["roots"] for m in r["maps"][1:]]
    seen: set = set()
    maps = [m for m in offered if not (tuple(sorted(m.items())) in seen or seen.add(tuple(sorted(m.items()))))]

    def covered(state) -> bool:
        empty = set(facts.empty_cells(state))
        for m in maps:
            levels = {c: k for c, k in m.items() if c in empty}
            if verify(facts, state, levels, b["n_levels"], b["cap"], deadline=time.monotonic() + SECONDS)["won"]:
                return True
        return False

    rows = []
    for r in load_evidence(ROOT / "evidence" / args.targets)["roots"]:
        if not r["complete"]:
            continue
        root = C4State(tuple(r["board"]), r["to_move"], None, False)
        leaves = [(C4State(tuple(board), to_move, None, False), node) for board, to_move, node in r["nodes"]
                  if "leaf" in node]
        hit = [bool(node.get("exceptions")) for s, node in leaves if covered(s)]
        rows.append({"index": r["index"], "bits": r["bits"]["nodes"], "root_covered": covered(root),
                     "leaves": len(leaves), "excepted": sum(1 for _s, n in leaves if n.get("exceptions")),
                     "leaves_covered": len(hit), "excepted_covered": sum(hit)})
        print(rows[-1], flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "c_sources": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest()[:12]
                                           for s in C_SOURCES},
                             "spec": args.spec, "maps_from": args.maps, "targets": args.targets,
                             "maps": len(maps), "seconds_per_check": SECONDS, "rows": rows})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
