"""§3.6 — the S4 strategies (evidence/c49_s4.json.gz) re-costed with exceptions coded in walk order
(harness.exception_coding): every leaf that carries exceptions is replayed with the game's rules, its map and its
exceptions, and charged the enumerative code over its contested positions plus each exception's move among the safe
moves left, instead of its count and the sparse list or mask the builder charged. No oracle is used; the strategies
are unchanged — only their size is recomputed. The charged cost is recomputed too and must match the run's.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_exception_coding.py --out evidence/c49_s4_exception_coding.json.gz
"""
from __future__ import annotations

import argparse
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "evidence" / "c49_s4.json.gz"
MEASUREMENT_MODULES = ("scripts/c4_exception_coding.py", "harness/exception_coding.py", "harness/steady_exceptions.py",
                       "harness/steady_state.py")


def recost(root: dict, n_levels: int) -> dict:
    from games.connect4 import C4State, Connect4
    from harness.exception_coding import walk_order_cost
    from harness.steady_exceptions import exception_bits
    from harness.steady_state import Facts

    game = Connect4()
    facts = Facts(game)
    maps = [{int(c): k for c, k in m.items()} for m in root["maps"]]
    charged = walk = 0
    leaves = []
    for board, to_move, node in root["nodes"]:
        if not node.get("exceptions"):
            continue
        state = C4State(tuple(board), to_move, None, False)
        empty = set(facts.empty_cells(state))
        levels = {c: k for c, k in maps[node["leaf"]].items() if c in empty}
        exc = {(tuple(b), t): m for b, t, m in node["exceptions"]}
        old = math.ceil(math.log2(node["own"] + 1)) + exception_bits(len(exc), node["own"], game.num_actions)
        new = walk_order_cost(facts, state, levels, n_levels, exc)
        charged += old
        walk += new["bits"]
        leaves.append({"exceptions": len(exc), "own": node["own"], "charged": old, **new})
    flags = root["bits"]["leaves"] if leaves else 0
    if root["bits"]["exception_bits"] != flags + charged:
        raise SystemExit(f"root #{root['index']}: recomputed exception bits {flags + charged} are not the run's "
                         f"{root['bits']['exception_bits']}")
    nodes = root["bits"]["nodes"] - charged + walk
    return {"index": root["index"], "complete": root["complete"], "own_positions": root["own_positions"],
            "charged_bits": charged, "walk_bits": walk, "flags": flags, "nodes_charged": root["bits"]["nodes"],
            "nodes_walk": nodes, "leaves": leaves}


def main() -> None:
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source = load_evidence(SOURCE)
    rows = []
    for root in source["roots"]:
        r = recost(root, source["builder"]["n_levels"])
        rows.append(r)
        print({k: v for k, v in r.items() if k != "leaves"}, flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "source": {"file": SOURCE.name, "started": source["started"]},
                             "roots": sorted(rows, key=lambda r: r["index"])})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
