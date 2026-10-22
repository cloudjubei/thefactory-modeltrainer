"""§3.6 R1 — a whole-game strategy's leaf references under exact cheaper codes (harness.reference_coding): every
root's leaves read in the decoder's order and coded with the builder's fixed width and with first_use. Every root
must be complete, its walk must meet every leaf the build kept, and the fixed code must reproduce the reference bits
the evidence charged root by root (its bits less one a node, the table moves, maps and exceptions), or nothing is
written. Saved: the totals per code and per root the leaves, maps, library size and both codes' bits.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_reference_coding.py --spec harness.floor_r1 \\
        --out evidence/c49_r1.json.gz
"""
from __future__ import annotations

import argparse
import importlib
import math
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_MODULES = ("scripts/c4_reference_coding.py", "harness/reference_coding.py", "games/connect4.py")


def main() -> None:
    from games.connect4 import C4State, Connect4
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.reference_coding import CODES, code_bits, walk_leaves

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    spec = importlib.import_module(args.spec).SPEC
    game = Connect4()
    move_bits = math.ceil(math.log2(game.num_actions))
    totals = Counter()
    rows = []
    for r in load_evidence(ROOT / "evidence" / spec["source"])["roots"]:
        if not r["complete"]:
            raise SystemExit(f"root #{r['index']} is unfinished — its references cannot be walked; nothing written")
        nodes = {(tuple(b), t): n for b, t, n in r["nodes"]}
        refs = walk_leaves(game, C4State(tuple(r["board"]), r["to_move"], None, False), nodes)
        if sorted(refs) != sorted(n["leaf"] for n in nodes.values() if "leaf" in n):
            raise SystemExit(f"root #{r['index']}: the walk does not meet every leaf the build kept; nothing written")
        b = r["bits"]
        charged = (b["nodes"] - len(nodes) - b["moves"] * move_bits - b["map_bits"] - b["exception_bits"])
        bits = {c: code_bits(c, refs, len(r["maps"]), r["shared"]) for c in CODES}
        if bits["fixed"] != charged:
            raise SystemExit(f"root #{r['index']}: the fixed code gives {bits['fixed']} reference bits, the evidence "
                             f"{charged} — not the evidence's accounting; nothing written")
        totals.update(bits)
        rows.append({"index": r["index"], "leaves": len(refs), "maps": len(r["maps"]), "shared": r["shared"],
                     "nodes": b["nodes"], **bits})
    print({**totals, "roots": len(rows)}, flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "spec": args.spec, "source": spec["source"], "roots": len(rows),
                             "totals": dict(totals), "rows": rows})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
