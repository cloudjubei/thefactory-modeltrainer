"""§3.6 Q1 — can a leaf's exceptions be told where to fall or what to play? From a seeded sample of a run's leaves
with exceptions, each leaf is walked as harness.exception_coding codes it (every contested position — the map defines
a move, no immediate win, two or more safe moves — flagged or not) and its contested positions are put in four
classes: whether the map's move comes from a levelled cell or the unlevelled last level, and whether the tactics order
(harness.steady_state.Facts `order` "tactics", over every safe move) would play the same. Per leaf: the flags as one
enumerative code over the contested positions and as one per class; each exception's move as one of the safe moves
the map did not play and, tactics first, a bit for whether it is the tactics move where that is one of them. The
coded flags and moves must reproduce the leaf's recorded walk-order bits, or nothing is written.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_exception_classes.py --spec harness.floor_q1 \\
        --out evidence/c49_q1.json.gz
"""
from __future__ import annotations

import argparse
import importlib
import math
import platform
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_MODULES = ("scripts/c4_exception_classes.py", "harness/exception_coding.py", "harness/steady_state.py",
                       "harness/steady_exceptions.py", "games/connect4.py")
CLASSES = ("level_agree", "level_disagree", "default_agree", "default_disagree")


def _width(n: int) -> int:
    return math.ceil(math.log2(n)) if n > 1 else 0


def leaf_classes(facts, tactics, root, levels: dict, n_levels: int, exceptions: dict) -> dict:
    from harness.exception_coding import enumerative_bits
    from harness.steady_exceptions import choose_with
    from harness.steady_state import _phase, choose

    game = facts.game
    player = game.current_player(root)
    contested, flagged, moves = Counter(), Counter(), Counter()
    seen: set = set()
    stack = [root]
    while stack:
        s = stack.pop()
        key = game.state_key(s)
        if key in seen or game.is_terminal(s):
            continue
        seen.add(key)
        if game.current_player(s) != player:
            stack.extend(game.step(s, b) for b in game.legal_actions(s))
            continue
        wins, _, cell = facts.of(s)
        safe = facts.safe_moves(s)
        move = choose(facts, s, levels, n_levels)
        fix = exceptions.get(key)
        if fix is not None:
            alternatives = len(safe) - (move is not None)
            moves["plain"] += _width(alternatives)
            hint = tactics.pick(s, safe)
            if hint == move or alternatives <= 1:
                moves["tactics_first"] += _width(alternatives)
            else:
                moves["tactics_first"] += 1 + (0 if fix == hint else _width(alternatives - 1))
            moves["exceptions"] += 1
            moves["hits"] += fix == hint
        if move is not None and not wins and len(safe) > 1:
            levelled = cell[move] in _phase(facts, s, levels)
            agree = tactics.pick(s, safe) == move
            c = f"{'level' if levelled else 'default'}_{'agree' if agree else 'disagree'}"
            contested[c] += 1
            flagged[c] += fix is not None
        stack.append(game.step(s, choose_with(facts, s, levels, n_levels, exceptions)))
    n, k = sum(contested.values()), sum(flagged.values())
    return {"contested": dict(contested), "flagged": dict(flagged), "one_class": enumerative_bits(n, k),
            "per_class": sum(enumerative_bits(contested[c], flagged[c]) for c in CLASSES), **moves}


def main() -> None:
    from games.connect4 import C4State, Connect4
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.steady_state import Facts

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    spec = importlib.import_module(args.spec).SPEC
    run = load_evidence(ROOT / "evidence" / spec["source"])
    b = run["builder"]
    game = Connect4()
    kw = {"ties": b.get("ties", "exact"), "safe_depth": b.get("safe_depth", 1), "forcing": b.get("forcing", False),
          "unlevelled": b.get("unlevelled", "none")}
    facts = Facts(game, **kw, order=b.get("order", "lowest"))
    tactics = Facts(game, **{**kw, "ties": "lowest"}, order="tactics")
    leaves = [(r, board, to_move, n) for r in run["roots"] for board, to_move, n in r["nodes"] if n.get("exceptions")]
    rows = []
    for r, board, to_move, n in random.Random(spec["seed"]).sample(leaves, spec["leaves"]):
        root = C4State(tuple(board), to_move, None, False)
        levels = {int(c): k for c, k in r["maps"][n["leaf"]].items()}
        exceptions = {(tuple(x), t): m for x, t, m in n["exceptions"]}
        row = leaf_classes(facts, tactics, root, levels, b["n_levels"], exceptions)
        if row["one_class"] + row["plain"] != n["coded"]:
            raise SystemExit(f"root #{r['index']}: the walk does not reproduce the leaf's {n['coded']} coded bits")
        rows.append({"index": r["index"], "coded": n["coded"], **row})
        print({k: row[k] for k in ("one_class", "per_class", "plain", "tactics_first")}, flush=True)
    totals = {k: sum(row[k] for row in rows) for k in ("coded", "one_class", "per_class", "plain", "tactics_first",
                                                        "exceptions", "hits")}
    print(totals, flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "spec": args.spec, "source": spec["source"], "seed": spec["seed"], "leaves": len(rows),
                             "totals": totals, "rows": rows})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
