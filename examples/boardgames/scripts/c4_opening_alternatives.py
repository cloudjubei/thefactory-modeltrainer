"""§3.6 — can the opening steer around the heavy frontier positions? For every first-player position at the frontier's
parent ply (the opening's last decisions, ply 6 under harness.floor_w6) the winning moves there (the label cache) and,
for each, the replies that leave a position the empty map does not win — the frontier positions that move would create
(their index in the opening's frontier when the opening reaches them). Saved with the built sizes of the frontier
positions W5b and W6 sampled, so the decisions above the heavy ones can be read against their alternatives.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_opening_alternatives.py --out evidence/c49_opening_alternatives.json.gz
"""
from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_MODULES = ("scripts/c4_opening_alternatives.py", "scripts/c4_strategy_s2.py", "harness/opening.py",
                       "harness/steady_state.py")
BUILT = ("c49_w5b.json.gz", "c49_w6.json.gz")


def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from c4_strategy_s2 import LABELS, _opening

    from games.connect4 import C4State, Connect4
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.floor_w6 import SPEC
    from harness.opening import frontier
    from harness.steady_state import Facts, verify

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    game = Connect4()
    facts = Facts(game)
    n_levels, cap = SPEC["builder"]["n_levels"], SPEC["builder"]["cap"]
    book = {(tuple(r["board"]), r["to_move"]): sorted(int(a) for a, v in r["values"].items() if v == 1)
            for r in load_evidence(LABELS)["positions"]}
    _stats, rows = _opening(SPEC)
    index = {(tuple(r["board"]), r["to_move"]): i for i, r in enumerate(rows)}
    opening = frontier(facts, game.initial_state(), lambda states: [set(book[game.state_key(s)]) for s in states],
                       SPEC["ply"], n_levels, cap)
    decisions = []
    for key, move in opening["moves"].items():
        if sum(1 for v in key[0] if v) != SPEC["ply"] - 2:
            continue
        s = C4State(tuple(key[0]), key[1], None, False)
        unwon = {}
        for a in book[key]:
            after = game.step(s, a)
            left = []
            for b in ([] if game.is_terminal(after) else game.legal_actions(after)):
                c = game.step(after, b)
                if not game.is_terminal(c) and not verify(facts, c, {}, n_levels, cap)["won"]:
                    left.append(index.get(game.state_key(c), -1))
            unwon[str(a)] = left
        decisions.append({"board": list(key[0]), "to_move": key[1], "move": move, "winning": book[key],
                          "unwon": unwon})
    built = {r["index"]: r["bits"]["nodes"] for f in BUILT for r in load_evidence(ROOT / "evidence" / f)["roots"]}
    print(f"{len(decisions)} decisions at ply {SPEC['ply'] - 2}, {sum(len(d['winning']) > 1 for d in decisions)} "
          f"with an alternative; {len(built)} frontier sizes", flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "spec": "harness.floor_w6", "frontier": len(rows), "decisions": decisions,
                             "built": {str(i): n for i, n in built.items()}})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
