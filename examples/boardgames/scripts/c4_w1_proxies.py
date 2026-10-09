"""§3.6 W1 — cheap stand-ins for a frontier position's strategy size, measured on W1's 8 sampled positions
(evidence/c49_w1.json.gz) to choose an opening for small frontier strategies: (1) the fewest first-player positions
two plies below that the empty map does not win, over the position's winning moves (label cache, else the exact
solver); (2) the exact solver's strong score (how soon the first player can force the win). Saved beside each
position's measured strategy size.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_w1_proxies.py --out evidence/c49_w1_proxies.json.gz
"""
from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_MODULES = ("scripts/c4_w1_proxies.py", "scripts/c4_steady_states.py", "harness/steady_state.py",
                       "harness/native_solver.py", "harness/c4_oracle.py")


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from c4_steady_states import _winning

    from games.connect4 import C4State, Connect4
    from harness import native_solver
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.solver import to_bitboard
    from harness.steady_state import Facts, verify

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    game = Connect4()
    facts = Facts(game)
    book = {(tuple(r["board"]), r["to_move"]): {int(a) for a, v in r["values"].items() if v == 1}
            for r in load_evidence(ROOT / "books" / "c4_labels.json.gz")["positions"]}
    lib = native_solver._library()

    def left_below(s) -> int:
        counts = []
        for a in sorted(book.get(game.state_key(s)) or _winning(game, [s])[0]):
            after = game.step(s, a)
            counts.append(0 if game.is_terminal(after) else sum(
                1 for b in game.legal_actions(after) for c in [game.step(after, b)]
                if not game.is_terminal(c) and not verify(facts, c, {}, 8, 1_000_000)["won"]))
        return min(counts)

    w1 = load_evidence(ROOT / "evidence" / "c49_w1.json.gz")
    rows = []
    for r in w1["roots"]:
        s = C4State(tuple(r["board"]), r["to_move"], None, False)
        position, mask, moves = to_bitboard(s)
        rows.append({"index": r["index"], "bits": r["bits"]["nodes"], "complete": r["complete"],
                     "left_below": left_below(s), "strong_score": int(lib.c4_solve(position, mask, moves, 0))})
        print(rows[-1], flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "source": {"file": "c49_w1.json.gz", "started": w1["started"]},
                             "rows": sorted(rows, key=lambda r: r["index"])})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
