"""Plan §2.3 E2 — how small certified first-player play in Kalah is. Per registered shape (harness.floor_e2), one at a
time in one process: the game graph's reachable positions, then the canonical and the tree-minimising first-player
strategies (harness.strategy_size), each certified over the whole game by harness.certify against the game's exact
values, and sized as a walk-order table (one move per decision, ceil(log2 holes) bits). Judged by
harness.floor_e2.e2_report.

    PYTHONPATH=. .venv/bin/python -u scripts/kalah_strategy_size.py --out evidence/kalah_strategy_size.json.gz
"""
from __future__ import annotations

import argparse
import math
import platform
import resource
import time
from datetime import datetime, timezone

MEASUREMENT_MODULES = ("scripts/kalah_strategy_size.py", "games/kalah.py", "harness/strategy_size.py",
                       "harness/certify.py")


def main() -> None:
    from games.kalah import Kalah
    from harness.certify import certify
    from harness.evidence import save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.floor_e2 import SPEC
    from harness.strategy_size import canonical_strategy, decisions, game_graph_size, min_tree_strategy

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for m, n in SPEC["shapes"]:
        t0 = time.time()
        game = Kalah(m, n)
        root = game.initial_state()
        move_bits = math.ceil(math.log2(m)) if m > 1 else 0

        def sized(choice):
            cert = certify(game, root, 0, lambda states: [choice[game.state_key(s)] for s in states],
                           game.position_value)
            count = decisions(game, root, 0, choice)
            return {"decisions": count, "bits": count * move_bits,
                    "certified": bool(cert["certified"] and cert["complete_game"])}

        best = min_tree_strategy(game, root, 0)
        row = {"shape": [m, n], "value": game.position_value(root), "margin": game.margin(root),
               "graph": game_graph_size(game, root), "canonical": sized(canonical_strategy(game, root, 0)),
               "min_tree": {**sized(best["choice"]), "tree": best["tree"]}, "seconds": round(time.time() - t0, 2)}
        rows.append(row)
        print(f"Kalah({m},{n}): graph {row['graph']}, canonical {row['canonical']['decisions']} decisions, "
              f"minimised {row['min_tree']['decisions']} (tree {row['min_tree']['tree']}), certified "
              f"{row['canonical']['certified']}/{row['min_tree']['certified']}, {row['seconds']}s, peak rss "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1_000_000} MB", flush=True)
        del game, best
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "shapes": rows})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
