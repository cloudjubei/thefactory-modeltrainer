"""§C.49 H1 — build each registered net's exception table (harness.opening_table) at each horizon and certify the
hybrid one White ply past it (harness.certify, through horizon + 2 plies, P-START). Exact values come from the label
cache where it records them and from the native solver otherwise (the process's "solve this position" step; the
evidence counts both). Judged by harness.floor_hybrid.hybrid_report.

    PYTHONPATH=. .venv/bin/python scripts/c4_hybrid.py --workers 8 --out evidence/c49_H1_hybrid.json.gz
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "books" / "c4_labels.json.gz"
MEASUREMENT_MODULES = ("scripts/c4_hybrid.py", "harness/opening_table.py", "harness/certify.py",
                       "harness/c4_oracle.py", "harness/native_solver.py")
EMPTY_BOARD_VALUE = 1


class _Values:
    """Exact values to the side to move: recorded ones first, then batched native solves."""

    def __init__(self, game, pool, rows: list):
        from games.connect4 import C4State

        self.game, self.pool = game, pool
        self.moves: dict = {}
        self.position: dict = {}
        self.counts = {"recorded": 0, "solved": 0}
        for row in rows:
            s = C4State(tuple(row["board"]), row["to_move"], None, False)
            values = {int(a): int(v) for a, v in row["values"].items()}
            for a, v in values.items():
                child = game.step(s, a)
                if not game.is_terminal(child):
                    self.position[game.state_key(child)] = -v
            if set(values) == set(game.legal_actions(s)):
                self.moves[game.state_key(s)] = values
                self.position[game.state_key(s)] = max(values.values())

    def positions(self, states: list) -> list:
        from harness.c4_oracle import solve

        todo = list({self.game.state_key(s): s for s in states if self.game.state_key(s) not in self.position}.values())
        for s, v in zip(todo, self.pool.map(solve, [(s.board, s.to_move) for s in todo], chunksize=2)):
            self.position[self.game.state_key(s)] = int(v)
        self.counts["solved"] += len(todo)
        self.counts["recorded"] += len(states) - len(todo)
        return [self.position[self.game.state_key(s)] for s in states]

    def move_values(self, states: list) -> list:
        todo = [s for s in states if self.game.state_key(s) not in self.moves]
        children = [(s, a, self.game.step(s, a)) for s in todo for a in self.game.legal_actions(s)]
        open_children = [c for _s, _a, c in children if not self.game.is_terminal(c)]
        values = iter(self.positions(open_children))
        for s, a, c in children:
            mover = self.game.current_player(s)
            v = round(self.game.returns(c)[mover]) if self.game.is_terminal(c) else -next(values)
            self.moves.setdefault(self.game.state_key(s), {})[a] = v
        return [self.moves[self.game.state_key(s)] for s in states]


def main() -> None:
    import hashlib
    import platform
    import random

    import torch

    from games.connect4 import Connect4
    from harness.certify import certify
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.floor_hybrid import SPEC
    from harness.neural import load_net
    from harness.opening_table import exception_table, hybrid_chooser
    from harness.strategy_tree import raw_chooser

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    game = Connect4()
    root = game.initial_state(random.Random(0))
    nets = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        values = _Values(game, pool, load_evidence(LABELS)["positions"])
        for run, seeds in SPEC["runs"]:
            trained = {r["seed"]: r for r in load_evidence(ROOT / "evidence" / run)["seeds"]}
            for seed in seeds:
                row = trained[seed]
                path = ROOT / row["net"]["path"]
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                if sha != row["net"]["file_sha256"]:
                    raise SystemExit(f"{run} seed {seed}: the saved net is not the one its run recorded")
                choose = raw_chooser(game, load_net(str(path)))
                readings = {}
                for h in SPEC["horizons"]:
                    t0 = time.time()
                    table, stats = exception_table(game, root, 0, choose, values.move_values, h)
                    cert = certify(game, root, 0, hybrid_chooser(game, table, choose),
                                   lambda s: values.positions([s])[0], max_depth=h + 2, value_many=values.positions,
                                   root_value=EMPTY_BOARD_VALUE)
                    readings[str(h)] = {"depth": h + 2, "entries": len(table), "positions": stats["positions"],
                                        "positions_by_ply": stats["positions_by_ply"],
                                        "table": [[list(k[0]), k[1], m] for k, m in table.items()],
                                        "certified": cert["certified"], "failures": cert["failures"],
                                        "failures_by_ply": {str(p): c for p, c in cert["failures_by_ply"].items()},
                                        "nodes": cert["nodes"], "seconds": round(time.time() - t0, 1)}
                    print(f"{run} seed {seed} horizon {h}: {len(table)} entries over {stats['positions']} positions; "
                          f"certified through {h + 2} = {cert['certified']} {cert['failures_by_ply']}; "
                          f"values {values.counts}", flush=True)
                nets.append({"run": run, "seed": seed, "net_file_sha256": sha, "horizons": readings})
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the measurement ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version(),
                                                                            "torch": torch.__version__},
                             "horizons": SPEC["horizons"], "runs": SPEC["runs"], "values": values.counts,
                             "nets": nets})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
