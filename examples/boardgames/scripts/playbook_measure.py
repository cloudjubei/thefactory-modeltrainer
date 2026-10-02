"""§C.50 E1 — measure written-rule playbooks against exact move values.

Sets:
  ttt_all          every raw non-terminal tic-tac-toe position (4,520)
  ttt_own_p0/p1    the positions Newell–Simon reaches in its own play as that player, against every reply
  c4_cache         the fully labelled positions of the Connect-4 label cache (the nets' own strategy trees)
  c4_sample        random-play Connect-4 positions in stone-count bands, labelled by the native solver

Every set is scored with the tactics playbook (win, block, fork); the tic-tac-toe sets also with Newell–Simon.
Results are broken out by stone count.

    PYTHONPATH=. .venv/bin/python scripts/playbook_measure.py --per-band 150 --workers 2 \\
        --out evidence/c50_E1_rules.json.gz
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

MEASUREMENT_MODULES = ("scripts/playbook_measure.py", "harness/playbook.py", "harness/coverage.py",
                       "harness/native_solver.py", "harness/benchmark.py")
BANDS = (10, 16, 22, 28)
LABELS = Path(__file__).resolve().parent.parent / "books" / "c4_labels.json.gz"


def _solve(job):
    from games.connect4 import C4State
    from harness import native_solver

    board, to_move = job
    return native_solver.move_values(C4State(tuple(board), to_move, None, False))


def _stones(state) -> int:
    return sum(1 for c in state.board if c)


def _score(books: dict, game, positions: list, values: list) -> dict:
    from harness.playbook import evaluate

    out = {}
    for name, book in books.items():
        whole = evaluate(book, game, positions, values)
        by_stones = {}
        for n in sorted({_stones(s) for s in positions}):
            idx = [i for i, s in enumerate(positions) if _stones(s) == n]
            r = evaluate(book, game, [positions[i] for i in idx], [values[i] for i in idx])
            by_stones[str(n)] = {k: r[k] for k in ("positions", "covered", "correct")}
        out[name] = {**whole, "by_stones": by_stones}
    return out


def main() -> None:
    import platform
    import random

    from games.connect4 import C4State, Connect4
    from games.tictactoe import TicTacToe
    from harness.benchmark import sample_solvable_positions
    from harness.coverage import move_values, reachable_states
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.playbook import newell_simon, own_play_positions, tactics

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-band", type=int, default=150)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    sets = {}

    ttt = TicTacToe()
    raw, complete = reachable_states(ttt, exact=True, symmetry=False)
    assert complete
    raw = [s for s in raw if not ttt.is_terminal(s)]
    both = {"tactics": tactics(), "newell_simon": newell_simon()}
    sets["ttt_all"] = _score(both, ttt, raw, [move_values(ttt, s) for s in raw])
    root = ttt.initial_state(random.Random(0))
    for player in (0, 1):
        own = own_play_positions(newell_simon(), ttt, root, player)
        sets[f"ttt_own_p{player}"] = _score(both, ttt, own, [move_values(ttt, s) for s in own])
    print({k: {b: (v[b]["covered"], v[b]["correct"], v[b]["positions"]) for b in v} for k, v in sets.items()},
          flush=True)

    c4 = Connect4()
    cache = [] if not LABELS.exists() else load_evidence(LABELS)["positions"]
    full = []
    for row in cache:
        s = C4State(tuple(row["board"]), row["to_move"], None, False)
        vals = {int(a): v for a, v in row["values"].items()}
        if not c4.is_terminal(s) and set(vals) == set(c4.legal_actions(s)):
            full.append((s, vals))
    sets["c4_cache"] = _score({"tactics": tactics()}, c4, [s for s, _v in full], [v for _s, v in full])
    print("c4_cache", {k: sets["c4_cache"]["tactics"][k] for k in ("positions", "covered", "correct")}, flush=True)

    sample = []
    for i, band in enumerate(BANDS):
        sample += sample_solvable_positions(c4, args.per_band, min_moves=band, seed=args.seed + i)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        values = list(pool.map(_solve, [(s.board, s.to_move) for s in sample], chunksize=4))
    sets["c4_sample"] = _score({"tactics": tactics()}, c4, sample, values)
    print("c4_sample", {k: sets["c4_sample"]["tactics"][k] for k in ("positions", "covered", "correct")}, flush=True)

    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the measurement ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "config": {"per_band": args.per_band, "bands": list(BANDS), "seed": args.seed},
                             "books": {"tactics": [r.text() for r in tactics()],
                                       "newell_simon": [r.text() for r in newell_simon()]},
                             "sets": sets})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
