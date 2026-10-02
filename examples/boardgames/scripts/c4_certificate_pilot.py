"""§C.50 C1 — the Allis-style value certificates (harness/c4_certificates.py) against exact values.

Sets:
  cache    the White-to-move positions of the Connect-4 label cache (the nets' own strategy trees, plies 0-8): every
           certificate and every move flag checked against the recorded values; certified positions whose value is
           not recorded are solved
  random   random-play positions in stone-count bands, every move solved by the native solver
  opening  the D2 opening positions (evidence/c49_D2_opening.json.gz): the share of the search's NON-OPTIMAL
           preferred White moves that the certificates flag as not winning, by budget and ply

A flag counts as BY CERTIFICATE when no Black reply simply ends the game (the rest are the tactic "gives an
immediate win"). A contradiction is a certified position the solver scores as a White win, or a flagged move that wins.

    PYTHONPATH=. .venv/bin/python scripts/c4_certificate_pilot.py --workers 6 --out evidence/c50_C1_certificates.json.gz
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "books" / "c4_labels.json.gz"
OPENING = ROOT / "evidence" / "c49_D2_opening.json.gz"
MEASUREMENT_MODULES = ("scripts/c4_certificate_pilot.py", "harness/c4_certificates.py", "harness/native_solver.py",
                       "harness/benchmark.py")
CONFIG = {"bands": [20, 26, 32, 38], "per_band": 150, "seed": 98, "opening_max_ply": 4}


def _move_values(job):
    from games.connect4 import C4State
    from harness import native_solver

    board, to_move = job
    return native_solver.move_values(C4State(tuple(board), to_move, None, False))


def _value(job):
    from games.connect4 import C4State
    from harness import native_solver

    board, to_move = job
    return native_solver.solve_position(C4State(tuple(board), to_move, None, False))


def _tally(game, state, values: dict, position_value) -> dict:
    from harness.c4_certificates import black_draw_certificate, move_not_winning

    certified = bool(black_draw_certificate(state))
    out = {"positions": 1, "certified": int(certified), "certified_won": int(certified and position_value == 1),
           "moves": 0, "non_winning": 0, "flagged": 0, "flagged_winning": 0, "flagged_by_certificate": 0}
    for m, v in values.items():
        flagged = move_not_winning(game, state, m)
        child = game.step(state, m)
        tactical = game.is_terminal(child) or any(game.is_terminal(game.step(child, b))
                                                  for b in game.legal_actions(child))
        out["moves"] += 1
        out["non_winning"] += int(v <= 0)
        out["flagged"] += int(flagged)
        out["flagged_winning"] += int(flagged and v == 1)
        out["flagged_by_certificate"] += int(flagged and not tactical)
    return out


def _add(into: dict, key, row: dict) -> None:
    slot = into.setdefault(str(key), {k: 0 for k in row})
    for k, v in row.items():
        slot[k] += v


def main() -> None:
    import platform

    from games.connect4 import C4State, Connect4
    from harness.benchmark import sample_solvable_positions
    from harness.c4_certificates import black_draw_certificate, move_not_winning
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    game = Connect4()
    stones = lambda s: sum(1 for c in s.board if c)

    cache, recorded = {}, {}
    rows = [r for r in load_evidence(LABELS)["positions"] if r["to_move"] == 0]
    for r in rows:
        s = C4State(tuple(r["board"]), 0, None, False)
        recorded[s.board] = {int(a): int(v) for a, v in r["values"].items()}
    unknown = []
    for board, vals in recorded.items():
        s = C4State(board, 0, None, False)
        full = set(vals) == set(game.legal_actions(s))
        if not full and 1 not in vals.values() and black_draw_certificate(s):
            unknown.append(s)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        solved = dict(zip([s.board for s in unknown], pool.map(_value, [(s.board, 0) for s in unknown])))
        for board, vals in recorded.items():
            s = C4State(board, 0, None, False)
            full = set(vals) == set(game.legal_actions(s))
            value = max(vals.values()) if full or 1 in vals.values() else solved.get(board)
            _add(cache, stones(s), _tally(game, s, vals, value))
        print("cache", cache, flush=True)

        sample = []
        for i, band in enumerate(CONFIG["bands"]):
            sample += [(band, s) for s in sample_solvable_positions(game, 2 * CONFIG["per_band"], min_moves=band,
                                                                    seed=CONFIG["seed"] + i) if s.to_move == 0
                       ][:CONFIG["per_band"]]
        values = list(pool.map(_move_values, [(s.board, 0) for _b, s in sample], chunksize=4))
    random_set = {}
    for (band, s), vals in zip(sample, values):
        _add(random_set, band, _tally(game, s, vals, max(vals.values())))
    print("random", random_set, flush=True)

    opening = {}
    optimal_flags = []
    for seed in load_evidence(OPENING)["seeds"]:
        for p in seed["positions"]:
            if p["to_move"] != 0 or p["ply"] > CONFIG["opening_max_ply"]:
                continue
            s = C4State(tuple(p["board"]), 0, None, False)
            for m in p["optimal"]:
                if move_not_winning(game, s, m):
                    optimal_flags.append(s)
            for budget, pref in p["preferred"].items():
                if pref not in p["optimal"]:
                    _add(opening.setdefault(budget, {}), p["ply"],
                         {"non_optimal": 1, "flagged": int(move_not_winning(game, s, pref))})
    won_flagged = sum(1 for s in optimal_flags if _value((s.board, 0)) == 1)
    print("opening", opening, "flagged optimal moves", len(optimal_flags), "in won positions", won_flagged, flush=True)

    contradictions = (sum(v["certified_won"] + v["flagged_winning"] for d in (cache, random_set) for v in d.values())
                      + won_flagged)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the measurement ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "config": CONFIG, "cache_unknown_solved": len(unknown),
                             "sets": {"cache": cache, "random": random_set, "opening": opening},
                             "opening_flagged_optimal": len(optimal_flags),
                             "opening_flagged_optimal_won": won_flagged, "contradictions": contradictions})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
