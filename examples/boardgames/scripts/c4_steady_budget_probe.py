"""§3.6 diagnostic — what stops steady states at ply 10: the search budget or the language? Re-searches the ply-10
positions the S1 pilot (evidence/c49_s1_pilot.json.gz) left out of budget, each under two settings: the registered
8 levels with a far larger budget, and 14 levels with the same budget. Per position and setting: status, constraints,
rounds, search seconds and, when found, the simplified map's bits and the first-player positions it plays (verified
by the solver-free walk and harness.certify, as in S1).

    PYTHONPATH=. .venv/bin/python -u scripts/c4_steady_budget_probe.py --workers 4 \\
        --out evidence/c49_s1_budget_probe.json.gz
"""
from __future__ import annotations

import argparse
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_MODULES = ("scripts/c4_steady_budget_probe.py", "scripts/c4_steady_states.py", "harness/steady_state.py",
                       "harness/steady_search.py", "harness/certify.py", "harness/c4_oracle.py",
                       "harness/native_solver.py")
SETTINGS = {"budget": {"n_levels": 8, "max_constraints": 200_000, "seconds": 1800.0},
            "levels": {"n_levels": 14, "max_constraints": 200_000, "seconds": 1800.0}}


def probe(job: dict) -> dict:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from c4_steady_states import _value, _winning

    from games.connect4 import C4State, Connect4
    from harness.certify import certify
    from harness.steady_search import find_steady_state
    from harness.steady_state import Facts, choose, map_bits, simplify, verify

    game = Connect4()
    root = C4State(tuple(job["board"]), job["to_move"], None, False)
    facts = Facts(game)
    s = SETTINGS[job["setting"]]
    t0 = time.time()
    found = find_steady_state(facts, root, lambda st: _winning(game, st), s["n_levels"],
                              max_constraints=s["max_constraints"], conflicts=1_000_000, seconds=s["seconds"],
                              cap=1_000_000, lines=64)
    row = {"index": job["index"], "setting": job["setting"], "status": found["status"],
           "constraints": found["constraints"], "iterations": found["iterations"],
           "search_seconds": round(time.time() - t0, 1)}
    if found["status"] != "found":
        return row
    levels = simplify(facts, root, found["levels"], s["n_levels"], 1_000_000)
    walk = verify(facts, root, levels, s["n_levels"], 1_000_000)
    cert = certify(game, root, 0, lambda xs: [choose(facts, x, levels, s["n_levels"]) for x in xs],
                   lambda x: _value(game, x), root_value=1)
    level_bits = max(1, (s["n_levels"] - 1).bit_length())
    return {**row, "bits": map_bits(levels, len(facts.empty_cells(root)), level_bits),
            "own_positions": walk["own_positions"], "verified": walk["won"],
            "certified": bool(cert["certified"] and cert["complete_game"])}


def main() -> None:
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pilot = load_evidence(ROOT / "evidence" / "c49_s1_pilot.json.gz")["results"]
    targets = [r for r in pilot if r["ply"] == 10 and r["status"] == "budget"]
    jobs = [{"index": r["index"], "board": r["board"], "to_move": r["to_move"], "setting": name}
            for r in targets for name in SETTINGS]
    print(f"{len(targets)} positions x {len(SETTINGS)} settings", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(probe, j) for j in jobs]):
            r = fut.result()
            rows.append(r)
            print(r, flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the probe ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "settings": SETTINGS, "rows": sorted(rows, key=lambda r: (r["index"], r["setting"]))})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
