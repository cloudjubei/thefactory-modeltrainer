"""§3.6 option 1 — map-plus-exceptions leaves at ply 10, searched by their size: on the eight ply-10 positions the S1
pilot left out of budget, harness.steady_local hill-climbs priority maps (8 levels, 3 bits a level, 30 minutes)
scored by the complete leaf each makes — map bits plus the exceptions it needs over every line (h170: the earlier
failure count stopped at an undefined root). The oracle's winning moves are cached across a position's walks. The
smallest leaf is checked by the solver-free walk with its exceptions and by harness.certify; the empty map's leaf
(the start) is recorded beside it.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_steady_leaf_probe.py --workers 4 \\
        --out evidence/c49_s1_leaf_probe.json.gz
"""
from __future__ import annotations

import argparse
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_MODULES = ("scripts/c4_steady_leaf_probe.py", "scripts/c4_steady_states.py", "harness/steady_local.py",
                       "harness/steady_exceptions.py", "harness/steady_state.py", "harness/certify.py",
                       "harness/c4_oracle.py", "harness/native_solver.py")
N_LEVELS, LEVEL_BITS, SECONDS, CAP = 8, 3, 1800.0, 1_000_000


def probe(job: dict) -> dict:
    import random
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from c4_steady_states import _value, _winning

    from games.connect4 import C4State, Connect4
    from harness.certify import certify
    from harness.steady_exceptions import choose_with, verify_with
    from harness.steady_local import leaf_bits, local_search
    from harness.steady_state import Facts

    game = Connect4()
    root = C4State(tuple(job["board"]), job["to_move"], None, False)
    facts = Facts(game)
    cache: dict = {}

    def winning(states):
        out = []
        for s in states:
            key = game.state_key(s)
            if key not in cache:
                cache[key] = _winning(game, [s])[0]
            out.append(cache[key])
        return out

    t0 = time.time()
    start = leaf_bits(facts, root, {}, N_LEVELS, LEVEL_BITS, winning, CAP)
    found = local_search(facts, root, N_LEVELS, LEVEL_BITS, winning, random.Random(job["index"]), job["seconds"], CAP)
    t1 = time.time()
    levels, exc = found["levels"], found["exceptions"]
    row = {"index": job["index"], "status": found["status"], "levels": {str(c): k for c, k in sorted(levels.items())},
           "exceptions": [[list(k[0]), k[1], m] for k, m in exc.items()], "bits": found["bits"],
           "own_positions": found["own_positions"], "evaluations": found["evaluations"],
           "start_bits": start["bits"], "start_exceptions": len(start["exceptions"]),
           "start_own_positions": start["own_positions"], "oracle_positions": len(cache),
           "search_seconds": round(t1 - t0, 1)}
    if found["bits"] is None:
        return row
    walk = verify_with(facts, root, levels, N_LEVELS, exc, CAP)
    cert = certify(game, root, 0, lambda xs: [choose_with(facts, x, levels, N_LEVELS, exc) for x in xs],
                   lambda x: _value(game, x), root_value=1)
    return {**row, "verified": walk["won"], "certified": bool(cert["certified"] and cert["complete_game"]),
            "check_seconds": round(time.time() - t1, 1)}


def main() -> None:
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", type=int, default=None, help="one position's index (a smoke run; not evidence)")
    ap.add_argument("--seconds", type=float, default=SECONDS)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pilot = load_evidence(ROOT / "evidence" / "c49_s1_pilot.json.gz")["results"]
    jobs = [{"index": r["index"], "board": r["board"], "to_move": r["to_move"], "seconds": args.seconds}
            for r in pilot if r["ply"] == 10 and r["status"] == "budget" and args.only in (None, r["index"])]
    print(f"{len(jobs)} positions", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(probe, j) for j in jobs]):
            r = fut.result()
            rows.append(r)
            print({k: v for k, v in r.items() if k not in ("levels", "exceptions")}, flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the probe ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "settings": {"n_levels": N_LEVELS, "level_bits": LEVEL_BITS, "seconds": args.seconds,
                                          "cap": CAP, "only": args.only},
                             "rows": sorted(rows, key=lambda r: r["index"])})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
