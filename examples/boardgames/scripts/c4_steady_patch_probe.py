"""SUPERSEDED by scripts/c4_steady_leaf_probe.py (h170) — kept as the record of how its evidence was made; the
harness.steady_local and harness.steady_exceptions APIs it calls have since changed.

§3.6 — map-plus-exceptions leaves at ply 10: on the eight ply-10 positions the S1 pilot left out of budget, local
search (harness.steady_local, 8 levels, 30 minutes, the h166 settings) proposes a map, harness.steady_exceptions
patches its failing lines with at most MAX_EXCEPTIONS table moves, and the leaf is checked by the solver-free walk
with its exceptions and by harness.certify. Size: the map's bits (1 per empty cell + 3 per levelled cell) plus the
exceptions' bits (an index among the leaf's own positions + a 3-bit move each). The near-miss maps are saved.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_steady_patch_probe.py --workers 4 \\
        --out evidence/c49_s1_patch_probe.json.gz
"""
from __future__ import annotations

import argparse
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_MODULES = ("scripts/c4_steady_patch_probe.py", "scripts/c4_steady_states.py", "harness/steady_local.py",
                       "harness/steady_exceptions.py", "harness/steady_state.py", "harness/certify.py",
                       "harness/c4_oracle.py", "harness/native_solver.py")
N_LEVELS, SECONDS, CAP, MAX_EXCEPTIONS = 8, 1800.0, 1_000_000, 100


def probe(job: dict) -> dict:
    import random
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from c4_steady_states import _value, _winning

    from games.connect4 import C4State, Connect4
    from harness.certify import certify
    from harness.steady_exceptions import choose_with, exception_bits, patch, verify_with
    from harness.steady_local import local_search
    from harness.steady_state import Facts, map_bits

    game = Connect4()
    root = C4State(tuple(job["board"]), job["to_move"], None, False)
    facts = Facts(game)
    t0 = time.time()
    proposal = local_search(facts, root, N_LEVELS, random.Random(job["index"]), SECONDS, CAP)
    levels = proposal["levels"]
    t1 = time.time()
    patched = patch(facts, root, levels, N_LEVELS, lambda st: _winning(game, st), CAP, MAX_EXCEPTIONS)
    row = {"index": job["index"], "levels": {str(c): k for c, k in sorted(levels.items())},
           "best_failures": proposal["best_failures"], "evaluations": proposal["evaluations"],
           "search_seconds": round(t1 - t0, 1), "patch": patched["status"],
           "exceptions": [[list(k[0]), k[1], m] for k, m in patched["exceptions"].items()],
           "patch_seconds": round(time.time() - t1, 1)}
    if patched["status"] != "patched":
        return row
    exc = patched["exceptions"]
    walk = verify_with(facts, root, levels, N_LEVELS, exc, CAP)
    cert = certify(game, root, 0, lambda xs: [choose_with(facts, x, levels, N_LEVELS, exc) for x in xs],
                   lambda x: _value(game, x), root_value=1)
    own_positions = _own(game, facts, root, levels, exc)
    bits = map_bits(levels, len(facts.empty_cells(root)), 3) + exception_bits(len(exc), own_positions, 7)
    return {**row, "bits": bits, "own_positions": own_positions, "verified": walk["won"],
            "certified": bool(cert["certified"] and cert["complete_game"])}


def _own(game, facts, root, levels, exc) -> int:
    from harness.steady_exceptions import choose_with

    seen, own, stack = set(), set(), [root]
    while stack:
        s = stack.pop()
        k = game.state_key(s)
        if k in seen or game.is_terminal(s):
            continue
        seen.add(k)
        if game.current_player(s) == game.current_player(root):
            own.add(k)
            stack.append(game.step(s, choose_with(facts, s, levels, N_LEVELS, exc)))
        else:
            stack.extend(game.step(s, b) for b in game.legal_actions(s))
    return len(own)


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
    jobs = [{"index": r["index"], "board": r["board"], "to_move": r["to_move"]}
            for r in pilot if r["ply"] == 10 and r["status"] == "budget"]
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
                             "settings": {"n_levels": N_LEVELS, "seconds": SECONDS, "cap": CAP,
                                          "max_exceptions": MAX_EXCEPTIONS},
                             "rows": sorted(rows, key=lambda r: r["index"])})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
