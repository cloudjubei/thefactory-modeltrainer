"""§3.6 S2 — complete certified first-player strategies (table moves + steady-state leaves) for the rest of the game
from sampled positions — by default ply 10 of the canonical exact Connect-4 table (evidence/c49_frontier_positions.json.gz,
harness.floor_s2), or as another judge's SPEC registers (`--spec`, e.g. harness.floor_s3: ply 8 of the label cache).
Leaves are searched by SAT (harness.steady_search), or with the SPEC's search kind "local" by size-scored local search
(harness.steady_local) over a shared bounded oracle cache (harness.winning_cache), a leaf then possibly carrying
exceptions (harness.floor_s4). With the SPEC's source "opening", the roots are the first player's positions at the
SPEC's ply that an opening from the empty board leaves to strategies (harness.opening, the label cache as its oracle;
harness.floor_w1), and the opening's own size is saved with them. A SPEC may name its roots ("indices") instead of
sampling them, and give the builder search budgets by depth ("budgets", harness.floor_p1). With "wave_size" the roots
are built in waves and each wave's builds start from a library of every map the earlier waves found
(harness.floor_w2); with "library_from" every build starts from the maps an earlier run's builds found
(harness.floor_w3). A local search with "walker": "native" walks its leaves and verifies its maps in C
(harness.native_leaf); every run
records the C sources' hashes, which the Python fingerprint cannot cover.
Roots: the first `roots` non-trivial positions in a seeded shuffle (harness.floor_s2), or with `--pilot N` the next N
after them. Per root, in parallel: harness.strategy_builder within the registered cap, then three independent checks
— the oracle-free walk (strategy_builder.check), every table move a winning move by the exact solver, and every leaf
with a non-empty map certified (harness.certify). The strategies themselves are saved. Judged by
harness.floor_s2.s2_report.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_strategy_s2.py --workers 8 --out evidence/c49_s2.json.gz \\
        --progress <scratch>/s2_progress.json [--pilot 1 --seconds 1200]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POSITIONS = ROOT / "evidence" / "c49_frontier_positions.json.gz"
LABELS = ROOT / "books" / "c4_labels.json.gz"
MEASUREMENT_MODULES = ("scripts/c4_strategy_s2.py", "harness/strategy_builder.py", "harness/steady_state.py",
                       "harness/steady_search.py", "harness/certify.py", "harness/c4_oracle.py",
                       "harness/native_solver.py", "scripts/c4_steady_states.py", "harness/steady_local.py",
                       "harness/steady_exceptions.py", "harness/winning_cache.py", "harness/exception_coding.py",
                       "harness/opening.py", "harness/native_leaf.py")
C_SOURCES = ("harness/c4leafwalk.c", "harness/c4solver.c")


def build_one(job: dict) -> dict:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from c4_steady_states import _value, _winning

    from games.connect4 import C4State, Connect4
    from harness.certify import certify
    from harness.native_leaf import needed as native_walk, verify as native_verify
    from harness.steady_exceptions import choose_with, needed
    from harness.steady_local import local_search
    from harness.steady_search import find_steady_state
    from harness.steady_state import Facts, verify
    from harness.strategy_builder import Builder, check
    from harness.winning_cache import CachedWinning

    game = Connect4()
    root = C4State(tuple(job["board"]), job["to_move"], None, False)
    facts = Facts(game)
    b, s = job["builder"], job["search"]
    if s.get("kind") == "local":
        winning = CachedWinning(lambda states: _winning(game, states), game.state_key, s["cache_limit"])
        rng = random.Random(job["index"])

        walker = native_walk if s.get("walker") == "native" else needed
        verifier = native_verify if s.get("walker") == "native" else verify

        def search(st, start, seconds):
            r = local_search(facts, st, b["n_levels"], b["level_bits"], winning, rng,
                             s["seconds"] if seconds is None else seconds, s["cap"], start, walker)
            return None if r["bits"] is None else {k: r[k] for k in ("levels", "exceptions", "own_positions")}
    else:
        winning = lambda states: _winning(game, states)
        verifier = verify

        def search(st, start, seconds):
            budget = {**s, **({} if seconds is None else {"seconds": seconds})}
            r = find_steady_state(facts, st, winning, b["n_levels"], **budget)
            return {"levels": r["levels"], "exceptions": {}, "own_positions": None} if r["status"] == "found" else None
    library = [({int(c): k for c, k in levels.items()}, empty) for levels, empty in job.get("library", [])]
    builder = Builder(facts, winning, search, b["n_levels"], b["level_bits"], b["cap"], b["min_leaf_depth"],
                      b["reuse_window"], b["seconds"], b.get("accept"), b.get("budgets"), library, verifier)
    t0 = time.time()
    complete = builder.build(root)
    row = {"index": job["index"], "board": job["board"], "to_move": job["to_move"], "complete": complete,
           "build_seconds": round(time.time() - t0, 1), "searches": builder.searches, "bits": builder.bits(),
           "shared": builder.shared,
           "found_maps": [[{str(c): v for c, v in m.items()}, e] for m, e in builder.found_maps()],
           "library_leaves": sum(1 for n in builder.nodes.values() if "leaf" in n and 1 <= n["leaf"] <= builder.shared),
           "nodes": [[list(k[0]), k[1], _jsonable(n)] for k, n in builder.nodes.items()], "maps":
           [{str(c): v for c, v in m.items()} for m in builder.maps], "map_empty": builder.map_empty}
    if not complete:
        return {**row, "own_positions": None, "checked": None, "moves_win": None, "leaves_certified": None}
    t1 = time.time()
    walk = check(facts, root, builder.nodes, builder.maps, b["n_levels"], b["cap"])
    states = {k: C4State(k[0], k[1], None, False) for k in builder.nodes}
    moves = [(states[k], n["move"]) for k, n in builder.nodes.items() if "move" in n]
    moves_win = all(m in w for (st, m), w in zip(moves, winning([st for st, _ in moves])))
    certified = True
    for k, n in builder.nodes.items():
        if "leaf" in n and (n["leaf"] != 0 or n.get("exceptions")):
            st = states[k]
            empty = set(facts.empty_cells(st))
            levels = {c: v for c, v in builder.maps[n["leaf"]].items() if c in empty}
            exc = n.get("exceptions", {})
            cert = certify(game, st, 0, lambda xs, lv=levels, ex=exc: [choose_with(facts, x, lv, b["n_levels"], ex)
                                                                        for x in xs],
                           lambda x: _value(game, x), root_value=1)
            certified = certified and bool(cert["certified"] and cert["complete_game"])
    kinds = {"trivial": 0, "mapped": 0, "excepted": 0}
    for n in builder.nodes.values():
        if "leaf" in n:
            kinds["excepted" if n.get("exceptions") else "trivial" if n["leaf"] == 0 else "mapped"] += 1
    return {**row, "own_positions": walk["own_positions"], "checked": walk["won"], "moves_win": moves_win,
            "leaves_certified": certified, "leaf_kinds": kinds, "check_seconds": round(time.time() - t1, 1)}


def _opening(spec: dict) -> tuple:
    """The opening's size and its frontier as root rows: winning moves from the label cache, which records every move's
    value at the first player's positions through ply 8 (a position missing from it stops the run)."""
    from games.connect4 import Connect4
    from harness.evidence import load_evidence
    from harness.opening import frontier
    from harness.steady_state import Facts

    game = Connect4()
    book = {(tuple(r["board"]), r["to_move"]): {int(a) for a, v in r["values"].items() if v == 1}
            for r in load_evidence(LABELS)["positions"]}
    found = frontier(Facts(game), game.initial_state(), lambda states: [book[game.state_key(s)] for s in states],
                     spec["ply"], spec["builder"]["n_levels"], spec["builder"]["cap"])
    rows = sorted(({"board": list(s.board), "to_move": s.to_move} for s in found["frontier"]),
                  key=lambda r: (r["board"], r["to_move"]))
    return {"moves": len(found["moves"]), "trivial": len(found["trivial"]), "frontier": len(rows)}, rows


def _jsonable(node: dict) -> dict:
    if "exceptions" not in node:
        return node
    return {**node, "exceptions": [[list(k[0]), k[1], m] for k, m in node["exceptions"].items()]}


def main() -> None:
    import platform

    from games.connect4 import C4State, Connect4
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    import hashlib
    import importlib

    from harness.steady_state import Facts, verify

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", required=True)
    ap.add_argument("--pilot", type=int, default=None, help="build this many roots after the registered ones")
    ap.add_argument("--seconds", type=float, default=None, help="pilot only: a shorter per-root cap")
    ap.add_argument("--spec", default="harness.floor_s2", help="the judge whose SPEC registers the study")
    args = ap.parse_args()
    SPEC = importlib.import_module(args.spec).SPEC
    if args.seconds is not None and not args.pilot:
        ap.error("--seconds changes the registered cap; it is for pilots only")
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    opening = None
    if SPEC.get("source") == "opening":
        opening, rows = _opening(SPEC)
        positions_fp = hashlib.sha256(LABELS.read_bytes()).hexdigest()[:12]
    elif SPEC.get("source") == "labels":
        rows = [r for r in load_evidence(LABELS)["positions"]
                if sum(1 for v in r["board"] if v) == SPEC["ply"] and r["to_move"] == 0
                and max(r["values"].values()) == 1]
        positions_fp = hashlib.sha256(LABELS.read_bytes()).hexdigest()[:12]
    else:
        positions = load_evidence(POSITIONS)
        rows = positions["plies"][str(SPEC["ply"])]
        positions_fp = positions["measurement_fingerprint"]
    order = list(range(len(rows)))
    random.Random(SPEC["seed"]).shuffle(order)
    game = Connect4()
    picked = []
    for i in order:
        s = C4State(tuple(rows[i]["board"]), rows[i]["to_move"], None, False)
        if not verify(Facts(game), s, {}, SPEC["builder"]["n_levels"], 100_000)["won"]:
            picked.append(i)
        if len(picked) == SPEC["roots"] + (args.pilot or 0):
            break
    picked = picked[SPEC["roots"]:] if args.pilot else picked
    if SPEC.get("indices") is not None:
        if args.pilot:
            ap.error("a SPEC that names its roots has no pilot roots")
        picked = list(SPEC["indices"])
    builder = {**SPEC["builder"], **({"seconds": args.seconds} if args.seconds is not None else {})}
    jobs = [{"index": i, "board": rows[i]["board"], "to_move": rows[i]["to_move"], "builder": builder,
             "search": SPEC["search"]} for i in picked]
    print(f"{len(jobs)} roots ({'pilot' if args.pilot else 'registered'}): {picked}", flush=True)
    results = []
    size = SPEC.get("wave_size") or len(jobs)
    earlier = load_evidence(ROOT / "evidence" / SPEC["library_from"])["roots"] if SPEC.get("library_from") else []
    library: list = [m for r in earlier for m in r["found_maps"]]
    for start in range(0, len(jobs), size):
        wave = [{**j, "library": list(library)} for j in jobs[start:start + size]]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for fut in as_completed([pool.submit(build_one, j) for j in wave]):
                r = fut.result()
                results.append(r)
                Path(args.progress).write_text(json.dumps({"roots": results}))
                print(f"root #{r['index']}: complete {r['complete']}, bits {r['bits']}, searches {r['searches']}, "
                      f"library {r['shared']} maps / {r['library_leaves']} leaves, own positions "
                      f"{r['own_positions']}, checked {r['checked']}, moves win {r['moves_win']}, "
                      f"leaves certified {r['leaves_certified']}, build {r['build_seconds']}s", flush=True)
        if SPEC.get("wave_size"):
            library += [m for r in results[start:] for m in r["found_maps"]]
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp,
                             "positions_fingerprint": positions_fp, "spec": args.spec,
                             "versions": {"python": platform.python_version()}, "pilot": args.pilot,
                             "builder": builder, "search": SPEC["search"],
                             "c_sources": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:12]
                                           for f in C_SOURCES},
                             "opening": opening, "roots": sorted(results, key=lambda r: r["index"])})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
