"""§3.6 — which zero-bit default keeps the win? Under a SPEC's rule, every leaf of one run's certified strategies is
walked (its map restricted to the cells empty there, its exceptions played where the map has no move) and the own
positions holding two or more safe moves and no winning move are pooled per root; a seeded sample of each pool is
solved exactly. At each sampled position every ORDER picks one move — the lowest, the highest, the one nearest the
board's centre, the one leaving the opponent fewest replies that stop an immediate win, and the lowest after first
preferring moves that win by force in two ("win2"), that do not lose by force in two ("safe2"), or both in that order
("tactics") — among all its safe moves (what the default would play with no map at all) and among the moves the rule
actually broke a tie between (the first deciding level, or the unlevelled last level, when it held two or more), and
the pick is scored by whether it keeps the win. Saved: per root and in total, the positions, the chance a random safe
move wins, and each order's hits.

    PYTHONPATH=. .venv/bin/python -u scripts/c4_default_accuracy.py --spec harness.floor_l5 \\
        --targets c49_l5.json.gz --out evidence/c49_d1.json.gz
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENT_MODULES = ("scripts/c4_default_accuracy.py", "harness/steady_state.py", "harness/native_solver.py",
                       "harness/native_build.py")
C_SOURCES = ("harness/c4solver.c",)
ORDERS = ("lowest", "highest", "centre", "forcing", "win2", "safe2", "tactics")


def tied(facts, state, levels: dict, n_levels: int) -> list:
    """The moves the rule broke a tie between at `state` — the first level holding any safe move, else the unlevelled
    last level — when that holds two or more; empty otherwise."""
    _wins, safe, cell = facts.of(state)
    groups = [[a for a in safe if levels.get(cell[a]) == k] for k in range(n_levels)]
    if facts.unlevelled == "last":
        groups.append([a for a in safe if cell[a] not in levels])
    first = next((g for g in groups if g), [])
    return first if len(first) >= 2 else []


def opponent_safe_replies(game, facts, state, action) -> int:
    child = game.step(state, action)
    if game.is_terminal(child):
        return 0
    return sum(1 for b in game.legal_actions(child)
               if not any(facts._wins_now(game.step(child, b), d) for d in game.legal_actions(game.step(child, b))))


def pick(order: str, moves: list, game, facts, state) -> int:
    if order == "lowest":
        return min(moves)
    if order == "highest":
        return max(moves)
    _wins, _safe, cell = facts.of(state)
    h, w = game.board_shape
    if order == "centre":
        return min(moves, key=lambda a: ((cell[a] // w - (h - 1) / 2) ** 2 + (cell[a] % w - (w - 1) / 2) ** 2, a))
    if order == "forcing":
        return min(moves, key=lambda a: (opponent_safe_replies(game, facts, state, a), a))
    if order == "win2":
        return min(moves, key=lambda a: (opponent_safe_replies(game, facts, state, a) > 0, a))
    if order == "safe2":
        return min(moves, key=lambda a: (facts._loses_in_two(state, a), a))
    if order == "tactics":
        return min(moves, key=lambda a: (opponent_safe_replies(game, facts, state, a) > 0,
                                         facts._loses_in_two(state, a), a))
    raise ValueError(f"unknown order {order!r}")


def main() -> None:
    from games.connect4 import C4State, Connect4
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.native_solver import move_values
    from harness.steady_state import Facts, choose

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--targets", required=True, help="evidence file whose certified leaves are walked")
    ap.add_argument("--per-leaf", type=int, default=400, help="own positions walked per leaf at most")
    ap.add_argument("--per-root", type=int, default=150, help="pooled positions solved per root at most")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    b = importlib.import_module(args.spec).SPEC["builder"]
    n_levels = b["n_levels"]
    game = Connect4()
    facts = Facts(game, ties=b.get("ties", "exact"), safe_depth=b.get("safe_depth", 1),
                  forcing=b.get("forcing", False), unlevelled=b.get("unlevelled", "none"))
    rows = []
    for r in load_evidence(ROOT / "evidence" / args.targets)["roots"]:
        if not r["complete"]:
            continue
        pool: dict = {}
        for board, to_move, node in r["nodes"]:
            if "leaf" not in node:
                continue
            leaf = C4State(tuple(board), to_move, None, False)
            player = game.current_player(leaf)
            empty = set(facts.empty_cells(leaf))
            levels = {int(c): k for c, k in r["maps"][node["leaf"]].items() if int(c) in empty}
            exceptions = {(tuple(eb), et): move for eb, et, move in node.get("exceptions", [])}
            stack, seen, own = [leaf], set(), 0
            while stack and own < args.per_leaf:
                s = stack.pop()
                key = game.state_key(s)
                if key in seen or game.is_terminal(s):
                    continue
                seen.add(key)
                if game.current_player(s) != player:
                    stack.extend(game.step(s, a) for a in game.legal_actions(s))
                    continue
                own += 1
                wins, safe, _cell = facts.of(s)
                if not wins and len(safe) >= 2 and key not in pool:
                    pool[key] = (s, tied(facts, s, levels, n_levels))
                move = exceptions.get((s.board, s.to_move))
                if move is None:
                    move = choose(facts, s, levels, n_levels)
                if move is not None:
                    stack.append(game.step(s, move))
        rng = random.Random(args.seed * 1_000_003 + r["index"])
        keys = sorted(pool)
        sample = [pool[k] for k in rng.sample(keys, min(args.per_root, len(keys)))]
        row = {"index": r["index"], "pooled": len(pool), "solved": len(sample), "tied": 0, "chance": 0.0,
               "tied_chance": 0.0, "hits": {o: 0 for o in ORDERS}, "tied_hits": {o: 0 for o in ORDERS}}
        for s, at in sample:
            values = move_values(s)
            safe = facts.of(s)[1]
            won = {a for a in safe if values.get(a) == 1}
            row["chance"] += len(won) / len(safe)
            for o in ORDERS:
                row["hits"][o] += pick(o, safe, game, facts, s) in won
            if at:
                row["tied"] += 1
                row["tied_chance"] += len(won & set(at)) / len(at)
                for o in ORDERS:
                    row["tied_hits"][o] += pick(o, at, game, facts, s) in won
        rows.append(row)
        print(row, flush=True)
    totals = {"solved": sum(x["solved"] for x in rows), "tied": sum(x["tied"] for x in rows),
              "chance": sum(x["chance"] for x in rows), "tied_chance": sum(x["tied_chance"] for x in rows),
              "hits": {o: sum(x["hits"][o] for x in rows) for o in ORDERS},
              "tied_hits": {o: sum(x["tied_hits"][o] for x in rows) for o in ORDERS}}
    print(totals, flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the run ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "c_sources": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest()[:12]
                                           for s in C_SOURCES},
                             "spec": args.spec, "targets": args.targets, "per_leaf": args.per_leaf,
                             "per_root": args.per_root, "seed": args.seed, "orders": list(ORDERS),
                             "rows": rows, "totals": totals})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
