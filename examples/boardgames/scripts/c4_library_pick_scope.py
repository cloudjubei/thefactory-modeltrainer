"""§3.6 O1 scope — where the builder's move rule, given G3's library, picks another move than the opening at ply 6.
E1 (h342) found its saving in that move: the opening (harness.opening) picks by the replies the empty map wins, the
builder (harness.strategy_builder._split) by the replies any known map covers, G3's 2,030 library maps included. For
each of the opening's 148 ply-6 positions the empty map does not win (sorted as scripts/c4_strategy_s2.py sorts them)
it records the winning moves, the opening's move, the builder's pick with its scores (covered replies, replies, move)
and G3's bits below the position (harness.floor_e1.e1_control). No search and no build: G3's builder settings, its
library, the native verifier. Writes evidence/c49_o1_scope.json.gz.

    PYTHONPATH=. .venv/bin/python scripts/c4_library_pick_scope.py"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import c4_strategy_s2 as s2  # noqa: E402
from games.connect4 import C4State, Connect4  # noqa: E402
from harness.evidence import load_evidence, save_evidence  # noqa: E402
from harness.floor_e1 import SPEC, e1_control  # noqa: E402
from harness.native_leaf import verify as native_verify  # noqa: E402
from harness.opening import frontier  # noqa: E402
from harness.steady_state import Facts  # noqa: E402
from harness.strategy_builder import Builder  # noqa: E402


def main() -> None:
    out = ROOT / "evidence" / "c49_o1_scope.json.gz"
    if out.exists():
        raise SystemExit(f"{out} exists")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    game = Connect4()
    b = SPEC["builder"]
    facts = Facts(game, ties=b.get("ties", "exact"), safe_depth=b.get("safe_depth", 1),
                  forcing=b.get("forcing", False), unlevelled=b.get("unlevelled", "none"),
                  order=b.get("order", "lowest"), net=s2._net(b))
    book = {(tuple(r["board"]), r["to_move"]): {int(a) for a, v in r["values"].items() if v == 1}
            for r in load_evidence(s2.LABELS)["positions"]}

    def winning(states):
        return [book[game.state_key(s)] for s in states]

    moves = frontier(Facts(game), game.initial_state(), winning, 8, b["n_levels"], b["cap"])["moves"]
    _opening, rows = s2._opening(SPEC)
    library = [({int(c): k for c, k in levels.items()}, empty)
               for r in load_evidence(ROOT / "evidence" / SPEC["library_from"])["roots"]
               for levels, empty in r["found_maps"]]
    g3 = load_evidence(ROOT / "evidence" / "c49_g3.json.gz")
    builder = Builder(facts, winning, None, b["n_levels"], b["level_bits"], b["cap"], b["min_leaf_depth"],
                      b["reuse_window"], 10.0 ** 9, b.get("accept"), b.get("budgets"), library, native_verify,
                      choose_by_size=b.get("choose_by_size", False), phases=b.get("phases", False))
    positions = []
    for i, r in enumerate(rows):
        s = C4State(tuple(r["board"]), r["to_move"], None, False)
        key = game.state_key(s)
        scored = []
        for a in sorted(book[key]):
            children = builder._children(s, a)
            covered = sum(1 for c in children if builder._covering_map(c) is not None)
            scored.append([covered, len(children), a])
        pick = min(scored, key=lambda x: (x[1] - x[0], x[1], x[2]))[2]
        positions.append({"index": i, "winning": sorted(book[key]), "opening": moves[key], "library_pick": pick,
                          "scored": scored, "g3_bits": e1_control([s], g3, moves)[0]})
    record = {"started": started, "positions_fp": hashlib.sha256(s2.LABELS.read_bytes()).hexdigest()[:12],
              "library_from": SPEC["library_from"], "library_maps": len(library), "positions": positions}
    save_evidence(out, record)
    differ = [p for p in positions if p["library_pick"] != p["opening"]]
    print(f"{len(positions)} positions, {len(differ)} picked differently, {sum(p['g3_bits'] for p in differ)} of "
          f"G3's {sum(p['g3_bits'] for p in positions)} bits below them: {[p['index'] for p in differ]}")


if __name__ == "__main__":
    main()
