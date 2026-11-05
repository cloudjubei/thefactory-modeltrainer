"""§3.6 X2's probe — how much of Z1's 51 rebuilds the library covers read on the mirrored board, from
evidence/c49_z1.json.gz and its library, with no build. Walking each rebuild from its root, a move node is a hit where
some library map wins every line from it, and a hit's subtree is not walked further. Two readings: the map on the
board's mirror image (`mirrored`), and the map with its cells flipped on the real board (`flipped`). Writes
evidence/c49_x2_probe.json.gz.

    PYTHONPATH=. .venv/bin/python scripts/c4_mirror_probe.py"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from c4_mirror_sharing import _grid_symmetries  # noqa: E402

from games.connect4 import COLS, ROWS, C4State, Connect4  # noqa: E402
from harness.evidence import load_evidence, save_evidence  # noqa: E402
from harness.floor_z1 import SPEC  # noqa: E402
from harness.mirror_builder import board_image  # noqa: E402
from harness.native_leaf import verify  # noqa: E402
from harness.steady_state import Facts, restrict  # noqa: E402

BUILDER = SPEC["builder"]
CELLS = _grid_symmetries(ROWS, COLS)["flip_columns"]
SECONDS = 20
WORKERS = 5


def _probe(job: tuple) -> dict:
    index, nodes, library = job
    game = Connect4()
    facts = Facts(game, ties=BUILDER["ties"], unlevelled=BUILDER["unlevelled"])
    image = board_image(CELLS)
    table = {(tuple(board), to_move): node for board, to_move, node in nodes}
    root = C4State(tuple(nodes[0][0]), nodes[0][1], None, False)

    def kids(state, move):
        child = game.step(state, move)
        if game.is_terminal(child):
            return []
        return [c for a in game.legal_actions(child) for c in [game.step(child, a)] if not game.is_terminal(c)]

    def wins(state, levels):
        return verify(facts, state, restrict(levels, facts.empty_cells(state)), BUILDER["n_levels"], BUILDER["cap"],
                      deadline=time.monotonic() + SECONDS)["won"]

    def nodes_under(state, seen):
        key = game.state_key(state)
        if key in seen or key not in table:
            return 0
        seen.add(key)
        return 1 + sum(nodes_under(c, seen) for c in (kids(state, table[key]["move"]) if "move" in table[key] else []))

    def walk(covers):
        seen, stack, hits, cut = set(), [root], 0, 0
        while stack:
            state = stack.pop()
            key = game.state_key(state)
            if key in seen or key not in table or "move" not in table[key]:
                continue
            seen.add(key)
            if covers(state):
                hits, cut = hits + 1, cut + nodes_under(state, set())
            else:
                stack.extend(kids(state, table[key]["move"]))
        return {"hits": hits, "nodes_cut": cut}

    flipped = [{CELLS[c]: k for c, k in levels.items()} for levels in library]
    return {"index": index, "nodes": len(table), "moves": sum("move" in n for n in table.values()),
            "mirrored": walk(lambda s: any(wins(image(s), levels) for levels in library)),
            "flipped": walk(lambda s: any(wins(s, levels) for levels in flipped))}


def main() -> None:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    evidence = ROOT / "evidence"
    library = [{int(c): k for c, k in levels.items()}
               for levels, _ in load_evidence(evidence / SPEC["library_from"])["roots"][0]["found_maps"]]
    rows = load_evidence(evidence / "c49_z1.json.gz")["roots"]
    with ProcessPoolExecutor(WORKERS) as pool:
        results = [f.result() for f in as_completed([pool.submit(_probe, (r["index"], r["nodes"], library))
                                                     for r in rows])]
    save_evidence(evidence / "c49_x2_probe.json.gz", {"started": started, "library": len(library),
                                                      "roots": sorted(results, key=lambda r: r["index"])})
    print(f"{len(results)} rebuilds probed", flush=True)


if __name__ == "__main__":
    main()
