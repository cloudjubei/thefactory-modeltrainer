"""PARALLEL winning-strategy grind for the committed Connect 4 book (SOLVE-IT M1).

`prove_winning_strategy_parallel` solves the winning-tree's independent leaves across a process pool in chunks,
persisting the committed book + TT after each chunk (via `on_save`), then assembles the tree by best-first once
every leaf is booked → the root is proven. Resumable: re-running continues from the booked leaves.

    PYTHONPATH=. .venv/bin/python scripts/ws_grind_parallel.py [wall_seconds] [workers] [max_exact_empty]
"""
from __future__ import annotations

import sys
import time

from harness import solver
from harness.book import _key, book_path, load_book, prove_winning_strategy_parallel, tt_cache_path
from harness.registry import resolve_game


def main() -> None:
    wall = float(sys.argv[1]) if len(sys.argv) > 1 else 3600.0
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    max_exact_empty = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    game = resolve_game("connect4")
    solver.load_book(tt_cache_path("connect4"))
    book = load_book("connect4")
    root_key = _key(game, game.initial_state(__import__("random").Random(0)))
    start = time.perf_counter()

    def save() -> None:
        book.save(book_path("connect4"))
        solver.save_book(tt_cache_path("connect4"))

    print(f"[ws||] start book={len(book)} workers={workers} E={max_exact_empty} wall={wall:.0f}s", flush=True)
    st = prove_winning_strategy_parallel(
        game, book, strategist=0, max_plies=42, max_exact_empty=max_exact_empty, workers=workers,
        max_seconds=8, deadline=start + wall, on_save=save, log=lambda m: print(m, flush=True),
    )
    save()
    el = time.perf_counter() - start
    rp = book.proven_value(root_key)
    print(f"[ws|| {el:.0f}s] rounds={st['rounds']} solved={st['proven']} assembled={st['assembled']} "
          f"deferred={st['deferred']} book={len(book)} root_proven={rp is not None} value={rp}", flush=True)
    if rp is not None:
        print(f"[ws||] *** ROOT PROVEN — value {rp} — SOLVED winning strategy ***", flush=True)
    print("[ws||] DONE", flush=True)


if __name__ == "__main__":
    main()
