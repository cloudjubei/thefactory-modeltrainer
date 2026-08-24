"""Standalone winning-strategy grind for the committed Connect 4 book (SOLVE-IT M1).

Best-first `prove_winning_strategy` (centre-first guide) on the committed book, resumable. Uses E=24 by default:
ply-18 leaves have only 24 empties (near-endgame) so every leaf solve is ~ms REGARDLESS of how warm the TT is —
avoiding the cold-leaf collapse E=28 (ply-14, ~5s cold solves) hits. Per-pass overhead is stripped: a CHEAP
root-proven check each pass, the full coverage walk only occasionally, and a save every few passes. Stops at
`root_proven` (the whole winning strategy is booked = a lookup-perfect first player), a wall limit, or a book cap.

    PYTHONPATH=. .venv/bin/python scripts/ws_grind.py [wall_seconds] [max_exact_empty] [book_cap]
"""
from __future__ import annotations

import sys
import time

from harness import solver
from harness.book import (
    book_coverage,
    book_path,
    load_book,
    prove_winning_strategy,
    tt_cache_path,
    winning_strategy_coverage,
)
from harness.registry import resolve_game
from harness.book import _key  # noqa: E402


def main() -> None:
    wall = float(sys.argv[1]) if len(sys.argv) > 1 else 3600.0
    max_exact_empty = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    book_cap = int(sys.argv[3]) if len(sys.argv) > 3 else 6_000_000
    game = resolve_game("connect4")
    solver.load_book(tt_cache_path("connect4"))
    book = load_book("connect4")
    root_key = _key(game, game.initial_state(__import__("random").Random(0)))
    start = time.perf_counter()
    print(f"[ws_grind] start book={len(book)} E={max_exact_empty} wall={wall:.0f}s", flush=True)
    passno = 0
    while True:
        el = time.perf_counter() - start
        if el >= wall or len(book) >= book_cap:
            print(f"[ws_grind] stop (elapsed {el:.0f}s, book {len(book)})", flush=True)
            break
        passno += 1
        before = len(book)
        st = prove_winning_strategy(
            game, book, strategist=0, max_plies=42, max_exact_empty=max_exact_empty,
            guide=None, max_seconds=2, deadline=time.perf_counter() + 60, max_positions=10**9,
        )
        root_proven = book.proven_value(root_key) is not None  # CHEAP check every pass
        if passno % 3 == 0:
            book.save(book_path("connect4"))
            solver.save_book(tt_cache_path("connect4"))
        el = time.perf_counter() - start
        rate = st["proven"] / max(1e-9, el - (el - 60))  # rough per-pass rate
        extra = ""
        if passno % 10 == 0 or root_proven:  # the expensive coverage walk only occasionally
            cov8 = book_coverage(game, book, plies=8)
            ws = winning_strategy_coverage(game, book, max_plies=12, max_exact_empty=max_exact_empty)
            extra = (f" | opening<=8ply proven {cov8['proven']}/{cov8['reachable']} "
                     f"| WS proven {ws['proven']}/{ws['nodes']} ({ws['provenFraction'] * 100:.1f}%)")
        print(f"[ws_grind {el:.0f}s p{passno}] +{st['proven']} (defer {st['deferred']}) book={len(book)} "
              f"root_proven={root_proven}{extra}", flush=True)
        if root_proven:
            book.save(book_path("connect4"))
            solver.save_book(tt_cache_path("connect4"))
            print(f"[ws_grind] *** ROOT PROVEN at {el:.0f}s value={book.proven_value(root_key)} — SOLVED strategy ***", flush=True)
            break
        if st["proven"] == 0 and st["deferred"] == 0 and len(book) == before:
            print("[ws_grind] tree exhausted with nothing new — done", flush=True)
            break
    print("[ws_grind] DONE", flush=True)


if __name__ == "__main__":
    main()
