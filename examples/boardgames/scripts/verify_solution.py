"""Verify the committed Connect 4 book yields a SOLVED model (SOLVE-IT M2).

Two checks, strongest first:
  1. EXACT optimality trace — BookAgent's whole line from the opening is graded against the EXACT reference
     (book proofs + solver); `first_blunder_ply == -1` and `verified_plies` = full game means every move it plays
     is provably optimal. Fast (book lookups).
  2. P1 conversion vs the EXACT oracle — BookAgent as first player must WIN vs perfect defence. The definitive
     gate; uses the warm solver TT so the oracle's opening replies are tractable.
"""
from __future__ import annotations

import random

from harness import solver
from harness.benchmark import exact_reference, optimality_trace, verify_solved
from harness.book import load_book, tt_cache_path, winning_strategy_coverage
from harness.bookagent import BookAgent
from harness.registry import resolve_game


def main() -> None:
    game = resolve_game("connect4")
    solver.load_book(tt_cache_path("connect4"))
    book = load_book("connect4")
    ws = winning_strategy_coverage(game, book, max_plies=12, max_exact_empty=28)
    print(f"book={len(book)}  WS root_proven={ws['root_proven']} value={ws['root_value']} complete={ws['complete']}")

    agent = BookAgent(book, "connect4", solve_endgame=28)
    ref = exact_reference(game, book=book, max_empty=28)
    trace = optimality_trace(game, lambda s: agent.act(game, s, random.Random(0)), ref, max_plies=42)
    print(f"[1] optimality trace: verified_plies={trace['verified_plies']} plies_played={trace['plies_played']} "
          f"first_blunder_ply={trace['first_blunder_ply']}  line={trace['line']}")

    print("[2] BookAgent as P1 vs the EXACT oracle (this can take a few minutes on the opening move)...", flush=True)
    res = verify_solved(game, lambda: BookAgent(book, "connect4", solve_endgame=28), games=1, strategist=0)
    print(f"    wins={res['wins']} draws={res['draws']} losses={res['losses']} rate={res['rate']}  SOLVED={res['solved']}")
    if res["solved"]:
        print("*** CONNECT 4 SOLVED (M2): BookAgent converts the first-player win vs the EXACT solver. ***")


if __name__ == "__main__":
    main()
