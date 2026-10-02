"""§C.50 E2 — the smallest position-sound tic-tac-toe playbook in harness.playbook's language, found and proven
minimal by SAT (harness.rule_search), then re-verified exactly at every raw position with harness.playbook.evaluate.

Every predicate in the language is unchanged by the board's symmetries, so the search sees one position per symmetry
class; the verification sees all 4,520.

    PYTHONPATH=. .venv/bin/python scripts/minimal_playbook.py --out evidence/c50_E2_minimal_playbook.json.gz
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

PREDICATES = ("wins", "blocks", "gives_win", "makes_threat", "forks", "opp_fork_at", "gives_fork", "centre", "corner",
              "side", "opposite_corner")
LANGUAGES = {"base": PREDICATES, "layered": PREDICATES + ("wins_in_5", "gives_loss_in_4", "gives_loss_in_6")}
MAX_RULES = 12
MEASUREMENT_MODULES = ("scripts/minimal_playbook.py", "harness/rule_search.py", "harness/playbook.py",
                       "harness/coverage.py")


def main() -> None:
    import platform

    from games.tictactoe import TicTacToe
    from harness.coverage import move_values, optimal_actions, reachable_states
    from harness.evidence import save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.playbook import PREDICATES as LIBRARY, Rule, evaluate
    from harness.rule_search import Case, decode_rules, minimal_playbook

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-rules", type=int, default=MAX_RULES)
    ap.add_argument("--language", choices=sorted(LANGUAGES), default="base")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()
    game = TicTacToe()
    names = LANGUAGES[args.language]
    raw, complete = reachable_states(game, exact=True, symmetry=False)
    assert complete
    raw = [s for s in raw if not game.is_terminal(s)]
    reps = {}
    for s in raw:
        reps.setdefault(game.canonical_key(s), s)
    cases = []
    for s in reps.values():
        moves = game.legal_actions(s)
        best = optimal_actions(game, s)
        cases.append(Case(tuple(tuple(int(LIBRARY[p](game, s, m)) for p in names) for m in moves),
                          frozenset(i for i, m in enumerate(moves) if m in best)))
    result = minimal_playbook(cases, len(names), args.max_rules)
    rules = decode_rules(result["rules"], names) if result["rules"] is not None else None
    verification = {"positions": len(raw), "covered": None, "correct": None, "rules": None}
    if rules is not None:
        book = [Rule(f"rule {i + 1}", lits) for i, lits in enumerate(rules)]
        check = evaluate(book, game, raw, [move_values(game, s) for s in raw])
        verification = {"positions": check["positions"], "covered": check["covered"], "correct": check["correct"],
                        "rules": check["rules"]}
    blocked = [{"board": list(cases[i].profiles), "optimal": sorted(cases[i].optimal)}
               for i in result["inseparable"][:20]]
    print(f"k={result['k']} literals={result.get('literals')} proven_minimal={result['proven_minimal']} "
          f"impossible_up_to={result['impossible_up_to']} inseparable={len(result['inseparable'])} "
          f"verified {verification['correct']}/{verification['positions']} [{time.time() - t0:.0f}s]", flush=True)
    for r in verification["rules"] or []:
        print("  ", r["text"], f"(decides {r['first_fires']})", flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the search ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version()},
                             "config": {"predicates": list(names), "max_rules": args.max_rules,
                                        "positions": len(raw)},
                             "classes": len(cases),
                             "result": {k: result[k] for k in result if k != "rules"} | {"rules": rules},
                             "inseparable_examples": blocked, "verification": verification,
                             "seconds": round(time.time() - t0, 1)})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
