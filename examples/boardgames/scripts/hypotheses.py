"""§C.30 the hypothesis register, from the command line — register a claim, link its evidence, read the board.

    PYTHONPATH=. .venv/bin/python scripts/hypotheses.py list
    PYTHONPATH=. .venv/bin/python scripts/hypotheses.py register --id h5 \
        --claim "lower search trains better per simulation" --a b143 --b a47 --direction a>b \
        --unit simulations
    PYTHONPATH=. .venv/bin/python scripts/hypotheses.py link --id h5
    PYTHONPATH=. .venv/bin/python scripts/hypotheses.py register --id t10 \
        --claim "an arm name means one measurement on one root family" \
        --proof tests/test_ledger.py::test_a_refused_name_collision_leaves_the_ORIGINAL_measurement_untouched
    PYTHONPATH=. .venv/bin/python scripts/hypotheses.py verify --id t10

A claim registered BEFORE its run is a prediction; the same claim registered after is a rationalisation, and
`list` shows which one you have (§C.30 H2). Status is never typed in — it is derived from the ledger."""
from __future__ import annotations

import argparse
import sys

from harness.hypotheses import Register
from harness.ledger import Ledger

REGISTER_PATH = "checkpoints/scaled_runs/hypotheses.json"
LEDGER_PATH = "checkpoints/scaled_runs/analysis_ledger.json"

MARK = {"supported": "SUPPORTED", "refuted": "REFUTED ", "contested": "CONTESTED",
        "null": "NULL    ", "inconclusive": "INCONCL.", "untested": "untested"}


def show(h: dict) -> None:
    flag = ("no evidence yet" if not h["evidence"]
            else "pre-registered" if h["pre_registered"] else "POST-HOC")
    print(f"{MARK[h['status']]:>10}  {h['id']:<6} {h['claim']}")
    if h.get("superseded_by"):
        print(f"{'':>10}  SUPERSEDED BY {h['superseded_by']} — do not build on this: "
              f"{h['supersession']['reason']}")
    if h.get("mode") == "test":
        print(f"{'':>10}  proved by {h['proof']}  [{flag}]")
        for e in h["evidence"]:
            print(f"{'':>10}    {'PASS' if e['ok'] else 'FAIL'}  {e['drawn_at']}")
        return
    print(f"{'':>10}  predicts {h['a']} {'>' if h['direction'] == 'a>b' else '<'} {h['b']} "
          f"in {h['unit']}  [{flag}]")
    for e in h["evidence"]:
        print(f"{'':>10}    {e['diff']:+.4f}  p={e['p']:.4g}  "
              f"{'significant' if e['significant'] else 'not significant'}  {e['drawn_at']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("register")
    r.add_argument("--id", required=True)
    r.add_argument("--claim", required=True)
    r.add_argument("--a", help="COMPARISON mode: the arm the claim says is better")
    r.add_argument("--b", help="COMPARISON mode: the arm it is measured against")
    r.add_argument("--direction", choices=("a>b", "a<b"),
                   help="COMPARISON mode: required — an undeclared direction lets any result read as support")
    r.add_argument("--unit", default="", help="COMPARISON mode: simulations | wall_clock | games | ...")
    r.add_argument("--proof", default="",
                   help="TEST mode: a pytest node id that fails if the claim is false. Most findings are not "
                        "A/B comparisons at all, and a claim with no regression test behind it is prose.")
    r.add_argument("--null-below", type=float, default=0.03,
                   help="COMPARISON mode: |diff| under this reads as NULL rather than INCONCLUSIVE")
    r.add_argument("--note", default="")
    li = sub.add_parser("link")
    li.add_argument("--id", help="omit to link every hypothesis whose evidence exists")
    v = sub.add_parser("verify")
    v.add_argument("--id", required=True, help="run a TEST-backed claim's proof and attach the outcome")
    sp = sub.add_parser("supersede")
    sp.add_argument("--id", required=True, help="the claim to retire — its status is KEPT, not rewritten")
    sp.add_argument("--by", required=True, help="the later claim that retires it; it must have evidence")
    sp.add_argument("--reason", required=True)
    for p in (ap, r, li, v, sp):
        p.add_argument("--register", default=REGISTER_PATH) if p is not ap else None
    ap.add_argument("--ledger", default=LEDGER_PATH)
    args = ap.parse_args()

    reg = Register(getattr(args, "register", REGISTER_PATH))
    if args.cmd == "register":
        show(reg.register(args.id, claim=args.claim, a=args.a, b=args.b, direction=args.direction,
                          unit=args.unit, proof=args.proof, note=args.note, null_below=args.null_below))
        return
    if args.cmd == "verify":
        show(reg.verify(args.id))
        return
    if args.cmd == "supersede":
        show(reg.supersede(args.id, by=args.by, reason=args.reason))
        return
    if args.cmd == "link":
        led = Ledger(args.ledger)
        ids = [args.id] if args.id else [h["id"] for h in reg.report()]
        linked = 0
        for i in ids:
            try:
                show(reg.link(i, led))
                linked += 1
            except ValueError as exc:
                if args.id:
                    raise SystemExit(f"{i}: {exc}")
        if not args.id:
            print(f"\nlinked evidence for {linked}/{len(ids)} hypotheses")
        return
    rows = sorted(reg.report(), key=lambda h: (h["status"], h["id"]))
    for h in rows:
        show(h)
    print(f"\n{len(rows)} hypotheses: " + ", ".join(
        f"{n} {s}" for s, n in sorted({x: sum(1 for h in rows if h["status"] == x) for x in
                                       {h['status'] for h in rows}}.items())))


if __name__ == "__main__":
    sys.exit(main())
