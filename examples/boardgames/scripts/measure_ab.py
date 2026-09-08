"""§C.16 A/B FINAL MEASUREMENT — one protocol, one entry point, straight into the ledger.

WHY (2026-09-08): the capacity A/B and the global-pool A/B were each measured by a hand-written heredoc.
Nothing carried the protocol from one to the next, so root count, `games_per_root`, seed role, simulation
budget and provenance were re-decided by hand every time — and a re-decided protocol is precisely how L1/L2/L3
(§C.11) happened. The ledger only checks what it is handed; this is what hands it the truth:

  * the seed must be a MEASUREMENT seed, so a run's gate can never have selected on these roots;
  * `params` and the training budget are READ OFF the run's own metrics, never typed in;
  * provenance is DERIVED from where the checkpoint sits in its run — final batch, or an earlier index whose
    budget then has to match the other arm's for the ledger to accept the comparison at all;
  * the simulation budget is folded into `roots_id`, so arms measured under different search cannot be paired;
  * the pre-registered read is printed beside the numbers, so a null cannot quietly become "promising".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from harness.benchmark import paired_conversion
from harness.ledger import Ledger, run_budget
from harness.measurement import MEASUREMENT_SEEDS, wilson_interval
from harness.registry import resolve_game

LEDGER_PATH = "checkpoints/scaled_runs/analysis_ledger.json"


def load_arm(spec: str, sims: int, device: str) -> dict:
    """`name=path/to/ckpt_N.pt` -> a scored arm: an agent factory plus everything the ledger demands."""
    from harness.neural import AlphaZeroAgent, load_net

    name, _, rest = spec.partition("=")
    path, _, code = rest.partition("@")
    if not name or not path:
        raise ValueError(f"--arm wants name=path/to/ckpt_N.pt[@fingerprint], got {spec!r}")
    ckpt = Path(path)
    net = load_net(str(ckpt), device)
    budget = run_budget(ckpt)
    if code and budget["code"] and code != budget["code"]:
        raise ValueError(f"{name}: fingerprint {code} was asserted but the run recorded {budget['code']}")
    budget["code"] = budget["code"] or code or None
    return {"name": name, "ckpt": ckpt, "params": sum(p.numel() for p in net.parameters()),
            "factory": lambda: AlphaZeroAgent(net, sims=sims, solve_endgame=0, gumbel=True, c_scale=0.1),
            **budget}


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=CKPT",
                    help="an arm to score; repeat once per arm (all arms share the roots). Append "
                         "@<fingerprint> for runs predating provenance.json, whose training code you have "
                         "established from git rather than from the run itself")
    ap.add_argument("--pair", action="append", default=[], metavar="A:B",
                    help="a comparison to draw from the ledger; repeat as needed")
    ap.add_argument("--game", default="connect4")
    ap.add_argument("--tag", default="", help="suffix for roots_id, naming this measurement's root family")
    ap.add_argument("--seed", type=int, default=257, help=f"must be a MEASUREMENT seed {sorted(MEASUREMENT_SEEDS)}")
    ap.add_argument("--n-roots", type=int, default=384)
    ap.add_argument("--empties", type=int, default=24)
    ap.add_argument("--games-per-root", type=int, default=1,
                    help="§C.13: roots buy power per unit compute, extra games per root do not")
    ap.add_argument("--max-empty", type=int, default=22)
    ap.add_argument("--sims", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ledger", default=LEDGER_PATH)
    ap.add_argument("--null-below", type=float, default=0.03,
                    help="pre-registered read: |difference| under this with no significant pair is a NULL")
    args = ap.parse_args()

    if args.seed not in MEASUREMENT_SEEDS:
        raise SystemExit(f"seed {args.seed} is not a MEASUREMENT seed {sorted(MEASUREMENT_SEEDS)} — "
                         f"reporting on a seed the gate may have selected on is selection-on-test")

    arms = [load_arm(spec, args.sims, args.device) for spec in args.arm]
    for a in arms:
        print(f"arm {a['name']:24s} {a['ckpt']}  batch={a['batch']} games={a['games']} "
              f"params={a['params']} provenance={a['provenance']} code={a['code']}")

    res = paired_conversion(resolve_game(args.game), {a["name"]: a["factory"] for a in arms},
                            n_roots=args.n_roots, empties=args.empties, seed=args.seed,
                            games_per_root=args.games_per_root, max_empty=args.max_empty)
    roots_id = f"{res['roots_id']}_sims{args.sims}" + (f"_{args.tag}" if args.tag else "")

    led = Ledger(args.ledger)
    for a in arms:
        outcomes = [1 if s >= 0.999 else 0 for s in res["arms"][a["name"]]["scores"]]
        e = led.record(a["name"], outcomes=outcomes, params=a["params"], games=a["games"],
                       provenance=a["provenance"], seed=args.seed, roots_id=roots_id, code=a["code"])
        lo, hi = wilson_interval(e["converted"], e["n"])
        print(f"\n{a['name']}: {e['converted']}/{e['n']} = {e['rate']:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    refused = 0
    for pair in args.pair:
        a, _, b = pair.partition(":")
        try:
            r = led.compare(a, b)
        except ValueError as exc:
            # A refusal is a RESULT — the arms were measured and recorded, and the ledger is saying the
            # comparison would attribute the difference to the wrong cause. Printing it as a verdict rather
            # than a traceback keeps it legible next to the rates it applies to.
            refused += 1
            print(f"\n{a} vs {b}\n  REFUSED: {exc}")
            continue
        d = r["rate_a"] - r["rate_b"]
        print(f"\n{a} vs {b}: {r['rate_a']:.4f} - {r['rate_b']:.4f} = {d:+.4f}")
        print(f"  a-only/b-only {r['only_a']}/{r['only_b']}  p={r['p']:.4f}  "
              f"alpha={r['alpha_corrected']:.4f} ({r['comparisons_on_family']} on this family)  "
              f"significant={r['significant']}")
        if r["budget_note"]:
            print(f"  {r['budget_note']}")
        if r["provenance_warning"]:
            print(f"  {r['provenance_warning']}")
        if r["code_warning"]:
            print(f"  {r['code_warning']}")
        verdict = ("NULL — below the pre-registered threshold and not significant"
                   if abs(d) < args.null_below and not r["significant"]
                   else "EFFECT — significant" if r["significant"]
                   else f"INCONCLUSIVE — |{d:+.4f}| exceeds {args.null_below} but does not reach significance")
        print(f"  READ: {verdict}")

    if refused:
        raise SystemExit(f"\n{refused} comparison(s) REFUSED — the arms were recorded, but the ledger will not "
                         f"draw a conclusion from them.")


if __name__ == "__main__":
    main()
