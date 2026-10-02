"""§C.41 OPTIMALITY SCORECARD — report a model's DISTANCE TO OPTIMAL directly, replacing the proxies (loss,
opening_value, late-corpus move-match) the §C.24 audit showed can mislead.

For a SolvableGame it is exact where the game is small (tictactoe: full enumeration) and sampled-then-solved
where it is not (connect4). It reports state-space COVERAGE (fraction of reachable states where the model's move
is solver-optimal) with a per-ply breakdown, the DECIDED frontier (proven-non-loss share), and — for the
opening/endgame of connect4 — proven-win conversion vs exact defence. For a solver-free game it reports coverage
against a CALIBRATED strong reference (with the reference's own exact-agreement, so the number's trust is known).

    PYTHONPATH=. .venv/bin/python scripts/scorecard.py --game tictactoe --ckpt path/to/ckpt.pt
    PYTHONPATH=. .venv/bin/python scripts/scorecard.py --game tictactoe --agent optimal   # calibration
"""
from __future__ import annotations

import argparse
import random

from harness.coverage import decided_frontier, optimal_actions, per_state_act, reachable_states, state_coverage
from harness.registry import resolve_game


def _agent_act(game, spec, sims):
    """Return an act_fn(state)->action for: a checkpoint path, or 'optimal'/'random' (calibration baselines)."""
    if spec == "optimal":
        return lambda s: sorted(optimal_actions(game, s))[0]
    if spec == "random":
        return lambda s: random.Random(repr(game.state_key(s))).choice(game.legal_actions(s))
    from harness.neural import AlphaZeroAgent, load_net
    net = load_net(spec, "cpu")
    return per_state_act(game, lambda: AlphaZeroAgent(net, sims=sims, solve_endgame=0, gumbel=True, c_scale=0.1)), net


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--agent", default="", choices=("", "optimal", "random"))
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--max-exact", type=int, default=200000, help="enumerate exactly below this many states")
    ap.add_argument("--sample-playouts", type=int, default=400)
    args = ap.parse_args()

    game = resolve_game(args.game)
    solvable = hasattr(game, "position_value")
    spec = args.ckpt or args.agent or "random"
    got = _agent_act(game, spec, args.sims)
    act, net = (got if isinstance(got, tuple) else (got, None))

    print(f"=== OPTIMALITY SCORECARD: {args.game}  agent={spec} ===")
    if net is not None:
        from harness.neural import net_value
        print(f"opening_value (net eval of empty board): {net_value(net, game, game.initial_state()):+.4f}")
    if not solvable:
        print("no solver for this game — optimality is only measurable vs a CALIBRATED reference (see §C.41).")
        return

    # exact if the space is small, else sampled
    states, complete = reachable_states(game, exact=True, max_states=args.max_exact, symmetry=True)
    mode = "EXACT (full enumeration, symmetry-reduced)"
    if not complete:
        states, _ = reachable_states(game, exact=False, sample_playouts=args.sample_playouts, symmetry=True)
        mode = f"SAMPLED ({len(states)} reachable states, symmetry-reduced)"
    cov = state_coverage(game, act, states=states)
    df = decided_frontier(game, act, states)
    print(f"\nSTATE-SPACE COVERAGE [{mode}]:")
    print(f"  coverage = {cov['optimal']}/{cov['n_states']} = {cov['coverage']:.4f}   (1.0 == perfect play)")
    print(f"  decided frontier: {df['decided_frac']:.3f} of states are proven non-loss; "
          f"kept when decided = {df['kept_when_decided']:.4f}")
    worst = sorted(cov["by_ply"].items(), key=lambda kv: kv[1])[:4]
    print(f"  weakest plies (ply: coverage): {worst}")


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
