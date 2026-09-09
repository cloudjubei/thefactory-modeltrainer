"""Distance-to-optimal benchmark for Connect 4 — how often a policy plays a game-theoretically OPTIMAL move,
scored against the perfect-play oracle (harness/solver.py).

`oracle_optimality_rate` = the fraction of benchmark positions where the policy's chosen move lies in the
oracle's optimal SET (several moves can be equally optimal). It is the honest "how close to solved" gauge: the
oracle scores 1.0 by construction; a model approaches 1.0 as it learns perfect play.

SCOPE / HONESTY: solving from the OPENING is a minutes-long search in pure Python (mid/late positions are
instant), so the default corpus samples positions with enough stones on the board that the exact solve is fast.
A high rate here is NECESSARY but not SUFFICIENT for "solved" — verifying opening / principal-variation play
needs a precomputed PV fixture (a one-time offline solve), tracked as a follow-on.
"""
from __future__ import annotations

import random
from typing import Callable, Optional

from harness import solver
from harness.book import book_optimal_actions
from harness.game import Game, State
from harness.solver import optimal_columns
from harness.tablebase import Tablebase

ActFn = Callable[[State], int]
RefFn = Callable[[State], Optional[list[int]]]


def exact_reference(game: Game, book: Tablebase | None = None, max_empty: int = 0) -> RefFn:
    """A ground-truth optimal-move-SET oracle that is EXACT wherever it answers and returns None where there is
    NO exact ground truth (so the caller treats that ply as UNVERIFIED, never a blunder): the book (one-ply
    lookahead over stored exact values) where it covers, else the exact solver where the position is cheap to
    solve (≤ `max_empty` empty cells), else None. As the book's opening coverage grows this reference reaches
    the opening and the trace below verifies the whole game — game-agnostic via the `exact_optimal_actions` hook."""

    def ref(state: State) -> list[int] | None:
        if book is not None:
            acts = book_optimal_actions(book, game, state)
            if acts is not None:
                return acts
        solve = getattr(game, "exact_optimal_actions", None)
        return solve(state, max_empty) if solve is not None else None

    return ref


def optimality_trace(
    game: Game, act_fn: ActFn, reference: RefFn, max_plies: int | None = None, rng: random.Random | None = None
) -> dict:
    """Walk `act_fn`'s greedy line from the standard start and grade each move against `reference(state)` (the
    optimal SET, or None where no ground truth exists). Reports the FIRST ply the agent left an AVAILABLE
    optimal set (localises WHERE it breaks — the opening-conversion failure a late-only positional benchmark
    cannot see), how many plies could be VERIFIED at all, and the line played."""
    rng = rng or random.Random(0)
    state = game.initial_state(rng)
    line: list[int] = []
    verified = 0
    first_blunder: int | None = None
    ply = 0
    while not game.is_terminal(state) and (max_plies is None or ply < max_plies):
        move = act_fn(state)
        ref_set = reference(state)
        if ref_set is not None:
            verified += 1
            if first_blunder is None and move not in ref_set:
                first_blunder = ply
        line.append(move)
        state = game.step(state, move)
        ply += 1
    return {"first_blunder_ply": first_blunder, "verified_plies": verified, "plies_played": ply, "line": line}


def exact_opponent_factory(game: Game, max_empty: int = 10**9) -> Callable[[], object]:
    """A factory for the game's EXACT (perfect-play) opponent — the ground-truth defender the SOLVE-IT
    verification plays against. Connect 4 uses the tuned `OracleAgent` (fast-win-preferring, its own solver);
    any other SolvableGame uses the generic `ExactOptimalAgent` (its `exact_optimal_actions` hook)."""
    if getattr(game, "name", "") == "connect4":
        from harness.solver import OracleAgent

        return lambda: OracleAgent()
    from harness.agents import ExactOptimalAgent

    return lambda: ExactOptimalAgent(max_empty=max_empty)


def p1_conversion(
    game: Game,
    model_factory: Callable[[], object],
    opponent_factory: Callable[[], object],
    games: int = 2,
    seed: int = 0,
    start: State | None = None,
    strategist: int = 0,
    opening_plies: int = 0,
) -> dict:
    """Play `games` games with the MODEL as the FIRST player (seat `strategist`, default 0) against `opponent`
    from `start` (default the empty board) and count how often the model reaches at least a DRAW (the winning
    seat of a first-player-win game must WIN; a drawn game's optimal seat must not LOSE). For a first-player-win
    game an optimal model converts the first move into a win every time vs perfect defence — `rate == 1.0`
    against the EXACT oracle is the definition of a solved model. A FRESH agent per game (no search-state bleed);
    deterministic per `seed`. `opening_plies` > 0 plays that many RANDOM opening moves before the agents take over,
    so a deterministic model+opponent no longer collapse to ONE repeated line — the rungs then sample DIVERSE
    defences (a robust frontier; a weak deterministic oracle can otherwise steer the single line off-distribution).
    Returns {wins, draws, losses, games, rate} where `rate` counts wins (a P1-win game)."""
    rng = random.Random(seed)
    wins = draws = losses = 0
    for _ in range(max(1, games)):
        model, opp = model_factory(), opponent_factory()
        s = game.initial_state(rng) if start is None else start
        ply = 0
        while not game.is_terminal(s):
            if ply < opening_plies:
                s = game.step(s, rng.choice(game.legal_actions(s)))
                ply += 1
                continue
            agent = model if game.current_player(s) == strategist else opp
            s = game.step(s, agent.act(game, s, rng))
            ply += 1
        w = game.winner(s)
        if w == strategist:
            wins += 1
        elif w is None:
            draws += 1
        else:
            losses += 1
    g = max(1, games)
    return {"wins": wins, "draws": draws, "losses": losses, "games": games, "rate": wins / g,
            "not_lost_rate": (wins + draws) / g}


def optimality_ladder(
    game: Game,
    model_factory: Callable[[], object],
    games: int = 8,
    seed: int = 0,
    depths: tuple[int, ...] = (6, 8, 10, 12),
    include_exact: bool = True,
    start: State | None = None,
    opening_plies: int = 0,
) -> dict:
    """The HONEST optimality ladder (Connect 4): test the model as first player against a rung of ever-stronger
    oracles — depth-6→8→10→12 near-perfect, then the EXACT solver — and report the DEEPEST rung it converts the
    first-player win against (its 'optimality frontier'). Beating a shallow depth is weak evidence (the oracle
    blunders past its horizon); only `solved` (converts vs the EXACT oracle) proves optimal play. Each rung's own
    rate is returned so a partial climb is visible. Connect 4-specific (the oracle lives in harness/solver)."""
    from harness.solver import NearPerfectOracle, OracleAgent

    rungs: list[dict] = []
    for d in depths:
        rungs.append({"label": f"depth-{d}", "kind": "oracle_depth", "depth": d,
                      "factory": (lambda d=d: NearPerfectOracle(depth=d, solve_endgame=22))})
    if include_exact:
        rungs.append({"label": "exact", "kind": "oracle_exact", "depth": None, "factory": (lambda: OracleAgent())})

    out: list[dict] = []
    frontier = "none"
    solved = False
    for r in rungs:
        res = p1_conversion(game, model_factory, r["factory"], games, seed=seed, start=start, opening_plies=opening_plies)
        cleared = res["rate"] >= 0.999
        out.append({"label": r["label"], "kind": r["kind"], "depth": r["depth"],
                    "rate": res["rate"], "games": res["games"], "cleared": cleared})
        if cleared:
            frontier = r["label"]
            if r["kind"] == "oracle_exact":
                solved = True
    return {"rungs": out, "frontier": frontier, "solved": solved}


def verify_solved(
    game: Game,
    model_factory: Callable[[], object],
    games: int = 2,
    seed: int = 0,
    start: State | None = None,
    strategist: int = 0,
    max_empty: int = 10**9,
) -> dict:
    """The SOLVE-IT gate: does the model convert the first-player win against the EXACT oracle every game? Returns
    the `p1_conversion` result plus `solved` (rate == 1.0). This is the one claim allowed to say a model plays
    perfectly — it is measured against ground-truth perfect defence, never a depth-limited proxy."""
    res = p1_conversion(game, model_factory, exact_opponent_factory(game, max_empty), games, seed=seed,
                        start=start, strategist=strategist)
    return {**res, "solved": res["rate"] >= 0.999}


def sample_forced_win_roots(game: Game, n: int, empties: int = 14, seed: int = 0,
                            shuffle: bool = True) -> list[State]:
    """`n` non-terminal positions with exactly `empties` empty cells where the SIDE TO MOVE has a PROVEN forced win
    (`position_value > 0`). These are small, FAST-to-solve roots: verifying the model converts them against the EXACT
    solver is an exact proof that SIDESTEPS the opening wall (solving a 14-empty position is milliseconds, not the
    ~minutes a from-the-opening solve costs). The reachable-set count is game-specific; Connect 4 has 42 cells."""
    from harness.book import position_value

    total_cells = getattr(game, "num_cells", None) or (getattr(game, "ROWS", 6) * getattr(game, "COLS", 7))
    target_stones = total_cells - empties
    rng = random.Random(seed)
    out: list[State] = []
    attempts = 0
    while len(out) < n and attempts < n * 20000:
        attempts += 1
        s = game.initial_state(rng)
        while sum(1 for v in s.board if v != 0) < target_stones:
            nxt = game.step(s, rng.choice(game.legal_actions(s)))
            if game.is_terminal(nxt):
                s = None
                break
            s = nxt
        if s is None or game.is_terminal(s):
            continue
        if sum(1 for v in s.board if v == 0) != empties or position_value(game, s) <= 0:
            continue
        out.append(s)
    if shuffle:
        # §C.13: return roots in a SHUFFLED order so any PREFIX is representative. The old generation order was
        # measured to be systematically easier at the front (mean conversion .804 on roots 0-31 vs .727 on
        # 32-255), and every gate / quick check reads a prefix — so the cheap checks were the biased ones.
        random.Random(seed ^ 0x5EED).shuffle(out)
    return out


def verify_forced_win_conversion(
    game: Game, model_factory: Callable[[], object], n_roots: int = 8, empties: int = 14,
    games_per_root: int = 1, seed: int = 0, max_empty: int = 10**9,
) -> dict:
    """EXACT proof, FAST: does the model convert PROVEN forced wins against the EXACT solver, from `n_roots`
    forced-win roots (few empties ⇒ each solve is cheap)? For each root the model plays the winning side vs perfect
    defence; a proven win the model fails to convert is a hard suboptimality certificate. Returns
    {converted, total, rate, roots} — `rate == 1.0` is an EXACT (not proxy) near-optimality proof over these roots,
    obtained without the from-the-opening solve. The complement to `verify_solved` (which needs the opening wall)."""
    roots = sample_forced_win_roots(game, n_roots, empties=empties, seed=seed)
    converted = 0
    details: list[dict] = []
    for i, root in enumerate(roots):
        res = verify_solved(game, model_factory, games=games_per_root, seed=seed + i, start=root,
                            strategist=root.to_move, max_empty=max_empty)
        ok = res["rate"] >= 0.999
        converted += 1 if ok else 0
        details.append({"empties": empties, "converted": ok, "wins": res["wins"], "losses": res["losses"]})
    total = len(roots)
    return {"converted": converted, "total": total, "rate": (converted / total) if total else 0.0, "roots": details}


def paired_exploitability(
    game: Game, factories: dict, depths: list[int], n_openings: int = 64, opening_plies: int = 4,
    seed: int = 0, refuter_factory: Callable[[int], object] | None = None, oracle_solve_endgame: int = 22,
) -> dict:
    """§C.19: every arm plays the SAME openings (both seats) against the SAME depth-k refuter, and we keep the
    per-cell loss outcome for each arm. Pairing on the opening is what cancels the opening-value confound that
    makes the raw single-arm exploit_rate meaningless: a position that is theoretically lost is lost for every
    arm, so it lands in neither arm's discordant set. The paired difference between two arms' outcome vectors
    (via `mcnemar_exact`) is therefore a clean RELATIVE exploitability — and because the refuter is a bounded
    searcher rather than a solver, it needs no exact oracle, which is what lets this metric transfer to games
    where forced-win conversion cannot go.

    `refuter_factory(depth) -> agent` is injectable so a transfer game can supply its own bounded searcher;
    the default is `NearPerfectOracle(depth=depth)`, the calibration game's refuter."""
    from harness.solver import NearPerfectOracle

    if refuter_factory is None:
        def refuter_factory(depth):
            return NearPerfectOracle(depth=depth, solve_endgame=oracle_solve_endgame)

    def opening(i):
        rng = random.Random(seed * 100003 + i)
        st = game.initial_state(rng)
        for _ in range(opening_plies):
            if game.is_terminal(st):
                break
            st = game.step(st, rng.choice(game.legal_actions(st)))
        return st

    by_depth: list[dict] = []
    for depth in depths:
        arms_out: dict = {name: {"outcomes": []} for name in factories}
        for model_seat in (0, 1):
            for i in range(n_openings):
                start = opening(i)
                for name, fac in factories.items():
                    seats = {model_seat: fac(), 1 - model_seat: refuter_factory(depth)}
                    st = start
                    rng = random.Random(90001 + i)  # greedy agents; fixed so the refuter is reproducible
                    while not game.is_terminal(st):
                        st = game.step(st, seats[game.current_player(st)].act(game, st, rng))
                    arms_out[name]["outcomes"].append(1 if game.returns(st)[model_seat] < 0 else 0)
        for name, a in arms_out.items():
            n = len(a["outcomes"])
            a["losses"] = sum(a["outcomes"])
            a["n"] = n
            a["loss_rate"] = a["losses"] / n if n else 0.0
        by_depth.append({"depth": depth, "arms": arms_out, "n": 2 * n_openings})
    return {"by_depth": by_depth, "opening_plies": opening_plies, "seed": seed,
            "roots_id": f"exploit_e{opening_plies}_s{seed}_n{n_openings}"}


def lbr_screen(
    game: Game, model_factory: Callable[[], object], depths: list[int], n_openings: int = 20,
    opening_plies: int = 4, seed: int = 0, oracle_solve_endgame: int = 22,
) -> dict:
    """LBR cheap-screen (§C.8 #14): the model plays BOTH seats against depth-k RESTRICTED best responders
    (`NearPerfectOracle(depth=k)`) from a fixed diverse-opening corpus. The per-depth loss rate profiles
    exploitability: how deep a refuter must look before it exploits the policy. An always-on, oracle-reusing
    robustness gauge — far cheaper than a learned adversary, strictly weaker (a k-ply screen, not a bound)."""
    from harness.solver import NearPerfectOracle

    by_depth: list[dict] = []
    for depth in depths:
        seats_out = {}
        losses = 0
        for model_seat in (0, 1):
            w = d = l = 0
            for i in range(n_openings):
                rng = random.Random(seed * 100003 + i)
                state = game.initial_state(rng)
                for _ in range(opening_plies):
                    if game.is_terminal(state):
                        break
                    state = game.step(state, rng.choice(game.legal_actions(state)))
                seats = {model_seat: model_factory(),
                         1 - model_seat: NearPerfectOracle(depth=depth, solve_endgame=oracle_solve_endgame)}
                while not game.is_terminal(state):
                    state = game.step(state, seats[game.current_player(state)].act(game, state, rng))
                r = game.returns(state)[model_seat]
                w, d, l = w + (r > 0), d + (r == 0), l + (r < 0)
            seats_out["as_p1" if model_seat == 0 else "as_p2"] = {
                "win": w / n_openings, "draw": d / n_openings, "loss": l / n_openings}
            losses += l
        by_depth.append({"depth": depth, **seats_out, "exploit_rate": losses / (2 * n_openings)})
    return {"by_depth": by_depth, "games_per_depth": 2 * n_openings, "opening_plies": opening_plies, "seed": seed}


def sim_scaling_curve(
    game: Game,
    model_factory: Callable[[int], object],
    reference_factory: Callable[[], object],
    sims_list: tuple[int, ...] = (2, 8, 32, 128),
    games: int = 20,
    seed: int = 0,
    start: State | None = None,
    strategist: int = 0,
) -> dict:
    """STRENGTH-PER-COMPUTE (the generic near-optimal headline, solver-OPTIONAL). Measure the model's first-player
    conversion vs a FIXED reference at each simulation budget — `model_factory(sims)` builds the model with that
    many sims. A near-optimal model shows a curve that FLATTENS early (little gain from few→many sims) AND whose
    low-sim point is already high: it has internalised optimal play rather than searching its way there each move.
    `reference_factory` is the EXACT oracle for the Connect-4 audit, or a frozen champion for the chess-realistic
    metric (no solver needed). `flatness = rate(max sims) − rate(min sims)` (→0 = the strength is in the NET, the
    win-per-sim goal); `auc` = the mean conversion across budgets (area-under-curve, the one-number summary)."""
    points: list[dict] = []
    for s in sims_list:
        res = p1_conversion(game, lambda s=s: model_factory(s), reference_factory, games, seed=seed,
                            start=start, strategist=strategist)
        points.append({"sims": s, "rate": res["rate"], "not_lost_rate": res["not_lost_rate"]})
    low, high = points[0]["rate"], points[-1]["rate"]
    return {"points": points, "low_sim_rate": low, "high_sim_rate": high, "flatness": high - low,
            "auc": sum(p["rate"] for p in points) / len(points)}


def sample_solvable_positions(game: Game, n: int, min_moves: int, seed: int) -> list[State]:
    """`n` random-play, non-terminal positions with at least `min_moves` stones already down (so the exact
    solve is cheap) and at least two legal replies (so 'optimal move' is a real choice)."""
    rng = random.Random(seed)
    out: list[State] = []
    attempts = 0
    while len(out) < n and attempts < n * 300:
        attempts += 1
        s = game.initial_state(rng)
        depth = min_moves + rng.randint(0, 6)
        for _ in range(depth):
            legal = game.legal_actions(s)
            if not legal:
                break
            s = game.step(s, rng.choice(legal))
            if s.done:
                break
        if not s.done and len(game.legal_actions(s)) >= 2:
            out.append(s)
    return out


def optimality_rate(states: list[State], act_fn: ActFn) -> tuple[int, int]:
    """(optimal moves, positions scored) — a move counts optimal iff it's in the oracle's optimal set."""
    optimal = 0
    for s in states:
        if act_fn(s) in optimal_columns(s, tt=solver._TT):
            optimal += 1
    return optimal, len(states)


def evaluate_optimality(
    game: Game,
    act_fn: ActFn,
    states: list[State] | None = None,
    n: int = 120,
    min_moves: int = 20,
    seed: int = 0,
) -> dict:
    """Score a policy's optimality against the oracle. Pass `states` to score the policy on positions IT
    reached (the meaningful corpus); omit to sample a fixed random-play corpus."""
    if states is None:
        states = sample_solvable_positions(game, n, min_moves, seed)
    optimal, total = optimality_rate(states, act_fn)
    return {
        "oracle_optimality_rate": (optimal / total) if total else 0.0,
        "oracle_positions": total,
        "oracle_min_moves": min_moves,
    }


def graded_conversion(game: Game, model_factory: Callable[[], object], roots: list[State],
                      games_per_root: int = 4, seed: int = 0, max_empty: int = 10**9,
                      randomized_defence: bool = True) -> dict:
    """GRADED forced-win conversion (§C.13): per root, the FRACTION of games converted against perfect defence
    that varies among equally-optimal replies. The binary metric asks "did it beat THE canonical defence" (1 bit,
    high variance); this asks "what share of perfect defences does it beat" — more bits per root at identical
    root-sampling cost, which is the cheap way to buy statistical power (roots cost seconds, training runs hours).

    Returns `scores` (per-root fractions), `rate` (their mean) and `binary_rate` (the old all-or-nothing metric,
    kept so historical numbers remain comparable)."""
    from harness.solver import OracleAgent, RandomizedOracleAgent

    if getattr(game, "name", "") == "connect4":
        opp = (lambda: RandomizedOracleAgent()) if randomized_defence else (lambda: OracleAgent())
    else:
        opp = exact_opponent_factory(game, max_empty)
    scores: list[float] = []
    for i, root in enumerate(roots):
        res = p1_conversion(game, model_factory, opp, games=max(1, games_per_root),
                            seed=seed + i * 7919, start=root, strategist=root.to_move)
        scores.append(res["rate"])
    n = max(1, len(scores))
    return {"scores": scores, "rate": sum(scores) / n, "n_roots": len(scores),
            "games_per_root": games_per_root,
            "binary_rate": sum(1 for x in scores if x >= 0.999) / n}


def paired_conversion(game: Game, factories: dict, n_roots: int = 256, empties: int = 24, seed: int = 0,
                      games_per_root: int = 4, max_empty: int = 22,
                      randomized_defence: bool = True) -> dict:
    """Score EVERY arm on the SAME roots in one call (§C.13 G2) — pairing becomes structural rather than a thing
    the caller has to remember, which is how a mixed-root comparison got made by hand once already. Returns each
    arm's graded scores plus a `roots_id` identifying the exact root family for the analysis ledger."""
    roots = sample_forced_win_roots(game, n_roots, empties=empties, seed=seed)
    arms = {name: graded_conversion(game, f, roots, games_per_root=games_per_root, seed=seed,
                                    max_empty=max_empty, randomized_defence=randomized_defence)
            for name, f in factories.items()}
    return {"arms": arms, "roots_id": f"e{empties}_s{seed}_n{len(roots)}_g{games_per_root}",
            "n_roots": len(roots), "games_per_root": games_per_root}
