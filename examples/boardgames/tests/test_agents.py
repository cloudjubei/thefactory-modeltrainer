import random

import pytest

from games.connect4 import Connect4
from games.tictactoe import TicTacToe
from harness.agents import HeuristicAgent, MctsAgent, RandomAgent, resolve_agent
from harness.book import build_book
from harness.tablebase import Tablebase


def test_mcts_proven_returns_reads_the_book_as_an_exact_leaf():
    game = TicTacToe()
    book = Tablebase(cap=10_000)
    build_book(game, book, max_plies=9, max_positions=100_000)  # a full PROVEN book
    agent = MctsAgent(sims=4, book=book)
    s = game.initial_state(random.Random(0))
    r = agent._proven_returns(game, s)  # the empty board is a proven DRAW for ttt → zero-sum [0, 0]
    assert r == [0.0, 0.0]
    # a bookless, non-solving agent has no proof to lean on here → falls back to a rollout (None)
    assert MctsAgent(sims=4)._proven_returns(game, s) is None


def test_book_aware_mcts_plays_optimally_from_either_seat_with_few_sims():
    # With every leaf a PROVEN exact value, the search backs up truth instead of noisy rollouts → optimal play
    # from a tiny sim budget. On tic-tac-toe (a draw) the book-aware agent never loses from either seat.
    game = TicTacToe()
    book = Tablebase(cap=10_000)
    build_book(game, book, max_plies=9, max_positions=100_000)
    agent = MctsAgent(sims=6, book=book)
    opp = RandomAgent()
    rng = random.Random(3)
    for seat in (0, 1):
        for _ in range(20):
            s = game.initial_state(rng)
            while not game.is_terminal(s):
                mover = game.current_player(s)
                s = game.step(s, agent.act(game, s, rng) if mover == seat else opp.act(game, s, rng))
            assert game.winner(s) != (1 - seat)  # the book-aware agent never loses


def test_random_agent_only_plays_legal_moves():
    game, agent, rng = Connect4(), RandomAgent(), random.Random(1)
    state = game.initial_state(rng)
    for _ in range(10):
        if game.is_terminal(state):
            break
        a = agent.act(game, state, rng)
        assert a in game.legal_actions(state)
        state = game.step(state, a)


def test_heuristic_agent_takes_an_immediate_win():
    game = Connect4()
    state = game.initial_state(random.Random(0))
    for c in [0, 6, 1, 6, 2, 6]:  # player 0 has 0,1,2 on the bottom, to move
        state = game.step(state, c)
    assert HeuristicAgent().act(game, state, random.Random(0)) == 3


def test_mcts_beats_random_convincingly():
    game = Connect4()
    rng = random.Random(7)
    wins = 0
    n = 20
    for i in range(n):
        model = MctsAgent(sims=60)
        opp = RandomAgent()
        seats = [model, opp] if i % 2 == 0 else [opp, model]
        model_seat = 0 if i % 2 == 0 else 1
        state = game.initial_state(rng)
        while not game.is_terminal(state):
            state = game.step(state, seats[game.current_player(state)].act(game, state, rng))
        if game.winner(state) == model_seat:
            wins += 1
    assert wins / n >= 0.7  # a 60-sim search should crush random in Connect 4


def test_mcts_counts_the_simulations_it_spends():
    game = Connect4()
    agent = MctsAgent(sims=50)
    agent.act(game, game.initial_state(random.Random(0)), random.Random(0))
    assert agent.sims_used == 50


def _replay(game, actions):
    state = game.initial_state(random.Random(0))
    for a in actions:
        state = game.step(state, a)
    return state


def test_mcts_takes_an_immediate_win():
    game = Connect4()
    # player 0 has three in column 3 (rows 0-2) and is to move: col 3 completes a vertical four.
    state = _replay(game, [3, 0, 3, 1, 3, 2])
    assert game.current_player(state) == 0
    assert MctsAgent(sims=200).act(game, state, random.Random(0)) == 3


def test_mcts_blocks_an_immediate_threat():
    game = Connect4()
    # player 1 has three in column 3; player 0 is to move and MUST block col 3 or lose next turn. A real
    # tree search sees the opponent's winning reply and blocks; flat Monte-Carlo often misses it.
    state = _replay(game, [0, 3, 1, 3, 2, 3])
    assert game.current_player(state) == 0
    assert MctsAgent(sims=400).act(game, state, random.Random(0)) == 3


def test_mcts_takes_the_win_without_needing_search():
    game = Connect4()
    state = _replay(game, [3, 0, 3, 1, 3, 2])  # P0 completes column 3 vertically
    assert MctsAgent(sims=1).act(game, state, random.Random(0)) == 3  # the tactical guard, not search depth


def test_mcts_never_hands_the_opponent_a_mate_in_one():
    game = Connect4()
    state = _replay(game, [0, 3, 1, 3, 2, 3])  # P1 threatens column 3; every other move loses on the spot
    assert MctsAgent(sims=1).act(game, state, random.Random(0)) == 3  # blocks even at a single simulation


def test_mcts_builds_a_tree_beyond_the_root():
    game = Connect4()
    agent = MctsAgent(sims=200)
    agent.act(game, game.initial_state(random.Random(0)), random.Random(0))
    # A real UCT tree expands past the root + its 7 children (8 nodes); flat MC would keep only the root.
    assert len(agent._tree) > 8


def test_mcts_reuses_its_tree_across_moves():
    game = Connect4()
    agent = MctsAgent(sims=150)
    rng = random.Random(3)
    state = game.initial_state(rng)
    agent.act(game, state, rng)
    n1 = len(agent._tree)
    state = game.step(state, 3)  # our move
    state = game.step(state, 2)  # opponent reply
    agent.act(game, state, rng)
    n2 = len(agent._tree)
    assert n2 >= n1  # the transposition table persists across moves (memory), it is not reset each turn


def test_exact_optimal_actions_gates_on_solve_cost():
    from harness.benchmark import sample_solvable_positions
    from harness.solver import optimal_columns

    game = Connect4()
    # The opening (42 empty cells) is a minutes-long exact solve → refused under a cheap threshold.
    assert game.exact_optimal_actions(game.initial_state(), max_empty=14) is None
    late = sample_solvable_positions(game, n=1, min_moves=32, seed=1)[0]  # ≤ ~10 empty → instant solve
    assert game.exact_optimal_actions(late, max_empty=40) == optimal_columns(late)
    assert game.exact_optimal_actions(late, max_empty=0) is None  # explicitly disabled


def test_mcts_solve_endgame_plays_perfectly_on_late_positions():
    from harness.benchmark import sample_solvable_positions
    from harness.solver import optimal_columns

    game = Connect4()
    # With the exact-endgame cutoff on, even a ONE-simulation mcts is game-theoretically optimal in the endgame:
    # the solver, not the tree, chooses. This is the mechanism that drives its endgame loss rate to zero.
    solver_mcts = MctsAgent(sims=1, solve_endgame=40)
    states = sample_solvable_positions(game, n=20, min_moves=30, seed=7)
    assert states
    for s in states:
        assert solver_mcts.act(game, s, random.Random(0)) in optimal_columns(s)


def test_mcts_solve_endgame_is_off_by_default_so_reference_rungs_stay_pure():
    # The fixed-strength reference rungs (opponent mcts) must NOT gain solver assistance, or the rating anchors
    # would shift — the cutoff is strictly opt-in.
    assert MctsAgent(sims=10).solve_endgame == 0
    assert resolve_agent("mcts", {}).solve_endgame == 0
    assert resolve_agent("mcts", {"mcts_sims": 25, "mcts_solve_endgame": 12}).solve_endgame == 12


def test_resolve_agent_builds_baselines_and_rejects_unknown():
    assert isinstance(resolve_agent("random", {}), RandomAgent)
    assert isinstance(resolve_agent("heuristic", {}), HeuristicAgent)
    m = resolve_agent("mcts", {"mcts_sims": 25})
    assert isinstance(m, MctsAgent) and m.sims == 25
    with pytest.raises(ValueError):
        resolve_agent("nope", {})
    with pytest.raises(NotImplementedError):
        resolve_agent("champion:abc", {})


# --- MCTS-Solver: proof PROPAGATION / SELECTION half (proven leaves already covered above) -----------------
_C4_FIND_CACHE: dict = {}


def _find_c4_position(game, empties, value):
    """The first sampled connect4 position with exactly `empties` empty cells, exact mover value `value`, and (for
    a win) NO immediate mate-in-1 — so the search cannot short-circuit and a proof must be DERIVED. Memoised so
    two tests sharing a target scan only once."""
    from games.connect4 import COLS, ROWS
    from harness.benchmark import sample_solvable_positions

    ck = (empties, value)
    if ck in _C4_FIND_CACHE:
        return _C4_FIND_CACHE[ck]
    target_ply = COLS * ROWS - empties
    found = None
    for seed in range(200):
        for s in sample_solvable_positions(game, n=4, min_moves=target_ply, seed=seed):
            if sum(1 for v in s.board if v != 0) != target_ply:
                continue
            me = game.current_player(s)
            if value > 0 and any(game.step(s, a).winner == me for a in game.legal_actions(s)):
                continue
            if game.position_value(s) != value:
                continue
            found = s
            break
        if found is not None:
            break
    _C4_FIND_CACHE[ck] = found
    return found


def _find_ttt_draw(game, empties):
    """A ply-(9-empties) tic-tac-toe position that is a proven DRAW with no immediate win, so a fixed-`empties`
    frontier below it forces the all-children (draw) branch of the solver."""
    target_ply = 9 - empties
    for seed in range(500):
        r = random.Random(seed)
        s = game.initial_state(r)
        while sum(1 for v in s.board if v != 0) < target_ply and not game.is_terminal(s):
            s = game.step(s, r.choice(game.legal_actions(s)))
        if game.is_terminal(s) or sum(1 for v in s.board if v != 0) != target_ply:
            continue
        me = game.current_player(s)
        if any(game.step(s, a).winner == me for a in game.legal_actions(s)):
            continue
        if game.position_value(s) == 0:
            return s
    return None


def test_mcts_solver_proves_the_root_and_plays_the_winning_move():
    from harness.agents import state_key
    from harness.solver import optimal_columns

    game = Connect4()
    pos = _find_c4_position(game, empties=12, value=1)  # a proven win that is NOT a mate-in-1 → a proof must be derived
    assert pos is not None
    empties = sum(1 for v in pos.board if v == 0)
    agent = MctsAgent(sims=60, solve_endgame=empties - 1)  # root (12 empty) NOT directly solved; children (11) are
    move = agent.act(game, pos, random.Random(0))
    assert agent._proven[state_key(game, pos)] == 1.0  # the root is now a DERIVED proof, not a mere high average
    assert move in optimal_columns(pos)  # and the played move is game-theoretically optimal


def test_mcts_solver_stops_searching_once_the_root_is_proven():
    game = Connect4()
    pos = _find_c4_position(game, empties=12, value=1)
    assert pos is not None
    empties = sum(1 for v in pos.board if v == 0)
    agent = MctsAgent(sims=500, solve_endgame=empties - 1)
    agent.act(game, pos, random.Random(0))
    assert agent.sims_used < 100  # search terminated as soon as the win was proven, not after the full 500 budget


def test_mcts_solver_propagates_a_proof_through_two_plies():
    from harness.agents import state_key

    game = Connect4()
    pos = _find_c4_position(game, empties=8, value=1)  # a proven win, 8 empty cells
    assert pos is not None
    empties = sum(1 for v in pos.board if v == 0)
    agent = MctsAgent(sims=400, solve_endgame=empties - 2)  # children (7 empty) NOT solvable; grandchildren (6) ARE
    agent.act(game, pos, random.Random(0))
    assert agent._proven[state_key(game, pos)] == 1.0  # the proof bubbled UP from solved grandchildren — leaves alone can't


def test_mcts_solver_proves_a_drawn_root_when_every_child_resolves():
    from harness.agents import state_key

    game = TicTacToe()
    pos = _find_ttt_draw(game, empties=4)  # a proven draw: no winning move, so the root proves only via ALL children
    assert pos is not None
    empties = sum(1 for v in pos.board if v == 0)
    agent = MctsAgent(sims=200, solve_endgame=empties - 1)  # root not directly solved; every child is
    agent.act(game, pos, random.Random(0))
    assert agent._proven[state_key(game, pos)] == 0.0  # ALL children resolved → the root is proven a DRAW, not a win/loss


def test_mcts_solver_matches_the_oracle_across_many_positions():
    # Differential check against ground truth: on many DISTINCT winning midgame positions whose root is NOT directly
    # solvable (children/grandchildren are), the propagated proof must make the played move game-theoretically
    # OPTIMAL. A sign error or a bad all-children/short-circuit rule would misplay at least one of these.
    from harness.benchmark import sample_solvable_positions
    from harness.solver import optimal_columns

    game = Connect4()
    checked = 0
    for seed in range(220):
        for s in sample_solvable_positions(game, n=3, min_moves=30, seed=seed):
            empties = sum(1 for v in s.board if v == 0)
            if not (10 <= empties <= 13):
                continue
            me = game.current_player(s)
            if any(game.step(s, a).winner == me for a in game.legal_actions(s)):
                continue  # skip mate-in-1: the tactical guard, not a derived proof, would answer it
            if game.position_value(s) != 1:
                continue  # a proven WIN → propagation resolves it deterministically
            agent = MctsAgent(sims=150, solve_endgame=empties - 2)  # root & children NOT solvable; grandchildren ARE
            assert agent.act(game, s, random.Random(0)) in optimal_columns(s)
            checked += 1
            if checked >= 20:
                return
    assert checked >= 20  # the corpus really exercised the two-ply propagation path


def test_mcts_without_book_or_solver_derives_no_proofs_so_reference_rungs_stay_pure():
    game = Connect4()
    pos = _find_c4_position(game, empties=8, value=1)
    assert pos is not None
    agent = MctsAgent(sims=200)  # pure rollout MCTS: no book, solve_endgame=0
    agent.act(game, pos, random.Random(0))
    assert agent._proven == {}  # the solver overlay is inert without a proof source
    assert agent.sims_used == 200  # and it spends its FULL budget — no early proof termination
