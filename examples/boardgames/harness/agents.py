"""Dependency-light agents — the first `model_name` cores AND the opponent-ladder rungs (no torch).

An Agent is any object with `act(game, state, rng) -> int`. The self-play harness hands agents the full
state (search needs it); a live-deployable LEARNED policy must instead rely only on `game.observation(...)`
(see `select_action` on a policy), which is what a BoardGameArena bridge would call.

The opponent axis is an extensible REGISTRY, not a fixed list: baseline rungs (`random`/`heuristic`/`mcts`)
live here; a game registers its protocol PERSONAS; and champion CHECKPOINTS are added as they are crowned —
the league/population an AlphaGo-style trainer plays against. `resolve_agent` builds an agent from a name.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Protocol

from harness.game import Game, State


class Agent(Protocol):
    def act(self, game: Game, state: State, rng: random.Random) -> int: ...


class RandomAgent:
    kind = "random"

    def act(self, game: Game, state: State, rng: random.Random) -> int:
        return rng.choice(game.legal_actions(state))


class ExactOptimalAgent:
    """Perfect play for ANY SolvableGame via its `exact_optimal_actions` hook — the game-AGNOSTIC exact oracle
    (Connect 4's `OracleAgent` is the tuned, fast-win-preferring specialisation for that game). Among equally
    optimal moves it plays centre-first, deterministically. On a game whose opening is expensive to solve
    (Connect 4) it is only affordable in the endgame; on a small game (tic-tac-toe) it is perfect everywhere,
    instantly — so it is the exact reference for the generic SOLVE-IT verification / ladder on such games."""

    kind = "exact"

    def __init__(self, max_empty: int = 10**9):
        self.max_empty = int(max_empty)

    def act(self, game: Game, state: State, rng: random.Random) -> int:
        solve = getattr(game, "exact_optimal_actions", None)
        acts = solve(state, self.max_empty) if solve is not None else None
        if not acts:
            raise ValueError("ExactOptimalAgent needs a SolvableGame position it can resolve within max_empty")
        mid = game.num_actions // 2
        return min(acts, key=lambda a: (abs(a - mid), a))


class HeuristicAgent:
    kind = "heuristic"

    def act(self, game: Game, state: State, rng: random.Random) -> int:
        return game.heuristic_action(state, rng)


class _Node:
    """One search-tree node = one game POSITION (keyed in the transposition table). Holds per-child edge
    statistics; `select` is UCT over the children."""

    __slots__ = ("legal", "child_n", "child_w", "total")

    def __init__(self, legal: list[int]):
        self.legal = legal
        self.child_n = {a: 0 for a in legal}
        self.child_w = {a: 0.0 for a in legal}
        self.total = 0

    def select(self, c_puct: float) -> int:
        log_total = math.log(self.total + 1.0)
        best_a, best_u = self.legal[0], -math.inf
        for a in self.legal:
            n = self.child_n[a]
            if n == 0:
                return a  # try every child once before exploiting
            u = self.child_w[a] / n + c_puct * math.sqrt(log_total / n)
            if u > best_u:
                best_u, best_a = u, a
        return best_a

    def update(self, action: int, value: float) -> None:
        self.child_n[action] += 1
        self.child_w[action] += value
        self.total += 1


def state_key(game: Game, state: State) -> object:
    """A hashable, canonical key for a position — a game's own `state_key` when it provides one (so
    transpositions collapse), else a fallback from the current player's observation + side to move."""
    fn = getattr(game, "state_key", None)
    if fn is not None:
        return fn(state)
    return (tuple(game.observation(state, game.current_player(state))), game.current_player(state))


# --- generic one-ply tactical helpers (domain-oblivious — only the Game protocol) -----------------------
def _hands_opponent_win(game: Game, state: State, action: int) -> bool:
    """Would playing `action` let the opponent win on their very next move? (never walk into a mate-in-1)."""
    s2 = game.step(state, action)
    if game.is_terminal(s2):
        return False
    opp = game.current_player(s2)
    return any(game.step(s2, b).winner == opp for b in game.legal_actions(s2))


def _tactical_move(game: Game, state: State, rng: random.Random) -> int:
    """A tactically-sane rollout move: the game's heuristic (win / block / centre) when it exposes one, else
    uniform-random. This is what stops MCTS's Monte-Carlo returns from being tactically blind."""
    fn = getattr(game, "heuristic_action", None)
    return fn(state, rng) if fn is not None else rng.choice(game.legal_actions(state))


# --- MCTS-Solver proof algebra (shared by the rollout and net-guided cores) --------------------------------
# A `proven` overlay maps a position key → its DERIVED game-theoretic sign (+1 win / 0 draw / -1 loss, relative
# to the side to move). These pure functions seed it at leaves and PROPAGATE it up a search path by negamax with
# a win short-circuit, so an internal node — even the root — can become a genuine PROOF rather than a mere average.
def _sign(v: float) -> float:
    return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)


def mover_returns(game: Game, state: State, mover_value: float) -> list[float]:
    """Zero-sum per-player returns from a single mover-relative value: the mover gets it, the opponent its
    negation (2-player; a lone value otherwise)."""
    r = [0.0] * game.num_players
    m = game.current_player(state)
    r[m] = mover_value
    if game.num_players == 2:
        r[1 - m] = -mover_value
    return r


def child_move_value(game: Game, state: State, action: int, proven: dict) -> float | None:
    """The PROVEN value (sign) to the mover of playing `action`, or None if not yet proven: a terminal child reads
    its own result; a proven non-terminal child negates its stored opponent-relative value (zero-sum, 2-player)."""
    child = game.step(state, action)
    if game.is_terminal(child):
        return _sign(game.returns(child)[game.current_player(state)])
    ck = state_key(game, child)
    if ck in proven and game.num_players == 2:
        return -proven[ck]
    return None


def prove_node(game: Game, state: State, key: object, proven: dict) -> None:
    """Try to prove `state` from its children (negamax, win short-circuit): a WINNING move proves it at once;
    otherwise it is proven only when EVERY child is (then the best child value stands). Idempotent — a proven
    value is final, so it is never rewritten."""
    if key in proven:
        return
    vals = [child_move_value(game, state, a, proven) for a in game.legal_actions(state)]
    if any(v == 1.0 for v in vals):
        proven[key] = 1.0
    elif vals and all(v is not None for v in vals):
        proven[key] = max(vals)


class MctsAgent:
    """Generic UCT Monte-Carlo Tree Search over the game protocol — a REAL tree, strength scales with `sims`.

    Each simulation SELECTS down the tree by UCT to a leaf, EXPANDS it, random-ROLLS OUT to a terminal, and
    BACKS UP the outcome along the visited edges (each from the perspective of the player who moved). Nodes
    are keyed by POSITION in a transposition table (`_tree`), so lines that reach the same board share their
    statistics AND the table persists across the agent's moves within a game — the search keeps what it has
    learned about mid-game positions instead of re-deriving it each turn. Domain-oblivious: only the Game
    protocol is used, so the same search plays every game. Reports `sims_used` for the harness cost tally.
    """

    kind = "mcts"

    def __init__(self, sims: int = 80, c_puct: float = 1.4, solve_endgame: int = 0, book=None):
        self.sims = max(1, int(sims))
        self.c_puct = c_puct
        # Opt-in EXACT-ENDGAME cutoff (empty-cell threshold): once a game exposes `exact_optimal_actions` and the
        # position is small enough to solve outright, play a provably-optimal move instead of trusting the tree —
        # so a deep-tactical miss can't cost a lost endgame. 0 = pure MCTS (the fixed-strength reference rungs).
        self.solve_endgame = int(solve_endgame)
        # Opt-in PROOF LEAVES: a `book` (proven-value store) makes the search back up the EXACT outcome wherever a
        # position is proven/solvable instead of a noisy rollout — truth propagates through the tree, so book
        # coverage pays off in PLAY. None = pure rollout MCTS (the reference rungs stay a fixed-strength baseline).
        self.book = book
        self.sims_used = 0
        self._tree: dict[object, _Node] = {}
        # SELECTION half of MCTS-Solver: a position key → its DERIVED game-theoretic sign (+1 win / 0 draw / -1 loss,
        # mover-relative). Seeded at proof leaves and PROPAGATED up (negamax with a win short-circuit) so an internal
        # node — even the root — becomes a genuine PROOF, not a high average: a proven node prunes selection and lets
        # the search STOP early. Inert unless a proof source exists, so the fixed-strength reference rungs are unchanged.
        self._proven: dict[object, float] = {}
        self._solving = book is not None or self.solve_endgame > 0

    def _proven_returns(self, game: Game, state: State) -> list[float] | None:
        """The EXACT per-player returns if this position is proven (booked) or cheap to solve (≤ solve_endgame),
        else None (the caller falls back to a rollout). Zero-sum: the mover gets the proven mover-relative value,
        the opponent its negation. This is the 'proof leaf' that lets a rollout collapse onto ground truth."""
        v = None
        if self.book is not None:
            from harness.book import book_value

            v = book_value(self.book, game, state)  # proven value to the mover, or None
        if v is None and self.solve_endgame > 0:
            solve = getattr(game, "exact_optimal_actions", None)
            if solve is not None and solve(state, self.solve_endgame) is not None:
                from harness.book import position_value

                v = position_value(game, state)
        if v is None:
            return None
        return mover_returns(game, state, float(v))

    def act(self, game: Game, state: State, rng: random.Random) -> int:
        legal = game.legal_actions(state)
        if len(legal) == 1:
            self.sims_used += 1
            return legal[0]
        me = game.current_player(state)
        # Tactical guards, BEFORE trusting visit counts — the two sources of the residual ~1.5% loss:
        # (1) take an immediate win outright; (2) never play a move that hands the opponent a mate-in-1
        # (unless every move does). Combined with the tactical rollout below, this drives loss toward 0.
        wins = [a for a in legal if game.step(state, a).winner == me]
        if wins:
            self.sims_used += 1
            return wins[0]
        if self.solve_endgame > 0:
            solve = getattr(game, "exact_optimal_actions", None)
            optimal = solve(state, self.solve_endgame) if solve is not None else None
            if optimal:
                self.sims_used += 1
                return min(optimal, key=lambda a: abs(a - (game.num_actions // 2)))
        safe = [a for a in legal if not _hands_opponent_win(game, state, a)]
        candidates = safe or legal
        root_key = state_key(game, state)
        for _ in range(self.sims):
            self.sims_used += 1
            self._simulate(game, state, root_key, rng)
            if self._solving and root_key in self._proven:  # the root is fully proven — more search cannot change it
                break
        if self._solving:
            # SELECTION half: if the search PROVED a win, play it outright (a proven loss is already shunned by the
            # value-tiebroken robust child below, which ranks a proof-leaf's -1 beneath every drawing move).
            proven_win = [a for a in legal if child_move_value(game, state, a, self._proven) == 1.0]
            if proven_win:
                return min(proven_win, key=lambda a: abs(a - (game.num_actions // 2)))
        root = self._tree[root_key]
        # robust child: the most-visited SAFE action, ties broken by mean value
        return max(candidates, key=lambda a: (root.child_n[a], root.child_w[a] / root.child_n[a] if root.child_n[a] else 0.0))

    def _simulate(self, game: Game, state: State, root_key: object, rng: random.Random) -> None:
        path: list[tuple[_Node, int, int, State]] = []  # (node, action, mover, node_state) edges walked this sim
        s = state
        key = root_key
        while True:
            if game.is_terminal(s):
                returns = game.returns(s)
                if self._solving:
                    self._proven.setdefault(key, _sign(returns[game.current_player(s)]))
                break
            if self._solving and key != root_key and key in self._proven:
                # a node PROVEN in a prior simulation → treat as a leaf; do not descend a solved subtree
                returns = mover_returns(game, s, self._proven[key])
                break
            node = self._tree.get(key)
            if node is None:
                # PROOF LEAF (descended positions only — the root is always expanded so `act` can rank its moves):
                # back up the exact outcome like a terminal, no rollout, and RECORD it so the proof can propagate up.
                if key != root_key:
                    proven = self._proven_returns(game, s)
                    if proven is not None:
                        returns = proven
                        self._proven[key] = _sign(proven[game.current_player(s)])
                        break
                self._tree[key] = _Node(game.legal_actions(s))  # EXPAND this leaf, evaluate with a random rollout
                returns = self._rollout(game, s, rng)
                break
            mover = game.current_player(s)
            action = node.select(self.c_puct)
            path.append((node, action, mover, s))
            s = game.step(s, action, rng)
            key = state_key(game, s)
        for node, action, mover, _s in path:
            node.update(action, returns[mover])
        if self._solving:  # PROPAGATE proofs deepest-first: a child just proven can now prove its parent, up to the root
            for _node, _action, _mover, s_node in reversed(path):
                prove_node(game, s_node, state_key(game, s_node), self._proven)

    def _rollout(self, game: Game, state: State, rng: random.Random) -> list[float]:
        while not game.is_terminal(state):
            state = game.step(state, _tactical_move(game, state, rng), rng)
        return game.returns(state)


# --- the opponent / core registry -------------------------------------------------------------------------

_BASELINE: dict[str, Callable[[dict], Agent]] = {
    "random": lambda cfg: RandomAgent(),
    "heuristic": lambda cfg: HeuristicAgent(),
    "mcts": lambda cfg: MctsAgent(
        sims=int(cfg.get("mcts_sims", 80)), solve_endgame=int(cfg.get("mcts_solve_endgame", 0))
    ),
}


def resolve_agent(name: str, cfg: dict, personas: dict[str, Callable[[dict], Agent]] | None = None) -> Agent:
    """Build an agent by NAME from the baseline cores, a game's registered personas, or (later) a champion
    checkpoint (`champion:<ref>`). Unknown names fail loudly so a mistyped opponent can't silently pass."""
    if name in _BASELINE:
        return _BASELINE[name](cfg)
    if name == "alphazero":
        # Lazy import: the neural core needs torch, but the light cores must not, so importing it is deferred
        # to the moment an alphazero agent is actually requested.
        from harness.neural import build_alphazero_agent

        return build_alphazero_agent(cfg)
    if name == "oracle":
        # Perfect-play Connect 4 reference (torch-free). Fast on mid/late positions; solving from the opening
        # is a minutes-long search in pure Python, so it's the benchmark labeller / endgame reference, not a
        # from-scratch opening opponent (that role belongs to a depth-limited near-perfect variant).
        from harness.solver import OracleAgent

        return OracleAgent()
    if name == "oracle_depth":
        # The fast, tactically-perfect reference opponent — depth-limited near-perfect solver.
        from harness.solver import NearPerfectOracle

        return NearPerfectOracle(depth=int(cfg.get("oracle_depth", 10)))
    if name == "book":
        # The deployable optimal agent: opening book + exact endgame solver + a near-perfect fallback. Loads
        # the project-committed book for the game; provably optimal wherever the book/solver reach, fast always.
        from harness.book import load_book
        from harness.bookagent import BookAgent

        gname = cfg.get("game", "connect4")
        return BookAgent(
            load_book(gname),
            gname,
            solve_endgame=int(cfg.get("book_solve_endgame", 22)),
            depth=int(cfg.get("oracle_depth", 12)),
        )
    if personas and name in personas:
        return personas[name](cfg)
    if name.startswith("champion:"):
        raise NotImplementedError("champion checkpoints load once league play lands")
    raise ValueError(f"unknown agent/opponent {name!r}; known: {sorted(_BASELINE)} + ['alphazero'] + personas {sorted(personas or {})}")
