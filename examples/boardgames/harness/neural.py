"""AlphaZero-style neural core — a LEARNED policy+value net trained by self-play, plugged into the harness.

Unlike the search cores (random / heuristic / mcts) whose "checkpoint" is a tiny config spec, this core learns
WEIGHTS: a small conv net maps a board (from the side-to-move's perspective) to a move POLICY + a win VALUE.
It is trained on its own self-play games — the AlphaZero loop: self-play with net-guided MCTS → train the net
on (position, visit-count policy, game outcome) → repeat, each round stronger.

At play time it is a net-guided MCTS: PUCT selection biased by the net's policy prior, and leaves evaluated by
the net's value head INSTEAD of a random rollout — so the same tree machinery as `mcts`, but guided by learned
knowledge that GENERALISES across positions (what a transposition table cannot do).

Needs torch (the light cores do not). Install once: `.venv/bin/pip install torch`.
"""
from __future__ import annotations

import math
import random
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.agents import Agent, _sign, child_move_value, prove_node, state_key
from harness.game import Game, State

ROWS, COLS = 6, 7  # connect4 board dims (this first neural core is connect4-shaped; moves stay generic)


# --- the network -----------------------------------------------------------------------------------------


class Connect4Net(nn.Module):
    """Two conv layers over a 2-plane board (own / opponent) → a policy head (per column) + a value head."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(2, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.policy_head = nn.Linear(channels * ROWS * COLS, COLS)
        self.value_head = nn.Linear(channels * ROWS * COLS, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.flatten(1)
        return self.policy_head(h), torch.tanh(self.value_head(h))


def encode(game: Game, state: State) -> torch.Tensor:
    """Encode a position as a (2, ROWS, COLS) tensor from the SIDE-TO-MOVE's perspective (plane 0 = own
    pieces, plane 1 = opponent), read from the game's own observation so the net is position-canonical."""
    player = game.current_player(state)
    cells = game.observation(state, player)[: ROWS * COLS]
    own = [1.0 if v == 1.0 else 0.0 for v in cells]
    opp = [1.0 if v == -1.0 else 0.0 for v in cells]
    plane_own = torch.tensor(own, dtype=torch.float32).reshape(ROWS, COLS)
    plane_opp = torch.tensor(opp, dtype=torch.float32).reshape(ROWS, COLS)
    return torch.stack([plane_own, plane_opp])


def net_value(net: "Connect4Net", game: Game, state: State, device: str = "cpu") -> float:
    """The net's value-head estimate for `state`, from the SIDE-TO-MOVE's perspective (+1 = the mover wins). On
    the standard opening of a first-player-win game a correctly-trained net returns ~+1; a value near 0 or
    negative reveals the self-play VALUE label (a draw/loss from WEAK play) has contaminated the opening belief —
    the mechanism behind a champion throwing away the forced first-player win."""
    net.eval()
    with torch.no_grad():
        _logits, value = net(encode(game, state).unsqueeze(0).to(device))
    return float(value[0, 0])


# --- net-guided MCTS -------------------------------------------------------------------------------------


class _AZNode:
    """A search node with the net's policy PRIORS on its edges (PUCT), not just visit counts."""

    __slots__ = ("legal", "prior", "child_n", "child_w", "total", "value")

    def __init__(self, legal: list[int], prior: dict[int, float], value: float):
        self.legal = legal
        self.prior = prior
        self.child_n = {a: 0 for a in legal}
        self.child_w = {a: 0.0 for a in legal}
        self.total = 0
        self.value = value

    def select(self, c_puct: float) -> int:
        sqrt_total = math.sqrt(self.total + 1e-8)
        best_a, best_u = self.legal[0], -math.inf
        for a in self.legal:
            q = self.child_w[a] / self.child_n[a] if self.child_n[a] > 0 else 0.0
            u = q + c_puct * self.prior[a] * sqrt_total / (1 + self.child_n[a])
            if u > best_u:
                best_u, best_a = u, a
        return best_a

    def update(self, action: int, value: float) -> None:
        self.child_n[action] += 1
        self.child_w[action] += value
        self.total += 1


class AlphaZeroAgent:
    """Net-guided MCTS. `sims` PUCT simulations per move; leaves scored by the value head (no random rollout).
    Holds a transposition table across its moves in a game, like `mcts`. `temperature`/`add_noise` are the
    self-play EXPLORATION knobs (0 temperature + no noise = greedy play, used for evaluation)."""

    kind = "alphazero"

    def __init__(
        self,
        net: Connect4Net,
        sims: int = 100,
        c_puct: float = 1.5,
        device: str = "cpu",
        temperature: float = 0.0,
        add_noise: bool = False,
        dirichlet_alpha: float = 0.9,
        noise_frac: float = 0.25,
        solve_endgame: int = 0,
        book=None,
    ):
        self.net = net
        self.sims = max(1, int(sims))
        self.c_puct = c_puct
        self.device = device
        self.temperature = temperature
        self.add_noise = add_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_frac = noise_frac
        # Opt-in PROOF LEAVES: a `book` (proven values) makes the search back up the EXACT outcome at a booked or
        # endgame-solvable leaf instead of the value HEAD's estimate — truth propagates through the tree, so book
        # coverage pays off in play and self-play value targets get exact where a proof exists. None = pure net.
        self.book = book
        # Opt-in EXACT-ENDGAME cutoff (empty-cell threshold): once the position is cheap to solve, play a
        # provably-optimal move instead of the net-guided search — a perfect endgame the value head needn't
        # approximate, driving loss toward 0 / optimality toward 1. 0 = pure net-guided MCTS (self-play default).
        self.solve_endgame = int(solve_endgame)
        self.sims_used = 0
        self._nodes: dict[object, _AZNode] = {}
        # SELECTION half of MCTS-Solver (see agents.prove_node): proofs seeded at leaves PROPAGATE up so the root
        # can become a genuine PROOF. Consumed in GREEDY deployment only — self-play keeps its visit-count policy
        # untouched (propagation writes this overlay but never changes descent/backup). Inert without a proof source.
        self._proven: dict[object, float] = {}
        self._solving = book is not None or self.solve_endgame > 0

    def _proven_value(self, game: Game, state: State) -> float | None:
        """The EXACT value to the side-to-move if this position is proven (booked) or cheap to solve
        (≤ solve_endgame), else None. Lets a search leaf collapse onto ground truth instead of the value head."""
        if self.book is not None:
            from harness.book import book_value

            bv = book_value(self.book, game, state)
            if bv is not None:
                return float(bv)
        if self.solve_endgame > 0:
            solve = getattr(game, "exact_optimal_actions", None)
            if solve is not None and solve(state, self.solve_endgame) is not None:
                from harness.book import position_value

                return float(position_value(game, state))
        return None

    def _policy_value(self, game: Game, state: State) -> tuple[dict[int, float], float]:
        self.net.eval()
        with torch.no_grad():
            logits, value = self.net(encode(game, state).unsqueeze(0).to(self.device))
        legal = game.legal_actions(state)
        masked = torch.full((COLS,), -1e9)
        for a in legal:
            masked[a] = logits[0, a]
        probs = F.softmax(masked, dim=0)
        return {a: float(probs[a]) for a in legal}, float(value[0, 0])

    def _expand(self, game: Game, state: State, key: object, rng: random.Random, root: bool) -> _AZNode:
        prior, value = self._policy_value(game, state)
        if root and self.add_noise and len(prior) > 1:
            noise = _dirichlet(len(prior), self.dirichlet_alpha, rng)
            prior = {a: (1 - self.noise_frac) * p + self.noise_frac * n for (a, p), n in zip(prior.items(), noise)}
        node = _AZNode(game.legal_actions(state), prior, value)
        self._nodes[key] = node
        return node

    def _simulate(self, game: Game, state: State, root_key: object, rng: random.Random) -> None:
        path: list[tuple[_AZNode, int, int, State]] = []  # (node, action, mover, node_state)
        s = state
        key = root_key
        while True:
            node = self._nodes.get(key)
            if node is None or game.is_terminal(s):
                leaf_player = game.current_player(s)
                if game.is_terminal(s):
                    v = game.returns(s)[leaf_player]
                    if self._solving:
                        self._proven.setdefault(key, _sign(v))
                else:
                    pv = self._proven_value(game, s)  # PROOF LEAF: exact value (booked/solvable), else the net
                    if pv is not None:
                        v = pv
                        if self._solving:
                            self._proven[key] = _sign(pv)
                    else:
                        v = self._expand(game, s, key, rng, root=False).value
                for n, a, mover, _s in path:
                    n.update(a, v if mover == leaf_player else -v)
                if self._solving:  # PROPAGATE proofs up — writes the overlay only, never touches visits, so π is unchanged
                    for _n, _a, _mover, s_node in reversed(path):
                        prove_node(game, s_node, state_key(game, s_node), self._proven)
                return
            mover = game.current_player(s)
            action = node.select(self.c_puct)
            path.append((node, action, mover, s))
            s = game.step(s, action, rng)
            key = state_key(game, s)

    def run_search(self, game: Game, state: State, rng: random.Random) -> dict[int, float]:
        """Run the searches and return the visit-count policy π over actions (the self-play training target)."""
        root_key = state_key(game, state)
        if root_key not in self._nodes:
            self._expand(game, state, root_key, rng, root=True)
        for _ in range(self.sims):
            self.sims_used += 1
            self._simulate(game, state, root_key, rng)
        root = self._nodes[root_key]
        total = sum(root.child_n.values()) or 1
        return {a: root.child_n[a] / total for a in root.legal}

    def act(self, game: Game, state: State, rng: random.Random) -> int:
        legal = game.legal_actions(state)
        if len(legal) == 1:
            self.sims_used += 1
            return legal[0]
        # Exact-endgame cutoff (greedy play only — self-play keeps exploring): a solved position is played
        # perfectly, so the net is never asked to approximate an endgame the solver can nail outright.
        if self.solve_endgame > 0 and self.temperature <= 1e-6:
            solve = getattr(game, "exact_optimal_actions", None)
            optimal = solve(state, self.solve_endgame) if solve is not None else None
            if optimal:
                self.sims_used += 1
                return min(optimal, key=lambda a: abs(a - (game.num_actions // 2)))
        pi = self.run_search(game, state, rng)
        if self._solving and self.temperature <= 1e-6:
            # SELECTION half (greedy deployment only — self-play keeps its π): if the search PROVED a win, play it,
            # even where an untrained value head would not. The proof came from propagated booked/solvable leaves.
            proven_win = [a for a in legal if child_move_value(game, state, a, self._proven) == 1.0]
            if proven_win:
                return min(proven_win, key=lambda a: abs(a - (game.num_actions // 2)))
        return sample_action(pi, self.temperature, rng)


def sample_action(pi: dict[int, float], temperature: float, rng: random.Random) -> int:
    """Pick a move from a visit-count policy: greedy (argmax) at temperature 0, else sample ∝ π^(1/T)."""
    actions = list(pi.keys())
    if temperature <= 1e-6:
        return max(actions, key=lambda a: pi[a])
    weights = [pi[a] ** (1.0 / temperature) for a in actions]
    total = sum(weights) or 1.0
    r = rng.random() * total
    acc = 0.0
    for a, w in zip(actions, weights):
        acc += w
        if r <= acc:
            return a
    return actions[-1]


def _dirichlet(n: int, alpha: float, rng: random.Random) -> list[float]:
    samples = [rng.gammavariate(alpha, 1.0) for _ in range(n)]
    s = sum(samples) or 1.0
    return [x / s for x in samples]


# --- self-play + training --------------------------------------------------------------------------------


def self_play_game(
    game: Game, agent: AlphaZeroAgent, rng: random.Random, temp_moves: int = 8
) -> list[tuple[torch.Tensor, list[float], float]]:
    """Play ONE self-play game and return training examples (encoded board, visit-count policy, outcome)."""
    agent._nodes = {}
    agent.add_noise = True
    pending: list[tuple[torch.Tensor, list[float], int]] = []
    state = game.initial_state(rng)
    move = 0
    while not game.is_terminal(state):
        agent.temperature = 1.0 if move < temp_moves else 0.0
        pi = agent.run_search(game, state, rng)
        player = game.current_player(state)
        pi_vec = [pi.get(a, 0.0) for a in range(game.num_actions)]
        pending.append((encode(game, state), pi_vec, player))
        state = game.step(state, sample_action(pi, agent.temperature, rng), rng)
        move += 1
    returns = game.returns(state)
    return [(x, pi_vec, returns[player]) for (x, pi_vec, player) in pending]


def vs_opponent_game(
    game: Game,
    learner: AlphaZeroAgent,
    opponent: Agent,
    learner_seat: int,
    rng: random.Random,
    temp_moves: int = 6,
) -> list[tuple[torch.Tensor, list[float], float]]:
    """League game: the LEARNER (net-guided, exploring) plays an arbitrary opponent (a strong mcts / heuristic
    / a past champion). Training examples are collected from ONLY the learner's moves — we learn to BEAT the
    opponent, we don't imitate it."""
    learner._nodes = {}
    learner.add_noise = True
    pending: list[tuple[torch.Tensor, list[float], int]] = []
    state = game.initial_state(rng)
    move = 0
    while not game.is_terminal(state):
        player = game.current_player(state)
        if player == learner_seat:
            learner.temperature = 1.0 if move < temp_moves else 0.0
            pi = learner.run_search(game, state, rng)
            pi_vec = [pi.get(a, 0.0) for a in range(game.num_actions)]
            pending.append((encode(game, state), pi_vec, player))
            action = sample_action(pi, learner.temperature, rng)
        else:
            action = opponent.act(game, state, rng)
        state = game.step(state, action, rng)
        move += 1
    returns = game.returns(state)
    return [(x, pi_vec, returns[player]) for (x, pi_vec, player) in pending]


def head_to_head(
    game: Game,
    model_factory: Callable[[], Agent],
    opponent_factory: Callable[[], Agent],
    n: int,
    rng: random.Random,
    opening_plies: int = 2,
) -> dict[str, float]:
    """Play `n` seat-alternated games of model vs opponent and return win/draw/loss rates for the model. The
    first `opening_plies` moves are RANDOM: two greedy (deterministic) nets would otherwise replay the exact
    same game every time, so a naive "n games" would really be one game repeated — random openings make the
    n games genuinely distinct, giving a robust win-rate for the promotion gate."""
    w = d = 0
    for i in range(n):
        model_seat = i % 2
        seats: list[Agent] = [model_factory(), opponent_factory()] if model_seat == 0 else [
            opponent_factory(),
            model_factory(),
        ]
        state = game.initial_state(rng)
        ply = 0
        while not game.is_terminal(state):
            if ply < opening_plies:
                action = rng.choice(game.legal_actions(state))
            else:
                action = seats[game.current_player(state)].act(game, state, rng)
            state = game.step(state, action, rng)
            ply += 1
        winner = game.winner(state)
        if winner is None:
            d += 1
        elif winner == model_seat:
            w += 1
    games = max(1, n)
    return {"win_rate": w / games, "draw_rate": d / games, "loss_rate": (games - w - d) / games, "games": n}


def augment_examples(
    examples: list[tuple[torch.Tensor, list[float], float]], perms: list[list[int]] | None
) -> list[tuple[torch.Tensor, list[float], float]]:
    """Multiply training examples by a game's symmetries (Lever 2 for the net). Each `perm` is a column
    source-permutation `dest <- src`: it reorders the board planes' column axis and the policy vector identically;
    the value is invariant. Bakes symmetry-invariance into the net and multiplies data — worth more the harder the
    game is to encode. Identity-only (or no perms) is a plain copy."""
    if not perms or len(perms) <= 1:
        return list(examples)
    identity = list(range(len(perms[0])))
    out: list[tuple[torch.Tensor, list[float], float]] = []
    for x, pi, v in examples:
        for perm in perms:
            if perm == identity:
                out.append((x, list(pi), v))
            else:
                out.append((x[..., perm], [pi[s] for s in perm], v))
    return out


def train_net(
    net: Connect4Net,
    examples: list[tuple[torch.Tensor, list[float], float]],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> float:
    """One training pass over the buffer (policy cross-entropy + value MSE). Returns the final mean loss."""
    if not examples:
        return 0.0
    x = torch.stack([e[0] for e in examples]).to(device)
    target_p = torch.tensor([e[1] for e in examples], dtype=torch.float32).to(device)
    target_v = torch.tensor([[e[2]] for e in examples], dtype=torch.float32).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()
    last = 0.0
    n = len(examples)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            b = perm[i : i + batch_size]
            logits, value = net(x[b])
            policy_loss = -(target_p[b] * F.log_softmax(logits, dim=1)).sum(1).mean()
            value_loss = F.mse_loss(value, target_v[b])
            loss = policy_loss + value_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.detach())
    return last


def distill_examples(
    game: Game, n: int, min_moves: int, seed: int, device: str = "cpu"
) -> list[tuple[torch.Tensor, list[float], float]]:
    """Supervised (state, optimal-policy, value) examples LABELLED BY THE PERFECT ORACLE — the biggest lever
    for reaching optimal play. The policy target is uniform over the oracle's OPTIMAL move set; the value is the
    position's game-theoretic sign (mover perspective). Sampled from fast-to-solve (mid/late) positions, so it
    teaches tactical perfection cheaply; the opening layer is left to self-play against the oracle league."""
    from harness.benchmark import sample_solvable_positions
    from harness.solver import move_values

    examples: list[tuple[torch.Tensor, list[float], float]] = []
    for state in sample_solvable_positions(game, n, min_moves, seed):
        values = move_values(state, weak=True)
        if not values:
            continue
        best = max(values.values())
        optimal = [c for c, v in values.items() if v == best]
        pi = [0.0] * COLS
        for c in optimal:
            pi[c] = 1.0 / len(optimal)
        value = 1.0 if best > 0 else (-1.0 if best < 0 else 0.0)
        examples.append((encode(game, state), pi, value))
    return examples


def book_distill_examples(
    game: Game, book, states, proof_copies: int = 3, estimate_copies: int = 1, device: str = "cpu"
) -> list[tuple[torch.Tensor, list[float], float]]:
    """(state, soft-policy, value) examples LABELLED BY THE BOOK — the bridge that lets the net learn from the
    GRADED opening the exact labeller can't reach. For each covered `state` the policy target is uniform over the
    entry's stored `best_actions` and the value target is the entry's value (exact for a PROOF, the bounded-search
    belief for an ESTIMATE, kept SOFT). Proofs outweigh beliefs by whole-copy REPLICATION (`proof_copies` vs
    `estimate_copies`) — the same oversampling the distill anchor uses, so no per-example loss weights are needed.
    Positions the book does not cover, or that carry no `best_actions`, are skipped."""
    from harness.book import _key
    from harness.tablebase import PROVEN

    examples: list[tuple[torch.Tensor, list[float], float]] = []
    for state in states:
        if game.is_terminal(state):
            continue
        entry = book.entry(_key(game, state))
        if entry is None or not entry.best_actions:
            continue
        acts = [c for c in range(COLS) if (entry.best_actions >> c) & 1]
        if not acts:
            continue
        pi = [0.0] * COLS
        for c in acts:
            pi[c] = 1.0 / len(acts)
        copies = proof_copies if entry.status == PROVEN else estimate_copies
        examples.extend([(encode(game, state), pi, float(entry.value))] * max(0, copies))
    return examples


def oracle_distill_games(
    game: Game,
    n_games: int,
    seed: int,
    oracle_depth: int = 14,
    exact_max_empty: int = 22,
    device: str = "cpu",
    book=None,
) -> list[tuple[torch.Tensor, list[float], float]]:
    """Distillation over the OPTIMAL-PLAY DISTRIBUTION (opening → endgame) — the layer `distill_examples`
    (late-only) can't reach, and the measured reason a champion keeps losing to the oracle: its OPENING plays an
    EDGE column first instead of centre, throwing away the first-player win. The fix is supervised optimal moves
    from the STANDARD start: the LEARNER seat plays optimally (labelled) while its OPPONENT varies (a tight
    near-perfect oracle on some games, a RANDOM agent on others — so the net learns the optimal RESPONSE to any
    deviation, not just the single main line). Label the learner's positions: policy = the EXACT optimal move-set
    from the OPENING BOOK where it reaches (`book`, instant one-ply lookup — the upgrade that makes opening labels
    truly optimal), else the exact solver when cheap (≤ `exact_max_empty` empty cells), else the near-perfect
    oracle's move; value = the game OUTCOME (mover view). No minutes-long from-the-opening solves — the book/oracle
    carry the opening, the solver labels mid/late. The learner labels the EMPTY board (→ centre)."""
    from harness.agents import RandomAgent
    from harness.book import book_optimal_actions, book_value
    from harness.solver import NearPerfectOracle, optimal_columns

    rng = random.Random(seed)
    oracle = NearPerfectOracle(depth=oracle_depth)
    opponents: list[Callable[[], Agent]] = [lambda: NearPerfectOracle(depth=oracle_depth), lambda: RandomAgent()]

    def label(state: State) -> tuple[list[float], int, float | None]:
        empty = sum(1 for v in state.board if v == 0)
        optimal = book_optimal_actions(book, game, state) if book is not None else None  # exact opening (instant)
        if optimal is None and empty <= exact_max_empty:
            optimal = optimal_columns(state)  # exact endgame
        # VALUE relabel: where the book PROVES this position, use its exact mover-relative value as the target,
        # not the noisy game outcome — the fix for the opening value-label contamination that forfeits the win.
        bv = book_value(book, game, state) if book is not None else None
        if optimal:
            pi = [0.0] * COLS
            for c in optimal:
                pi[c] = 1.0 / len(optimal)
            action = optimal[0] if len(optimal) == 1 else min(optimal, key=lambda c: abs(c - COLS // 2))
            return pi, action, bv
        action = oracle.act(game, state, rng)
        return [1.0 if c == action else 0.0 for c in range(COLS)], action, bv

    examples: list[tuple[torch.Tensor, list[float], float]] = []
    for g in range(n_games):
        learner_seat = g % 2  # alternate seats so the net learns BOTH first- and second-player optimal play
        opponent = opponents[g % len(opponents)]()
        state = game.initial_state(rng)
        pending: list[tuple[torch.Tensor, list[float], int, float | None]] = []
        while not game.is_terminal(state):
            mover = game.current_player(state)
            if mover == learner_seat:
                pi, action, bv = label(state)
                pending.append((encode(game, state), pi, mover, bv))
            else:
                action = opponent.act(game, state, rng)
            state = game.step(state, action, rng)
        returns = game.returns(state)
        examples.extend(
            (x, pi, bv if bv is not None else returns[mover]) for (x, pi, mover, bv) in pending
        )
    return examples


def build_distill_corpus(
    game: Game, spec: dict, cache_dir: str | None = None, device: str = "cpu",
    log: Callable[[str], None] | None = None, book=None,
) -> list[tuple[torch.Tensor, list[float], float]]:
    """Build (or LOAD from disk) a broad distillation corpus per `spec`, so the expensive solves happen ONCE and
    every champion generation reuses the same optimal-play anchor. `spec` = { games, seed, oracle_depth,
    exact_max_empty, opening_plies, late: {n, min_moves} }. `book` upgrades the opening labels to EXACT wherever
    it reaches (include a `book` identity field in `spec` so a grown book re-keys the cache). Cached by a hash of
    the spec under `cache_dir`."""
    import hashlib
    import json
    import os

    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"{game.name}-distill-{key}.pt") if cache_dir else None
    if path and os.path.isfile(path):
        blob = torch.load(path, map_location=device)
        if log:
            log(f"distill corpus: {len(blob['v'])} examples (cache hit)")
        return [(x, list(pi), float(v)) for x, pi, v in zip(blob["x"], blob["pi"], blob["v"])]
    examples: list[tuple[torch.Tensor, list[float], float]] = []
    if int(spec.get("games", 0)) > 0:
        examples += oracle_distill_games(
            game, int(spec["games"]), int(spec.get("seed", 0)), int(spec.get("oracle_depth", 14)),
            int(spec.get("exact_max_empty", 22)), device, book=book,
        )
    late = spec.get("late") or {}
    if int(late.get("n", 0)) > 0:
        examples += distill_examples(game, int(late["n"]), int(late.get("min_moves", 16)), int(spec.get("seed", 0)), device)
    if path and examples:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({"x": [e[0] for e in examples], "pi": [e[1] for e in examples], "v": [e[2] for e in examples]}, path)
    if log:
        log(f"distill corpus: {len(examples)} examples (cache {'built' if path else 'no-cache'})")
    return examples


def _mix_training_set(buffer: list, distilled: list, distill_fraction: float) -> list:
    """Keep the exact distilled anchor at a FIXED FRACTION of each training pass (a DQfD-style fixed-ratio mix),
    so it never dilutes below `distill_fraction` as the self-play buffer grows — the fix for the net DRIFTING off
    the optimal opening it was distilled on (plain `buffer + distilled` sinks the ~400 anchor examples to ~5% of
    an 8000-buffer). Oversamples the small anchor by whole copies to hit the ratio (equivalent to weighting it in
    the loss). No anchor → the buffer unchanged; no buffer / fraction 0 → the old plain concatenation."""
    if not distilled:
        return list(buffer)
    if not buffer or distill_fraction <= 0:
        return list(buffer) + list(distilled)
    frac = min(distill_fraction, 0.9)
    k = max(1, round(frac / (1 - frac) * len(buffer) / len(distilled)))
    return list(buffer) + list(distilled) * k


def train_alphazero(
    game: Game,
    iterations: int = 8,
    selfplay_games: int = 32,
    sims: int = 100,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 1e-3,
    channels: int = 32,
    buffer_cap: int = 8000,
    seed: int = 0,
    device: str = "cpu",
    init_net: Connect4Net | None = None,
    opponent_pool: list[Callable[[], Agent]] | None = None,
    pool_frac: float = 0.5,
    distill_positions: int = 0,
    distill_min_moves: int = 16,
    distill_corpus: list[tuple[torch.Tensor, list[float], float]] | None = None,
    distill_fraction: float = 0.34,
    book=None,
    book_distill_positions: int = 0,
    book_distill_min_moves: int = 6,
    book_proof_copies: int = 3,
    book_estimate_copies: int = 1,
    augment: bool = True,
    log: Callable[[str], None] | None = None,
) -> tuple[Connect4Net, list[dict]]:
    """The AlphaZero loop with WARM-START + LEAGUE + optional ORACLE DISTILLATION. Starts from `init_net` (the
    champion) when given instead of a random net — so training compounds across runs rather than restarting
    from zero. When `distill_positions > 0` it first imprints the net on oracle-labelled optimal play and keeps
    those examples in EVERY training pass (a persistent 'this is the perfect move' anchor). Each iteration then
    mixes pure self-play with games against the `opponent_pool` (strong mcts / heuristic / near-perfect oracle /
    past champions) at rate `pool_frac`, and trains on the accumulated buffer + the distilled anchor."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    net = init_net if init_net is not None else Connect4Net(channels=channels).to(device)
    buffer: list[tuple[torch.Tensor, list[float], float]] = []
    perms = game.symmetries() if augment and hasattr(game, "symmetries") else None
    # A prebuilt BROAD corpus (opening→endgame, cached) is the persistent anchor when given — the layer that
    # teaches the OPENING to hold the first-player win; else fall back to the late-only sampled distillation.
    distilled = (
        distill_corpus
        if distill_corpus is not None
        else (distill_examples(game, distill_positions, distill_min_moves, seed, device) if distill_positions > 0 else [])
    )
    if book is not None and book_distill_positions > 0:
        # Fold the BOOK's proofs + graded-opening beliefs into the anchor: soft optimal-move policy targets the
        # exact late-only labeller can't reach, proofs oversampled over estimates.
        from harness.benchmark import sample_solvable_positions

        book_states = sample_solvable_positions(game, book_distill_positions, book_distill_min_moves, seed)
        distilled = distilled + book_distill_examples(
            game, book, book_states, proof_copies=book_proof_copies, estimate_copies=book_estimate_copies, device=device
        )
    distilled = augment_examples(distilled, perms)  # a position + its mirror are the same exact lesson
    if distilled:
        train_net(net, distilled, epochs, batch_size, lr, device)  # imprint optimal play before self-play
    history: list[dict] = []
    for it in range(iterations):
        learner = AlphaZeroAgent(net, sims=sims, device=device)
        fresh: list[tuple[torch.Tensor, list[float], float]] = []
        vs_pool = 0
        for _ in range(selfplay_games):
            if opponent_pool and rng.random() < pool_frac:
                opponent = opponent_pool[rng.randrange(len(opponent_pool))]()
                fresh.extend(vs_opponent_game(game, learner, opponent, rng.randint(0, 1), rng))
                vs_pool += 1
            else:
                fresh.extend(self_play_game(game, learner, rng))
        fresh = augment_examples(fresh, perms)  # symmetry-augment self-play too (2× data, invariance baked in)
        buffer = (buffer + fresh)[-buffer_cap:]
        loss = train_net(net, _mix_training_set(buffer, distilled, distill_fraction), epochs, batch_size, lr, device)
        history.append(
            {"iteration": it + 1, "examples": len(fresh), "vs_pool_games": vs_pool, "buffer": len(buffer),
             "distilled": len(distilled), "loss": loss}
        )
        if log:
            log(f"iter {it + 1}/{iterations}: +{len(fresh)} ex ({vs_pool} vs-pool), buffer {len(buffer)}, "
                f"distilled {len(distilled)}, loss {loss:.3f}")
    return net, history


def save_net(net: Connect4Net, path: str) -> None:
    torch.save({"state_dict": net.state_dict(), "channels": net.conv1.out_channels}, path)


def load_net(path: str, device: str = "cpu") -> Connect4Net:
    blob = torch.load(path, map_location=device)
    net = Connect4Net(channels=int(blob.get("channels", 32)))
    net.load_state_dict(blob["state_dict"])
    net.to(device)
    net.eval()
    return net


def build_alphazero_agent(cfg: dict) -> Agent:
    """Registry seam (lazy-imported by `resolve_agent`, so the light cores never import torch): build a
    net-guided agent, loading weights from the checkpoint's `az_weights` when present, else a fresh net."""
    device = str(cfg.get("device", "cpu"))
    weights = cfg.get("az_weights") or cfg.get("weights")
    net = load_net(weights, device) if weights else Connect4Net(channels=int(cfg.get("az_channels", 32))).to(device)
    from harness.config import DEFAULT_AZ_SOLVE_ENDGAME

    return AlphaZeroAgent(
        net,
        sims=int(cfg.get("az_sims", cfg.get("mcts_sims", 100))),
        device=device,
        solve_endgame=int(cfg.get("az_solve_endgame", DEFAULT_AZ_SOLVE_ENDGAME)),
    )
