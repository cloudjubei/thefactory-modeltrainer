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

from harness.agents import Agent, state_key
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
    ):
        self.net = net
        self.sims = max(1, int(sims))
        self.c_puct = c_puct
        self.device = device
        self.temperature = temperature
        self.add_noise = add_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_frac = noise_frac
        # Opt-in EXACT-ENDGAME cutoff (empty-cell threshold): once the position is cheap to solve, play a
        # provably-optimal move instead of the net-guided search — a perfect endgame the value head needn't
        # approximate, driving loss toward 0 / optimality toward 1. 0 = pure net-guided MCTS (self-play default).
        self.solve_endgame = int(solve_endgame)
        self.sims_used = 0
        self._nodes: dict[object, _AZNode] = {}

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
        path: list[tuple[_AZNode, int, int]] = []
        s = state
        key = root_key
        while True:
            node = self._nodes.get(key)
            if node is None or game.is_terminal(s):
                if game.is_terminal(s):
                    leaf_player = game.current_player(s)
                    v = game.returns(s)[leaf_player]
                else:
                    leaf_player = game.current_player(s)
                    v = self._expand(game, s, key, rng, root=False).value
                for n, a, mover in path:
                    n.update(a, v if mover == leaf_player else -v)
                return
            mover = game.current_player(s)
            action = node.select(self.c_puct)
            path.append((node, action, mover))
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
    distilled = distill_examples(game, distill_positions, distill_min_moves, seed, device) if distill_positions > 0 else []
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
        buffer = (buffer + fresh)[-buffer_cap:]
        loss = train_net(net, buffer + distilled, epochs, batch_size, lr, device)
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
    return AlphaZeroAgent(
        net,
        sims=int(cfg.get("az_sims", cfg.get("mcts_sims", 100))),
        device=device,
        solve_endgame=int(cfg.get("az_solve_endgame", 0)),
    )
