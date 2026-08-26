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


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    """A head: a bare Linear (hidden ≤ 0, the legacy readout) or a Linear→ReLU→Linear tower — the nonlinearity a
    single affine map lacks, needed to represent forks (AND-of-threats) and odd/even threat-parity."""
    if hidden <= 0:
        return nn.Linear(in_dim, out_dim)
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class _ResBlock(nn.Module):
    """A pre-activation-free residual block (conv-[bn]-relu-conv-[bn] + skip) — BatchNorm+residual are what make a
    DEEP tower trainable; adding depth to the plain legacy stack without them silently fails to train."""

    def __init__(self, filters: int, batchnorm: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters) if batchnorm else None
        self.bn2 = nn.BatchNorm2d(filters) if batchnorm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.bn1(h) if self.bn1 is not None else h
        h = F.relu(h)
        h = self.conv2(h)
        h = self.bn2(h) if self.bn2 is not None else h
        return F.relu(h + x)


class Connect4Net(nn.Module):
    """Policy+value net over a 2-plane board (own / opponent). Config-driven capacity (§C.7): the default is the
    legacy 2-conv/bare-linear-head net; `residual=True` builds a ResNet tower with head towers to the ~1.5-2M-param
    strong-C4 floor. Architecture is persisted with the weights (see save_net/load_net) so any net round-trips."""

    def __init__(self, channels: int = 32, blocks: int = 0, residual: bool = False,
                 batchnorm: bool = False, head_hidden: int = 0, input_planes: int = 2):
        super().__init__()
        # The architecture is a CONFIG, persisted with the weights so any net round-trips. The DEFAULT reproduces
        # the legacy 2-conv/bare-linear-head net EXACTLY (same module names) so the 306 old checkpoints still load;
        # `residual=True` builds the deep tower (§C.7: the capacity the 20K-param toy lacked). See save_net/load_net.
        self.arch = {"channels": int(channels), "blocks": int(blocks), "residual": bool(residual),
                     "batchnorm": bool(batchnorm), "head_hidden": int(head_hidden), "input_planes": int(input_planes)}
        self.residual = bool(residual)
        if not self.residual:
            self.conv1 = nn.Conv2d(input_planes, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.policy_head = nn.Linear(channels * ROWS * COLS, COLS)
            self.value_head = nn.Linear(channels * ROWS * COLS, 1)
            return
        # SCALED: stem → residual tower → policy/value HEAD TOWERS (a hidden layer, not a bare linear — the
        # dominant nonlinearity gap that let forks / odd-even parity be represented).
        self.stem = nn.Conv2d(input_planes, channels, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels) if batchnorm else None
        self.blocks = nn.ModuleList([_ResBlock(channels, batchnorm) for _ in range(max(1, blocks))])
        self.p_conv = nn.Conv2d(channels, 2, 1)
        self.p_bn = nn.BatchNorm2d(2) if batchnorm else None
        self.policy_head = _mlp(2 * ROWS * COLS, head_hidden, COLS)
        self.v_conv = nn.Conv2d(channels, 1, 1)
        self.v_bn = nn.BatchNorm2d(1) if batchnorm else None
        self.value_head = _mlp(1 * ROWS * COLS, max(1, head_hidden), 1)  # value ALWAYS gets a hidden layer

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.residual:
            h = F.relu(self.conv1(x))
            h = F.relu(self.conv2(h))
            h = h.flatten(1)
            return self.policy_head(h), torch.tanh(self.value_head(h))
        h = self.stem(x)
        h = self.stem_bn(h) if self.stem_bn is not None else h
        h = F.relu(h)
        for b in self.blocks:
            h = b(h)
        p = self.p_conv(h)
        p = self.p_bn(p) if self.p_bn is not None else p
        p = F.relu(p).flatten(1)
        v = self.v_conv(h)
        v = self.v_bn(v) if self.v_bn is not None else v
        v = F.relu(v).flatten(1)
        return self.policy_head(p), torch.tanh(self.value_head(v))


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
    """The net's value-head estimate for `state`, from the SIDE-TO-MOVE's perspective (+1 = the mover wins). NOTE
    (§C.7): this is an ON-POLICY value — it is CAPPED at the current policy's actual win margin, not the
    game-theoretic value. On a first-player-win opening it only climbs toward +1 as the policy learns to CONVERT
    the win; a value near 0 mid-training is partly EXPECTED (equal-strength self-play scores the opening ~50/50),
    NOT proof of a bug on its own. Read it as a strength-dependent progress signal, not a fixed +1 truth."""
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


def completed_q_values(
    prior: dict[int, float],
    child_n: dict[int, int],
    child_w: dict[int, float],
    root_value: float,
    legal: list[int],
) -> tuple[dict[int, float], int, int]:
    """COMPLETED Q-values (Danihelka 2022) — every legal action gets a Q, even the ones the search never visited.
    A visited action keeps its own mean Q (`child_w/child_n`); an UNVISITED action is completed with `v_mix`, the
    root value blended with the visited children's prior-weighted mean Q. This is what makes a low-sim policy
    target sane: with only a handful of visits over a wide root, the raw visit counts have no improvement
    guarantee, but the completed-Q set does. Returns `(q, sum_n, max_n)` (the visit totals feed σ's scale)."""
    sum_n = sum(child_n[a] for a in legal)
    max_n = max((child_n[a] for a in legal), default=0)
    visited = [a for a in legal if child_n[a] > 0]
    if visited and sum_n > 0:
        sum_pi_visited = sum(prior[a] for a in visited) or 1e-12
        weighted_q = sum(prior[a] * (child_w[a] / child_n[a]) for a in visited)
        v_mix = (root_value + sum_n * (weighted_q / sum_pi_visited)) / (1 + sum_n)
    else:
        v_mix = root_value
    q = {a: (child_w[a] / child_n[a] if child_n[a] > 0 else v_mix) for a in legal}
    return q, sum_n, max_n


V_MIN, V_MAX = -1.0, 1.0  # the game's value bounds (loss/win); a tanh value head + {-1,0,1} outcomes live here


def _norm_q(q_val: float) -> float:
    """Normalise a completed-Q value to [0,1] against the FIXED value range [V_MIN, V_MAX] — NOT a per-root
    min-max. Softmax is shift-invariant, so σ then acts on the CARDINAL Q gap (q_a − q_b)/2: near-tied actions
    stay near-uniform, a genuine win/loss gap peaks. Per-root min-max instead stretches any nonzero gap to the
    full [0,1] range, manufacturing a near-one-hot target from low-sim search NOISE — the measured cause of the
    completed-Q-trained net regressing. Fixed-range is the actual generic/chess-safe choice (MuZero known bounds)."""
    return min(1.0, max(0.0, (q_val - V_MIN) / (V_MAX - V_MIN)))


def completed_q_policy(
    prior: dict[int, float],
    child_n: dict[int, int],
    child_w: dict[int, float],
    root_value: float,
    legal: list[int],
    c_visit: float = 50.0,
    c_scale: float = 1.0,
) -> dict[int, float]:
    """The IMPROVED policy target: `softmax(logits + σ(completedQ))` over legal actions, where `logits = log π`
    (the net prior) and `σ(q̂) = (c_visit + maxN)·c_scale·q̂` with the completed-Q values normalised to [0,1]
    against the FIXED value range (`_norm_q`), so σ scales with the true Q magnitude. Unlike raw visit fractions
    this is a GUARANTEED policy improvement over the prior even at 2–32 sims, so it gives the net a corrective
    gradient the visit-count target cannot — the fix for the structural low-sim plateau."""
    q, _sum_n, max_n = completed_q_values(prior, child_n, child_w, root_value, legal)
    scale = (c_visit + max_n) * c_scale
    logits = {a: math.log(max(prior[a], 1e-12)) + scale * _norm_q(q[a]) for a in legal}
    mx = max(logits.values())
    exps = {a: math.exp(logits[a] - mx) for a in legal}
    z = sum(exps.values()) or 1.0
    return {a: exps[a] / z for a in legal}


def _sample_gumbel(rng: random.Random) -> float:
    """One Gumbel(0) sample: −log(−log U), U ~ Uniform(0,1). Added to the root logits it turns argmax into a
    draw ∝ the prior — the exploration Gumbel AlphaZero uses at the root INSTEAD of Dirichlet noise."""
    return -math.log(-math.log(rng.random() + 1e-12) + 1e-12)


class AlphaZeroAgent:
    """Net-guided MCTS. `sims` PUCT simulations per move; leaves scored by the value head (no random rollout).
    Holds a transposition table across its moves in a game, like `mcts`. `temperature`/`add_noise` are the
    self-play EXPLORATION knobs (0 temperature + no noise = greedy play, used for evaluation).

    With `gumbel=True` the ROOT uses Gumbel action-selection + Sequential Halving and returns the completed-Q
    improved policy (Danihelka 2022) instead of raw visit fractions — a guaranteed per-move policy improvement
    that holds at low sims, so the same strength needs far fewer simulations. The interior stays PUCT (it only
    supplies the Q estimates the root completes over). `gumbel_m` = how many root actions Sequential Halving
    considers; `c_visit`/`c_scale` set σ's scale."""

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
        gumbel: bool = False,
        gumbel_m: int = 16,
        c_visit: float = 50.0,
        c_scale: float = 0.1,
    ):
        self.net = net
        self.sims = max(1, int(sims))
        self.c_puct = c_puct
        self.device = device
        self.temperature = temperature
        self.add_noise = add_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_frac = noise_frac
        # GUMBEL root (opt-in): Sequential Halving + completed-Q policy target. `gumbel_m` root actions considered;
        # `_gumbel_selected` records the action Sequential Halving chose (the move to PLAY), separate from the
        # completed-Q TARGET run_search returns. See `completed_q_policy` / `_gumbel_search`.
        self.gumbel = bool(gumbel)
        self.gumbel_m = max(2, int(gumbel_m))
        self.c_visit = float(c_visit)
        self.c_scale = float(c_scale)
        self._gumbel_selected: int | None = None
        # Opt-in PROOF LEAVES: a `book` (proven values) makes the search back up the EXACT outcome at a booked or
        # endgame-solvable leaf instead of the value HEAD's estimate — truth propagates through the tree, so book
        # coverage pays off in play and self-play value targets get exact where a proof exists. None = pure net.
        self.book = book
        # Opt-in EXACT-ENDGAME cutoff (empty-cell threshold): once the position is cheap to solve, play a
        # provably-optimal move instead of the net-guided search — a perfect endgame the value head needn't
        # approximate, driving loss toward 0 / optimality toward 1. 0 = pure net-guided MCTS (self-play default).
        self.solve_endgame = int(solve_endgame)
        # §C.7 #2 amortisation gauge: endgame_solves = positions the agent solved from scratch (and wrote through
        # into `book`); endgame_hits = solves it AVOIDED via a memo lookup. Reported so the added self-solving is
        # shown to have stayed bounded (NOT a vs-#1 speedup number — pure #1 solves nothing to amortise).
        self.endgame_solves = 0
        self.endgame_hits = 0
        self.sims_used = 0
        self._nodes: dict[object, _AZNode] = {}
        # SELECTION half of MCTS-Solver (see agents.prove_node): proofs seeded at leaves PROPAGATE up so the root
        # can become a genuine PROOF. Consumed in GREEDY deployment only — self-play keeps its visit-count policy
        # untouched (propagation writes this overlay but never changes descent/backup). Inert without a proof source.
        self._proven: dict[object, float] = {}
        self._solving = book is not None or self.solve_endgame > 0

    def _proven_value(self, game: Game, state: State) -> float | None:
        """The EXACT value to the side-to-move if this position is proven (booked) or cheap to solve
        (≤ solve_endgame), else None. Lets a search leaf collapse onto ground truth instead of the value head.
        WRITE-THROUGH: when a book is present, a fresh cheap solve is recorded (value-only) so the next visit is a
        free lookup — each endgame is solved ONCE per run, the amortisation the online loop is built on."""
        if self.book is not None:
            from harness.book import book_value

            bv = book_value(self.book, game, state)
            if bv is not None:
                self.endgame_hits += 1
                return float(bv)
        if self.solve_endgame > 0:
            solve = getattr(game, "exact_optimal_actions", None)
            if solve is not None and solve(state, self.solve_endgame) is not None:
                from harness.book import _empties, _key, _ply, position_value

                v = float(position_value(game, state, book=self.book))  # reads booked children when a book is present
                if self.book is not None and getattr(game, "canonical_key", None) is not None:
                    emp = _empties(state)  # more empties = harder to recompute = higher keep-priority (see Tablebase)
                    self.book.put_proven(_key(game, state), int(v), best_actions=0,
                                         priority=emp if emp is not None else _ply(game, state))
                self.endgame_solves += 1
                return v
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

    def _simulate(
        self, game: Game, state: State, root_key: object, rng: random.Random, first_action: int | None = None
    ) -> None:
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
            # Gumbel Sequential Halving forces the ROOT's first descent to a chosen action; the interior stays PUCT.
            action = first_action if (first_action is not None and not path) else node.select(self.c_puct)
            path.append((node, action, mover, s))
            s = game.step(s, action, rng)
            key = state_key(game, s)

    def run_search(self, game: Game, state: State, rng: random.Random) -> dict[int, float]:
        """Run the searches and return the policy π over actions (the self-play training target). Default = the
        raw visit-count policy; `gumbel=True` = the completed-Q improved policy from a Sequential-Halving root."""
        if self.gumbel:
            return self._gumbel_search(game, state, rng)
        root_key = state_key(game, state)
        if root_key not in self._nodes:
            self._expand(game, state, root_key, rng, root=True)
        for _ in range(self.sims):
            self.sims_used += 1
            self._simulate(game, state, root_key, rng)
        root = self._nodes[root_key]
        total = sum(root.child_n.values()) or 1
        return {a: root.child_n[a] / total for a in root.legal}

    def _gumbel_scores(
        self, root: _AZNode, logits: dict[int, float], gumbel: dict[int, float]
    ) -> dict[int, float]:
        """The Sequential-Halving ranking score g(a) + logit(a) + σ(completedQ(a)) — the SAME σ(completedQ) the
        returned policy target uses, so the action Sequential Halving keeps and the target's argmax agree."""
        q, _sum_n, max_n = completed_q_values(root.prior, root.child_n, root.child_w, root.value, root.legal)
        scale = (self.c_visit + max_n) * self.c_scale
        return {a: gumbel[a] + logits[a] + scale * _norm_q(q[a]) for a in root.legal}

    def _gumbel_search(self, game: Game, state: State, rng: random.Random) -> dict[int, float]:
        """Gumbel AlphaZero root: sample Gumbel noise on the prior logits, take the top `gumbel_m` actions, then
        Sequential Halving — repeatedly give the survivors equal visits and drop the worse half by the g+logit+σ(Q)
        score — until one remains (recorded as `_gumbel_selected`, the move to play). Returns the completed-Q
        improved policy over ALL legal actions as the training target. Total simulations ≤ `sims` (an honest
        budget, comparable to a raw n-sim search)."""
        root_key = state_key(game, state)
        if root_key not in self._nodes:
            self._expand(game, state, root_key, rng, root=True)
        root = self._nodes[root_key]
        legal = list(root.legal)
        if len(legal) == 1:
            self.sims_used += 1
            self._gumbel_selected = legal[0]
            return {legal[0]: 1.0}
        logits = {a: math.log(max(root.prior[a], 1e-12)) for a in legal}
        # Gumbel noise is the SELF-PLAY exploration device; at greedy eval (temperature 0) it is OFF, so greedy
        # deployment is deterministic and doesn't weaken play by scattering the few root visits.
        explore = self.temperature > 1e-6
        gumbel = {a: (_sample_gumbel(rng) if explore else 0.0) for a in legal}
        m = min(self.gumbel_m, len(legal))
        considered = sorted(legal, key=lambda a: gumbel[a] + logits[a], reverse=True)[:m]
        budget = self.sims
        remaining = list(considered)
        while budget > 0 and len(remaining) > 1:
            phases_left = max(1, math.ceil(math.log2(len(remaining))))
            phase_budget = budget if phases_left == 1 else max(len(remaining), budget // phases_left)
            per = max(1, min(phase_budget, budget) // len(remaining))
            for a in remaining:
                for _ in range(per):
                    if budget <= 0:
                        break
                    self.sims_used += 1
                    self._simulate(game, state, root_key, rng, first_action=a)
                    budget -= 1
            scores = self._gumbel_scores(root, logits, gumbel)
            remaining.sort(key=lambda a: scores[a], reverse=True)
            remaining = remaining[: max(1, len(remaining) // 2)]
        while budget > 0:  # any rounding remainder refines the surviving action (keeps total sims ≈ budget)
            self.sims_used += 1
            self._simulate(game, state, root_key, rng, first_action=remaining[0])
            budget -= 1
        scores = self._gumbel_scores(root, logits, gumbel)
        self._gumbel_selected = remaining[0] if len(remaining) == 1 else max(considered, key=lambda a: scores[a])
        return completed_q_policy(
            root.prior, root.child_n, root.child_w, root.value, legal, self.c_visit, self.c_scale
        )

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


def _root_search_value(agent: "AlphaZeroAgent", game: Game, state: State) -> float:
    """The SEARCH-improved root value (mover-relative): the visit-weighted mean of the root children's Q after the
    search, i.e. the backed-up root value — a stronger estimate than the raw net value, and what Reanalyze uses to
    refresh a stored position's value target with the current net."""
    node = agent._nodes.get(state_key(game, state))
    if node is None:
        return 0.0
    total = sum(node.child_n[a] for a in node.legal)
    if total == 0:
        return float(node.value)
    return sum(node.child_w[a] for a in node.legal) / total


def reanalyze_examples(
    game: Game, agent: "AlphaZeroAgent", states: list[State], rng: random.Random
) -> list[tuple[torch.Tensor, list[float], float]]:
    """MuZero REANALYZE (#2) — re-label stored positions with the CURRENT net for ~free data efficiency. For each
    stored `state`, re-run the current agent's GREEDY search (temperature 0 → the improved policy target, no
    exploration noise) to regenerate a fresh policy target AND the search-improved value. Old buffer entries were
    labelled by a weaker past net; refreshing them with the current, stronger net de-stales the targets without any
    new self-play. Returns `(encoded, pi_vec, value)` ready for the training buffer."""
    agent.temperature = 0.0
    out: list[tuple[torch.Tensor, list[float], float]] = []
    for state in states:
        if game.is_terminal(state):
            continue
        agent._nodes = {}
        pi = agent.run_search(game, state, rng)
        pi_vec = [pi.get(a, 0.0) for a in range(game.num_actions)]
        value = max(-1.0, min(1.0, _root_search_value(agent, game, state)))
        out.append((encode(game, state), pi_vec, value))
    return out


def n_step_value_targets(vt: list[float], outcome_for: list[float], n: int) -> list[float]:
    """The n-step / TD value target (MuZero) — the fix for opening value-label CONTAMINATION. The raw-MC target
    labels every position with the FINAL game outcome, so an opening gets blamed for a blunder 20 plies later. The
    n-step target instead bootstraps from the LAGGED target-net's value `n` plies ahead (`vt[i+n]`, sign-corrected
    to mover-i: n even → same mover +1, n odd → opponent −1), falling back to the real terminal `outcome_for[i]`
    only when the terminal is within n plies. Large n → mostly real outcome (low bias); small n → mostly bootstrap
    (low variance, but needs a decent target net). `n ≥ trajectory length` reproduces the pure-MC target exactly."""
    length = len(vt)
    sign = 1.0 if n % 2 == 0 else -1.0
    return [outcome_for[i] if i + n >= length else sign * vt[i + n] for i in range(length)]


def _value_batch(net: "Connect4Net", xs: list[torch.Tensor], device: str = "cpu") -> list[float]:
    """The value head over a batch of already-encoded positions (mover-relative), for the lagged target net."""
    if not xs:
        return []
    net.eval()
    with torch.no_grad():
        _logits, value = net(torch.stack(xs).to(device))
    return [float(value[i, 0]) for i in range(len(xs))]


def self_play_game(
    game: Game, agent: AlphaZeroAgent, rng: random.Random, temp_moves: int = 8,
    target_net: "Connect4Net | None" = None, n_step: int = 0, device: str = "cpu",
    return_states: bool = False, opening_plies: int = 0,
    endgame_tb=None, exact_value_targets: bool = False,
) -> list:
    """Play ONE self-play game and return training examples (encoded board, policy, value). With `return_states`,
    each example is prefixed with the game STATE `(state, x, pi, v)` so the buffer can be REANALYZED (#2) — the
    stored state is what lets the current net re-search and re-label the position later. `opening_plies` > 0 plays
    that many RANDOM opening moves before net-guided play begins (those plies are NOT recorded as training
    examples) — so the net TRAINS on positions reached from DIVERSE openings, not just its own main line. This is
    the robustness lever: a net trained only on its canonical line loses AWAY from it (measured); off-line coverage
    teaches it to never lose a drawable position. Generic (no game knowledge)."""
    agent._nodes = {}
    agent.add_noise = not agent.gumbel  # Gumbel supplies its own root exploration; Dirichlet would double it
    pending: list[tuple[State, torch.Tensor, list[float], int]] = []
    state = game.initial_state(rng)
    for _ in range(opening_plies):  # DIVERSE random opening (unrecorded) → off-main-line training coverage
        if game.is_terminal(state):
            break
        state = game.step(state, rng.choice(game.legal_actions(state)), rng)
    move = 0
    while not game.is_terminal(state):
        agent.temperature = 1.0 if move < temp_moves else 0.0
        pi = agent.run_search(game, state, rng)
        player = game.current_player(state)
        pi_vec = [pi.get(a, 0.0) for a in range(game.num_actions)]
        pending.append((state, encode(game, state), pi_vec, player))
        # Gumbel PLAYS the Sequential-Halving winner (exploration already baked into the Gumbel noise); the raw
        # loop samples the visit-count policy at the temperature schedule.
        action = agent._gumbel_selected if agent.gumbel else sample_action(pi, agent.temperature, rng)
        state = game.step(state, action, rng)
        move += 1
    returns = game.returns(state)
    outcome_for = [returns[player] for (_s, _x, _pi, player) in pending]
    if n_step > 0 and target_net is not None:
        vt = _value_batch(target_net, [x for (_s, x, _pi, _p) in pending], device)  # lagged target-net bootstrap
        values = n_step_value_targets(vt, outcome_for, n_step)
    else:
        values = outcome_for  # raw-MC outcome (default / unchanged)
    if exact_value_targets and endgame_tb is not None:
        # EXACT-TARGET override: where the endgame tablebase PROVES a position, its game-theoretic value REPLACES
        # the (noisy MC / bootstrap) value target — mover-relative direct-assign, matching outcome_for's frame.
        from harness.book import _key

        values = [
            float(pv) if (pv := endgame_tb.proven_value(_key(game, s))) is not None else v
            for (s, _x, _pi, _p), v in zip(pending, values)
        ]
    if return_states:
        return [(s, x, pi_vec, v) for (s, x, pi_vec, _p), v in zip(pending, values)]
    return [(x, pi_vec, v) for (_s, x, pi_vec, _p), v in zip(pending, values)]


def extend_endgame_frontier(game: Game, run_tb, states, max_empty: int, max_positions: int,
                            deadline_seconds: float) -> int:
    """RETROGRADE frontier climb: try to PROVE each of `states` and write the result (VALUE-ONLY) into `run_tb`,
    processing DEEPEST-first so a just-proven child lets its shallower parent prove in the SAME pass. Uses only
    book._prove (a winning/terminal child, free minimax once every child is proven, or a ≤ `max_empty` cheap solve
    — all inherently bounded), never an unbounded full solve. Budgeted by `max_positions` proofs and a wall-clock
    `deadline_seconds` checked BETWEEN positions (thread-safe — no SIGALRM). Returns the number of positions proven.
    This is what marches the proven frontier opening-ward across iterations."""
    import time

    from harness.book import _empties, _key, _ply, _prove

    if max_positions <= 0:
        return 0
    seen: dict[int, State] = {}
    for s in states:
        if game.is_terminal(s):
            continue
        k = _key(game, s)
        if k not in seen and run_tb.proven_value(k) is None:  # skip already-proven (no wasted re-prove)
            seen[k] = s
    ordered = sorted(seen.values(), key=lambda s: _ply(game, s), reverse=True)  # deepest first → children before parents
    deadline = time.monotonic() + deadline_seconds if deadline_seconds > 0 else None
    proven = 0
    for s in ordered:
        if proven >= max_positions or (deadline is not None and time.monotonic() > deadline):
            break
        res = _prove(game, s, run_tb, max_empty)
        if res is not None:
            emp = _empties(s)  # more empties = harder to recompute = higher keep-priority
            run_tb.put_proven(_key(game, s), int(res[0]), best_actions=0,
                              priority=emp if emp is not None else _ply(game, s))
            proven += 1
    return proven


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
    learner.add_noise = not learner.gumbel  # Gumbel supplies its own root exploration; Dirichlet would double it
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
            action = learner._gumbel_selected if learner.gumbel else sample_action(pi, learner.temperature, rng)
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


# --- parallel self-play (§C.7 speedup) ------------------------------------------------------------------
# Self-play is sequential (one tiny 6x7 forward at a time), so a single process uses ~1 core and the rest sit
# idle. These play the per-iteration GAMES across worker processes to fill the idle cores. macOS uses 'spawn',
# so the worker fn + initializer are MODULE-LEVEL and args are picklable; the net is shared ONCE per iteration
# via a temp file (version = iteration) that each worker version-caches, never re-serialised per game.
_SELFPLAY_WORKER: dict = {}


def _selfplay_worker_init(game_name: str) -> None:
    import torch as _torch

    _torch.set_num_threads(1)  # tiny forwards don't use threads; 1/worker ⇒ W workers = W cores, no oversubscription
    from harness.registry import resolve_game

    _SELFPLAY_WORKER.clear()
    _SELFPLAY_WORKER["game"] = resolve_game(game_name)


def _selfplay_worker(task: tuple):
    (net_path, net_ver, tgt_path, tgt_ver, sims, gumbel, gumbel_m, c_scale, n_step, opening_plies,
     temp_moves, game_seed) = task
    st = _SELFPLAY_WORKER
    if st.get("net_ver") != net_ver:  # reload only when the iteration's weights changed
        st["net"] = load_net(net_path)
        st["net_ver"] = net_ver
    tgt = None
    if tgt_path is not None:
        if st.get("tgt_ver") != tgt_ver:
            st["tgt"] = load_net(tgt_path)
            st["tgt_ver"] = tgt_ver
        tgt = st["tgt"]
    agent = AlphaZeroAgent(st["net"], sims=sims, gumbel=gumbel, gumbel_m=gumbel_m, c_scale=c_scale)
    return self_play_game(st["game"], agent, random.Random(game_seed), temp_moves=temp_moves,
                          target_net=tgt, n_step=n_step, opening_plies=opening_plies)


def _run_parallel_selfplay(pool, tmpdir: str, net, target_net, version: int, n_games: int, sims: int,
                           gumbel: bool, gumbel_m: int, c_scale: float, n_step: int, opening_plies: int,
                           rng: random.Random, temp_moves: int = 8) -> list:
    """Play `n_games` self-play games across `pool`'s workers using the CURRENT net. Deterministic per (parent
    rng, n_games): the parent draws each game's seed, so the set of games is reproducible; workers never mutate
    shared state. Returns the flat (x, pi, v) examples (UN-augmented — the caller augments once, as sequentially)."""
    import os

    net_path = os.path.join(tmpdir, "net.pt")
    save_net(net, net_path)  # 7MB, written ONCE per iteration (pool.map is synchronous ⇒ no read/write race)
    tgt_path = None
    if target_net is not None:
        tgt_path = os.path.join(tmpdir, "target.pt")
        save_net(target_net, tgt_path)
    tasks = [(net_path, version, tgt_path, version, sims, gumbel, gumbel_m, c_scale, n_step, opening_plies,
              temp_moves, rng.randrange(2**31)) for _ in range(n_games)]
    out: list = []
    for game_examples in pool.map(_selfplay_worker, tasks):
        out.extend(game_examples)
    return out


def _endgame_enabled(game: Game, endgame_tb) -> bool:
    """§C.7 #2 GENERIC GATE: the online endgame loop runs ONLY when a run tablebase is present AND the game exposes
    the exact hooks (canonical_key + exact_optimal_actions). Absent either, the caller builds the learner with
    book=None/solve_endgame=0 — byte-identical to pure #1 (chess opening / Go degrade cleanly, never crash)."""
    return (
        endgame_tb is not None
        and getattr(game, "canonical_key", None) is not None
        and getattr(game, "exact_optimal_actions", None) is not None
    )


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
    gumbel: bool = False,
    gumbel_m: int = 16,
    c_scale: float = 0.1,
    value_n_step: int = 0,
    target_refresh: int = 4,
    reanalyze_frac: float = 0.0,
    selfplay_opening_plies: int = 0,
    endgame_tb=None,
    endgame_max_empty: int = 14,
    endgame_exact_targets: int = 1,
    endgame_extend_positions: int = 2000,
    endgame_extend_seconds: float = 5.0,
    net_arch: dict | None = None,
    init_buffer: list | None = None,
    return_buffer: bool = False,
    selfplay_workers: int = 1,
    log: Callable[[str], None] | None = None,
):
    """The AlphaZero loop with WARM-START + LEAGUE + optional ORACLE DISTILLATION. Starts from `init_net` (the
    champion) when given instead of a random net — so training compounds across runs rather than restarting
    from zero. When `distill_positions > 0` it first imprints the net on oracle-labelled optimal play and keeps
    those examples in EVERY training pass (a persistent 'this is the perfect move' anchor). Each iteration then
    mixes pure self-play with games against the `opponent_pool` (strong mcts / heuristic / near-perfect oracle /
    past champions) at rate `pool_frac`, and trains on the accumulated buffer + the distilled anchor."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    # net_arch (§C.7 capacity levers) overrides the legacy `channels`-only shape; init_net (warm start / batch resume)
    # wins over both so a resumed run keeps its architecture.
    net = init_net if init_net is not None else Connect4Net(**(net_arch or {"channels": channels})).to(device)
    # init_buffer/return_buffer (§C.7 batched training): carry the replay buffer ACROSS batches so a resumed run is
    # equivalent to a continuous one — a big net starved of history relearns from scratch each batch. (Non-reanalyze path.)
    buffer: list[tuple[torch.Tensor, list[float], float]] = list(init_buffer) if init_buffer else []
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
    # LAGGED TARGET NET for the n-step value target (#3): a frozen copy refreshed every `target_refresh` iters, so
    # self-play VALUE labels bootstrap off a STABLE net instead of chasing the live weights (and off the target
    # net's mid-game read n plies ahead instead of the noisy final outcome — the opening-contamination fix).
    import copy

    target_net = copy.deepcopy(net) if value_n_step > 0 else None
    # REANALYZE (#2) holds STATES in the buffer so old entries can be re-labelled by the current net. The training
    # set is always the (x, pi, v) view; the state is carried only to re-search. `state_buffer` mirrors `buffer`.
    state_buffer: list[tuple[State, torch.Tensor, list[float], float]] = []
    # §C.7 #2: the online endgame loop is armed only for a game with the exact hooks — else pure #1 (no crash).
    endgame_on = _endgame_enabled(game, endgame_tb)
    eg_targets = endgame_on and bool(endgame_exact_targets)
    # §C.7 PARALLEL self-play: only the PURE-#1 path is safe to fan out (no shared endgame tablebase, no league
    # opponent, no reanalyze state-buffer). Otherwise stay sequential (byte-identical). Needs a game name to respawn.
    parallel_ok = (int(selfplay_workers) > 1 and opponent_pool is None and reanalyze_frac == 0.0
                   and not endgame_on and getattr(game, "name", None) is not None)
    _pool = _tmpdir = None
    if parallel_ok:
        import multiprocessing as _mp
        import os as _os
        import tempfile

        _tmpdir = tempfile.mkdtemp(prefix="az_sp_")  # UNIQUE per process ⇒ the 3 concurrent seeds never collide
        # THREAD DECOUPLING (macOS Accelerate follows OMP_NUM_THREADS, NOT torch.set_num_threads): spawn the
        # self-play workers with 1 BLAS thread each — measured 2.46x self-play speedup, vs a 27-thread thrash at
        # OMP=3 — while the PARENT keeps its OMP threads for TRAINING (its BLAS is already initialised). spawn
        # children inherit os.environ AT SPAWN TIME, so set it around Pool() only, then restore.
        _saved = {k: _os.environ.get(k) for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}
        for k in _saved:
            _os.environ[k] = "1"
        _pool = _mp.get_context("spawn").Pool(int(selfplay_workers), initializer=_selfplay_worker_init,
                                              initargs=(game.name,))  # daemonic workers ⇒ die with the parent
        for k, v in _saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
    for it in range(iterations):
        if value_n_step > 0 and it > 0 and it % max(1, target_refresh) == 0:
            target_net = copy.deepcopy(net)  # refresh the lag every k iters
        # When armed, the learner carries the run tablebase as its proof book + a cheap-endgame cutoff, so search
        # backs up EXACT endgame values AND records/memoises each solve (write-through) as it plays.
        learner = AlphaZeroAgent(net, sims=sims, device=device, gumbel=gumbel, gumbel_m=gumbel_m, c_scale=c_scale,
                                 book=(endgame_tb if endgame_on else None),
                                 solve_endgame=(endgame_max_empty if endgame_on else 0))
        vs_pool = 0
        reanalyzed = 0
        eg_visited: list[State] = []
        if reanalyze_frac > 0.0:
            fresh_s: list[tuple[State, torch.Tensor, list[float], float]] = []
            for _ in range(selfplay_games):
                fresh_s.extend(self_play_game(game, learner, rng, target_net=target_net, n_step=value_n_step,
                                              device=device, return_states=True, opening_plies=selfplay_opening_plies,
                                              endgame_tb=(endgame_tb if endgame_on else None),
                                              exact_value_targets=eg_targets))
            if endgame_on:
                eg_visited.extend(s for (s, *_rest) in fresh_s)
            state_buffer = (state_buffer + fresh_s)[-buffer_cap:]
            # Re-label a random sample of the buffer with the CURRENT net (fresh policy + search-improved value).
            k = int(reanalyze_frac * len(state_buffer))
            if it > 0 and k > 0:
                idxs = rng.sample(range(len(state_buffer)), k)
                relabelled = reanalyze_examples(game, learner, [state_buffer[i][0] for i in idxs], rng)
                # POLICY-ONLY refresh: replace the POLICY target with the current net's, but KEEP the stored n-step
                # VALUE target — measured: overwriting the value with a search-root estimate DEGRADES the n-step
                # value (the #3 lever), so reanalyze must not touch it.
                for i, (x, pi, _v_search) in zip(idxs, relabelled):
                    state_buffer[i] = (state_buffer[i][0], x, pi, state_buffer[i][3])
                reanalyzed = len(relabelled)
            buffer = augment_examples([(x, pi, v) for (_s, x, pi, v) in state_buffer], perms)
        elif parallel_ok:  # PURE-#1 fanned out across worker processes (fills the idle cores; ~2-2.5x faster)
            fresh = _run_parallel_selfplay(_pool, _tmpdir, net, target_net, it, selfplay_games, sims, gumbel,
                                           gumbel_m, c_scale, value_n_step, selfplay_opening_plies, rng)
            fresh = augment_examples(fresh, perms)
            buffer = (buffer + fresh)[-buffer_cap:]
        else:
            fresh: list[tuple[torch.Tensor, list[float], float]] = []
            for _ in range(selfplay_games):
                if opponent_pool and rng.random() < pool_frac:
                    opponent = opponent_pool[rng.randrange(len(opponent_pool))]()
                    fresh.extend(vs_opponent_game(game, learner, opponent, rng.randint(0, 1), rng))
                    vs_pool += 1
                elif endgame_on:  # collect STATES for the frontier extension (return_states only changes the shape)
                    ex = self_play_game(game, learner, rng, target_net=target_net, n_step=value_n_step, device=device,
                                        opening_plies=selfplay_opening_plies, return_states=True,
                                        endgame_tb=endgame_tb, exact_value_targets=eg_targets)
                    eg_visited.extend(s for (s, *_rest) in ex)
                    fresh.extend((x, pi, v) for (_s, x, pi, v) in ex)
                else:  # unchanged pure-#1 path (byte-identical when the loop is off)
                    fresh.extend(self_play_game(game, learner, rng, target_net=target_net, n_step=value_n_step,
                                                device=device, opening_plies=selfplay_opening_plies))
            fresh = augment_examples(fresh, perms)  # symmetry-augment self-play too (2× data, invariance baked in)
            buffer = (buffer + fresh)[-buffer_cap:]
        eg_booked = 0
        if endgame_on:  # RETROGRADE climb: prove the iteration's frontier positions backward from booked terminals
            eg_booked = extend_endgame_frontier(game, endgame_tb, eg_visited, endgame_max_empty,
                                                endgame_extend_positions, endgame_extend_seconds)
        loss = train_net(net, _mix_training_set(buffer, distilled, distill_fraction), epochs, batch_size, lr, device)
        # CHEAP per-iteration quality probe (one forward pass): the net's value on the standard opening. Connect 4
        # is a first-player WIN, so this should climb toward +1 — the live curve the #2-vs-#1 A/B compares.
        opening_value = round(net_value(net, game, game.initial_state(random.Random(seed)), device), 4)
        entry = {"iteration": it + 1, "examples": len(buffer), "vs_pool_games": vs_pool, "reanalyzed": reanalyzed,
                 "buffer": len(buffer), "distilled": len(distilled), "loss": loss, "opening_value": opening_value}
        if endgame_on:
            entry.update({"endgame_booked": eg_booked, "endgame_solves": learner.endgame_solves,
                          "endgame_hits": learner.endgame_hits, "endgame_total": len(endgame_tb)})
        history.append(entry)
        if log:
            eg_note = (f", endgame +{eg_booked}/{len(endgame_tb)} (solve {learner.endgame_solves} hit {learner.endgame_hits})"
                       if endgame_on else "")
            log(f"iter {it + 1}/{iterations}: buffer {len(buffer)} ({vs_pool} vs-pool, {reanalyzed} reanalyzed), "
                f"distilled {len(distilled)}, loss {loss:.3f}{eg_note}")
    if _pool is not None:
        _pool.close()
        _pool.join()
        import shutil

        shutil.rmtree(_tmpdir, ignore_errors=True)
    if return_buffer:
        return net, history, buffer
    return net, history


def save_net(net: Connect4Net, path: str) -> None:
    # Persist the full architecture so any net (legacy or scaled) reconstructs exactly; `channels` kept for
    # backward-readability of the legacy field.
    torch.save({"state_dict": net.state_dict(), "arch": net.arch, "channels": net.arch["channels"]}, path)


def load_net(path: str, device: str = "cpu") -> Connect4Net:
    blob = torch.load(path, map_location=device)
    arch = blob.get("arch")  # absent ⇒ a pre-levers checkpoint ⇒ the legacy net keyed only by `channels`
    net = Connect4Net(**arch) if arch else Connect4Net(channels=int(blob.get("channels", 32)))
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
        # Deploy under the trained-for search operator (a gumbel-trained net is measurably weaker under plain PUCT).
        gumbel=bool(cfg.get("az_gumbel", False)),
        gumbel_m=int(cfg.get("az_gumbel_m", 16)),
        c_scale=float(cfg.get("az_c_scale", 0.1)),
    )
