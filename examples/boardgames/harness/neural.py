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
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.agents import Agent, _sign, child_move_value, prove_node, state_key
from harness.game import Game, State
from harness.strategy_tree import disagreements, raw_chooser, strategy_tree_positions

ROWS, COLS = 6, 7  # the DEFAULT board (Connect-4); a net carries its own board_shape/num_actions in its arch


def _board_shape(game: Game) -> tuple[int, int]:
    """A game's (rows, cols); games that predate `board_shape` are Connect-4-shaped."""
    h, w = getattr(game, "board_shape", (ROWS, COLS))
    return int(h), int(w)


def legacy_arch_as_built(arch: dict) -> dict:
    """The net an arch recorded BEFORE §C.46 actually built. Without `residual` the legacy branch ignored blocks,
    head_hidden, batchnorm and global_pool, so {"channels": 32, "blocks": 3, "head_hidden": 32} trained a
    12,746-parameter 2-conv net. Re-analysing old evidence must rebuild that net, which `arch_for_game` now refuses
    to build from the misleading dict; a residual arch is returned unchanged."""
    if arch.get("residual"):
        return dict(arch)
    return {k: v for k, v in arch.items() if k not in ("blocks", "head_hidden", "batchnorm", "global_pool")}


def arch_for_game(net_arch: dict | None, game: Game) -> dict:
    """A net arch with `board_shape`/`num_actions` taken FROM THE GAME when the config omits them — a config never
    hand-types 65 for Othello — and REFUSED when it states them differently: a net built for the wrong game is the
    §C.21 D1 defect, and it is caught here at construction, before a single game is played."""
    arch = dict(net_arch or {})
    if not arch.get("residual"):
        stray = [k for k in ("blocks", "head_hidden", "batchnorm", "global_pool") if arch.get(k)]
        if stray:
            raise ValueError(f"net_arch sets {stray} without residual=True — the legacy net would silently ignore them "
                             f"(§C.43-§C.45 trained a 12,746-parameter net believed to be 32/3/32)")
    h, w = _board_shape(game)
    n = getattr(game, "num_actions", None)
    if "board_shape" in arch and [int(v) for v in arch["board_shape"]] != [h, w]:
        raise ValueError(f"net_arch board_shape {list(arch['board_shape'])} but {game.name} is {[h, w]}")
    if "num_actions" in arch and n is not None and int(arch["num_actions"]) != int(n):
        raise ValueError(f"net_arch num_actions {arch['num_actions']} but {game.name} has {n}")
    planes = getattr(game, "input_planes", None)
    if planes is not None and "input_planes" in arch and int(arch["input_planes"]) != int(planes):
        raise ValueError(f"net_arch input_planes {arch['input_planes']} but {game.name} emits {planes}")
    arch.setdefault("board_shape", [h, w])
    if n is not None:
        arch.setdefault("num_actions", int(n))
    if planes is not None:
        arch.setdefault("input_planes", int(planes))
    # A game whose board has structurally dead cells (checkers' light squares) hands the net its mask, so the
    # masked pooling built in Increment 1 excludes them instead of averaging structural zeros into every feature.
    mask = getattr(game, "valid_mask", None)
    if mask is not None:
        arch.setdefault("valid_mask", list(mask))
    if arch.get("canonical_input") and "symmetries" not in arch and hasattr(game, "symmetries"):
        arch["symmetries"] = [[list(c), list(a)] for c, a in game.symmetries()]
    return arch


# --- the network -----------------------------------------------------------------------------------------


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    """A head: a bare Linear (hidden ≤ 0, the legacy readout) or a Linear→ReLU→Linear tower — the nonlinearity a
    single affine map lacks, needed to represent forks (AND-of-threats) and odd/even threat-parity."""
    if hidden <= 0:
        return nn.Linear(in_dim, out_dim)
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


def _masked_pool(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Per-channel (mean, max) over the board — over ONLY the valid cells when a mask is given. §C.21 D2: a padded
    board's pad cells must not reach the aggregate; an unmasked mean is diluted 2.7x for a 24-of-64 board and the
    max picks up whatever the pad emits. `mask` is None for a full board, which is the legacy path unchanged."""
    if mask is None:
        return torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
    n = mask.sum().clamp_min(1.0)
    return torch.cat([(h * mask).sum(dim=(2, 3)) / n,
                      h.masked_fill(mask == 0, float("-inf")).amax(dim=(2, 3))], dim=1)


class _ResBlock(nn.Module):
    """A pre-activation-free residual block (conv-[bn]-relu-conv-[bn] + skip) — BatchNorm+residual are what make a
    DEEP tower trainable; adding depth to the plain legacy stack without them silently fails to train. With
    `global_pool` (§C.8 #2, KataGo-style) the block also conditions mid-tower on WHOLE-BOARD aggregates
    (per-channel mean+max → FC → per-channel bias) — the aggregate a stack of 3×3 convs represents poorly but
    odd/even threat-parity (Allis zugzwang) needs."""

    def __init__(self, filters: int, batchnorm: bool, global_pool: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters) if batchnorm else None
        self.bn2 = nn.BatchNorm2d(filters) if batchnorm else None
        self.gpool_fc = nn.Linear(2 * filters, filters) if global_pool else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(x)
        h = self.bn1(h) if self.bn1 is not None else h
        h = F.relu(h)
        pooled = _masked_pool(h, mask) if self.gpool_fc is not None else None
        h = self.conv2(h)
        h = self.bn2(h) if self.bn2 is not None else h
        if pooled is not None:
            # Bias AFTER bn2 (before the skip-add): a per-channel-constant added before BatchNorm is exactly
            # cancelled by its mean subtraction (provably dead at batch size 1) — here nothing normalizes it away.
            h = h + self.gpool_fc(pooled)[:, :, None, None]
        return F.relu(h + x)


class Connect4Net(nn.Module):
    """Policy+value net over a 2-plane board (own / opponent). Config-driven capacity (§C.7): the default is the
    legacy 2-conv/bare-linear-head net; `residual=True` builds a ResNet tower with head towers to the ~1.5-2M-param
    strong-C4 floor. Architecture is persisted with the weights (see save_net/load_net) so any net round-trips."""

    def __init__(self, channels: int = 32, blocks: int = 0, residual: bool = False,
                 batchnorm: bool = False, head_hidden: int = 0, input_planes: int = 2,
                 global_pool: bool = False, value_bins: int = 0, aux_heads: bool = False,
                 board_shape=(ROWS, COLS), num_actions: int = COLS, valid_mask=None,
                 mlp_hidden: list[int] | None = None, canonical_input: bool = False, symmetries=None):
        super().__init__()
        if aux_heads and not residual:
            raise ValueError("aux_heads requires the residual tower (spatial trunk features)")
        if mlp_hidden is not None and (not mlp_hidden or any(int(h) < 1 for h in mlp_hidden) or residual or blocks
                                       or value_bins or aux_heads or global_pool or batchnorm or head_hidden):
            raise ValueError(f"mlp_hidden {mlp_hidden} needs positive widths and no conv-tower options — the MLP "
                             "body is a plain tower with linear policy and tanh value heads")
        if canonical_input and aux_heads:
            raise ValueError("canonical_input cannot serve aux heads — their per-cell targets are not mapped back")
        # §C.21 Increment 1: the board and the action count are part of the ARCH, not module constants, so one
        # net class serves every game; the defaults are Connect-4, so every existing checkpoint builds identically.
        self.board_h, self.board_w = int(board_shape[0]), int(board_shape[1])
        self.num_actions = int(num_actions)
        cells = self.board_h * self.board_w
        if valid_mask is not None and len(valid_mask) != cells:
            raise ValueError(f"valid_mask has {len(valid_mask)} cells for a {self.board_h}x{self.board_w} board")
        # The architecture is a CONFIG, persisted with the weights so any net round-trips. The DEFAULT reproduces
        # the legacy 2-conv/bare-linear-head net EXACTLY (same module names) so the 306 old checkpoints still load;
        # `residual=True` builds the deep tower (§C.7: the capacity the 20K-param toy lacked). See save_net/load_net.
        # §C.8 ceiling levers, both OPT-IN so every existing checkpoint/config is untouched: `global_pool` threads
        # whole-board aggregates through the tower; `value_bins=K>0` swaps the tanh scalar for a K-bin categorical
        # value head (consumers still see a scalar — forward returns the expectation over the support).
        self.arch = {"channels": int(channels), "blocks": int(blocks), "residual": bool(residual),
                     "batchnorm": bool(batchnorm), "head_hidden": int(head_hidden), "input_planes": int(input_planes),
                     "global_pool": bool(global_pool), "value_bins": int(value_bins), "aux_heads": bool(aux_heads),
                     "board_shape": [self.board_h, self.board_w], "num_actions": self.num_actions,
                     "valid_mask": None if valid_mask is None else [int(m) for m in valid_mask]}
        if mlp_hidden is not None:
            self.arch["mlp_hidden"] = [int(h) for h in mlp_hidden]
        if canonical_input:
            self._init_canonical(symmetries, input_planes, cells)
        if valid_mask is not None:  # only when a game pads its board — legacy checkpoints carry no such buffer
            self.register_buffer("valid_mask", torch.tensor([float(m) for m in valid_mask])
                                 .reshape(1, 1, self.board_h, self.board_w))
        self.residual = bool(residual)
        self.value_bins = int(value_bins)
        if self.value_bins > 0:
            self.register_buffer("value_support", torch.linspace(-1.0, 1.0, self.value_bins))
        v_out = self.value_bins if self.value_bins > 0 else 1
        self.mlp_body = None
        if mlp_hidden is not None:
            layers, width = [], int(input_planes) * cells
            for h in mlp_hidden:
                layers += [nn.Linear(width, int(h)), nn.ReLU()]
                width = int(h)
            self.mlp_body = nn.Sequential(*layers)
            self.policy_head = nn.Linear(width, self.num_actions)
            self.value_head = nn.Linear(width, v_out)
            return
        if not self.residual:
            self.conv1 = nn.Conv2d(input_planes, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.policy_head = nn.Linear(channels * cells, self.num_actions)
            self.value_head = nn.Linear(channels * cells, v_out)
            return
        # SCALED: stem → residual tower → policy/value HEAD TOWERS (a hidden layer, not a bare linear — the
        # dominant nonlinearity gap that let forks / odd-even parity be represented).
        self.stem = nn.Conv2d(input_planes, channels, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels) if batchnorm else None
        self.blocks = nn.ModuleList([_ResBlock(channels, batchnorm, global_pool) for _ in range(max(1, blocks))])
        self.p_conv = nn.Conv2d(channels, 2, 1)
        self.p_bn = nn.BatchNorm2d(2) if batchnorm else None
        self.policy_head = _mlp(2 * cells, head_hidden, self.num_actions)
        self.v_conv = nn.Conv2d(channels, 1, 1)
        self.v_bn = nn.BatchNorm2d(1) if batchnorm else None
        self.value_head = _mlp(1 * cells, max(1, head_hidden), v_out)  # value ALWAYS gets a hidden layer
        if aux_heads:  # §C.8 #4: ownership map (per-cell, from the spatial trunk) + opponent-reply logits
            self.own_head = nn.Conv2d(channels, 1, 1)
            self.reply_head = _mlp(2 * cells, head_hidden, self.num_actions)

    def _init_canonical(self, symmetries, planes: int, cells: int) -> None:
        """§C.49: the net takes every input in ONE standardised orientation — the image, under the game's verified
        symmetries, with the smallest fixed-weight key — and maps its policy back, so it is equivariant by
        construction and never has to learn an orientation twice."""
        perms = [(list(c), list(a)) for c, a in (symmetries or [])]
        if len(perms) < 2 or any(len(c) != cells or sorted(c) != list(range(cells)) or len(a) != self.num_actions
                                 or sorted(a) != list(range(self.num_actions)) for c, a in perms):
            raise ValueError("canonical_input needs the game's verified symmetries: at least two (cell_perm, "
                             f"action_perm) pairs over {cells} cells and {self.num_actions} actions")
        self.arch["canonical_input"] = True
        self.arch["symmetries"] = [[c, a] for c, a in perms]
        self.register_buffer("canon_cells", torch.tensor([c for c, _a in perms]), persistent=False)
        self.register_buffer("canon_actions", torch.tensor([a for _c, a in perms]), persistent=False)
        weights = torch.rand(planes * cells, generator=torch.Generator().manual_seed(20260927), dtype=torch.float64)
        self.register_buffer("canon_weights", weights, persistent=False)

    def canonical_images(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(the standardised image of each input, a [batch, symmetries] mask of the symmetries that produce it).
        Several symmetries produce it exactly when the position is symmetric."""
        b, planes = x.shape[0], x.shape[1]
        images = x.reshape(b, planes, -1)[:, :, self.canon_cells].permute(0, 2, 1, 3)
        keys = (images.reshape(b, images.shape[1], -1).double() * self.canon_weights).sum(-1)
        tied = keys == keys.min(dim=1, keepdim=True).values
        chosen = images[torch.arange(b), tied.int().argmax(dim=1)].reshape(x.shape)
        return chosen, tied

    def _map_back(self, logits: torch.Tensor, tied: torch.Tensor) -> torch.Tensor:
        """The raw-frame policy: the standardised-frame logits mapped back through each symmetry that produced the
        standardised image, averaged. Summed in sorted order, so a symmetric position gets bit-identical output
        whichever of its orientations came in."""
        b, g, n = logits.shape[0], self.canon_actions.shape[0], logits.shape[1]
        every = torch.zeros(b, g, n, dtype=logits.dtype, device=logits.device).scatter(
            2, self.canon_actions.unsqueeze(0).expand(b, g, n), logits.unsqueeze(1).expand(b, g, n))
        every = every.masked_fill(~tied.unsqueeze(2), float("inf")).sort(dim=1).values
        count = tied.sum(dim=1)
        total = every.masked_fill(torch.isinf(every), 0.0).cumsum(dim=1)[torch.arange(b), count - 1]
        return total / count.unsqueeze(1).to(logits.dtype)

    def _body(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Residual tower → (policy features, value features, SPATIAL trunk) — the aux heads read the trunk."""
        h = self.stem(x)
        h = self.stem_bn(h) if self.stem_bn is not None else h
        h = F.relu(h)
        mask = getattr(self, "valid_mask", None)
        for b in self.blocks:
            h = b(h, mask)
        p = self.p_conv(h)
        p = self.p_bn(p) if self.p_bn is not None else p
        p = F.relu(p).flatten(1)
        v = self.v_conv(h)
        v = self.v_bn(v) if self.v_bn is not None else v
        v = F.relu(v).flatten(1)
        return p, v, h

    def _trunk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared body → (policy logits, RAW value-head output: scalar pre-tanh, or K bin logits)."""
        if self.arch.get("canonical_input"):
            chosen, tied = self.canonical_images(x)
            logits, value = self._raw_trunk(chosen)
            return self._map_back(logits, tied), value
        return self._raw_trunk(x)

    def _raw_trunk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mlp_body is not None:
            h = self.mlp_body(x.flatten(1))
            return self.policy_head(h), self.value_head(h)
        if not self.residual:
            h = F.relu(self.conv1(x))
            h = F.relu(self.conv2(h))
            h = h.flatten(1)
            return self.policy_head(h), self.value_head(h)
        p, v, _h = self._body(x)
        return self.policy_head(p), self.value_head(v)

    def forward_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """TRAIN-path forward with the aux targets (§C.8 #4): (policy logits, value in train form — bin logits
        when categorical, tanh scalar otherwise — ownership map in [-1,1] per cell, opponent-reply logits)."""
        p_feat, v_feat, h = self._body(x)
        v_raw = self.value_head(v_feat)
        v_out = v_raw if self.value_bins > 0 else torch.tanh(v_raw)
        return (self.policy_head(p_feat), v_out,
                torch.tanh(self.own_head(h)).flatten(1), self.reply_head(p_feat))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, v_raw = self._trunk(x)
        if self.value_bins > 0:  # deployed value = EXPECTATION over the support — consumers still see a scalar
            return logits, (F.softmax(v_raw, dim=1) * self.value_support).sum(dim=1, keepdim=True)
        return logits, torch.tanh(v_raw)

    def forward_train(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """TRAIN-path forward: with bins, the value output is the K bin LOGITS (for cross-entropy); without,
        identical to forward — one path, no drift."""
        if self.value_bins > 0:
            return self._trunk(x)
        return self.forward(x)


def encode(game: Game, state: State) -> torch.Tensor:
    """Encode a position as a (2, rows, cols) tensor from the SIDE-TO-MOVE's perspective (plane 0 = own
    pieces, plane 1 = opponent), read from the game's own observation so the net is position-canonical."""
    player = game.current_player(state)
    h, w = _board_shape(game)
    obs = game.observation(state, player)
    planes = int(getattr(game, "input_planes", 2))
    if planes != 2:
        # A game with more than two piece kinds (checkers' men and kings) cannot be expressed as own/opponent,
        # so it declares `input_planes` and emits its own planes as the first `planes * h * w` observations.
        return torch.tensor(obs[: planes * h * w], dtype=torch.float32).reshape(planes, h, w)
    cells = obs[: h * w]
    own = [1.0 if v == 1.0 else 0.0 for v in cells]
    opp = [1.0 if v == -1.0 else 0.0 for v in cells]
    plane_own = torch.tensor(own, dtype=torch.float32).reshape(h, w)
    plane_opp = torch.tensor(opp, dtype=torch.float32).reshape(h, w)
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
        n = self.net.num_actions
        if getattr(game, "num_actions", n) != n:  # §C.21 D1: a net built for another game must fail loudly
            raise ValueError(f"net has num_actions={n} but {game.name} has {game.num_actions}")
        self.net.eval()
        with torch.no_grad():
            logits, value = self.net(encode(game, state).unsqueeze(0).to(self.device))
        legal = game.legal_actions(state)
        masked = torch.full((n,), -1e9)
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


def _root_q(agent: "AlphaZeroAgent", game: Game, state: State) -> dict:
    """The search's Q (mover-relative) for every legal root move — None for a move the search never visited."""
    node = agent._nodes.get(state_key(game, state))
    if node is None:
        return {a: None for a in game.legal_actions(state)}
    return {a: (node.child_w[a] / node.child_n[a] if node.child_n[a] else None) for a in node.legal}


def reanalyze_examples(
    game: Game, agent: "AlphaZeroAgent", states: list[State], rng: random.Random, with_q: bool = False
) -> list[tuple]:
    """MuZero REANALYZE (#2) — re-label stored positions with the CURRENT net for ~free data efficiency. For each
    stored `state`, re-run the current agent's GREEDY search (temperature 0 → the improved policy target, no
    exploration noise) to regenerate a fresh policy target AND the search-improved value. Old buffer entries were
    labelled by a weaker past net; refreshing them with the current, stronger net de-stales the targets without any
    new self-play. Returns `(encoded, pi_vec, value)` ready for the training buffer; `with_q` appends each legal
    move's root Q (`_root_q`) as a fourth field."""
    agent.temperature = 0.0
    out: list[tuple[torch.Tensor, list[float], float]] = []
    for state in states:
        if game.is_terminal(state):
            raise ValueError("reanalyze_examples was handed a terminal state — skipping it would shift every later "
                             "label onto the wrong buffer entry")
        agent._nodes = {}
        pi = agent.run_search(game, state, rng)
        pi_vec = [pi.get(a, 0.0) for a in range(game.num_actions)]
        value = max(-1.0, min(1.0, _root_search_value(agent, game, state)))
        row = (encode(game, state), pi_vec, value)
        out.append(row + (_root_q(agent, game, state),) if with_q else row)
    return out


def _checked_policy_target(game: Game, state: State, fn: Callable) -> list[float]:
    """A `policy_target_fn` label, refused unless it is a distribution over exactly the legal moves — a malformed
    oracle would otherwise train silently."""
    pi = [float(p) for p in fn(game, state)]
    legal = set(game.legal_actions(state))
    if not all(math.isfinite(p) for p in pi) \
            or len(pi) != game.num_actions or abs(sum(pi) - 1.0) > 1e-6 or any(p < 0 for p in pi) \
            or any(p > 0 for a, p in enumerate(pi) if a not in legal):
        raise ValueError("policy_target_fn must return a probability vector over num_actions with mass only on "
                         "legal moves")
    return pi


def one_ply_siblings(game: Game, states: list, key_fn, holdout: dict | None = None) -> tuple[list, dict]:
    """§C.46 SIBLINGS: the positions one move away from the recorded ones along a move self-play did NOT take —
    exactly the off-line positions self-play stops generating (§C.45 h30: after the random plies, 200-sim play never
    reaches a position where the second player wins). Pure and rng-free: parents in the order given, actions in
    `legal_actions` order; a child is kept when it is non-terminal, its key is not a recorded parent's, it is not
    held out, and it was not already kept (the first raw image found stands for its key). `holdout`
    {"mod", "salt"} withholds keys k with sha256(f"{salt}:{k!r}") % mod == 0, so what the net learns there can be
    measured as GENERALISATION rather than recall. Returns (children, counts): the skip counts are per (parent,
    move) EDGE, so a child reached from several parents counts once per edge; `added` counts distinct keys."""
    import hashlib

    if holdout is not None and int(holdout.get("mod", 0)) < 2:
        raise ValueError(f"sibling holdout mod must be >= 2, got {holdout.get('mod')}")

    def held(k) -> bool:
        if holdout is None:
            return False
        digest = hashlib.sha256(f"{holdout['salt']}:{k!r}".encode()).hexdigest()
        return int(digest, 16) % int(holdout["mod"]) == 0

    parents = {key_fn(s) for s in states}
    seen: set = set()
    out: list = []
    stats = {"terminal_skipped": 0, "recorded_skipped": 0, "holdout_skipped": 0, "added": 0}
    for s in states:
        for a in game.legal_actions(s):
            child = game.step(s, a)
            if game.is_terminal(child):
                stats["terminal_skipped"] += 1
                continue
            k = key_fn(child)
            if k in parents:
                stats["recorded_skipped"] += 1
                continue
            if held(k):
                stats["holdout_skipped"] += 1
                continue
            if k in seen:
                continue
            seen.add(k)
            out.append(child)
    stats["added"] = len(out)
    return out, stats


def sibling_positions(game: Game, states: list, key_fn, holdout: dict | None, depth: int) -> tuple[list, dict]:
    """§C.48 coverage: siblings out to `depth` moves off the recorded positions. Ring 1 is exactly
    `one_ply_siblings`; each further ring is the new non-terminal children of the ring before it, in the same fixed
    order, keeping only keys not recorded, not in an earlier ring and not held out. Pure and rng-free. The counts
    add up over rings, `added` counts distinct keys and `rings` the size of each ring."""
    if depth < 1:
        raise ValueError(f"sibling depth must be >= 1, got {depth}")
    out, stats = one_ply_siblings(game, states, key_fn, holdout)
    rings = [len(out)]
    known = {key_fn(st) for st in states} | {key_fn(st) for st in out}
    frontier = out
    for _ in range(depth - 1):
        children, ring_stats = one_ply_siblings(game, frontier, key_fn, holdout)
        fresh = [c for c in children if key_fn(c) not in known]
        stats["recorded_skipped"] += ring_stats["recorded_skipped"] + len(children) - len(fresh)
        stats["terminal_skipped"] += ring_stats["terminal_skipped"]
        stats["holdout_skipped"] += ring_stats["holdout_skipped"]
        known |= {key_fn(c) for c in fresh}
        out = out + fresh
        rings.append(len(fresh))
        frontier = fresh
    return out, {**stats, "added": len(out), "rings": rings}


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
    endgame_tb=None, exact_value_targets: bool = False, record_aux: bool = False,
    forced_opening: list[int] | None = None, record: dict | None = None, start_state: State | None = None,
) -> list:
    """Play ONE self-play game and return training examples (encoded board, policy, value). With `return_states`,
    each example is prefixed with the game STATE `(state, x, pi, v)` so the buffer can be REANALYZED (#2) — the
    stored state is what lets the current net re-search and re-label the position later. `opening_plies` > 0 plays
    that many RANDOM opening moves before net-guided play begins (those plies are NOT recorded as training
    examples) — so the net TRAINS on positions reached from DIVERSE openings, not just its own main line. This is
    the robustness lever: a net trained only on its canonical line loses AWAY from it (measured); off-line coverage
    teaches it to never lose a drawable position. Generic (no game knowledge). `start_state` starts the game from
    that position instead (no opening plies) — the backward curriculum's late starts."""
    if start_state is not None and game.is_terminal(start_state):
        raise ValueError("a self-play game cannot start from a finished position")
    agent._nodes = {}
    agent.add_noise = not agent.gumbel  # Gumbel supplies its own root exploration; Dirichlet would double it
    pending: list[tuple[State, torch.Tensor, list[float], int]] = []
    actions: list[int] = []
    if start_state is not None:
        state = start_state
    else:
        state = game.initial_state(rng)
        if forced_opening:  # §C.8 #5: replay a REFUTED line exactly (scripted, unrecorded — the search didn't pick
            for a in forced_opening:  # these moves, so they get no policy targets); recording starts where it led.
                if game.is_terminal(state) or a not in game.legal_actions(state):
                    break
                state = game.step(state, a, rng)
        else:
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
        actions.append(action)
        state = game.step(state, action, rng)
        move += 1
    returns = game.returns(state)
    if record is not None:
        # The ACTUAL game result, per seat — review-confirmed bug guard: the refutation resolver must read THIS,
        # never the (n-step-bootstrapped / tablebase-overridden) training TARGET, and it is filled even when the
        # forced prefix consumed the whole game and zero examples were recorded.
        record["returns"] = list(returns)
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
    if record_aux:
        # §C.8 #4 aux targets, mover-relative like everything else: `own` = the FINAL board read through THIS
        # position's mover perspective (game.observation — generic, no C4 knowledge); `reply` = the opponent's
        # actual next move (-1 for the last position). Consecutive examples are exact negations of each other.
        _h, _w = _board_shape(game)
        owns = [torch.tensor(game.observation(state, p)[: _h * _w], dtype=torch.float32)
                for (_s, _x, _pi, p) in pending]
        replies = [actions[i + 1] if i + 1 < len(actions) else -1 for i in range(len(pending))]
        if return_states:
            return [(s, x, pi_vec, v, o, r)
                    for (s, x, pi_vec, _p), v, o, r in zip(pending, values, owns, replies)]
        return [(x, pi_vec, v, o, r)
                for (_s, x, pi_vec, _p), v, o, r in zip(pending, values, owns, replies)]
    if return_states:
        return [(s, x, pi_vec, v) for (s, x, pi_vec, _p), v in zip(pending, values)]
    return [(x, pi_vec, v) for (_s, x, pi_vec, _p), v in zip(pending, values)]


def _nogood_prefix(actions: list[int], plies: int) -> tuple[int, ...]:
    """The prefix stored for a refuted game (§C.8 #5): capped at `plies` and ALWAYS excluding the final move —
    a full-game prefix replays to a terminal state with zero recorded examples (review-confirmed dead weight),
    so the replay must keep at least one ply to diverge on."""
    return tuple(int(a) for a in actions[: min(int(plies), len(actions) - 1)])


def _resolve_refutation(store, forced, record: dict | None) -> None:
    """After a forced replay of a refuted line (§C.8 #5): did P1 lose it AGAIN? Read from the game's ACTUAL
    per-seat returns (self_play_game's `record`) — review-confirmed bug: the first example's VALUE TARGET is
    n-step-bootstrapped / tablebase-overridden, so judging by it made retirement track the net's opinion instead
    of the replay outcome. The record is filled even for a replay that recorded zero examples (terminal prefix),
    so a nogood can always retire."""
    if store is None or not forced or not record or "returns" not in record:
        return
    store.resolve(tuple(forced), lost=float(record["returns"][0]) < 0)


def _frontier_order(game: Game, states, value_fn):
    """Frontier processing order (§C.8 #10, rescoped): ply-DESC stays PRIMARY (children before parents — the
    retrograde invariant the climb depends on); a `value_fn` (batched net values) only breaks ties WITHIN a ply
    tier, most-UNCERTAIN (|v| smallest) first — so a tight proof budget is spent where an exact target corrects
    the net the most. `value_fn=None` reproduces the legacy deepest-first order exactly."""
    from harness.book import _ply

    if value_fn is None:
        return sorted(states, key=lambda s: _ply(game, s), reverse=True)
    conf = {id(s): abs(float(v)) for s, v in zip(states, value_fn(states))}
    return sorted(states, key=lambda s: (-_ply(game, s), conf[id(s)]))


def _batched_net_values(net: "Connect4Net", game: Game, states, device: str = "cpu") -> list[float]:
    """One batched forward over `states` → mover-relative scalar values (the §C.8 #10 priority signal: a single
    net call for the whole frontier, never per-node inside the prover — per-node NN ordering would be a large
    SLOWDOWN in a µs-node pure-python prover)."""
    if not states:
        return []
    net.eval()
    with torch.no_grad():
        _p, v = net(torch.stack([encode(game, s) for s in states]).to(device))
    return [float(vi) for vi in v[:, 0]]


def extend_endgame_frontier(game: Game, run_tb, states, max_empty: int, max_positions: int,
                            deadline_seconds: float, value_fn=None) -> int:
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
    ordered = _frontier_order(game, list(seen.values()), value_fn)  # deepest first → children before parents
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
    record: dict | None = None,
    return_states: bool = False,
    opening_plies: int = 0,
) -> list[tuple]:
    """League game: the LEARNER (net-guided, exploring) plays an arbitrary opponent (a strong mcts / heuristic
    / a past champion). Training examples are collected from ONLY the learner's moves — we learn to BEAT the
    opponent, we don't imitate it. A caller-supplied `record` dict is filled with the played `actions` and the
    `learner_return` — the refutation-replay loop (§C.8 #5) reads it to learn WHICH lines the opponent refutes.
    `return_states` prefixes each example with its state (for the reanalyze buffer); `opening_plies` random moves are
    played first and not recorded, as in self-play."""
    learner._nodes = {}
    learner.add_noise = not learner.gumbel  # Gumbel supplies its own root exploration; Dirichlet would double it
    pending: list[tuple[State, torch.Tensor, list[float], int]] = []
    actions: list[int] = []
    state = game.initial_state(rng)
    for _ in range(opening_plies):
        if game.is_terminal(state):
            break
        state = game.step(state, rng.choice(game.legal_actions(state)), rng)
    move = 0
    while not game.is_terminal(state):
        player = game.current_player(state)
        if player == learner_seat:
            learner.temperature = 1.0 if move < temp_moves else 0.0
            pi = learner.run_search(game, state, rng)
            pi_vec = [pi.get(a, 0.0) for a in range(game.num_actions)]
            pending.append((state, encode(game, state), pi_vec, player))
            action = learner._gumbel_selected if learner.gumbel else sample_action(pi, learner.temperature, rng)
        else:
            action = opponent.act(game, state, rng)
        actions.append(action)
        state = game.step(state, action, rng)
        move += 1
    returns = game.returns(state)
    if record is not None:
        record["actions"] = actions
        record["learner_return"] = returns[learner_seat]
    if return_states:
        return [(s, x, pi_vec, returns[player]) for (s, x, pi_vec, player) in pending]
    return [(x, pi_vec, returns[player]) for (_s, x, pi_vec, player) in pending]


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
    examples: list[tuple[torch.Tensor, list[float], float]],
    perms: list[tuple[list[int], list[int]]] | None,
) -> list[tuple[torch.Tensor, list[float], float]]:
    """Multiply training examples by a game's symmetries (Lever 2 for the net). Each symmetry is a PAIR
    `(cell_perm, action_perm)`, both source-permutations `dest <- src`:

      - `cell_perm` (length rows*cols) reorders the board planes' FLATTENED cell axis, so a 2D isometry (a
        rotation, a diagonal flip) is expressible — not only a reordering of the last tensor axis. Before this,
        `augment` reindexed the width axis with the action perm, which is correct ONLY when the action space IS
        the board width (Connect-4's columns). tictactoe's 9-cell dihedral perms indexed a size-3 axis and threw
        IndexError; othello and checkers could not expose symmetries at all.
      - `action_perm` (length num_actions) reorders the policy vector.

    The value is invariant. Identity-only (or no perms) is a plain copy. Connect-4 is byte-identical to the old
    single-perm path (its cell_perm is the column mirror lifted to rows*cols, its action_perm the column mirror)."""
    if not perms or len(perms) <= 1:
        return list(examples)
    out: list[tuple] = []
    for e in examples:
        x, pi, v = e[0], e[1], e[2]
        flat = x.reshape(x.shape[0], -1)
        ident_cells = list(range(flat.shape[1]))
        ident_acts = list(range(len(pi)))
        for cell_perm, action_perm in perms:
            if cell_perm == ident_cells and action_perm == ident_acts:
                base = (x, list(pi), v)
            else:
                bx = flat[:, cell_perm].reshape(x.shape)
                base = (bx, [pi[a] for a in action_perm], v)
            if len(e) >= 5:  # aux targets (§C.8 #4): ownership is per-CELL, the reply is an ACTION index
                own, reply = e[3], e[4]
                if cell_perm == ident_cells and action_perm == ident_acts:
                    out.append(base + (own, reply))
                else:
                    own_m = own[cell_perm]
                    out.append(base + (own_m, action_perm.index(reply) if reply >= 0 else -1))
            else:
                out.append(base)
    return out


def _two_hot(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Project scalar targets in [-1,1] onto a bin support as TWO-HOT distributions (§C.8 #3): an exact bin
    centre takes all the mass, anything between two centres splits linearly. This is what turns value learning
    into classification (cross-entropy) instead of the MSE whose optimum under 50/50 outcomes is the ~0 collapse."""
    k = support.shape[0]
    v = values.reshape(-1).clamp(float(support[0]), float(support[-1]))
    step = (float(support[-1]) - float(support[0])) / (k - 1)
    pos = (v - float(support[0])) / step
    lo = pos.floor().long().clamp(0, k - 1)
    hi = (lo + 1).clamp(0, k - 1)
    frac = (pos - lo.float()).unsqueeze(1)
    dist = torch.zeros(v.shape[0], k, device=values.device)
    dist.scatter_(1, lo.unsqueeze(1), 1.0 - frac)
    dist.scatter_add_(1, hi.unsqueeze(1), frac)
    return dist


def _value_loss(value: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
    """Value MSE over the examples that CARRY a value target. A NaN target marks a policy-only example (§C.46
    siblings: a position self-play never reached, labelled by search but with no game outcome to learn from) and
    is left out; with no NaN present this is exactly `F.mse_loss`, so every existing run trains byte-identically."""
    mask = ~torch.isnan(target_v)
    if bool(mask.all()):
        return F.mse_loss(value, target_v)
    if not bool(mask.any()):
        return (value * 0.0).sum()
    return ((value - torch.nan_to_num(target_v)) ** 2)[mask].mean()


def train_net(
    net: Connect4Net,
    examples: list[tuple[torch.Tensor, list[float], float]],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    opt_state: dict | None = None,
    epoch_examples: int | None = None,
    lr_end: float | None = None,
) -> float:
    """One training pass over the buffer (policy cross-entropy + value MSE — or value cross-entropy against a
    two-hot target when the net carries a categorical value head). Returns the final mean loss.

    `epoch_examples` caps each epoch at that many examples of a fresh random permutation (§C.46 steps-matched
    mode: an arm that ADDS examples keeps the optimisation steps of the arm without them, so exposure is not
    confounded with extra gradient steps). None — or the full size — trains exactly as before.

    `lr_end` decays the learning rate linearly, step by step, from `lr` to `lr_end` at the last step (§C.48 settling:
    a constant rate leaves the net on a noise floor around the fit). None keeps the optimizer's rate untouched."""
    if not examples:
        return 0.0
    if epoch_examples is not None and not 1 <= epoch_examples <= len(examples):
        raise ValueError(f"epoch_examples {epoch_examples} outside 1..{len(examples)}")
    if lr_end is not None and not 0.0 < lr_end <= lr:
        raise ValueError(f"lr_end {lr_end} outside (0, lr={lr}] — the schedule only decays")
    bins = int(net.arch.get("value_bins", 0))
    aux_on = bool(net.arch.get("aux_heads", False))
    x = torch.stack([e[0] for e in examples]).to(device)
    target_p = torch.tensor([e[1] for e in examples], dtype=torch.float32).to(device)
    target_v = torch.tensor([[e[2]] for e in examples], dtype=torch.float32).to(device)
    if aux_on and bool(torch.isnan(target_v).any()):
        raise ValueError("policy-only examples (NaN value target) are not supported with aux heads — their ownership "
                         "and reply terms would train on positions with no outcome")
    if bins > 0 and bool(torch.isnan(target_v).any()):
        raise ValueError("policy-only examples (NaN value target) are not supported with a categorical value head — "
                         "a two-hot of NaN would poison the value loss")
    target_dist = _two_hot(target_v, net.value_support) if bins > 0 else None
    if aux_on:
        # §C.8 #4: aux targets apply ONLY to examples that carry them — league/distill 3-tuples train alongside
        # aux-recorded self-play with their ownership/reply terms masked out (weight 0), never faked.
        zeros = torch.zeros(net.board_h * net.board_w)
        target_own = torch.stack([e[3] if len(e) >= 5 else zeros for e in examples]).to(device)
        target_reply = torch.tensor([e[4] if len(e) >= 5 else -1 for e in examples], dtype=torch.long).to(device)
        aux_mask = torch.tensor([1.0 if len(e) >= 5 else 0.0 for e in examples]).to(device)
    # §C.9 BUG FIX: a fresh Adam per call reset the moment estimates on EVERY training iteration (~200x in a
    # long run), so the optimizer never accumulated any state. Callers that pass `opt_state` keep ONE Adam for
    # the life of the run; passing nothing preserves the old behaviour byte-for-byte.
    if opt_state is None:
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    else:
        opt = opt_state.get("opt")
        if opt is None:
            opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
            # §C.22: a RESUMED run hands back the Adam state its earlier process saved, so the moment estimates
            # continue instead of restarting — otherwise an interrupted run is a different experiment.
            carried = opt_state.pop("load_state", None)
            if carried is not None:
                opt.load_state_dict(carried)
            opt_state["opt"] = opt
    net.train()
    last = 0.0
    epoch_sum = epoch_seen = 0.0
    n = len(examples)
    per_epoch = n if epoch_examples is None else int(epoch_examples)
    total_steps = epochs * math.ceil(per_epoch / batch_size)
    step = 0
    for _ in range(epochs):
        perm = torch.randperm(n)
        if per_epoch < n:
            perm = perm[:per_epoch]
        for i in range(0, per_epoch, batch_size):
            b = perm[i : i + batch_size]
            if aux_on:
                logits, value, own_pred, reply_logits = net.forward_aux(x[b])
            else:
                logits, value = net.forward_train(x[b])
            policy_loss = -(target_p[b] * F.log_softmax(logits, dim=1)).sum(1).mean()
            if bins > 0:
                value_loss = -(target_dist[b] * F.log_softmax(value, dim=1)).sum(1).mean()
            else:
                value_loss = _value_loss(value, target_v[b])
            loss = policy_loss + value_loss
            if aux_on:
                m = aux_mask[b]
                if float(m.sum()) > 0:
                    own_per = ((own_pred - target_own[b]) ** 2).mean(1)  # per-example ownership MSE
                    loss = loss + 0.15 * (own_per * m).sum() / m.sum()
                    rb = target_reply[b]
                    valid = rb >= 0  # last-move examples (and masked 3-tuples, rb=-1) carry no reply target
                    if bool(valid.any()):
                        loss = loss + 0.15 * F.cross_entropy(reply_logits[valid], rb[valid])
            opt.zero_grad()
            loss.backward()
            step += 1
            if lr_end is not None:
                for group in opt.param_groups:
                    group["lr"] = lr + (lr_end - lr) * step / total_steps
            opt.step()
            # §C.10 BUILD #3: accumulate an EPOCH MEAN. The old `last = float(loss.detach())` reported one
            # mini-batch — a single-sample statistic (within-run SD ~0.23) that we mistook for a fit-quality
            # signal across architectures. Loss-vs-skill correlation was ~-0.09; the column meant nothing.
            epoch_sum += float(loss.detach()) * len(b)
            epoch_seen += len(b)
        last = epoch_sum / max(1, epoch_seen)
        epoch_sum = epoch_seen = 0.0
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
        pi = [0.0] * game.num_actions
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
        acts = [c for c in range(game.num_actions) if (entry.best_actions >> c) & 1]
        if not acts:
            continue
        pi = [0.0] * game.num_actions
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
            pi = [0.0] * game.num_actions
            for c in optimal:
                pi[c] = 1.0 / len(optimal)
            action = optimal[0] if len(optimal) == 1 else min(optimal, key=lambda c: abs(c - game.num_actions // 2))
            return pi, action, bv
        action = oracle.act(game, state, rng)
        return [1.0 if c == action else 0.0 for c in range(game.num_actions)], action, bv

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


# Relabelling is the other sequential cost: the 200-sim search over the buffer, its siblings and the strategy tree
# runs one position at a time. It parallelises EXACTLY: each position's label depends only on the net and the
# position (the relabel search is greedy, with no noise), so splitting the list across workers and joining the
# results in order gives the serial labels bit for bit. The workers get an rng that refuses every draw, so a
# configuration that WOULD draw (root noise, a stochastic game) fails loudly instead of silently diverging.
_RELABEL_WORKER: dict = {}
# A pool whose result thread dies (a torch shared-memory handoff timed out under load, §C.49 T10) never answers; a
# relabel call that waits longer than this raises instead of hanging the run.
RELABEL_TIMEOUT_S = 3 * 3600


class _NoRandom:
    """An rng that refuses every draw: a parallel relabel must not depend on how the positions were split."""

    def __getattr__(self, name):
        raise RuntimeError(f"a parallel relabel drew from the rng ({name}) — its labels would depend on the split")


def _relabel_worker_init(game_name: str) -> None:
    import torch as _torch

    _torch.set_num_threads(1)
    from harness.registry import resolve_game

    _RELABEL_WORKER.clear()
    _RELABEL_WORKER["game"] = resolve_game(game_name)


def _relabel_worker(task: tuple) -> list:
    net_path, version, agent_kwargs, states, with_q = task
    st = _RELABEL_WORKER
    if st.get("version") != version:
        st["net"] = load_net(net_path)
        st["version"] = version
    labels = reanalyze_examples(st["game"], AlphaZeroAgent(st["net"], **agent_kwargs), states, _NoRandom(), with_q)
    return [(row[0].numpy(), *row[1:]) for row in labels]


def _parallel_relabel(pool, workers: int, tmpdir: str, net, version: int, agent_kwargs: dict, states: list,
                      with_q: bool = False) -> list:
    """`reanalyze_examples` over `states` across `pool`, joined in order — the serial labels, bit for bit. Workers
    hand back plain arrays (a tensor would travel through torch's shared-memory manager, which can time out), and
    a pool that does not answer within RELABEL_TIMEOUT_S raises."""
    import multiprocessing
    import os

    if not states:
        return []
    net_path = os.path.join(tmpdir, "relabel_net.pt")
    save_net(net, net_path)
    size = max(1, -(-len(states) // (4 * workers)))
    chunks = [states[i:i + size] for i in range(0, len(states), size)]
    pending = pool.map_async(_relabel_worker, [(net_path, version, agent_kwargs, c, with_q) for c in chunks])
    try:
        parts = pending.get(timeout=RELABEL_TIMEOUT_S)
    except multiprocessing.TimeoutError:
        raise RuntimeError(f"the relabel pool did not answer within {RELABEL_TIMEOUT_S}s — its result thread may have "
                           f"died; the run stops rather than hang") from None
    return [(torch.from_numpy(row[0]), *row[1:]) for part in parts for row in part]


def _endgame_enabled(game: Game, endgame_tb) -> bool:
    """§C.7 #2 GENERIC GATE: the online endgame loop runs ONLY when a run tablebase is present AND the game exposes
    the exact hooks (canonical_key + exact_optimal_actions). Absent either, the caller builds the learner with
    book=None/solve_endgame=0 — byte-identical to pure #1 (chess opening / Go degrade cleanly, never crash)."""
    return (
        endgame_tb is not None
        and getattr(game, "canonical_key", None) is not None
        and getattr(game, "exact_optimal_actions", None) is not None
    )


def _self_agreement(net: Connect4Net, game: Game, rows: list, device: str) -> dict:
    """§C.48: the share of buffer positions where the raw policy's argmax over legal moves is one of its own label's
    best moves — a solver-free reading of whether the net has fit what it is being taught. Observes only: eval mode,
    no gradient, no RNG, and the net is put back in the mode it was in."""
    if not rows:
        return {"self_agreement": 0.0, "self_agreement_positions": 0}
    was_training = net.training
    net.eval()
    with torch.no_grad():
        logits, _v = net(torch.stack([x for (_s, x, _pi, _v) in rows]).to(device))
    net.train(was_training)
    hits = 0
    for (s, _x, pi, _v), row in zip(rows, logits.cpu(), strict=True):
        legal = game.legal_actions(s)
        move = max(legal, key=lambda a: float(row[a]))
        hits += pi[move] >= max(pi)
    return {"self_agreement": hits / len(rows), "self_agreement_positions": len(rows)}


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
    reanalyze_sims: int | None = None,
    reanalyze_siblings: bool = False,
    steps_matched: bool = False,
    buffer_unique: bool = False,
    settle_epochs: int = 0,
    settle_lr_final: float = 1e-5,
    settle_lr: float | None = None,
    record_self_agreement: bool = False,
    sibling_depth: int = 1,
    strategy_tree: dict | None = None,
    relabel_workers: int = 1,
    stop_on_agreement: bool = False,
    stop_value_delta: float | None = None,
    tree_value_target: bool = False,
    backplay: dict | None = None,
    exploiter: dict | None = None,
    selfplay_starts: list | None = None,
    tree_roots: list | None = None,
    sibling_holdout: dict | None = None,
    policy_target_fn: Callable | None = None,
    selfplay_opening_plies: int = 0,
    opening_plies_zero_frac: float = 0.0,
    endgame_net_priority: bool = False,
    opt_state: dict | None = None,
    refutation_frac: float = 0.0,
    refutation_prefix_plies: int = 6,
    refutation_store=None,
    endgame_tb=None,
    endgame_max_empty: int = 14,
    endgame_exact_targets: int = 1,
    endgame_extend_positions: int = 2000,
    endgame_extend_seconds: float = 5.0,
    net_arch: dict | None = None,
    init_buffer: list | None = None,
    return_buffer: bool = False,
    selfplay_workers: int = 1,
    league_p1_frac: float = 0.5,
    opening_anchor_cap: int = 0,
    league_anchor_frac: float = 0.0,
    league_frozen_self: bool = False,
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
    net = init_net if init_net is not None else Connect4Net(
        **arch_for_game(net_arch or {"channels": channels}, game)).to(device)
    # §C.8 #4: an aux-headed net auto-records its own targets in self-play (no extra knob to forget); the
    # reanalyze state-buffer path drops aux fields, so the combination is refused rather than silently degraded.
    aux_on = bool(net.arch.get("aux_heads", False))
    canonical = bool(net.arch.get("canonical_input", False))
    if canonical and augment:
        raise ValueError("a canonical_input net sees every orientation as one input — augment would only duplicate "
                         "its training rows")
    if aux_on and reanalyze_frac > 0.0:
        raise ValueError("aux_heads + reanalyze_frac are not combinable (the state buffer drops aux targets)")
    # §C.45: `reanalyze_sims` relabels the buffer with its OWN search budget while self-play keeps `sims` — the one
    # treatment that changes the policy LABEL alone (play, outcomes and visited states stay at the self-play budget).
    if reanalyze_sims is not None and reanalyze_frac <= 0.0:
        raise ValueError("reanalyze_sims is set but reanalyze_frac is 0 — nothing is relabelled, so the knob would "
                         "silently do nothing")
    # §C.46: siblings, steps matching and the policy-target hook all act at RELABEL time on the reanalyze path, so
    # each is refused wherever it would silently do nothing or mix with a path that cannot carry it.
    if reanalyze_siblings and reanalyze_frac != 1.0:
        raise ValueError("reanalyze_siblings needs reanalyze_frac == 1.0 — siblings are relabelled with the whole buffer")
    if (buffer_unique or settle_epochs > 0 or record_self_agreement) and reanalyze_frac <= 0.0:
        raise ValueError("buffer_unique / settle_epochs / record_self_agreement act on the reanalyze state buffer — "
                         "set reanalyze_frac")
    if relabel_workers != 1 and (relabel_workers < 1 or reanalyze_frac <= 0.0 or reanalyze_sims is None
                                 or getattr(game, "name", None) is None):
        raise ValueError("relabel_workers spreads the separate relabel search (reanalyze_frac > 0 and reanalyze_sims "
                         "set) over >= 1 worker processes of a named game")
    if stop_on_agreement and strategy_tree is None:
        raise ValueError("stop_on_agreement reads the strategy-tree walk — set strategy_tree")
    if stop_value_delta is not None and (strategy_tree is None or stop_value_delta < 0):
        raise ValueError("stop_value_delta reads the strategy-tree walk's search values — set strategy_tree and a "
                         "delta >= 0")
    if backplay is not None and (reanalyze_frac <= 0.0 or not 0.0 < backplay["frac"] < 1.0 or backplay["ramp"] < 1):
        raise ValueError("backplay starts games from the state buffer's recorded games — it needs reanalyze_frac "
                         "> 0, 0 < frac < 1 (some games must start normally to refill its pool) and ramp >= 1")
    if exploiter is not None and (reanalyze_frac <= 0.0 or not 0.0 < exploiter["frac"] <= 1.0
                                  or exploiter["sims_factor"] <= 1):
        raise ValueError("an exploiter plays games for the state buffer — it needs reanalyze_frac > 0, 0 < frac <= 1 "
                         "and sims_factor > 1 (more search than the learner)")
    if selfplay_starts is not None and (reanalyze_frac <= 0.0 or not selfplay_starts or backplay is not None
                                        or exploiter is not None):
        raise ValueError("selfplay_starts starts every state-buffer self-play game at one of the given positions — it "
                         "needs reanalyze_frac > 0, at least one position, and no backplay or exploiter")
    if tree_roots is not None and (strategy_tree is None or not tree_roots):
        raise ValueError("tree_roots walks the strategy tree from the given positions — set strategy_tree and give at "
                         "least one root")
    if tree_value_target and strategy_tree is None:
        raise ValueError("tree_value_target gives the strategy-tree positions a value target — set strategy_tree")
    if strategy_tree is not None and reanalyze_frac != 1.0:
        raise ValueError("strategy_tree relabels its positions with the whole-buffer search — set reanalyze_frac=1.0")
    if settle_lr is not None and settle_epochs <= 0:
        raise ValueError("settle_lr sets the settle's starting rate — it needs settle_epochs")
    if sibling_depth != 1 and not reanalyze_siblings:
        raise ValueError("sibling_depth acts only on siblings — set reanalyze_siblings")
    if (steps_matched or sibling_holdout is not None) and not reanalyze_siblings:
        raise ValueError("steps_matched / sibling_holdout act only on siblings — set reanalyze_siblings")
    if sibling_holdout is not None and int(sibling_holdout.get("mod", 0)) < 2:
        raise ValueError(f"sibling_holdout mod must be >= 2, got {sibling_holdout.get('mod')}")
    if sibling_holdout is not None and not isinstance(sibling_holdout.get("salt"), str):
        raise ValueError("sibling_holdout needs a string 'salt' — without one the held-out set is undefined, and the "
                         "run would fail only after the first pass had been paid for")
    if reanalyze_siblings and (aux_on or int(net.arch.get("value_bins", 0)) > 0):
        raise ValueError("siblings carry no value target, which aux heads and a categorical value head cannot mask")
    if policy_target_fn is not None and (reanalyze_frac != 1.0 or reanalyze_sims is not None):
        raise ValueError("policy_target_fn replaces the relabel search, so it needs reanalyze_frac == 1.0 and no "
                         "reanalyze_sims")
    if refutation_frac > 0.0 and reanalyze_frac > 0.0:
        raise ValueError("refutation_frac is inert on the reanalyze path (its store is never replayed there)")
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
    if reanalyze_siblings and distilled:
        raise ValueError("siblings are refused with a distilled anchor — its fixed-fraction mix would change what the "
                         "steps-matched pass samples")
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
    backplay_pool: list[list[State]] = []
    # §C.7 #2: the online endgame loop is armed only for a game with the exact hooks — else pure #1 (no crash).
    endgame_on = _endgame_enabled(game, endgame_tb)
    eg_targets = endgame_on and bool(endgame_exact_targets)
    # §C.7 PARALLEL self-play: only the PURE-#1 path is safe to fan out (no shared endgame tablebase, no league
    # opponent, no reanalyze state-buffer). Otherwise stay sequential (byte-identical). Needs a game name to respawn.
    # §C.8 #13 mixed openings: 0 < frac ≤ 1 sends that share of self-play games to the CANONICAL line (0 plies)
    # and the rest to diverse openings — sharpness AND coverage from one buffer. frac == 0.0 draws NOTHING from
    # rng, so existing runs stay byte-identical.
    def _game_plies(r: random.Random) -> int:
        if opening_plies_zero_frac > 0.0 and r.random() < opening_plies_zero_frac:
            return 0
        return selfplay_opening_plies

    # §C.8 #5 refutation-replay: nogoods live in `_refut_store` (caller-supplied for persistence across batches,
    # else run-local). frac == 0.0 with no store leaves EVERYTHING untouched — no extra rng draws, no records.
    _refut_store = refutation_store
    if _refut_store is None and refutation_frac > 0.0:
        from harness.refutation import RefutationStore

        _refut_store = RefutationStore()
    parallel_ok = (int(selfplay_workers) > 1 and opponent_pool is None and reanalyze_frac == 0.0
                   and opening_plies_zero_frac == 0.0 and not aux_on and refutation_frac == 0.0
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
    _relabel_pool = _relabel_dir = None
    relabel_calls = [0]
    if relabel_workers > 1:
        if endgame_on:
            raise ValueError("relabel_workers cannot share the run's endgame tablebase with worker processes")
        import multiprocessing as _mp
        import os as _os
        import tempfile

        _relabel_dir = tempfile.mkdtemp(prefix="az_rl_")
        _saved = {k: _os.environ.get(k) for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}
        for k in _saved:
            _os.environ[k] = "1"
        _relabel_pool = _mp.get_context("spawn").Pool(int(relabel_workers), initializer=_relabel_worker_init,
                                                      initargs=(game.name,))
        for k, v in _saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
    # §C.7 #3 SOLVER-FREE LEAGUE: the opening anchor accumulates seat-0 (empty-board) league examples with their
    # TRUE outcomes (honest, never win-filtered) so the opening signal isn't diluted below distillation's pin; the
    # frozen-self rung is a batch-start deepcopy — an equal-strength opponent that beats a genuinely-lost opening,
    # the counter-pressure keeping the anchor honest. league off ⇒ opponent_pool is None ⇒ all inert.
    opening_anchor: list = []
    _league_pool = None
    if opponent_pool is not None:
        _league_pool = list(opponent_pool)
        if league_frozen_self:
            _frozen = copy.deepcopy(net)
            _league_pool.append(lambda fn=_frozen: AlphaZeroAgent(fn, sims=sims, device=device, gumbel=gumbel,
                                                                  gumbel_m=gumbel_m, c_scale=c_scale))
    # §C.14: the RUN owns this (scaled_run passes it across batches). A local default keeps other
    # callers working, but then Adam resets per call — which is the bug this parameter fixes.
    _opt_state: dict = opt_state if opt_state is not None else {}
    stopped = False
    for it in range(iterations):
        reanalyze_note: dict = {}
        epoch_cap: int | None = None
        sibling_relabel_s = relabel_s = 0.0
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
            t_selfplay = time.time()
            full_games: list[list[State]] = []
            backplay_games = exploiter_games = 0
            if exploiter is not None:
                exploiter_agent = AlphaZeroAgent(net, sims=sims * exploiter["sims_factor"], device=device,
                                                 gumbel=gumbel, gumbel_m=gumbel_m, c_scale=c_scale)
                exploiter_agent.temperature = 0.0
                exploiter_agent.add_noise = False
            for _ in range(selfplay_games):
                start = None
                if backplay is not None and backplay_pool and rng.random() < backplay["frac"]:
                    source = backplay_pool[rng.randrange(len(backplay_pool))]
                    reach = min(len(source), math.ceil(len(source) * (it + 1) / backplay["ramp"]))
                    start = source[len(source) - rng.randint(1, reach)]
                    backplay_games += 1
                if selfplay_starts is not None:
                    start = selfplay_starts[rng.randrange(len(selfplay_starts))]
                if exploiter is not None and start is None and rng.random() < exploiter["frac"]:
                    fresh_s.extend(vs_opponent_game(game, learner, exploiter_agent, rng.randrange(game.num_players),
                                                    rng, return_states=True, opening_plies=_game_plies(rng)))
                    exploiter_games += 1
                    continue
                played = self_play_game(game, learner, rng, target_net=target_net, n_step=value_n_step,
                                        device=device, return_states=True, opening_plies=_game_plies(rng),
                                        endgame_tb=(endgame_tb if endgame_on else None),
                                        exact_value_targets=eg_targets, start_state=start)
                fresh_s.extend(played)
                if start is None and played:
                    full_games.append([row[0] for row in played])
            if backplay is not None and full_games:
                backplay_pool = full_games
            if endgame_on:
                eg_visited.extend(s for (s, *_rest) in fresh_s)
            selfplay_s = time.time() - t_selfplay
            if buffer_unique:
                row_key = game.canonical_key if canonical else game.state_key
                held = {row_key(row[0]): row for row in state_buffer}
                for row in fresh_s:
                    held.pop(row_key(row[0]), None)
                    held[row_key(row[0])] = row
                merged = len(state_buffer) + len(fresh_s) - len(held)
                evicted = max(0, len(held) - buffer_cap)
                state_buffer = list(held.values())[-buffer_cap:]
            else:
                merged = 0
                evicted = max(0, len(state_buffer) + len(fresh_s) - buffer_cap)
                state_buffer = (state_buffer + fresh_s)[-buffer_cap:]
            t_relabel = time.time()
            sib_rows: list = []
            tree_rows: list = []
            tree_walked = tree_disagree = tree_share_disagree = 0
            sib_stats = {"terminal_skipped": 0, "recorded_skipped": 0, "holdout_skipped": 0, "added": 0}
            # Re-label a random sample of the buffer with the CURRENT net (fresh policy + search-improved value).
            k = int(reanalyze_frac * len(state_buffer))
            if it > 0 and k > 0:
                idxs = rng.sample(range(len(state_buffer)), k)
                relabeler = learner if reanalyze_sims is None else AlphaZeroAgent(
                    net, sims=reanalyze_sims, device=device, gumbel=gumbel, gumbel_m=gumbel_m, c_scale=c_scale,
                    book=(endgame_tb if endgame_on else None), solve_endgame=(endgame_max_empty if endgame_on else 0))
                if relabeler is not learner:
                    relabeler.add_noise = learner.add_noise  # self-play left the learner's root-noise state set

                def relabel(states: list, with_q: bool = False) -> list:
                    relabel_calls[0] += 1
                    if _relabel_pool is None:
                        return reanalyze_examples(game, relabeler, states, rng, with_q)
                    return _parallel_relabel(_relabel_pool, relabel_workers, _relabel_dir, net, relabel_calls[0],
                                             {"sims": reanalyze_sims, "gumbel": gumbel, "gumbel_m": gumbel_m,
                                              "c_scale": c_scale, "add_noise": relabeler.add_noise}, states, with_q)
                if policy_target_fn is not None:
                    relabelled = [(encode(game, s), _checked_policy_target(game, s, policy_target_fn), None)
                                  for s in (state_buffer[i][0] for i in idxs)]
                else:
                    relabelled = relabel([state_buffer[i][0] for i in idxs])
                if relabeler is not learner:
                    learner.endgame_solves += relabeler.endgame_solves
                    learner.endgame_hits += relabeler.endgame_hits
                # POLICY-ONLY refresh: replace the POLICY target with the current net's, but KEEP the stored n-step
                # VALUE target — measured: overwriting the value with a search-root estimate DEGRADES the n-step
                # value (the #3 lever), so reanalyze must not touch it.
                for i, (x, pi, _v_search) in zip(idxs, relabelled, strict=True):
                    state_buffer[i] = (state_buffer[i][0], x, pi, state_buffer[i][3])
                reanalyzed = len(relabelled)
                relabel_s = time.time() - t_relabel
                if reanalyze_siblings:
                    key_fn = (game.canonical_key if (augment or canonical) and hasattr(game, "canonical_key")
                              else game.state_key)
                    sibs, sib_stats = sibling_positions(game, [st for (st, *_rest) in state_buffer], key_fn,
                                                        sibling_holdout, sibling_depth)
                    t_sib = time.time()
                    if policy_target_fn is not None:
                        sib_rows = [(encode(game, st), _checked_policy_target(game, st, policy_target_fn), float("nan"))
                                    for st in sibs]
                    else:
                        sib_rows = [(x, pi, float("nan")) for (x, pi, _v) in relabel(sibs)]
                    sibling_relabel_s = time.time() - t_sib
                if strategy_tree is not None:
                    choose = raw_chooser(game, net)
                    if tree_roots is None:
                        walked = strategy_tree_positions(game, game.initial_state(random.Random(0)),
                                                         strategy_tree["player"], choose, strategy_tree.get("depth"))
                    else:
                        by_key: dict = {}
                        for tree_root in tree_roots:
                            for st in strategy_tree_positions(game, tree_root, strategy_tree["player"], choose,
                                                              strategy_tree.get("depth")):
                                by_key.setdefault(game.state_key(st), st)
                        walked = list(by_key.values())
                    tree_walked = len(walked)
                    if walked:
                        tree_labels = relabel(walked, stop_value_delta is not None)
                        raw_moves, tree_pis = choose(walked), [row[1] for row in tree_labels]
                        if stop_value_delta is None:
                            tree_disagree = disagreements(raw_moves, tree_pis)
                        else:
                            tree_disagree = disagreements(raw_moves, tree_pis, [row[3] for row in tree_labels],
                                                          stop_value_delta)
                            tree_share_disagree = disagreements(raw_moves, tree_pis)
                        tree_key = (game.canonical_key if (augment or canonical) and hasattr(game, "canonical_key")
                                    else game.state_key)
                        held = {tree_key(st) for (st, *_rest) in state_buffer}
                        tree_rows = [(row[0], row[1], row[2] if tree_value_target else float("nan"))
                                     for st, row in zip(walked, tree_labels, strict=True) if tree_key(st) not in held]
            sp_aug = augment_examples([(x, pi, v) for (_s, x, pi, v) in state_buffer], perms)
            sib_aug = augment_examples(sib_rows, perms) if sib_rows else []
            tree_aug = augment_examples(tree_rows, perms) if tree_rows else []
            buffer = sp_aug + sib_aug + tree_aug
            reanalyze_note = {"selfplay_states": len(fresh_s), "state_buffer": len(state_buffer), "evicted": evicted,
                              **({"backplay_games": backplay_games} if backplay is not None else {}),
                              **({"exploiter_games": exploiter_games} if exploiter is not None else {}),
                              **({"merged": merged} if buffer_unique else {}), "siblings": sib_stats["added"],
                              **({"sibling_rings": sib_stats.get("rings", [])} if reanalyze_siblings else {}),
                              "sibling_terminal_skipped": sib_stats["terminal_skipped"],
                              "sibling_recorded_skipped": sib_stats["recorded_skipped"],
                              "sibling_holdout_skipped": sib_stats["holdout_skipped"],
                              "nan_value_examples": len(sib_aug) + (0 if tree_value_target else len(tree_aug)),
                              "selfplay_s": round(selfplay_s, 3),
                              "relabel_s": round(relabel_s, 3), "sibling_relabel_s": round(sibling_relabel_s, 3)}
            epoch_cap = len(sp_aug) if (steps_matched and (sib_aug or tree_aug)) else None
            if strategy_tree is not None:
                reanalyze_note.update({"tree_walked": tree_walked, "tree_positions": len(tree_rows),
                                       "tree_disagreements": tree_disagree,
                                       **({"tree_disagreements_share": tree_share_disagree}
                                          if stop_value_delta is not None else {})})
                stopped = stop_on_agreement and tree_walked > 0 and tree_disagree == 0
        elif parallel_ok:  # PURE-#1 fanned out across worker processes (fills the idle cores; ~2-2.5x faster)
            fresh = _run_parallel_selfplay(_pool, _tmpdir, net, target_net, it, selfplay_games, sims, gumbel,
                                           gumbel_m, c_scale, value_n_step, selfplay_opening_plies, rng)
            fresh = augment_examples(fresh, perms)
            buffer = (buffer + fresh)[-buffer_cap:]
        else:
            fresh: list[tuple[torch.Tensor, list[float], float]] = []
            for _ in range(selfplay_games):
                if opponent_pool and rng.random() < pool_frac:
                    # SEAT-PRIORITY: sit the learner on the won P1 seat most of the time (where the collapse lives
                    # and opening_value is read), so wins from the TRUE opening label the empty board +1 via outcomes.
                    learner_seat = 0 if rng.random() < league_p1_frac else 1
                    opponent = _league_pool[rng.randrange(len(_league_pool))]()
                    _rec = {} if (_refut_store is not None and learner_seat == 0) else None
                    ex_lg = vs_opponent_game(game, learner, opponent, learner_seat, rng, record=_rec)
                    if _rec is not None and _rec.get("learner_return", 0.0) < 0:
                        # the opponent REFUTED a line the learner entered as P1 → store its opening as a nogood
                        _refut_store.add(_nogood_prefix(_rec["actions"], refutation_prefix_plies))
                    fresh.extend(ex_lg)
                    if learner_seat == 0 and opening_anchor_cap > 0 and ex_lg:
                        opening_anchor.append(ex_lg[0])  # empty board + its TRUE outcome (honest, not win-filtered)
                        if len(opening_anchor) > opening_anchor_cap:
                            del opening_anchor[: len(opening_anchor) - opening_anchor_cap]
                    vs_pool += 1
                elif endgame_on:  # collect STATES for the frontier extension (return_states only changes the shape)
                    _forced = (_refut_store.sample(rng)
                               if _refut_store is not None and refutation_frac > 0.0
                               and rng.random() < refutation_frac else None)
                    _frec = {} if _forced else None
                    ex = self_play_game(game, learner, rng, target_net=target_net, n_step=value_n_step, device=device,
                                        opening_plies=_game_plies(rng), return_states=True,
                                        endgame_tb=endgame_tb, exact_value_targets=eg_targets, record_aux=aux_on,
                                        forced_opening=list(_forced) if _forced else None, record=_frec)
                    _resolve_refutation(_refut_store, _forced, _frec)
                    eg_visited.extend(s for (s, *_rest) in ex)
                    fresh.extend(tuple(e[1:]) for e in ex)  # drop the state prefix; aux fields (if any) ride along
                else:  # unchanged pure-#1 path (byte-identical when the loop is off)
                    _forced = (_refut_store.sample(rng)
                               if _refut_store is not None and refutation_frac > 0.0
                               and rng.random() < refutation_frac else None)
                    _frec = {} if _forced else None
                    ex = self_play_game(game, learner, rng, target_net=target_net, n_step=value_n_step,
                                        device=device, opening_plies=_game_plies(rng), record_aux=aux_on,
                                        forced_opening=list(_forced) if _forced else None, record=_frec)
                    _resolve_refutation(_refut_store, _forced, _frec)
                    fresh.extend(ex)
            fresh = augment_examples(fresh, perms)  # symmetry-augment self-play too (2× data, invariance baked in)
            buffer = (buffer + fresh)[-buffer_cap:]
        eg_booked = 0
        if endgame_on:  # RETROGRADE climb: prove the iteration's frontier positions backward from booked terminals
            _vfn = (lambda ss: _batched_net_values(net, game, ss, device)) if endgame_net_priority else None
            eg_booked = extend_endgame_frontier(game, endgame_tb, eg_visited, endgame_max_empty,
                                                endgame_extend_positions, endgame_extend_seconds, value_fn=_vfn)
        # ANCHOR: the solver-free opening anchor (league) is pinned at league_anchor_frac via the SAME fixed-fraction
        # mix the oracle distillation uses — so a few-hundred empty-board examples aren't diluted in a 400k buffer.
        if stopped:
            history.append({"iteration": it + 1, "stopped": True, **reanalyze_note})
            break
        _anchor = distilled + opening_anchor
        _afrac = league_anchor_frac if opening_anchor else distill_fraction
        train_set = _mix_training_set(buffer, _anchor, _afrac)
        t_train = time.time()
        loss = train_net(net, train_set, epochs, batch_size, lr, device, opt_state=_opt_state,
                         epoch_examples=epoch_cap)
        if reanalyze_note:
            per_epoch = epoch_cap if epoch_cap is not None else len(train_set)
            reanalyze_note.update({"train_examples": len(train_set), "epoch_examples": per_epoch,
                                   "steps": epochs * math.ceil(per_epoch / batch_size),
                                   "train_s": round(time.time() - t_train, 3)})
        # CHEAP per-iteration quality probe (one forward pass): the net's value on the standard opening. Connect 4
        # is a first-player WIN, so this should climb toward +1 — the live curve the #2-vs-#1 A/B compares.
        opening_value = round(net_value(net, game, game.initial_state(random.Random(seed)), device), 4)
        entry = {"iteration": it + 1, "examples": len(buffer), "vs_pool_games": vs_pool, "reanalyzed": reanalyzed,
                 "buffer": len(buffer), "distilled": len(distilled), "loss": loss, "opening_value": opening_value}
        entry.update(reanalyze_note)
        if record_self_agreement:
            entry.update(_self_agreement(net, game, state_buffer, device))
        if endgame_on:
            entry.update({"endgame_booked": eg_booked, "endgame_solves": learner.endgame_solves,
                          "endgame_hits": learner.endgame_hits, "endgame_total": len(endgame_tb)})
        history.append(entry)
        if log:
            eg_note = (f", endgame +{eg_booked}/{len(endgame_tb)} (solve {learner.endgame_solves} hit {learner.endgame_hits})"
                       if endgame_on else "")
            log(f"iter {it + 1}/{iterations}: buffer {len(buffer)} ({vs_pool} vs-pool, {reanalyzed} reanalyzed), "
                f"distilled {len(distilled)}, loss {loss:.3f}{eg_note}")
    if settle_epochs > 0 and iterations > 0 and not stopped:
        loss = train_net(net, train_set, settle_epochs, batch_size, lr if settle_lr is None else settle_lr, device,
                         opt_state=_opt_state, epoch_examples=epoch_cap, lr_end=settle_lr_final)
        entry = {"iteration": "settle", "epochs": settle_epochs, "lr_final": settle_lr_final,
                 "train_examples": len(train_set), "state_buffer": len(state_buffer), "loss": loss}
        if record_self_agreement:
            entry.update(_self_agreement(net, game, state_buffer, device))
        history.append(entry)
    if _pool is not None:
        _pool.close()
        _pool.join()
        import shutil

        shutil.rmtree(_tmpdir, ignore_errors=True)
    if _relabel_pool is not None:
        _relabel_pool.close()
        _relabel_pool.join()
        import shutil

        shutil.rmtree(_relabel_dir, ignore_errors=True)
    if return_buffer:
        return net, history, buffer
    return net, history


def average_checkpoints(paths: list[str], device: str = "cpu") -> Connect4Net:
    """Average the PARAMETERS of several checkpoints of the SAME architecture (SWA-style, §C.14).

    Why: run-to-run (training-seed) variance is the constraint that measurement cannot reduce — better instruments
    make the seed floor more precisely visible, not smaller. Averaging the tail of a run's checkpoints is the
    classic cheap variance reducer, and because we save a checkpoint every batch it can be applied POST-HOC to
    finished runs at zero training cost.

    Averages every floating-point entry in the state dict, INCLUDING BatchNorm running statistics (dropping them
    silently changes eval-time behaviour). Integer buffers such as `num_batches_tracked` are taken from the first
    checkpoint. NOTE: averaging BN running stats is the cheap approximation — the textbook SWA recipe recomputes
    them with a forward pass over data; if the averaged net underperforms its members, that is the first thing to
    try (the run's persisted replay buffer is the natural data source)."""
    if not paths:
        raise ValueError("no checkpoints to average")
    nets = [load_net(p, device) for p in paths]
    arch = nets[0].arch
    for p, n in zip(paths[1:], nets[1:]):
        if n.arch != arch:
            raise ValueError(f"cannot average mismatched arch: {p} has {n.arch}, expected {arch}")
    out = Connect4Net(**arch).to(device)
    sds = [n.state_dict() for n in nets]
    merged = {}
    for key, first in sds[0].items():
        if first.is_floating_point():
            acc = torch.zeros_like(first, dtype=torch.float64)
            for sd in sds:
                acc += sd[key].to(torch.float64)
            merged[key] = (acc / len(sds)).to(first.dtype)
        else:
            merged[key] = first.clone()
    out.load_state_dict(merged)
    out.eval()
    return out


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
