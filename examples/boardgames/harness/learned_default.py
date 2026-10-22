"""A learned default move (§3.6 N1): a small net reads a position — every cell as the mover sees it (own, the
opponent's, empty) and, for every action, whether it is safe, wins by force in two and loses by force in two
(harness.steady_state.Facts) — and plays the safe move it scores highest. It is trained to put its weight on the safe
moves that keep the win (minus the log of their share of its softmax over the safe moves), and its weights are paid
once, quantized to a few bits each with one 32-bit scale a tensor."""
from __future__ import annotations

import copy

import torch
from torch import nn

SCALE_BITS = 32


class Net(nn.Module):
    def __init__(self, n_in: int, hidden: int, n_out: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def n_features(game) -> int:
    return 3 * len(game.initial_state().board) + 3 * game.num_actions


def features(game, facts, state) -> list:
    """The cells as the mover sees them, three a cell, then three facts an action (zero for an unsafe one)."""
    own = game.current_player(state) + 1
    out = []
    for v in state.board:
        out += [int(v == own), int(v not in (0, own)), int(v == 0)]
    safe = set(facts.safe_moves(state))
    for a in range(game.num_actions):
        if a in safe:
            out += [1, int(facts._wins_in_two(state, a)), int(facts._loses_in_two(state, a))]
        else:
            out += [0, 0, 0]
    return out


def _masked(net: Net, x: torch.Tensor, safe: torch.Tensor) -> torch.Tensor:
    return net(x).masked_fill(~safe, float("-inf"))


def picks(net: Net, x: torch.Tensor, safe: torch.Tensor) -> torch.Tensor:
    """The safe action each row's net scores highest (the lowest on a tie)."""
    with torch.no_grad():
        return _masked(net, x, safe).argmax(dim=1)


def train(x: torch.Tensor, safe: torch.Tensor, won: torch.Tensor, hidden: int, epochs: int, batch: int, lr: float,
          seed: int) -> Net:
    if not (won & safe).any(dim=1).all():
        raise ValueError("every training position needs a safe move that keeps the win")
    torch.manual_seed(seed)
    net = Net(x.shape[1], hidden, safe.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    order = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for at in torch.randperm(len(x), generator=order).split(batch):
            log_p = torch.log_softmax(_masked(net, x[at], safe[at]), dim=1)
            loss = -torch.logsumexp(log_p.masked_fill(~(won[at] & safe[at]), float("-inf")), dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def quantized(net: Net, bits: int) -> Net:
    """A copy with every tensor's weights rounded to 2**(bits-1)-1 steps either side of zero, its largest at the end."""
    steps = 2 ** (bits - 1) - 1
    q = copy.deepcopy(net)
    with torch.no_grad():
        for p in q.parameters():
            top = p.abs().max()
            if top > 0:
                p.copy_((p / (top / steps)).round().clamp(-steps, steps) * (top / steps))
    return q


def net_bits(net: Net, bits: int) -> int:
    params = list(net.parameters())
    return sum(p.numel() for p in params) * bits + SCALE_BITS * len(params)
