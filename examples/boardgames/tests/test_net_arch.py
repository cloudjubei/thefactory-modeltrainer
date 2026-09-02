"""§C.7 capacity levers — the config-driven net: legacy shape stays the default (old checkpoints + tests unaffected),
residual/BN/head-tower scale it to the ~1.5-2M-param reproduction floor, and save/load round-trips the architecture."""
import torch

from games.connect4 import Connect4
from harness.neural import Connect4Net, encode, load_net, save_net


def _nparams(net):
    return sum(p.numel() for p in net.parameters())


def test_legacy_net_is_the_default_and_unchanged():
    net = Connect4Net()  # no args ⇒ today's 2-conv, bare-linear-head net (so 306 old checkpoints still load)
    assert hasattr(net, "conv1") and hasattr(net, "conv2")
    assert hasattr(net, "policy_head") and hasattr(net, "value_head")
    assert net.arch["residual"] is False
    assert 20000 <= _nparams(net) <= 21000  # ~20.6K — the measured under-capacity net
    g = Connect4()
    x = encode(g, g.initial_state(__import__("random").Random(0))).unsqueeze(0)
    p, v = net(x)
    assert p.shape == (1, 7) and v.shape == (1, 1)
    assert -1.0 <= float(v[0, 0]) <= 1.0


def test_residual_net_reaches_the_reproduction_floor():
    net = Connect4Net(channels=128, blocks=6, residual=True, batchnorm=True, head_hidden=64)
    assert net.arch["residual"] is True and net.arch["blocks"] == 6
    assert _nparams(net) >= 1_000_000  # >= ~1M params — the strong-C4 floor, not the 20K toy
    g = Connect4()
    x = encode(g, g.initial_state(__import__("random").Random(1))).unsqueeze(0)
    net.eval()
    p, v = net(x)
    assert p.shape == (1, 7) and v.shape == (1, 1)
    assert -1.0 <= float(v[0, 0]) <= 1.0


def test_save_load_roundtrips_the_architecture(tmp_path):
    net = Connect4Net(channels=64, blocks=3, residual=True, batchnorm=True, head_hidden=32)
    net.eval()
    g = Connect4()
    x = encode(g, g.initial_state(__import__("random").Random(2))).unsqueeze(0)
    with torch.no_grad():
        p0, v0 = net(x)
    path = str(tmp_path / "scaled.pt")
    save_net(net, path)
    back = load_net(path)  # must reconstruct the SAME architecture, not the legacy default
    assert back.arch == net.arch
    back.eval()
    with torch.no_grad():
        p1, v1 = back(x)
    assert torch.allclose(p0, p1, atol=1e-5) and torch.allclose(v0, v1, atol=1e-5)


def test_old_format_checkpoint_loads_as_legacy(tmp_path):
    # Simulate a pre-levers checkpoint blob ({state_dict, channels} with NO arch field): must load as the legacy net.
    legacy = Connect4Net()
    path = str(tmp_path / "old.pt")
    torch.save({"state_dict": legacy.state_dict(), "channels": 32}, path)
    back = load_net(path)
    assert back.arch["residual"] is False and back.arch["channels"] == 32
    assert hasattr(back, "conv1")


def test_residual_value_head_has_a_real_nonlinearity():
    # The dominant capacity gap was the BARE linear value head; the scaled value tower must carry a hidden layer.
    net = Connect4Net(channels=32, blocks=2, residual=True, batchnorm=False, head_hidden=32)
    linears = [m for m in net.value_head.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) >= 2  # hidden + output ⇒ can represent the nonlinear AND-of-threats / parity


def test_global_pool_defaults_off_and_legacy_arch_unchanged():
    # §C.8 lever #2 is OPT-IN: the default arch must stay byte-identical (old checkpoints, running screens).
    net = Connect4Net(channels=32, blocks=2, residual=True, batchnorm=True, head_hidden=32)
    assert net.arch["global_pool"] is False
    assert all(getattr(b, "gpool_fc", None) is None for b in net.blocks)
    legacy = Connect4Net()
    assert legacy.arch["global_pool"] is False and legacy.arch["value_bins"] == 0


def test_global_pool_branch_exists_gets_gradient_and_roundtrips(tmp_path):
    # KataGo-style whole-board conditioning (§C.8 #2): every block carries a mean+max pool → FC → per-channel
    # bias path, it must actually TRAIN (gradient flows), and the arch must round-trip through save/load.
    net = Connect4Net(channels=16, blocks=2, residual=True, batchnorm=True, head_hidden=8, global_pool=True)
    assert net.arch["global_pool"] is True
    for b in net.blocks:
        assert b.gpool_fc is not None and b.gpool_fc.out_features == 16 and b.gpool_fc.in_features == 32
    g = Connect4()
    # Probe with NON-EMPTY positions at batch 2: on the all-zero empty board BatchNorm zeroes every constant
    # activation, so block-0's pooled input (and thus its weight grad) is legitimately 0 — an input artifact.
    rng = __import__("random").Random(3)
    states = []
    for _ in range(2):
        s = g.initial_state(rng)
        for _ in range(4):
            s = g.step(s, rng.choice(g.legal_actions(s)))
        states.append(s)
    x = torch.stack([encode(g, s) for s in states])
    p, v = net(x)
    assert p.shape == (2, 7) and v.shape == (2, 1) and all(-1.0 <= float(vi) <= 1.0 for vi in v[:, 0])
    (p.sum() + v.sum()).backward()
    grads = [b.gpool_fc.weight.grad for b in net.blocks]
    assert all(gr is not None and float(gr.abs().sum()) > 0 for gr in grads)  # the pool path is live, not dead
    net.eval()
    with torch.no_grad():
        p0, v0 = net(x)
    path = str(tmp_path / "gpool.pt")
    save_net(net, path)
    back = load_net(path)
    assert back.arch == net.arch and back.blocks[0].gpool_fc is not None
    with torch.no_grad():
        p1, v1 = back(x)
    assert torch.allclose(p0, p1, atol=1e-5) and torch.allclose(v0, v1, atol=1e-5)


def test_value_bins_head_returns_scalar_expectation_and_roundtrips(tmp_path):
    # §C.8 lever #3: categorical value head. CONSUMERS stay untouched — forward still returns a scalar in
    # [-1, 1] (the expectation over the bin support), so MCTS/eval/probes need no changes.
    net = Connect4Net(channels=16, blocks=2, residual=True, batchnorm=True, head_hidden=8, value_bins=9)
    assert net.arch["value_bins"] == 9
    assert net.value_support.shape == (9,)
    assert float(net.value_support[0]) == -1.0 and float(net.value_support[-1]) == 1.0
    g = Connect4()
    x = encode(g, g.initial_state(__import__("random").Random(4))).unsqueeze(0)
    net.eval()
    p, v = net(x)
    assert p.shape == (1, 7) and v.shape == (1, 1) and -1.0 <= float(v[0, 0]) <= 1.0
    logits_p, bin_logits = net.forward_train(x)
    assert bin_logits.shape == (1, 9)  # the TRAIN path exposes the distribution for cross-entropy
    with torch.no_grad():
        p0, v0 = net(x)
    path = str(tmp_path / "bins.pt")
    save_net(net, path)
    back = load_net(path)
    assert back.arch == net.arch
    with torch.no_grad():
        p1, v1 = back(x)
    assert torch.allclose(p0, p1, atol=1e-5) and torch.allclose(v0, v1, atol=1e-5)


def test_value_bins_on_the_legacy_tower_too():
    # The lever is arch-orthogonal: a non-residual net with bins still returns a scalar expectation.
    net = Connect4Net(channels=8, value_bins=5)
    g = Connect4()
    x = encode(g, g.initial_state(__import__("random").Random(5))).unsqueeze(0)
    p, v = net(x)
    assert v.shape == (1, 1) and -1.0 <= float(v[0, 0]) <= 1.0
    _p, bl = net.forward_train(x)
    assert bl.shape == (1, 5)


def test_forward_train_matches_forward_for_scalar_nets():
    # With bins OFF, forward_train IS forward — one train path, no drift between the two.
    net = Connect4Net(channels=16, blocks=1, residual=True, batchnorm=False, head_hidden=8)
    net.eval()
    g = Connect4()
    x = encode(g, g.initial_state(__import__("random").Random(6))).unsqueeze(0)
    with torch.no_grad():
        p0, v0 = net(x)
        p1, v1 = net.forward_train(x)
    assert torch.allclose(p0, p1) and torch.allclose(v0, v1)


def test_aux_heads_default_off_and_require_the_residual_tower():
    # §C.8 #4 aux heads are OPT-IN (default net byte-identical) and read SPATIAL trunk features — scaled tower only.
    import pytest

    net = Connect4Net(channels=16, blocks=1, residual=True, batchnorm=False, head_hidden=8)
    assert net.arch["aux_heads"] is False and not hasattr(net, "own_head")
    with pytest.raises(ValueError):
        Connect4Net(channels=8, aux_heads=True)  # legacy tower has no aux seam


def test_aux_heads_forward_aux_shapes_and_roundtrip(tmp_path):
    # forward stays (p, v) — consumers untouched; forward_aux adds ownership map (tanh, per-cell) + reply logits.
    net = Connect4Net(channels=16, blocks=1, residual=True, batchnorm=False, head_hidden=8, aux_heads=True)
    assert net.arch["aux_heads"] is True
    g = Connect4()
    x = encode(g, g.initial_state(__import__("random").Random(7))).unsqueeze(0)
    net.eval()
    p, v = net(x)
    assert p.shape == (1, 7) and v.shape == (1, 1)
    p2, v2, own, reply = net.forward_aux(x)
    assert torch.allclose(p, p2) and torch.allclose(v, v2)
    assert own.shape == (1, 42) and float(own.abs().max()) <= 1.0
    assert reply.shape == (1, 7)
    path = str(tmp_path / "aux.pt")
    save_net(net, path)
    back = load_net(path)
    assert back.arch == net.arch and hasattr(back, "own_head") and hasattr(back, "reply_head")
