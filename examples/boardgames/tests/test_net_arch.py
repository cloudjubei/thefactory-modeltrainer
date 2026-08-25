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
