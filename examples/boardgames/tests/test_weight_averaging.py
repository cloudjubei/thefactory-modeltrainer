"""§C.14 WEIGHT AVERAGING — reduce RUN-TO-RUN variance, the constraint measurement cannot fix.

Measured 2026-09-05: two identical configs differing only by training seed land 0.036 apart (per-run SD ~0.026),
against an architecture effect of 0.060 we want to resolve. Averaging the tail of a run's checkpoints is the
classic variance reducer and — crucially — can be tested POST-HOC on checkpoints we already have, at zero
training cost.
"""
import random

import pytest
import torch

from games.connect4 import Connect4
from harness.neural import Connect4Net, average_checkpoints, encode, load_net, save_net


def _mk(tmp_path, n, arch, seed0=0):
    paths = []
    for i in range(n):
        torch.manual_seed(seed0 + i)
        net = Connect4Net(**arch)
        p = str(tmp_path / f"ckpt_{i}.pt")
        save_net(net, p)
        paths.append(p)
    return paths


def test_average_is_the_arithmetic_mean_of_parameters(tmp_path):
    arch = {"channels": 8}
    paths = _mk(tmp_path, 3, arch)
    nets = [load_net(p) for p in paths]
    avg = average_checkpoints(paths)
    assert avg.arch == nets[0].arch                       # architecture preserved
    for key, param in avg.state_dict().items():
        if not param.is_floating_point():
            continue
        want = sum(n.state_dict()[key].float() for n in nets) / 3
        assert torch.allclose(param.float(), want, atol=1e-6), key


def test_averaging_one_checkpoint_is_identity(tmp_path):
    paths = _mk(tmp_path, 1, {"channels": 8})
    a, b = load_net(paths[0]), average_checkpoints(paths)
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k])


def test_averaged_net_runs_and_stays_in_range(tmp_path):
    arch = {"channels": 16, "blocks": 2, "residual": True, "batchnorm": True, "head_hidden": 8}
    avg = average_checkpoints(_mk(tmp_path, 4, arch))
    g = Connect4()
    x = encode(g, g.initial_state(random.Random(0))).unsqueeze(0)
    avg.eval()
    p, v = avg(x)
    assert p.shape == (1, 7) and v.shape == (1, 1) and -1.0 <= float(v[0, 0]) <= 1.0


def test_refuses_to_average_mismatched_architectures(tmp_path):
    a = _mk(tmp_path, 1, {"channels": 8})
    torch.manual_seed(9)
    p2 = str(tmp_path / "other.pt")
    save_net(Connect4Net(channels=16), p2)
    with pytest.raises(ValueError, match="(?i)arch"):
        average_checkpoints(a + [p2])


def test_batchnorm_running_stats_are_averaged_not_dropped(tmp_path):
    # BatchNorm buffers (running_mean/var) must be carried through — dropping them silently changes the net's
    # eval-time behaviour, which is exactly the kind of quiet corruption an averaging helper can introduce.
    arch = {"channels": 8, "blocks": 1, "residual": True, "batchnorm": True, "head_hidden": 4}
    paths = _mk(tmp_path, 3, arch)
    for i, p in enumerate(paths):          # give each a DISTINCT running_mean so averaging is observable
        blob = torch.load(p, map_location="cpu")
        blob["state_dict"]["stem_bn.running_mean"] = torch.full((8,), float(i + 1))
        torch.save(blob, p)
    avg = average_checkpoints(paths)
    assert torch.allclose(avg.state_dict()["stem_bn.running_mean"], torch.full((8,), 2.0), atol=1e-6)
    assert "num_batches_tracked" in " ".join(avg.state_dict().keys())
