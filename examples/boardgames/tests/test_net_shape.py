"""§C.21 Increment 1 — the net is board-shape- and action-count-aware, and Connect-4 is BYTE-IDENTICAL.

Three defects were verified live in the code (they never bit Connect-4, so the suite was green; each corrupts
silently the moment a second game lands):
  D1 `_policy_value` built its mask as torch.full((COLS,)) — 7 wide, whatever the game's num_actions.
  D2 the global-pool branch averaged/maxed over ALL cells, so a padded board's pad zeros dilute the aggregate.
  D3 ROWS,COLS=6,7 and Linear(...,COLS) heads were module constants — board dims and action count hardwired.
The fix threads board_shape / num_actions / valid_mask through the ARCH (persisted with the weights) with
defaults that reproduce the legacy net exactly, so the 306 existing checkpoints load unchanged. The goldens
below were computed with the pre-refactor code; if construction order or shape ever drifts they fail.
"""
import random
from pathlib import Path

import pytest
import torch

from harness.neural import (AlphaZeroAgent, Connect4Net, arch_for_game, augment_examples, encode, load_net,
                            net_value, save_net)
from harness.registry import resolve_game

CKPT = Path("checkpoints/scaled_runs/ctrl302_postfix_s0/ckpt_23.pt")


class _FakeGame:
    """A board of any shape with any action count — enough surface for encode() and _policy_value()."""
    name = "fake"
    num_players = 2

    def __init__(self, shape, num_actions, legal):
        self.board_shape = shape
        self.num_actions = num_actions
        self._legal = legal

    def current_player(self, state):
        return 0

    def observation(self, state, player):
        h, w = self.board_shape
        return [1.0 if i % 3 == 0 else (-1.0 if i % 3 == 1 else 0.0) for i in range(h * w)]

    def legal_actions(self, state):
        return list(self._legal)


def test_default_arch_is_byte_identical_to_legacy():
    torch.manual_seed(0)
    legacy = Connect4Net()
    torch.manual_seed(0)
    tower = Connect4Net(channels=64, blocks=4, residual=True, batchnorm=True, head_hidden=32,
                        global_pool=True, value_bins=21)
    x = torch.linspace(-1, 1, 2 * 6 * 7).reshape(1, 2, 6, 7)
    legacy.eval(); tower.eval()
    with torch.no_grad():
        lp, lv = legacy(x)
        tp, tv = tower(x)
    assert len(legacy.state_dict()) == 8 and len(tower.state_dict()) == 94
    assert lp[0, :3].tolist() == pytest.approx([0.029392480850219727, 0.1103089451789856, 0.10748609900474548], abs=1e-6)
    assert float(lv[0, 0]) == pytest.approx(-0.04646496, abs=1e-6)
    assert tp[0, :3].tolist() == pytest.approx([0.16367337107658386, 0.05983233451843262, 0.1233791932463646], abs=1e-6)
    assert float(tv[0, 0]) == pytest.approx(-0.00849554, abs=1e-6)
    assert legacy.arch["board_shape"] == [6, 7] and legacy.arch["num_actions"] == 7
    assert legacy.arch["valid_mask"] is None


def test_existing_checkpoint_loads_strictly_and_evaluates_unchanged():
    if not CKPT.exists():
        pytest.skip("trained checkpoint not present")
    game = resolve_game("connect4")
    net = load_net(str(CKPT), "cpu")
    assert net_value(net, game, game.initial_state(random.Random(0))) == pytest.approx(0.50067037, abs=1e-6)
    assert net.arch["board_shape"] == [6, 7] and net.arch["num_actions"] == 7


def test_net_builds_for_another_board_and_action_count():
    for kw in ({}, {"residual": True, "blocks": 1, "batchnorm": True, "head_hidden": 8}):
        net = Connect4Net(channels=8, board_shape=(8, 8), num_actions=65, **kw)
        p, v = net(torch.zeros(1, 2, 8, 8))
        assert p.shape == (1, 65) and v.shape == (1, 1)


def test_policy_mask_is_num_actions_wide():
    # D1: legal action ids beyond 6 must be scored, not truncated or IndexError'd.
    game = _FakeGame((8, 8), 65, legal=[0, 40, 64])
    agent = AlphaZeroAgent(Connect4Net(channels=4, board_shape=(8, 8), num_actions=65), sims=1)
    prior, value = agent._policy_value(game, state=None)
    assert set(prior) == {0, 40, 64}
    assert sum(prior.values()) == pytest.approx(1.0, abs=1e-6)
    assert -1.0 <= value <= 1.0


def test_policy_value_refuses_a_net_built_for_another_game():
    # A legacy 7-wide net handed a 65-action game must fail LOUDLY, not silently score the first 7 ids.
    game = _FakeGame((8, 8), 65, legal=[0, 40, 64])
    agent = AlphaZeroAgent(Connect4Net(channels=4), sims=1)
    with pytest.raises(ValueError, match="num_actions"):
        agent._policy_value(game, state=None)


def test_masked_pool_ignores_pad_cells():
    # D2, checked as the PROPERTY itself: the aggregate must not depend on pad cells. (A first version asserted
    # this on the net's value output and passed vacuously — the randomly-initialised value head was dead-ReLU on
    # the probe input, so nothing reached it either way; the mutation guard below is what caught that.)
    from harness.neural import _masked_pool

    h = torch.rand(2, 5, 4, 4)
    noisy = h.clone()
    noisy[..., 2:] += 50.0
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0]).repeat(4).reshape(1, 1, 4, 4)
    assert torch.allclose(_masked_pool(h, mask), _masked_pool(noisy, mask))
    assert not torch.allclose(_masked_pool(h, None), _masked_pool(noisy, None))  # mutation guard: unmasked leaks
    valid = h[..., :2]
    assert torch.allclose(_masked_pool(h, mask), torch.cat([valid.mean(dim=(2, 3)), valid.amax(dim=(2, 3))], 1),
                          atol=1e-6)


def test_unmasked_pool_is_the_legacy_path():
    from harness.neural import _masked_pool

    h = torch.rand(2, 5, 6, 7)
    assert torch.equal(_masked_pool(h, None), torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], 1))


def test_net_routes_its_valid_mask_into_every_block(monkeypatch):
    # Wiring: a net built with a mask hands THAT mask to the pool; a net without one hands None (legacy path).
    import harness.neural as neural

    seen = []
    real = neural._masked_pool
    monkeypatch.setattr(neural, "_masked_pool", lambda h, m: (seen.append(m), real(h, m))[1])
    mask = [1, 1, 0, 0] * 4
    for valid_mask, blocks in ((mask, 2), (None, 2)):
        seen.clear()
        net = Connect4Net(channels=4, blocks=blocks, residual=True, global_pool=True, board_shape=(4, 4),
                          num_actions=16, valid_mask=valid_mask).eval()
        with torch.no_grad():
            net(torch.zeros(1, 2, 4, 4))
        assert len(seen) == blocks
        if valid_mask is None:
            assert all(m is None for m in seen)
        else:
            assert all(torch.equal(m, net.valid_mask) for m in seen)


def test_valid_mask_roundtrips_through_save_load(tmp_path):
    mask = [1, 1, 0, 0] * 4
    net = Connect4Net(channels=4, blocks=1, residual=True, global_pool=True, board_shape=(4, 4),
                      num_actions=16, valid_mask=mask)
    save_net(net, str(tmp_path / "n.pt"))
    loaded = load_net(str(tmp_path / "n.pt"))
    assert loaded.arch == net.arch
    assert loaded.valid_mask.flatten().tolist() == [float(m) for m in mask]


def test_encode_follows_the_games_board_shape():
    game = _FakeGame((3, 4), 12, legal=[0])
    assert tuple(encode(game, None).shape) == (2, 3, 4)
    c4 = resolve_game("connect4")
    assert tuple(encode(c4, c4.initial_state(random.Random(0))).shape) == (2, 6, 7)


def test_augment_reshapes_aux_by_perm_width_not_module_constant():
    # D3 in the augmenter: ownership planes were reshaped to (6,7) whatever the board — a 3x4 board crashed.
    own = torch.arange(12, dtype=torch.float32)  # 3 rows x 4 cols
    x = torch.zeros(2, 3, 4)
    mirror = [3, 2, 1, 0]
    out = augment_examples([(x, [0.25] * 4, 0.0, own, 1)], [list(range(4)), mirror])
    assert len(out) == 2
    own_m = out[1][3]
    assert own_m.tolist() == own.reshape(3, 4)[:, mirror].reshape(-1).tolist()
    assert out[1][4] == mirror.index(1)


def test_arch_for_game_fills_shape_and_actions_from_the_game():
    oth = resolve_game("othello")
    arch = arch_for_game({"channels": 8}, oth)
    assert arch["board_shape"] == [8, 8] and arch["num_actions"] == 65 and arch["channels"] == 8
    c4 = resolve_game("connect4")
    assert arch_for_game(None, c4) == {"board_shape": [6, 7], "num_actions": 7}
    # a Connect-4 config that never mentioned shape builds the identical legacy net
    assert Connect4Net(**arch_for_game({"channels": 32}, c4)).arch == Connect4Net().arch


def test_arch_for_game_refuses_a_config_for_the_wrong_game():
    oth = resolve_game("othello")
    with pytest.raises(ValueError, match="num_actions"):
        arch_for_game({"num_actions": 7}, oth)
    with pytest.raises(ValueError, match="board_shape"):
        arch_for_game({"board_shape": [6, 7]}, oth)
