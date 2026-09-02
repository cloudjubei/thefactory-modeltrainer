"""§C.7 batched/resumable scaled-run driver — a checkpoint + metrics per batch, and RESUME picks up where it left off."""
import json
from pathlib import Path

from harness.scaled_run import run_scaled_experiment

ARCH = {"channels": 16, "blocks": 1, "residual": True, "batchnorm": True, "head_hidden": 8}


def _req(run_dir, batches):
    return {"game": "connect4", "run_dir": str(run_dir), "net_arch": ARCH, "seed": 0, "sims": 6,
            "iters_per_batch": 1, "games": 3, "epochs": 1, "buffer_cap": 2000, "opening_plies": 2,
            "batches": batches, "benchmark_positions": 6, "offline_openings": 3}


def test_batches_checkpoint_and_resume(tmp_path):
    run_dir = Path(tmp_path) / "run"
    # Batch 0 only.
    r1 = run_scaled_experiment(_req(run_dir, batches=1))
    assert (run_dir / "ckpt_0.pt").exists()
    assert len(r1["metrics"]) == 1 and r1["metrics"][0]["batch"] == 0
    m0 = r1["metrics"][0]
    assert "opening_value" in m0 and "oracle_optimality_rate" in m0 and "offline" in m0
    assert "p1_loss" in m0["offline"]
    lines1 = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(lines1) == 1

    # RESUME to batch 2: must SKIP batch 0 (checkpoint present) and only run batches 1..1.
    r2 = run_scaled_experiment(_req(run_dir, batches=2))
    assert (run_dir / "ckpt_1.pt").exists()
    batches_seen = [m["batch"] for m in r2["metrics"]]
    assert batches_seen == [0, 1]  # batch 0 read back from metrics.jsonl, batch 1 freshly trained
    lines2 = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(lines2) == 2  # exactly one NEW line appended on resume (batch 0 not re-run)
    # iterations_done is cumulative across batches
    assert json.loads(lines2[-1])["iterations_done"] == 2


def test_scaled_run_league_routes_games_and_off_is_pure(tmp_path):
    from harness.scaled_run import run_scaled_experiment
    arch = {"channels": 16, "blocks": 1, "residual": True, "batchnorm": True, "head_hidden": 8}
    base = {"game": "connect4", "net_arch": arch, "seed": 0, "sims": 6, "iters_per_batch": 1, "games": 8,
            "epochs": 1, "buffer_cap": 2000, "opening_plies": 2, "batches": 1, "benchmark_positions": 4,
            "offline_openings": 2}
    # league ON: some games routed to the solver-free pool
    on = run_scaled_experiment({**base, "run_dir": str(tmp_path / "on"), "league": True, "league_frac": 1.0,
                                "league_snapshots": 0, "league_frozen_self": True})
    assert on["metrics"][0]["league_vs_pool"] > 0
    # league OFF: pure self-play, no pool games
    off = run_scaled_experiment({**base, "run_dir": str(tmp_path / "off")})
    assert off["metrics"][0]["league_vs_pool"] == 0


def test_gate_promotes_only_on_a_strictly_better_probe(tmp_path):
    # §C.8 #6 promotion gate: champion.json/champion.pt track the BEST probe rate ever seen and never regress
    # (the push run's b23→b39 slide is exactly what this prevents).
    import torch

    from harness.neural import Connect4Net, save_net
    from harness.scaled_run import update_gate

    run_dir = Path(tmp_path)
    net_a, net_b = Connect4Net(channels=8), Connect4Net(channels=8)
    assert update_gate(run_dir, batch=0, rate=0.5, net=net_a) is True  # first probe always crowns
    champ = json.loads((run_dir / "champion.json").read_text())
    assert champ == {"batch": 0, "rate": 0.5} and (run_dir / "champion.pt").exists()
    assert update_gate(run_dir, batch=1, rate=0.5, net=net_b) is False  # tie does NOT overwrite
    assert update_gate(run_dir, batch=2, rate=0.4, net=net_b) is False  # worse does NOT overwrite
    assert json.loads((run_dir / "champion.json").read_text())["batch"] == 0
    assert update_gate(run_dir, batch=3, rate=0.75, net=net_b) is True  # strictly better crowns
    assert json.loads((run_dir / "champion.json").read_text()) == {"batch": 3, "rate": 0.75}


def test_gate_probe_runs_inside_the_batch_loop_when_requested(tmp_path, monkeypatch):
    # gate_roots > 0 wires a per-batch forced-win probe into run_scaled_experiment; 0 (default) stays off.
    import harness.scaled_run as sr

    calls = []
    monkeypatch.setattr(sr, "gate_probe", lambda game, net, roots, sims: calls.append(roots) or 0.5)
    run_dir = tmp_path / "gated"
    r = run_scaled_experiment(_req(run_dir, batches=1) | {"gate_roots": 4})
    assert calls == [4]
    assert r["metrics"][0]["gate_rate"] == 0.5
    assert json.loads((run_dir / "champion.json").read_text()) == {"batch": 0, "rate": 0.5}
    # default OFF: no probe, no champion artifacts
    calls.clear()
    r2 = run_scaled_experiment(_req(tmp_path / "ungated", batches=1))
    assert calls == [] and "gate_rate" not in r2["metrics"][0]
    assert not (tmp_path / "ungated" / "champion.json").exists()


def test_reanalyze_and_priority_knobs_reach_train_alphazero(tmp_path, monkeypatch):
    # §C.8 B8 + C10 exposure: request knobs must flow through to train_alphazero (default off = absent/0.0).
    import harness.scaled_run as sr
    from harness.neural import Connect4Net

    seen = {}

    def fake_train(game, **kw):
        seen.update(kw)
        return Connect4Net(**kw["net_arch"]), [{"loss": 1.0, "vs_pool_games": 0}], []

    monkeypatch.setattr(sr, "train_alphazero", fake_train)
    run_scaled_experiment(_req(tmp_path / "r1", batches=1) | {"reanalyze_frac": 0.25, "endgame_net_priority": 1})
    assert seen["reanalyze_frac"] == 0.25 and seen["endgame_net_priority"] is True
    assert seen["opening_plies_zero_frac"] == 0.0  # mixed-openings default OFF
    seen.clear()
    run_scaled_experiment(_req(tmp_path / "r2", batches=1) | {"opening_plies_zero_frac": 0.5})
    assert seen["reanalyze_frac"] == 0.0 and seen["endgame_net_priority"] is False
    assert seen["opening_plies_zero_frac"] == 0.5
    assert seen["refutation_frac"] == 0.0 and seen["refutation_store"] is None  # §C.8 #5 default OFF
    seen.clear()
    run_scaled_experiment(_req(tmp_path / "r3", batches=1)
                          | {"refutation_frac": 0.2, "refutation_prefix_plies": 4,
                             "league": True, "league_snapshots": 0})  # league required — nogoods come from it
    assert seen["refutation_frac"] == 0.2 and seen["refutation_prefix_plies"] == 4
    assert seen["refutation_store"] is not None  # run-owned store, persisted per batch for resume
    assert (tmp_path / "r3" / "refutations.json").exists()


def test_gate_probe_seed_is_held_out_from_the_measurement_seed():
    # Review-confirmed HIGH bug guard: with a shared seed the gate's roots were a SUBSET of the final n=32
    # scorecard's roots — the champion would be max-selected on 37.5% of the exact final measurement.
    from harness.scaled_run import GATE_PROBE_SEED

    assert GATE_PROBE_SEED != 7  # 7 = the standing measurement seed (verify_forced_win_conversion scorecards)


def test_json_helpers_are_atomic_and_tolerant(tmp_path):
    # Review-confirmed (FileStorage incident class): a kill mid-write must never leave truncated JSON that
    # crashes the next resume — writes go temp+rename, reads degrade corrupt files to 'absent'.
    from harness.scaled_run import _read_json_tolerant, _write_json_atomic

    p = tmp_path / "state.json"
    _write_json_atomic(p, {"a": 1})
    assert _read_json_tolerant(p) == {"a": 1}
    assert not p.with_suffix(".json.tmp").exists()  # no droppings
    p.write_text('{"a": 1')  # truncated (simulated mid-write kill)
    assert _read_json_tolerant(p) is None
    from harness.neural import Connect4Net
    from harness.scaled_run import update_gate

    (tmp_path / "champion.json").write_text("{corrupt")
    assert update_gate(tmp_path, batch=1, rate=0.5, net=Connect4Net(channels=8)) is True  # degrades, re-crowns


def test_refutation_without_league_is_refused(tmp_path):
    # Review-confirmed: nogoods only come from league losses — league off + refutation on is a dead config.
    import pytest

    with pytest.raises(ValueError):
        run_scaled_experiment(_req(tmp_path / "dead", batches=1) | {"refutation_frac": 0.2})
