"""The PROVE-IT-GOOD scorecard capability (harness/process_eval.py): the verdict is honest — only the EXACT
oracle earns "solved" — and the end-to-end assembly runs on the shipped harness."""
from harness.process_eval import process_scorecard, scorecard_verdict


def test_scorecard_verdict_only_the_exact_oracle_earns_solved():
    # verify_solved (exact) or the ladder's exact rung ⇒ solved; a depth-proxy win never claims perfection.
    assert scorecard_verdict({"verify_solved": {"solved": True}})[0] == "solved"
    assert scorecard_verdict({"ladder": {"frontier": "exact"}})[0] == "solved"
    # a strong PROXY result is "near-optimal", explicitly UNPROVEN (clears a deep rung, never loses, near-optimal moves)
    v, why = scorecard_verdict({"ladder": {"frontier": "depth-10"}, "oracle_match": 0.96, "p1": {"not_lost_rate": 0.97}})
    assert v == "near-optimal" and "unproven" in why
    assert scorecard_verdict({"ladder": {"frontier": "depth-6"}, "oracle_match": 0.85, "p1": {"not_lost_rate": 0.6}})[0] == "strong"
    assert scorecard_verdict({"ladder": {"frontier": "none"}, "oracle_match": 0.5, "p1": {"not_lost_rate": 0.4}})[0] == "developing"
    # EXACT-proven near-optimal: converts EVERY proven forced win vs the EXACT solver + mid-game optimal (not a proxy)
    v2, why2 = scorecard_verdict({"exact_forced_wins": {"rate": 1.0, "converted": 8, "total": 8}, "oracle_match": 1.0})
    assert v2 == "near-optimal (exact-proven)" and "EXACT solver" in why2


def test_process_scorecard_runs_end_to_end_and_yields_an_honest_card():
    # A fast smoke on a light (non-neural) competitor: the scorecard assembles the ladder + P1 conversion +
    # mid-game distance-to-optimal into a verdict, and NEVER claims "solved" without the exact gate.
    events: list[dict] = []
    card = process_scorecard(
        {"game": "connect4", "model": {"model_name": "heuristic", "label": "heuristic"},
         "games": 4, "ladder_depths": [6], "reference_depth": 6, "corpus": {"n": 6, "min_moves": 28, "seed": 1},
         "forced_win_roots": {"n": 0}, "exact": False},  # forced-win proof off for a fast smoke
        on_progress=events.append,
    )
    assert card["game"] == "connect4" and card["model"] == "heuristic"
    assert "ladder" in card and "frontier" in card["ladder"]
    assert set(card["p1"]) >= {"rate", "not_lost_rate"}
    assert 0.0 <= card["oracle_match"] <= 1.0
    assert card["verdict"] in {"solved", "near-optimal", "near-optimal (exact-proven)", "strong", "developing"}
    assert card["verify_solved"] is None if "verify_solved" in card else True  # exact off ⇒ no solved claim
    assert card["verdict"] != "solved"  # never solved without the exact oracle
    assert events[0]["phase"] == "start" and events[-1]["phase"] == "done"


def test_neural_model_at_returns_an_agent_per_sims_not_a_factory(tmp_path):
    # Guard: sim_scaling_curve calls the builder as model_factory(sims) and treats the RESULT as the agent, so
    # `_neural_model_at(spec)(sims)` must return an AGENT (with .act), a fresh one per call — not a factory.
    import torch
    from harness.neural import Connect4Net, save_net
    from harness.process_eval import _neural_model_at
    from games.connect4 import Connect4

    torch.manual_seed(0)
    weights = tmp_path / "net.pt"
    save_net(Connect4Net(), str(weights))
    at = _neural_model_at({"az_weights": str(weights), "az_gumbel": True}, Connect4())
    agent1, agent2 = at(8), at(32)
    assert hasattr(agent1, "act") and hasattr(agent2, "act")  # agents, not factories
    assert agent1 is not agent2 and agent1.sims == 8 and agent2.sims == 32 and agent1.gumbel is True
