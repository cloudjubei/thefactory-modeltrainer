"""§C.46 post-hoc — score an arm's final nets in EVERY orientation. The evidence of §C.41-§C.46 scored the raw policy
on one canonical image per state; a net trained with augmentation is not exactly symmetric, so that reading can
call a net perfect that misplays a rotated board. This retrains each seed of an evidence file deterministically,
REFUSES any net whose weights hash differs from the final-pass `weights_sha` the evidence recorded (so the net
scored is provably the net that was measured), and scores it three ways with `coverage.orientation_failures`.

    PYTHONPATH=. .venv/bin/python scripts/orientation_evidence.py --evidence evidence/c46_R200S.json.gz \\
        --out evidence/c46_R200S_orientation.json.gz
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

MODULES = ("scripts/orientation_evidence.py", "harness/coverage.py", "harness/symmetry.py", "harness/targets.py")


def score_seed(job: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    from harness.coverage import orientation_failures
    from harness.neural import encode, train_alphazero
    from harness.registry import resolve_game
    from harness.symmetry import verified_isometries
    from harness.targets import POLICY_TARGETS, _weights_sha

    cfg = job["config"]
    game = resolve_game(cfg["game"])
    knobs: dict = {}
    if cfg.get("reanalyze_frac"):
        knobs.update(reanalyze_frac=cfg["reanalyze_frac"], reanalyze_sims=cfg["reanalyze_sims"])
    if cfg.get("reanalyze_siblings"):
        knobs.update(reanalyze_siblings=True, steps_matched=bool(cfg.get("steps_matched")),
                     sibling_holdout=cfg.get("sibling_holdout"))
    if cfg.get("policy_target"):
        knobs["policy_target_fn"] = POLICY_TARGETS[cfg["policy_target"]]
    net, _history = train_alphazero(
        game, iterations=cfg["iterations"], selfplay_games=cfg["selfplay"], sims=cfg["train_sims"],
        channels=cfg["arch"]["channels"], net_arch=cfg["arch"], augment=True, gumbel=True, seed=job["seed"],
        selfplay_opening_plies=cfg["opening_plies"], opening_plies_zero_frac=cfg["opening_zero_frac"],
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"], buffer_cap=cfg["buffer_cap"], **knobs)
    sha = _weights_sha(net)
    if sha != job["weights_sha"]:
        return {"seed": job["seed"], "reproduced": False, "weights_sha": sha, "expected": job["weights_sha"]}
    net.eval()

    def logits_fn(state):
        with torch.no_grad():
            lg, _v = net(encode(game, state).unsqueeze(0))
        return [float(x) for x in lg[0]]
    return {"seed": job["seed"], "reproduced": True, "weights_sha": sha,
            **orientation_failures(game, logits_fn, [iso for iso, _f in verified_isometries(game)])}


def main() -> None:
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    ev = load_evidence(args.evidence)
    if training_fingerprint(ev["config"]["game"]) != ev["training_fingerprint"]:
        raise SystemExit("the training code moved since this evidence was produced — its nets cannot be rebuilt")
    jobs = [{"config": ev["config"], "seed": s["seed"],
             "weights_sha": max(s["passes"], key=lambda p: p["pass"])["weights_sha"]} for s in ev["seeds"]]
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(score_seed, jobs))
    bad = [r for r in rows if not r["reproduced"]]
    if bad:
        raise SystemExit(f"seeds {[r['seed'] for r in bad]} did not rebuild the recorded net — refusing to score a "
                         f"different net")
    canon = {s["seed"]: sorted(s["policy_fail_keys"]) for s in ev["seeds"]}
    drift = [r["seed"] for r in rows if r["canonical_fail_keys"] != canon[r["seed"]]]
    if drift:
        raise SystemExit(f"seeds {drift}: the rebuilt net's canonical failures differ from the evidence")
    out = {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "evidence": args.evidence, "arm_started": ev["started"], "training_fingerprint": ev["training_fingerprint"],
           "orientation_fingerprint": training_fingerprint(modules=MODULES), "seeds": rows}
    save_evidence(args.out, out)
    for r in rows:
        print(f"seed {r['seed']}: canonical {len(r['canonical_fail_keys'])}  images {r['failing_images']} over "
              f"{len(r['image_fail_keys'])} keys  symmetrized {len(r['symmetrized_fail_keys'])}")
    n = len(rows)
    print(f"perfect seeds — canonical {sum(not r['canonical_fail_keys'] for r in rows)}/{n}, every image "
          f"{sum(not r['image_fail_keys'] for r in rows)}/{n}, symmetrized {sum(not r['symmetrized_fail_keys'] for r in rows)}/{n}")


if __name__ == "__main__":
    main()
