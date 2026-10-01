"""§C.49 T10 — does the SOLVER-FREE process reach perfect play from the start on Connect-4? Pre-registered.

The process is T9's (h73/h74: the strategy tree instead of siblings), at the depth-10 oracle setup (standardised-
input conv-32, 20,616 params, h72), trained with the solver forbidden and stopped by the h69/h74 signal — the first
walk of its own first-player tree (through 10 plies) that reads zero disagreements with its own search. Only after
training does the solver look: it walks the stopped net's own first-player tree through `certify_depth` plies
against every reply (P-START).

A seed SUCCEEDS when its run stopped on full agreement AND the solver certifies the net it stopped at. SUPPORTED at
>= `support_at` successes, REFUTED at <= `refute_at`, INCONCLUSIVE between. A run that never stops fails even if its
final net happens to be certified: the process must know when it is done. Any integrity failure, or unreadable
evidence, is NOT_RUN."""
from __future__ import annotations

CONFIG = {"game": "connect4", "arch": {"channels": 32, "canonical_input": True}, "params": 20616, "selfplay": 48,
          "train_sims": 32, "opening_plies": 2, "opening_zero_frac": 0.5, "reanalyze_frac": 1.0, "reanalyze_sims": 200,
          "reanalyze_siblings": False, "steps_matched": False, "epochs": 6, "batch_size": 64, "lr": 1e-3,
          "buffer_cap": 50000, "gumbel": True, "c_scale": 0.1, "augment": False, "buffer_unique": True,
          "settle_epochs": 30, "settle_lr_final": 1e-5, "stop_on_agreement": True}
T10_CONFIG = {**CONFIG, "iterations": 60, "strategy_tree": {"player": 0, "depth": 10}, "relabel_workers": 8}
SPEC = {"config": T10_CONFIG, "seeds": tuple(range(361, 371)), "era": "9e2609b15600", "measurement_fp": "297228f7d9d2",
        "params": CONFIG["params"], "certify_depth": 10, "support_at": 8, "refute_at": 5}


def _row_problems(row: dict, spec: dict) -> list:
    cfg, where = spec["config"], f"seed {row.get('seed')}"
    problems = []
    if row.get("params") != spec["params"]:
        problems.append(f"{where}: built {row.get('params')} parameters, not {spec['params']}")
    cert = row.get("certificate")
    if not isinstance(cert, dict) or not isinstance(cert.get("certified"), bool):
        return problems + [f"{where}: no certificate verdict"]
    if cert.get("net_file_sha256") != row["net"]["file_sha256"]:
        problems.append(f"{where}: the certificate is for another net")
    if cert.get("horizon") != spec["certify_depth"]:
        problems.append(f"{where}: certified through {cert.get('horizon')} plies, not {spec['certify_depth']}")
    history = row["history"]
    last = history[-1]
    if row.get("stopped") != bool(last.get("stopped")):
        problems.append(f"{where}: the stop flag does not match the history")
    if last.get("stopped"):
        if last.get("tree_disagreements") != 0 or not last.get("tree_walked"):
            problems.append(f"{where}: stopped without a walk reading full agreement")
        if row.get("games_per_pass") != [cfg["selfplay"]] * (len(history) - 1):
            problems.append(f"{where}: not the registered games on every trained iteration")
    elif len(history) != cfg["iterations"] + 1 or last.get("iteration") != "settle" \
            or row.get("games_per_pass") != [cfg["selfplay"]] * cfg["iterations"] + [0]:
        problems.append(f"{where}: did not stop, yet did not run every iteration and settle")
    walks = [h for h in history[1:] if h.get("iteration") != "settle"]
    if any((h.get("siblings") or 0) != 0 for h in history if h.get("iteration") != "settle"):
        problems.append(f"{where}: siblings were added")
    if not all(isinstance(h.get("tree_walked"), int) and h["tree_walked"] > 0 for h in walks):
        problems.append(f"{where}: an iteration did not walk the tree")
    return problems


def integrity(evidence, spec: dict) -> list:
    if not isinstance(evidence, dict) or not isinstance(evidence.get("seeds"), list) \
            or not isinstance(evidence.get("config"), dict):
        return ["no seeds/config"]
    problems = []
    if evidence.get("training_fingerprint") != spec["era"]:
        problems.append(f"training era {evidence.get('training_fingerprint')}, registered {spec['era']}")
    if evidence.get("measurement_fingerprint") != spec["measurement_fp"]:
        problems.append(f"measurement code {evidence.get('measurement_fingerprint')}, registered "
                        f"{spec['measurement_fp']}")
    if evidence["config"] != {**spec["config"], "seeds": list(spec["seeds"]), "certify_depth": spec["certify_depth"]}:
        problems.append("config is not the registered recipe")
    if sorted(r.get("seed") for r in evidence["seeds"] if isinstance(r, dict)) != list(spec["seeds"]):
        problems.append(f"seeds are not the registered {list(spec['seeds'])}")
    for row in evidence["seeds"]:
        problems.extend(_row_problems(row, spec))
    return problems


def c4_report(evidence, spec: dict = SPEC) -> dict:
    """The pre-registered verdict, or NOT_RUN when the evidence is not the registered run."""
    try:
        problems = integrity(evidence, spec)
        if not problems:
            rows = sorted(evidence["seeds"], key=lambda r: r["seed"])
            ok = [r["stopped"] and r["certificate"]["certified"] for r in rows]
            successes = sum(ok)
            verdict = ("supported" if successes >= spec["support_at"] else
                       "refuted" if successes <= spec["refute_at"] else "inconclusive")
            return {"integrity": [], "verdict": verdict, "successes": successes, "n": len(rows),
                    "descriptives": {
                        "stopped": sum(1 for r in rows if r["stopped"]),
                        "stop_iterations": [r["history"][-1]["iteration"] for r in rows if r["stopped"]],
                        "false_stops": sum(1 for r in rows if r["stopped"] and not r["certificate"]["certified"]),
                        "certified_without_stopping": sum(1 for r in rows
                                                          if not r["stopped"] and r["certificate"]["certified"]),
                        "failures": [r["certificate"].get("failures") for r in rows],
                        "values": [r["certificate"].get("values") for r in rows]}}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError) as e:
        problems = [f"the evidence could not be read: {type(e).__name__}: {e}"]
    return {"integrity": problems, "verdict": "not_run"}
