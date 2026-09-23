"""§C.46 G0 CAPACITY GATE — can the net FIT the exact tic-tac-toe policy in the steps a run gives it?

Every §C.46 arm changes what a net is TAUGHT (sibling exposure, deeper labels). That question is empty if the net
cannot represent the exact policy at the recipe's step budget: a treatment that "does not help" would then be the
net's ceiling, not the labels'. So before any arm is run, a fresh net is trained OFFLINE on the whole answer —
every canonical non-terminal state with its symmetric images, the policy target uniform over the solver's
optimal set, the value target the exact game value to the mover — and its RAW policy (no search) is graded on
every canonical state.

Registered (spec §4 G0): residual PASSES iff at least 4 of torch seeds 61-65 reach 0 raw-policy failures after 62
epochs (62 x 79 batches = 4,898 steps, R200's budget). Legacy runs at 62, 124 and 186 epochs, descriptive only.

    PYTHONPATH=. .venv/bin/python scripts/capacity_gate.py --arch residual --epochs 62 --seeds 61-65 \\
        --out evidence/c46_G0_capacity.json
    PYTHONPATH=. .venv/bin/python scripts/capacity_gate.py --arch legacy --epochs 62,124,186 --seeds 61-65 \\
        --out evidence/c46_G0_capacity.json

A later invocation MERGES into an existing --out (a run with the same arch, epochs and seed is replaced) and is
refused when the file was written under other training or measurement code, or from another supervised set.

WHY the 627 canonical states x 8 images, not the 4,520 raw positions: it is how the recipe itself augments —
every training row is multiplied by the whole verified symmetry group, so a position with a symmetric board is
repeated — and it is the 5,016-row set whose 79 batches per epoch make 62 epochs the registered 4,898 steps. The
raw space is CHECKED instead of trained on: every raw position must appear among the images, each image with its
own exact label, so a wrong action permutation cannot teach the mirror's move."""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

GAME = "tictactoe"
ARCHS = {"residual": {"channels": 32, "blocks": 3, "head_hidden": 32, "residual": True},
         "legacy": {"channels": 32}}
PARAMS_EXPECTED = {"residual": 57453, "legacy": 12746}
CANONICAL_STATES = 627
IMAGES = 8
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
GATE = {"arch": "residual", "epochs": 62, "seeds": [61, 62, 63, 64, 65], "min_zero_failure_seeds": 4}
MEASUREMENT_MODULES = ("scripts/capacity_gate.py", "harness/coverage.py", "harness/symmetry.py")


def _parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def _parse_epochs(text: str) -> list[int]:
    out = [int(x) for x in text.split(",")]
    if any(e < 1 for e in out):
        raise ValueError(f"every epoch budget must be at least 1, got {out}")
    return out


def exact_rows(game, states: list) -> list:
    """One (encoding, policy, value) row per state with the labels a PERFECT relabeller would write: the policy
    uniform over the solver's optimal set, the value the exact game value to the mover. Whatever the net then gets
    wrong is the net's, not the label's."""
    from harness.coverage import optimal_actions
    from harness.neural import encode

    rows = []
    for s in states:
        opt = optimal_actions(game, s)
        rows.append((encode(game, s), [1.0 / len(opt) if a in opt else 0.0 for a in range(game.num_actions)],
                     float(game.position_value(s))))
    return rows


def check_images(game, examples: list) -> dict:
    """Prove the augmented set IS the raw state space with the right labels: every example must encode a reachable
    non-terminal position and carry that position's own exact policy and value, and every such position must
    appear. Counting rows cannot see an action permutation that sends a label to the wrong cell; this can."""
    from harness.coverage import raw_encoding_lookup
    from harness.neural import encode

    lookup = raw_encoding_lookup(game, encode)
    labels: dict = {}
    covered: set = set()
    for i, (x, pi, v) in enumerate(examples):
        s = lookup.get(x.detach().cpu().numpy().tobytes())
        if s is None:
            raise ValueError(f"example #{i} encodes no reachable non-terminal position")
        sk = game.state_key(s)
        if sk not in labels:
            _x, want_pi, want_v = exact_rows(game, [s])[0]
            labels[sk] = (want_pi, want_v)
        if (list(pi), float(v)) != labels[sk]:
            raise ValueError(f"example #{i} carries the wrong label for the position it encodes: policy {list(pi)} "
                             f"value {v}, exact {labels[sk][0]} value {labels[sk][1]}")
        covered.add(sk)
    if len(covered) != len(lookup):
        raise ValueError(f"{len(lookup) - len(covered)} reachable position(s) appear in no example — the net "
                         f"would be graded on states it never saw in any orientation")
    return {"raw_positions": len(lookup), "covered": len(covered)}


def supervised_set(game) -> tuple[list, list, dict]:
    """(canonical states, training examples, facts): the exact rows of every canonical state, multiplied by the
    game's verified symmetries exactly as training augments them. Refused unless the enumeration is complete at
    the registered state count, every state carries exactly the registered number of images, and check_images
    passes — the gate is only a statement about capacity if the data is the whole, correctly labelled answer."""
    from harness.coverage import reachable_states
    from harness.neural import augment_examples

    states, complete = reachable_states(game, exact=True, symmetry=True)
    if not complete or len(states) != CANONICAL_STATES:
        raise ValueError(f"expected the complete space of {CANONICAL_STATES} canonical states, got {len(states)} "
                         f"(complete={complete})")
    perms = game.symmetries()
    examples = augment_examples(exact_rows(game, states), perms)
    if len(examples) != len(states) * IMAGES:
        raise ValueError(f"expected {IMAGES} symmetric images per state ({len(states) * IMAGES} examples), got "
                         f"{len(perms)} symmetries and {len(examples)} examples")
    raw = check_images(game, examples)
    floor = sum(math.log(sum(1 for p in e[1] if p > 0)) for e in examples) / len(examples)
    return states, examples, {"canonical_states": len(states), "images": len(perms), "examples": len(examples),
                              "raw_positions": raw["raw_positions"],
                              "steps_per_epoch": math.ceil(len(examples) / BATCH_SIZE), "loss_floor": floor}


def policy_fail_keys(game, net, states: list, optimal: list) -> list:
    """Canonical keys of the states whose RAW policy — the argmax of the eval-mode logits over the LEGAL moves, no
    search — is not an optimal move. One forward per state and ties to the first legal move: the same reading as
    localize_ceiling's policy_fail_keys, so a G0 count and an arm's count mean the same thing."""
    import torch

    from harness.neural import encode

    net.eval()
    out = []
    for s, opt in zip(states, optimal, strict=True):
        with torch.no_grad():
            logits, _v = net(encode(game, s).unsqueeze(0))
        a = max(game.legal_actions(s), key=lambda x: float(logits[0, x]))
        if a not in opt:
            out.append(game.canonical_key(s))
    return out


def optimizer_facts(opt) -> dict:
    """What the optimizer that trained the net actually was, read from the object rather than from train_net's
    source: its class, lr, weight decay and the distinct step counts its parameters took."""
    group = opt.param_groups[0]
    return {"name": type(opt).__name__, "lr": group["lr"], "weight_decay": group["weight_decay"],
            "steps": sorted({int(st["step"]) for st in opt.state.values()})}


def check_training(arch: str, params: int, optimizer: dict, expected_steps: int) -> None:
    """Refuse a run whose net or optimizer is not the registered one. §C.43-45 trained the legacy net while their
    config said 32/3/32; a parameter count is the check that would have caught it, and the optimizer's own step
    count is the check that the budget the evidence states is the budget that ran."""
    if params != PARAMS_EXPECTED[arch]:
        raise ValueError(f"{arch} net has {params} parameters, registered {PARAMS_EXPECTED[arch]}")
    want = {"name": "Adam", "lr": LR, "weight_decay": WEIGHT_DECAY, "steps": [expected_steps]}
    if optimizer != want:
        raise ValueError(f"optimizer {optimizer} is not the registered {want}")


def make_tasks(arch: str, epochs: list[int], seeds: list[int], threads: int, states: list, examples: list,
               optimal: list) -> list[dict]:
    """One task per (epochs, seed), longest budget first. The examples travel as plain lists so no tensor crosses a
    process boundary through torch's shared-memory reducers."""
    import torch

    x = torch.stack([e[0] for e in examples]).tolist()
    pi = [list(e[1]) for e in examples]
    v = [e[2] for e in examples]
    return [{"arch": arch, "epochs": e, "seed": s, "threads": threads, "x": x, "pi": pi, "v": v,
             "states": states, "optimal": optimal} for e in sorted(epochs, reverse=True) for s in seeds]


def train_run(task: dict) -> dict:
    """Train ONE fresh net on the G0 set and grade its raw policy. The torch seed fixes both the initial weights
    and every epoch's batch order, and one thread keeps the run bit-reproducible. `opt_state={}` makes train_net
    build the same fresh Adam it builds without it, and hands the optimizer back so its facts can be checked."""
    import torch

    torch.set_num_threads(task["threads"])
    from harness.fingerprint import training_fingerprint
    from harness.neural import Connect4Net, arch_for_game, train_net
    from harness.registry import resolve_game

    stamp = training_fingerprint(GAME)
    game = resolve_game(GAME)
    arch = task["arch"]
    x = torch.tensor(task["x"], dtype=torch.float32)
    examples = [(x[i], task["pi"][i], task["v"][i]) for i in range(len(task["pi"]))]
    torch.manual_seed(task["seed"])
    net = Connect4Net(**arch_for_game(ARCHS[arch], game))
    params = sum(p.numel() for p in net.parameters())
    opt_state: dict = {}
    t0 = time.time()
    loss = train_net(net, examples, task["epochs"], BATCH_SIZE, LR, "cpu", opt_state=opt_state)
    seconds = time.time() - t0
    steps = task["epochs"] * math.ceil(len(examples) / BATCH_SIZE)
    optimizer = optimizer_facts(opt_state["opt"])
    check_training(arch, params, optimizer, steps)
    fails = sorted(policy_fail_keys(game, net, task["states"], [set(o) for o in task["optimal"]]))
    return {"arch": arch, "net_arch": ARCHS[arch], "params": params, "epochs": task["epochs"], "steps": steps,
            "seed": task["seed"], "failures": len(fails), "failing_keys": fails, "train_loss": loss,
            "optimizer": optimizer, "train_seconds": round(seconds, 1), "training_fingerprint": stamp}


def gate_verdict(runs: list[dict]) -> dict:
    """G0 as registered: only the gate's own arch, epoch budget and seeds count. A missing registered seed is
    `not_run` whatever the others scored, and an extra seed can neither rescue nor sink the gate."""
    at = {r["seed"]: r for r in runs if r["arch"] == GATE["arch"] and r["epochs"] == GATE["epochs"]}
    missing = [s for s in GATE["seeds"] if s not in at]
    zero = [s for s in GATE["seeds"] if s in at and at[s]["failures"] == 0]
    if missing:
        verdict = "not_run"
    else:
        verdict = "passed" if len(zero) >= GATE["min_zero_failure_seeds"] else "failed"
    return {**GATE, "zero_failure_seeds": zero, "missing_seeds": missing, "verdict": verdict}


def summarise(runs: list[dict]) -> list[dict]:
    """Failures per seed for every (arch, epochs) — the legacy step curve (D7) reads straight off this."""
    groups: dict = {}
    for r in sorted(runs, key=lambda r: (r["arch"], r["epochs"], r["seed"])):
        g = groups.setdefault((r["arch"], r["epochs"]), {"arch": r["arch"], "epochs": r["epochs"],
                                                         "steps": r.get("steps"), "seeds": [], "failures": []})
        g["seeds"].append(r["seed"])
        g["failures"].append(r["failures"])
    for g in groups.values():
        g["zero_failure_seeds"] = sum(1 for f in g["failures"] if f == 0)
        g["mean_failures"] = sum(g["failures"]) / len(g["failures"])
    return list(groups.values())


def merge_evidence(existing: dict | None, new: dict) -> dict:
    """Fold one invocation into the file already at --out. One file must describe one experiment, so a merge
    across training code, measurement code, game or supervised set is refused; a run with the same (arch, epochs,
    seed) is replaced, never duplicated. The verdict and summary are recomputed over the merged runs."""
    def key(r):
        return (r["arch"], r["epochs"], r["seed"])

    runs = list(new["runs"])
    started, invocations = new["started"], list(new["invocations"])
    if existing is not None:
        for field in ("game", "training_fingerprint", "measurement_fingerprint", "dataset"):
            if existing.get(field) != new.get(field):
                raise ValueError(f"refusing to merge: {field} is {new.get(field)!r} here but {existing.get(field)!r} "
                                 f"in the existing file — rerun every arch into a fresh --out")
        fresh = {key(r) for r in runs}
        runs = [r for r in existing["runs"] if key(r) not in fresh] + runs
        started, invocations = existing["started"], existing["invocations"] + invocations
    runs.sort(key=key)
    return {**new, "started": started, "invocations": invocations, "verdict": gate_verdict(runs),
            "summary": summarise(runs), "runs": runs}


def check_stamps(before: tuple, after: tuple, runs: list[dict]) -> None:
    """Refuse evidence whose code moved under it: the training and measurement stamps must hold from before the
    first run to after the last, and every worker must have trained under the same training stamp."""
    seen = {r["training_fingerprint"] for r in runs}
    if after != before or seen - {before[0]}:
        raise ValueError(f"training or measurement code changed while the runs ran (before {before}, after {after}, "
                         f"workers {sorted(seen)}) — no stamp describes this evidence, so it is not written")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=sorted(ARCHS))
    ap.add_argument("--epochs", default="62", help="comma-separated epoch budgets, each a separate fresh run")
    ap.add_argument("--seeds", default="61-65", help="torch seeds: a range a-b or a comma list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    from harness.coverage import optimal_actions
    from harness.fingerprint import training_fingerprint
    from harness.registry import resolve_game

    epochs, seeds = _parse_epochs(args.epochs), _parse_seeds(args.seeds)
    out = Path(args.out)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    before = (training_fingerprint(GAME), training_fingerprint(modules=MEASUREMENT_MODULES))
    game = resolve_game(GAME)
    states, examples, dataset = supervised_set(game)
    existing = json.loads(out.read_text()) if out.exists() else None
    probe = {"game": GAME, "training_fingerprint": before[0], "measurement_fingerprint": before[1],
             "dataset": dataset, "started": started, "invocations": [], "runs": []}
    merge_evidence(existing, probe)
    print(f"G0 set: {dataset['canonical_states']} canonical states x {dataset['images']} images = "
          f"{dataset['examples']} examples covering {dataset['raw_positions']} raw positions; "
          f"{dataset['steps_per_epoch']} steps/epoch; loss floor {dataset['loss_floor']:.4f}", flush=True)
    optimal = [sorted(optimal_actions(game, s)) for s in states]
    tasks = make_tasks(args.arch, epochs, seeds, args.threads, states, examples, optimal)
    runs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(train_run, tasks):
            print(f"{r['arch']:>8} seed {r['seed']:>3} epochs {r['epochs']:>4} ({r['steps']} steps): failures "
                  f"{r['failures']:>2} {r['failing_keys']}  loss {r['train_loss']:.4f}  [{r['train_seconds']:.0f}s]",
                  flush=True)
            runs.append(r)
    after = (training_fingerprint(GAME), training_fingerprint(modules=MEASUREMENT_MODULES))
    check_stamps(before, after, runs)
    new = {**probe, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "config": {"archs": ARCHS, "params_expected": PARAMS_EXPECTED, "batch_size": BATCH_SIZE, "lr": LR,
                      "weight_decay": WEIGHT_DECAY, "device": "cpu", "gate": GATE},
           "invocations": [{"started": started, "arch": args.arch, "epochs": epochs, "seeds": seeds,
                            "threads": args.threads}],
           "runs": runs}
    evidence = merge_evidence(existing, new)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=1))
    for g in evidence["summary"]:
        print(f"{g['arch']:>8} @ {g['epochs']:>4} epochs: failures {g['failures']} (seeds {g['seeds']}), "
              f"{g['zero_failure_seeds']}/{len(g['seeds'])} at zero", flush=True)
    v = evidence["verdict"]
    print(f"G0 verdict ({v['arch']} @ {v['epochs']} epochs, seeds {v['seeds']}): {v['verdict']} — zero-failure "
          f"seeds {v['zero_failure_seeds']}, need {v['min_zero_failure_seeds']}", flush=True)


if __name__ == "__main__":
    main()
