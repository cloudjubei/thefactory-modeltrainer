"""§C.49 D3 — does each T10 net's value head back its own wrong moves? For every label-cache position with White to
move and every move's exact value recorded (plies 0-8), read the net's raw move and its value head's rating of each
move: minus the value head at the resulting position, or the exact result where the move ends the game. Judged by
harness.value_diagnosis.d3_report on D3_SPEC. No solver is used: the exact values are the cache's.

    PYTHONPATH=. .venv/bin/python scripts/c4_value_diagnosis.py --out evidence/c49_D3_value_head.json.gz
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NETS = ROOT / "checkpoints" / "c49_sf" / "c49_T10_solver_free"
LABELS = ROOT / "books" / "c4_labels.json.gz"
MEASUREMENT_MODULES = ("scripts/c4_value_diagnosis.py",)


def _value_head(game, net, states: list) -> list:
    import torch

    from harness.neural import encode

    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(states), 4096):
            _logits, v = net(torch.stack([encode(game, s) for s in states[i:i + 4096]]))
            out.extend(float(x) for x in v[:, 0])
    return out


def main() -> None:
    import hashlib
    import platform

    import torch

    from games.connect4 import C4State, Connect4
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.neural import load_net
    from harness.strategy_tree import raw_chooser
    from harness.value_diagnosis import D3_SPEC

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    stamps = (training_fingerprint("connect4"), training_fingerprint(modules=MEASUREMENT_MODULES))
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    game = Connect4()
    plies = D3_SPEC["config"]["plies"]
    cases = []
    for row in load_evidence(LABELS)["positions"]:
        s = C4State(tuple(row["board"]), row["to_move"], None, False)
        ply = sum(1 for c in s.board if c)
        values = {int(a): int(v) for a, v in row["values"].items()}
        if s.to_move == 0 and ply in plies and not game.is_terminal(s) and set(values) == set(game.legal_actions(s)):
            cases.append((s, ply, values))
    children = [(i, a, game.step(s, a)) for i, (s, _p, values) in enumerate(cases) for a in values]
    open_children = [c for _i, _a, c in children if not game.is_terminal(c)]
    positions, nets = [], {}
    for seed in D3_SPEC["seeds"]:
        path = NETS / f"seed_{seed}.pt"
        net = load_net(str(path))
        nets[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = raw_chooser(game, net)([s for s, _p, _v in cases])
        heads = iter(_value_head(game, net, open_children))
        q_hat = [{} for _ in cases]
        for i, a, child in children:
            q_hat[i][a] = (game.returns(child)[0] if game.is_terminal(child) else -next(heads))
        positions += [{"seed": seed, "ply": p, "raw": m, "q_hat": {str(a): round(q, 6) for a, q in q.items()},
                       "values": {str(a): v for a, v in values.items()}}
                      for (s, p, values), m, q in zip(cases, raw, q_hat)]
        print(f"seed {seed}: {len(cases)} positions", flush=True)
    if (training_fingerprint("connect4"), training_fingerprint(modules=MEASUREMENT_MODULES)) != stamps:
        raise SystemExit("training or measurement code changed while the measurement ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "training_fingerprint": stamps[0], "measurement_fingerprint": stamps[1],
                             "versions": {"python": platform.python_version(), "torch": torch.__version__},
                             "config": D3_SPEC["config"], "seeds": list(D3_SPEC["seeds"]), "nets": nets,
                             "positions": positions})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
