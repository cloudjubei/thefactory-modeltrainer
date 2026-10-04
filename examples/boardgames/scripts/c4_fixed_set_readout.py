"""Score every net of the named Connect-4 runs on ONE FIXED SET of exactly valued positions (harness.fixed_set): the
label cache's White-to-move positions at plies 0-8 with every move's exact value and a real choice. For each run
prefix, every training evidence file <prefix>_<arm>.json.gz is read and each seed's saved net (hash-checked) plays its
raw move at every position; each net's certificate verdict from its run is carried along. Forward passes only; no
solver.

    PYTHONPATH=. .venv/bin/python scripts/c4_fixed_set_readout.py --runs c49_T14 c49_T16 \\
        --out evidence/c49_P1_fixed_set.json.gz
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "books" / "c4_labels.json.gz"
MEASUREMENT_MODULES = ("scripts/c4_fixed_set_readout.py", "harness/fixed_set.py")
PLIES = [0, 2, 4, 6, 8]


def main() -> None:
    import hashlib
    import platform

    import torch

    from games.connect4 import Connect4
    from harness.evidence import load_evidence, save_evidence
    from harness.fingerprint import training_fingerprint
    from harness.fixed_set import fixed_positions, score
    from harness.neural import load_net
    from harness.strategy_tree import raw_chooser

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    stamp = training_fingerprint(modules=MEASUREMENT_MODULES)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    game = Connect4()
    cases = fixed_positions(load_evidence(LABELS)["positions"], game, player=0, plies=PLIES)
    states = [s for s, _p, _v in cases]
    nets, runs = [], {}
    for run in args.runs:
        for path in sorted((ROOT / "evidence").glob(f"{run}_*.json.gz")):
            arm = path.name[len(run) + 1:-len(".json.gz")]
            if arm == "readout":
                continue
            trained = load_evidence(path)
            runs[f"{run}/{arm}"] = {"training_fingerprint": trained["training_fingerprint"],
                                    "config": trained["config"]}
            for row in sorted(trained["seeds"], key=lambda r: r["seed"]):
                net_path = ROOT / row["net"]["path"]
                if hashlib.sha256(net_path.read_bytes()).hexdigest() != row["net"]["file_sha256"]:
                    raise SystemExit(f"{run}/{arm} seed {row['seed']}: the saved net is not the one its run recorded")
                result = score(raw_chooser(game, load_net(str(net_path)))(states), cases)
                cert = row.get("certificate") or {}
                nets.append({"run": run, "arm": arm, "seed": row["seed"], "net_file_sha256": row["net"]["file_sha256"],
                             "certified": cert.get("certified"), "failures": cert.get("failures"), **result})
                print(f"{run}/{arm} seed {row['seed']}: {result['optimal']}/{result['positions']}", flush=True)
    if training_fingerprint(modules=MEASUREMENT_MODULES) != stamp:
        raise SystemExit("measurement code changed while the readout ran — evidence not written")
    save_evidence(args.out, {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "measurement_fingerprint": stamp, "versions": {"python": platform.python_version(),
                                                                            "torch": torch.__version__},
                             "plies": PLIES, "positions": len(cases), "runs": runs, "nets": nets})


if __name__ == "__main__":
    from harness.trials import logged

    logged(main, __file__)
