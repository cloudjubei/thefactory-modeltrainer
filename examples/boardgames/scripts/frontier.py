"""§C.49 — find the ORACLE FRONTIER of a game: for each family of nets and each training recipe, the smallest width
at which every seed's net, taught the exact answers, holds the target. One search per (family, recipe) runs in its
own worker; every probe is recorded. Judged and summarised from the evidence file; see harness/frontier.py.

    PYTHONPATH=. .venv/bin/python scripts/frontier.py --game tictactoe \\
        --families mlp,mlp2,conv,residual,canon_mlp,canon_mlp2,canon_conv,canon_residual \\
        --recipes default,slow,fast --seeds 1-5 --workers 10 --out evidence/c49_frontier_tictactoe.json.gz
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

RECIPES = {
    "default": {"lr": 2e-3, "batch": 256, "max_epochs": 6000, "check_every": 25, "patience": 1500},
    "slow": {"lr": 1e-3, "batch": 256, "max_epochs": 12000, "check_every": 25, "patience": 3000},
    "fast": {"lr": 5e-3, "batch": 256, "max_epochs": 6000, "check_every": 25, "patience": 1500},
}
CAPS = {"mlp": 1024, "mlp2": 512, "conv": 128, "residual": 128}
MEASUREMENT_MODULES = ("scripts/frontier.py", "harness/frontier.py", "harness/coverage.py")


def _family(name: str) -> dict:
    canonical = name.startswith("canon_")
    return {"body": name[len("canon_"):] if canonical else name, "canonical": canonical}


def search(job: dict) -> dict:
    import time

    import torch

    torch.set_num_threads(1)
    from harness.frontier import enumerated_target, probe_width, smallest_width
    from harness.registry import resolve_game

    game = resolve_game(job["game"])
    family = _family(job["family"])
    target = enumerated_target(game, family["canonical"])
    recipe = RECIPES[job["recipe"]]
    t0 = time.time()
    records = []

    def probe(width: int) -> bool:
        rec = probe_width(game, family, width, target, job["seeds"], recipe)
        records.append(rec)
        print(f"  {job['family']:>15} {job['recipe']:>7} width {width:>4} ({rec['params']:>6} params): "
              f"{'ok' if rec['ok'] else 'fails'} {[r['best_failures'] for r in rec['runs']]}", flush=True)
        return rec["ok"]

    result = smallest_width(probe, start=job["start"], cap=CAPS[family["body"]])
    frontier = result["frontier"]
    params = next((r["params"] for r in records if r["width"] == frontier), None)
    return {"family": job["family"], "recipe": job["recipe"], "frontier": frontier, "frontier_params": params,
            "non_monotone": result["non_monotone"], "probes": records, "seconds": round(time.time() - t0, 1)}


def _seeds(text: str) -> list:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def main() -> None:
    import platform

    import torch

    from harness.evidence import save_evidence
    from harness.fingerprint import training_fingerprint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--recipes", default="default")
    ap.add_argument("--seeds", default="1-5")
    ap.add_argument("--start", type=int, default=2)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    families, recipes = args.families.split(","), args.recipes.split(",")
    unknown = [f for f in families if _family(f)["body"] not in CAPS] + [r for r in recipes if r not in RECIPES]
    if unknown:
        raise SystemExit(f"unknown families/recipes: {unknown}")
    seeds = _seeds(args.seeds)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stamps = (training_fingerprint(args.game), training_fingerprint(modules=MEASUREMENT_MODULES))
    jobs = [{"game": args.game, "family": f, "recipe": r, "seeds": seeds, "start": args.start}
            for f in families for r in recipes]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(search, jobs):
            print(f"{r['family']:>15} {r['recipe']:>7}: frontier width {r['frontier']} ({r['frontier_params']} params)"
                  f"{' NON-MONOTONE ' + str(r['non_monotone']) if r['non_monotone'] else ''} [{r['seconds']:.0f}s]",
                  flush=True)
            rows.append(r)
    if (training_fingerprint(args.game), training_fingerprint(modules=MEASUREMENT_MODULES)) != stamps:
        raise SystemExit("training or measurement code changed while the searches ran — evidence not written")
    found = [r for r in rows if r["frontier"] is not None]
    best = min(found, key=lambda r: r["frontier_params"]) if found else None
    save_evidence(args.out, {"game": args.game, "started": started,
                             "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "training_fingerprint": stamps[0], "measurement_fingerprint": stamps[1],
                             "versions": {"python": platform.python_version(), "torch": torch.__version__},
                             "config": {"families": families, "recipes": {r: RECIPES[r] for r in recipes},
                                        "seeds": seeds, "start": args.start, "caps": CAPS},
                             "searches": rows,
                             "smallest": None if best is None else {k: best[k] for k in
                                                                    ("family", "recipe", "frontier", "frontier_params")}})
    if best:
        print(f"smallest: {best['family']} / {best['recipe']} at width {best['frontier']} = "
              f"{best['frontier_params']} params", flush=True)


if __name__ == "__main__":
    main()
