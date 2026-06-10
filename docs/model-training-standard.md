# The model-training standard

The contract every training project conforms to so the Overseer can orchestrate it
identically — launch, calibrate, run, judge, compare, reproduce. A project that satisfies
this is "trainer-conformant." `examples/cartpole` (no data) and `examples/tabular` (small
data) are the **executable specifications** of this document; BlackSwanExperiments migrates
to it (see `implementation-plan.md`, BlackSwan phase).

The standard is three things: a **manifest** (data the orchestrator reads), a **CLI
contract** (what the program implements), and a set of **practices** (what makes a run
reproducible and judgeable). Keep it minimal — every clause earns its place by being
something the orchestrator or a comparison actually depends on.

---

## 1. `TrainerManifest` — `.factory/trainer.json` (data the orchestrator reads)

```jsonc
{
  "name": "cartpole",
  "recordType": "cartpole-run", // namespaces all DataStorage records
  "run": "python -m trainer.run --config-json {configPath} --summary-out {summaryOut}",
  "calibrate": "python -m trainer.run --calibrate --summary-out {summaryOut}",
  "objective": { "name": "eval_return", "direction": "max" }, // single north-star
  "levers": {
    // the sweepable config space (renders the launch form)
    "learning_rate": { "type": "number", "default": 3e-4, "range": [1e-5, 1e-2] },
    "gamma": { "type": "number", "default": 0.99 },
    "net_arch": {
      "type": "choice",
      "choices": [
        [64, 64],
        [256, 256],
      ],
    },
  },
  "data": [], // dataset requirements; [] = self-contained/no data
  "resources": { "gpu": false, "memory": "2g", "cpus": 2 }, // declared needs → "how big a machine"
  "image": "thefactory/trainer-cartpole:latest", // reproducible run image (optional locally)
}
```

- `run` / `calibrate` are command **templates**; the orchestrator substitutes `{configPath}`
  and `{summaryOut}`. The program reads neither env vars nor hardcoded paths for run config.
- `levers` drives both the launch-form inputs and the matrix planner's sweep. Identity of a
  run = the resolved config (its hash) — the planner's skip-if-fresh key.
- `data[]` entries (when present): `{ id, files | glob, source, credentialRef? }` — consumed
  by the content-addressed cache + the proxy allowlist. Never inline secrets.
- `resources` is declarative truth for scheduling + for telling a human the machine class.

---

## 2. CLI contract (what the program implements)

A single entrypoint (`python -m trainer.run`) honoring:

- `--config-json <path>` — fully build and run from a machine-written config. **No code edits
  to launch a new experiment.** Deterministic given the config's `seed`.
- `--summary-out <path>` — on completion, write a `RunSummary` (§3) to that path. This is the
  machine-readable result the orchestrator ingests; it is the source of truth, not stdout/CSV.
- `--calibrate` — a deliberately tiny end-to-end pass (few steps, one config). Reports
  throughput (`fps`/`secondsObserved`) for ETA **and** doubles as a fast correctness/smoke
  gate (non-zero exit ⇒ the project is broken; usable in CI).

Streaming during a run is encouraged (progress/metrics to stdout + the tracker) but the
contract's hard requirement is the final `RunSummary`.

---

## 3. `RunSummary` — the machine-readable result

```jsonc
{
  "objective": 487.3, // the single north-star value (matches manifest.objective.name)
  "metrics": {
    // standard battery; project-specific keys allowed
    "train_return": 495.0,
    "eval_return": 487.3,
    "baseline_return": 21.0,
  },
  "health": { "status": "ok", "flags": [] }, // e.g. ["degenerate_policy","nan_loss","zero_trades"]
  "seed": 0,
  "seedAggregate": { "mean": 480.1, "median": 488.0, "std": 12.4, "n": 5 }, // when multi-seed
  "config": {
    /* the fully resolved config that produced this run */
  },
  "provenance": { "gitCommit": "abc123", "dataVersion": "dvc:…", "configHash": "…", "ranAt": "…" },
  "artifacts": { "checkpoint": "checkpoints/…", "best": true }, // ref, not the bytes
}
```

- **One objective.** A run is comparable to another by a single number + direction. Everything
  else is diagnostic. (BlackSwan today has no single objective — Sharpe/CAGR/max-DD are stubbed;
  fixing that is part of its migration.)
- **Health flags** make degenerate runs auto-rejectable by the judge (constant-action policy,
  NaN/inf loss, zero trades) — cheap, high-signal.
- **Provenance** makes a run reproducible from `(gitCommit, config, seed, dataVersion)`.
- **Artifacts are referenced**, never shipped in the summary; "best" is selected by the objective.

---

## 4. Practices (what makes a run reproducible + judgeable)

1. **Config as data, not code.** One declarative, typed config fully specifies a run; no editing
   Python to launch. (Hydra/pydantic both fine; the JSON contract is what the orchestrator sees.)
2. **Reproducibility.** Explicit seed control; record seed + git commit + config hash + data
   version on every run. Pinned deps (lockfile).
3. **Structured tracking, not ephemeral.** Log params + metrics + artifacts to a tracker (MLflow),
   streamed during training — not a CSV dumped in the cwd at the end.
4. **One objective + a standard battery + baseline-relative.** Held-out eval; report vs a trivial
   baseline (random / buy-and-hold / majority-class). Degenerate-run detection.
5. **Seed aggregation.** Multiple seeds aggregated (mean/median/std), not N raw rows the human eyeballs.
6. **Train/val/test discipline + leakage guards.** Temporal split for time-series; no test leakage.
7. **Checkpoint-best + resume.** Standard checkpoint format, traceable to its run; "best" by the
   objective; resume-from-checkpoint supported.
8. **Content-addressed data.** DVC (or equivalent) so data is reproducible + the orchestrator can
   fetch only what a run needs. `[]` data is allowed (self-contained projects).
9. **CI smoke gate.** `--calibrate` runs a tiny end-to-end pass fast; green = the project still trains.
10. **Reproducible environment.** A container image + declared `resources`; this is also how the
    machine class is known.

---

## 5. Recommended layout (predictable + separable)

```
<project>/
  .factory/trainer.json          # the manifest
  trainer/
    config.py                    # typed config + JSON loader (the --config-json contract)
    data.py                      # data interface (empty/synthetic for no-data projects)
    model.py                     # model/policy factory
    train.py                     # training loop (streams metrics to the tracker)
    evaluate.py                  # held-out eval → metrics + health flags
    run.py                       # the CLI entry: honors --config-json / --summary-out / --calibrate
    summary.py                   # builds + writes the RunSummary
  configs/                       # named base configs (composition)
  pyproject.toml                 # pinned deps + lockfile
  Dockerfile                     # reproducible image (uid 10001 to match the sandbox runner)
```

`data | model | train | evaluate | serve` stay separable behind clear interfaces. BlackSwan
already has this factory skeleton (`data_factory`/`model_factory`/`env_factory`) — its migration
is mostly conforming the entrypoint + summary + objective + tracking, and pruning dead paths.
