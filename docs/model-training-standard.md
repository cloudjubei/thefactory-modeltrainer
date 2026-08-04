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
  "objective": { "name": "eval_return", "direction": "max" }, // the REWARD/steering signal (in-sample)
  // The SCORECARD (gates + fitness) — the machine-readable definition of SUCCESS, SEPARATE from the
  // reward. Both optional; omit them and the project COLLAPSES to the objective (reward = success, e.g.
  // CartPole/Wine). Declare them when a high reward does NOT equal a good model (e.g. a trading run that
  // games a metric): selection/convergence/ranking + the viewer then read the scorecard, not the reward.
  "gates": [
    // accept/reject predicates over the summary — a run is ACCEPTED only if EVERY gate passes. `value` is
    // a literal OR { "metric": "..." } to compare against another metric (e.g. beat a benchmark).
    { "metric": "eval_return", "op": ">", "value": 0, "label": "profitable" }
  ],
  "fitness": [
    // how ACCEPTED runs are RANKED — a single scalar or a Pareto set; the first entry is the primary key.
    { "metric": "eval_return", "direction": "max" }
  ],
  // How a SINGLE-CONTEXT hypothesis is PROVEN: the best matching run's `metric` clears `threshold`
  // (toward `direction`, default max). Omitted ⇒ the trading default (return_vs_hold_pct > 0) — declare
  // it for any non-trading project or its hypotheses can never be judged.
  "hypothesisBenchmark": { "metric": "eval_return", "threshold": 475 },
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
- `data[]` entries (when present): `{ id, files: [{ relPath, url, sha256? }], credentialRef? }`.
  The program NEVER downloads — the compute runner materialises declared files into the
  workspace before the run via a content-addressed cache (identical bytes are fetched once,
  ever; a missing file at run time should exit non-zero with a clear message). Never inline
  secrets. `examples/tabular` is the executable spec of this clause.
- `resources` is declarative truth for scheduling + for telling a human the machine class.
- **Three declared layers — `objective` / `gates` / `fitness`.** `objective` is the REWARD (what training
  steers on, in-sample). `gates` are accept/reject predicates and `fitness` is the ranking metric(s) — together
  the **scorecard**, the definition of SUCCESS (out-of-sample), computed post-hoc from the summary and
  INDEPENDENT of the reward. The engine reads the scorecard, not the reward, to accept runs (all gates pass),
  rank/select the incumbent (primary fitness), and gate convergence. **Simple projects collapse all three**:
  omit `gates`/`fitness` and success = the objective (CartPole/Wine — reward *is* success). **They diverge**
  when a high reward does not mean a good model (a trading run that games a metric): declare the scorecard so
  "good" is machine-readable. Governing rule: in-sample behavioural SHAPE (trade frequency, no-op fraction) →
  the reward, as a constraint; out-of-sample OUTCOME (beat-a-benchmark, risk-adjusted return) → the scorecard
  ONLY, never the reward — rewarding on the metric you then select on re-creates single-window overfit. The
  `diagnoseSearch` tool reports the **reward–success alignment** (the correlation of the reward with each
  scorecard metric across the cohort): strong = a good proxy, near-zero = a misaligned proxy.

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
- `--evaluate` (optional; manifest `evaluate` command template) — re-test a saved checkpoint:
  the config it receives carries the original run's levers plus `checkpoint` (and optionally
  `eval_episodes`). No training, no checkpoint writing; writes a `RunSummary` whose
  `objective` is the fresh evaluation result, with a top-level
  `evaluation: { checkpoint, episodes }`. Missing/unloadable checkpoint ⇒ non-zero exit.

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
  "artifacts": {
    "checkpoint": "checkpoints/…", // ref, not the bytes
    "best": true,
    "decisionTrace": {
      /* optional xAI trace — see below; the hub's Explain view reads it */
    },
  },
  "series": {
    "episode_return": [
      /* ≤200 downsampled points */
    ],
  }, // optional per-run curves
}
```

- **One objective.** A run is comparable to another by a single number + direction. Everything
  else is diagnostic. (BlackSwan today has no single objective — Sharpe/CAGR/max-DD are stubbed;
  fixing that is part of its migration.)
- **Health flags** make degenerate runs auto-rejectable by the judge (constant-action policy,
  NaN/inf loss, zero trades) — cheap, high-signal.
- **Provenance** makes a run reproducible from `(gitCommit, config, seed, dataVersion)`.
- **Artifacts are referenced**, never shipped in the summary; "best" is selected by the objective.
- **Decision trace (optional, xAI).** A project MAY attach `artifacts.decisionTrace` — a generic
  `DecisionTrace` ({`steps[]` of {`step`, `action` label, optional `confidence`, `actionValues`,
  `alternativeAction`, `forced`, `reward`}, `actionCounts`, optional `featureAttribution`,
  `totalSteps`}) — for the hub's **Explain** view (action distribution, per-action value over time,
  confidence, input attribution). It is domain-oblivious: arbitrary action strings, no domain
  vocabulary. The embedded trace is downsampled to ≤200 steps to share the chart axis; the full
  per-step trace (with raw observations) is an optional sidecar at `artifacts.decisionTraceFile`. The
  engine validates it softly (a missing/unusable trace is dropped, never an error).

---

## 4. Practices (what makes a run reproducible + judgeable)

1. **Config as data, not code.** One declarative, typed config fully specifies a run; no editing
   Python to launch. (Hydra/pydantic both fine; the JSON contract is what the orchestrator sees.)
2. **Reproducibility.** Explicit seed control; record seed + git commit + config hash + data
   version on every run. Pinned deps (lockfile).
3. **Structured tracking, not ephemeral.** Log params + metrics + artifacts to a tracker (MLflow),
   streamed during training — not a CSV dumped in the cwd at the end.
4. **One objective (the reward) + the scorecard (success) + a standard battery, baseline-relative.** The
   objective is the single steering signal; `gates`/`fitness` define success separately (collapse to the
   objective when reward = success). Held-out eval; report vs a trivial baseline (random / buy-and-hold /
   majority-class). Degenerate-run detection.
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

---

## 6. AI action parity (the chat can do what the user can do)

Every mutating or launching action a user can take in the project's app has an equivalent, **approval-gated**
chat capability — so a user can drive the whole loop as a conversation. The surface is declared in ONE place
(the engine's `trainerCapabilities.ts`, mirrored by the viewer's `trainerCapabilities.js` twin) and audited so
it can't silently regress:

- **Launches** (train, explore, judge, evaluate, cross-test, mine data, research, …) go through a bespoke
  `startTrainerActivity` tool that injects the project context (`recordType`/`dir`) the chat can't know and
  validates a train/explore spec against the manifest BEFORE admitting it. The generic `startProjectActivity`
  path is deliberately NOT used for trainer activities — it can neither inject that context nor describe an
  activity's params.
- **Record edits** (environments, datasets, scorecards, hypotheses, papers, per-run flags, project settings)
  ride the generic `createProjectRecord`/`updateProjectRecord` tools against the declared record types, each
  with an editable/creatable field allow-list.
- **Destructive** verbs (delete runs, with their derived records) are separate tools, ALWAYS shown for
  explicit approval.
- **Code** changes (new models / levers / components) are never in-place edits — the AI files a story/feature.
- **Exempt** from parity, with a stated reason: UI-only state (seen marks, drain order), backend-derived
  records (evaluations, verdicts, stats), hub registration, and automatic boot maintenance (migrate/invalidate
  sweeps — not user actions).

**The audit is a hard gate.** A test enumerates the viewer's `startOrEnqueue` / `putData` / `deleteData` sites
and fails the build if any activity or record type isn't declared launchable/editable or explicitly exempted —
so adding a new user action without a chat path breaks CI.
