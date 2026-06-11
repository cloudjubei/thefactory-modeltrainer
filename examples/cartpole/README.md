# cartpole — executable specification of the model-training standard

A complete, working reference trainer conforming to
[`docs/model-training-standard.md`](../../docs/model-training-standard.md). It mirrors the
BlackSwan stack (stable-baselines3 + gymnasium) but is self-contained: no dataset
(`data: []`), trains in seconds on CPU.

The three pieces of the standard, made concrete:

- **Manifest** — `.factory/trainer.json`: name, record type, run/calibrate/evaluate
  command templates, the single objective (`eval_return_mean`, max), the sweepable levers,
  declared resources.
- **CLI contract** — `python -m trainer.run` honoring `--config-json`, `--summary-out`,
  `--calibrate`, `--evaluate`. Config is data, not code; runs are deterministic given the
  config's seed.
- **RunSummary** — written to `--summary-out`: objective, metric battery, training-curve
  series (`series.episode_return`, downsampled to ≤200 points), health flags
  (`nan_metrics`, `degenerate_policy`), seed, resolved config, provenance (configHash,
  ranAt), checkpoint reference. `--calibrate` additionally reports
  `calibration.unitsPerSecond` for ETA.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install stable-baselines3 gymnasium
```

## Run

Calibrate (tiny end-to-end pass; doubles as the CI smoke gate):

```sh
.venv/bin/python -m trainer.run --calibrate --summary-out /tmp/cartpole-cal.json
```

Train from a config:

```sh
.venv/bin/python -m trainer.run --config-json configs/default.json --summary-out /tmp/cartpole-sum.json
```

Checkpoints are saved to `checkpoints/<configHash>.zip`.

Evaluate a saved checkpoint (no training; the config adds `"checkpoint"` and optionally
`"eval_episodes"`, default 50):

```sh
.venv/bin/python -m trainer.run --evaluate --config-json /tmp/cartpole-eval.json --summary-out /tmp/cartpole-eval-sum.json
```

The evaluate RunSummary carries the same objective/metrics/health/provenance plus an
`evaluation: { checkpoint, episodes }` block.

## Layout

```
.factory/trainer.json   # the manifest
trainer/
  config.py             # typed config + JSON loader + configHash
  model.py              # SB3 PPO/DQN factory on CartPole-v1
  train.py              # training loop + wall-clock throughput
  evaluate.py           # deterministic eval -> metrics + health flags
  summary.py            # builds + writes the RunSummary
  run.py                # the CLI entry
configs/default.json    # the default config
```
