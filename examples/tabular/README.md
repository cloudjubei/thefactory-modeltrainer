# tabular — executable specification of the model-training standard (DATA path)

A complete, working reference trainer conforming to
[`docs/model-training-standard.md`](../../docs/model-training-standard.md). Where
`examples/cartpole` exercises the no-data path (`data: []`), this project exercises the
**declared-data** path: wine-quality regression (UCI red-wine, ~100KB semicolon-CSV)
with a sklearn `GradientBoostingRegressor`, predicting `quality` from 11 features over a
seeded 80/20 train/val split. Trains in seconds on CPU.

The three pieces of the standard, made concrete:

- **Manifest** — `.factory/trainer.json`: name, record type, run/calibrate/evaluate
  command templates, the single objective (`val_rmse`, **min**), the sweepable levers,
  the declared `data[]` requirement, declared resources.
- **CLI contract** — `python -m trainer.run` honoring `--config-json`, `--summary-out`,
  `--calibrate`, `--evaluate`. Config is data, not code; runs are deterministic given the
  config's seed.
- **RunSummary** — written to `--summary-out`: objective, metric battery (`val_rmse`,
  `train_rmse`, `val_r2`, `n_train`, `n_val`), per-stage validation curve
  (`series.val_rmse` via `staged_predict`, downsampled to ≤200 points), health flags
  (`nan_metrics`), seed, resolved config, provenance (configHash, ranAt), checkpoint
  reference. `--calibrate` additionally reports `calibration.unitsPerSecond` for ETA.

## The data contract

**This program never downloads.** It requires `data/winequality-red.csv` to exist and
exits non-zero if it is missing. The orchestrator materialises the manifest's declared
`data[]` before runs — launch via the Model Trainer app and the file appears
automatically. For a manual run, download it once yourself:

```sh
curl -fsSL -o data/winequality-red.csv \
  https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
```

`data/` is gitignored: the CSV is reproducible from its URL, never committed.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install scikit-learn pandas joblib
```

## Run

Calibrate (tiny end-to-end pass, `n_estimators=20`; doubles as the CI smoke gate):

```sh
.venv/bin/python -m trainer.run --calibrate --summary-out /tmp/tabular-cal.json
```

Train from a config:

```sh
.venv/bin/python -m trainer.run --config-json configs/default.json --summary-out /tmp/tabular-sum.json
```

Checkpoints are saved to `checkpoints/<configHash>.joblib`.

Evaluate a saved checkpoint (no training; the config adds `"checkpoint"`; the seeded
validation split is recomputed from the config's seed, so `eval_episodes` is ignored):

```sh
.venv/bin/python -m trainer.run --evaluate --config-json /tmp/tabular-eval.json --summary-out /tmp/tabular-eval-sum.json
```

The evaluate RunSummary carries the same objective/metrics/health/provenance plus an
`evaluation: { checkpoint, episodes }` block.

## Layout

```
.factory/trainer.json   # the manifest (incl. the declared data[] requirement)
trainer/
  config.py             # typed config + JSON loader + configHash
  data.py               # data contract + seeded 80/20 train/val split
  model.py              # GradientBoostingRegressor factory + joblib checkpoint IO
  train.py              # fit + wall-clock throughput + per-stage validation curve
  evaluate.py           # seeded held-out eval -> metrics + health flags
  summary.py            # builds + writes the RunSummary
  run.py                # the CLI entry
configs/default.json    # the default config
data/                   # gitignored; materialised by the orchestrator
```
