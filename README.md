# thefactory-modeltrainer

Generic model-training orchestration for the Overseer: plan experiment campaigns over any
**trainer-conformant** project, run them through a compute runner, persist every result as a
record, and (coming) judge results + propose the next experiments with LLMs.
BlackSwanExperiments is the first consumer.

- **The standard** a training project conforms to: [docs/model-training-standard.md](docs/model-training-standard.md)
- **How it all fits**: [docs/architecture.md](docs/architecture.md)
- **What's left to build**: [docs/implementation-plan.md](docs/implementation-plan.md)
- **Repo standards**: [docs/CODE_STANDARD.md](docs/CODE_STANDARD.md) ·
  [docs/TESTING.md](docs/TESTING.md) · [docs/FILE_ORGANISATION.md](docs/FILE_ORGANISATION.md)

## Layout

- `src/` — the library (`createModelTrainerTools`): manifest reading, matrix planning,
  campaign orchestration over a `ComputeRunner`.
- `examples/cartpole/` — the executable specification of the standard: SB3 + gymnasium RL,
  no data, trains in seconds on CPU.
- `viewer/` — the no-build static **hub app** served as this project's Overseer App view (via
  the `appDir: "viewer"` project config): register training projects by directory, then
  run/judge/propose/evaluate per project.

## Develop

```bash
npm install        # links ../thefactory-tools
npm test           # vitest, TDD, near-100% coverage enforced
npm run build      # dist/ — dependents (the backend) import this
node scripts/smoke.mjs   # real end-to-end campaign against examples/cartpole (~15s)
```

`examples/cartpole` needs a one-time venv: `cd examples/cartpole && python3 -m venv .venv &&
.venv/bin/pip install stable-baselines3 gymnasium`.
