# Testing Guide

Vitest, strict TDD, near-100% coverage. This adopts `thefactory-tools/docs/TESTING.md`; repo
specifics below.

## TDD — mandatory

Red-green-refactor for every change in `src/`: write the failing test first, watch it fail,
implement until green, then refactor. No implementation lands without a test that failed
before it.

## Atomicity

One test file per source file (`modelTrainerUtils.ts` → `modelTrainerUtils.test.ts`). Each
test checks a single code path or return. Direct tests per file — transitive coverage through
a parent module does not count.

## Coverage

Near-100% for `src/`: `npm run test:cov` enforces thresholds (statements 95 / branches 90 /
functions 95 / lines 95). New/changed code gets an atomic test per branch and edge case.

## Rules

- Tests are deterministic and offline. Mock external dependencies: the injected
  `ComputeRunner` (never spawn real training in unit tests), `DataStorage` (in-memory stub),
  LLM calls, the filesystem where practical.
- Filesystem tests use temp dirs under the repo (cleaned up per test); never write outside.
- Validate malformed inputs explicitly (bad manifests, malformed summaries, empty sweeps).
- NEVER fix code just to make tests pass — tests poke at holes; the code must make sense.

## What is NOT unit-tested

- `examples/*` Python projects — exercised by integration smoke runs (a real tiny campaign),
  not vitest.
- the root app files (`index.html`/`app.js`/`bridge.js`/`style.css`) — a no-build static app;
  logical helpers may be extracted and tested if they grow, UI behaviour is verified in the
  running Overseer.

## Integration smoke

A real end-to-end campaign against `examples/cartpole` (LocalComputeRunner, tiny
total_timesteps) validates the full path before sign-off. It is run on demand, not in the
default `npm test`.
