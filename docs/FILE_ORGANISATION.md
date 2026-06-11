# Project File Organisation

- `docs/` — standards + plans:
  - `model-training-standard.md` — THE contract every training project conforms to
    (TrainerManifest, CLI contract, RunSummary, practices). The `examples/` projects are its
    executable specification.
  - `implementation-plan.md` — the staged build plan (holds only remaining work).
  - `architecture.md` — how the pieces fit (engine ↔ thefactory-tools ↔ backend ↔ viewer ↔
    conformant projects). Keep current as work lands.
  - `CODE_STANDARD.md`, `TESTING.md`, this file.
- `src/` — the library (see CODE_STANDARD.md module layout):
  - `index.ts` (barrel), `ModelTrainerTools.ts` (toolset factory), `modelTrainerTypes.ts`
    (all public types), `modelTrainerUtils.ts` (pure), `modelTrainerHelpers.ts` (node-touching),
    `modelTrainerConstants.ts`, co-located `*.test.ts` per file.
- `examples/` — trainer-conformant reference Python projects:
  - `cartpole/` — RL, no data, CPU/seconds (SB3 + gymnasium). The first executable spec.
  - `tabular/` — (planned, Phase 6) small DVC-tracked dataset; establishes the data path.
  - Each has `.factory/trainer.json`, a `trainer/` Python package with the CLI entry
    (`python -m trainer.run`), `configs/`, `pyproject.toml`, its own `.venv/` (gitignored).
- `viewer/` — the no-build static **hub app** (`index.html`, `app.js`, `bridge.js`,
  `style.css`), served via the project's `appDir: "viewer"` config (set in project settings):
  a home screen of registered training projects (directories) + a per-project dashboard
  (runs/launch/judge/hypotheses/charts).
- `dist/` — build output (gitignored); the surface dependents import.
