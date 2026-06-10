# Code Standard

Architecture and coding standards for this repository. Intentionally concise; follow it when
adding or modifying code. It adopts the thefactory-tools standard (see
`thefactory-tools/docs/CODE_STANDARD.md`) with this repo's specifics below — where the two
disagree for code in THIS repo, this document wins.

- Language and Modules
  - TypeScript strict mode; target/module ES2022; ESM imports. Node built-ins via `node:` prefix.
  - Prefer named exports.
  - This package is consumed by the backend via its `dist/` build — run `npm run build` after
    source changes that dependents need.

- Formatting
  - Prettier is the source of truth (`.prettierrc.json`). Do not commit unformatted code.

- Comments
  - Default to none; naming explains WHAT, comments exist only for WHY (hidden constraint,
    invariant, workaround). Never inline project conventions in source — they live here.
  - `*Types.ts` is the exception: TSDoc on exported types/members is the public API
    documentation. Keep it tight and accurate.

- Naming
  - Private fields `_camelCase`; expose via `get` accessors, not `getFoo()` methods.
  - Verb conventions: `get*` pure reads; `read*`/`list*` disk reads; `write*`/`upsert*` disk
    writes; action verbs for mutators.
  - Toolset interface methods are globally unique and self-describing (verb-then-noun naming the
    entity, e.g. `runTrainingCampaign`, not `run`).

- Module layout (mirrors a thefactory-tools toolset folder, flattened to `src/`)
  - `src/ModelTrainerTools.ts` — the deps-injected toolset factory (`createModelTrainerTools`).
  - `src/modelTrainerTypes.ts` — ALL public types/interfaces, TSDoc'd. Never inline an exported
    type elsewhere.
  - `src/modelTrainerUtils.ts` — pure functions only (zero node imports).
  - `src/modelTrainerHelpers.ts` — anything touching node (fs/crypto/network/process).
  - `src/modelTrainerConstants.ts` — exported constants.
  - `src/index.ts` — the barrel; the package's only export surface.
  - Internal implementations are plain exported classes (interface in `modelTrainerTypes.ts`,
    `Xyz implements I`); `create*Tools` is reserved for the toolset factory.

- Boundaries
  - Generic infra (process spawning, compute runners, caches, file/JSON primitives, activity
    engine) belongs in `thefactory-tools` — import it, never duplicate it here. This repo owns
    the training DOMAIN only: manifest, matrix planning, campaign orchestration, judging,
    proposing, the viewer.
  - The engine is domain-oblivious: no model-specific term (e.g. "cartpole", "blackswan")
    appears in `src/`. Model specifics live in each project's `TrainerManifest` + the
    `examples/` projects.
  - LLM calls go through thefactory-tools' inference seam; raw-JSON completions must pass
    `structuredOutput: false`.

- `examples/`
  - Each example is a self-contained, trainer-conformant Python project — the executable
    specification of `docs/model-training-standard.md`. Examples are reference code, not
    library code: no TypeScript tests; they are exercised by integration smoke runs.

- `viewer/`
  - A no-build static web app (plain HTML/JS/CSS) served by the Overseer into a sandboxed
    iframe. No frameworks, no deps, no build step. Talks to the host only via
    `window.OverseerBridge`.

- Testing: TDD is mandatory — see `docs/TESTING.md`.
