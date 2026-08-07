# thefactory-modeltrainer — implementation plan

**Remaining work only** — shipped history lives in git + session memory. Architecture: `docs/architecture.md`.
Contract: `docs/model-training-standard.md`. The core loop (engine, backend activities, viewer, remote
runner) and three conformant consumers (`examples/cartpole`, `examples/tabular`, **BlackSwan**) are built;
the engine stays domain-oblivious — any further model is _data + the thin CLI contract_, not engine code.

**Structure:** §A = things to **BUILD** now (the ready backlog of platform improvements — keep adding to it).
§B = bigger deferred projects. §C = open questions + trigger-blocked builds (act when the dependency lands).
(Experiments are no longer a plan section — the exploration autopilot / Diagnosis plan drives them; findings
live in git + memory.)

## North star (frames prioritization)

1. **Best generic pipeline for creating ANY model** end to end (propose → run → judge → explore), with a
   self-explanatory results UI, a minimal-storage/derive-at-runtime data layer, and guidance from "here's my
   problem" to "here's what data to mine."
2. **Use it to make BlackSwan the best trading model**, in STRICT ORDER: **(A) correctness → (B) find ONE
   setup that trades well → (C) huge-space exploration.** BlackSwan is the forcing function that hardens the
   generic pipeline. "Trades well" is defined by the **scorecard** (gates + fitness), NOT the reward — the
   reward is a training proxy that for BlackSwan does not equal success (the shipped scorecard defines "good").
3. **Fully AI-operable, shipped as a template.** Anything a user can do in the app, the API/CLI AI can do via
   tools through the approval gate (shipped) — so a user drives the whole loop as a CONVERSATION where the AI
   researches/proposes/runs and the human mostly approves (companion shipped); and the app is a SINGLE-PURPOSE
   template a project copies + a library it consumes for engine + base UI (A5), with BlackSwan the first consumer.

## Repo split (governs where work lands)

| Repo | Owns |
| --- | --- |
| **thefactory-modeltrainer** (this repo) | `ModelTrainerTools`; matrix planner; campaign loop; judge/propose orchestration; the viewer; the standard + `examples/`. |
| **thefactory-tools** | Generic infra: `ComputeRunner` seam (+ future `RemoteComputeRunner`, `ContentAddressedDataCache`, pairing); work-item engine. |
| **thefactory-backend** | Activity registration + composition; app-view serving; future PIN-pairing endpoints + runner WS channel. |
| **clients** | Future Compute Runners settings/pairing screen (native, cross-project). |
| **the runner agent** | Future Docker-packaged connect-out program. |
| **BlackSwan** (the trading repo) | Its `TrainerManifest` + additive `trainer/` CLI conformance. No Overseer code. |

---

## A. Build — implementation tasks

Ordered by value. Each is something to **implement**.

### A5. modeltrainer as the base + a one-time SEED (NOT a live template)

**Model (owner-set).** modeltrainer is the BASE that demonstrates ALL generic functionality — the engine and
the full viewer. A project is **seeded from it ONCE** and then carries on independently: it USES the generic
functionality and adds only what cannot be generic (BlackSwan: its trading env, its rewards — already in its
Python + `.factory/trainer.json`). There is deliberately **NO update-absorption path** — a seeded project does
not track or pull modeltrainer changes, and we do not build machinery for it to. The multi-project HUB stays for
dev/multi-project use; single-purpose boot is what a seeded app runs.

This retires the old "template + LIBRARY consumers pin + updates flow" framing that earlier drafts of A5 (and
`architecture.md`) carried: the versioned-publish-for-pinning step and the `npm update` + re-copy ceremony are
**dropped, not built**, and the "override layer so updates still flow" loses its reason to exist — a seeded
project owns its copy and edits it directly.

1. **Single-purpose boot (viewer). ✅ SHIPPED.** `window.__TRAINER_BOOT__ = {mode:'single', …}` boots straight
   into ONE project's dashboard, bypassing the hub. Decision extracted to a pure, node-tested module
   `viewer/boot.js` (`TrainerBoot.resolveBoot(bootConfig, {embedded})` → `{mode, project, manifest,
   needsInspect, error}`; dual-loaded like `hypothesis.js`, 10 tests in `src/bootViewer.test.ts`). Config forms:
   a **bundled** manifest (opens directly, works standalone via localStorage) or a **dir** to `inspect-trainer`
   (embedded only). Any misconfig is fail-SAFE — resolves to the hub with a surfaced banner, so the no-config
   default is byte-identical to before. `app.js`: `openProject` split so hub and boot share ONE
   `openResolvedProject` path (no duplication); `bootProject` seeds the single project into
   `projectsCache`/`manifestsCache` so every downstream read works unchanged; `hubMode` flag hides Back and
   guards `goHome`. **Remaining in this item:** single-purpose deep-link wiring (embedded boot currently skips
   the `getDeepLink` pull + `onNavOpen` registration) — a refinement, folded into step 5's de-monolith or done
   when a single-purpose app first needs in-app deep-links.
2. **The SEED mechanism. ✅ SHIPPED.** A project declares single-purpose mode through a `boot.config.js` the
   viewer loads BEFORE `boot.js` (added to `index.html`; base modeltrainer ships it EMPTY → hub, so default
   behaviour is unchanged). The seed writes a project-specific `boot.config.js` that sets `window.__TRAINER_BOOT__`.
   Pure, node-tested content generator `TrainerBoot.renderBootConfig(cfg)` (in `viewer/boot.js`) round-trips
   through `resolveBoot`, so a malformed seed can't silently brick the app. `scripts/seed-single-purpose.mjs`
   copies `viewer/` → a target project's app dir and drops that `boot.config.js`. This REPLACES the old "override
   layer" — a seeded project owns its copy; it customizes by editing it, not by layering to survive updates that
   never come.
3. ~~**Package + publish a versioned build for consumers to pin.**~~ **DROPPED** — no update flow, so nothing
   pins-for-updates. (The engine `dist` remains a normal npm lib the BACKEND consumes; that is unchanged and is
   not part of the seed.)
4. **Seed BlackSwan as its own app. ✅ SHIPPED (files) — one runtime step remains.** The overseer serves app
   files ONLY from inside the project checkout (`files.ts` escape guard → `metadata.appDir`, checkout-relative,
   falls back to root), so the viewer bytes must live in BlackSwan. The seed wrote `BlackSwan/app/` (24 files)
   with a `boot.config.js` of `{mode:'single', dir:'.', name:'BlackSwan', manifestRelPath:'.factory/trainer.json'}`
   — the `needsInspect` path loads BlackSwan's REAL manifest fresh via `inspect-trainer dir:'.'`, nothing bundled
   stale. `BlackSwan/app/boot.js` is byte-identical to source (`diff -q` clean). One-time; BlackSwan owns the copy
   (gitignore-able — regenerable by re-running the seed). No Node toolchain imposed on the Python repo. **Remaining
   (overseer runtime, NOT a repo file — cannot be done from the repo):** set the BlackSwan project record's
   `metadata.hasApp=true`, `metadata.appDir="app"`. Until then the App tab won't appear / won't serve `app/`.
5. ~~**`app.js` de-monolith.**~~ **DROPPED as an A5 requirement** — its only A5 justification was overriding
   without editing the monolith; a seeded project owns and may edit its copy freely. Optional hygiene, not the
   seed's blocker.
6. **Update docs. ✅ SHIPPED.** `architecture.md` now carries a "Two deployment modes, one codebase" section
   (hub for dev + one-time seed for a shipped single-purpose app) and the "registered, not forked" bullet notes
   the seed as the one place a project takes a copy it then owns.
7. **THE SOLE REMAINDER — deferred xAI per-step attention scrubber + snapshot diff-consecutive arm.**
   The producers are shipped (`write_per_step_attention`; the `snapshotTraces` index) but the heavy traces +
   attention live in JSONL SIDECAR FILES, and today's viewer reads only RECORDS (no file-fetch bridge verb) — so
   this is blocked in the hub model. Once BlackSwan is its own single-purpose app (steps 1–4), the served viewer
   bytes AND those sidecars both live inside BlackSwan's checkout, so the viewer can fetch the sidecars DIRECTLY
   (relative fetch from the same served tree) with NO host→viewer file-read primitive. Then: fetch + animate/scrub
   the per-step attention over the rollout, and lazily fetch two snapshots' traces → feed the shipped
   `DecisionTraceDiff`. (This is the whole of the old A6 that was blocked — the rest of A6 shipped.)

---

## B. Deferred — bigger implementation projects

### B1. Multi-asset portfolio / cross-sectional long-short — a SEPARATE project

The one genuine project split. BlackSwan's single-asset env hardcodes one asset everywhere, so multi-asset
needs a fundamentally different env: a **3-D observation** (`asset × lookback × features`), a **portfolio
action space** (per-asset long/short/weight), a **timestamp-aligning N-symbol data provider** (misalignment =
silent P&L corruption — unit-test against a 3-coin × 100-bar fixture), and a **scorecard for a portfolio**
(reward = portfolio return minus fees + a correlation-penalty CONSTRAINT; fitness = Sharpe/Calmar — not a
"Sharpe objective" that steers directly). REUSES BlackSwan's reward components, feature engineering, SB3 algos,
walk-forward harness; ~3–4 week project, blocks none of the in-place wins. Research calls cross-sectional
long-short the strongest-edge config — promising, deliberately sequenced last.

### B2. Position-blind signal model — a SEPARATE objective

The trading line trains a position MANAGER (actions are position-gated; out-of-position output is never shaped
— `blocked_signal_ratio` ~0.95). A model whose raw per-step output IS a long/flat(/short) signal, independent
of position, is a **different objective** (the "Case-1" signal case). Probe it cheaply FIRST via the shipped
forward-horizon signal lens (`signal_expectancy`/`_hit_rate`/`_coverage`) on existing manager checkpoints —
build this full project only if that lens shows a generalisable per-signal edge. First confirm the clean manager (via `combo_noop_penalty`, which
drops `blocked_signal_ratio` ~0.95→~0.02 by going silent on no-ops) isn't already enough. Then: reuse the env's
`buy_sell_signal` reward family or the supervised direction head; its OWN objective (signal precision/recall/
coverage vs realized forward returns, or a signal-following backtest); likely its OWN manifest
(`trainer-signal.json`); its own verdict rule + signal-overlay chart (the verdict+objective plumbing is the
bulk, not the model). Runs with `combo_noop_penalty>0` are CLEAN-MANAGER runs — don't migrate them; represent
the split as a DERIVED `approach` facet. Compare manager vs forecaster on a SHARED signal-following backtest.

### B3. Data platform — remaining source coverage + productionization

**Remaining source coverage** — the lower-noise asset-class line (this supersedes the old "strategic
direction": the mine that enables it is largely built). Macro `rates`/`macro_core`, `majors`, `market`, and
stocks are already mined + tradeable (context/data-fusion + the shipped stocks class). Left: make **commodities
+ FX** tradeable classes (catalog + `asset` lever + backfill, per the shipped stocks pattern — absorbs the old
"other single assets" item) and mine the **fundamental panels crypto lacks** — earnings, COT/inventories, trade
balances, an event calendar — each under the correctness rules below. The edge-search across these classes is
the exploration autopilot's job + the Diagnosis cross-class read, not a build here.

**Productionization (D5) — only when a SECOND trainer needs the same data:** extract catalog+miners+CLI to a
standalone `thefactory-datamine`; generalize `derive_cache` to a central 1m→fidelities service; wire the
`ContentAddressedDataCache` + remote-runner data path from one curated origin.

**Data correctness rules (enforce in the loader, for all future mining):** store minimal raw only (derive at
runtime); **join by TIMESTAMP, never date string** (each row carries `barCloseTz`); macro is point-in-time
(ALFRED vintages, stamp at the real per-release datetime, forward-fill, MoM/YoY = diff of the SAME vintage;
daily rates that exceed the vintage cap fetch standard obs stamped as-of since they're non-revised); post-close
series (H.15 ~16:15 ET) publish next session; fundamentals stamp at filing/acceptance not period-end
(restatements = new rows); commodity continuous roll is a look-ahead machine (back-adjust with roll dates); FX
`Volume≡0` is a constant not data (neutral-fill; 6-char `=X`); never forward-fill one leg of a ratio;
idempotent + validated mining (monotonic timestamps, positive prices, split/div-adjust); licence-gate shareable
output (only Frankfurter/ECB is redistribution-safe).

### B4. Code-change risk model — a third ML consumer (research first)

A trainer-conformant project scoring an agent's diff/PR by bug-likelihood. (1) Research: survey public
JIT-defect datasets vs mining our own from the `thefactory-*` git histories → cited report + go/no-go.
(2) Data (via the mine): SZZ-style labeling + codeIntel features (churn, complexity, coverage, diff size).
(3) Train: a `risk-classifier` project (objective = AUC / precision-at-k). (4) Consume: wire the score into
the review / expert-panel / verifier path. _Further out: a FastContext-style repo-explorer subagent — gated by
the same go/no-go + LLM-training compute the ComputeRunner has never exercised._

### B5. Optional + small deferred

- **Live handoff** — tag the exploration autopilot's global-max checkpoint for live trading (`run_server_model.py`).
- **Jupyter notebooks (UNDERSCOPED)** — view/edit/execute a project's `.ipynb`; scope kernel location + security.
- **Runner-channel WebSocket upgrade** — dispatch is already ~instant; a WS only shaves ~1.5s log latency.
- **Remote git repoRefs** — wire git refs + project bootstrap when a real remote machine needs it.

## C. Open questions + trigger-blocked builds (act when the dependency lands)

- **Context — `with_extra_data` projection rung** (blocked on data). The 4th projection rung: an asset's OWN
  fused series + the obs-signature gate as a REAL replay guard. Every context panel today is GLOBAL
  (macro/majors/market); no per-asset series is mined (`fundamentals/`, EDGAR are empty). The fusion substrate
  + loader already exist, so the rung is small — build it once ≥1 asset-specific series is mined.
- **Context — post-release window restriction** in `config_builder` (blocked + currently a no-op). Bar
  pre-release context bars via `fillna(0)`. Every context series on disk is deep-history (FRED to 1776;
  GOLD/SPY from 2018), so `fillna(0)` never zeroes a real signal — build it when a LATE-STARTING context series
  is mined.
- **Host→iframe `data:updated` push channel** — the viewer is poll-only; the bridge forwards only `nav.open`.
  A push channel (thefactory-ui `appBridge` + `ProjectAppView` + `bridge.js`) would let the viewer live-refresh
  on a data change instead of polling. Unblocks the cross-test (`-settest`) auto-refresh (the viewer half is a
  trivial `onDataUpdated` handler debouncing a `readCrossTests()`) + trims poll latency generally.
- **Remote artifact/checkpoint storage** — keep-on-runner + reference vs upload; how a winning remote
  checkpoint reaches the live trading server. Meaningful once remote runs AND live handoff both exist.
- **GPU + sandbox profile for training images** — `--read-only` rootfs vs ML caches; `--gpus` is wired but
  unexercised (the runner runs jobs directly, not through Docker-sandboxed `SandboxTools`).
- **Judge/proposer model transport** — `ModelSelection` (API vs CLI), overtaken by the in-flight refactor.
  Revisit once the CLI inference stage lands.
