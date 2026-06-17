import type {
  ComputeRunner,
  DataStorage,
  InferenceExecutor,
  LLMConfig,
} from 'thefactory-tools/types'

/** Throughput measured by a calibrate run, plus the campaign ETA derived from it. */
export interface TrainingCalibration {
  secondsObserved: number
  unitsPerSecond?: number
  /** Predicted whole-campaign duration when total units and throughput are known. */
  etaSeconds?: number
}

/** One sweepable lever in a {@link TrainerManifest}: the launch form renders it, the planner sweeps it. */
export interface TrainerLeverSpec {
  /** Input kind: `number` (numeric input), `choice` (select over `choices`), `boolean` (checkbox). */
  type: 'number' | 'choice' | 'boolean'
  /** Default value used when neither `fixed` nor `sweep` provides one. */
  default?: unknown
  /** Inclusive [min, max] hint for `number` levers. */
  range?: [number, number]
  /** Allowed values for `choice` levers. */
  choices?: unknown[]
  /** Plain-language explanation shown as a help tooltip in the launch form (for newcomers). */
  description?: string
  /**
   * Whether this lever configures the MODEL (default), the ENVIRONMENT the agent acts in (market
   * mechanics — e.g. fees, take-profit/stop-loss), or the DATASET it trains/tests on (asset, time
   * window, fidelity). Environment and dataset levers are managed as named bundles the hub runs
   * models against, not as ordinary model knobs. Omitted ⇒ `model`.
   */
  scope?: 'model' | 'environment' | 'dataset'
}

/** The single north-star metric a run is judged by. */
export interface TrainerObjective {
  /** Metric name as reported in the RunSummary `objective` field, e.g. `eval_return_mean`. */
  name: string
  /** Whether higher (`max`) or lower (`min`) is better. */
  direction: 'max' | 'min'
}

/** One file a dataset materialises: workspace destination + where to fetch it. */
export interface TrainerDataFile {
  /** Path relative to the project root, e.g. `data/winequality-red.csv`. */
  relPath: string
  url: string
  /** Expected content hash; a fetched object that doesn't match fails the run. */
  sha256?: string
}

/** A dataset a training project declares; the compute runner materialises it before runs. */
export interface TrainerDataRequirement {
  id: string
  files: TrainerDataFile[]
  credentialRef?: string
}

/** Declared machine needs — how the compute target is chosen and sized. */
export interface TrainerResources {
  gpu?: boolean
  memory?: string
  cpus?: number
}

/**
 * The contract a trainer-conformant project ships at `.factory/trainer.json`
 * (see docs/model-training-standard.md). The engine reads ONLY this — no
 * model-specific knowledge lives in the engine.
 */
export interface TrainerManifest {
  name: string
  /** Namespaces every DataStorage record this project produces, e.g. `cartpole-run`. */
  recordType: string
  /**
   * Plain-language description of what this project trains and what "good" means — the domain,
   * the model, the objective in human terms, and any key mechanics. Surfaced to the user and used
   * to brief the in-app chat agent so a discussion is about THIS project, not the generic trainer.
   */
  description?: string
  /** Run-command template containing `{configPath}` and `{summaryOut}` placeholders. */
  run: string
  /** Calibrate-command template containing `{summaryOut}`; omit if the project cannot calibrate. */
  calibrate?: string
  /**
   * Evaluate-command template (same `{configPath}`/`{summaryOut}` contract as `run`); the
   * config it receives carries the original run's levers plus `checkpoint`. Omit if the
   * project cannot re-evaluate saved checkpoints.
   */
  evaluate?: string
  objective: TrainerObjective
  levers: Record<string, TrainerLeverSpec>
  /** Names the lever whose numeric value measures work (e.g. `total_timesteps`) for ETA math. */
  eta?: { unitsLever: string }
  /**
   * A recommended "fast first run" preset: fixed lever values the launch form can pre-fill so a
   * new user's first campaign returns quickly, without changing the tuned best-known defaults.
   */
  quickStart?: { label?: string; fixed: Record<string, unknown> }
  /**
   * Curated launch presets the launch form can load with one pick. A preset is either a single
   * known-good setup (`fixed` lever values) or a whole ready-to-run EXPERIMENT — `sweep` (lever →
   * candidate values), a `seeds` count, and a `thesis`/`thesisTarget` tag — so a designed campaign
   * is point-and-launch, not hand-assembled.
   */
  presets?: Array<{
    label: string
    fixed?: Record<string, unknown>
    sweep?: Record<string, unknown[]>
    seeds?: number
    thesis?: string
    thesisTarget?: string
  }>
  data?: TrainerDataRequirement[]
  resources?: TrainerResources
  /** Reproducible run image (Phase 6 remote runners). */
  image?: string
  /**
   * The CURRENT data/scoring pipeline version. Bump it ONLY on a BREAKING change — one that changes
   * how data is fed or scored, so scores are no longer comparable to prior runs (additive changes
   * like a new model/optimizer option do NOT bump it). Each run is tagged with the version it ran
   * under; a bump invalidates `skipExplored`/`unrunnable` so every setup is re-explored under the
   * new pipeline. Omitted ⇒ treated as `"1"`.
   */
  pipelineVersion?: string
  /** Per-version changelog (newest first), so the hub can show what each version changed. */
  pipelineChangelog?: PipelineVersionEntry[]
}

/** One entry in a project's pipeline changelog. */
export interface PipelineVersionEntry {
  /** The version string this entry documents (matches `TrainerManifest.pipelineVersion` when current). */
  version: string
  /** When the version landed (ISO date), if recorded. */
  date?: string
  /** True when this version changed data/scoring in a way that makes prior scores incomparable. */
  breaking?: boolean
  /** One- or two-line description of what changed in this version. */
  summary: string
}

/** Which lever values a campaign explores. Levers absent from both fall back to their defaults. */
export interface ExperimentSpec {
  /** Per-lever value lists; the planner takes the cartesian product. */
  sweep?: Record<string, unknown[]>
  /** Per-lever pinned values overriding defaults. */
  fixed?: Record<string, unknown>
  /**
   * Environment BUNDLES to run every configuration against — each is a set of (environment) lever
   * values applied together (NOT a cartesian product, unlike `sweep`). Used to test one model
   * across several named environments (e.g. different fee / TP-SL regimes) in a single campaign.
   */
  environments?: Array<Record<string, unknown>>
  /**
   * Dataset BUNDLES to run every configuration against — like {@link environments}, a set of
   * (dataset) lever values applied together (NOT a cartesian product). Used to test one model across
   * several named datasets (e.g. different assets / walk-forward windows / fidelity stacks) in a
   * single campaign. Crosses the model matrix alongside `environments`.
   */
  datasets?: Array<Record<string, unknown>>
  /** Seeds to repeat every configuration with (sets `config.seed`); omit to run each config once. */
  seeds?: number[]
  /** Safety cap overriding the default maximum planned items. */
  maxItems?: number
}

/** One planned unit of work: a fully resolved config and its stable identity. */
export interface PlannedTrainingItem {
  /** Stable hash of the resolved config — the record key and skip-if-fresh identity. */
  key: string
  config: Record<string, unknown>
  /** Value of the manifest's `eta.unitsLever` in this config, when declared and numeric. */
  units?: number
}

/** What data a run trained on — generic across trainer projects, for the hub's data-visibility surface. */
export interface TrainingRunDataset {
  /** Dataset/asset identifier (e.g. a symbol like `BTCUSDT`, or a named dataset). */
  asset?: string
  /** Sampling timeframe / fidelity (e.g. `1d`, `1h`). */
  timeframe?: string
  /** Number of samples/candles the run saw. */
  candles?: number
  /** ISO start of the data window, when the source carries timestamps. */
  from?: string
  /** ISO end of the data window, when the source carries timestamps. */
  to?: string
}

/** Per-exit-reason aggregate in a trading run's behavioural breakdown (sell/cover/tp/trailing/sl/open). */
export interface ExitReasonStats {
  count: number
  wins: number
  win_pct: number
  total_pnl_pct: number
  avg_pnl_pct: number
}

/** One equal-time sub-window of the test span: the market's move next to the model's realized P&L. */
export interface RegimeWindow {
  market_return_pct: number
  realized_pnl_pct: number
  n_trades: number
  win_pct: number
}

/** Per market regime (up/flat/down by trailing trend): realized performance + how much of the window it occupied. */
export interface RegimeTrendStats {
  n_trades: number
  win_pct: number
  realized_pnl_pct: number
  bars_pct: number
}

/** One reconstructed round-trip in a trading run's explainability ledger. */
export interface TradeLedgerEntry {
  entry_step: number
  entry_price: number
  exit_step: number
  exit_price: number
  side: 'long' | 'short'
  reason: 'sell' | 'cover' | 'tp' | 'trailing' | 'sl' | 'open'
  pnl: number
  pnl_pct: number
  bars_held: number
}

/** The machine-readable result a conformant run writes via `--summary-out`. */
export interface TrainingRunSummary {
  /** The objective metric value (matches the manifest's `objective.name`). */
  objective: number
  metrics?: Record<string, number>
  health?: { status: string; flags?: string[] }
  seed?: number
  config?: Record<string, unknown>
  provenance?: Record<string, unknown>
  artifacts?: Record<string, unknown>
  /** What data this run trained on (asset, timeframe, sample count, date span) — for the hub's data-visibility surface. */
  dataset?: TrainingRunDataset
  /** Present on calibrate runs: throughput for ETA math. */
  calibration?: { unitsPerSecond?: number; secondsObserved?: number; units?: number }
  /** Optional per-run curves (e.g. `episode_return`) for the viewer's training-curve chart. */
  series?: Record<string, number[]>
  /** Trading runs: per-exit-reason breakdown (how each position was closed). */
  exits?: Record<string, ExitReasonStats>
  /** Trading runs: skill-vs-luck regime split — equal time windows and trailing-trend buckets. */
  regimes?: { windows?: RegimeWindow[]; trend?: Record<string, RegimeTrendStats> }
  /** Trading runs: reconstructed round-trips (entry/exit, reason, P&L) — the explainability ledger. */
  ledger?: TradeLedgerEntry[]
  /** Present on evaluate runs: which checkpoint was re-tested and how hard. */
  evaluation?: { checkpoint?: string; episodes?: number }
}

export interface EvaluateTrainingRunParams {
  scope: string
  projectRoot: string
  manifest?: TrainerManifest
  /** Manifest file relative to `projectRoot` (default `.factory/trainer.json`); names a second conformant line in the same repo. */
  manifestRelPath?: string
  /** Key of the completed run record whose checkpoint gets re-evaluated. */
  runKey: string
  /** Named compute target to evaluate on; omit for the default (local) runner. */
  computeTarget?: string
  abortSignal?: AbortSignal
  /** Fired after the evaluation record upsert so the host can broadcast `data:updated`. */
  onRecordWritten?: (type: string, key: string) => void
}

export interface EvaluateTrainingRunResult {
  recordType: string
  runKey: string
  /** The evaluation's objective value (e.g. mean return over the eval episodes). */
  objective: number
  evaluatedAt: string
}

export interface EvaluateTrainingRunsParams {
  scope: string
  projectRoot: string
  manifest?: TrainerManifest
  /** Manifest file relative to `projectRoot` (default `.factory/trainer.json`). */
  manifestRelPath?: string
  /** Keys of completed run records whose checkpoints get re-evaluated, in parallel up to `concurrency`. */
  runKeys: string[]
  /** Max evaluations dispatched at once (a bounded pool); defaults to 1 (sequential). */
  concurrency?: number
  /** Named compute target to evaluate on; omit for the default (local) runner. */
  computeTarget?: string
  abortSignal?: AbortSignal
  /** Fired after each evaluation record upsert so the host can broadcast `data:updated`. */
  onRecordWritten?: (type: string, key: string) => void
  /** Streamed cumulative progress as evaluations settle. */
  onProgress?: (progress: { done: number; total: number; failed: number }) => void
}

export interface EvaluateTrainingRunsResult {
  recordType: string
  /** Number of runs evaluated successfully. */
  evaluated: number
  /** Number of runs whose evaluation threw. */
  failed: number
  /** The per-run results, one per successful evaluation. */
  results: EvaluateTrainingRunResult[]
  /** The runs that failed, with their error; omitted when none failed. */
  failures?: { runKey: string; error: string }[]
}

/** Streamed campaign progress — written to the `{recordType}-progress` record by the host activity. */
export interface TrainingCampaignProgress {
  phase: 'calibrate' | 'train' | 'done'
  done: number
  total: number
  skipped: number
  failed: number
  etaSeconds?: number
  /** True when `etaSeconds` is a coarse wall-clock estimate (no calibration), so the UI can mark it approximate. */
  etaApprox?: boolean
  /** Key of the most recently finished item. */
  lastKey?: string
}

export interface TrainingCampaignParams {
  /** DataStorage scope (the projectId). */
  scope: string
  /** Absolute path of the trainer-conformant project checkout. */
  projectRoot: string
  /** Pre-read manifest; omitted → read from `projectRoot`. */
  manifest?: TrainerManifest
  /** Manifest file relative to `projectRoot` (default `.factory/trainer.json`); names a second conformant line in the same repo. */
  manifestRelPath?: string
  spec: ExperimentSpec
  /** Re-run items that already have a completed record. */
  refresh?: boolean
  /**
   * Maximum number of runs dispatched at once (default 1, sequential). Each run is
   * isolated (unique jobId + its own temp config/summary), so the real cap is host
   * CPU/GPU/RAM — this is the safety valve.
   */
  concurrency?: number
  /**
   * Exploration mode: skip any planned item whose SETUP (config minus seed) was already
   * run under any seed, so an overlapping sweep doesn't re-run setups expected to produce
   * similar results. Turn off when homing in (to run a setup across multiple seeds).
   */
  skipExplored?: boolean
  /**
   * Free-text experiment/thesis this campaign tests (e.g. "fee-penalty reward"), stamped on every
   * run so the hub can group + compare runs by experiment — even theses outside the lever set
   * (a new data prep, a code change). Optional.
   */
  thesis?: string
  /** The lever this thesis is varying, when it maps to one (for the by-experiment view to highlight). */
  thesisTarget?: string
  /**
   * Named compute target to run on (resolved via the deps' `resolveComputeRunner`);
   * omit for the default (local) runner. Also the provenance label.
   */
  computeTarget?: string
  /** Provenance label stamped on each run record; defaults to the compute target or `local`. */
  ranBy?: string
  abortSignal?: AbortSignal
  onProgress?: (progress: TrainingCampaignProgress) => void | Promise<void>
  /**
   * Fired with sub-progress for the item currently running — `phase: 'starting'`
   * when an item begins, then whatever the trainer streams via `@@PROGRESS`
   * markers (e.g. `{ phase: 'train', done, total }`). Lets the host show real
   * within-run progress for long, data-driven runs.
   */
  onItemProgress?: (key: string, progress: Record<string, unknown>) => void | Promise<void>
  /** Fired after each run record upsert so the host can broadcast `data:updated`. */
  onRecordWritten?: (type: string, key: string) => void
}

export interface TrainingCampaignResult {
  recordType: string
  planned: number
  /** Stable keys of every planned item — the campaign's run-record identities. */
  keys: string[]
  completed: number
  skipped: number
  failed: number
  aborted: boolean
  /** Per-item failure reasons; absent when every run succeeded. */
  failures?: { key: string; error: string }[]
  /** Best completed run across ALL stored records of this type (not just this campaign). */
  bestKey?: string
  bestObjective?: number
  direction: 'max' | 'min'
  calibration?: TrainingCalibration
  finishedAt: string
}

export interface CalibrateTrainingParams {
  projectRoot: string
  manifest: TrainerManifest
  abortSignal?: AbortSignal
}

export interface TrainerLogger {
  info: (message: string, meta?: Record<string, unknown>) => void
  warn: (message: string, meta?: Record<string, unknown>) => void
}

export interface ModelTrainerToolsDeps {
  computeRunner: ComputeRunner
  /** Resolve a named compute target (e.g. a paired remote runner) to its runner. */
  resolveComputeRunner?: (target: string) => ComputeRunner | undefined
  storage: DataStorage
  /** Required for judging/proposing; the train/calibrate surface works without it. */
  inferenceExecutor?: InferenceExecutor
  logger?: TrainerLogger
  /** Injectable clock for deterministic tests; defaults to ISO now. */
  now?: () => string
}

/** One judged run: the deterministic objective blended with the LLM's verdict. */
export interface TrainingVerdict {
  /** The judged run record's key. */
  key: string
  /** Blended 0–100 score the viewer ranks by. */
  score: number
  /** Min–max-normalised objective (0–100, direction-aware). */
  objectiveScore: number
  /** The LLM's raw 0–100 score; absent when the model returned no row for this run. */
  llmScore?: number
  why: string
  /** Health-flagged runs are auto-rejected without consulting the LLM. */
  rejected?: boolean
  /** Provenance label of the judging model. */
  judgedBy?: string
  judgedAt: string
}

export interface JudgeTrainingRunsParams {
  scope: string
  projectRoot: string
  manifest?: TrainerManifest
  /** Manifest file relative to `projectRoot` (default `.factory/trainer.json`); names a second conformant line in the same repo. */
  manifestRelPath?: string
  llmConfig: LLMConfig
  /** Extra rubric appended to the judge prompt. */
  instructions?: string
  /** Restrict judging to these run keys (the selection); when omitted, judge every completed run in scope. */
  runKeys?: string[]
  abortSignal?: AbortSignal
  /** Fired after each verdict record upsert so the host can broadcast `data:updated`. */
  onRecordWritten?: (type: string, key: string) => void
}

export interface JudgeTrainingRunsResult {
  recordType: string
  judged: number
  rejected: number
  verdicts: TrainingVerdict[]
  judgedBy: string
  judgedAt: string
}

/** A proposed experiment in the durable backlog — nothing gets lost between sessions. */
export interface TrainingHypothesis {
  /** Stable hash of the proposed spec — identical proposals dedupe. */
  id: string
  title: string
  rationale: string
  spec: ExperimentSpec
  status: 'pending' | 'accepted' | 'rejected'
  source: 'human' | 'llm'
  /** Provenance label of the proposing model (absent for human entries). */
  proposedBy?: string
  createdAt: string
  updatedAt: string
}

export interface ProposeTrainingHypothesesParams {
  scope: string
  projectRoot: string
  manifest?: TrainerManifest
  /** Manifest file relative to `projectRoot` (default `.factory/trainer.json`); names a second conformant line in the same repo. */
  manifestRelPath?: string
  llmConfig: LLMConfig
  /** How many proposals to ask for; defaults to {@link DEFAULT_HYPOTHESIS_COUNT}. */
  count?: number
  /** Extra guidance appended to the proposer prompt. */
  instructions?: string
  abortSignal?: AbortSignal
  /** Fired after each hypothesis record upsert so the host can broadcast `data:updated`. */
  onRecordWritten?: (type: string, key: string) => void
}

export interface ProposeTrainingHypothesesResult {
  recordType: string
  proposed: number
  /** Proposals whose spec already exists as a hypothesis record (any status). */
  skippedExisting: number
  hypotheses: TrainingHypothesis[]
  proposedBy: string
  proposedAt: string
}

/**
 * The model-training toolset: plans experiment matrices from a TrainerManifest,
 * runs them through a ComputeRunner, and persists each RunSummary as a record.
 * Domain-oblivious — everything model-specific arrives via the manifest + spec.
 */
export interface ModelTrainerTools {
  /** Read + validate a project's `.factory/trainer.json`. */
  readTrainerManifest(projectRoot: string, manifestRelPath?: string): Promise<TrainerManifest>
  /** Expand a spec into the fully resolved, stably keyed work items. */
  planTrainingMatrix(manifest: TrainerManifest, spec: ExperimentSpec): PlannedTrainingItem[]
  /** Run the project's calibrate command; `undefined` when the manifest declares none. */
  calibrateTrainingThroughput(
    params: CalibrateTrainingParams,
  ): Promise<TrainingCalibration | undefined>
  /** Plan → skip-if-fresh → run each item → persist records → report progress. */
  runTrainingCampaign(params: TrainingCampaignParams): Promise<TrainingCampaignResult>
  /**
   * Re-test a completed run's saved checkpoint via the manifest's `evaluate`
   * command, persisting the result as a `{recordType}-evaluation` record.
   */
  evaluateTrainingRun(params: EvaluateTrainingRunParams): Promise<EvaluateTrainingRunResult>
  /**
   * Re-test many completed runs' checkpoints in parallel (a bounded pool),
   * persisting one `{recordType}-evaluation` record per run; a failure isolates
   * to its run and is reported in `failures` without stopping the rest.
   */
  evaluateTrainingRuns(params: EvaluateTrainingRunsParams): Promise<EvaluateTrainingRunsResult>
  /**
   * Score every completed run: auto-reject health-flagged ones, blend the
   * normalised objective with an LLM verdict, persist `{recordType}-verdict` records.
   */
  judgeTrainingRuns(params: JudgeTrainingRunsParams): Promise<JudgeTrainingRunsResult>
  /**
   * Ask an LLM for the next experiments given run history + verdicts; persist new
   * `{recordType}-hypothesis` records (deduped by spec hash, existing statuses kept).
   */
  proposeTrainingHypotheses(
    params: ProposeTrainingHypothesesParams,
  ): Promise<ProposeTrainingHypothesesResult>
}
