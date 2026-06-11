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
}

/** The single north-star metric a run is judged by. */
export interface TrainerObjective {
  /** Metric name as reported in the RunSummary `objective` field, e.g. `eval_return_mean`. */
  name: string
  /** Whether higher (`max`) or lower (`min`) is better. */
  direction: 'max' | 'min'
}

/** A dataset a training project needs; consumed by the data cache + proxy allowlist (Phase 6). */
export interface TrainerDataRequirement {
  id: string
  files?: string[]
  glob?: string
  source?: string
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
  data?: TrainerDataRequirement[]
  resources?: TrainerResources
  /** Reproducible run image (Phase 6 remote runners). */
  image?: string
}

/** Which lever values a campaign explores. Levers absent from both fall back to their defaults. */
export interface ExperimentSpec {
  /** Per-lever value lists; the planner takes the cartesian product. */
  sweep?: Record<string, unknown[]>
  /** Per-lever pinned values overriding defaults. */
  fixed?: Record<string, unknown>
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
  /** Present on calibrate runs: throughput for ETA math. */
  calibration?: { unitsPerSecond?: number; secondsObserved?: number; units?: number }
  /** Optional per-run curves (e.g. `episode_return`) for the viewer's training-curve chart. */
  series?: Record<string, number[]>
  /** Present on evaluate runs: which checkpoint was re-tested and how hard. */
  evaluation?: { checkpoint?: string; episodes?: number }
}

export interface EvaluateTrainingRunParams {
  scope: string
  projectRoot: string
  manifest?: TrainerManifest
  /** Key of the completed run record whose checkpoint gets re-evaluated. */
  runKey: string
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

/** Streamed campaign progress — written to the `{recordType}-progress` record by the host activity. */
export interface TrainingCampaignProgress {
  phase: 'calibrate' | 'train' | 'done'
  done: number
  total: number
  skipped: number
  failed: number
  etaSeconds?: number
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
  spec: ExperimentSpec
  /** Re-run items that already have a completed record. */
  refresh?: boolean
  /** Provenance label stamped on each run record (e.g. a compute target name). */
  ranBy?: string
  abortSignal?: AbortSignal
  onProgress?: (progress: TrainingCampaignProgress) => void | Promise<void>
  /** Fired after each run record upsert so the host can broadcast `data:updated`. */
  onRecordWritten?: (type: string, key: string) => void
}

export interface TrainingCampaignResult {
  recordType: string
  planned: number
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
  llmConfig: LLMConfig
  /** Extra rubric appended to the judge prompt. */
  instructions?: string
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
  readTrainerManifest(projectRoot: string): Promise<TrainerManifest>
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
