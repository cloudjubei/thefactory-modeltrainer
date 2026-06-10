import type { ComputeRunner, DataStorage } from 'thefactory-tools/types'

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
  logger?: TrainerLogger
  /** Injectable clock for deterministic tests; defaults to ISO now. */
  now?: () => string
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
}
