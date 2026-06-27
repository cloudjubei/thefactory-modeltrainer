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
  /**
   * Conditional applicability: this lever only applies when ANOTHER lever holds one of the listed
   * values — e.g. `{ reward_model: ['combo_all_noop'] }` means the lever is relevant only for that
   * reward model. The launch form disables (greys out) the lever when the condition isn't met, so a
   * setting that some models have and others don't isn't swept/pinned where it does nothing. ALL named
   * keys must match (AND). Omitted ⇒ always applies.
   */
  appliesWhen?: Record<string, unknown[]>
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
  /**
   * Command template (only `{summaryOut}`) that benchmarks ONE model on CPU vs MPS and writes a
   * `deviceBenchmark` summary (best device + per-device us/step). The model to benchmark is passed via the
   * `BENCH_MODEL_NAME` env var. Powers the Models tab's per-model "Benchmark device" button.
   */
  benchmarkDevice?: string
  /**
   * How many math threads ONE run of this project wants (its `run` command's per-process thread cap).
   * When set, a campaign with no explicit `concurrency` packs `floor(hostCpus / maxThreadsPerRun)` runs
   * in parallel (instead of the safe sequential default that leaves cores idle), and exports this many
   * threads into each run's process env so N runs × this ≈ host cores. Omit to keep the sequential default.
   */
  maxThreadsPerRun?: number
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
    /** Dataset bundles to run the preset across (each applied together), e.g. walk-forward windows. */
    datasets?: Array<Record<string, unknown>>
    /** Environment bundles to run the preset across (each applied together), e.g. exit/fee regimes. */
    environments?: Array<Record<string, unknown>>
    seeds?: number
    thesis?: string
    thesisTarget?: string
  }>
  /** Starter approaches/papers the viewer imports into the Papers registry once (keyed by id). */
  papers?: TrainingPaperSeed[]
  /** Starter models the viewer imports into the Models catalog once (keyed by slug id). */
  models?: TrainingModelSeed[]
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
  /**
   * One-time, idempotent record migrations (see {@link TrainerMigrationRule}). The host's boot-time
   * migration sweep applies them via {@link ModelTrainerTools.migrateTrainingRuns} to every stored run
   * (and pending-queue) config so a config-shape change — e.g. collapsing several reward-model names
   * into one parameterised model, or pruning retired ones — is rolled forward across the whole history
   * without re-running anything. Applying twice is a no-op (rules only match the OLD shape).
   */
  migrations?: TrainerMigrationRule[]
  /**
   * One-time run-validity invalidations applied by the host's boot sweep (see {@link RunInvalidationRule}
   * and {@link ModelTrainerTools.invalidateRuns}): mark runs produced by a since-fixed code bug as
   * `status: 'invalid'` (so they stop counting toward any aggregation/xAI) and optionally cancel matching
   * pending-queue items. Gated by pipeline major, so re-runs with the fix are never re-flagged.
   */
  runInvalidations?: RunInvalidationRule[]
}

/**
 * A declarative run-invalidation. Runs produced BEFORE a behaviour fix shipped (pipeline major <
 * `beforePipelineMajor`) that ALSO match the bug's affected set (resolved from `kind` by the host to a
 * predicate, e.g. `fidelityDesync`) are marked `status: 'invalid'` with `reason` — excluded from every
 * aggregation thereafter. When `cancelPendingQueue` is set, matching pending-queue items are removed ONCE
 * (guarded by a marker record keyed on `id`) so post-fix re-queued runs survive. Idempotent.
 */
export interface RunInvalidationRule {
  id: string
  kind: string
  beforePipelineMajor: number
  reason: string
  cancelPendingQueue?: boolean
}

/**
 * A single declarative migration rule. A record is handled by the FIRST rule it matches — `match`
 * fields must all be PRESENT and loosely equal (so JSON `0` matches a stored `0`/`"0"`), and `matchNot`
 * fields must all be PRESENT and loosely UNEQUAL (so a record missing the field is never matched by a
 * `matchNot` clause — runs without that key are left alone). A rule needs at least one of `match` /
 * `matchNot`; one with neither matches nothing.
 *
 * A matched rule either REWRITES the record (`set` + `keepOrDefault`: the new config is
 * `{...config, ...set}`, then each `keepOrDefault` key keeps the config's current value when present
 * else the default) or, when `delete` is true, REMOVES the record entirely. When no rule matches, the
 * record is left untouched — so applying the rules twice is a no-op (idempotency).
 */
export interface TrainerMigrationRule {
  /** Field/value pairs the config must all have AND equal for this rule to fire. */
  match?: Record<string, unknown>
  /** Field/value pairs the config must all have AND NOT equal (a negative match, e.g. "not combo_unified"). */
  matchNot?: Record<string, unknown>
  /** Field values written unconditionally onto the migrated config (rewrite rules only). */
  set?: Record<string, unknown>
  /** Field → fallback: keep the config's current value when present, else write this default. */
  keepOrDefault?: Record<string, unknown>
  /** Config keys to REMOVE (e.g. retiring a dead lever). On its own (no match/matchNot) the rule fires
   * for any config that still has one of these keys, then stops — so the sweep is idempotent. */
  unset?: string[]
  /** When true, a matched record is DELETED rather than rewritten (irreversible). */
  delete?: boolean
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
   * Explicit runs to plan VERBATIM (no sweep/bundle/seed expansion). When present and non-empty it DEFINES
   * the matrix (`sweep`/`seeds` are ignored). Each entry's `config` is merged onto the lever defaults; its
   * optional `key` becomes the planned item's record key UNCHANGED — so re-running an EXISTING run (passing
   * its record key) updates that same record in place instead of hashing a fresh, divergent key (the stored
   * config rarely re-hashes to the original key). Omit `key` to hash the merged config like the rest of the
   * matrix. Used to re-run a SET of runs as ONE campaign/activity — batch version-upgrade or failure-retry.
   * Each entry's `config` is migrated like {@link ExperimentSpec.fixed}; its `key` is preserved.
   */
  configs?: Array<{ config: Record<string, unknown>; key?: string }>
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

/** Aggregate input attribution for a {@link DecisionTrace} — generic saliency over the model's observation. */
export interface DecisionFeatureAttribution {
  /** Mean absolute saliency per observation-feature index (length = the observation dimension). */
  perFeature?: number[]
  /**
   * Saliency aggregated into named groups — domain-defined (e.g. the trading line groups by the
   * observation's fidelity layer), so the Explain view can show which input GROUP drove decisions.
   */
  byGroup?: Record<string, number>
  /** How attribution was computed, e.g. `gradient-saliency`. */
  method?: string
  /** Number of decision steps the attribution was averaged over. */
  samples?: number
  /**
   * The Adebayo "Sanity Checks for Saliency Maps" model-randomization test: a FAITHFUL attribution
   * changes when the model's weights are randomized. Lets the viewer warn when the saliency reflects the
   * input/architecture rather than what the model learned, instead of trusting a plausible-but-unfaithful map.
   */
  sanityCheck?: {
    /** How the check was run, e.g. `model-randomization`. */
    method?: string
    /** Rank correlation between the real and weight-randomized saliency — low = faithful. */
    rankCorrelation?: number
    /** True when the saliency passed (changed enough under randomization to be trusted). */
    passed?: boolean
  }
}

/**
 * One step in a model's {@link DecisionTrace}. Domain-oblivious — the action is an arbitrary
 * project-defined label (no trading vocabulary in the engine), optionally annotated with how confident
 * the policy was and what it nearly did instead, so the Explain view can answer "what did it do, and why".
 */
export interface DecisionStep {
  /** Step index within the rollout, aligned to the run's `series`/`artifacts.runChart` step axis. */
  step: number
  /** The action the policy took, as a project-defined label (e.g. `hold`, `buy`, `sell`). */
  action: string
  /** Confidence in the chosen action in [0,1] (softmax over values / chosen probability), when derivable. */
  confidence?: number
  /** Per-action value the policy assigned (Q-values or action probabilities), keyed by action label. */
  actionValues?: Record<string, number>
  /** The runner-up action (second-highest value) — what the model nearly did instead. */
  alternativeAction?: string
  /** True when the environment FORCED this action (e.g. an automatic exit), so it was not the policy's choice. */
  forced?: boolean
  /** The reward the environment returned for this step, when recorded. */
  reward?: number
  /** A free-form environment-state label at this step (e.g. position state); project-defined. */
  state?: string
  /** Raw observation the policy saw; usually only in the full sidecar, omitted from the embedded trace. */
  features?: number[]
}

/**
 * A model's decision trace over a test/eval rollout — the explainability artifact stored at
 * `summary.artifacts.decisionTrace`. Domain-oblivious: `steps` carry arbitrary action labels and
 * `actionCounts` tallies them so the Explain view flags distribution anomalies generically. The embedded
 * trace is downsampled to the run's chart step axis; the full per-step trace (with raw observations) is an
 * optional sidecar file the run may also write, referenced by `artifacts.decisionTraceFile`.
 */
export interface DecisionTrace {
  /** Per-step decisions, downsampled to the chart step axis so the Explain timeline aligns to price/equity. */
  steps: DecisionStep[]
  /** Count of every action label over the FULL rollout (not just the downsampled `steps`). */
  actionCounts?: Record<string, number>
  /** Aggregate input attribution for the run's decisions, when computed. */
  featureAttribution?: DecisionFeatureAttribution
  /** Total steps in the full rollout before downsampling (so the view can show "N steps, showing M"). */
  totalSteps?: number
  /**
   * Named additive contributions to the run's total reward ("why this reward") — each value is a signed
   * contribution and they sum to `total`. Domain-defined (e.g. the trading line splits base earnings from
   * the turnover/no-op penalties), so the Explain view can show what drove vs dragged the reward.
   */
  rewardBreakdown?: Record<string, number>
  /** A 2-D projection of the policy's internal (penultimate-layer) representation — how it organises states. */
  latentMap?: DecisionLatentMap
}

/** One projected state in a {@link DecisionLatentMap}: its 2-D coordinates + the action taken there. */
export interface DecisionLatentPoint {
  x: number
  y: number
  /** The action label chosen at this state (the point's colour). */
  action: string
}

/**
 * A deterministic 2-D projection (PCA) of the policy's penultimate-layer activations over the rollout —
 * the model's INTERNAL state representation. Clusters by action reveal how it organises states by
 * decision. Domain-oblivious: arbitrary action labels.
 */
export interface DecisionLatentMap {
  /** The projected states (downsampled to the chart cap), coloured by their action. */
  points: DecisionLatentPoint[]
  /** Fraction of the activation variance the two projection axes capture, in [0,1]. */
  varianceExplained?: number
  /** The original activation dimensionality (before projection). */
  dim?: number
  /** How the projection was computed, e.g. `pca`. */
  method?: string
  /**
   * A linear PROBE (Alain & Bengio) of the latent: how well a linear classifier predicts the action from
   * the activations. `accuracy` well above `baseline` (the majority-class rate) ⇒ the representation
   * linearly encodes the decision.
   */
  probe?: {
    /** Held-out accuracy of the linear probe, in [0,1]. */
    accuracy?: number
    /** Majority-class accuracy on the held-out set — the trivial baseline to beat. */
    baseline?: number
    /** Number of action classes probed. */
    classes?: number
    /** How the probe was fit, e.g. `ridge-linear`. */
    method?: string
    /** Held-out sample count the accuracy rests on. */
    testSize?: number
  }
}

/** One aligned step in a {@link DecisionTraceDiff}: what each run did at the SAME step index, with per-step deltas. */
export interface DecisionStepDelta {
  /** Shared step index on the downsampled axis (present in BOTH traces). */
  step: number
  /** Baseline run's action label at this step. */
  baselineAction: string
  /** +Tweak run's action label at this step. */
  tweakAction: string
  /** True when the two runs took different actions here (a divergence point). */
  changed: boolean
  /** `tweak.reward − baseline.reward` at this step, when BOTH recorded a reward; the per-step quality delta. */
  rewardDelta?: number
  /** `tweak.confidence − baseline.confidence`, when both recorded confidence. */
  confidenceDelta?: number
}

/**
 * A HEURISTIC, explicitly-caveated read of whether a tweak's DECISION changes look better — never a causal
 * claim. Rests on the realized per-step reward AT the steps where the decision changed, controlled against
 * the reward shift on steps where it did NOT change (so a whole-rollout regime move isn't mistaken for a
 * decision improvement).
 */
export interface DecisionQualitySignal {
  /** Number of CHANGED (divergent) steps for which BOTH runs recorded a reward — the population this read rests on. */
  scoredChangedSteps: number
  /** Mean `rewardDelta` over CHANGED steps. `>0` ⇒ where decisions changed, the tweak earned more per-step reward. */
  meanRewardDeltaOnChanges?: number
  /** Mean `rewardDelta` over UNCHANGED aligned steps — the CONTROL: a real decision gain shows AT the changes, not everywhere. */
  meanRewardDeltaOnUnchanged?: number
  /** Coarse verdict, confounds folded in. */
  verdict: 'better' | 'worse' | 'mixed' | 'unchanged' | 'insufficient'
  /** One-line plain read that NEVER claims causation; always carries "heuristic, not causal". */
  summary: string
}

/**
 * A step-by-step counterfactual diff of two runs' {@link DecisionTrace}s that share a dataset/window
 * (baseline vs +tweak) — the "did this new information change the model's DECISIONS, and for the better?"
 * read. Domain-oblivious (arbitrary action labels). The decision-quality verdict is kept deliberately
 * SEPARATE from the objective so a tweak can read positive on decisions even when the score hasn't moved.
 */
export interface DecisionTraceDiff {
  /** Whether the traces are step-alignable (same dataset signature + same `totalSteps` + a shared step axis). */
  aligned: boolean
  /** Why not aligned, when `aligned` is false (e.g. "different dataset", "different totalSteps", "no shared steps"). */
  alignmentNote?: string
  /** The dataset signature both runs share, echoed for display. */
  datasetSignature?: string
  /** Count of step indices present in BOTH downsampled traces — the diffable population. */
  alignedSteps: number
  /** Count of aligned steps where the action differs. */
  changedSteps: number
  /** `changedSteps / alignedSteps` in [0,1] — how much the tweak moved decisions. */
  divergenceRate: number
  /** Per-aligned-step deltas, in step order. */
  steps: DecisionStepDelta[]
  /** Per-action FULL-rollout count delta (`tweak.actionCounts − baseline.actionCounts`), non-zero entries only. */
  actionCountDeltas: Record<string, number>
  /** Mean `confidenceDelta` over steps where both have confidence; undefined when unavailable. */
  meanConfidenceShift?: number
  /** `tweak.objective − baseline.objective` — CONTEXT only, NOT the decision-quality verdict. */
  objectiveDelta?: number
  /** The decision-quality read, kept deliberately separate from the objective. */
  quality: DecisionQualitySignal
}

/**
 * The minimal, domain-oblivious run shape the xAI config-effect engine analyses — projected from a
 * stored run record. Only completed runs (with a numeric criterion value) participate.
 */
export interface AnalysisRun {
  key: string
  /** The resolved lever values that produced the run (the config-level "knobs"). */
  config: Record<string, unknown>
  metrics?: Record<string, number>
  objective?: number
  /** Wall-clock duration (the `runtime` criterion). */
  durationMs?: number
  seed?: number
  dataset?: TrainingRunDataset
  /** Only `completed` runs are analysed. */
  status?: string
  /** When the run completed (ISO), for the time-ordered convergence series. */
  ranAt?: string
  /** On an AGGREGATED setup (from {@link aggregateToSetupRuns}): bootstrap CI of the criterion's IQM. */
  ci?: [number, number]
  /** On an AGGREGATED setup: how many distinct seeds were folded in. */
  seeds?: number
}

/**
 * A criterion to analyse runs by — the metric and which direction is "better". Drives the xAI
 * config-effect engine and viewer (selectable in the UI: objective / any `metrics.*` key / runtime).
 */
export interface AnalysisCriterion {
  /** Metric key: `objective`, a `metrics.*` key, or `durationMs` (runtime). */
  key: string
  /** Whether higher (`max`) or lower (`min`) is better for THIS criterion. */
  direction: 'max' | 'min'
  /** Display label; defaults to `key`. */
  label?: string
}

/**
 * A robust interval estimate of one criterion over a set of runs — the rliable-recommended way to
 * summarise seed variance (point estimates over few seeds are unreliable). `iqm` (interquartile mean)
 * is the headline aggregate; `ci` is its bootstrap confidence interval.
 */
export interface RunValueAggregate {
  /** Number of runs (seeds) in the sample. */
  n: number
  mean: number
  /** Interquartile mean — robust + efficient aggregate (trims the top/bottom 25%, then means). */
  iqm: number
  median: number
  std: number
  min: number
  max: number
  /** Bootstrap confidence interval of the IQM, `[lo, hi]`. */
  ci: [number, number]
}

/** One value of a lever in an {@link OfatAnalysis}: the matched runs (identical on every other lever) + their aggregate. */
export interface OfatLevel {
  /** The lever's value at this level (stringified for display/keying). */
  value: string
  /** Run keys contributing — all identical on every OTHER lever + dataset/env (a clean one-factor contrast). */
  runKeys: string[]
  /** Distinct seeds among the contributing runs. */
  seeds: number
  aggregate: RunValueAggregate
}

/** A pairwise effect between two {@link OfatLevel}s — is the change REAL, not seed noise? */
export interface OfatEffect {
  /** The baseline level's value. */
  from: string
  /** The compared level's value. */
  to: string
  /** `IQM(to) − IQM(from)`, oriented so a positive delta is BETTER per the criterion's direction. */
  delta: number
  /** Bootstrap CI of the DIFFERENCE — the honest significance route (excludes 0 ⇒ significant), never CI-overlap. */
  diffCi: [number, number]
  /** True when `diffCi` excludes 0 AND survives Benjamini-Hochberg FDR across the analysis's comparisons. */
  significant: boolean
  /** Two-sided bootstrap p-value, BEFORE the FDR correction. */
  pValue: number
}

/**
 * One lever's effect on one {@link AnalysisCriterion}, controlling for everything else — the answer to
 * "how does changing THIS lever affect the score, holding all other levers + the dataset fixed?".
 * Built from exact one-factor-at-a-time (OFAT) matches over stored runs, with seed-variance rigor.
 */
export interface OfatAnalysis {
  lever: string
  criterion: AnalysisCriterion
  /** A stable signature of the held-fixed context (the other levers + dataset/env) this contrast is within. */
  controlSignature: string
  /** The lever's observed values and their aggregates, ordered best-first by the criterion. */
  levels: OfatLevel[]
  /** Pairwise effects (each non-baseline level vs the baseline), with significance. */
  effects: OfatEffect[]
}

/** A lever's cheap, surrogate-free importance under a criterion — the spread of its per-value marginal IQMs. */
export interface LeverImportance {
  lever: string
  /** Fraction of the explained between-lever variance attributable to this lever's marginal means, in [0,1]. */
  importance: number
  /** Number of distinct values observed for the lever. */
  values: number
  /** Best and worst marginal IQM across the lever's values (oriented to the criterion). */
  bestValue: string
  worstValue: string
  /** Fewest runs observed for any single value of this lever — the weakest leg of the estimate. */
  minRuns: number
  /**
   * Whether every value has enough runs to trust the importance ({@link minRuns} ≥ the min-seeds bar).
   * When false the number rests on too few trials and the UI should flag it + suggest more runs.
   */
  confident: boolean
}

/**
 * A fitted config→criterion surrogate (a small seeded random forest of regression trees) — the
 * retraining-free model the xAI tree, fANOVA importance, and interaction views predict against. Opaque:
 * build it with `fitConfigSurrogate` and read it with `predictConfig` / the analysis functions.
 */
export interface ConfigSurrogate {
  /** The forest — each tree is a nested split/leaf node (shape is an implementation detail). */
  trees: unknown[]
  /** The levers the surrogate splits on, with how each is treated. */
  levers: { name: string; kind: 'num' | 'cat' }[]
  /** Mean target — the fallback prediction for an empty forest. */
  mean: number
}

/** One greedy single-lever change in an {@link AblationPath}, with the surrogate-predicted effect. */
export interface AblationStep {
  lever: string
  /** The baseline value before this change. */
  from: string
  /** The incumbent value applied at this step. */
  to: string
  /** Surrogate-predicted criterion after applying this change. */
  predicted: number
  /** Improvement vs the previous step, oriented so positive = better. */
  gain: number
}

/**
 * A greedy ablation path from a baseline config to the incumbent (best) — at each step the single lever
 * change that most improves the surrogate prediction, so "what handful of changes drove this result?"
 * reads off as an ordered tree. The validated local hyperparameter-importance method (Fawcett & Hoos).
 */
export interface AblationPath {
  baseline: Record<string, unknown>
  incumbent: Record<string, unknown>
  baselinePredicted: number
  incumbentPredicted: number
  steps: AblationStep[]
}

/** A lever's fANOVA MAIN-effect importance from the surrogate — the variance its marginal explains, in [0,1]. */
export interface FanovaImportance {
  lever: string
  /** MAIN-effect importance (Sobol first-order): variance of the lever's marginal / total. */
  importance: number
  /**
   * TOTAL-effect importance (Sobol total-order): the variance contributed by varying this lever at each
   * fixed combination of the others, averaged / total — so it includes the lever's INTERACTIONS. `total
   * − importance` is how much of the lever's effect is interactive; a tiny `total` means the lever is
   * inert across the explored range (a "stop sweeping it" signal).
   */
  total: number
  /** Number of distinct values marginalised over. */
  values: number
  /** The distinct observed values themselves, so the viewer can link each to its runs. */
  valueList: unknown[]
}

/** Interaction (coupling) strength between a lever pair — the 2-way ANOVA term as a fraction of total variance. */
export interface LeverCoupling {
  leverA: string
  leverB: string
  /** Variance explained by the pair's INTERACTION (joint minus the two main effects), normalised by total. */
  strength: number
}

/**
 * The surrogate-predicted criterion across two levers' grid — answers "does A help universally or only at
 * some B?" (the interaction view). `cells[i*valuesB.length + j]` is `(valuesA[i], valuesB[j])`.
 */
export interface InteractionGrid {
  leverA: string
  leverB: string
  valuesA: string[]
  valuesB: string[]
  /** Surrogate-predicted criterion per (a, b) cell, row-major over valuesA × valuesB. */
  cells: number[]
}

/** One setup projected onto the 2-D PCA plane, coloured by its criterion value. */
export interface PcaPoint {
  x: number
  y: number
  /** The setup's criterion value (IQM across its seeds) — the colour. */
  value: number
  /** A representative run key (for tooltips / drill-down). */
  key: string
  /** All run keys in this setup. */
  runKeys: string[]
}

/**
 * A deterministic PCA projection of the explored configs onto 2 axes, coloured by performance — a
 * VISUALISATION/intuition layer (spot clusters + outliers), NOT a space you navigate (PC axes aren't
 * levers). One point per SETUP (config minus seed). Numeric levers are standardised; categorical levers
 * one-hot encoded; the top-2 principal components are found by deterministic power iteration.
 */
export interface PcaProjection {
  points: PcaPoint[]
  /** Fraction of total variance each of PC1/PC2 explains (each in [0,1]). */
  explainedVariance: [number, number]
  /** Number of encoded feature columns the projection ran over. */
  features: number
}

/**
 * The whole-space xAI analysis precomputed in ONE pass over every run (seeds folded into setups), so the
 * heavy surrogate/fANOVA/coupling/PCA work happens server-side and the viewer just renders + recomputes
 * cheap interaction grids off the embedded surrogate. Deterministic; cached as a `{recordType}-config-space`
 * record and refreshed as runs land.
 */
export interface ConfigSpaceAnalysis {
  criterion: { key: string; direction: 'max' | 'min' }
  /** Runs behind the analysis (every seed of every config). */
  runCount: number
  /** Distinct configurations after folding seeds into one IQM each — what the surrogate trained on. */
  setupCount: number
  /** Every lever (model + dataset + environment) the space varies over. */
  levers: string[]
  /** The distinct configs (seeds folded, each carrying its IQM where the criterion reads it) — so the
   * viewer can marginalise the surrogate live for any lever-pair interaction grid + label PCA points. */
  setups: AnalysisRun[]
  /** Surrogate-free screening: each lever's importance from the spread of its per-value marginal IQMs. */
  screening: LeverImportance[]
  /** One-factor-at-a-time contrasts per lever (each lever → its controlled contrasts), so the viewer's
   * Config-effects card renders from the cache instead of recomputing bootstraps in the browser. */
  ofat: Record<string, OfatAnalysis[]>
  /** The seeded forest the reads derive from; embedded so the viewer can compute interaction grids live. */
  surrogate: ConfigSurrogate
  importances: FanovaImportance[]
  /** Pairwise interaction strengths, computed ONLY among {@link coupledLevers}. */
  couplings: LeverCoupling[]
  /** The high-effect levers coupling was searched among (top-K by total effect); the rest are too inert. */
  coupledLevers: string[]
  ablation: AblationPath | null
  pca: PcaProjection | null
  recommendations: ExperimentRecommendation[]
  /**
   * The ENVIRONMENT this bundle is scoped to — the held-fixed values of the environment/dataset levers
   * (market mechanics + which data). The surrogate/importances/recommender above are computed ONLY within
   * it, over the MODEL levers, so configs aren't blended across environments. Null when the project
   * declares no environment/dataset levers (whole space analysed together).
   */
  environment: Record<string, unknown> | null
  /** Every distinct environment present in the runs — drives the selector + cross-environment comparison. */
  environments: EnvironmentSummary[]
  /** Importance of each ENVIRONMENT/DATASET lever across ALL runs — "how much does this context matter?"
   * for the compare view. These are context, never recommended; the viewer tags them 🔒. */
  contextImportances: LeverImportance[]
  /** The environment + dataset levers (context); the rest are model levers. */
  contextLevers: string[]
  /** Best-so-far over the environment's runs in time order — for the "is the search still improving?" view. */
  convergence: ConvergencePoint[]
}

export interface ConvergencePoint {
  /** 1-based run count (x-axis). */
  index: number
  /** Best criterion value among the first `index` runs (oriented by the criterion's direction). */
  best: number
  /** When that run completed (ISO), when known. */
  at?: string
}

/** One distinct environment (a combination of environment+dataset lever values) across the runs. */
export interface EnvironmentSummary {
  /** Canonical signature of the context-lever values — the environment's stable identity. */
  signature: string
  /** The held-fixed context-lever values that define this environment. */
  values: Record<string, unknown>
  /** Completed runs in this environment. */
  runCount: number
  /** Best (criterion-oriented) setup IQM achieved in this environment — for ranking environments. */
  best: number
}

export interface AnalyzeConfigSpaceParams {
  scope: string
  projectRoot: string
  manifest?: TrainerManifest
  manifestRelPath?: string
  /** Criterion to analyse over the whole space; defaults to the manifest objective. */
  criterionKey?: string
  criterionDir?: 'max' | 'min'
  /** The environment to scope the analysis to (its context-lever values). Omit ⇒ the most-run environment. */
  environment?: Record<string, unknown>
}

export interface AnalyzeConfigSpaceResult {
  recordType: string
  criterion: AnalysisCriterion
  /** Null when there are no completed runs to analyse. */
  analysis: ConfigSpaceAnalysis | null
}

/**
 * A deterministic, NON-LLM recommendation for the next experiments to run — each carries a launchable
 * {@link ExperimentSpec} the viewer fires as a batched campaign, closing the analyse→run→re-analyse loop.
 */
export interface ExperimentRecommendation {
  /**
   * Why this batch: `acquisition` (the surrogate's highest Expected-Improvement unrun config — climb
   * toward the optimum), `thin-seeds` (a variance-thin setup needing more seeds), `missing-cell` (an
   * untested factorial combo), or `interaction` (an untested lever pair).
   */
  kind: 'acquisition' | 'missing-cell' | 'thin-seeds' | 'interaction'
  /** One-line human reason, e.g. "batch_size=256 never run with the current best setup". */
  reason: string
  /** Number of runs this batch would launch. */
  runCount: number
  /** The launchable spec (`sweep`/`fixed`/`seeds`) — fed straight to `runTrainingCampaign`. */
  spec: ExperimentSpec
  /** Deterministic priority (higher = more valuable to run next). */
  priority: number
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
  /**
   * Free-form run outputs. Known keys the hub reads: `checkpoint` (path), `runChart`, and
   * `decisionTrace` (a {@link DecisionTrace} for the Explain view) plus `decisionTraceFile` (path to
   * the optional full per-step sidecar). Unknown keys are stored and ignored.
   */
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
  /** Injectable host CPU count for deterministic tests; defaults to os.availableParallelism(). */
  availableParallelism?: () => number
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

/** Whether a hypothesis is proven/disproved by its matching runs — `untested` until evidence exists. */
export type HypothesisStatus = 'untested' | 'proven' | 'disproved'

/** The aggregate read of a hypothesis's matching runs — the numbers behind a verdict. */
export interface MeasuredSummary {
  /** Count of non-failed matching runs. */
  runs: number
  /** Best objective among the matching runs (per the manifest objective direction). */
  objective: number
  /** Whether the best matching run beats buy-and-hold OOS; `null` when no run carries that metric. */
  beatsHold: boolean | null
}

/** A snapshot of the last auto-evaluation — diffed on the next refresh to detect which runs are new. */
export interface HypothesisEvidence {
  /** When this snapshot was taken (ISO). */
  at: string
  /** The auto-derived status at this snapshot. */
  status: HypothesisStatus
  /** Sorted keys of the runs that matched the spec at this snapshot. */
  matchedKeys: string[]
  /** The measured read at this snapshot (`null` when no runs matched). */
  measured: MeasuredSummary | null
}

/** One recorded change of a hypothesis's auto-verdict — names the runs that flipped it and the read. */
export interface HypothesisTransition {
  /** When the flip was observed (ISO). */
  at: string
  /** The status before the flip. */
  from: HypothesisStatus
  /** The status after the flip. */
  to: HypothesisStatus
  /** Keys of the runs new since the prior snapshot — the evidence that caused the flip. */
  byRunKeys: string[]
  /** The measured read at the flip (`null` when no runs matched). */
  measured: MeasuredSummary | null
}

/** What a context-spanning hypothesis CLAIMS about how the objective moves across its context cells. */
export type HypothesisComparisonKind = 'beats-baseline' | 'invariant' | 'differs'

/**
 * How to read a context-spanning hypothesis (a `spec` with `environments`/`datasets` bundles) into a
 * verdict. Its runs are grouped per context cell and compared ACROSS cells — never pooled, since runs in
 * different environments/datasets are a different comparison (and the context levers are held-fixed
 * context, never tuned). Absent ⇒ the default `beats-baseline`.
 */
export interface HypothesisComparison {
  /**
   * `beats-baseline` — the best non-baseline cell's objective beats the baseline cell's (e.g. long+short
   * beats long-only). `invariant` — the objective is stable across cells (a robustness thesis).
   * `differs` — the objective moves across cells (a sensitivity thesis).
   */
  kind: HypothesisComparisonKind
  /** Index (into the spec's ordered context cells) of the baseline cell for `beats-baseline`; default 0. */
  baselineIndex?: number
  /** Fractional objective-spread tolerance for `invariant`/`differs` (relative to the baseline); default 0.1. */
  tolerance?: number
}

/** One context cell's measured read — produced per environment/dataset cell a hypothesis spans, never pooled. */
export interface ContextGroupMeasured {
  /** The context-lever values defining this cell (e.g. `{ allow_shorting: true }`). */
  context: Record<string, unknown>
  /** Sorted keys of the runs that ran in this cell. */
  runKeys: string[]
  /** The aggregate read over this cell's runs only (`null` when the cell has no non-failed run). */
  measured: MeasuredSummary | null
}

/**
 * A registry entry for a CLAIM that runs prove or disprove — an architecture, a paper's method, or an
 * ad-hoc idea. Its `spec` both LAUNCHES the runs that test it AND identifies them: a run is evidence iff
 * its config is consistent with `spec` (every `fixed` lever matches; every swept lever's value is one of
 * the options). The verdict is AUTO-derived from those runs (beats buy-and-hold OOS) and re-checked when
 * runs land, with manual override. Domain-oblivious; stored as a `<recordType>-hypothesis` record
 * (key = `hashTrainingConfig(spec)`, so identical specs dedupe across human/llm/paper/migrated sources).
 */
export interface TrainingHypothesis {
  /** Stable hash of the spec — the canonical identity; identical specs dedupe. */
  id: string
  title: string
  rationale: string
  spec: ExperimentSpec
  /** The verdict: auto-derived from matching runs, or pinned when `verdictSource` is `manual`. */
  status: HypothesisStatus
  /**
   * For a context-spanning `spec` (`environments`/`datasets` bundles), how the cross-context comparison is
   * read into the verdict. Ignored for a single-context spec (which uses the pooled beats-hold rule).
   */
  comparison?: HypothesisComparison
  /** Whether `status` is auto-derived from runs or a manual override that refresh must not overwrite. */
  verdictSource: 'auto' | 'manual'
  /** Free-text note recorded with a manual verdict. */
  verdictNote?: string
  /** Hidden from the default view (a rejected/irrelevant proposal) without deleting it. */
  dismissed?: boolean
  /** Where the entry came from. */
  source: 'human' | 'llm' | 'paper' | 'migrated-model'
  /** Provenance label of the proposing model (absent for human entries). */
  proposedBy?: string
  /** Metrics a source/author claims (free-form), shown alongside the measured read. */
  claimedMetrics?: Record<string, number>
  /** The last auto-evaluation snapshot — the baseline the next refresh diffs against. */
  evidence?: HypothesisEvidence
  /** History of auto-verdict flips, newest last — each names the runs that caused it. */
  transitions?: HypothesisTransition[]
  /** Ids of Papers that link this hypothesis (reverse of `TrainingPaperRecord.hypothesisIds`). */
  paperIds?: string[]
  /** The last launched campaign for live "running" status (verdict derives from ALL matching runs, not these). */
  campaign?: TrainingHypothesisCampaign
  createdAt: string
  updatedAt: string
}

/** The runtime block the viewer stamps when a hypothesis launches a campaign — for the live badge only. */
export interface TrainingHypothesisCampaign {
  activityId: string
  launchedAt: string
  status: 'queued' | 'running' | 'completed' | 'aborted'
  keys?: string[]
  bestObjective?: number
  bestKey?: string
  completed?: number
  failed?: number
  finishedAt?: string
  queueId?: string
}

/**
 * A registry entry for an APPROACH/paper — "an approach with a source and a claim". A CONTAINER: it
 * links to N {@link TrainingHypothesis} (created by LLM extraction, manual add, or linking an existing
 * one) and its verdict ROLLS UP from theirs. Domain-oblivious; the trading line's first consumers are
 * the published methods it replicates under real costs. Stored as a `<recordType>-paper` record (key = id).
 */
export interface TrainingPaperRecord {
  /** Stable id (random hex). */
  id: string
  title: string
  /** Link to the paper / source (arXiv, blog, repo). */
  url?: string
  authors?: string
  year?: number
  /** The headline claim in plain words, e.g. "RL beats buy-and-hold by 30% out-of-sample". */
  claim: string
  /** Metrics the SOURCE claims (free-form, e.g. `{ return_pct: 30, sharpe: 1.5 }`). */
  claimedMetrics?: Record<string, number>
  /** Honesty checklist — assumptions that commonly inflate published results. */
  assumptions?: {
    /** Are realistic transaction fees modelled? */
    fees?: boolean
    /** Returns reported NET of costs (vs gross)? */
    netOfCosts?: boolean
    /** How often the model is retrained (free text, e.g. "monthly"). */
    retrainCadence?: string
    /** Assumes frictionless execution (no slippage/fees)? */
    frictionless?: boolean
    /** Result depends on a multi-asset universe? */
    multiAsset?: boolean
    /** Any other caveats. */
    notes?: string
  }
  /** How the approach works (prose). */
  approach?: string
  /** Ids of the hypotheses this paper creates/links; the paper's verdict ROLLS UP from theirs. */
  hypothesisIds?: string[]
  /** Slug ids of the catalog Models this paper introduces or improves (set by `analyzePaperModels`). */
  modelIds?: string[]
  /** Lifecycle: untested → replicating → holds-up | fluff (legacy/manual; the card badge rolls up from `hypothesisIds`). */
  status: 'untested' | 'replicating' | 'holds-up' | 'fluff'
  /** Free-text verdict / notes recorded by the user. */
  verdictNote?: string
  /** Hidden from the default Papers view (marked "not wanted") without deleting it. */
  dismissed?: boolean
  /** Where the entry came from. */
  source: 'manual' | 'research'
  tags?: string[]
  createdAt: string
  updatedAt: string
}

/**
 * A starter Paper a manifest can ship (parallel to `presets`/`quickStart`): curated approaches the
 * viewer imports into the registry once, keyed by `id` so re-import never duplicates or clobbers an
 * entry the user has since edited. Timestamps + status default on import.
 */
export type TrainingPaperSeed = Omit<TrainingPaperRecord, 'createdAt' | 'updatedAt' | 'status'> & {
  status?: TrainingPaperRecord['status']
}

/** What KIND of model a catalog entry is — drives grouping + the heuristic category guess in a scan. */
export type ModelCategory = 'rl' | 'supervised' | 'baseline' | 'component'

/**
 * Lifecycle of a catalog Model. `proposed` = named (often by a paper) but not yet implemented;
 * `implemented` = present in the project and training healthily; `failing` = implemented but its runs
 * are health-flagged/degenerate; `needs-improvement` = works but a paper or result asks for more;
 * `deferred` = a real candidate, but deliberately NOT being built now (e.g. blocked on missing infra);
 * `deprecated` = retired. `proposed`/`implemented`/`failing` auto-derive from runs (see `deriveModelStatus`
 * in `viewer/models.js`); `needs-improvement`/`deferred`/`deprecated` are pins (manual OR manifest-seeded)
 * that auto-derivation won't override.
 */
export type ModelStatus =
  | 'proposed'
  | 'implemented'
  | 'failing'
  | 'needs-improvement'
  | 'deferred'
  | 'deprecated'

/** Where a catalog Model entry came from. */
export type ModelSource = 'scan' | 'paper' | 'manual' | 'llm'

/**
 * One concrete VARIANT of a model — how it actually appears in run configs. A run trains this flavor iff
 * its `config.model_name` equals `modelName` AND every key in `config` (the extra matchers, e.g.
 * `{ lstm_hidden_size: 3 }`) loosely-equals the run's value. Recording flavors makes a family (e.g.
 * "Dueling DQN") map PRECISELY to the variants in code, lets a missing flavor be spotted + added, and
 * lets papers/hypotheses attach to the exact variant they concern.
 */
export interface ModelFlavor {
  /** Short label distinguishing this variant, e.g. "custom + LSTM (h3)". */
  name?: string
  /** The `config.model_name` value this flavor trains as. */
  modelName: string
  /** Extra config matchers that pin this flavor WITHIN a shared `model_name` (omit when the name alone identifies it). */
  config?: Record<string, unknown>
  /** Where this flavor is implemented — a repo-relative path hint. */
  implPath?: string
  /** Papers specific to THIS flavor (the model-level `paperIds` cover the whole family). */
  paperIds?: string[]
  /** Hypotheses specific to THIS flavor. */
  hypothesisIds?: string[]
  notes?: string
}

/**
 * A catalog entry for a MODEL ARCHITECTURE/algorithm the project can train — the aggregating layer the
 * Models tab renders. It OWNS its runs through its `flavors`: a run trains this model iff it matches one
 * of the flavors (by `model_name`, optionally narrowed by config), so run counts, verdict and "is it
 * failing" derive from an all-runs aggregate. It LINKS the papers that introduce/improve it (`paperIds`)
 * and the hypotheses that test it (`hypothesisIds`). Domain-oblivious; stored as a `<recordType>-model`
 * record (key = `id`, which is the `slug`). Distinct from a hypothesis (a falsifiable claim): a Model is
 * the thing a claim, paper or run is ABOUT.
 */
/**
 * The result of benchmarking a model on the available compute devices (CPU vs Apple MPS) — what backs a
 * model's {@link TrainingModel.preferredDevice}. `usPerStep` is per device; `speedup` is the winner's
 * margin over the runner-up (>= 1).
 */
export interface ModelDeviceBenchmark {
  bestDevice: 'cpu' | 'mps'
  speedup: number
  usPerStep: Record<string, number>
  availableDevices: string[]
  benchmarkedAt: string
}

export interface TrainingModel {
  /** Stable id — equals `slug`; identical slugs dedupe across scan/paper/manual/seed sources. */
  id: string
  /** Human name, e.g. "Rainbow DQN (custom)". */
  name: string
  /** Canonical kebab identity, e.g. `rainbow-dqn-custom`. */
  slug: string
  /** One or two sentences: what the model is + how it differs. */
  description: string
  category: ModelCategory
  /** Lifecycle verdict — auto-derived from runs unless `statusSource` is `manual`. */
  status: ModelStatus
  /** Whether `status` is auto-derived from runs or a manual pin that the run roll-up must not overwrite. */
  statusSource: 'auto' | 'manual'
  /** Free-text note recorded with a manual status, or why a scan flagged it. */
  statusNote?: string
  /**
   * The concrete VARIANTS that bind runs to this model. A run is this model's evidence iff it matches one
   * flavor. Empty for a purely-proposed model not yet wired to any run config. (Records written before
   * flavors carried a flat `modelNames: string[]`; the viewer reads either.)
   */
  flavors: ModelFlavor[]
  /** Ids of Papers that introduce or improve this model. */
  paperIds?: string[]
  /** Ids (spec hashes) of Hypotheses that test this model. */
  hypothesisIds?: string[]
  /** Where the model is (or, when `proposed`, should be) implemented — a repo-relative path hint. */
  implPath?: string
  /** When `proposed`: what a paper/scan asks to add or change, in plain words (the agent's brief). */
  proposal?: string
  source: ModelSource
  /** Provenance label of the proposing model (absent for human/seed entries). */
  proposedBy?: string
  /** Free-text notes. */
  notes?: string
  /** Hidden from the default view (a rejected/irrelevant candidate) without deleting it. */
  dismissed?: boolean
  /** The device the last benchmark found fastest for this model (the cpu-vs-mps winner). */
  preferredDevice?: 'cpu' | 'mps'
  /** The last device-benchmark measurement backing `preferredDevice`. */
  deviceBenchmark?: ModelDeviceBenchmark
  createdAt: string
  updatedAt: string
}

/**
 * A starter Model a manifest can ship (parallel to `papers`/`hypotheses`): curated catalog entries the
 * viewer imports once, keyed by `id`/`slug` so re-import never duplicates or clobbers a user's edit.
 * `statusSource` + timestamps default on import.
 */
export type TrainingModelSeed = Omit<TrainingModel, 'createdAt' | 'updatedAt' | 'statusSource'> & {
  statusSource?: TrainingModel['statusSource']
}

/**
 * A model a paper PROPOSES that has no catalog entry yet — the payload the Models tab turns into a
 * one-click "Add to catalog" affordance (persisting a `proposed` {@link TrainingModel}). Returned by
 * {@link ModelTrainerTools.analyzePaperModels}; never auto-persisted.
 */
export interface ProposedModel {
  name: string
  slug: string
  description: string
  category: ModelCategory
  /** What the paper asks to add/change — becomes the proposed model's `proposal`. */
  proposal: string
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

/** An LLM-proposed experiment surfaced as a runnable suggestion in the xAI recommender (NOT a hypothesis). */
export interface TrainingExperimentSuggestion {
  /** Stable hash of the spec — the canonical identity; identical specs dedupe. */
  id: string
  title: string
  rationale: string
  spec: ExperimentSpec
  source: 'llm'
  /** Provenance label of the proposing model. */
  proposedBy: string
  proposedAt: string
}

export interface ProposeTrainingExperimentsParams {
  scope: string
  projectRoot: string
  manifest?: TrainerManifest
  /** Manifest file relative to `projectRoot` (default `.factory/trainer.json`); names a second conformant line in the same repo. */
  manifestRelPath?: string
  llmConfig: LLMConfig
  /** How many suggestions to ask for; defaults to {@link DEFAULT_HYPOTHESIS_COUNT}. */
  count?: number
  /** Extra guidance appended to the proposer prompt. */
  instructions?: string
  abortSignal?: AbortSignal
  /** Fired after each suggestion record upsert so the host can broadcast `data:updated`. */
  onRecordWritten?: (type: string, key: string) => void
}

export interface ProposeTrainingExperimentsResult {
  recordType: string
  proposed: number
  /** Proposals whose spec already exists as a suggestion record. */
  skippedExisting: number
  suggestions: TrainingExperimentSuggestion[]
  proposedBy: string
  proposedAt: string
}

export interface AnalyzePaperFromUrlParams {
  scope: string
  projectRoot: string
  manifest?: TrainerManifest
  /** Manifest file relative to `projectRoot` (default `.factory/trainer.json`). */
  manifestRelPath?: string
  /** The paper / source URL to read and summarise. */
  url: string
  /** Optional extra steering for the summary (e.g. "focus on the cost assumptions"). */
  notes?: string
  llmConfig: LLMConfig
  abortSignal?: AbortSignal
  /** Fired after the draft paper record is upserted so the host can broadcast `data:updated`. */
  onRecordWritten?: (type: string, key: string) => void
  /**
   * Injectable URL→text fetcher (for tests); defaults to a real HTTP fetch + HTML/abstract extraction.
   * The TOOL supplies the page text to the model — the model needs no web tools.
   */
  fetchPaperText?: (url: string, abortSignal?: AbortSignal) => Promise<string>
}

export interface AnalyzePaperFromUrlResult {
  recordType: string
  /** The drafted, persisted Paper record (status 'untested', source 'research') for the user to verify. */
  paper: TrainingPaperRecord
  /** The hypotheses extracted from the paper and persisted (new + pre-existing it linked). */
  hypotheses: TrainingHypothesis[]
  /** Ids of every hypothesis now linked to the paper (`paper.hypothesisIds`). */
  linkedHypothesisIds: string[]
  /** Provenance label of the summarising model. */
  analyzedBy: string
  analyzedAt: string
}

export interface SuggestPaperHypothesesParams {
  scope: string
  projectRoot: string
  manifest?: TrainerManifest
  /** Manifest file relative to `projectRoot` (default `.factory/trainer.json`). */
  manifestRelPath?: string
  /** The paper to enrich — read from `{recordType}-paper`. */
  paperId: string
  llmConfig: LLMConfig
  abortSignal?: AbortSignal
  /** Fired after each paper/hypothesis record upsert so the host can broadcast `data:updated`. */
  onRecordWritten?: (type: string, key: string) => void
  /**
   * Injectable URL→text fetcher (for tests); defaults to a real HTTP fetch. When the paper carries a
   * URL the tool tries to fetch its text as extra context; a fetch failure is non-fatal (the paper's
   * stored fields + the existing hypotheses are enough).
   */
  fetchPaperText?: (url: string, abortSignal?: AbortSignal) => Promise<string>
}

export interface SuggestPaperHypothesesResult {
  recordType: string
  /** The paper with its updated `hypothesisIds`. */
  paper: TrainingPaperRecord
  /** Ids of PRE-EXISTING hypotheses the model matched to this paper and that were linked. */
  linkedExistingIds: string[]
  /** The NEW hypotheses created from the model's suggestions and linked. */
  newHypotheses: TrainingHypothesis[]
  /** Every hypothesis id now linked to the paper (matched existing ∪ new). */
  linkedHypothesisIds: string[]
  /** Provenance label of the suggesting model. */
  suggestedBy: string
  suggestedAt: string
}

export interface ScanProjectModelsParams {
  scope: string
  projectRoot: string
  manifest?: TrainerManifest
  /** Manifest file relative to `projectRoot` (default `.factory/trainer.json`). */
  manifestRelPath?: string
  /** When present, an LLM enriches each discovered candidate (category/description/paper links); when
   * absent the scan is heuristic-only (the candidate's guessed name/category and no paper links). */
  llmConfig?: LLMConfig
  abortSignal?: AbortSignal
  /** Fired after each model record upsert so the host can broadcast `data:updated`. */
  onRecordWritten?: (type: string, key: string) => void
}

export interface BenchmarkModelDeviceParams {
  scope: string
  projectRoot: string
  manifest?: TrainerManifest
  /** Manifest file relative to `projectRoot` (default `.factory/trainer.json`). */
  manifestRelPath?: string
  /** The model record key (slug) to benchmark + update with its preferred device. */
  modelId: string
  /** Override the benchmarked `model_name` (defaults to the model's first flavor's modelName, else its slug). */
  modelName?: string
  /** Named compute target to benchmark on; omit for the default (local) runner. */
  computeTarget?: string
  abortSignal?: AbortSignal
  /** Fired after the model record upsert so the host can broadcast `data:updated`. */
  onRecordWritten?: (type: string, key: string) => void
}

export interface BenchmarkModelDeviceResult {
  recordType: string
  modelId: string
  preferredDevice: 'cpu' | 'mps'
  deviceBenchmark: ModelDeviceBenchmark
}

export interface ScanProjectModelsResult {
  recordType: string
  /** How many candidate models the heuristic found that were not already in the catalog. */
  discovered: number
  /** How many new `<recordType>-model` records were persisted. */
  created: number
  /** Candidates skipped because a model with the same slug already exists. */
  skippedExisting: number
  /** The newly-created models. */
  models: TrainingModel[]
  /** Provenance label of the enriching model (absent for a heuristic-only scan). */
  scannedBy?: string
  scannedAt: string
}

export interface AnalyzePaperModelsParams {
  scope: string
  projectRoot: string
  manifest?: TrainerManifest
  manifestRelPath?: string
  paperId: string
  llmConfig: LLMConfig
  /** Optionally re-fetch the paper URL for extra context (injected in tests). */
  fetchPaperText?: (url: string, abortSignal?: AbortSignal) => Promise<string>
  abortSignal?: AbortSignal
  onRecordWritten?: (type: string, key: string) => void
}

export interface AnalyzePaperModelsResult {
  recordType: string
  /** The paper with its updated `modelIds`. */
  paper: TrainingPaperRecord
  /** Slug ids of EXISTING catalog models the paper was matched + linked to. */
  linkedModelIds: string[]
  /** Models the paper proposes that have NO catalog entry yet — the "Add to catalog" candidates. */
  missingModels: ProposedModel[]
  /** Provenance label of the analysing model. */
  analyzedBy: string
  analyzedAt: string
}

export interface XaiNarrateParams {
  scope: string
  projectRoot: string
  manifest?: TrainerManifest
  /** Manifest file relative to `projectRoot` (default `.factory/trainer.json`). */
  manifestRelPath?: string
  llmConfig: LLMConfig
  /** The run to narrate — the narrative is PER RUN, keyed by this run's key. */
  runKey: string
  /** Optional nearest comparable run for the decision-diff context line. */
  siblingKey?: string
  /** Criterion the run is ranked + the levers screened by; defaults to the manifest objective. */
  criterion?: AnalysisCriterion
  abortSignal?: AbortSignal
  /** Fired after the narrative record upsert so the host can broadcast `data:updated`. */
  onRecordWritten?: (type: string, key: string) => void
}

export interface XaiNarrateResult {
  recordType: string
  /** The narrated run's key (the narrative record's key). */
  runKey: string
  /** Completed-run count at generation — the viewer shows "N new runs since" as the cross-run context drifts. */
  runCount: number
  /** Provenance label of the narrating model. */
  narratedBy: string
  narratedAt: string
}

/**
 * The compact, structured deterministic xAI analysis of ONE run — its decisions, what drives them, how
 * trustworthy that is, how it compares to a sibling, and its standing among all runs. The narrative
 * builder ({@link buildXaiNarrateUserContent}) AND the agent-facing `getRunXAI` tool both consume it, so the
 * facts live in one place and the LLM never computes them.
 */
export interface RunXaiDigest {
  runKey: string
  config: Record<string, unknown>
  objective?: number
  criterion: AnalysisCriterion
  /** Where this run ranks by the criterion among all completed runs. */
  rank?: { position: number; total: number }
  /** Full-rollout action label counts. */
  actionCounts?: Record<string, number>
  attribution?: {
    /** Top input GROUPS by absolute saliency, `[group, value]`. */
    topGroups: [string, number][]
    method?: string
    /** The Adebayo sanity-check verdict — a FAILED check means the attribution is untrustworthy. */
    sanityPassed?: boolean
    sanityRankCorr?: number
  }
  /** Named additive reward contributions ("why this reward"). */
  rewardBreakdown?: Record<string, number>
  /** Linear-probe read of the penultimate-layer representation. */
  latent?: { varianceExplained?: number; probeAccuracy?: number; probeBaseline?: number }
  /** Cross-run lever importance (CONFOUNDED screening) for context. */
  importances: { lever: string; importance: number; bestValue: string }[]
  /** The decision-diff vs the nearest comparable run, when a sibling was given + the traces align. */
  sibling?: {
    key: string
    changed: string
    divergencePct: number
    qualityVerdict?: string
    qualitySummary?: string
  }
}

export interface GetRunDataParams {
  /** The HOST project scope the run records live in. */
  scope: string
  /** The run id (config-hash key). The recordType is resolved from the host's registered training projects. */
  runKey: string
}

export interface GetRunDataResult {
  found: boolean
  /** The training project's record type the run was found under. */
  recordType?: string
  /** The run record, with the heavy per-step trace + series stripped (only a compact trace digest kept). */
  run?: Record<string, unknown>
  error?: string
}

export interface MigrateTrainingRunsParams {
  /** The HOST project scope the run + queue records live in. */
  scope: string
  projectRoot: string
  /** Pre-loaded manifest (carries `recordType` + `migrations`); read from disk when omitted. */
  manifest?: TrainerManifest
  /** Manifest file relative to `projectRoot` (default `.factory/trainer.json`). */
  manifestRelPath?: string
  /**
   * Record type of the host's pending-run queue (e.g. `trainer-queue`). When given, each queued
   * train item's `params.spec.fixed` is migrated too, so pending runs launch under the new shape.
   * Omit to migrate only completed run records.
   */
  queueRecordType?: string
  /** Fired after each record rewrite OR deletion so the host can broadcast `data:updated`. */
  onRecordWritten?: (type: string, key: string) => void
}

export interface MigrateTrainingRunsResult {
  recordType: string
  /** Completed run records examined / rewritten / deleted (a delete rule removes the record). */
  examinedRuns: number
  migratedRuns: number
  deletedRuns: number
  /** Pending-queue records examined / rewritten / deleted (0 when no `queueRecordType` was given). */
  examinedQueue: number
  migratedQueue: number
  deletedQueue: number
}

export interface InvalidateRunsParams {
  /** The HOST project scope the run + queue records live in. */
  scope: string
  projectRoot: string
  /** Pre-loaded manifest (carries `recordType`); read from disk when omitted. */
  manifest?: TrainerManifest
  manifestRelPath?: string
  /** The invalidation id (from the manifest rule) — also the one-time pending-cancel marker key. */
  invalidationId: string
  /** Stamped onto each invalidated run as `invalidReason`. */
  reason: string
  /** Only runs whose pipeline major is BELOW this are stale (re-runs at/after it are never flagged). */
  beforePipelineMajor: number
  /** True for a stored run CONFIG produced by the bug. */
  affectsRun: (config: Record<string, unknown>) => boolean
  /** True for a pending launch SPEC (`{fixed, sweep}`) that would produce an affected run. */
  affectsPending?: (spec: Record<string, unknown>) => boolean
  /** Record type of the host's pending-run queue (e.g. `trainer-queue`). */
  queueRecordType?: string
  /** Remove affected pending-queue items ONCE (guarded by a marker keyed on `invalidationId`). */
  cancelPendingQueue?: boolean
  /** Fired after each record rewrite OR deletion so the host can broadcast `data:updated`. */
  onRecordWritten?: (type: string, key: string) => void
}

export interface InvalidateRunsResult {
  recordType: string
  examinedRuns: number
  invalidatedRuns: number
  examinedQueue: number
  cancelledQueue: number
  /** True when the one-time pending cancellation was already applied (marker present) and so skipped. */
  pendingAlreadyApplied: boolean
}

export interface GetRunXaiParams {
  scope: string
  runKey: string
  /** Optional nearest comparable run for the decision-diff context. */
  siblingKey?: string
  /** Criterion to rank + screen by; defaults to the run's training project objective. */
  criterion?: AnalysisCriterion
}

export interface GetRunXaiResult {
  found: boolean
  recordType?: string
  /** Completed-run count the analysis was computed over. */
  runCount?: number
  analysis?: RunXaiDigest
  error?: string
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
  /**
   * Ask an LLM for the next experiments given run history + verdicts; persist new
   * `{recordType}-xai-suggestion` records (deduped by spec hash) for the xAI recommender to surface as
   * runnable suggestions. Unlike {@link proposeTrainingHypotheses} this creates NO hypothesis records.
   */
  proposeTrainingExperiments(
    params: ProposeTrainingExperimentsParams,
  ): Promise<ProposeTrainingExperimentsResult>
  /**
   * Read a paper/source URL, summarise it with an LLM (the tool fetches the page text — no web tools),
   * and persist a DRAFT `{recordType}-paper` record (status 'untested', source 'research') for the user
   * to verify. Powers the Papers tab's "Automatic Fill".
   */
  analyzePaperFromUrl(params: AnalyzePaperFromUrlParams): Promise<AnalyzePaperFromUrlResult>
  /**
   * Enrich an EXISTING paper with hypotheses: an LLM matches the paper against the project's existing
   * hypotheses (linking the ones that test its claims) AND proposes any NEW testable hypotheses not yet
   * covered (created + linked). Powers the Papers tab's "Suggest hypotheses". Works for any paper (URL
   * optional — its text is used as extra context when present).
   */
  suggestPaperHypotheses(
    params: SuggestPaperHypothesesParams,
  ): Promise<SuggestPaperHypothesesResult>
  /**
   * Discover MODELS the project declares (its `model_name` lever choices) that the catalog does not yet
   * cover, optionally enrich each with an LLM (category, description, paper links), and persist them as
   * `<recordType>-model` records. Heuristic-first (works with no LLM); deduped by slug so re-scanning
   * never duplicates. Powers the Models tab's "Scan Project" button.
   */
  scanProjectModels(params: ScanProjectModelsParams): Promise<ScanProjectModelsResult>
  /**
   * Benchmark ONE catalog model on CPU vs MPS (via the manifest's `benchmarkDevice` command) and persist
   * the winner as the model's `preferredDevice` (+ the `deviceBenchmark` numbers). Powers the Models
   * tab's per-model "Benchmark device" button.
   */
  benchmarkModelDevice(params: BenchmarkModelDeviceParams): Promise<BenchmarkModelDeviceResult>
  /**
   * Analyse ONE paper for the MODELS it introduces/improves: an LLM matches it against the existing
   * catalog (linking the models it is about, updating the paper's `modelIds` and each model's
   * `paperIds`) AND names any models the paper proposes that have no catalog entry yet. The missing ones
   * are RETURNED (not persisted) for the Models tab's one-click "Add to catalog". Powers the Papers
   * tab's "Find models" + the per-paper "add missing model" affordance.
   */
  analyzePaperModels(params: AnalyzePaperModelsParams): Promise<AnalyzePaperModelsResult>
  /**
   * Synthesise the campaign's DETERMINISTIC xAI analysis (lever importances, the surrogate ablation
   * path, the recommender's gaps) into a short LLM narrative — "what's been learned + what to try next" —
   * persisted as a `{recordType}-xai-narrative` 'latest' record. The computation stays deterministic;
   * the LLM only narrates the facts.
   */
  xaiNarrate(params: XaiNarrateParams): Promise<XaiNarrateResult>
  /**
   * Agent-facing READ tool: fetch ONE run's stored record by id, resolving its training project from the
   * host's registered projects. The heavy per-step trace + series are stripped (a compact trace digest is
   * kept) so the result is agent-sized. Read-only.
   */
  getRunData(params: GetRunDataParams): Promise<GetRunDataResult>
  /**
   * Agent-facing READ tool: compute the deterministic xAI analysis ({@link RunXaiDigest}) for ONE run by
   * id — the same facts the narrative is built from, returned as structured data (the LLM never computes
   * them). Read-only.
   */
  getRunXAI(params: GetRunXaiParams): Promise<GetRunXaiResult>
  /**
   * Compute the whole-space xAI bundle ({@link ConfigSpaceAnalysis}) over EVERY completed run — surrogate,
   * fANOVA, coupling, PCA, recommendations — folding seeds into setups. Heavy + deterministic, so the host
   * runs it server-side and caches the result for the viewer to render.
   */
  analyzeConfigSpace(params: AnalyzeConfigSpaceParams): Promise<AnalyzeConfigSpaceResult>
  /**
   * Apply the manifest's one-time `migrations` to every stored run config (and, when a
   * `queueRecordType` is given, every pending-queue item's `spec.fixed`), rewriting each in place and
   * recomputing its `setupKey`. Idempotent — rules only match the old config shape, so re-firing is a
   * no-op. Returns how many records were examined vs actually changed.
   */
  migrateTrainingRuns(params: MigrateTrainingRunsParams): Promise<MigrateTrainingRunsResult>
  /**
   * Mark runs produced by a since-fixed bug as `status: 'invalid'` (excluded from all aggregation) and,
   * once, cancel matching pending-queue items. Version-gated so re-runs with the fix are never re-flagged.
   */
  invalidateRuns(params: InvalidateRunsParams): Promise<InvalidateRunsResult>
}
