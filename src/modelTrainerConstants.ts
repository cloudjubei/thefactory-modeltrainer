/** Where a trainer-conformant project ships its manifest, relative to the project root. */
export const TRAINER_MANIFEST_RELPATH = '.factory/trainer.json'

/** Default safety cap on planned items per campaign; override via `ExperimentSpec.maxItems`. */
export const MAX_CAMPAIGN_ITEMS = 500

/** Provenance label stamped on run records when the caller names no compute target. */
export const DEFAULT_RAN_BY = 'local'

/**
 * Content paths on a run record that hold the unbounded per-step arrays (chart series, ledger, regime
 * spans, the full decision trace). Every list/aggregate scan that does NOT render a chart passes these to
 * `listRecords({ omit })` so the DB backend never reads the heavy subtrees off the socket — detail / xAI
 * views re-fetch the FULL record by key. Mirrors the viewer's `HEAVY_RUN_FIELDS` (a separate runtime).
 */
export const HEAVY_RUN_FIELDS = [
  'series',
  'ledger',
  'regimes',
  'artifacts.runChart',
  'artifacts.decisionTrace',
]

/**
 * Safety valve: most points a single `series` metric array is stored with. The trainer downsamples charts
 * for display anyway, so this only clamps a producer that streams a per-timestep metric without bound —
 * beyond it the array is uniformly downsampled (first + last preserved). Generous, so real runs never hit it.
 */
export const MAX_SERIES_POINTS = 10000

/**
 * Safety valve: most steps a stored `artifacts.decisionTrace.steps` is kept with. The embedded trace is
 * meant to be producer-downsampled already (the full rollout lives in the `decisionTraceFile` sidecar), so
 * this only clamps a runaway trace — the kept steps are a contiguous prefix (adjacency preserved for the
 * trace diff) and `totalSteps` records the true pre-cap rollout length.
 */
export const MAX_DECISION_TRACE_STEPS = 20000

/** How much the LLM verdict weighs against the normalised objective in a blended score. */
export const JUDGE_LLM_WEIGHT = 0.5

/** Most runs sent to the judge/proposer in one prompt; the best-by-objective are kept. */
export const MAX_JUDGE_RUNS = 100

/** Proposals requested from the LLM when the caller names no count. */
export const DEFAULT_HYPOTHESIS_COUNT = 5

/** Candidate papers `researchTrainingPapers` discovers when the caller names no count. */
export const DEFAULT_RESEARCH_PAPER_COUNT = 8

/** Hard ceiling on the number of papers a research run will DRAFT (the target `count`). */
export const MAX_RESEARCH_PAPER_COUNT = 12

/**
 * How many candidates to discover per DRAFTED paper target — an over-scan so the paper-host ranker has a
 * pool to prefer from. The run verifies ranked candidates until it hits the target, so the low-affinity
 * tail (blogs/marketing) is only fetched+verified when the target can't be met from paper-venue hits.
 */
export const RESEARCH_DISCOVERY_OVERSCAN = 3

/**
 * Minimum verify confidence for a discovered candidate to be admitted as a draft. The verdict must also
 * sit on the supported side of the ladder (`confirmed`/`implied`); below this floor it is rejected so a
 * weakly-supported (possibly hallucinated or off-domain) paper is never drafted.
 */
export const PAPER_VERIFY_MIN_CONFIDENCE = 0.5

/**
 * Fewest divergent steps that carry a reward in both runs before a decision-trace diff will assert a
 * `better`/`worse` quality verdict; below it the read is `insufficient`. Guards against reading too much
 * into a handful of changed decisions.
 */
export const DECISION_QUALITY_MIN_SCORED_STEPS = 5

/**
 * The |mean per-step reward delta| under which a decision-trace diff reads `unchanged` — a small absolute
 * dead-band so floating-point noise isn't called a decision improvement.
 */
export const DECISION_QUALITY_REWARD_EPSILON = 1e-6

/** Bootstrap resamples for the xAI engine's interval estimates + difference tests (fixed for determinism). */
export const XAI_BOOTSTRAP_ITERATIONS = 2000

/** Confidence level for the xAI engine's interval/difference CIs. */
export const XAI_CI_LEVEL = 0.95

/** Benjamini-Hochberg false-discovery-rate level for the xAI engine's many lever-vs-baseline comparisons. */
export const XAI_FDR_ALPHA = 0.1

/**
 * Seeds a setup should have before the xAI engine trusts its interval; below it the recommender suggests
 * more seeds. The RL-reproducibility consensus warns N<5 averaging is unreliable.
 */
export const XAI_MIN_SEEDS = 5

/**
 * Fraction of host TOTAL memory the campaign pool may budget for concurrent runs when sizing the
 * RAM-aware concurrency ceiling. Deliberately off `os.totalmem()`, NOT `os.freemem()`: freemem excludes
 * reclaimable page cache (Linux `MemFree`, macOS genuinely-free pages), so after the trainer reads
 * multi-GB kline files it reads chronically low and would throttle the pool to 1 on a host with ample
 * RAM. 0.8 leaves ~20% headroom for the OS + the backend itself.
 */
export const MEMORY_BUDGET_FRACTION = 0.8

/**
 * Conservative fallback for a run's peak memory (bytes) when the manifest declares no
 * `maxMemoryBytesPerRun`. Makes the RAM-aware concurrency ceiling DEFAULT-ON: the campaign pool is
 * always bounded by host memory, erring toward under-parallelizing (never a host OOM) until a project
 * sets a measured per-run figure. 2 GiB ≈ a typical RL training process; override per-manifest to tune.
 */
export const DEFAULT_RUN_MEMORY_ESTIMATE_BYTES = 2 * 1024 * 1024 * 1024

// --- Exploration autopilot (the closed-loop config-space search) ---

/** Consecutive `global`-stage rounds that must add NO new basin before the search moves on to climb them. */
export const EXPLORATION_DRY_ROUNDS = 2

/**
 * Screening keeps a lever ACTIVE (searched) when its importance fraction clears this floor. Below it the
 * lever barely moves the objective and is frozen at its best-so-far value to spare the search budget.
 */
export const EXPLORATION_ACTIVE_IMPORTANCE_FLOOR = 0.1

/** Hard cap on how many levers the search varies at once — the highest-importance ones win the slots. */
export const EXPLORATION_MAX_ACTIVE_LEVERS = 5

/** Space-filling samples the screen stage (S1) draws across the full space before scoring lever importance. */
export const EXPLORATION_SCREEN_SAMPLES = 12

/**
 * A discrete region qualifies as a basin only when its best setup beats the trivial baseline by more than
 * this many noise-floor multiples — so seed-noise alone can never mint a spurious "maximum".
 */
export const EXPLORATION_BASIN_NOISE_MARGIN = 3

/**
 * ...AND captures at least this fraction of the baseline→global-best span. An absolute-ish bar (relative
 * to the trivial baseline, not to the observed worst) so a genuine second-best basin close to the global
 * max is still kept, while a region only marginally above random is rejected.
 */
export const EXPLORATION_BASIN_MIN_SPAN_FRACTION = 0.1
