/** Where a trainer-conformant project ships its manifest, relative to the project root. */
export const TRAINER_MANIFEST_RELPATH = '.factory/trainer.json'

/** Default safety cap on planned items per campaign; override via `ExperimentSpec.maxItems`. */
export const MAX_CAMPAIGN_ITEMS = 500

/** Provenance label stamped on run records when the caller names no compute target. */
export const DEFAULT_RAN_BY = 'local'

/** How much the LLM verdict weighs against the normalised objective in a blended score. */
export const JUDGE_LLM_WEIGHT = 0.5

/** Most runs sent to the judge/proposer in one prompt; the best-by-objective are kept. */
export const MAX_JUDGE_RUNS = 100

/** Proposals requested from the LLM when the caller names no count. */
export const DEFAULT_HYPOTHESIS_COUNT = 5

/** Candidate papers `researchTrainingPapers` discovers when the caller names no count. */
export const DEFAULT_RESEARCH_PAPER_COUNT = 8

/** Hard ceiling on candidate papers per research run (each costs a fetch + verify panel + synthesis). */
export const MAX_RESEARCH_PAPER_COUNT = 12

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
