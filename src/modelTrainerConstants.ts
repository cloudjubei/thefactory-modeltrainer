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
