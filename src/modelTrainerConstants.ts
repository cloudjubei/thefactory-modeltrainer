/** Where a trainer-conformant project ships its manifest, relative to the project root. */
export const TRAINER_MANIFEST_RELPATH = '.factory/trainer.json'

/** Default safety cap on planned items per campaign; override via `ExperimentSpec.maxItems`. */
export const MAX_CAMPAIGN_ITEMS = 500

/** Provenance label stamped on run records when the caller names no compute target. */
export const DEFAULT_RAN_BY = 'local'
