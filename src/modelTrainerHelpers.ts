import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { TrainerManifest } from './modelTrainerTypes.js'
import { TRAINER_MANIFEST_RELPATH } from './modelTrainerConstants.js'
import { canonicalConfigString, validateTrainerManifest } from './modelTrainerUtils.js'

/** Stable 12-hex identity of a resolved config — the run-record key. */
export function hashTrainingConfig(config: Record<string, unknown>): string {
  return createHash('sha256').update(canonicalConfigString(config)).digest('hex').slice(0, 12)
}

/** Read + validate a project's manifest from `.factory/trainer.json`. */
export async function readTrainerManifest(projectRoot: string): Promise<TrainerManifest> {
  const manifestPath = join(projectRoot, TRAINER_MANIFEST_RELPATH)
  let text: string
  try {
    text = await readFile(manifestPath, 'utf8')
  } catch {
    throw new Error(`no trainer.json manifest at ${manifestPath}`)
  }
  return validateTrainerManifest(JSON.parse(text))
}
