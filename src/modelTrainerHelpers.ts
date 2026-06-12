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

/**
 * Read + validate a project's manifest. Defaults to `.factory/trainer.json`; pass
 * `manifestRelPath` to read a second conformant line in the same repo (e.g.
 * `.factory/trainer-dip.json`), so one checkout can register as several hub projects.
 */
export async function readTrainerManifest(
  projectRoot: string,
  manifestRelPath: string = TRAINER_MANIFEST_RELPATH,
): Promise<TrainerManifest> {
  const manifestPath = join(projectRoot, manifestRelPath || TRAINER_MANIFEST_RELPATH)
  let text: string
  try {
    text = await readFile(manifestPath, 'utf8')
  } catch {
    throw new Error(`no trainer manifest at ${manifestPath}`)
  }
  return validateTrainerManifest(JSON.parse(text))
}
