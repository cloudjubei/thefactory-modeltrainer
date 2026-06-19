import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { TrainerManifest } from './modelTrainerTypes.js'
import { TRAINER_MANIFEST_RELPATH } from './modelTrainerConstants.js'
import {
  canonicalConfigString,
  extractPaperText,
  validateTrainerManifest,
} from './modelTrainerUtils.js'

/** Stable 12-hex identity of a resolved config — the run-record key. */
export function hashTrainingConfig(config: Record<string, unknown>): string {
  return createHash('sha256').update(canonicalConfigString(config)).digest('hex').slice(0, 12)
}

/**
 * Identity of a config's SETUP — the config minus its `seed` — so exploration can skip a
 * setup already run under any seed (vs the seed-inclusive {@link hashTrainingConfig}). When
 * homing in, seeds matter and this is not used.
 */
export function setupKeyOf(config: Record<string, unknown>): string {
  const rest = { ...config }
  delete rest.seed
  return hashTrainingConfig(rest)
}

/** arXiv pdf links are binary/JS-heavy; the /abs/ page carries title+authors+abstract as HTML. */
function arxivAbsUrl(url: string): string {
  const match = String(url).match(/arxiv\.org\/(?:pdf|abs)\/([^?#\s]+?)(?:\.pdf)?(?:[?#].*)?$/i)
  return match ? `https://arxiv.org/abs/${match[1]}` : url
}

/**
 * Fetch a paper/source URL and reduce it to readable text for the "Automatic Fill" summariser. arXiv
 * pdf/abs links are normalised to the abstract page; PDFs aren't supported (use the landing page).
 * The pure HTML→text reduction lives in {@link extractPaperText} (utils); only the network call is here.
 */
export async function fetchPaperText(url: string, abortSignal?: AbortSignal): Promise<string> {
  const target = arxivAbsUrl(url)
  const res = await fetch(target, {
    redirect: 'follow',
    signal: abortSignal,
    headers: { 'user-agent': 'thefactory-modeltrainer' },
  })
  if (!res.ok) throw new Error(`could not fetch paper (HTTP ${res.status}) from ${target}`)
  if ((res.headers.get('content-type') || '').includes('application/pdf')) {
    throw new Error(`PDF pages aren't supported yet — use the paper's abstract / landing page URL`)
  }
  const text = extractPaperText(await res.text())
  if (!text) throw new Error(`no readable text could be extracted from ${target}`)
  return text
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
