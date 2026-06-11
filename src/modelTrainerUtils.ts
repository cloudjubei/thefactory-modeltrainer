import type {
  ExperimentSpec,
  PlannedTrainingItem,
  TrainerLeverSpec,
  TrainerManifest,
  TrainingRunSummary,
} from './modelTrainerTypes.js'
import { MAX_CAMPAIGN_ITEMS } from './modelTrainerConstants.js'

const LEVER_TYPES: ReadonlySet<string> = new Set(['number', 'choice', 'boolean'])

export function validateTrainerManifest(raw: unknown): TrainerManifest {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('trainer manifest must be a JSON object')
  }
  const m = raw as Record<string, unknown>
  if (typeof m.name !== 'string' || !m.name) throw new Error('trainer manifest requires a name')
  if (typeof m.recordType !== 'string' || !m.recordType) {
    throw new Error('trainer manifest requires a recordType')
  }
  if (typeof m.run !== 'string' || !m.run.includes('{configPath}')) {
    throw new Error('trainer manifest run template must contain {configPath}')
  }
  if (!m.run.includes('{summaryOut}')) {
    throw new Error('trainer manifest run template must contain {summaryOut}')
  }
  if (m.calibrate !== undefined) {
    if (typeof m.calibrate !== 'string' || !m.calibrate.includes('{summaryOut}')) {
      throw new Error('trainer manifest calibrate template must contain {summaryOut}')
    }
  }
  if (m.evaluate !== undefined) {
    if (typeof m.evaluate !== 'string' || !m.evaluate.includes('{configPath}')) {
      throw new Error('trainer manifest evaluate template must contain {configPath}')
    }
    if (!m.evaluate.includes('{summaryOut}')) {
      throw new Error('trainer manifest evaluate template must contain {summaryOut}')
    }
  }
  const objective = m.objective as Record<string, unknown> | undefined
  if (!objective || typeof objective.name !== 'string' || !objective.name) {
    throw new Error('trainer manifest requires an objective name')
  }
  if (objective.direction !== 'max' && objective.direction !== 'min') {
    throw new Error('trainer manifest objective direction must be "max" or "min"')
  }
  const levers = m.levers as Record<string, TrainerLeverSpec> | undefined
  if (!levers || typeof levers !== 'object' || Array.isArray(levers)) {
    throw new Error('trainer manifest requires levers')
  }
  for (const [key, lever] of Object.entries(levers)) {
    if (!lever || typeof lever !== 'object' || !LEVER_TYPES.has(lever.type)) {
      throw new Error(`trainer manifest lever "${key}" has an invalid type`)
    }
  }
  const eta = m.eta as { unitsLever?: unknown } | undefined
  if (eta !== undefined) {
    if (typeof eta.unitsLever !== 'string' || !(eta.unitsLever in levers)) {
      throw new Error('trainer manifest eta.unitsLever must name a declared lever')
    }
  }
  return m as unknown as TrainerManifest
}

export function canonicalConfigString(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(canonicalConfigString).join(',')}]`
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => (a < b ? -1 : 1))
      .map(([k, v]) => `${JSON.stringify(k)}:${canonicalConfigString(v)}`)
    return `{${entries.join(',')}}`
  }
  return JSON.stringify(value)
}

export function expandExperimentMatrix(
  manifest: TrainerManifest,
  spec: ExperimentSpec,
  hashConfig: (config: Record<string, unknown>) => string,
): PlannedTrainingItem[] {
  const leverKeys = Object.keys(manifest.levers)
  for (const key of Object.keys(spec.fixed ?? {})) {
    if (!leverKeys.includes(key)) throw new Error(`fixed value "${key}" names no manifest lever`)
  }
  const sweep = spec.sweep ?? {}
  for (const [key, values] of Object.entries(sweep)) {
    if (!leverKeys.includes(key)) throw new Error(`sweep "${key}" names no manifest lever`)
    if (!Array.isArray(values) || values.length === 0) {
      throw new Error(`sweep "${key}" must be a non-empty array`)
    }
  }

  const base: Record<string, unknown> = {}
  for (const [key, lever] of Object.entries(manifest.levers)) {
    if (lever.default !== undefined) base[key] = lever.default
  }
  Object.assign(base, spec.fixed ?? {})

  let configs: Record<string, unknown>[] = [base]
  for (const [key, values] of Object.entries(sweep)) {
    configs = configs.flatMap((config) => values.map((value) => ({ ...config, [key]: value })))
  }
  if (spec.seeds && spec.seeds.length > 0) {
    configs = configs.flatMap((config) => spec.seeds!.map((seed) => ({ ...config, seed })))
  }

  const cap = spec.maxItems ?? MAX_CAMPAIGN_ITEMS
  if (configs.length > cap) {
    throw new Error(`campaign plans ${configs.length} items, exceeding the cap of ${cap}`)
  }

  const unitsLever = manifest.eta?.unitsLever
  return configs.map((config) => {
    const units =
      unitsLever && typeof config[unitsLever] === 'number'
        ? (config[unitsLever] as number)
        : undefined
    return { key: hashConfig(config), config, ...(units !== undefined ? { units } : {}) }
  })
}

export function pickBestRun(
  entries: { key: string; objective: number }[],
  direction: 'max' | 'min',
): { key: string; objective: number } | undefined {
  let best: { key: string; objective: number } | undefined
  for (const entry of entries) {
    if (
      !best ||
      (direction === 'max' ? entry.objective > best.objective : entry.objective < best.objective)
    ) {
      best = entry
    }
  }
  return best
}

export function totalCampaignUnits(items: PlannedTrainingItem[]): number | undefined {
  if (items.length === 0) return undefined
  let total = 0
  for (const item of items) {
    if (typeof item.units !== 'number') return undefined
    total += item.units
  }
  return total
}

export function normalizeObjectiveScores(
  entries: { key: string; objective: number }[],
  direction: 'max' | 'min',
): Map<string, number> {
  const scores = new Map<string, number>()
  if (entries.length === 0) return scores
  let min = Infinity
  let max = -Infinity
  for (const entry of entries) {
    if (entry.objective < min) min = entry.objective
    if (entry.objective > max) max = entry.objective
  }
  for (const entry of entries) {
    if (max === min) {
      scores.set(entry.key, 50)
      continue
    }
    const normalized = ((entry.objective - min) / (max - min)) * 100
    scores.set(entry.key, Math.round(direction === 'max' ? normalized : 100 - normalized))
  }
  return scores
}

function clamp(value: number, low: number, high: number): number {
  if (!Number.isFinite(value)) return low
  return Math.min(high, Math.max(low, value))
}

export function blendJudgeScore(
  objectiveScore: number,
  llmScore: number,
  llmWeight: number,
): number {
  const objective = clamp(objectiveScore, 0, 100)
  const llm = clamp(llmScore, 0, 100)
  const weight = clamp(llmWeight, 0, 1)
  return Math.round(llm * weight + objective * (1 - weight))
}

export function coerceVerdictRows(raw: unknown[]): { key: string; score: number; why: string }[] {
  if (!Array.isArray(raw)) return []
  const rows: { key: string; score: number; why: string }[] = []
  for (const item of raw) {
    if (!item || typeof item !== 'object') continue
    const row = item as Record<string, unknown>
    if (typeof row.key !== 'string' || row.key.length === 0) continue
    rows.push({
      key: row.key,
      score: Math.round(clamp(typeof row.score === 'number' ? row.score : 0, 0, 100)),
      why: typeof row.why === 'string' ? row.why : '',
    })
  }
  return rows
}

export function coerceHypothesisItems(
  raw: unknown[],
  manifest: TrainerManifest,
): { title: string; rationale: string; spec: ExperimentSpec }[] {
  if (!Array.isArray(raw)) return []
  const leverKeys = new Set(Object.keys(manifest.levers))
  const items: { title: string; rationale: string; spec: ExperimentSpec }[] = []
  for (const item of raw) {
    if (!item || typeof item !== 'object') continue
    const obj = item as Record<string, unknown>
    if (typeof obj.title !== 'string' || !obj.title) continue
    if (typeof obj.rationale !== 'string' || !obj.rationale) continue
    const rawSpec = obj.spec as Record<string, unknown> | undefined
    if (!rawSpec || typeof rawSpec !== 'object') continue

    const spec: ExperimentSpec = {}
    let valid = true
    const sweep = rawSpec.sweep as Record<string, unknown> | undefined
    if (sweep && typeof sweep === 'object') {
      const entries = Object.entries(sweep)
      for (const [key, values] of entries) {
        if (!leverKeys.has(key) || !Array.isArray(values) || values.length === 0) {
          valid = false
          break
        }
      }
      if (valid && entries.length > 0) spec.sweep = sweep as Record<string, unknown[]>
    }
    const fixed = rawSpec.fixed as Record<string, unknown> | undefined
    if (valid && fixed && typeof fixed === 'object') {
      const entries = Object.entries(fixed)
      for (const [key] of entries) {
        if (!leverKeys.has(key)) {
          valid = false
          break
        }
      }
      if (valid && entries.length > 0) spec.fixed = fixed
    }
    if (!valid || (!spec.sweep && !spec.fixed)) continue
    if (Array.isArray(rawSpec.seeds)) {
      spec.seeds = rawSpec.seeds
        .filter((s): s is number => typeof s === 'number' && Number.isFinite(s))
        .map((s) => Math.trunc(s))
    }
    items.push({ title: obj.title, rationale: obj.rationale, spec })
  }
  return items
}

export function buildJudgeSystemPrompt(manifest: TrainerManifest, instructions?: string): string {
  return [
    `You are an exacting ML experiment judge for the "${manifest.name}" training project.`,
    `Each run reports the objective "${manifest.objective.name}" (direction: ${manifest.objective.direction} is better) plus its config and metrics.`,
    `Score how PROMISING each run's configuration is for further investment, 0-100 — weigh the objective against signs of luck, instability or overfitting visible in the metrics, and prefer configurations whose neighbours also perform well.`,
    instructions ? `Additional rubric: ${instructions}` : '',
    `Return ONLY a JSON array, one row per run: [{"key": "<run key>", "score": <0-100>, "why": "<one concise sentence>"}]. No prose around it.`,
  ]
    .filter(Boolean)
    .join('\n')
}

export function buildJudgeUserContent(
  runs: {
    key: string
    objective: number
    config?: Record<string, unknown>
    metrics?: Record<string, number>
    seed?: number
  }[],
): string {
  return JSON.stringify(runs)
}

export function buildProposeSystemPrompt(
  manifest: TrainerManifest,
  count: number,
  instructions?: string,
): string {
  return [
    `You are an ML experiment designer for the "${manifest.name}" training project.`,
    `Objective: ${manifest.objective.name} (${manifest.objective.direction} is better).`,
    `The ONLY tunable levers, with their allowed shapes, are: ${JSON.stringify(manifest.levers)}.`,
    `Given the run history and verdicts, propose up to ${count} NEW experiment specs likely to beat the best run. Explore promising neighbourhoods and untested regions; avoid repeating configurations already run.`,
    instructions ? `Additional guidance: ${instructions}` : '',
    `Return ONLY a JSON array: [{"title": "<short name>", "rationale": "<why this is promising>", "spec": {"sweep": {"<lever>": [values]}, "fixed": {"<lever>": value}, "seeds": [0]}}]. Use only declared lever names; sweep arrays must be non-empty; every spec needs a sweep or a fixed. No prose around it.`,
  ]
    .filter(Boolean)
    .join('\n')
}

export function buildProposeUserContent(input: {
  manifest: TrainerManifest
  runs: { key: string; objective: number; config?: Record<string, unknown> }[]
  verdicts: { key: string; score: number; why: string }[]
  bestObjective?: number
}): string {
  return JSON.stringify({
    objective: input.manifest.objective,
    bestObjective: input.bestObjective,
    runs: input.runs,
    verdicts: input.verdicts,
  })
}

export function validateTrainingRunSummary(raw: unknown): TrainingRunSummary {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('run summary must be a JSON object')
  }
  const summary = raw as Record<string, unknown>
  if (typeof summary.objective !== 'number' || Number.isNaN(summary.objective)) {
    throw new Error('run summary requires a numeric objective')
  }
  return summary as unknown as TrainingRunSummary
}
