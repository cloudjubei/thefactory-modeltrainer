import type {
  DecisionFeatureAttribution,
  DecisionQualitySignal,
  DecisionStep,
  DecisionStepDelta,
  DecisionTrace,
  DecisionTraceDiff,
  ExperimentSpec,
  ModelCategory,
  PlannedTrainingItem,
  ProposedModel,
  RunXaiDigest,
  TrainerDataFile,
  TrainerLeverSpec,
  TrainerManifest,
  TrainerMigrationRule,
  TrainingPaperRecord,
  TrainingRunSummary,
} from './modelTrainerTypes.js'
import {
  DECISION_QUALITY_MIN_SCORED_STEPS,
  DECISION_QUALITY_REWARD_EPSILON,
  MAX_CAMPAIGN_ITEMS,
} from './modelTrainerConstants.js'

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
  if (m.data !== undefined) {
    if (!Array.isArray(m.data)) throw new Error('trainer manifest data must be an array')
    for (const entry of m.data as Record<string, unknown>[]) {
      if (!entry || typeof entry.id !== 'string' || !entry.id) {
        throw new Error('trainer manifest data entries require an id')
      }
      if (!Array.isArray(entry.files) || entry.files.length === 0) {
        throw new Error(`trainer manifest data entry "${entry.id}" requires non-empty files`)
      }
      for (const file of entry.files as Record<string, unknown>[]) {
        if (!file || typeof file.relPath !== 'string' || !file.relPath) {
          throw new Error(`trainer manifest data entry "${entry.id}" has a file without a relPath`)
        }
        if (typeof file.url !== 'string' || !file.url) {
          throw new Error(`trainer manifest data entry "${entry.id}" has a file without a url`)
        }
      }
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

/**
 * The FIRST {@link TrainerMigrationRule} that handles `config`, or `null` when none do. A rule matches
 * when every `match` field is present and loosely equal (so a JSON `0` matches a stored number/string
 * `0`) AND every `matchNot` field is present and loosely UNEQUAL — so a record missing a `matchNot`
 * field is never matched (runs without that key are left alone). A rule with neither clause matches
 * nothing.
 */
export function findMigrationRule(
  config: Record<string, unknown>,
  rules: TrainerMigrationRule[],
): TrainerMigrationRule | null {
  const matches = (rule: TrainerMigrationRule): boolean => {
    if (!rule.match && !rule.matchNot) return false
    const eq = Object.entries(rule.match ?? {}).every(
      ([k, v]) => k in config && String(config[k]) === String(v),
    )
    const neq = Object.entries(rule.matchNot ?? {}).every(
      ([k, v]) => k in config && String(config[k]) !== String(v),
    )
    return eq && neq
  }
  return rules.find(matches) ?? null
}

/**
 * Roll one config forward through a project's rules for the REWRITE path: returns the new config
 * (`{...config, ...set}` with each `keepOrDefault` key resolved to the config's current value when
 * present, else the rule's default), or `null` when no rule matches OR the matched rule is a `delete`
 * rule (deletion isn't a config the caller can use — see {@link findMigrationRule} for that decision).
 * Re-running once everything is migrated is a no-op.
 */
export function applyMigrationRules(
  config: Record<string, unknown>,
  rules: TrainerMigrationRule[],
): Record<string, unknown> | null {
  const rule = findMigrationRule(config, rules)
  if (!rule || rule.delete) return null
  const out: Record<string, unknown> = { ...config, ...rule.set }
  for (const [k, fallback] of Object.entries(rule.keepOrDefault ?? {})) {
    out[k] = k in config ? config[k] : fallback
  }
  return out
}

/**
 * Roll an {@link ExperimentSpec} forward through the same migration rules before it is planned, so a
 * run dispatched from an OLD queued/pending spec (e.g. `reward_model: "combo_all"`) executes under the
 * migrated shape rather than the retired one. Only `spec.fixed` is migrated (the pinned config a single
 * run carries); a `sweep` over a migrated lever is left untouched (its values are a list, not a config).
 * Returns the SAME spec object when no rule matches, so callers can treat it as a cheap pass-through.
 */
export function migrateExperimentSpec(
  spec: ExperimentSpec,
  migrations: TrainerMigrationRule[] | undefined,
): ExperimentSpec {
  if (!migrations || migrations.length === 0 || !spec || !spec.fixed) return spec
  const migrated = applyMigrationRules(spec.fixed, migrations)
  return migrated ? { ...spec, fixed: migrated } : spec
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
  const environments = spec.environments ?? []
  for (const bundle of environments) {
    for (const key of Object.keys(bundle)) {
      if (!leverKeys.includes(key)) {
        throw new Error(`environment value "${key}" names no manifest lever`)
      }
    }
  }
  const datasets = spec.datasets ?? []
  for (const bundle of datasets) {
    for (const key of Object.keys(bundle)) {
      if (!leverKeys.includes(key)) {
        throw new Error(`dataset value "${key}" names no manifest lever`)
      }
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
  // Dataset + environment bundles apply TOGETHER (not cartesian): each crosses the whole model matrix.
  if (datasets.length > 0) {
    configs = configs.flatMap((config) => datasets.map((bundle) => ({ ...config, ...bundle })))
  }
  if (environments.length > 0) {
    configs = configs.flatMap((config) => environments.map((bundle) => ({ ...config, ...bundle })))
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

/** Flatten a manifest's declared datasets into the per-job data file list. */
export function manifestDataFiles(manifest: TrainerManifest): TrainerDataFile[] | undefined {
  if (!manifest.data || manifest.data.length === 0) return undefined
  return manifest.data.flatMap((entry) => entry.files)
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

/**
 * A backstop for the "must be a TESTABLE claim" rule the propose/extract/suggest prompts enforce: reject
 * items that are pure DATA-GATHERING (more seeds/runs of a known setup to tighten a confidence interval,
 * reproduce, or reduce noise) rather than a falsifiable conjecture. Such "establish a trustworthy interval"
 * work is handled by the min-runs gate, "Launch more runs", and the xAI tab — never as a hypothesis. The
 * patterns are deliberately high-precision (a real claim rarely uses this phrasing) to avoid false drops.
 */
export function looksLikeDataGathering(title: string, rationale: string): boolean {
  const t = `${title || ''} ${rationale || ''}`.toLowerCase()
  return (
    /\b(\d+\s+more|more|additional|extra|further|repeat(?:ed)?)\s+seeds?\b/.test(t) ||
    /\b(trustworthy|confidence|reliable|tighter|narrower|stable|robust)\s+(interval|ci|estimate)\b/.test(
      t,
    ) ||
    /\bestablish(?:ing)?\b[^.]*\binterval\b/.test(t) ||
    /\b(gather|collect|accumulate|get)\b[^.]*\bmore\s+(data|runs|seeds|samples|evidence)\b/.test(
      t,
    ) ||
    /\b(more|additional|extra)\s+runs?\b[^.]*\b(establish|confirm|tighten|interval|trust|reproduc)/.test(
      t,
    )
  )
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
    // Drop "run more seeds to establish an interval"-style data-gathering — not a falsifiable test.
    if (looksLikeDataGathering(obj.title, obj.rationale)) continue
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

// The bar every proposed/extracted/suggested hypothesis must clear, shared by all three prompts: a
// FALSIFIABLE test-claim, never "gather more data for a known setup" (that's the min-runs gate + xAI tab).
const HYPOTHESIS_RULE =
  'Each hypothesis MUST be a FALSIFIABLE claim a run can PROVE OR DISPROVE — that a specific lever value, ' +
  'or a comparison against a baseline, CHANGES the objective in a stated direction. It must be able to turn ' +
  'out FALSE. Do NOT propose pure DATA-GATHERING: running more seeds/runs of an already-identified config to ' +
  'tighten a confidence interval, reduce variance/noise, reproduce, or "establish a trustworthy interval" is ' +
  'NOT a hypothesis — that is handled separately (the min-runs threshold and the xAI tab). Every spec must ' +
  'introduce or compare a lever setting whose effect is what is being tested.'

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
    HYPOTHESIS_RULE,
    instructions ? `Additional guidance: ${instructions}` : '',
    `Return ONLY a JSON array: [{"title": "<short name>", "rationale": "<the falsifiable claim being tested>", "spec": {"sweep": {"<lever>": [values]}, "fixed": {"<lever>": value}, "seeds": [0]}}]. Use only declared lever names; sweep arrays must be non-empty; every spec needs a sweep or a fixed. No prose around it.`,
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

function round2(v: number): number {
  return Math.round(v * 100) / 100
}

/** The system prompt for the one-shot PER-RUN xAI narrative — explain this run, hedge on weak signals. */
export function buildXaiNarrateSystemPrompt(manifest: TrainerManifest): string {
  return [
    `You are an ML interpretability analyst for the "${manifest.name}" training project.`,
    `Objective: ${manifest.objective.name} (${manifest.objective.direction} is better).`,
    `Below is the DETERMINISTIC xAI analysis of ONE specific run. Write a SHORT narrative (3–6 sentences, plain prose — NO headings or bullet lists): what this model is DOING (its decisions/action mix), what DRIVES those decisions (input attribution), how TRUSTWORTHY that explanation is, how it compares to its nearest sibling, and the single most valuable thing to try next.`,
    `Be specific and HONEST about uncertainty: if the attribution FAILED its sanity check, say the input explanation is unreliable; lever importances are CONFOUNDED screening signals; a decision-quality verdict is heuristic, not causal; low-data estimates are weak. Synthesise; don't restate the numbers verbatim. No preamble.`,
  ].join('\n')
}

/** Compact, model-readable digest of ONE run's deterministic xAI analysis for the narrative. Pure. */
export function buildXaiNarrateUserContent(input: RunXaiDigest): string {
  const label = input.criterion.label || input.criterion.key
  const pct = (v: number) => `${Math.round(v * 100)}%`
  const cfg = Object.entries(input.config)
    .filter(([k]) => k !== 'seed')
    .map(([k, v]) => `${k}=${v}`)
    .join(' ')
  const lines = [
    `Run ${input.runKey.slice(0, 8)} — config {${cfg}}.`,
    `${label}: ${input.objective === undefined ? 'n/a' : round2(input.objective)}${
      input.rank ? ` (ranks #${input.rank.position} of ${input.rank.total})` : ''
    }.`,
  ]
  if (input.actionCounts && Object.keys(input.actionCounts).length) {
    lines.push(
      `Action mix: ` +
        Object.entries(input.actionCounts)
          .sort((a, b) => b[1] - a[1])
          .map(([a, c]) => `${a}=${c}`)
          .join(', ') +
        '.',
    )
  }
  if (input.attribution && input.attribution.topGroups.length) {
    const top = input.attribution.topGroups.map(([k, v]) => `${k}(${round2(v)})`).join(', ')
    const sanity =
      input.attribution.sanityPassed === undefined
        ? ''
        : input.attribution.sanityPassed
          ? ` — sanity check PASSED (rank corr ${round2(input.attribution.sanityRankCorr ?? 0)}), so this attribution is trustworthy`
          : ` — sanity check FAILED (rank corr ${round2(input.attribution.sanityRankCorr ?? 0)}): the attribution barely changes under weight randomization, so it likely reflects the input/architecture, NOT what the model learned`
    lines.push(`Input attribution (${input.attribution.method || 'saliency'}): ${top}${sanity}.`)
  }
  if (input.rewardBreakdown && Object.keys(input.rewardBreakdown).length) {
    lines.push(
      `Reward breakdown (why this reward): ` +
        Object.entries(input.rewardBreakdown)
          .filter(([k]) => k !== 'total')
          .map(([k, v]) => `${k} ${Number(v) >= 0 ? '+' : ''}${round2(Number(v))}`)
          .join(', ') +
        '.',
    )
  }
  if (input.latent && input.latent.probeAccuracy !== undefined) {
    lines.push(
      `Latent: a linear probe predicts the action from the penultimate activations at ${pct(
        input.latent.probeAccuracy,
      )} vs a ${pct(input.latent.probeBaseline ?? 0)} majority baseline${
        input.latent.varianceExplained !== undefined
          ? ` (2-D projection keeps ${pct(input.latent.varianceExplained)} variance)`
          : ''
      }.`,
    )
  }
  if (input.sibling) {
    lines.push(
      `Vs nearest sibling ${input.sibling.key.slice(0, 8)} (changed ${input.sibling.changed}): ${input.sibling.divergencePct}% of aligned decisions differ; decision-quality ${input.sibling.qualityVerdict || 'n/a'}${
        input.sibling.qualitySummary ? ` — ${input.sibling.qualitySummary}` : ''
      } (heuristic, not causal).`,
    )
  }
  if (input.importances.length) {
    lines.push(
      `Cross-run lever importance (CONFOUNDED screening) for context: ` +
        input.importances
          .slice(0, 4)
          .map((i) => `${i.lever} ${pct(i.importance)} (best≈${i.bestValue})`)
          .join(', ') +
        '.',
    )
  }
  return lines.join('\n')
}

/** Upper bound on extracted paper text handed to the model — enough for an abstract/intro, bounded cost. */
export const PAPER_TEXT_CAP = 12000

const _HTML_ENTITIES: Record<string, string> = {
  '&amp;': '&',
  '&lt;': '<',
  '&gt;': '>',
  '&quot;': '"',
  '&#39;': "'",
  '&nbsp;': ' ',
}

/** Strip HTML to readable text (scripts/styles/tags removed, entities decoded, whitespace collapsed),
 * capped to {@link PAPER_TEXT_CAP}. Pure — the network fetch lives in the helpers layer. */
export function extractPaperText(raw: string): string {
  let text = String(raw || '')
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
  for (const [entity, ch] of Object.entries(_HTML_ENTITIES)) text = text.split(entity).join(ch)
  text = text
    .replace(/&#\d+;/g, ' ')
    .replace(/&[a-z]+;/gi, ' ')
    .replace(/\s+/g, ' ')
    .trim()
  return text.length > PAPER_TEXT_CAP ? text.slice(0, PAPER_TEXT_CAP) : text
}

/** System prompt for "Automatic Fill": the model is GIVEN the paper text and must return one honest
 * registry-entry JSON object (no browsing, no prose). */
export function buildAnalyzePaperSystemPrompt(manifest: TrainerManifest, notes?: string): string {
  return [
    `You are a research librarian for the "${manifest.name}" training project.`,
    `Objective: ${manifest.objective.name} (${manifest.objective.direction} is better).`,
    `You are given the TEXT of a paper/source (already fetched — DO NOT browse). Summarise it HONESTLY as a registry entry AND break it into the distinct, runnable HYPOTHESES it makes for this project.`,
    `The project's tunable levers (for the hypothesis specs) are: ${JSON.stringify(manifest.levers)}.`,
    HYPOTHESIS_RULE,
    notes ? `Extra guidance: ${notes}` : '',
    `Return ONLY a single JSON object (no prose, no code fence): {"title": string (required), "authors"?: string, ` +
      `"year"?: number, "claim": string (the source's headline claim in its own terms, required), "approach"?: string, ` +
      `"claimedMetrics"?: {"<name>": number}, "assumptions"?: {"fees"?: boolean, "netOfCosts"?: boolean, ` +
      `"frictionless"?: boolean, "multiAsset"?: boolean, "retrainCadence"?: string, "notes"?: string}, ` +
      `"verdictNote"?: string (skeptical — does it likely survive real costs + walk-forward OOS?), "tags"?: [string], ` +
      `"hypotheses"?: [{"title": string, "rationale": string, "spec": {"fixed"?: {"<lever>": value}, ` +
      `"sweep"?: {"<lever>": [values]}, "seeds"?: [number]}}] ` +
      `(one per DISTINCT testable claim, each a runnable spec using ONLY declared lever names; [] if it maps to no runnable setup)}. ` +
      `Be honest about assumptions that inflate results (no fees, in-sample, single split).`,
  ]
    .filter(Boolean)
    .join('\n')
}

export function buildAnalyzePaperUserContent(input: {
  url: string
  text: string
  notes?: string
}): string {
  return JSON.stringify({ url: input.url, notes: input.notes, text: input.text })
}

/** System prompt for "Suggest hypotheses": match the paper against EXISTING hypotheses (link the ones
 * that test its claims) AND propose NEW testable hypotheses not already covered. */
export function buildSuggestHypothesesSystemPrompt(manifest: TrainerManifest): string {
  return [
    `You are a research librarian for the "${manifest.name}" training project.`,
    `Objective: ${manifest.objective.name} (${manifest.objective.direction} is better).`,
    `You are given a PAPER (its fields, and its text when available) and the project's EXISTING hypotheses.`,
    `The project's tunable levers (for any new hypothesis spec) are: ${JSON.stringify(manifest.levers)}.`,
    `Do TWO things: (1) MATCH — pick the existing hypotheses (by id) that are genuine tests of THIS paper's claims; (2) SUGGEST — propose NEW testable hypotheses for the paper's claims that NO existing hypothesis already covers (each a runnable spec using ONLY declared lever names). Do not duplicate an existing hypothesis as a new one.`,
    HYPOTHESIS_RULE,
    `Return ONLY a single JSON object (no prose, no code fence): {"matchExistingIds": [string], ` +
      `"newHypotheses": [{"title": string, "rationale": string, "spec": {"fixed"?: {"<lever>": value}, ` +
      `"sweep"?: {"<lever>": [values]}, "seeds"?: [number]}}]}. Use [] for either when there is nothing to add.`,
  ].join('\n')
}

export function buildSuggestHypothesesUserContent(input: {
  paper: Record<string, unknown>
  existingHypotheses: Array<{ id: string; title?: string; rationale?: string; spec?: unknown }>
  text?: string
}): string {
  return JSON.stringify({
    paper: input.paper,
    existingHypotheses: input.existingHypotheses,
    ...(input.text ? { text: input.text } : {}),
  })
}

/** Coerce the model's suggest response into matched-existing ids + validated new hypothesis items. */
export function coerceSuggestedHypotheses(
  raw: unknown,
  manifest: TrainerManifest,
): { matchIds: string[]; newItems: { title: string; rationale: string; spec: ExperimentSpec }[] } {
  const o =
    raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {}
  const matchIds = Array.isArray(o.matchExistingIds)
    ? o.matchExistingIds.filter((x): x is string => typeof x === 'string' && x.length > 0)
    : []
  const newItems = coerceHypothesisItems(
    Array.isArray(o.newHypotheses) ? o.newHypotheses : [],
    manifest,
  )
  return { matchIds, newItems }
}

/** Defensively coerce the model's JSON into a Paper draft — `undefined` unless title + claim are
 * present (mirrors {@link coerceHypothesisItems}). Drops unknown/ill-typed fields; the tool stamps
 * id/url/status/source/timestamps. */
export function coercePaperDraft(raw: unknown): Partial<TrainingPaperRecord> | undefined {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return undefined
  const o = raw as Record<string, unknown>
  const str = (v: unknown) => (typeof v === 'string' && v.trim() ? v.trim() : undefined)
  const title = str(o.title)
  const claim = str(o.claim)
  if (!title || !claim) return undefined
  const draft: Partial<TrainingPaperRecord> = { title, claim }
  const authors = str(o.authors)
  if (authors) draft.authors = authors
  if (typeof o.year === 'number' && Number.isFinite(o.year)) draft.year = o.year
  const approach = str(o.approach)
  if (approach) draft.approach = approach
  const verdictNote = str(o.verdictNote)
  if (verdictNote) draft.verdictNote = verdictNote
  if (
    o.claimedMetrics &&
    typeof o.claimedMetrics === 'object' &&
    !Array.isArray(o.claimedMetrics)
  ) {
    const metrics: Record<string, number> = {}
    for (const [k, v] of Object.entries(o.claimedMetrics as Record<string, unknown>)) {
      if (typeof v === 'number' && Number.isFinite(v)) metrics[k] = v
    }
    if (Object.keys(metrics).length) draft.claimedMetrics = metrics
  }
  if (o.assumptions && typeof o.assumptions === 'object' && !Array.isArray(o.assumptions)) {
    draft.assumptions = o.assumptions as TrainingPaperRecord['assumptions']
  }
  if (Array.isArray(o.tags)) {
    const tags = o.tags.filter((x): x is string => typeof x === 'string')
    if (tags.length) draft.tags = tags
  }
  return draft
}

const PROGRESS_MARKER = '@@PROGRESS '

/**
 * Extract a structured progress object from a `@@PROGRESS {json}` log line a
 * conformant trainer emits during a run; `undefined` for any other line. Lets
 * the campaign surface real within-run sub-progress (phase, data done/total)
 * without the engine knowing anything domain-specific.
 */
export function parseProgressMarker(line: string): Record<string, unknown> | undefined {
  const at = line.indexOf(PROGRESS_MARKER)
  if (at < 0) return undefined
  const rest = line.slice(at + PROGRESS_MARKER.length).trim()
  const end = rest.lastIndexOf('}')
  if (!rest.startsWith('{') || end < 0) return undefined
  try {
    const parsed = JSON.parse(rest.slice(0, end + 1))
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : undefined
  } catch {
    return undefined
  }
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

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === 'object' && !Array.isArray(v)

const isFiniteNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

/** Coerce a record to label → finite number, dropping non-numeric entries; `undefined` when empty. */
function coerceNumberMap(raw: unknown): Record<string, number> | undefined {
  if (!isPlainObject(raw)) return undefined
  const out: Record<string, number> = {}
  for (const [k, v] of Object.entries(raw)) if (isFiniteNumber(v)) out[k] = v
  return Object.keys(out).length ? out : undefined
}

/** Keep only the finite numbers from an array; `undefined` when the input is not an array. */
function coerceNumberArray(raw: unknown): number[] | undefined {
  if (!Array.isArray(raw)) return undefined
  return raw.filter(isFiniteNumber)
}

function coerceDecisionStep(raw: unknown): DecisionStep | undefined {
  if (!isPlainObject(raw)) return undefined
  if (!isFiniteNumber(raw.step) || typeof raw.action !== 'string') return undefined
  const step: DecisionStep = { step: raw.step, action: raw.action }
  if (isFiniteNumber(raw.confidence)) step.confidence = raw.confidence
  const actionValues = coerceNumberMap(raw.actionValues)
  if (actionValues) step.actionValues = actionValues
  if (typeof raw.alternativeAction === 'string') step.alternativeAction = raw.alternativeAction
  if (typeof raw.forced === 'boolean') step.forced = raw.forced
  if (isFiniteNumber(raw.reward)) step.reward = raw.reward
  if (typeof raw.state === 'string') step.state = raw.state
  const features = coerceNumberArray(raw.features)
  if (features && features.length) step.features = features
  return step
}

function coerceFeatureAttribution(raw: unknown): DecisionFeatureAttribution | undefined {
  if (!isPlainObject(raw)) return undefined
  const out: DecisionFeatureAttribution = {}
  const perFeature = coerceNumberArray(raw.perFeature)
  if (perFeature && perFeature.length) out.perFeature = perFeature
  const byGroup = coerceNumberMap(raw.byGroup)
  if (byGroup) out.byGroup = byGroup
  if (typeof raw.method === 'string') out.method = raw.method
  if (isFiniteNumber(raw.samples)) out.samples = raw.samples
  if (isPlainObject(raw.sanityCheck)) {
    const sc = raw.sanityCheck
    const sanityCheck: NonNullable<DecisionFeatureAttribution['sanityCheck']> = {}
    if (typeof sc.method === 'string') sanityCheck.method = sc.method
    if (isFiniteNumber(sc.rankCorrelation)) sanityCheck.rankCorrelation = sc.rankCorrelation
    if (typeof sc.passed === 'boolean') sanityCheck.passed = sc.passed
    if (Object.keys(sanityCheck).length) out.sanityCheck = sanityCheck
  }
  return out.perFeature || out.byGroup ? out : undefined
}

/**
 * Soft-validate a stored `artifacts.decisionTrace` into a clean {@link DecisionTrace}, dropping malformed
 * steps and fields rather than throwing — a missing or unusable trace is NOT an error (returns
 * `undefined`), so a run without explainability data ingests normally.
 */
export function validateDecisionTrace(raw: unknown): DecisionTrace | undefined {
  if (!isPlainObject(raw) || !Array.isArray(raw.steps)) return undefined
  const steps = raw.steps.map(coerceDecisionStep).filter((s): s is DecisionStep => s !== undefined)
  if (!steps.length) return undefined
  const trace: DecisionTrace = { steps }
  const actionCounts = coerceNumberMap(raw.actionCounts)
  if (actionCounts) trace.actionCounts = actionCounts
  const featureAttribution = coerceFeatureAttribution(raw.featureAttribution)
  if (featureAttribution) trace.featureAttribution = featureAttribution
  if (isFiniteNumber(raw.totalSteps)) trace.totalSteps = raw.totalSteps
  const rewardBreakdown = coerceNumberMap(raw.rewardBreakdown)
  if (rewardBreakdown) trace.rewardBreakdown = rewardBreakdown
  const latentMap = coerceLatentMap(raw.latentMap)
  if (latentMap) trace.latentMap = latentMap
  return trace
}

function coerceLatentMap(raw: unknown): DecisionTrace['latentMap'] | undefined {
  if (!isPlainObject(raw) || !Array.isArray(raw.points)) return undefined
  const points = raw.points
    .filter(
      (p): p is { x: number; y: number; action: string } =>
        isPlainObject(p) &&
        isFiniteNumber(p.x) &&
        isFiniteNumber(p.y) &&
        typeof p.action === 'string',
    )
    .map((p) => ({ x: p.x, y: p.y, action: p.action }))
  if (points.length < 3) return undefined
  const out: NonNullable<DecisionTrace['latentMap']> = { points }
  if (isFiniteNumber(raw.varianceExplained)) out.varianceExplained = raw.varianceExplained
  if (isFiniteNumber(raw.dim)) out.dim = raw.dim
  if (typeof raw.method === 'string') out.method = raw.method
  if (isPlainObject(raw.probe)) {
    const p = raw.probe
    const probe: NonNullable<NonNullable<DecisionTrace['latentMap']>['probe']> = {}
    if (isFiniteNumber(p.accuracy)) probe.accuracy = p.accuracy
    if (isFiniteNumber(p.baseline)) probe.baseline = p.baseline
    if (isFiniteNumber(p.classes)) probe.classes = p.classes
    if (typeof p.method === 'string') probe.method = p.method
    if (isFiniteNumber(p.testSize)) probe.testSize = p.testSize
    if (Object.keys(probe).length) out.probe = probe
  }
  return out
}

// The dataset fields that determine the STEP AXIS (so two runs sharing them tested the same bars).
// Deliberately excludes observation-only fields (fidelity_set/layers) — those are exactly the "new
// information" tweaks we want to diff, and they don't change the step count.
const ALIGNMENT_DATASET_KEYS = ['asset', 'timeframe', 'candles', 'from', 'to'] as const

/**
 * A stable dataset/window signature for step-alignment, read off `summary.dataset` — only runs with the
 * SAME signature share a step axis and are safely diffable. Empty when no dataset is recorded (callers
 * treat two empty signatures as NOT auto-alignable).
 */
export function datasetAlignmentSignature(summary: TrainingRunSummary): string {
  const dataset = summary.dataset as Record<string, unknown> | undefined
  if (!dataset || typeof dataset !== 'object') return ''
  const parts: string[] = []
  for (const key of ALIGNMENT_DATASET_KEYS) {
    const value = dataset[key]
    if (value !== undefined && value !== null) parts.push(`${key}=${value}`)
  }
  return parts.join('|')
}

function averageOf(values: number[]): number | undefined {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : undefined
}

function fmtDelta(value: number | undefined): string {
  if (value === undefined) return 'n/a'
  const rounded = Number(value.toFixed(4))
  return rounded >= 0 ? `+${rounded}` : `${rounded}`
}

/**
 * Map the changed-step and unchanged-step reward deltas to an HONEST decision-quality verdict. The
 * changed-step gain must clear a dead-band AND beat the unchanged-step CONTROL to read `better`/`worse`
 * (so a whole-rollout regime move isn't mistaken for a decision improvement); too few scored steps reads
 * `insufficient`. Never claims causation. Pure.
 */
function classifyDecisionQuality(
  changedRewardDeltas: number[],
  unchangedRewardDeltas: number[],
): DecisionQualitySignal {
  const scoredChangedSteps = changedRewardDeltas.length
  const onChanges = averageOf(changedRewardDeltas)
  const onUnchanged = averageOf(unchangedRewardDeltas)
  const base: DecisionQualitySignal = {
    scoredChangedSteps,
    ...(onChanges !== undefined ? { meanRewardDeltaOnChanges: onChanges } : {}),
    ...(onUnchanged !== undefined ? { meanRewardDeltaOnUnchanged: onUnchanged } : {}),
    verdict: 'insufficient',
    summary: '',
  }
  if (scoredChangedSteps < DECISION_QUALITY_MIN_SCORED_STEPS) {
    return {
      ...base,
      summary: `Only ${scoredChangedSteps}/${DECISION_QUALITY_MIN_SCORED_STEPS} changed steps carry a reward — too few to read decision quality (heuristic, not causal).`,
    }
  }
  const change = onChanges ?? 0
  const control = onUnchanged ?? 0
  const eps = DECISION_QUALITY_REWARD_EPSILON
  let verdict: DecisionQualitySignal['verdict']
  if (Math.abs(change) <= eps) verdict = 'unchanged'
  else if (change > eps) verdict = change > control + eps ? 'better' : 'mixed'
  else verdict = change < control - eps ? 'worse' : 'mixed'
  const summary =
    verdict === 'unchanged'
      ? `Where decisions changed, per-step reward barely moved (${fmtDelta(change)}) — heuristic, not causal.`
      : verdict === 'mixed'
        ? `At changed steps reward moved ${fmtDelta(change)}, but unchanged steps shifted ~equally (${fmtDelta(control)}) — likely the rollout, not the decisions (heuristic, not causal).`
        : `At the ${scoredChangedSteps} changed steps the tweak averaged ${fmtDelta(change)} reward vs baseline (control ${fmtDelta(control)} on unchanged) — decisions look ${verdict} (heuristic, not causal).`
  return { ...base, verdict, summary }
}

function notAlignedDiff(note: string, signature: string): DecisionTraceDiff {
  return {
    aligned: false,
    alignmentNote: note,
    ...(signature ? { datasetSignature: signature } : {}),
    alignedSteps: 0,
    changedSteps: 0,
    divergenceRate: 0,
    steps: [],
    actionCountDeltas: {},
    quality: classifyDecisionQuality([], []),
  }
}

/**
 * Diff two runs' decision traces step-by-step — how a lever tweak (the "new information") changed the
 * model's DECISIONS, with a decision-quality read kept separate from the objective. Returns `undefined`
 * when EITHER run has no usable trace; an `aligned:false` diff (with `alignmentNote`) when traces exist
 * but can't be step-aligned (different dataset, `totalSteps`, or no shared step indices). Never throws.
 */
export function diffDecisionTraces(
  baseline: TrainingRunSummary,
  tweak: TrainingRunSummary,
): DecisionTraceDiff | undefined {
  const traceA = validateDecisionTrace(baseline.artifacts?.decisionTrace)
  const traceB = validateDecisionTrace(tweak.artifacts?.decisionTrace)
  if (!traceA || !traceB) return undefined

  const sigA = datasetAlignmentSignature(baseline)
  const sigB = datasetAlignmentSignature(tweak)
  if (!sigA || !sigB || sigA !== sigB) {
    return notAlignedDiff('different dataset — not step-comparable', sigA || sigB)
  }
  const totalA = traceA.totalSteps ?? traceA.steps.length
  const totalB = traceB.totalSteps ?? traceB.steps.length
  if (totalA !== totalB) {
    return notAlignedDiff(`different totalSteps (${totalA} vs ${totalB})`, sigA)
  }

  const mapA = new Map(traceA.steps.map((s) => [s.step, s]))
  const mapB = new Map(traceB.steps.map((s) => [s.step, s]))
  const sharedSteps = [...mapA.keys()].filter((step) => mapB.has(step)).sort((x, y) => x - y)
  if (!sharedSteps.length) return notAlignedDiff('no shared steps', sigA)

  const steps: DecisionStepDelta[] = []
  const changedRewardDeltas: number[] = []
  const unchangedRewardDeltas: number[] = []
  const confidenceDeltas: number[] = []
  let changedSteps = 0
  for (const step of sharedSteps) {
    const a = mapA.get(step)!
    const b = mapB.get(step)!
    const changed = a.action !== b.action
    if (changed) changedSteps += 1
    const delta: DecisionStepDelta = {
      step,
      baselineAction: a.action,
      tweakAction: b.action,
      changed,
    }
    if (typeof a.reward === 'number' && typeof b.reward === 'number') {
      const rewardDelta = b.reward - a.reward
      delta.rewardDelta = rewardDelta
      ;(changed ? changedRewardDeltas : unchangedRewardDeltas).push(rewardDelta)
    }
    if (typeof a.confidence === 'number' && typeof b.confidence === 'number') {
      delta.confidenceDelta = b.confidence - a.confidence
      confidenceDeltas.push(delta.confidenceDelta)
    }
    steps.push(delta)
  }

  const actionCountDeltas: Record<string, number> = {}
  const labels = new Set([
    ...Object.keys(traceA.actionCounts ?? {}),
    ...Object.keys(traceB.actionCounts ?? {}),
  ])
  for (const label of labels) {
    const d = (traceB.actionCounts?.[label] ?? 0) - (traceA.actionCounts?.[label] ?? 0)
    if (d !== 0) actionCountDeltas[label] = d
  }
  const meanConfidenceShift = averageOf(confidenceDeltas)

  return {
    aligned: true,
    datasetSignature: sigA,
    alignedSteps: sharedSteps.length,
    changedSteps,
    divergenceRate: changedSteps / sharedSteps.length,
    steps,
    actionCountDeltas,
    ...(meanConfidenceShift !== undefined ? { meanConfidenceShift } : {}),
    objectiveDelta: tweak.objective - baseline.objective,
    quality: classifyDecisionQuality(changedRewardDeltas, unchangedRewardDeltas),
  }
}

// --- Models catalog ----------------------------------------------------------

const MODEL_CATEGORIES: ReadonlySet<string> = new Set(['rl', 'supervised', 'baseline', 'component'])

/** Tokens rendered verbatim (uppercased acronym or expanded alias) when humanizing a model_name. */
const MODEL_NAME_ACRONYMS: Record<string, string> = {
  dqn: 'DQN',
  ppo: 'PPO',
  a2c: 'A2C',
  a3c: 'A3C',
  trpo: 'TRPO',
  iqn: 'IQN',
  qrdqn: 'QR-DQN',
  tcn: 'TCN',
  lstm: 'LSTM',
  gru: 'GRU',
  gbm: 'GBM',
  mlp: 'MLP',
  rl: 'RL',
  sac: 'SAC',
  td3: 'TD3',
  ddpg: 'DDPG',
  ars: 'ARS',
  sbx: 'SBX',
  reppo: 'RecurrentPPO',
  hodl: 'Buy-and-Hold',
}

/** Canonical kebab slug for a model name/identifier — lowercase, alphanumeric runs joined by hyphens. */
export function modelSlug(name: string): string {
  return String(name || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

/** Heuristic {@link ModelCategory} for a model_name/identifier; RL is the default for an algorithm name. */
export function inferModelCategory(modelName: string): ModelCategory {
  const s = String(modelName || '').toLowerCase()
  if (/(^|[-_])(hodl|buy-?and-?hold|technical|time-strategy|random)([-_]|$)/.test(s))
    return 'baseline'
  if (/(supervised|logreg|gbm|xgboost|regression|classifier|forecast)/.test(s)) return 'supervised'
  if (/(extractor|policy|buffer|network|encoder|backbone|component|optimizer)/.test(s))
    return 'component'
  return 'rl'
}

/** Heuristic human name for a model_name lever value, e.g. `rainbow-dqn-custom` → `Rainbow DQN Custom`. */
export function humanizeModelName(modelName: string): string {
  const slug = modelSlug(modelName)
  if (!slug) return String(modelName || '')
  return slug
    .split('-')
    .map((tok) => MODEL_NAME_ACRONYMS[tok] ?? (tok ? tok[0].toUpperCase() + tok.slice(1) : ''))
    .filter(Boolean)
    .join(' ')
}

/**
 * From a manifest's `model_name` choice lever, the candidate models whose slug is not already in the
 * catalog and whose `model_name` is not already a binding of an existing model — each with a heuristic
 * name + category. `[]` when the manifest declares no `model_name` choice lever.
 */
export function discoverManifestModelCandidates(
  manifest: TrainerManifest,
  existingModels: Array<{ slug?: string; modelNames?: string[] }>,
): { modelName: string; slug: string; name: string; category: ModelCategory }[] {
  const lever = manifest.levers?.model_name as TrainerLeverSpec | undefined
  if (!lever || lever.type !== 'choice' || !Array.isArray(lever.choices)) return []
  const haveSlugs = new Set<string>()
  const haveBindings = new Set<string>()
  for (const m of existingModels) {
    if (m.slug) haveSlugs.add(m.slug)
    for (const n of m.modelNames ?? []) haveBindings.add(n)
  }
  const out: { modelName: string; slug: string; name: string; category: ModelCategory }[] = []
  const seen = new Set<string>()
  for (const choice of lever.choices) {
    if (typeof choice !== 'string' || !choice || haveBindings.has(choice)) continue
    const slug = modelSlug(choice)
    if (!slug || haveSlugs.has(slug) || seen.has(slug)) continue
    seen.add(slug)
    out.push({
      modelName: choice,
      slug,
      name: humanizeModelName(choice),
      category: inferModelCategory(choice),
    })
  }
  return out
}

/** Keep the first occurrence of each `slug` (entries without a slug pass through unchanged). Pure. */
export function dedupeModelsBySlug<T extends { slug?: string }>(models: T[]): T[] {
  const seen = new Set<string>()
  const out: T[] = []
  for (const m of models) {
    const slug = typeof m.slug === 'string' ? m.slug : ''
    if (slug && seen.has(slug)) continue
    if (slug) seen.add(slug)
    out.push(m)
  }
  return out
}

/**
 * Coerce the model's scan-enrichment response (an array of `{slug,name?,description?,category?,paperIds?}`)
 * into a map keyed by slug, keeping only known candidate slugs + known paper ids and dropping ill-typed
 * fields. An empty map for a non-array response.
 */
export function coerceScannedModels(
  raw: unknown,
  candidateSlugs: Set<string>,
  knownPaperIds: Set<string>,
): Map<
  string,
  { name?: string; description?: string; category?: ModelCategory; paperIds?: string[] }
> {
  const out = new Map<
    string,
    { name?: string; description?: string; category?: ModelCategory; paperIds?: string[] }
  >()
  if (!Array.isArray(raw)) return out
  for (const item of raw) {
    if (!item || typeof item !== 'object') continue
    const o = item as Record<string, unknown>
    const slug = typeof o.slug === 'string' ? o.slug : ''
    if (!candidateSlugs.has(slug)) continue
    const enrich: {
      name?: string
      description?: string
      category?: ModelCategory
      paperIds?: string[]
    } = {}
    if (typeof o.name === 'string' && o.name.trim()) enrich.name = o.name.trim()
    if (typeof o.description === 'string' && o.description.trim())
      enrich.description = o.description.trim()
    if (typeof o.category === 'string' && MODEL_CATEGORIES.has(o.category)) {
      enrich.category = o.category as ModelCategory
    }
    if (Array.isArray(o.paperIds)) {
      const ids = o.paperIds.filter(
        (x): x is string => typeof x === 'string' && knownPaperIds.has(x),
      )
      if (ids.length) enrich.paperIds = ids
    }
    out.set(slug, enrich)
  }
  return out
}

/** System prompt for "Scan Project": enrich the given candidate models into honest catalog entries. */
export function buildScanModelsSystemPrompt(manifest: TrainerManifest): string {
  return [
    `You are a model librarian for the "${manifest.name}" training project.`,
    `Objective: ${manifest.objective.name} (${manifest.objective.direction} is better).`,
    `You are given CANDIDATE models (each a model_name the project can train, with a heuristic name + category) and the project's PAPERS.`,
    `For each candidate, return an enriched catalog entry: a clearer human NAME, a one/two-sentence DESCRIPTION of what the model is and how it differs, the best CATEGORY (rl|supervised|baseline|component), and the ids of any given PAPERS that introduce or improve it (only ids from the list; [] if none).`,
    `Return ONLY a JSON array, one object per candidate you can enrich: [{"slug": "<the candidate slug, verbatim>", "name": string, "description": string, "category": "rl|supervised|baseline|component", "paperIds": [string]}]. No prose.`,
  ].join('\n')
}

export function buildScanModelsUserContent(input: {
  candidates: { slug: string; modelName: string; name: string; category: ModelCategory }[]
  papers: { id: string; title: string; claim?: string }[]
  leverDescription?: string
}): string {
  return JSON.stringify({
    candidates: input.candidates,
    papers: input.papers,
    ...(input.leverDescription ? { leverDescription: input.leverDescription } : {}),
  })
}

/**
 * Coerce the model's paper→models response into the existing-model ids it matched + the proposed (missing)
 * models. A proposed model needs a non-empty name; its slug defaults from the name and category defaults
 * to `rl`. Empty for a malformed response.
 */
export function coerceAnalyzedPaperModels(raw: unknown): {
  matchModelIds: string[]
  proposedModels: ProposedModel[]
} {
  const o =
    raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {}
  const matchModelIds = Array.isArray(o.matchModelIds)
    ? o.matchModelIds.filter((x): x is string => typeof x === 'string' && x.length > 0)
    : []
  const proposedModels: ProposedModel[] = []
  const rawProposed = Array.isArray(o.proposedModels) ? o.proposedModels : []
  for (const item of rawProposed) {
    if (!item || typeof item !== 'object') continue
    const p = item as Record<string, unknown>
    const name = typeof p.name === 'string' && p.name.trim() ? p.name.trim() : ''
    if (!name) continue
    const slug = (typeof p.slug === 'string' && modelSlug(p.slug)) || modelSlug(name)
    if (!slug) continue
    const description = typeof p.description === 'string' ? p.description.trim() : ''
    const category =
      typeof p.category === 'string' && MODEL_CATEGORIES.has(p.category)
        ? (p.category as ModelCategory)
        : 'rl'
    const proposal = typeof p.proposal === 'string' ? p.proposal.trim() : ''
    proposedModels.push({ name, slug, description, category, proposal })
  }
  return { matchModelIds, proposedModels }
}

/** Of `proposed`, the models with NO catalog entry — matched by slug, by a catalog model's name slug, or
 *  by a catalog model's `model_name` binding. The Models tab's "Add to catalog" candidates. */
export function detectMissingPaperModels(
  proposed: ProposedModel[],
  catalog: Array<{ slug?: string; name?: string; modelNames?: string[] }>,
): ProposedModel[] {
  const slugs = new Set<string>()
  const bindings = new Set<string>()
  for (const m of catalog) {
    if (m.slug) slugs.add(m.slug)
    if (m.name) slugs.add(modelSlug(m.name))
    for (const n of m.modelNames ?? []) bindings.add(n)
  }
  return proposed.filter((p) => !slugs.has(p.slug) && !bindings.has(p.slug))
}

/** System prompt for "Find models": match the paper to existing catalog models + propose missing ones. */
export function buildAnalyzePaperModelsSystemPrompt(manifest: TrainerManifest): string {
  return [
    `You are a model librarian for the "${manifest.name}" training project.`,
    `Objective: ${manifest.objective.name} (${manifest.objective.direction} is better).`,
    `You are given a PAPER (its fields, and its text when available) and the project's EXISTING catalog models.`,
    `Do TWO things: (1) MATCH — the ids of existing models this paper INTRODUCES or IMPROVES (a genuine "this paper is about that model" link, not a passing mention); (2) PROPOSE — any models the paper introduces or requires that are NOT in the catalog, each with what to add.`,
    `Return ONLY a single JSON object (no prose, no code fence): {"matchModelIds": [string], "proposedModels": [{"name": string, "slug": string, "description": string, "category": "rl|supervised|baseline|component", "proposal": string}]}. Use [] for either when there is nothing to add.`,
  ].join('\n')
}

export function buildAnalyzePaperModelsUserContent(input: {
  paper: Record<string, unknown>
  existingModels: {
    id: string
    name: string
    slug: string
    category: string
    modelNames?: string[]
  }[]
  text?: string
}): string {
  return JSON.stringify({
    paper: input.paper,
    existingModels: input.existingModels,
    ...(input.text ? { text: input.text } : {}),
  })
}
