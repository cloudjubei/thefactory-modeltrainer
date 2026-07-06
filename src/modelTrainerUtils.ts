import type {
  ComputeDevice,
  ConsolidationGroup,
  DecisionFeatureAttribution,
  DecisionQualitySignal,
  DecisionStep,
  DecisionStepDelta,
  DecisionTrace,
  DecisionTraceDiff,
  ExperimentSpec,
  HypothesisComparison,
  HypothesisComparisonKind,
  ModelCategory,
  ModelDeviceBenchmark,
  ModelFlavor,
  PaperCandidate,
  PlannedTrainingItem,
  ProposedImprovement,
  ProposedImprovementKind,
  ProposedModel,
  RunXaiDigest,
  StepAttributionSummary,
  TrainerDataFile,
  TrainerLeverSpec,
  TrainerManifest,
  TrainerMigrationRule,
  TrainingHypothesis,
  TrainingModel,
  TrainingPaperRecord,
  TrainingRunSummary,
} from './modelTrainerTypes.js'
import type { ClaimVerdict } from 'thefactory-tools/types'
import {
  DECISION_QUALITY_MIN_SCORED_STEPS,
  DECISION_QUALITY_REWARD_EPSILON,
  MAX_CAMPAIGN_ITEMS,
} from './modelTrainerConstants.js'

const LEVER_TYPES: ReadonlySet<string> = new Set(['number', 'choice', 'boolean'])

/**
 * Process-env keys that cap a run's math-thread count. The standard BLAS/OpenMP knobs are universal
 * (any numpy/torch project honours them); BS_NUM_THREADS is the BlackSwan trainer's own knob, harmless
 * for projects that ignore it. All are set to the same per-run thread count so a run can't grab every
 * core out from under its siblings.
 */
export const THREAD_ENV_VARS: readonly string[] = [
  'BS_NUM_THREADS',
  'OMP_NUM_THREADS',
  'MKL_NUM_THREADS',
  'OPENBLAS_NUM_THREADS',
  'NUMEXPR_NUM_THREADS',
  'VECLIB_MAXIMUM_THREADS',
]

/**
 * Resolve a campaign's run-pool size and per-run thread-cap env from the manifest + host.
 *
 * An explicit `concurrency` always wins. Otherwise, when the manifest declares `maxThreadsPerRun > 1`,
 * default the pool to `floor(cpus / threadsPerRun)` so N runs × threadsPerRun ≈ host cores — instead of
 * the safe-but-idle sequential default. Projects that don't declare `maxThreadsPerRun` keep the
 * sequential default and get NO env override (byte-compatible).
 *
 * The per-run thread cap is each run's FAIR SHARE of cores, `floor(cpus / concurrency)` — NOT a fixed
 * `threadsPerRun`. So a lone/final run (concurrency 1) uses the WHOLE host (measured ~1.24× faster for the
 * LSTM model at 6 vs 2 threads) instead of idling 8 cores, a packed auto sweep still lands on
 * `threadsPerRun` each, and an over-packed sweep (concurrency > auto) caps below it to avoid
 * oversubscription. The MLP models don't scale with threads but aren't harmed by the extra.
 *
 * A RAM-aware ceiling caps the pool at `floor(availableMemoryBytes / maxMemoryBytesPerRun)` so a packed
 * sweep can never launch more concurrent runs than host memory holds (the OOM guard). It is a HARD
 * ceiling — it caps an explicit `concurrency` too — but opt-in: a manifest that declares no per-run
 * memory estimate keeps the CPU-only derivation byte-for-byte. `memoryCapped` reports when it bit, so
 * the caller can log the reduction rather than silently shrinking the pool.
 */
export function resolveCampaignParallelism(opts: {
  concurrency?: number
  maxThreadsPerRun?: number
  availableParallelism: number
  maxMemoryBytesPerRun?: number
  availableMemoryBytes?: number
}): { concurrency: number; runEnv?: Record<string, string>; memoryCapped?: boolean } {
  const threadsPerRun = Math.max(1, Math.floor(opts.maxThreadsPerRun ?? 1))
  const cpus = Math.max(1, Math.floor(opts.availableParallelism))
  const auto = threadsPerRun > 1 ? Math.max(1, Math.floor(cpus / threadsPerRun)) : 1
  const requested =
    opts.concurrency !== undefined ? Math.max(1, Math.floor(opts.concurrency)) : auto
  const memoryCap =
    opts.maxMemoryBytesPerRun &&
    opts.maxMemoryBytesPerRun > 0 &&
    opts.availableMemoryBytes &&
    opts.availableMemoryBytes > 0
      ? Math.max(1, Math.floor(opts.availableMemoryBytes / opts.maxMemoryBytesPerRun))
      : Number.POSITIVE_INFINITY
  const concurrency = Math.min(requested, memoryCap)
  const memoryCapped = concurrency < requested ? true : undefined
  // Each run's fair share of cores — EXCEPT when the RAM cap lowered the pool: expanding threads to fill
  // the freed cores would raise each run's RSS above the estimate the cap was sized against, fighting the
  // cap. So a memory-capped pool holds per-run threads at the declared appetite instead of expanding.
  const perRunThreads = memoryCapped
    ? Math.max(1, Math.min(Math.floor(cpus / concurrency), threadsPerRun))
    : Math.max(1, Math.floor(cpus / concurrency))
  const runEnv =
    threadsPerRun > 1
      ? Object.fromEntries(THREAD_ENV_VARS.map((v) => [v, String(perRunThreads)]))
      : undefined
  return { concurrency, runEnv, memoryCapped }
}

/**
 * The device a benchmarked model wants for ONE run — or undefined to leave the run's device untouched.
 *
 * Applies a model's `preferredDevice` (set by the device benchmark) to a config that doesn't already name
 * a device, matched by `config.model_name` against the model's flavor names. CRITICAL: an `mps` preference
 * is NEVER applied to a PARALLEL sweep (concurrency > 1) — MPS is one shared GPU the runs would contend
 * for, so a packed sweep stays on CPU (which parallelises). An explicit `config.device` always wins.
 */
export function resolveModelDeviceForConfig(opts: {
  config: Record<string, unknown>
  models: Array<{
    preferredDevice?: 'cpu' | 'mps'
    flavors?: { modelName?: string }[]
    modelNames?: string[]
  }>
  concurrency: number
}): 'cpu' | 'mps' | undefined {
  if (opts.config.device !== undefined) return undefined
  const modelName = opts.config.model_name
  if (typeof modelName !== 'string') return undefined
  const match = opts.models.find(
    (m) => m.preferredDevice && modelBindingNames(m).includes(modelName),
  )
  const pref = match?.preferredDevice
  if (!pref) return undefined
  if (pref === 'mps' && opts.concurrency > 1) return undefined
  return pref
}

/**
 * Coerce a `benchmarkDevice` command's `{summaryOut}` JSON (`{ deviceBenchmark: {...} }`) into a typed
 * {@link ModelDeviceBenchmark}, defaulting safely to CPU on anything malformed so a flaky benchmark can
 * never set a bogus device. `bestDevice` is `mps` only when the summary explicitly says so.
 */
export function parseDeviceBenchmark(
  summary: unknown,
  benchmarkedAt: string,
): ModelDeviceBenchmark {
  const db = (summary as { deviceBenchmark?: Record<string, unknown> } | undefined)?.deviceBenchmark
  const bestDevice: ComputeDevice =
    db?.bestDevice === 'mps' || db?.bestDevice === 'cuda' ? db.bestDevice : 'cpu'
  const speedup =
    typeof db?.speedup === 'number' && Number.isFinite(db.speedup) && db.speedup >= 1
      ? db.speedup
      : 1
  const numberMap = (raw: unknown): Record<string, number> => {
    const out: Record<string, number> = {}
    for (const [d, v] of Object.entries((raw ?? {}) as Record<string, unknown>)) {
      if (typeof v === 'number' && Number.isFinite(v) && v > 0) out[d] = v
    }
    return out
  }
  const usPerStep = numberMap(db?.usPerStep)
  const seconds = numberMap(db?.seconds)
  const errors: Record<string, string> = {}
  for (const [d, v] of Object.entries((db?.errors ?? {}) as Record<string, unknown>)) {
    if (typeof v === 'string' && v) errors[d] = v
  }
  const rawDevices = Array.isArray(db?.availableDevices) ? (db?.availableDevices as unknown[]) : []
  const availableDevices = rawDevices.filter((d): d is string => typeof d === 'string')
  const result: ModelDeviceBenchmark = {
    bestDevice,
    speedup,
    usPerStep,
    availableDevices: availableDevices.length
      ? availableDevices
      : Object.keys(usPerStep).length
        ? Object.keys(usPerStep)
        : ['cpu'],
    benchmarkedAt,
  }
  if (Object.keys(seconds).length) result.seconds = seconds
  if (typeof db?.budget === 'number' && Number.isFinite(db.budget) && db.budget > 0)
    result.budget = db.budget
  if (Object.keys(errors).length) result.errors = errors
  return result
}

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
  if (m.benchmarkDevice !== undefined) {
    if (typeof m.benchmarkDevice !== 'string' || !m.benchmarkDevice.includes('{summaryOut}')) {
      throw new Error('trainer manifest benchmarkDevice template must contain {summaryOut}')
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
    const hasClause = !!(rule.match || rule.matchNot)
    const hasUnset = !!(rule.unset && rule.unset.length)
    if (!hasClause && !hasUnset) return false
    const eq = Object.entries(rule.match ?? {}).every(
      ([k, v]) => k in config && String(config[k]) === String(v),
    )
    const neq = Object.entries(rule.matchNot ?? {}).every(
      ([k, v]) => k in config && String(config[k]) !== String(v),
    )
    const clauseOk = hasClause ? eq && neq : true
    // An unset-ONLY rule fires only while a target key remains, so the sweep converges (idempotent).
    const unsetOk = hasUnset && !hasClause ? rule.unset!.some((k) => k in config) : true
    return clauseOk && unsetOk
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
  for (const k of rule.unset ?? []) delete out[k]
  return out
}

/**
 * Roll an {@link ExperimentSpec} forward through the same migration rules before it is planned, so a
 * run dispatched from an OLD queued/pending spec (e.g. `reward_model: "combo_all"`) executes under the
 * migrated shape rather than the retired one. `spec.fixed` and each `spec.configs` entry are migrated (the
 * pinned/explicit configs runs carry); a `sweep` over a migrated lever is left untouched (its values are a
 * list, not a config). Returns the SAME spec object when no rule matches, so callers can treat it as a
 * cheap pass-through.
 */
export function migrateExperimentSpec(
  spec: ExperimentSpec,
  migrations: TrainerMigrationRule[] | undefined,
): ExperimentSpec {
  if (!migrations || migrations.length === 0 || !spec) return spec
  const next: ExperimentSpec = { ...spec }
  let changed = false
  if (spec.fixed) {
    const migrated = applyMigrationRules(spec.fixed, migrations)
    if (migrated) {
      next.fixed = migrated
      changed = true
    }
  }
  if (spec.configs && spec.configs.length > 0) {
    const migratedConfigs = spec.configs.map((entry) => {
      const migrated = applyMigrationRules(entry.config, migrations)
      return migrated ? { ...entry, config: migrated } : entry
    })
    if (migratedConfigs.some((entry, i) => entry !== spec.configs![i])) {
      next.configs = migratedConfigs
      changed = true
    }
  }
  return changed ? next : spec
}

/**
 * Predicted wall-clock SECONDS for the campaign's REMAINING runs, from the ACTUAL durations of runs that
 * have completed THIS session — not elapsed/done (which is diluted toward zero by instantly-skipped, already
 * completed runs). Runs over the same data+model take a similar time, so the average per-run duration × the
 * number of remaining concurrency "waves" (ceil(remaining / concurrency)) is a stable total estimate.
 * Returns undefined until at least one real run has finished (an honest "no estimate yet").
 */
export function estimateRemainingCampaignSeconds(input: {
  durationsMs: number[]
  remaining: number
  concurrency: number
}): number | undefined {
  const durs = (input.durationsMs || []).filter(
    (d) => typeof d === 'number' && isFinite(d) && d > 0,
  )
  if (!durs.length || input.remaining <= 0) return undefined
  const avgMs = durs.reduce((a, b) => a + b, 0) / durs.length
  const conc = Math.max(1, Math.min(Math.floor(input.concurrency) || 1, input.remaining))
  const waves = Math.ceil(input.remaining / conc)
  return (avgMs / 1000) * waves
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

// ── Hypothesis consolidation ──────────────────────────────────────────────────────────────────────────
// Two hypotheses are the "same" when they share the same MAIN PARAMETERS and differ only in how WIDE their
// sweep is — so an LLM re-suggesting an existing setup (or a wider sweep of it) folds into the original
// instead of spawning a near-duplicate. The id IS the spec hash, so "extend the sweep" means a new id: the
// group merges to one wider hypothesis and the others are ABSORBED (links + per-paper weights repointed,
// records deleted). Deterministic — no LLM, unlike model consolidation. Used by both the manual
// "Consolidate" action and the auto-pass after suggest.

type HypothesisLike = Pick<
  TrainingHypothesis,
  | 'id'
  | 'spec'
  | 'title'
  | 'rationale'
  | 'status'
  | 'verdictSource'
  | 'verdictNote'
  | 'source'
  | 'comparison'
  | 'claim'
  | 'claimedMetrics'
  | 'proposedBy'
  | 'paperIds'
  | 'createdAt'
  | 'dismissed'
  | 'campaign'
>

/** Union a list of lever VALUES (which may be objects/arrays): dedupe by canonical string, sort stably. */
function unionLeverValues(values: unknown[]): unknown[] {
  const seen = new Map<string, unknown>()
  for (const v of values) {
    const k = canonicalConfigString(v)
    if (!seen.has(k)) seen.set(k, v)
  }
  return [...seen.entries()].sort(([a], [b]) => (a < b ? -1 : 1)).map(([, v]) => v)
}

/**
 * The consolidation KEY: the spec's identity IGNORING sweep BREADTH — its fixed lever values, which levers
 * are swept (keys only, so a wider sweep collides), the compare lever, and the env/dataset bundles (order
 * preserved — bundle order is semantic). Seeds, sweep/compare VALUES, configs and maxItems are excluded.
 */
export function hypothesisConsolidationKey(spec: ExperimentSpec | undefined): string {
  const s = spec || {}
  return canonicalConfigString({
    fixed: s.fixed ?? {},
    sweepLevers: Object.keys(s.sweep ?? {}).sort(),
    compareLever: s.compare?.lever ?? null,
    environments: s.environments ?? [],
    datasets: s.datasets ?? [],
  })
}

/**
 * Merge a group of specs that share a consolidation key into ONE wider spec: fixed/environments/datasets
 * kept verbatim (identical by the key), sweep values / compare values / seeds UNIONED (deduped + sorted, so
 * the merged spec hashes deterministically regardless of member order — idempotent). configs/maxItems are
 * dropped (not part of a hypothesis's identity).
 */
export function mergeHypothesisSpecs(specs: ExperimentSpec[]): ExperimentSpec {
  const first = specs[0] || {}
  const out: ExperimentSpec = {}
  if (first.fixed && Object.keys(first.fixed).length) out.fixed = first.fixed
  const sweepLevers = new Set<string>()
  for (const s of specs) for (const k of Object.keys(s.sweep ?? {})) sweepLevers.add(k)
  if (sweepLevers.size) {
    const sweep: Record<string, unknown[]> = {}
    for (const lever of [...sweepLevers].sort())
      sweep[lever] = unionLeverValues(specs.flatMap((s) => s.sweep?.[lever] ?? []))
    out.sweep = sweep
  }
  const compareLever = first.compare?.lever
  if (compareLever)
    out.compare = {
      lever: compareLever,
      values: unionLeverValues(specs.flatMap((s) => s.compare?.values ?? [])),
    }
  if (first.environments && first.environments.length) out.environments = first.environments
  if (first.datasets && first.datasets.length) out.datasets = first.datasets
  const seeds = [...new Set(specs.flatMap((s) => s.seeds ?? []))].sort((a, b) => a - b)
  if (seeds.length) out.seeds = seeds
  return out
}

const HYP_SOURCE_PRIORITY: Record<string, number> = {
  human: 0,
  paper: 1,
  llm: 2,
  'migrated-model': 3,
}

function rankHypothesesForCanonical<T extends HypothesisLike>(members: T[]): T[] {
  return [...members].sort((a, b) => {
    const pa = HYP_SOURCE_PRIORITY[a.source] ?? 9
    const pb = HYP_SOURCE_PRIORITY[b.source] ?? 9
    if (pa !== pb) return pa - pb
    const ta = a.createdAt || ''
    const tb = b.createdAt || ''
    if (ta !== tb) return ta < tb ? -1 : 1
    return a.id < b.id ? -1 : 1
  })
}

/**
 * Which member's metadata (title/rationale/verdict pin/comparison…) the merged hypothesis inherits — the id
 * is always the merged-spec hash, so this governs metadata ONLY. A lone manual-verdict member wins; else
 * rank by source (human > paper > llm > migrated-model) then earliest createdAt. CONFLICTING manual
 * verdicts (proven vs disproved) return `conflict` so the caller skips the group rather than silently
 * pinning one (a manual verdict is returned verbatim by the viewer — a wrong inherited pin is invisible).
 */
export function pickCanonicalHypothesis<T extends HypothesisLike>(
  members: T[],
): { canonical: T | null; conflict: boolean } {
  const manual = members.filter((m) => m.verdictSource === 'manual')
  if (manual.length === 1) return { canonical: manual[0], conflict: false }
  if (manual.length > 1) {
    if (new Set(manual.map((m) => m.status)).size > 1) return { canonical: null, conflict: true }
    return { canonical: rankHypothesesForCanonical(manual)[0], conflict: false }
  }
  return { canonical: rankHypothesesForCanonical(members)[0] ?? null, conflict: false }
}

/**
 * Bucket hypotheses by consolidation key into the groups worth merging: skips dismissed members, singleton
 * buckets, and buckets whose members already share a single id (already consolidated — so a re-run is a
 * no-op and the auto-pass cannot loop).
 */
export function groupHypothesesForConsolidation<T extends HypothesisLike>(
  hypotheses: T[],
  _hashFn?: (config: Record<string, unknown>) => string,
): { key: string; members: T[] }[] {
  const buckets = new Map<string, T[]>()
  for (const h of hypotheses) {
    if (h.dismissed) continue
    const key = hypothesisConsolidationKey(h.spec)
    const bucket = buckets.get(key)
    if (bucket) bucket.push(h)
    else buckets.set(key, [h])
  }
  const groups: { key: string; members: T[] }[] = []
  for (const [key, members] of buckets) {
    if (members.length < 2) continue
    if (new Set(members.map((m) => m.id)).size < 2) continue
    // Defer a group while any member has an in-flight campaign — absorbing (deleting) it would orphan the
    // campaign's pending results write. It consolidates on the next pass once the campaign settles.
    if (
      members.some(
        (m) => m.campaign && (m.campaign.status === 'running' || m.campaign.status === 'queued'),
      )
    )
      continue
    groups.push({ key, members })
  }
  return groups
}

export interface HypothesisConsolidationPlan {
  unionRecord: TrainingHypothesis
  changedPapers: TrainingPaperRecord[]
  changedModels: TrainingModel[]
  deletedIds: string[]
  mergedFrom: string[]
}

function unionStrings(...lists: (string[] | undefined)[]): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const list of lists)
    for (const s of list || [])
      if (s && !seen.has(s)) {
        seen.add(s)
        out.push(s)
      }
  return out
}

/**
 * Plan one group's consolidation: the union hypothesis (at the merged-spec id, inheriting the canonical's
 * metadata, with NO evidence/transitions so the wider verdict recomputes cleanly), plus every record that
 * must be repointed off the absorbed ids — papers (`hypothesisIds` + the per-paper `hypothesisWeights` map,
 * max on collision so the user's stated importance is PRESERVED onto the survivor) and models/flavors
 * (`hypothesisIds`). Returns `{ skipped: 'conflict' }` for conflicting manual verdicts, or `null` when the
 * group is already consolidated. The caller performs the writes (papers/models first, delete absorbed last).
 */
export function planHypothesisConsolidation(
  group: { members: HypothesisLike[] },
  papers: TrainingPaperRecord[],
  models: TrainingModel[],
  nowIso: string,
  hashFn: (config: Record<string, unknown>) => string,
): HypothesisConsolidationPlan | { skipped: 'conflict'; members: string[] } | null {
  const members = group.members
  const { canonical, conflict } = pickCanonicalHypothesis(members)
  if (conflict || !canonical) return { skipped: 'conflict', members: members.map((m) => m.id) }
  const mergedSpec = mergeHypothesisSpecs(members.map((m) => m.spec || {}))
  const newId = hashFn(mergedSpec as unknown as Record<string, unknown>)
  const absorbed = members.filter((m) => m.id !== newId)
  if (!absorbed.length) return null // already consolidated to one id
  const absorbedIds = new Set(absorbed.map((m) => m.id))
  const existingUnion = members.find((m) => m.id === newId)
  const base = existingUnion || canonical
  const earliest = members
    .map((m) => m.createdAt)
    .filter(Boolean)
    .sort()[0]
  const keepManual = canonical.verdictSource === 'manual'
  const unionRecord: TrainingHypothesis = {
    id: newId,
    title: (base.title as string) || 'Hypothesis',
    rationale: (base.rationale as string) || '',
    spec: mergedSpec,
    status: keepManual ? canonical.status : 'untested',
    verdictSource: keepManual ? 'manual' : 'auto',
    source: (base.source as TrainingHypothesis['source']) || 'llm',
    paperIds: unionStrings(...members.map((m) => m.paperIds)),
    createdAt: earliest || nowIso,
    updatedAt: nowIso,
  }
  if (keepManual && canonical.verdictNote) unionRecord.verdictNote = canonical.verdictNote
  if (canonical.comparison) unionRecord.comparison = canonical.comparison
  // Carry the paper-claim label so a consolidated union still groups under its claim in the paper view.
  if (canonical.claim) unionRecord.claim = canonical.claim
  if (canonical.claimedMetrics) unionRecord.claimedMetrics = canonical.claimedMetrics
  if (base.proposedBy) unionRecord.proposedBy = base.proposedBy as string

  const changedPapers: TrainingPaperRecord[] = []
  for (const p of papers) {
    const ids = p.hypothesisIds || []
    if (!ids.some((id) => absorbedIds.has(id))) continue
    const nextIds = unionStrings(ids.map((id) => (absorbedIds.has(id) ? newId : id)))
    const weights = { ...(p.hypothesisWeights || {}) }
    for (const aid of absorbedIds) {
      if (!(aid in weights)) continue
      const w = weights[aid]
      weights[newId] = newId in weights ? Math.max(weights[newId], w) : w
      delete weights[aid]
    }
    const next: TrainingPaperRecord = { ...p, hypothesisIds: nextIds, updatedAt: nowIso }
    if (Object.keys(weights).length) next.hypothesisWeights = weights
    else delete next.hypothesisWeights
    changedPapers.push(next)
  }

  const repoint = (ids: string[] | undefined): { ids: string[]; touched: boolean } => {
    const list = ids || []
    if (!list.some((id) => absorbedIds.has(id))) return { ids: list, touched: false }
    return {
      ids: unionStrings(list.map((id) => (absorbedIds.has(id) ? newId : id))),
      touched: true,
    }
  }
  const changedModels: TrainingModel[] = []
  for (const m of models) {
    const top = repoint(m.hypothesisIds)
    let touched = top.touched
    const flavors = (m.flavors || []).map((f: ModelFlavor) => {
      const r = repoint(f.hypothesisIds)
      if (r.touched) touched = true
      return r.touched ? { ...f, hypothesisIds: r.ids } : f
    })
    if (touched) changedModels.push({ ...m, hypothesisIds: top.ids, flavors, updatedAt: nowIso })
  }

  return {
    unionRecord,
    changedPapers,
    changedModels,
    deletedIds: [...absorbedIds],
    mergedFrom: members.map((m) => m.id),
  }
}

/**
 * The conditional-applicability map for a manifest: lever name → its `appliesWhen` conditions, for the
 * levers that declare one. Feeds {@link normalizeConditionalLevers} so a conditional lever (e.g.
 * `forward_horizon`, which only applies to the supervised models) can be pinned to the `n/a` sentinel
 * wherever it doesn't apply — keeping stored configs, the interaction grid, and fANOVA honest.
 */
export function appliesWhenMap(
  manifest: TrainerManifest,
): Record<string, Record<string, unknown[]>> {
  const out: Record<string, Record<string, unknown[]>> = {}
  for (const [name, spec] of Object.entries(manifest.levers ?? {})) {
    if (spec.appliesWhen) out[name] = spec.appliesWhen
  }
  return out
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
  const compare = spec.compare
  if (compare && compare.lever !== undefined) {
    if (!leverKeys.includes(compare.lever)) {
      throw new Error(`compare lever "${compare.lever}" names no manifest lever`)
    }
    if (!Array.isArray(compare.values) || compare.values.length === 0) {
      throw new Error(`compare "${compare.lever}" must list values to compare`)
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
  const explicitConfigs = spec.configs ?? []
  for (const entry of explicitConfigs) {
    for (const key of Object.keys(entry.config)) {
      if (!leverKeys.includes(key)) {
        throw new Error(`config value "${key}" names no manifest lever`)
      }
    }
  }

  const base: Record<string, unknown> = {}
  for (const [key, lever] of Object.entries(manifest.levers)) {
    if (lever.default !== undefined) base[key] = lever.default
  }
  Object.assign(base, spec.fixed ?? {})

  // Explicit configs run VERBATIM (merged onto defaults): when present they DEFINE the matrix, so the
  // sweep/bundle/seed expansion below starts from nothing and only they remain.
  let configs: Record<string, unknown>[] = explicitConfigs.length ? [] : [base]
  for (const [key, values] of Object.entries(sweep)) {
    configs = configs.flatMap((config) => values.map((value) => ({ ...config, [key]: value })))
  }
  // A `compare` runs every value of its lever (like a sweep) — the verdict then PARTITIONS the resulting
  // runs by that value (via contextCells) to judge the values against each other.
  if (
    compare &&
    compare.lever !== undefined &&
    Array.isArray(compare.values) &&
    compare.values.length
  ) {
    configs = configs.flatMap((config) =>
      compare.values.map((value) => ({ ...config, [compare.lever]: value })),
    )
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

  const unitsLever = manifest.eta?.unitsLever
  const toItem = (key: string, config: Record<string, unknown>): PlannedTrainingItem => {
    const units =
      unitsLever && typeof config[unitsLever] === 'number'
        ? (config[unitsLever] as number)
        : undefined
    return { key, config, ...(units !== undefined ? { units } : {}) }
  }
  // Swept items hash their config; explicit items honour a preassigned `key` (re-run identity) or hash.
  const items = configs.map((config) => toItem(hashConfig(config), config))
  for (const entry of explicitConfigs) {
    const config = { ...base, ...entry.config }
    items.push(toItem(entry.key ?? hashConfig(config), config))
  }

  const cap = spec.maxItems ?? MAX_CAMPAIGN_ITEMS
  if (items.length > cap) {
    throw new Error(`campaign plans ${items.length} items, exceeding the cap of ${cap}`)
  }
  return items
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

// Validate an LLM-proposed comparison criterion; drop it unless `kind` is one of the known kinds.
function coerceComparison(raw: unknown): HypothesisComparison | undefined {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return undefined
  const o = raw as Record<string, unknown>
  const kinds: HypothesisComparisonKind[] = ['beats-baseline', 'invariant', 'differs']
  if (typeof o.kind !== 'string' || !kinds.includes(o.kind as HypothesisComparisonKind))
    return undefined
  const c: HypothesisComparison = { kind: o.kind as HypothesisComparisonKind }
  if (
    typeof o.baselineIndex === 'number' &&
    Number.isFinite(o.baselineIndex) &&
    o.baselineIndex >= 0
  ) {
    c.baselineIndex = Math.trunc(o.baselineIndex)
  }
  if (typeof o.tolerance === 'number' && Number.isFinite(o.tolerance)) c.tolerance = o.tolerance
  return c
}
// Resolve the lever ROLES used by the suggestion prompt + the precision guards: the config-DEFINING
// (model-scope or unscoped) levers, the context-scoped sets, and the model-IDENTITY lever (`model_name`
// when declared, else none — keyed on model_name specifically so generic manifests without it never get
// false-dropped by the under-pinned guard).
export function resolveModelLevers(manifest: TrainerManifest): {
  identityLever: string | undefined
  modelLevers: Set<string>
  envLevers: Set<string>
  datasetLevers: Set<string>
} {
  const scopedLevers = (scope: string) =>
    new Set(
      Object.entries(manifest.levers)
        .filter(([, s]) => (s as TrainerLeverSpec).scope === scope)
        .map(([k]) => k),
    )
  const modelLevers = new Set(
    Object.entries(manifest.levers)
      .filter(([, s]) => {
        const sc = (s as TrainerLeverSpec).scope
        return sc === 'model' || sc === undefined
      })
      .map(([k]) => k),
  )
  return {
    identityLever: manifest.levers.model_name ? 'model_name' : undefined,
    modelLevers,
    envLevers: scopedLevers('environment'),
    datasetLevers: scopedLevers('dataset'),
  }
}

// A title/rationale that asserts a COMPARISON between configs ("A outperforms B", "X is necessary",
// "recurrent beats non-recurrent") — used to reject the unjudgeable comparative-as-pooled-model-sweep form.
// "beats hold"/"beats buy" is NOT comparative (that's the verdict itself).
export function looksComparative(title: string, rationale: string): boolean {
  // No trailing \b — so conjugations match (outperform/outperforms/outperformed). The beats? lookahead
  // excludes "beats hold/buy" (that's the verdict itself, not a config-vs-config comparison).
  return /\b(outperform|out-perform|better than|worse than|superior|necessity of|necessary|versus|vs\.?|compared? (?:to|with)|more effective|beats?\b(?!\s+(?:hold|buy)))/i.test(
    `${title} ${rationale}`,
  )
}

export function coerceHypothesisItems(
  raw: unknown[],
  manifest: TrainerManifest,
): {
  title: string
  rationale: string
  spec: ExperimentSpec
  comparison?: HypothesisComparison
  claim?: string
}[] {
  if (!Array.isArray(raw)) return []
  const leverKeys = new Set(Object.keys(manifest.levers))
  const { identityLever, modelLevers, envLevers, datasetLevers } = resolveModelLevers(manifest)
  // A context bundle array is valid iff every bundle is a non-empty object keyed ONLY by levers of the
  // matching scope (environments hold environment levers; datasets hold dataset levers).
  const coerceBundles = (
    value: unknown,
    allowed: Set<string>,
  ): Record<string, unknown>[] | null => {
    if (!Array.isArray(value)) return null
    const out: Record<string, unknown>[] = []
    for (const b of value) {
      if (!b || typeof b !== 'object' || Array.isArray(b)) return null
      const keys = Object.keys(b as Record<string, unknown>)
      if (!keys.length || keys.some((k) => !allowed.has(k))) return null
      out.push(b as Record<string, unknown>)
    }
    return out.length ? out : null
  }
  const items: {
    title: string
    rationale: string
    spec: ExperimentSpec
    comparison?: HypothesisComparison
    claim?: string
  }[] = []
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
    // Context-spanning bundles: environments/datasets must carry ONLY levers of the matching scope. An
    // ill-scoped bundle invalidates the whole item (it's the wrong dimension for that lever).
    if (valid && rawSpec.environments !== undefined) {
      const envs = coerceBundles(rawSpec.environments, envLevers)
      if (!envs) valid = false
      else spec.environments = envs
    }
    if (valid && rawSpec.datasets !== undefined) {
      const dss = coerceBundles(rawSpec.datasets, datasetLevers)
      if (!dss) valid = false
      else spec.datasets = dss
    }
    // `compare` pits a single lever's values against each other (the judgeable "A vs B" form) —
    // require {lever: a declared lever, values: an array of >= 2}.
    const compareRaw = rawSpec.compare as Record<string, unknown> | undefined
    if (valid && compareRaw !== undefined) {
      if (
        !compareRaw ||
        typeof compareRaw !== 'object' ||
        Array.isArray(compareRaw) ||
        typeof compareRaw.lever !== 'string' ||
        !leverKeys.has(compareRaw.lever) ||
        !Array.isArray(compareRaw.values) ||
        (compareRaw.values as unknown[]).length < 2
      ) {
        valid = false
      } else {
        spec.compare = { lever: compareRaw.lever, values: compareRaw.values as unknown[] }
      }
    }
    if (
      !valid ||
      (!spec.sweep && !spec.fixed && !spec.environments && !spec.datasets && !spec.compare)
    )
      continue
    // PRECISION GUARDS (deterministic backstop — the prompt rules are advisory).
    // GUARD A: a COMPARATIVE claim must NOT be a pooled multi-value sweep over a model lever — that pools
    // the arms and never compares them (the canonical failure). Drop it; the model must use `compare`.
    const hasMultiValueModelSweep =
      !!spec.sweep &&
      Object.entries(spec.sweep).some(
        ([k, vals]) => modelLevers.has(k) && Array.isArray(vals) && vals.length > 1,
      )
    if (looksComparative(obj.title, obj.rationale) && hasMultiValueModelSweep) continue
    // GUARD B: a SINGLE-CONTEXT spec (no environments/datasets/compare) is judged as a pooled beats-hold
    // over its matched runs — so it MUST pin the model-identity lever, else it matches the whole backlog.
    const isSingleContext = !spec.environments && !spec.datasets && !spec.compare
    if (isSingleContext && identityLever && !(spec.fixed && identityLever in spec.fixed)) continue
    if (Array.isArray(rawSpec.seeds)) {
      spec.seeds = rawSpec.seeds
        .filter((s): s is number => typeof s === 'number' && Number.isFinite(s))
        .map((s) => Math.trunc(s))
    }
    const comparison = coerceComparison(obj.comparison)
    const claim = typeof obj.claim === 'string' && obj.claim.trim() ? obj.claim.trim() : undefined
    items.push({
      title: obj.title,
      rationale: obj.rationale,
      spec,
      ...(comparison ? { comparison } : {}),
      ...(claim ? { claim } : {}),
    })
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
  'Each hypothesis MUST be a FALSIFIABLE claim a run can PROVE OR DISPROVE, and it MUST match how the verdict ' +
  'is computed. A SINGLE-CONTEXT hypothesis (just fixed/sweep) is judged ONLY as "does the BEST matching run ' +
  'beat buy-and-hold OOS net of fees" — NOT an A-vs-B comparison — so frame it as "THIS one fully-pinned ' +
  'configuration beats buy-and-hold" and pin in spec.fixed EVERY defining lever (the model-identity lever plus ' +
  'the other model and data/market levers) so it matches a small coherent run family, not the backlog. For a ' +
  'COMPARATIVE claim ("A outperforms B", "X is necessary"), use a "compare" block (the contrasted lever + its ' +
  'values, with the shared config pinned in fixed) plus a "comparison" kind — NEVER a pooled sweep over the ' +
  'comparison axis, which cannot test which value wins. Reserve cross-CONTEXT comparison (environment/dataset ' +
  'bundles + comparison) for claims about the effect of the context. Do NOT propose pure DATA-GATHERING ' +
  '(more seeds/runs of a known config to tighten an interval, reduce variance, or reproduce) — that is the ' +
  'min-runs gate + the xAI tab, not a hypothesis. Every spec must introduce a configuration whose beats-hold ' +
  'outcome (or, for a comparison, whose cross-value/context comparison) is what is being tested.'

// How to express a CROSS-CONTEXT hypothesis. Some levers are held-fixed CONTEXT, not model knobs —
// managed as bundles, never swept. A claim about the EFFECT OF THE CONTEXT uses these instead of sweep/fixed.
const CONTEXT_SPANNING_RULE =
  'Some levers are CONTEXT, not model knobs — those marked "scope":"environment" (the action space / market ' +
  'mechanics) or "scope":"dataset" (which data). To test a claim that compares behaviour ACROSS contexts ' +
  '(e.g. long-only vs long+short, or one asset vs another), do NOT put a context lever in sweep/fixed; ' +
  'instead give "environments" and/or "datasets" as arrays of bundles (each bundle a set of context-lever ' +
  'values applied together) plus a "comparison": {"kind":"beats-baseline"|"invariant"|"differs", ' +
  '"baselineIndex"?: number, "tolerance"?: number}. beats-baseline = a non-baseline context beats the ' +
  'baseline; invariant = the objective holds steady across contexts (robustness); differs = it changes ' +
  '(sensitivity). Use this whenever the hypothesis is really about the effect of the context itself.'

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
  if (input.attribution?.driverCounts && input.attribution.driverCounts.length) {
    const drivers = input.attribution.driverCounts.map(([k, c]) => `${k}=${c}`).join(', ')
    lines.push(`Per-step drivers (dominant input group by decision count): ${drivers}.`)
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
    CONTEXT_SPANNING_RULE,
    notes ? `Extra guidance: ${notes}` : '',
    `Return ONLY a single JSON object (no prose, no code fence): {"title": string (required), "authors"?: string, ` +
      `"year"?: number, "claim": string (the source's headline claim in its own terms, required), "approach"?: string, ` +
      `"claimedMetrics"?: {"<name>": number}, "assumptions"?: {"fees"?: boolean, "netOfCosts"?: boolean, ` +
      `"frictionless"?: boolean, "multiAsset"?: boolean, "retrainCadence"?: string, "notes"?: string}, ` +
      `"verdictNote"?: string (skeptical — does it likely survive real costs + walk-forward OOS?), "tags"?: [string], ` +
      `"hypotheses"?: [{"title": string, "rationale": string, "comparison"?: {"kind": "beats-baseline"|"invariant"|"differs", "baselineIndex"?: number, "tolerance"?: number}, ` +
      `"spec": {"fixed"?: {"<lever>": value}, "sweep"?: {"<lever>": [values]}, ` +
      `"environments"?: [{"<environment-lever>": value}], "datasets"?: [{"<dataset-lever>": value}], "seeds"?: [number]}}] ` +
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
  const idName = resolveModelLevers(manifest).identityLever || 'the model-identity lever'
  return [
    `You are a research librarian for the "${manifest.name}" training project.`,
    `Objective: ${manifest.objective.name} (${manifest.objective.direction} is better).`,
    `You are given a PAPER (its fields, and its text when available), the project's EXISTING hypotheses, and a leverGuide naming each lever's ROLE.`,
    `Levers carry a "scope": MODEL levers (scope "model" or unscoped) DEFINE a run (e.g. ${idName}, reward_model, net_arch, learning_rate, lookback_window); ENVIRONMENT levers are the action space / market mechanics; DATASET levers are which data (asset, timeframe, fidelity_set). Levers marked "ignore" are infrastructure — NEVER pin or sweep them. Levers: ${JSON.stringify(manifest.levers)}.`,
    `Do TWO things: (1) MATCH — pick the EXISTING hypotheses (by id) that genuinely test THIS paper's claims; (2) SUGGEST — propose NEW testable hypotheses the existing ones don't cover, each a runnable spec using ONLY declared lever names. Never duplicate an existing hypothesis as a new one.`,
    ``,
    `HOW A SUGGESTED HYPOTHESIS IS JUDGED — every spec MUST fit one of these or its verdict is meaningless:`,
    `• SINGLE-CONTEXT (just fixed/sweep): a POOLED beats-buy-and-hold test — PROVEN iff the BEST matching run beats buy-and-hold OOS net of fees (return_vs_hold_pct > 0) over enough matching runs that report it; else DISPROVED/UNTESTED. It NEVER compares two configs. A run is evidence IFF it equals every "fixed" lever AND each "sweep" value is among the options — so MORE fixed levers = a narrow coherent match; too few = a flood of unrelated runs (a spec pinning only "timeframe" once matched ~4968 runs and was meaningless).`,
    `• COMPARISON (a "compare" block): partitions the matching runs by ONE lever's values and judges them AGAINST each other — the ONLY judgeable form of "A vs B".`,
    ``,
    `RULE 1 — PRECISION: "fixed" MUST pin EVERY lever that DEFINES the configuration — the identity lever (${idName}) PLUS the other model levers PLUS the data/market context (timeframe, fidelity_set, asset) — so the match is one coherent family (single digits to low tens). A single-context spec that omits ${idName} is almost always too broad and will be rejected.`,
    `RULE 2 — SWEEP IS THE RARE EXCEPTION: prefer ONE fully-pinned config with NO sweep (the tightest verdict, your default). If you sweep, vary AT MOST the ONE tuning dimension under test (e.g. learning_rate) with everything else pinned. NEVER put ${idName}/reward_model/net_arch in "sweep".`,
    `RULE 3 — COMPARATIVE CLAIMS ("A outperforms B", "X is necessary", "recurrent beats feedforward"): use a "compare" block — pin the SHARED config in "fixed" and put the contrasted lever + its values in "compare": {"lever": "<lever>", "values": [a, b]}, with "comparison": {"kind": "beats-baseline", "baselineIndex": 0} (index 0 = the value the others must beat). This judges the values against each other. NEVER express a comparison as a pooled sweep — it WILL be rejected.`,
    `RULE 4 — TRUE CROSS-CONTEXT CLAIMS (about the EFFECT OF THE CONTEXT — long-only vs long+short, one asset vs another): ${CONTEXT_SPANNING_RULE}`,
    `RULE 5 — PIN TO CONFIGS THAT PLAUSIBLY RUN: a spec matching no completed, metric-bearing run stays UNTESTED forever. Use lever values the paper maps onto that real runs would instantiate.`,
    HYPOTHESIS_RULE,
    `CLAIM LABEL — give EACH new hypothesis a short "claim": the SPECIFIC paper claim it tests (e.g. "Momentum predicts returns" vs "Vol-targeting lifts Sharpe"). Hypotheses that test the SAME claim share the SAME claim string verbatim; a paper that argues several distinct things will have several distinct claims — that is how the paper is scored claim-by-claim. Use the SAME wording for the same claim so they group.`,
    ``,
    `WORKED EXAMPLE — BAD: {"title":"Recurrent outperforms non-recurrent","spec":{"sweep":{"model_name":["ppo-custom","reppo-custom"]},"fixed":{"timeframe":"1h"}}} — only timeframe pinned (matches the whole backlog) AND a comparison as a pooled sweep (never compares the arms). GOOD: {"title":"Recurrence beats a feedforward policy on 1h","claim":"Recurrent policies beat feedforward","rationale":"Comparative claim -> a compare over the identity lever with the shared config pinned; beats-baseline judges the recurrent value against the baseline.","comparison":{"kind":"beats-baseline","baselineIndex":0},"spec":{"fixed":{"timeframe":"1h","reward_model":"combo_unified","lookback_window":64},"compare":{"lever":"model_name","values":["ppo-custom","reppo-custom"]}}}`,
    ``,
    `Return ONLY a single JSON object (no prose, no code fence): {"matchExistingIds": [string], "newHypotheses": [{"title": string, "claim": string, "rationale": string, "comparison"?: {"kind": "beats-baseline"|"invariant"|"differs", "baselineIndex"?: number, "tolerance"?: number}, "spec": {"fixed"?: {"<lever>": value}, "sweep"?: {"<lever>": [values]}, "compare"?: {"lever": "<lever>", "values": [v1, v2]}, "environments"?: [{"<env-lever>": value}], "datasets"?: [{"<dataset-lever>": value}], "seeds"?: [number]}}]}. ONLY declared lever names; a single-context spec MUST pin ${idName}; use [] for either list when empty.`,
  ].join('\n')
}

export function buildSuggestHypothesesUserContent(input: {
  manifest: TrainerManifest
  paper: Record<string, unknown>
  existingHypotheses: Array<{ id: string; title?: string; rationale?: string; spec?: unknown }>
  text?: string
}): string {
  // Surface lever ROLES so the prompt's "pin the defining levers / route comparisons correctly" rules are
  // grounded in THIS manifest — resolved identically to the coercion guard so prompt + guard agree.
  const { identityLever, modelLevers, envLevers, datasetLevers } = resolveModelLevers(
    input.manifest,
  )
  const leverGuide = {
    identityLever: identityLever ?? null,
    modelLevers: [...modelLevers],
    environmentLevers: [...envLevers],
    datasetLevers: [...datasetLevers],
  }
  return JSON.stringify({
    paper: input.paper,
    existingHypotheses: input.existingHypotheses,
    leverGuide,
    ...(input.text ? { text: input.text } : {}),
  })
}

/** Coerce the model's suggest response into matched-existing ids + validated new hypothesis items. */
export function coerceSuggestedHypotheses(
  raw: unknown,
  manifest: TrainerManifest,
): {
  matchIds: string[]
  newItems: {
    title: string
    rationale: string
    spec: ExperimentSpec
    comparison?: HypothesisComparison
    claim?: string
  }[]
} {
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

export const HYPOTHESIS_WEIGHT_MIN = 1
export const HYPOTHESIS_WEIGHT_MAX = 5

export function buildWeighHypothesesSystemPrompt(manifest: TrainerManifest): string {
  return [
    `You are a rigorous research librarian for the "${manifest.name}" training project.`,
    `You are given a PAPER (its claim and fields, plus its text when available) and the hypotheses currently LINKED to it (each with its compact spec).`,
    `Your job is a SCRUTINOUS COVERAGE + WEIGHT pass: make sure the linked hypotheses, TAKEN TOGETHER, coherently PROVE OR DISPROVE the paper.`,
    `STEP 1 — DECOMPOSE: break the paper into its distinct testable CLAIMS (the things it argues).`,
    `STEP 2 — MAP: for each claim, list the linked hypotheses that would PROVE OR DISPROVE it. A hypothesis only COVERS a claim if its spec, WHEN RUN, actually settles that claim under how a verdict is computed: a single-context spec is judged "does the best matching run beat buy-and-hold OOS net of fees"; a compare spec judges its contrasted lever's arms against each other. Do NOT count a loosely-related hypothesis as covering a claim it can't actually settle.`,
    `STEP 3 — GAPS: any claim with NO covering hypothesis goes in "uncoveredClaims" (verbatim claim text). Do NOT inflate an unrelated hypothesis's weight to paper over a gap — an honest gap is more useful than a fake cover.`,
    `STEP 4 — WEIGH: assign each linked hypothesis an integer importance WEIGHT from ${HYPOTHESIS_WEIGHT_MIN} (a minor / supporting detail) to ${HYPOTHESIS_WEIGHT_MAX} (the paper's CENTRAL claim — what it most stands or falls on), reflecting how LOAD-BEARING the claim it covers is FOR THIS PAPER. The same hypothesis can be central to one paper and peripheral in another — judge against THIS paper's argument, not general merit. Most papers have ONE or a few central claims (4–5) and several supporting ones (1–2); do NOT flatten everything to the same weight.`,
    `Return ONLY a single JSON object (no prose, no code fence): {"claims": [{"claim": string, "hypothesisIds": [string-or-1-based-index]}], "weights": [{"index": number, "id": string, "weight": number, "reason": string}], "uncoveredClaims": [string]}. In "weights" use the "index" shown for each hypothesis below (copy the "id" too if you can), include EVERY linked hypothesis exactly once, and give a one-clause "reason". Use [] for "uncoveredClaims" when every claim is covered.`,
  ].join('\n')
}

export function buildWeighHypothesesUserContent(input: {
  paper: {
    title?: string
    claim?: string
    approach?: string
    assumptions?: unknown
    claimedMetrics?: unknown
    url?: string
  }
  hypotheses: { id: string; title?: string; rationale?: string; spec?: unknown }[]
  text?: string
}): string {
  // Number the hypotheses (1-based) so the model can refer to each by a stable INDEX even if it doesn't
  // echo the hash id — coerceHypothesisCoverage resolves either back to the id. Include each hypothesis's
  // spec so the model can judge what each WOULD actually settle (coverage), not just its title.
  const linkedHypotheses = input.hypotheses.map((h, i) => ({ index: i + 1, ...h }))
  return JSON.stringify({ paper: input.paper, linkedHypotheses, paperText: input.text }, null, 2)
}

/**
 * Coerce the scrutinous weigh response into the per-hypothesis weights PLUS the coverage signal: claims with
 * no covering hypothesis (`uncoveredClaims`) and the claim→hypotheses map (`coverageByClaim`). Tolerant of the
 * legacy weights-only shape (a bare array or `{weights:[...]}`), in which case coverage is simply unknown
 * (empty), never "no gaps". Reuses {@link coerceHypothesisWeights}' id-or-1-based-index resolution.
 */
export function coerceHypothesisCoverage(
  raw: unknown,
  linkedIds: string[],
): {
  weights: { id: string; weight: number; reason: string }[]
  uncoveredClaims: string[]
  coverageByClaim: { claim: string; hypothesisIds: string[] }[]
} {
  const o =
    raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {}
  const weights = coerceHypothesisWeights(raw, linkedIds)
  const uncoveredClaims = Array.isArray(o.uncoveredClaims)
    ? (o.uncoveredClaims as unknown[])
        .filter((c): c is string => typeof c === 'string' && c.trim().length > 0)
        .map((c) => c.trim())
    : []
  const allow = new Set(linkedIds)
  const resolveId = (v: unknown): string => {
    if (typeof v === 'string' && allow.has(v)) return v
    const idx = Number(v)
    if (Number.isInteger(idx) && idx >= 1 && idx <= linkedIds.length) return linkedIds[idx - 1]
    return ''
  }
  const coverageByClaim: { claim: string; hypothesisIds: string[] }[] = []
  if (Array.isArray(o.claims)) {
    for (const c of o.claims as unknown[]) {
      if (!c || typeof c !== 'object') continue
      const rec = c as Record<string, unknown>
      const claim = typeof rec.claim === 'string' ? rec.claim.trim() : ''
      if (!claim) continue
      const ids: string[] = []
      const list = Array.isArray(rec.hypothesisIds) ? (rec.hypothesisIds as unknown[]) : []
      for (const v of list) {
        const id = resolveId(v)
        if (id && ids.indexOf(id) < 0) ids.push(id)
      }
      coverageByClaim.push({ claim, hypothesisIds: ids })
    }
  }
  return { weights, uncoveredClaims, coverageByClaim }
}

// Parse the model's weight response into {id, weight, reason} rows. Resolve each row to a hypothesis by its
// `id` (when in the linked set) or its 1-based `index` into `linkedIds` — robust to a model that doesn't
// echo the hash id. Each hypothesis once, weight rounded + clamped to [MIN, MAX]; tolerant of a bare array.
export function coerceHypothesisWeights(
  raw: unknown,
  linkedIds: string[],
): { id: string; weight: number; reason: string }[] {
  const allow = new Set(linkedIds)
  const o =
    raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {}
  const rows = Array.isArray((o as { weights?: unknown }).weights)
    ? ((o as { weights: unknown[] }).weights as unknown[])
    : Array.isArray(raw)
      ? (raw as unknown[])
      : []
  const seen = new Set<string>()
  const out: { id: string; weight: number; reason: string }[] = []
  for (const r of rows) {
    if (!r || typeof r !== 'object') continue
    const rec = r as Record<string, unknown>
    let id = typeof rec.id === 'string' && allow.has(rec.id) ? rec.id : ''
    if (!id) {
      const idx = Number(rec.index)
      if (Number.isInteger(idx) && idx >= 1 && idx <= linkedIds.length) id = linkedIds[idx - 1]
    }
    if (!id || seen.has(id)) continue
    const n = Number(rec.weight)
    if (!Number.isFinite(n)) continue
    const weight = Math.min(HYPOTHESIS_WEIGHT_MAX, Math.max(HYPOTHESIS_WEIGHT_MIN, Math.round(n)))
    seen.add(id)
    out.push({ id, weight, reason: typeof rec.reason === 'string' ? rec.reason : '' })
  }
  return out
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

// ──────────────────────────────────────────────────────────────────────────
// Open-ended paper research (ModelTrainerTools.researchTrainingPapers). These pure helpers shape the
// discover → verify → synthesize pipeline: the discovery query, candidate coercion/dedup, the domain
// relevance claim, and the admission rule. The impure orchestration (deep-research seam + storage)
// lives in ModelTrainerTools; these stay node-free and directly tested.
// ──────────────────────────────────────────────────────────────────────────

/**
 * Compose the open-ended discovery query for `researchTrainingPapers` from the same manifest
 * ingredients {@link buildAnalyzePaperSystemPrompt} uses — the project name, its plain-language domain
 * description, the objective (name + direction), and the candidate model families (the identity
 * lever's choices) — then steer the search toward scholarly sources. Missing pieces are omitted so no
 * dangling label or `undefined` leaks into the query.
 */
export function buildPaperResearchGoal(
  manifest: TrainerManifest,
  opts?: { notes?: string },
): string {
  const { identityLever } = resolveModelLevers(manifest)
  const families = identityLever
    ? (manifest.levers[identityLever]?.choices ?? []).filter(
        (c): c is string => typeof c === 'string' && c.trim().length > 0,
      )
    : []
  const parts: string[] = []
  const desc = typeof manifest.description === 'string' ? manifest.description.trim() : ''
  parts.push(desc ? `${manifest.name}: ${desc}` : manifest.name)
  parts.push(`Objective: ${manifest.objective.name} (${manifest.objective.direction} is better).`)
  if (families.length) parts.push(`Candidate model families: ${families.join(', ')}.`)
  parts.push(
    'Find recent research papers (arXiv, OpenReview, peer-reviewed) proposing methods to improve this.',
  )
  const notes = opts?.notes?.trim()
  if (notes) parts.push(notes)
  return parts.join(' ')
}

/**
 * Canonicalize a paper URL for identity comparison: lowercase host, drop the fragment and `utm_*`
 * tracking params, strip a trailing slash, and collapse arxiv `abs`/`pdf`/`vN`/`.pdf` variants of the
 * SAME paper to one key (so a paper added as `/abs/ID` and re-discovered as `/pdf/IDv2` dedupes). An
 * unparseable string degrades to its trimmed, lowercased self rather than throwing.
 */
export function normalizeResearchUrl(url: string): string {
  const raw = (url ?? '').trim()
  let parsed: URL
  try {
    parsed = new URL(raw)
  } catch {
    return raw.toLowerCase()
  }
  const host = parsed.host.toLowerCase()
  const arxiv = /(?:^|\.)arxiv\.org$/.test(host)
  if (arxiv) {
    // Collapse abs/pdf/version/.pdf variants of the SAME paper (e.g. /abs/2401.12345, /pdf/2401.12345v2,
    // /pdf/2401.12345.pdf) to one id, preserving old-style category ids (hep-th/9901001).
    const id = parsed.pathname
      .replace(/^\/+/, '')
      .replace(/^(?:abs|pdf)\//, '')
      .replace(/\.pdf$/i, '')
      .replace(/v\d+$/i, '')
    if (id) return `arxiv:${id.toLowerCase()}`
  }
  let path = parsed.pathname.replace(/\/+$/, '')
  const search = [...parsed.searchParams.entries()].filter(([k]) => !/^utm_/i.test(k))
  const query = search.length ? '?' + search.map(([k, v]) => `${k}=${v}`).join('&') : ''
  return `${parsed.protocol}//${host}${path}${query}`
}

/**
 * Coerce the raw discovery seam output ({@link import('thefactory-tools/types').DiscoveredSource}[] or
 * loose records) into validated {@link PaperCandidate}s. A candidate must have a non-empty title
 * (`title` or the seam's `name`) AND a well-formed http(s) URL — the identity requirement; anything
 * else is dropped so nothing untitled/unlinkable enters the pipeline.
 */
export function coercePaperCandidates(raw: unknown[]): PaperCandidate[] {
  if (!Array.isArray(raw)) return []
  const out: PaperCandidate[] = []
  for (const r of raw) {
    if (!r || typeof r !== 'object') continue
    const o = r as Record<string, unknown>
    const title =
      typeof o.title === 'string' && o.title.trim()
        ? o.title.trim()
        : typeof o.name === 'string' && o.name.trim()
          ? o.name.trim()
          : ''
    const url = typeof o.url === 'string' ? o.url.trim() : ''
    if (!title || !/^https?:\/\/\S+/i.test(url)) continue
    const candidate: PaperCandidate = { title, url }
    if (Array.isArray(o.hints)) {
      const hints = o.hints.filter((h): h is string => typeof h === 'string')
      if (hints.length) candidate.hints = hints
    }
    out.push(candidate)
  }
  return out
}

/**
 * Drop candidates already represented in the papers registry (by normalized URL OR normalized title)
 * and intra-batch duplicates, preserving the input order of the survivors. A candidate with an empty
 * title matches on URL only. This is what guarantees a research run never re-drafts a paper already
 * present — whether added manually, via `analyzePaperFromUrl`, or by an earlier research run.
 */
export function dedupePaperCandidates(
  candidates: PaperCandidate[],
  existing: { url?: string; title?: string }[],
): PaperCandidate[] {
  const titleKey = (t?: string) => (t ?? '').trim().toLowerCase()
  const urlKeys = new Set<string>()
  const titleKeys = new Set<string>()
  for (const e of existing) {
    if (typeof e.url === 'string' && e.url.trim()) urlKeys.add(normalizeResearchUrl(e.url))
    const tk = titleKey(e.title)
    if (tk) titleKeys.add(tk)
  }
  const out: PaperCandidate[] = []
  for (const c of candidates) {
    const uk = normalizeResearchUrl(c.url)
    const tk = titleKey(c.title)
    if (urlKeys.has(uk)) continue
    if (tk && titleKeys.has(tk)) continue
    out.push(c)
    urlKeys.add(uk)
    if (tk) titleKeys.add(tk)
  }
  return out
}

/**
 * The natural-language claim the deep-research verify gate judges a discovered candidate against —
 * grounded in the project DOMAIN (name + description + objective), not merely the metric, so an
 * off-domain paper that is trivially "relevant to <metric>" is rejected. The verdict on the paper's
 * real fetched page is what admits or rejects it.
 */
export function paperRelevanceClaim(candidate: PaperCandidate, manifest: TrainerManifest): string {
  const desc = typeof manifest.description === 'string' ? manifest.description.trim() : ''
  const domain = desc ? `${manifest.name}: ${desc}` : manifest.name
  return (
    `"${candidate.title}" (${candidate.url}) is a real research paper whose method is applicable to ` +
    `${domain} (objective: ${manifest.objective.name}).`
  )
}

/**
 * Admission rule for a candidate: the verify verdict must sit on the supported side of the ladder
 * (`confirmed` or `implied`) AND meet the confidence floor. `unverifiable`/`refuted` are rejected at
 * any confidence — so a hallucinated or off-domain paper (no supporting evidence on its own page) is
 * never drafted.
 */
export function isPaperVerdictAdmitted(verdict: ClaimVerdict, minConfidence: number): boolean {
  if (!verdict) return false
  const ok = verdict.status === 'confirmed' || verdict.status === 'implied'
  return ok && typeof verdict.confidence === 'number' && verdict.confidence >= minConfidence
}

/**
 * Known scholarly/paper-venue host patterns. `discoverSources` is domain-blind (it ranks by generic
 * richness, not paper-ness), so a candidate on one of these hosts is far more likely to be a real paper
 * than a blog/marketing page. Used only to ORDER candidates — never to drop them (the verify gate stays
 * the real filter).
 */
const PAPER_HOST_PATTERNS: readonly RegExp[] = [
  /(^|\.)arxiv\.org$/,
  /(^|\.)openreview\.net$/,
  /(^|\.)aclanthology\.org$/,
  /(^|\.)semanticscholar\.org$/,
  /(^|\.)doi\.org$/,
  /(^|\.)biorxiv\.org$/,
  /(^|\.)mlr\.press$/,
  /(^|\.)neurips\.cc$/,
  /(^|\.)nature\.com$/,
  /(^|\.)sciencedirect\.com$/,
  /(^|\.)springer\.com$/,
  /(^|\.)ieee\.org$/,
  /(^|\.)acm\.org$/,
  /\.edu$/,
]

/**
 * Paper-likeness score for a discovered candidate URL: `2` for a known paper venue, `1` for a bare
 * PDF link on any host, `0` for a generic page. Never throws on a malformed URL (scores `0`). Higher =
 * try it sooner within the discovery budget.
 */
export function paperHostAffinity(url: string): number {
  let parsed: URL
  try {
    parsed = new URL((url ?? '').trim())
  } catch {
    return 0
  }
  const host = parsed.host.toLowerCase()
  if (PAPER_HOST_PATTERNS.some((re) => re.test(host))) return 2
  if (/\.pdf$/i.test(parsed.pathname)) return 1
  return 0
}

/**
 * Stable re-rank of discovered candidates: paper-venue hosts first, bare PDFs next, generic pages last,
 * preserving input order within each tier. Never drops a candidate — it only decides who is TRIED first,
 * so with an over-scanned discovery pool the low-affinity tail is verified only if the target isn't hit.
 */
export function rankPaperCandidates(candidates: PaperCandidate[]): PaperCandidate[] {
  return candidates
    .map((candidate, index) => ({ candidate, index, score: paperHostAffinity(candidate.url) }))
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map((entry) => entry.candidate)
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
  const saliencyByGroup = coerceNumberMap(raw.saliencyByGroup)
  if (saliencyByGroup) step.saliencyByGroup = saliencyByGroup
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

/**
 * Whether a run used BlackSwan's MULTI-timeline (resolved-from-fidelity) data provider rather than the
 * SINGLE-timeline one — mirrors trainer/fidelity.py `resolve_fidelity` + data_factory's single-vs-multi
 * routing. ONLY the multi path was hit by the observation/reward desync (fixed in pipeline v5); the single
 * path (`1d`@`1d`, `1h`@`1h`) is unaffected. A run is SINGLE iff its (auto-resolved) `fidelity_set` equals
 * its `timeframe` and is a base cadence (`1h`/`1d`); everything else — multi-layer (`1h+1d`, `1h+1d+1w`,
 * `1d+1w`), a coarser single layer at a finer step (`1d`@`1h`), a finer base at a coarser step (`1h`@`1d`),
 * or `auto`@`1h` (= `1h+1d`) — is MULTI and AFFECTED. Absent fields fall back to the lever defaults
 * (`timeframe` `1d`, `fidelity_set` `auto`). Pure; returns false for a missing config (never invalidate the
 * unknown).
 */
export function isRunAffectedByFidelityDesync(
  config: Record<string, unknown> | undefined,
): boolean {
  if (!config || typeof config !== 'object') return false
  const timeframe = String(config.timeframe ?? '1d')
  const raw = config.fidelity_set
  const isAuto = raw === undefined || raw === null || raw === '' || raw === 'auto'
  // `auto` follows the step: an hourly step observes 1h+1d (MULTI), a daily step observes just 1d (SINGLE).
  const fidelitySet = isAuto ? (timeframe === '1h' ? '1h+1d' : '1d') : String(raw)
  const isSingle = fidelitySet === timeframe && (fidelitySet === '1h' || fidelitySet === '1d')
  return !isSingle
}

/**
 * Whether a PENDING launch spec (`{fixed, sweep}`) would produce ANY run on the affected multi-timeline
 * path — so a queued campaign that SWEEPS `timeframe`/`fidelity_set` is caught even when its `fixed` base
 * is on the single path. Builds the timeframe × fidelity_set candidates from fixed+sweep and defers to
 * {@link isRunAffectedByFidelityDesync}. Conservative: any affected combination flags the whole item.
 */
export function isSpecAffectedByFidelityDesync(spec: Record<string, unknown> | undefined): boolean {
  if (!spec || typeof spec !== 'object') return false
  const fixed = (spec.fixed && typeof spec.fixed === 'object' ? spec.fixed : {}) as Record<
    string,
    unknown
  >
  const sweep = (spec.sweep && typeof spec.sweep === 'object' ? spec.sweep : {}) as Record<
    string,
    unknown
  >
  const timeframes =
    Array.isArray(sweep.timeframe) && sweep.timeframe.length ? sweep.timeframe : [fixed.timeframe]
  const fidelitySets =
    Array.isArray(sweep.fidelity_set) && sweep.fidelity_set.length
      ? sweep.fidelity_set
      : [fixed.fidelity_set]
  for (const timeframe of timeframes) {
    for (const fidelity_set of fidelitySets) {
      if (isRunAffectedByFidelityDesync({ ...fixed, timeframe, fidelity_set })) return true
    }
  }
  return false
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

/**
 * Summarize a trace's per-step {@link DecisionStep.saliencyByGroup} into which input GROUP drove each
 * attributed decision over the rollout — the temporal companion to the run-aggregate `byGroup`. The
 * dominant group per step is the largest |saliency| (ties broken by group name for determinism). Returns
 * `undefined` when no step carries per-step group saliency (a run without it ingests normally). Pure.
 */
export function summarizeStepAttribution(
  trace: Pick<DecisionTrace, 'steps'>,
): StepAttributionSummary | undefined {
  const perStep: StepAttributionSummary['perStep'] = []
  const groups = new Set<string>()
  const dominanceCounts: Record<string, number> = {}
  for (const step of trace.steps) {
    const entries = Object.entries(step.saliencyByGroup ?? {}).filter(([, v]) => isFiniteNumber(v))
    if (!entries.length) continue
    let dominantGroup = entries[0][0]
    let best = Math.abs(entries[0][1])
    for (const [group, value] of entries) {
      groups.add(group)
      const magnitude = Math.abs(value)
      if (magnitude > best || (magnitude === best && group < dominantGroup)) {
        best = magnitude
        dominantGroup = group
      }
    }
    dominanceCounts[dominantGroup] = (dominanceCounts[dominantGroup] ?? 0) + 1
    perStep.push({ step: step.step, dominantGroup, byGroup: Object.fromEntries(entries) })
  }
  if (!perStep.length) return undefined
  return { groups: [...groups].sort(), perStep, dominanceCounts, samples: perStep.length }
}

// --- Models catalog ----------------------------------------------------------

const MODEL_CATEGORIES: ReadonlySet<string> = new Set(['rl', 'supervised', 'baseline', 'component'])
const PROPOSED_IMPROVEMENT_KINDS: ReadonlySet<string> = new Set([
  'model',
  'data',
  'metric',
  'environment',
  'capability',
  'other',
])

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
/** The `model_name` values a model binds runs through — from its `flavors`, falling back to a legacy
 * flat `modelNames[]`. The one place that reconciles the two shapes for the tool side. */
export function modelBindingNames(model: {
  flavors?: { modelName?: string }[]
  modelNames?: string[]
}): string[] {
  if (Array.isArray(model.flavors) && model.flavors.length) {
    return model.flavors
      .map((f) => f.modelName)
      .filter((n): n is string => typeof n === 'string' && !!n)
  }
  return Array.isArray(model.modelNames)
    ? model.modelNames.filter((n): n is string => typeof n === 'string' && !!n)
    : []
}

export function discoverManifestModelCandidates(
  manifest: TrainerManifest,
  existingModels: Array<{
    slug?: string
    modelNames?: string[]
    flavors?: { modelName?: string }[]
  }>,
): { modelName: string; slug: string; name: string; category: ModelCategory }[] {
  const lever = manifest.levers?.model_name as TrainerLeverSpec | undefined
  if (!lever || lever.type !== 'choice' || !Array.isArray(lever.choices)) return []
  const haveSlugs = new Set<string>()
  const haveBindings = new Set<string>()
  for (const m of existingModels) {
    if (m.slug) haveSlugs.add(m.slug)
    for (const n of modelBindingNames(m)) haveBindings.add(n)
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
  proposedImprovements: ProposedImprovement[]
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
  const proposedImprovements: ProposedImprovement[] = []
  const rawImprovements = Array.isArray(o.proposedImprovements) ? o.proposedImprovements : []
  for (const item of rawImprovements) {
    if (!item || typeof item !== 'object') continue
    const p = item as Record<string, unknown>
    const title = typeof p.title === 'string' && p.title.trim() ? p.title.trim() : ''
    if (!title) continue
    const detail = typeof p.detail === 'string' ? p.detail.trim() : ''
    const kind =
      typeof p.kind === 'string' && PROPOSED_IMPROVEMENT_KINDS.has(p.kind)
        ? (p.kind as ProposedImprovementKind)
        : 'capability'
    proposedImprovements.push({ title, detail, kind })
  }
  return { matchModelIds, proposedModels, proposedImprovements }
}

/** Normalized identity for a proposed improvement — title, case- and whitespace-insensitive. */
export function proposedImprovementKey(title: string): string {
  return (title || '').trim().replace(/\s+/g, ' ').toLowerCase()
}

/**
 * Merge a fresh `incoming` list of proposed improvements (from a Find-models re-run) with the `existing`
 * persisted ones, so a user's `inapplicable` mark SURVIVES: an incoming item matching an existing
 * inapplicable one (by {@link proposedImprovementKey}) inherits `inapplicable:true` (never resurrected as
 * active), and an inapplicable item the new run DROPPED is kept (appended) so it stays listed + referable.
 * Applicable items follow the fresh list (dropped ones go, exactly as an overwrite would).
 */
export function mergeProposedImprovements(
  existing: ProposedImprovement[] | undefined,
  incoming: ProposedImprovement[],
): ProposedImprovement[] {
  const inapplicable = new Map<string, ProposedImprovement>()
  for (const e of existing ?? []) {
    if (e && e.inapplicable) {
      const k = proposedImprovementKey(e.title)
      if (k) inapplicable.set(k, e)
    }
  }
  const out: ProposedImprovement[] = []
  const seen = new Set<string>()
  for (const item of incoming) {
    const k = proposedImprovementKey(item.title)
    if (k) seen.add(k)
    out.push(k && inapplicable.has(k) ? { ...item, inapplicable: true } : item)
  }
  for (const [k, e] of inapplicable) {
    if (!seen.has(k)) out.push(e)
  }
  return out
}

/** Of `proposed`, the models with NO catalog entry — matched by slug, by a catalog model's name slug, or
 *  by a catalog model's `model_name` binding. The Models tab's "Add to catalog" candidates. */
export function detectMissingPaperModels(
  proposed: ProposedModel[],
  catalog: Array<{
    slug?: string
    name?: string
    modelNames?: string[]
    flavors?: { modelName?: string }[]
    aliases?: string[]
  }>,
): ProposedModel[] {
  const slugs = new Set<string>()
  const bindings = new Set<string>()
  for (const m of catalog) {
    // Normalize on compare (a proposal's slug is already modelSlug'd) so an un-kebab manifest slug still matches.
    if (m.slug) slugs.add(modelSlug(m.slug))
    if (m.name) slugs.add(modelSlug(m.name))
    // A model also answers to its aliases — so "Policy Gradient" is not "missing" when a2c lists it.
    for (const a of m.aliases ?? []) slugs.add(modelSlug(a))
    for (const n of modelBindingNames(m)) bindings.add(n)
  }
  return proposed.filter((p) => !slugs.has(p.slug) && !bindings.has(p.slug))
}

/** System prompt for "Find models": match the paper to existing catalog models + propose missing ones. */
export function buildAnalyzePaperModelsSystemPrompt(manifest: TrainerManifest): string {
  return [
    `You are a model librarian for the "${manifest.name}" training project.`,
    `Objective: ${manifest.objective.name} (${manifest.objective.direction} is better).`,
    `You are given a PAPER (its fields, and its text when available) and the project's EXISTING catalog models.`,
    `Each existing model lists its aliases (other names it is known by) — MATCH a paper to an existing model when it refers to it by any alias (e.g. a paper on "Policy Gradient" matches the model whose aliases include "policy-gradient"), and do NOT propose it as missing.`,
    `Do THREE things: (1) MATCH — the ids of existing models this paper INTRODUCES or IMPROVES (a genuine "this paper is about that model" link, not a passing mention); (2) PROPOSE MODELS — any models the paper introduces or requires that are NOT in the catalog (nor an alias of one), each with what to add; (3) PROPOSE IMPROVEMENTS — the project FUNCTIONALITY that is missing for this paper's claims to be TESTED here. A proposed model is one KIND of improvement (kind:"model"); the others are new input features / data streams (kind:"data"), evaluation metrics (kind:"metric"), environment/market mechanics like fees or funding (kind:"environment"), or general capabilities (kind:"capability"). Only list functionality that genuinely BLOCKS testing a claim — if everything needed already exists, return [].`,
    `Return ONLY a single JSON object (no prose, no code fence): {"matchModelIds": [string], "proposedModels": [{"name": string, "slug": string, "description": string, "category": "rl|supervised|baseline|component", "proposal": string}], "proposedImprovements": [{"title": string, "detail": string, "kind": "model|data|metric|environment|capability|other"}]}. Use [] for any list when there is nothing to add. Every proposedModel should also appear as a proposedImprovement with kind:"model".`,
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
    aliases?: string[]
  }[]
  text?: string
}): string {
  return JSON.stringify({
    paper: input.paper,
    existingModels: input.existingModels,
    ...(input.text ? { text: input.text } : {}),
  })
}

/**
 * Validate the LLM's consolidation proposal against the REAL catalog ids: keep only groups whose
 * `canonicalId` is a known model, drop `duplicateIds` that are unknown / equal the canonical / repeat, and
 * drop a group left with no duplicates. A model id appears in at most ONE group (first wins) so the
 * proposed merges never conflict. Accepts either `{groups:[...]}` or a bare array.
 */
export function coerceConsolidationGroups(
  raw: unknown,
  validIds: Iterable<string>,
): ConsolidationGroup[] {
  const valid = validIds instanceof Set ? validIds : new Set(validIds)
  const o =
    raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {}
  const list = Array.isArray(raw) ? raw : Array.isArray(o.groups) ? o.groups : []
  const used = new Set<string>()
  const out: ConsolidationGroup[] = []
  for (const item of list) {
    if (!item || typeof item !== 'object') continue
    const g = item as Record<string, unknown>
    const canonicalId = typeof g.canonicalId === 'string' ? g.canonicalId : ''
    if (!valid.has(canonicalId) || used.has(canonicalId)) continue
    const seen = new Set<string>()
    const duplicateIds: string[] = []
    for (const d of Array.isArray(g.duplicateIds) ? g.duplicateIds : []) {
      if (typeof d !== 'string') continue
      if (d === canonicalId || !valid.has(d) || used.has(d) || seen.has(d)) continue
      seen.add(d)
      duplicateIds.push(d)
    }
    if (!duplicateIds.length) continue
    used.add(canonicalId)
    for (const d of duplicateIds) used.add(d)
    out.push({
      canonicalId,
      duplicateIds,
      reason: typeof g.reason === 'string' ? g.reason.trim() : '',
    })
  }
  return out
}

/** System prompt for "Consolidate": find groups of catalog models that are really the SAME approach. */
export function buildConsolidateModelsSystemPrompt(manifest: TrainerManifest): string {
  return [
    `You are a model librarian for the "${manifest.name}" training project.`,
    `Objective: ${manifest.objective.name} (${manifest.objective.direction} is better).`,
    `You are given the project's catalog MODELS (each with id, name, the model_name bindings it trains as, a description, and any aliases — other names it is ALREADY known by from prior merges).`,
    `Find groups of entries that are REALLY THE SAME model/algorithm — typically the same approach proposed from several papers under slightly different names — and should be ONE catalog entry (the duplicates fold into a canonical model as flavors).`,
    `Treat aliases as already-consolidated names: never propose merging an entry into another whose aliases already cover it, and prefer the entry that already carries aliases (it has absorbed others) as the canonical.`,
    `Be conservative: group only genuine duplicates of the same method, NOT merely related or same-family-but-distinct models (e.g. "DQN" and "Dueling DQN" are DIFFERENT). When unsure, do not group.`,
    `For each group pick the best-named, most-complete entry as the canonical and list the rest as duplicates.`,
    `Return ONLY a single JSON object (no prose, no code fence): {"groups": [{"canonicalId": string, "duplicateIds": [string], "reason": string}]}. Use {"groups": []} when there is nothing to merge. Every id MUST be one of the given model ids.`,
  ].join('\n')
}

export function buildConsolidateModelsUserContent(input: {
  models: {
    id: string
    name: string
    slug: string
    category: string
    description?: string
    modelNames?: string[]
    aliases?: string[]
  }[]
}): string {
  return JSON.stringify({ models: input.models })
}
