import { describe, it, expect } from 'vitest'
import {
  TRAINER_LAUNCHABLE_ACTIVITIES,
  TRAINER_EXEMPT_ACTIVITY_TYPES,
  TRAINER_CAPABILITY_RECORD_TYPES,
  TRAINER_EXEMPT_RECORDS,
  TRAINER_RUN_DELETE_CASCADE_SUFFIXES,
  TRAINER_RUN_KEYED_CHILD_SUFFIXES,
  TRAINER_RUN_SETUP_KEYED_SUFFIX,
  buildTrainerDataCapabilityManifest,
} from './trainerCapabilities.js'

describe('trainer launchable activities', () => {
  const byType = new Map(TRAINER_LAUNCHABLE_ACTIVITIES.map((a) => [a.activityType, a]))

  it('covers every chat-launchable trainer activity with a description', () => {
    // Every project operation a user can start from the viewer (both launch paths). inspect-trainer is
    // deliberately NOT here — it is the hub bootstrap (re-registers the manifest), an exempt activity.
    const expected = [
      'side-experiment',
      'train',
      'explore',
      'train-champion',
      'autopilot',
      'tournament',
      'judge',
      'evaluate',
      'cross-test',
      'continue-training',
      'config-space-analyze',
      'run-xai-analyze',
      'xai-narrate',
      'diagnose',
      'propose',
      'propose-experiments',
      'analyze-paper',
      'research-training-papers',
      'suggest-paper-hypotheses',
      'weigh-paper-hypotheses',
      'analyze-paper-models',
      'scan-models',
      'consolidate-models',
      'consolidate-hypotheses',
      'benchmark-model-device',
      'data-catalog',
      'mine-data',
      'discover-data',
      'approve-data-source',
    ].sort()
    expect([...byType.keys()].sort()).toEqual(expected)
    for (const a of TRAINER_LAUNCHABLE_ACTIVITIES) {
      expect(a.label.length).toBeGreaterThan(0)
      expect(a.description.length).toBeGreaterThan(0)
    }
  })

  it('pins requiresApi on the API-only vs compute/CLI activities (a launcher gate)', () => {
    // requiresApi:true ⇒ a CLI-only model is rejected at admission. Pin the consumers so a wrong flag breaks.
    expect(byType.get('judge')?.requiresApi).toBe(true)
    expect(byType.get('propose')?.requiresApi).toBe(true)
    expect(byType.get('scan-models')?.requiresApi).toBe(true)
    expect(byType.get('xai-narrate')?.requiresApi).toBe(true)
    // Pure-compute (no LLM) — never requiresApi.
    expect(byType.get('train')?.requiresApi ?? false).toBe(false)
    expect(byType.get('explore')?.requiresApi ?? false).toBe(false)
    expect(byType.get('evaluate')?.requiresApi ?? false).toBe(false)
    expect(byType.get('cross-test')?.requiresApi ?? false).toBe(false)
    // Model-selectable research (CLI allowed) — not requiresApi.
    expect(byType.get('research-training-papers')?.requiresApi ?? false).toBe(false)
    expect(byType.get('discover-data')?.requiresApi ?? false).toBe(false)
  })

  it('declares required params where the activity cannot run without them', () => {
    expect(byType.get('train')?.requiredParams).toContain('spec')
    expect(byType.get('cross-test')?.requiredParams).toContain('runKey')
    expect(byType.get('continue-training')?.requiredParams).toContain('runKey')
    expect(byType.get('run-xai-analyze')?.requiredParams).toContain('runKey')
    expect(byType.get('evaluate')?.requiredParams).toContain('runKeys')
    expect(byType.get('analyze-paper')?.requiredParams).toContain('url')
    expect(byType.get('suggest-paper-hypotheses')?.requiredParams).toContain('paperId')
    expect(byType.get('benchmark-model-device')?.requiredParams).toContain('modelId')
    // A side-experiment cannot run without the claim it tests and the matrix that tests it.
    expect(byType.get('side-experiment')?.requiredParams).toContain('thesis')
    expect(byType.get('side-experiment')?.requiredParams).toContain('matrix')
    expect(byType.get('side-experiment')?.optionalParams).toContain('hypothesisId')
    // A parameterless activity declares no required params.
    expect(byType.get('scan-models')?.requiredParams ?? []).toEqual([])
  })

  it('does not overlap the exempt set (a partition)', () => {
    for (const t of TRAINER_EXEMPT_ACTIVITY_TYPES) expect(byType.has(t)).toBe(false)
    expect(TRAINER_EXEMPT_ACTIVITY_TYPES).toContain('inspect-trainer')
  })

})

describe('trainer capability record types', () => {
  const bySuffix = new Map(
    TRAINER_CAPABILITY_RECORD_TYPES.filter((t) => t.suffix).map((t) => [t.suffix, t]),
  )

  it('every editable type has a non-empty editableFields allow-list', () => {
    for (const t of TRAINER_CAPABILITY_RECORD_TYPES) {
      if (t.editable) expect((t.editableFields ?? []).length).toBeGreaterThan(0)
      if (t.creatable) expect((t.creatableFields ?? []).length).toBeGreaterThan(0)
      expect((t.suffix !== undefined) !== (t.fixedType !== undefined)).toBe(true) // exactly one of suffix / fixedType
    }
  })

  it('declares the config record types the chat can now edit/create', () => {
    expect(bySuffix.get('-environment')?.creatable).toBe(true)
    expect(bySuffix.get('-environment')?.creatableFields).toContain('settings')
    expect(bySuffix.get('-dataset')?.creatable).toBe(true)
    expect(bySuffix.get('-dataset')?.editableFields).toContain('settings')
    expect(bySuffix.get('-filter-rule')?.creatable).toBe(true)
    // Per-run flags: editable overrides.
    expect(bySuffix.get('-reliability')?.editableFields).toContain('level')
    expect(bySuffix.get('-unrunnable')?.editableFields).toContain('unrunnable')
    // The existing declared types are preserved.
    expect(bySuffix.get('-hypothesis')?.editable).toBe(true)
    expect(bySuffix.get('-scorecard')?.editable).toBe(true)
    expect(bySuffix.get('-paper')?.editable).toBe(true)
  })

  it('declares the side-experiment record (A4) as a chat-owned, hypothesis-linkable thesis evidence type', () => {
    const exp = bySuffix.get('-experiment')
    expect(exp?.creatable).toBe(true)
    expect(exp?.editable).toBe(true)
    // Chat files an experiment as thesis evidence and links it to a hypothesis.
    expect(exp?.creatableFields).toContain('thesis')
    expect(exp?.creatableFields).toContain('matrix')
    expect(exp?.creatableFields).toContain('hypothesisId')
    expect(exp?.editableFields).toContain('hypothesisId')
    // Cells/aggregate/verdict are produced by the campaign, never set by chat.
    expect(exp?.creatableFields).not.toContain('cells')
    expect(exp?.creatableFields).not.toContain('verdict')
    expect(exp?.createDefaults).toMatchObject({ source: 'llm' })
  })

  it('declares the AI-companion strategy record (A3.4) as a chat-owned singleton', () => {
    // The companion's working memory: the AI reads+writes it to persist its plan across the long
    // launch→wait→read→decide loop; the human sees it in the app. Chat-owned ⇒ editable AND creatable.
    const strategy = bySuffix.get('-strategy')
    expect(strategy?.editable).toBe(true)
    expect(strategy?.creatable).toBe(true)
    for (const f of ['summary', 'decided', 'open', 'nextSteps']) {
      expect(strategy?.editableFields).toContain(f)
      expect(strategy?.creatableFields).toContain(f)
    }
  })

  it('the run record + its derived children are covered (queryable run, exempt/edit children)', () => {
    // The run record itself is a declared (queryable, non-editable) type; deletion is the bespoke deleteRuns tool.
    const run = TRAINER_CAPABILITY_RECORD_TYPES.find((t) => t.suffix === '')
    expect(run).toBeTruthy()
    expect(run?.editable ?? false).toBe(false)
    // The delete cascade the bespoke deleteRuns tool replicates — pinned EXACTLY (not a subset) so dropping
    // any suffix (e.g. -settest) fails the guard, and derived from the run-keyed source + setup-keyed marker.
    expect([...TRAINER_RUN_DELETE_CASCADE_SUFFIXES].sort()).toEqual(
      ['-evaluation', '-reliability', '-settest', '-unrunnable', '-verdict', '-xai-narrative'].sort(),
    )
    expect(TRAINER_RUN_DELETE_CASCADE_SUFFIXES).toEqual([
      ...TRAINER_RUN_KEYED_CHILD_SUFFIXES,
      TRAINER_RUN_SETUP_KEYED_SUFFIX,
    ])
  })

  it('exempts UI/derived/hub records with a stated reason', () => {
    const exemptKeys = new Set(TRAINER_EXEMPT_RECORDS.map((e) => e.suffix ?? e.fixedType))
    for (const k of ['-dismissed-failure', '-model-stats', '-exploration']) expect(exemptKeys.has(k)).toBe(true)
    for (const k of ['trainer-seen', 'trainer-queue', 'trainer-project', 'trainer-queue-order'])
      expect(exemptKeys.has(k)).toBe(true)
    for (const k of ['-evaluation', '-verdict', '-xai-narrative']) expect(exemptKeys.has(k)).toBe(true)
    for (const e of TRAINER_EXEMPT_RECORDS) expect(e.reason.length).toBeGreaterThan(0)
  })
})

describe('buildTrainerDataCapabilityManifest', () => {
  const m = buildTrainerDataCapabilityManifest('demo-run')

  it('resolves suffixes against the recordType into absolute types', () => {
    const types = new Map(m.types.map((t) => [t.type, t]))
    expect(types.has('demo-run')).toBe(true) // the run record
    expect(types.get('demo-run-environment')?.creatable).toBe(true)
    expect(types.get('demo-run-scorecard')?.editable).toBe(true)
    expect(types.get('demo-run-xai-suggestion')).toBeTruthy() // queryable suggestion
  })

  it('leaves generic activities empty — trainer launches go through the bespoke startTrainerActivity tool', () => {
    expect(m.activities).toEqual([])
  })

  it('never emits a type that is on the exempt list', () => {
    const exemptAbsolute = new Set(
      TRAINER_EXEMPT_RECORDS.map((e) => (e.suffix !== undefined ? `demo-run${e.suffix}` : e.fixedType)),
    )
    for (const t of m.types) expect(exemptAbsolute.has(t.type)).toBe(false)
  })
})
