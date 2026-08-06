import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'
import {
  TRAINER_CAPABILITY_RECORD_TYPES,
  TRAINER_EXEMPT_RECORDS,
  TRAINER_EXEMPT_ACTIVITY_TYPES,
  TRAINER_LAUNCHABLE_ACTIVITY_TYPES,
  TRAINER_RUN_DELETE_CASCADE_SUFFIXES,
} from './trainerCapabilities.js'

// A2 parity audit: scrape the REAL viewer launch + write sites out of viewer/app.js and assert every one is
// covered by the capability spec — launchable-or-exempt for activities, declared-or-exempt for record types.
// This is the regression guard: adding a new startOrEnqueue('x') / OverseerBridge.startActivity('x') or a new
// putData/deleteData({type}) in the viewer WITHOUT declaring it here fails the build. The scrape resolves the
// top-of-file `const NAME = '...'` record constants so `manifest.recordType + SUFFIX_CONST` normalizes too.

const here = dirname(fileURLToPath(import.meta.url))
const appJs = readFileSync(join(here, '..', 'viewer', 'app.js'), 'utf8')

const RT = '<rt>'

/** Top-of-file record/type constants (`const NAME = 'value'`), for resolving concatenated type expressions. */
function readConstants(src: string): Map<string, string> {
  const out = new Map<string, string>()
  const re = /const\s+([A-Z][A-Z0-9_]*)\s*=\s*'([^']+)'/g
  for (let m = re.exec(src); m; m = re.exec(src)) out.set(m[1], m[2])
  return out
}
const CONSTS = readConstants(appJs)

/** The first string literal passed to `fn(` when it is the immediate first argument (else undefined). */
function launchTypes(src: string, fn: string): string[] {
  const out: string[] = []
  const re = new RegExp(fn.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\(', 'g')
  for (let m = re.exec(src); m; m = re.exec(src)) {
    const seg = src.slice(m.index + m[0].length, m.index + m[0].length + 200)
    const sm = /^[\s\n]*'([a-z][a-z0-9-]*)'/.exec(seg)
    if (sm) out.push(sm[1])
  }
  return out
}

/** Normalize a put/deleteData `type:` expression to a token: `<rt>`, `<rt>-suffix`, or a resolved constant. */
function normalizeType(expr: string): string {
  const e = expr.trim()
  let t: RegExpExecArray | null
  if ((t = /^`\$\{manifest\.recordType\}([^`]*)`$/.exec(e))) return RT + t[1]
  if ((t = /^`\$\{recordType\}([^`]*)`$/.exec(e))) return RT + t[1]
  if ((t = /^manifest\.recordType\s*\+\s*'([^']+)'$/.exec(e))) return RT + t[1]
  if ((t = /^recordType\s*\+\s*'([^']+)'$/.exec(e))) return RT + t[1]
  if ((t = /^manifest\.recordType\s*\+\s*([A-Z0-9_]+)$/.exec(e))) return RT + (CONSTS.get(t[1]) ?? '?' + t[1])
  if ((t = /^`\$\{manifest\.recordType\}\$\{([A-Z0-9_]+)\}`$/.exec(e))) return RT + (CONSTS.get(t[1]) ?? '?' + t[1])
  if (e === 'manifest.recordType' || e === 'recordType') return RT
  if ((t = /^([A-Z0-9_]+)$/.exec(e))) return CONSTS.get(t[1]) ?? '?' + t[1]
  return '??' + e
}

// Every record-type token written via putData/deleteData({ type: <expr>, ... }). The `type:` match is ANCHORED
// to the head of the object literal so a property-SHORTHAND write (`deleteData({ type, key })`, used by the
// run-delete cascade loop) does NOT silently borrow the next call's `type:` — it yields nothing and is covered
// instead by the trainerCapabilities twin test (the viewer iterates the pinned run-keyed cascade constant).
function writtenRecordTokens(src: string): Set<string> {
  const out = new Set<string>()
  const re = /(?:putData|deleteData)\(\s*\{/g
  for (let m = re.exec(src); m; m = re.exec(src)) {
    const seg = src.slice(m.index + m[0].length, m.index + m[0].length + 300)
    const tm = /^\s*type:\s*([^,\n]+)/.exec(seg)
    if (tm) out.add(normalizeType(tm[1].replace(/[,\s]+$/, '')))
  }
  return out
}

const usedActivities = new Set([
  ...launchTypes(appJs, 'startOrEnqueue'),
  ...launchTypes(appJs, 'launchPaperOp'),
  ...launchTypes(appJs, 'window.OverseerBridge.startActivity'),
])
const usedRecordTokens = writtenRecordTokens(appJs)

const coveredActivities = new Set([...TRAINER_LAUNCHABLE_ACTIVITY_TYPES, ...TRAINER_EXEMPT_ACTIVITY_TYPES])
const coveredRecordTokens = new Set([
  ...TRAINER_CAPABILITY_RECORD_TYPES.map((t) => t.fixedType ?? `${RT}${t.suffix ?? ''}`),
  ...TRAINER_EXEMPT_RECORDS.map((e) => e.fixedType ?? `${RT}${e.suffix ?? ''}`),
])

describe('A2 parity audit — viewer launch surface', () => {
  it('scraped a plausible number of launch sites (guards against a broken scraper)', () => {
    expect(usedActivities.size).toBeGreaterThanOrEqual(20)
    // Proof the scraper resolves all three launch paths + is not vacuous.
    for (const a of ['train', 'judge', 'mine-data', 'inspect-trainer', 'suggest-paper-hypotheses'])
      expect(usedActivities.has(a)).toBe(true)
  })

  it('every activity the viewer launches is launchable-by-chat or explicitly exempt', () => {
    const undeclared = [...usedActivities].filter((a) => !coveredActivities.has(a))
    expect(undeclared).toEqual([])
  })

  it('every launchable activity actually exists as a viewer launch (no dead declarations)', () => {
    const dead = [...TRAINER_LAUNCHABLE_ACTIVITY_TYPES].filter((a) => !usedActivities.has(a))
    expect(dead).toEqual([])
  })
})

describe('A2 parity audit — viewer record-write surface', () => {
  it('resolved every written type (no unresolved token — guards the normalizer)', () => {
    const unresolved = [...usedRecordTokens].filter((t) => t.startsWith('??') || t.includes('?'))
    expect(unresolved).toEqual([])
    expect(usedRecordTokens.size).toBeGreaterThanOrEqual(20)
    // Proof the normalizer resolves suffix, template, and constant forms.
    for (const t of ['<rt>', '<rt>-environment', 'trainer-project', 'trainer-queue'])
      expect(usedRecordTokens.has(t)).toBe(true)
  })

  it('every record type the viewer writes/deletes is declared-editable or explicitly exempt', () => {
    const undeclared = [...usedRecordTokens].filter((t) => !coveredRecordTokens.has(t))
    expect(undeclared).toEqual([])
  })

  it('the run-delete cascade suffixes are all classified (declared or exempt)', () => {
    for (const suffix of TRAINER_RUN_DELETE_CASCADE_SUFFIXES)
      expect(coveredRecordTokens.has(`${RT}${suffix}`)).toBe(true)
  })
})
