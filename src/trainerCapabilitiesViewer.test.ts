import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'
import {
  buildTrainerDataCapabilityManifest,
  TRAINER_CAPABILITY_RECORD_TYPES,
  TRAINER_RUN_KEYED_CHILD_SUFFIXES,
} from './trainerCapabilities.js'

// viewer/trainerCapabilities.js is the no-build browser twin of the record-type half of the engine's
// trainerCapabilities.ts; load it as CommonJS (same mechanism as scorecardViewer.test.ts) and assert it
// builds the SAME data-capability manifest, so the viewer's reported capabilities, the backend tool's
// declarations, and the parity audit can never drift.
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'trainerCapabilities.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const JS: any = mod.exports

describe('viewer trainerCapabilities twin', () => {
  it('builds a byte-identical manifest to the engine for a sample recordType', () => {
    expect(JS.buildTrainerDataCapabilityManifest('demo-run')).toEqual(
      buildTrainerDataCapabilityManifest('demo-run'),
    )
  })

  it('builds identically for a different recordType (suffix resolution matches)', () => {
    expect(JS.buildTrainerDataCapabilityManifest('blackswan-run')).toEqual(
      buildTrainerDataCapabilityManifest('blackswan-run'),
    )
  })

  it('mirrors the engine record-type declarations verbatim', () => {
    expect(JS.TRAINER_CAPABILITY_RECORD_TYPES).toEqual(TRAINER_CAPABILITY_RECORD_TYPES)
  })

  it('mirrors the engine run-delete cascade verbatim (viewer deleteRelatedRunRecords iterates this)', () => {
    expect(JS.TRAINER_RUN_KEYED_CHILD_SUFFIXES).toEqual([...TRAINER_RUN_KEYED_CHILD_SUFFIXES])
  })
})
