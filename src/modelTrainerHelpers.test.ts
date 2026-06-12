import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { hashTrainingConfig, readTrainerManifest, setupKeyOf } from './modelTrainerHelpers.js'

const tempDirs: string[] = []

async function makeProjectDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'modeltrainer-test-'))
  tempDirs.push(dir)
  return dir
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })))
})

const VALID_MANIFEST = {
  name: 'demo',
  recordType: 'demo-run',
  run: 'python -m trainer.run --config-json {configPath} --summary-out {summaryOut}',
  objective: { name: 'score', direction: 'max' },
  levers: { lr: { type: 'number', default: 0.01 } },
}

describe('hashTrainingConfig', () => {
  it('returns a 12-char lowercase hex string', () => {
    expect(hashTrainingConfig({ a: 1 })).toMatch(/^[0-9a-f]{12}$/)
  })

  it('is deterministic', () => {
    expect(hashTrainingConfig({ a: 1, b: 'x' })).toBe(hashTrainingConfig({ a: 1, b: 'x' }))
  })

  it('is insensitive to key order', () => {
    expect(hashTrainingConfig({ a: 1, b: 2 })).toBe(hashTrainingConfig({ b: 2, a: 1 }))
  })

  it('differs for different configs', () => {
    expect(hashTrainingConfig({ a: 1 })).not.toBe(hashTrainingConfig({ a: 2 }))
  })
})

describe('setupKeyOf', () => {
  it('is identical for the same setup under different (or absent) seeds', () => {
    const base = setupKeyOf({ lr: 0.1, steps: 100, seed: 0 })
    expect(setupKeyOf({ lr: 0.1, steps: 100, seed: 7 })).toBe(base)
    expect(setupKeyOf({ lr: 0.1, steps: 100 })).toBe(base)
  })

  it('differs when a non-seed lever changes', () => {
    expect(setupKeyOf({ lr: 0.1, seed: 0 })).not.toBe(setupKeyOf({ lr: 0.2, seed: 0 }))
  })
})

describe('readTrainerManifest', () => {
  it('reads and validates a manifest from .factory/trainer.json', async () => {
    const root = await makeProjectDir()
    await mkdir(join(root, '.factory'), { recursive: true })
    await writeFile(join(root, '.factory', 'trainer.json'), JSON.stringify(VALID_MANIFEST))
    const manifest = await readTrainerManifest(root)
    expect(manifest.recordType).toBe('demo-run')
  })

  it('reads a second line from a custom manifestRelPath in the same repo', async () => {
    const root = await makeProjectDir()
    await mkdir(join(root, '.factory'), { recursive: true })
    await writeFile(join(root, '.factory', 'trainer.json'), JSON.stringify(VALID_MANIFEST))
    await writeFile(
      join(root, '.factory', 'trainer-dip.json'),
      JSON.stringify({ ...VALID_MANIFEST, recordType: 'demo-dip-run' }),
    )
    const dip = await readTrainerManifest(root, '.factory/trainer-dip.json')
    expect(dip.recordType).toBe('demo-dip-run')
    const trading = await readTrainerManifest(root)
    expect(trading.recordType).toBe('demo-run')
  })

  it('throws a descriptive error when the file is missing', async () => {
    const root = await makeProjectDir()
    await expect(readTrainerManifest(root)).rejects.toThrow(/trainer\.json/)
  })

  it('throws on malformed JSON', async () => {
    const root = await makeProjectDir()
    await mkdir(join(root, '.factory'), { recursive: true })
    await writeFile(join(root, '.factory', 'trainer.json'), '{not json')
    await expect(readTrainerManifest(root)).rejects.toThrow()
  })

  it('throws when the manifest fails validation', async () => {
    const root = await makeProjectDir()
    await mkdir(join(root, '.factory'), { recursive: true })
    await writeFile(join(root, '.factory', 'trainer.json'), JSON.stringify({ name: 'x' }))
    await expect(readTrainerManifest(root)).rejects.toThrow(/recordType/)
  })
})
