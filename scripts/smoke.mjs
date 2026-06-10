// On-demand end-to-end smoke (see docs/TESTING.md): runs a real 2-config
// campaign against examples/cartpole through LocalComputeRunner. Not part of
// `npm test`. Usage: node scripts/smoke.mjs
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { LocalComputeRunner } from 'thefactory-tools'
import { createModelTrainerTools } from '../dist/index.js'

const repoRoot = dirname(dirname(fileURLToPath(import.meta.url)))
const projectRoot = join(repoRoot, 'examples', 'cartpole')

const rows = new Map()
const keyOf = (r) => `${r.scope}|${r.type}|${r.key ?? ''}`
const storage = {
  async upsertRecord(input) {
    const record = { ...input, key: input.key ?? null, createdAt: '', updatedAt: '' }
    rows.set(keyOf(input), record)
    return record
  },
  async readRecord(ref) {
    return rows.get(keyOf(ref))
  },
  async listRecords(query) {
    return [...rows.values()].filter(
      (r) => r.scope === query.scope && (!query.type || r.type === query.type),
    )
  },
  async deleteRecord(ref) {
    return rows.delete(keyOf(ref))
  },
}

const tools = createModelTrainerTools({
  computeRunner: new LocalComputeRunner(),
  storage,
  logger: {
    info: (m, meta) => console.log(`[info] ${m}`, meta ?? ''),
    warn: (m, meta) => console.warn(`[warn] ${m}`, meta ?? ''),
  },
})

const started = Date.now()
const result = await tools.runTrainingCampaign({
  scope: 'smoke',
  projectRoot,
  spec: { sweep: { learning_rate: [0.0003, 0.001] }, fixed: { total_timesteps: 2000 } },
  onProgress: (p) => console.log(`[progress]`, JSON.stringify(p)),
})

console.log(`\n[result]`, JSON.stringify(result, null, 2))
const records = await storage.listRecords({ scope: 'smoke', type: result.recordType })
console.log(`\n[records] ${records.length} run record(s):`)
for (const r of records) {
  const c = r.content
  console.log(
    `  ${r.key}  objective=${c.objective}  health=${c.health?.status}  lr=${c.config?.learning_rate}`,
  )
}
console.log(`\n[smoke] total wall: ${((Date.now() - started) / 1000).toFixed(1)}s`)
if (result.completed !== 2 || result.failed !== 0) {
  console.error('[smoke] FAILED — expected 2 completed runs')
  process.exit(1)
}
console.log('[smoke] PASS')
