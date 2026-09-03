// A5 SEED. Stamp a single-purpose trainer app into a target project ONCE: copy this repo's viewer/ into the
// project's app dir and drop a validated boot.config.js so that copy boots straight into the project's own
// dashboard instead of the multi-project hub. The project OWNS the copy afterwards — there is no update path
// back to modeltrainer by design (docs/implementation-plan.md A5).
//
// Usage:
//   node scripts/seed-single-purpose.mjs --target <projectDir> [--app-dir app] \
//        [--name <label>] [--project-dir .] [--manifest-rel-path .factory/trainer.json]
//
// The default boot config uses the `dir`/`manifestRelPath` (inspect) path, so the seeded app loads the
// project's REAL manifest fresh at boot rather than a bundled snapshot that could drift.
import { createRequire } from 'module'
import Module from 'module'
import { readFileSync, cpSync, writeFileSync, existsSync, mkdirSync } from 'fs'
import { dirname, join, resolve } from 'path'
import { fileURLToPath } from 'url'

const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const VIEWER = resolve(here, '..', 'viewer')

// Load the browser boot module as CommonJS to reuse renderBootConfig — the SAME validator the tests pin, so
// the seed cannot emit a config that fails to open single.
const bpath = join(VIEWER, 'boot.js')
const bm = new Module(bpath)
bm.filename = bpath
bm.paths = []
bm._compile(readFileSync(bpath, 'utf8'), bpath)
const { renderBootConfig } = bm.exports

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name)
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : dflt
}

const target = arg('target')
if (!target) {
  console.error('seed-single-purpose: --target <projectDir> is required')
  process.exit(1)
}
const appDir = arg('app-dir', 'app')
const projectDir = arg('project-dir', '.')
const manifestRelPath = arg('manifest-rel-path', '.factory/trainer.json')
const name = arg('name', undefined)

const dest = resolve(target, appDir)
if (!existsSync(target)) {
  console.error(`seed-single-purpose: target ${target} does not exist`)
  process.exit(1)
}
mkdirSync(dest, { recursive: true })

// Copy the whole viewer verbatim. The project owns this copy from here on.
cpSync(VIEWER, dest, { recursive: true })

const cfg = { mode: 'single', dir: projectDir, manifestRelPath }
if (name) cfg.name = name
// BUNDLE the manifest so the app boots straight into the dashboard SYNCHRONOUSLY — no inspect round-trip, so
// the hub never flashes and there is no blank loading gap. The dir/manifestRelPath stay in the config, so the
// viewer still re-inspects in the background on open and self-heals if trainer.json changed since the seed.
// If the manifest can't be read, fall back to the inspect path (config without an inline manifest).
const manifestPath = resolve(target, manifestRelPath)
let bundled = false
if (existsSync(manifestPath)) {
  try {
    cfg.manifest = JSON.parse(readFileSync(manifestPath, 'utf8'))
    bundled = true
  } catch (e) {
    console.warn(`could not parse ${manifestPath} (${e.message}); falling back to inspect-on-boot`)
  }
}
// Throws (and aborts the seed) if the config would not open single — a loud failure beats a bricked app.
writeFileSync(join(dest, 'boot.config.js'), renderBootConfig(cfg))

console.log(`seeded single-purpose viewer → ${dest}`)
console.log(
  `boot: ${bundled ? 'bundled manifest (instant, self-refreshing)' : 'inspect-on-boot'} — ${JSON.stringify({ ...cfg, manifest: bundled ? '<' + (cfg.manifest.recordType || 'manifest') + '>' : undefined })}`,
)
console.log('')
console.log('Final wiring (overseer project metadata, NOT a repo file):')
console.log(`  metadata.hasApp = true`)
console.log(`  metadata.appDir = "${appDir}"`)
