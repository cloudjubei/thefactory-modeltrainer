import { createRequire } from 'module'
import Module from 'module'
import { readFileSync } from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
import { describe, it, expect } from 'vitest'

// viewer/boot.js is the no-build browser module that decides, at load, whether the viewer opens as the
// multi-project HUB (today's default) or boots straight into ONE project's dashboard (A5 single-purpose).
// Load it as CommonJS the same way hypothesisViewer.test.ts loads viewer/hypothesis.js, so the ACTUAL
// viewer decision logic is under test rather than a paraphrase of it.
const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))
const mpath = join(here, '..', 'viewer', 'boot.js')
const mod = new Module(mpath)
mod.filename = mpath
mod.paths = []
mod._compile(readFileSync(mpath, 'utf8'), mpath)
const B: any = mod.exports

const MANIFEST = { name: 'BlackSwan', recordType: 'blackswan-run', objective: { name: 'x', direction: 'max' } }
const emb = { embedded: true }

describe('resolveBoot — default is the hub, unchanged', () => {
  it('no boot config → hub', () => {
    expect(B.resolveBoot(undefined, emb).mode).toBe('hub')
    expect(B.resolveBoot(null, emb).mode).toBe('hub')
    expect(B.resolveBoot(undefined, emb).project).toBe(null)
  })
  it('a boot config that does not request single mode → hub', () => {
    expect(B.resolveBoot({}, emb).mode).toBe('hub')
    expect(B.resolveBoot({ mode: 'hub' }, emb).mode).toBe('hub')
    // a stray manifest without mode:'single' must NOT silently hijack the hub
    expect(B.resolveBoot({ manifest: MANIFEST }, emb).mode).toBe('hub')
  })
})

describe('resolveBoot — single-purpose with a bundled manifest', () => {
  const r = () => B.resolveBoot({ mode: 'single', manifest: MANIFEST }, emb)
  it('opens the dashboard directly, no inspect needed', () => {
    expect(r().mode).toBe('single')
    expect(r().needsInspect).toBe(false)
    expect(r().manifest).toEqual(MANIFEST)
    expect(r().error).toBe(null)
  })
  it('synthesizes the project the dashboard needs: key/dir/name', () => {
    const p = r().project
    expect(p.key).toBe('blackswan-run') // defaults to the manifest recordType
    expect(p.dir).toBe('.') // a bundled single-purpose app is rooted at the project itself
    expect(p.name).toBe('BlackSwan') // defaults to the manifest name
  })
  it('honours explicit key / name / dir / manifestRelPath overrides', () => {
    const p = B.resolveBoot(
      { mode: 'single', manifest: MANIFEST, key: 'k', name: 'N', dir: 'sub', manifestRelPath: '.factory/t.json' },
      emb,
    ).project
    expect(p).toEqual({ key: 'k', name: 'N', dir: 'sub', manifestRelPath: '.factory/t.json' })
  })
  it('a bundled manifest works even standalone (localStorage fallback, no host to inspect with)', () => {
    const res = B.resolveBoot({ mode: 'single', manifest: MANIFEST }, { embedded: false })
    expect(res.mode).toBe('single')
    expect(res.needsInspect).toBe(false)
  })
})

describe('resolveBoot — single-purpose that must inspect its own dir', () => {
  it('with only a dir, embedded: single + needsInspect, manifest deferred', () => {
    const res = B.resolveBoot({ mode: 'single', dir: '.' }, emb)
    expect(res.mode).toBe('single')
    expect(res.needsInspect).toBe(true)
    expect(res.manifest).toBe(null)
    expect(res.project.dir).toBe('.')
    expect(res.project.key).toBe('.') // no manifest yet ⇒ the dir is the stable key
  })
  it('a manifest MISSING recordType is not usable inline; falls back to inspecting the dir', () => {
    const res = B.resolveBoot({ mode: 'single', dir: '.', manifest: { name: 'x' } }, emb)
    expect(res.needsInspect).toBe(true)
    expect(res.manifest).toBe(null)
  })
})

describe('resolveBoot — misconfiguration is fail-SAFE (usable hub + a surfaced reason)', () => {
  it('single with neither a usable manifest nor a dir → hub, with an error to show', () => {
    const res = B.resolveBoot({ mode: 'single' }, emb)
    expect(res.mode).toBe('hub')
    expect(typeof res.error).toBe('string')
    expect(res.error.length).toBeGreaterThan(0)
  })
  it('single needing inspect but NOT embedded cannot load a manifest → hub, with an error', () => {
    const res = B.resolveBoot({ mode: 'single', dir: '.' }, { embedded: false })
    expect(res.mode).toBe('hub')
    expect(typeof res.error).toBe('string')
  })
})
