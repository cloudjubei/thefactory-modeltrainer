// Single-purpose boot decision (A5). The viewer ships as BOTH a multi-project HUB and a single-purpose app a
// project embeds to open straight into its own dashboard. This module is the pure decision at the seam: given
// the host-injected boot config (`window.__TRAINER_BOOT__`) and whether we run embedded, it returns whether to
// render the hub (the unchanged default) or to boot ONE project — and, when single, the synthesized project the
// dashboard needs (`{ key, dir, name, manifestRelPath }`, the whole of `currentProject`) plus either the bundled
// manifest or a flag that it must be loaded by inspecting the project's own dir.
//
// It is deliberately pure and side-effect-free so it is node-tested directly (src/bootViewer.test.ts), the same
// dual-load pattern as hypothesis.js. app.js owns the effects (inspect-trainer, cache seeding, chrome).
;(function (root) {
  // A manifest is usable INLINE only if it carries the recordType the dashboard scopes every record by. A
  // manifest without it is treated as absent, so the app re-inspects the dir for a real one rather than
  // opening a dashboard that cannot read or write anything.
  function usableManifest(m) {
    return !!(m && typeof m === 'object' && typeof m.recordType === 'string' && m.recordType)
  }

  function nonEmpty(s) {
    return typeof s === 'string' && s.length > 0
  }

  function hub(error) {
    return { mode: 'hub', project: null, manifest: null, needsInspect: false, error: error || null }
  }

  // The dashboard's `currentProject` shape. manifestRelPath is only carried when the config sets it (a second
  // line in the same repo); it is omitted rather than set to undefined so the synthesized object equals a
  // hand-written one.
  function projectFrom(cfg, keyFallback, nameFallback) {
    const p = {
      key: nonEmpty(cfg.key) ? cfg.key : keyFallback,
      dir: nonEmpty(cfg.dir) ? cfg.dir : '.',
      name: nonEmpty(cfg.name) ? cfg.name : nameFallback,
    }
    if (nonEmpty(cfg.manifestRelPath)) p.manifestRelPath = cfg.manifestRelPath
    return p
  }

  // The one decision. Returns { mode:'hub'|'single', project, manifest, needsInspect, error }.
  function resolveBoot(bootConfig, env) {
    const embedded = !!(env && env.embedded)
    if (!bootConfig || bootConfig.mode !== 'single') return hub(null)

    const manifest = usableManifest(bootConfig.manifest) ? bootConfig.manifest : null
    if (manifest) {
      const project = projectFrom(bootConfig, manifest.recordType, manifest.name || 'Trainer')
      return { mode: 'single', project, manifest, needsInspect: false, error: null }
    }

    if (nonEmpty(bootConfig.dir)) {
      // No inline manifest — it has to be read from the project's own trainer.json, which only the host can
      // do. Standalone there is no host, so this configuration cannot open; fall back to the (usable) hub and
      // say why rather than boot into a dashboard with no manifest.
      if (!embedded) {
        return hub(
          'single-purpose boot needs the Overseer host to load the project manifest; showing the hub',
        )
      }
      const project = projectFrom(bootConfig, bootConfig.dir, bootConfig.name || 'Trainer')
      return { mode: 'single', project, manifest: null, needsInspect: true, error: null }
    }

    return hub('single-purpose boot config declared no manifest and no dir; showing the hub')
  }

  // Generate the source of a seeded project's boot.config.js (loaded before boot.js). It is checked through
  // resolveBoot FIRST and throws if the config would not open a single-purpose dashboard — a seed that would
  // silently leave the app on the hub (or a blank screen) is a bug, and it fails at seed time rather than in
  // the user's browser. The config is embedded via JSON.stringify, so any string in it is safely escaped.
  function renderBootConfig(cfg) {
    const check = resolveBoot(cfg, { embedded: true })
    if (check.mode !== 'single') {
      throw new Error(
        'renderBootConfig: config does not resolve to a single-purpose boot (' +
          (check.error || 'not single') +
          ')',
      )
    }
    return (
      '// Generated single-purpose boot config (A5 seed) — see scripts/seed-single-purpose.mjs.\n' +
      '// Base modeltrainer ships this file EMPTY, which leaves the viewer in multi-project hub mode.\n' +
      'window.__TRAINER_BOOT__ = ' +
      JSON.stringify(cfg) +
      '\n'
    )
  }

  const TrainerBoot = {
    resolveBoot: resolveBoot,
    usableManifest: usableManifest,
    renderBootConfig: renderBootConfig,
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = TrainerBoot
  if (root) root.TrainerBoot = TrainerBoot
})(typeof window !== 'undefined' ? window : null)
