/*
 * Exploration view — the config-space search map, in-app and always available per project.
 *
 * Renders the `{recordType}-exploration` autopilot state (stage, basins, regret, decision log) plus a live
 * COVERAGE HEATMAP binned from the project's actual runs. Axes are chosen by lever IMPORTANCE (via the
 * parity-tested window.Xai engine, falling back to variance), both numeric and categorical (algo/net_arch),
 * with a lever picker + peg controls so you can hold one lever (e.g. net_arch="64,64") and vary the rest.
 * Empty cells read as unexplored gaps, so "how well are we covering the space" is legible at a glance.
 *
 * window.Exploration.render(container, data, actions)
 *   data    = { manifest, state|null, runs:[{config,objective,metrics,status}], activity:{status}|null }
 *   actions = { onLaunch(budget), onExploreMore(budget), onPause(), onResume(), onAbort() }
 *
 * Exposed as window.Exploration in the browser and module.exports under CommonJS (pure analyze/magma tested).
 */
;(function (root) {
  'use strict'

  const STAGES = ['calibrate', 'screen', 'global', 'local', 'converged']
  const STAGE_HELP = {
    calibrate:
      "Measures the objective's run-to-run noise on the default config across several seeds — the bar for deciding whether a later difference is real, not luck.",
    screen:
      'Samples the whole space to find which levers actually move the objective. The ones that barely matter are frozen at their best value so the search budget isn’t wasted on them.',
    global:
      'Hunts for distinct basins — regions of the space that clearly beat the baseline. Keeps proposing new points until no new basin appears for two rounds.',
    local:
      'Climbs each basin to its peak: finer sweeps around the best config plus extra seeds, to tell a genuine peak from a lucky seed.',
    converged:
      'Done — the global maximum is declared: the best basin that is robust across seeds and no longer improves beyond the noise floor.',
  }
  const BUDGET_HELP = {
    maxRuns: 'Most training runs the autopilot may spend before it stops. Leave blank to run until it converges on its own.',
    maxConcurrent: 'How many training runs to launch at once. The host’s RAM/CPU caps still apply on top of this.',
    targetObjective:
      'The known best score, if you have one (e.g. CartPole solved = 500 reward). Reaching it stops the search early. Leave blank if unknown — for most projects the objective is the run’s reward/return.',
  }
  const BASE_GRID = 24
  const MAX_CAT_BINS = 14
  // A search tries a DISCRETE set of values per lever, so each heatmap cell is a CONCRETE tried value (not a
  // range bin). Cap the distinct values shown per axis; beyond this a lever is too finely sampled to grid.
  const MAX_AXIS_VALUES = 80
  const MAG = [
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 253, 191],
  ]
  function magma(t) {
    t = Math.max(0, Math.min(1, t))
    const p = t * (MAG.length - 1)
    const i = Math.floor(p)
    const f = p - i
    const a = MAG[i]
    const b = MAG[Math.min(i + 1, MAG.length - 1)]
    return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(a[2] + (b[2] - a[2]) * f)})`
  }
  const esc = (s) =>
    String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c])
  const num = (v) => (typeof v === 'number' && isFinite(v) ? v : (isFinite(Number(v)) && v !== '' && v !== null ? Number(v) : NaN))
  const fmt = (v, d = 0) => (isFinite(v) ? Number(v).toFixed(d) : '—')
  const isNumericValue = (v) => typeof v === 'number' || (typeof v === 'string' && v !== '' && isFinite(Number(v)))

  // Per-project view state (chosen axes / pegs / zoom) — survives the live re-render poll.
  const view = {}
  // Terminal-log stickiness: auto-scroll the decision log to the newest line UNLESS the user scrolled up.
  let logStick = true
  let ctx = null // { container, data, actions } — cached so a selector change can re-render locally

  let stylesInjected = false
  function injectStyles() {
    if (stylesInjected || typeof document === 'undefined') return
    stylesInjected = true
    const css = `
    .expl{--e-line:#e2e7ef;--e-panel:#fff;--e-panel2:#f6f8fb;--e-text:#141d29;--e-muted:#5c6a7e;--e-faint:#8a97a9;--e-gold:#c8891a;--e-ice:#2b7fb8;--e-ok:#12936a;--e-mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;color:var(--e-text)}
    @media (prefers-color-scheme:dark){.expl{--e-line:#1f2a3b;--e-panel:#111725;--e-panel2:#0d121d;--e-text:#dbe4f1;--e-muted:#8492a8;--e-faint:#5c6a80;--e-gold:#f4b740;--e-ice:#8fd6ff;--e-ok:#4bd4a0}}
    .expl{padding:20px 22px 60px;font-feature-settings:"tnum" 1}
    .expl h2{font-size:17px;margin:0 0 4px;font-weight:640}
    .expl h3{font-size:12.5px;margin:0 0 12px;font-weight:620}
    .expl .lede{color:var(--e-muted);font-size:13px;max-width:76ch;margin:0 0 18px}
    .expl-status{background:var(--e-panel);border:1px solid var(--e-line);border-radius:12px;padding:16px 18px;margin:0 0 20px}
    .expl-status.live{border-color:color-mix(in srgb,var(--e-gold) 45%,var(--e-line))}
    .expl-status .row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
    .expl-status .title{font-size:14px;font-weight:640;display:flex;align-items:center;gap:9px}
    .expl-status .pulse{width:9px;height:9px;border-radius:50%;background:var(--e-gold);box-shadow:0 0 0 0 color-mix(in srgb,var(--e-gold) 60%,transparent);animation:expl-pulse 1.8s infinite}
    @keyframes expl-pulse{0%{box-shadow:0 0 0 0 color-mix(in srgb,var(--e-gold) 55%,transparent)}70%{box-shadow:0 0 0 8px transparent}100%{box-shadow:0 0 0 0 transparent}}
    @media (prefers-reduced-motion:reduce){.expl-status .pulse{animation:none}}
    .expl-status .now{font-family:var(--e-mono);font-size:12.5px;color:var(--e-muted);margin:8px 0 0}
    .expl-status .now b{color:var(--e-text);font-weight:600}
    .expl-status .now.sub{font-size:11.5px;color:var(--e-faint);margin-top:5px}
    .expl-logwrap{margin:10px 0 0;padding-top:8px;border-top:1px solid var(--e-line);max-height:150px;overflow-y:auto;overflow-x:hidden;scrollbar-width:thin}
    .expl-log{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:0}
    .expl-log li{display:flex;gap:10px;font-family:var(--e-mono);font-size:11.5px;padding:4px 2px;color:var(--e-muted);border-top:1px solid color-mix(in srgb,var(--e-line) 45%,transparent)}
    .expl-log li:first-child{border-top:none}
    .expl-log li .st{color:var(--e-gold);min-width:74px;flex:none}
    .expl-log li .ms{flex:1;min-width:0}
    .expl-log li .rn{color:var(--e-faint);min-width:52px;text-align:right;flex:none}
    .expl-status .now .bl{color:var(--e-gold)}
    .expl-controls{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end}
    .expl-field{display:flex;flex-direction:column;gap:4px;font-family:var(--e-mono);font-size:11px;color:var(--e-muted)}
    .expl-field .lbl{display:inline-flex;align-items:center;gap:5px}
    .expl-field input,.expl-sel select{padding:7px 9px;border:1px solid var(--e-line);border-radius:7px;background:var(--e-panel);color:var(--e-text);font-family:var(--e-mono);font-size:12.5px}
    .expl-field input{width:118px}
    .expl-btn{font:inherit;font-family:var(--e-mono);font-size:12.5px;padding:8px 15px;border-radius:8px;border:1px solid var(--e-line);background:var(--e-panel);color:var(--e-text);cursor:pointer}
    .expl-btn.primary{background:var(--e-gold);border-color:transparent;color:#1a1204;font-weight:600}
    .expl-btn:disabled{opacity:.5;cursor:default}
    .expl-btn:focus-visible,.expl-ico:focus-visible,.expl-open:focus-visible,.expl-sel select:focus-visible,.expl-field input:focus-visible{outline:2px solid var(--e-ice);outline-offset:2px}
    .expl-ctrls{display:flex;align-items:center;gap:8px}
    .expl-ico{display:inline-flex;align-items:center;justify-content:center;width:34px;height:34px;border-radius:8px;border:1px solid var(--e-line);background:var(--e-panel);color:var(--e-text);font-size:12px;line-height:1;cursor:pointer;font-family:var(--e-mono)}
    .expl-ico:hover{border-color:var(--e-muted)}
    .expl-ico.primary{background:var(--e-gold);border-color:transparent;color:#1a1204}
    .spin{width:12px;height:12px;border-radius:50%;border:2px solid color-mix(in srgb,var(--e-gold) 30%,transparent);border-top-color:var(--e-gold);display:inline-block;animation:expl-spin .8s linear infinite}
    @keyframes expl-spin{to{transform:rotate(360deg)}}
    @media (prefers-reduced-motion:reduce){.spin{animation:none}}
    .expl-lede{color:var(--e-muted);font-size:12.5px;line-height:1.5;max-width:74ch;margin:12px 0 0}
    .expl-open{font:inherit;font-family:var(--e-mono);font-size:11px;padding:3px 8px;border-radius:6px;border:1px solid var(--e-line);background:var(--e-panel);color:var(--e-ice);cursor:pointer}
    .expl-open:hover{border-color:var(--e-ice)}
    .expl-status .expl-readouts{margin:15px 0 0}
    .expl-status .expl-stages{margin:12px 0 0}
    .expl-readouts{display:flex;flex-wrap:wrap;gap:12px 30px;font-family:var(--e-mono);margin:0 0 18px}
    .expl-ro{display:flex;flex-direction:column;gap:2px}
    .expl-ro .k{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--e-faint)}
    .expl-ro .v{font-size:18px;font-weight:600}
    .expl-ro .v.ok{color:var(--e-ok)}.expl-ro .v.gold{color:var(--e-gold)}
    .expl-stages{display:flex;flex-wrap:wrap;gap:7px;margin:0 0 22px;padding:0;list-style:none}
    .expl-stages li{flex:1 1 130px;display:flex;align-items:center;gap:9px;padding:9px 12px;border:1px solid var(--e-line);border-radius:8px;background:var(--e-panel);font-family:var(--e-mono);font-size:12px}
    .expl-stages li .n{width:18px;height:18px;border-radius:50%;display:grid;place-items:center;font-size:10px;background:var(--e-panel2);color:var(--e-muted);border:1px solid var(--e-line)}
    .expl-stages li.done{border-color:color-mix(in srgb,var(--e-ok) 40%,var(--e-line))}
    .expl-stages li.done .n{background:var(--e-ok);color:#04120c;border-color:transparent}
    .expl-stages li.cur{border-color:color-mix(in srgb,var(--e-gold) 55%,var(--e-line))}
    .expl-mapbar{display:flex;align-items:center;gap:10px 14px;flex-wrap:wrap;margin:2px 0 12px}
    .expl-sel{display:flex;align-items:center;gap:6px;font-family:var(--e-mono);font-size:11px;color:var(--e-muted)}
    .expl-sel select{padding:5px 8px;font-size:12px}
    .expl-zoom{display:inline-flex;border:1px solid var(--e-line);border-radius:7px;overflow:hidden;margin-left:auto}
    .expl-zoom button{font:inherit;font-family:var(--e-mono);font-size:13px;border:none;background:var(--e-panel);color:var(--e-text);padding:5px 11px;cursor:pointer}
    .expl-zoom button+button{border-left:1px solid var(--e-line)}
    .expl-legend{display:flex;align-items:center;gap:8px;font-family:var(--e-mono);font-size:11px;color:var(--e-muted)}
    .expl-legend canvas{border-radius:3px;border:1px solid var(--e-line);display:block}
    .expl-mapcard{background:var(--e-panel);border:1px solid var(--e-line);border-radius:11px;padding:14px}
    .expl-mapscroll{max-height:560px;overflow:auto;border-radius:7px;border:1px solid var(--e-line)}
    .expl-mapcanvas{display:block}
    .expl-axis-y{writing-mode:vertical-rl;transform:rotate(180deg);font-family:var(--e-mono);font-size:10.5px;color:var(--e-faint);text-align:center}
    .expl-maprow{display:grid;grid-template-columns:auto 1fr;gap:8px;align-items:stretch}
    .expl-axis-x{text-align:center;font-family:var(--e-mono);font-size:10.5px;color:var(--e-faint);margin-top:6px}
    .expl-mapmain{position:relative}
    .expl-mapcanvas{cursor:crosshair;display:block}
    .expl-tip{position:absolute;z-index:20;pointer-events:none;max-width:270px;background:var(--e-panel);border:1px solid var(--e-line);border-radius:8px;padding:8px 10px;box-shadow:0 8px 24px rgba(0,0,0,.35);font-family:var(--e-mono);font-size:11px;color:var(--e-muted)}
    .expl-tip .th{color:var(--e-text);font-weight:600;margin-bottom:3px}
    .expl-tip .tr{color:var(--e-faint);margin-bottom:5px}
    .expl-tip .tr .tw,.expl-tip .tl b{color:var(--e-gold)}
    .expl-tip .tl{padding:2px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .expl-tip .tl b{font-weight:600;margin-right:5px}
    .expl-tip .tm{color:var(--e-faint);margin-top:3px}
    .expl-grid2{display:grid;grid-template-columns:1.1fr 1fr;gap:18px;margin-top:22px}
    @media (max-width:860px){.expl-grid2{grid-template-columns:1fr}}
    .expl-card{background:var(--e-panel);border:1px solid var(--e-line);border-radius:11px;padding:16px 18px}
    .expl-card canvas{width:100%;height:auto;display:block}
    .expl-card .cap{color:var(--e-muted);font-size:12px;margin:10px 0 0}
    table.expl-basins{width:100%;border-collapse:collapse;font-family:var(--e-mono);font-size:12px}
    table.expl-basins th{text-align:left;font-weight:500;color:var(--e-faint);font-size:10px;letter-spacing:.06em;text-transform:uppercase;padding:0 8px 8px 0;border-bottom:1px solid var(--e-line)}
    table.expl-basins td{padding:9px 8px 9px 0;border-bottom:1px solid var(--e-line);white-space:nowrap}
    table.expl-basins tr:last-child td{border-bottom:none}
    table.expl-basins .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}
    .expl-badge{font-family:var(--e-mono);font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;padding:2px 6px;border-radius:20px;background:color-mix(in srgb,var(--e-gold) 20%,transparent);color:var(--e-gold);border:1px solid color-mix(in srgb,var(--e-gold) 45%,transparent)}
    .expl-empty{color:var(--e-muted);padding:26px 0;font-size:13px;text-align:center}
    .expl-note{font-family:var(--e-mono);font-size:11.5px;color:var(--e-faint);margin-top:8px}
    .expl-err{color:#e0644f;font-family:var(--e-mono);font-size:11.5px;margin-top:8px}
    `
    const s = document.createElement('style')
    s.id = 'exploration-styles'
    s.textContent = css
    document.head.appendChild(s)
  }

  // --- analysis ---------------------------------------------------------------------------------

  function rankLevers(manifest, runs) {
    const levers = manifest.levers || {}
    const searchable = Object.keys(levers).filter((l) => l !== 'seed' && ((levers[l].scope || 'model') === 'model'))
    const kindOf = (l) => (levers[l] && levers[l].type === 'number' ? 'num' : 'cat')
    const distinct = (l) => new Set(runs.map((r) => String(r.config[l])).filter((v) => v && v !== 'undefined')).size

    // Importance via the parity-tested Xai engine when available (numeric + categorical), else variance.
    let importance = {}
    if (root && root.Xai && typeof root.Xai.leverImportances === 'function' && runs.length) {
      try {
        const crit = { key: 'objective', direction: (manifest.objective && manifest.objective.direction) || 'max', label: 'objective' }
        for (const imp of root.Xai.leverImportances(runs, crit) || []) importance[imp.lever] = imp.importance
      } catch {
        importance = {}
      }
    }
    const varOf = (l) => {
      const vals = runs.map((r) => num(r.config[l])).filter((v) => isFinite(v))
      if (vals.length < 2) return 0
      const m = vals.reduce((a, b) => a + b, 0) / vals.length
      return vals.reduce((a, b) => a + (b - m) ** 2, 0) / vals.length
    }
    const hasImp = Object.keys(importance).length > 0
    const ranked = searchable
      .map((l) => ({ lever: l, kind: kindOf(l), values: distinct(l), score: hasImp ? importance[l] || 0 : varOf(l) }))
      .filter((r) => r.values > 1) // a lever with one observed value can't be an axis
      .sort((a, b) => b.score - a.score)
    return ranked
  }

  function analyze(data) {
    const manifest = data.manifest || { levers: {}, objective: { direction: 'max' } }
    const dir = (manifest.objective && manifest.objective.direction) || 'max'
    const runs = (data.runs || []).filter((r) => r && r.config && isFinite(num(r.objective)))
    const ranked = rankLevers(manifest, runs)
    const rt = manifest.recordType || '_'
    const vs = view[rt] || (view[rt] = { axisX: null, axisY: null, pegs: {}, zoom: 1 })

    // default axes = top-2 ranked levers (prefer numeric for a continuous surface, but allow categorical)
    const rankedKeys = ranked.map((r) => r.lever)
    if (!vs.axisX || !rankedKeys.includes(vs.axisX)) vs.axisX = rankedKeys[0] || null
    if (!vs.axisY || !rankedKeys.includes(vs.axisY) || vs.axisY === vs.axisX)
      vs.axisY = rankedKeys.find((k) => k !== vs.axisX) || null

    // objective orientation → color scale (hot = better)
    const objs = runs.map((r) => num(r.objective)).filter(isFinite)
    const oMin = objs.length ? Math.min(...objs) : 0
    const oMax = objs.length ? Math.max(...objs) : 1
    const nrm = (v) => {
      if (oMax === oMin) return 0.5
      const t = (v - oMin) / (oMax - oMin)
      return dir === 'max' ? t : 1 - t
    }
    return { manifest, dir, runs, ranked, rankedKeys, vs, nrm, oMin, oMax, basins: (data.state && data.state.basins) || [] }
  }

  // Build an axis descriptor: every cell is a CONCRETE tried value (numeric axes sort numerically + label
  // with the value; categorical axes sort as strings). Not range bins — a config-space search probes a
  // discrete set of values, so "0.004–0.004" ranges read as noise; the actual values are what matter.
  function makeAxis(lever, runs, manifest) {
    const levers = manifest.levers || {}
    const spec = levers[lever] || {}
    const vals = runs.map((r) => r.config[lever]).filter((v) => v !== undefined && v !== null)
    const numeric = spec.type === 'number' || (vals.length && vals.every(isNumericValue))
    const distinct = numeric
      ? [...new Set(vals.map(Number).filter(isFinite))].sort((a, b) => a - b).slice(0, MAX_AXIS_VALUES)
      : [...new Set(vals.map(String))].sort().slice(0, MAX_AXIS_VALUES)
    const key = (v) => (numeric ? String(Number(v)) : String(v))
    const idx = new Map(distinct.map((v, i) => [key(v), i]))
    const n = Math.max(1, distinct.length)
    return {
      lever,
      kind: numeric ? 'num' : 'cat',
      numeric,
      n,
      distinct,
      index: (v) => (idx.has(key(v)) ? idx.get(key(v)) : -1),
      labels: distinct.map((v) => (numeric ? fmtVal(v) : String(v))),
      coordOf: (v) => (idx.has(key(v)) ? (idx.get(key(v)) + 0.5) / n : -1),
      cellLabel: (i) => (numeric ? fmtVal(distinct[i]) : String(distinct[i])),
    }
  }

  // Bin the (peg-filtered) runs into the X×Y grid, KEEPING every run per cell (not just the best) so the
  // heatmap can subdivide a busy cell + a hover can list the exact configs behind a square. Pure + tested.
  function heatmapCells(a) {
    const xA = makeAxis(a.vs.axisX, a.runs, a.manifest)
    const yA = makeAxis(a.vs.axisY, a.runs, a.manifest)
    const pegs = a.vs.pegs || {}
    const runs = a.runs.filter((r) =>
      Object.keys(pegs).every((l) => pegs[l] == null || String(r.config[l]) === String(pegs[l])),
    )
    const cells = new Array(xA.n * yA.n)
    for (let k = 0; k < cells.length; k++) cells[k] = { i: k % xA.n, j: Math.floor(k / xA.n), runs: [], best: null }
    for (const r of runs) {
      const gi = xA.index(r.config[a.vs.axisX])
      const gj = yA.index(r.config[a.vs.axisY])
      if (gi < 0 || gj < 0) continue
      const o = num(r.objective)
      const t = a.nrm(o)
      const c = cells[gj * xA.n + gi]
      c.runs.push({ config: r.config, objective: o, key: r.key, t: isFinite(t) ? t : 0 })
      if (c.best === null || t > c.best) c.best = t
    }
    for (const c of cells) c.runs.sort((p, q) => q.t - p.t) // hottest-first so the mosaic + tooltip read best→worst
    return { xA, yA, cells }
  }

  // --- drawing ----------------------------------------------------------------------------------

  function drawHeatmap(container, a) {
    const wrap = container.querySelector('[data-expl-map]')
    if (!wrap || !a.vs.axisX || !a.vs.axisY) return
    const { xA, yA, cells } = heatmapCells(a)
    const pegs = a.vs.pegs || {}

    const canvas = wrap.querySelector('canvas')
    const dpr = Math.min(2, (typeof window !== 'undefined' && window.devicePixelRatio) || 1)
    const zoom = a.vs.zoom || 1
    const cellPx = Math.max(16, Math.round(26 * zoom))
    // Reserve a margin for axis value ticks — left of the rows (y) and under the columns (x).
    const mL = 60
    const mB = 30
    const mT = 6
    const mR = 8
    const gridW = xA.n * cellPx
    const gridH = yA.n * cellPx
    // No clamp: the canvas lives in a scroll container, so it takes its natural size and scrolls rather
    // than clipping cells off the right/bottom edge.
    const W = mL + gridW + mR
    const H = mT + gridH + mB
    canvas.style.width = W + 'px'
    canvas.style.height = H + 'px'
    canvas.width = W * dpr
    canvas.height = H * dpr
    const x = canvas.getContext('2d')
    x.scale(dpr, dpr)
    const cs = getComputedStyle(container.querySelector('.expl'))
    const empty = (cs.getPropertyValue('--e-panel2') || '#0d121d').trim()
    const faint = (cs.getPropertyValue('--e-faint') || '#8a97a9').trim()
    const mono = cs.fontFamily || 'monospace'
    const CX = (i) => mL + i * cellPx // grid-left of column i
    const CY = (jScreen) => mT + jScreen * cellPx // grid-top of screen-row jScreen (0 = top)

    for (let jScreen = 0; jScreen < yA.n; jScreen++) {
      const cj = yA.n - 1 - jScreen // y points UP: top screen row = highest tried value
      for (let i = 0; i < xA.n; i++) {
        const c = cells[cj * xA.n + i]
        const px = CX(i)
        const py = CY(jScreen)
        if (!c.runs.length) {
          x.fillStyle = empty
          x.fillRect(px, py, cellPx + 0.6, cellPx + 0.6)
          continue
        }
        if (c.runs.length === 1) {
          x.fillStyle = magma(c.runs[0].t)
          x.fillRect(px, py, cellPx + 0.6, cellPx + 0.6)
          continue
        }
        // Multiple configs share this X/Y square (they differ on OTHER levers) — subdivide into a mosaic so
        // the spread of results inside one cell is visible instead of collapsed to a single best colour.
        const sub = Math.ceil(Math.sqrt(c.runs.length))
        const sw = cellPx / sub
        for (let s = 0; s < sub * sub; s++) {
          const run = c.runs[s]
          x.fillStyle = run ? magma(run.t) : empty
          x.fillRect(px + (s % sub) * sw, py + Math.floor(s / sub) * sw, sw + 0.5, sw + 0.5)
        }
      }
    }
    // grid lines
    x.strokeStyle = 'rgba(128,140,160,.16)'
    x.lineWidth = 1
    for (let g = 0; g <= xA.n; g++) { x.beginPath(); x.moveTo(CX(g), mT); x.lineTo(CX(g), mT + gridH); x.stroke() }
    for (let g = 0; g <= yA.n; g++) { x.beginPath(); x.moveTo(mL, mT + g * cellPx); x.lineTo(mL + gridW, mT + g * cellPx); x.stroke() }

    // axis value ticks — concrete values under each column / left of each row, thinned so they never crowd
    x.fillStyle = faint
    x.font = `10px ${mono}`
    const xStep = Math.max(1, Math.ceil(xA.n / Math.max(1, Math.floor(gridW / 46))))
    x.textAlign = 'center'
    x.textBaseline = 'top'
    for (let i = 0; i < xA.n; i += xStep) x.fillText(String(xA.labels[i]), CX(i) + cellPx / 2, mT + gridH + 6, cellPx * xStep - 2)
    const yStep = Math.max(1, Math.ceil(yA.n / Math.max(1, Math.floor(gridH / 20))))
    x.textAlign = 'right'
    x.textBaseline = 'middle'
    for (let jScreen = 0; jScreen < yA.n; jScreen += yStep) {
      const cj = yA.n - 1 - jScreen
      x.fillText(String(yA.labels[cj]), mL - 6, CY(jScreen) + cellPx / 2, mL - 10)
    }

    // peak crosshairs for basins whose center matches the current pegs
    for (const b of a.basins) {
      if (!b.centerConfig) continue
      if (!Object.keys(pegs).every((l) => pegs[l] == null || String(b.centerConfig[l]) === String(pegs[l]))) continue
      const gi = xA.index(b.centerConfig[a.vs.axisX])
      const gj = yA.index(b.centerConfig[a.vs.axisY])
      if (gi < 0 || gj < 0) continue
      const px = CX(gi) + cellPx / 2
      const py = CY(yA.n - 1 - gj) + cellPx / 2
      const isG = ctx && ctx.data.state && b.id === ctx.data.state.declaredBasinId
      x.strokeStyle = isG ? '#f4b740' : '#8fd6ff'
      x.lineWidth = 2
      x.beginPath(); x.arc(px, py, 7, 0, 7); x.stroke()
      x.beginPath(); x.moveTo(px - 11, py); x.lineTo(px + 11, py); x.moveTo(px, py - 11); x.lineTo(px, py + 11); x.lineWidth = 1.3; x.stroke()
    }

    wireHeatmapHover(wrap, canvas, a, { xA, yA, cells, W, H, mL, mT, cellPx, gridW, gridH })
  }

  // Hover a cell → a tooltip with the exact config(s) + result behind that square (each a CONCRETE tried
  // value); hover an empty cell → the X/Y values it sits at, so gaps read as "nothing tried here".
  function wireHeatmapHover(wrap, canvas, a, geom) {
    let tip = wrap.parentNode.querySelector('.expl-tip')
    if (!tip) {
      tip = document.createElement('div')
      tip.className = 'expl-tip'
      tip.style.display = 'none'
      wrap.parentNode.appendChild(tip)
    }
    const { xA, yA, cells, W, H, mL, mT, cellPx, gridW, gridH } = geom
    const dir = a.dir === 'min' ? 'min' : 'max'
    canvas.onmousemove = (ev) => {
      const rect = canvas.getBoundingClientRect()
      const mx = ((ev.clientX - rect.left) / rect.width) * W - mL
      const my = ((ev.clientY - rect.top) / rect.height) * H - mT
      if (mx < 0 || my < 0 || mx >= gridW || my >= gridH) { tip.style.display = 'none'; return }
      const i = Math.max(0, Math.min(xA.n - 1, Math.floor(mx / cellPx)))
      const jScreen = Math.max(0, Math.min(yA.n - 1, Math.floor(my / cellPx)))
      const cj = yA.n - 1 - jScreen
      const cell = cells[cj * xA.n + i]
      const where = `<span class="tw">${esc(a.vs.axisX)}</span> = ${esc(xA.cellLabel(i))} · <span class="tw">${esc(a.vs.axisY)}</span> = ${esc(yA.cellLabel(cj))}`
      let body
      if (!cell || !cell.runs.length) {
        body = `<div class="th">no runs here</div><div class="tr">${where}</div>`
      } else {
        const rows = cell.runs.slice(0, 8).map((r) => {
          const cfg = Object.keys(r.config)
            .filter((k) => k !== a.vs.axisX && k !== a.vs.axisY && k !== 'seed')
            .map((k) => `${esc(k)}=${esc(fmtVal(r.config[k]))}`)
            .join(' ')
          return `<div class="tl"><b>${isFinite(r.objective) ? fmt(r.objective, 2) : '—'}</b> <span>${cfg || '—'}</span></div>`
        }).join('')
        const more = cell.runs.length > 8 ? `<div class="tm">+${cell.runs.length - 8} more</div>` : ''
        body = `<div class="th">${cell.runs.length} run${cell.runs.length === 1 ? '' : 's'} · best ${dir === 'max' ? '↑' : '↓'}</div><div class="tr">${where}</div>${rows}${more}`
      }
      tip.innerHTML = body
      tip.style.display = 'block'
      const host = wrap.parentNode.getBoundingClientRect()
      let left = ev.clientX - host.left + 14
      let top = ev.clientY - host.top + 14
      if (left + 260 > host.width) left = Math.max(4, ev.clientX - host.left - 260 - 14)
      tip.style.left = left + 'px'
      tip.style.top = top + 'px'
    }
    canvas.onmouseleave = () => { tip.style.display = 'none' }
  }

  function drawRegret(container, a) {
    const c = container.querySelector('[data-expl-regret]')
    if (!c) return
    const pts = (ctx && ctx.data.state && ctx.data.state.regret) || []
    const dpr = Math.min(2, (typeof window !== 'undefined' && window.devicePixelRatio) || 1)
    const box = c.getBoundingClientRect()
    const W = box.width || 520
    const H = 220
    c.width = W * dpr
    c.height = H * dpr
    const x = c.getContext('2d')
    x.scale(dpr, dpr)
    const mono = getComputedStyle(c).fontFamily
    if (pts.length < 2) {
      x.fillStyle = '#8a97a9'
      x.font = `12px ${mono}`
      x.textAlign = 'center'
      x.fillText('awaiting runs…', W / 2, H / 2)
      return
    }
    const pad = { l: 46, r: 12, t: 12, b: 24 }
    const iw = W - pad.l - pad.r
    const ih = H - pad.t - pad.b
    const maxR = pts[pts.length - 1].runsSpent || 1
    const vals = pts.map((p) => p.bestObjective)
    let vmin = Math.min(...vals)
    let vmax = Math.max(...vals)
    if (vmin === vmax) { vmin -= 1; vmax += 1 }
    const X = (r) => pad.l + (r / maxR) * iw
    const Y = (v) => pad.t + ih - ((v - vmin) / (vmax - vmin)) * ih
    x.strokeStyle = 'rgba(128,140,160,.18)'
    x.lineWidth = 1
    x.font = `10px ${mono}`
    x.fillStyle = '#8a97a9'
    x.textAlign = 'right'
    for (let g = 0; g <= 4; g++) {
      const v = vmin + (vmax - vmin) * (g / 4)
      const y = Y(v)
      x.beginPath(); x.moveTo(pad.l, y); x.lineTo(W - pad.r, y); x.stroke()
      x.fillText(Math.round(v), pad.l - 7, y + 3)
    }
    const gold = '#f4b740'
    const grd = x.createLinearGradient(0, pad.t, 0, pad.t + ih)
    grd.addColorStop(0, 'rgba(244,183,64,.28)')
    grd.addColorStop(1, 'rgba(244,183,64,.02)')
    x.beginPath()
    x.moveTo(X(pts[0].runsSpent), Y(pts[0].bestObjective))
    for (const p of pts) x.lineTo(X(p.runsSpent), Y(p.bestObjective))
    x.lineTo(X(pts[pts.length - 1].runsSpent), pad.t + ih)
    x.lineTo(X(pts[0].runsSpent), pad.t + ih)
    x.closePath()
    x.fillStyle = grd
    x.fill()
    x.beginPath()
    x.moveTo(X(pts[0].runsSpent), Y(pts[0].bestObjective))
    for (const p of pts) x.lineTo(X(p.runsSpent), Y(p.bestObjective))
    x.strokeStyle = gold
    x.lineWidth = 2.2
    x.lineJoin = 'round'
    x.stroke()
    const last = pts[pts.length - 1]
    x.beginPath(); x.arc(X(last.runsSpent), Y(last.bestObjective), 4, 0, 7); x.fillStyle = gold; x.fill()
    x.fillStyle = '#8a97a9'
    x.textAlign = 'center'
    x.fillText('runs spent →', pad.l + iw / 2, H - 5)
  }

  // --- html -------------------------------------------------------------------------------------

  function statusHtml(data, a) {
    const st = data.activity && data.activity.status
    const live = st === 'running' || st === 'starting' || st === 'queued'
    const paused = !!(data.state && data.state.paused)
    const done = !!(data.state && data.state.done)
    const hasState = !!data.state
    const stage = data.state && data.state.stage
    const log = (data.state && data.state.log) || []
    const last = log[log.length - 1]

    // A running/queued child `train` means the search IS working even if the `explore` controller activity
    // isn't found live in this (eventually-consistent) snapshot — so treat it as working, never a Stopped view.
    const pc = data.pendingChild
    const childActive = !!(pc && (pc.status === 'running' || pc.status === 'starting' || pc.status === 'queued'))
    const launching = !!data.launching && !live && !childActive // optimistic: just clicked Start/Resume
    const working = (live || childActive) && !paused
    const inProgress = hasState && !done && !paused && !working && !launching // mid-search, nothing executing
    const started = hasState || working || launching

    const total = typeof data.totalRuns === 'number' ? data.totalRuns : a.runs.length

    // Icon-only transport controls (pause/resume/stop); Start / Explore-again stay labelled (primary action).
    const ico = (act, glyph, label, cls) =>
      `<button class="expl-ico ${cls || ''}" data-act="${act}" title="${label}" aria-label="${label}">${glyph}</button>`
    let head, ctrls, mode
    if (launching) {
      mode = 'launching'
      head = '<span class="spin"></span> Starting exploration…'
      ctrls = ico('abort', '■', 'Stop')
    } else if (working) {
      mode = 'working'
      head = `<span class="pulse"></span> Exploring — <span data-help="${esc(STAGE_HELP[stage] || '')}">${esc(stage || 'running')}</span>`
      ctrls = ico('pause', '❚❚', 'Pause') + ico('abort', '■', 'Stop') // no Resume while working
    } else if (paused) {
      mode = 'paused'
      head = 'Paused'
      ctrls = ico('resume', '▶', 'Resume', 'primary') + ico('abort', '■', 'Stop')
    } else if (inProgress) {
      mode = 'stopped'
      head = `Stopped — <span data-help="${esc(STAGE_HELP[stage] || '')}">${esc(stage || '')}</span>`
      ctrls = ico('resume', '▶', 'Resume', 'primary') + ico('abort', '■', 'Stop')
    } else if (done) {
      mode = 'done'
      head = 'Converged'
      // Continue the SAME search (keep the runs, push the frontier deeper/wider) rather than restarting.
      ctrls = '<button class="expl-btn primary" data-act="explore-more">Explore more</button>'
    } else {
      mode = 'idle'
      head = 'Ready to explore'
      ctrls = '<button class="expl-btn primary" data-act="launch">Start</button>'
    }

    // Readouts + stage progress are PART of the status (they only exist once the search has started), so they
    // live at the top of the card rather than as a separate strip.
    const bestObj =
      data.state && data.state.regret && data.state.regret.length
        ? data.state.regret[data.state.regret.length - 1].bestObjective
        : a.runs.length ? (a.dir === 'max' ? a.oMax : a.oMin) : NaN
    const declared = a.basins.find((x) => data.state && x.id === data.state.declaredBasinId)
    const readouts = started
      ? [
          ['runs', total, ''],
          ['basins', a.basins.length, ''],
          ['best objective', isFinite(bestObj) ? fmt(bestObj, 2) : '—', 'gold'],
          ...(declared ? [['global max', basinLabel(declared, a), 'gold']] : []),
        ]
      : []
    const readoutsHtml = readouts.length
      ? `<div class="expl-readouts">${readouts.map((r) => `<div class="expl-ro"><span class="k">${esc(r[0])}</span><span class="v ${r[2]}">${esc(r[1])}</span></div>`).join('')}</div>`
      : ''
    const reachedIdx = data.state ? STAGES.indexOf(data.state.stage) : -1
    const stagesHtml = started
      ? `<ol class="expl-stages">${STAGES.map((s, i) => {
          const d = reachedIdx >= i
          const cur = data.state && data.state.stage === s && !done
          return `<li class="${d ? 'done' : ''} ${cur ? 'cur' : ''}" data-help="${esc(STAGE_HELP[s])}"><span class="n">${i + 1}</span><span>${esc(s)}</span></li>`
        }).join('')}</ol>`
      : ''

    // What's happening now — the last decision + the CURRENT campaign's concrete run progress (X/Y), not an
    // opaque "N being analyzed" backlog (the live total sits in the readouts above).
    const rationale = (last && last.rationale) || (stage || 'starting')
    let progress = ''
    if (working) {
      if (childActive && pc.status === 'queued') progress = ' · <span class="bl">run queued — waiting for a free experiment slot</span>'
      else if (data.childProgress && data.childProgress.total) progress = ` · running ${data.childProgress.done}/${data.childProgress.total} runs`
      else if (childActive) progress = ' · training run in progress…'
      else progress = ' · planning the next batch…'
    }
    const nowLine =
      working || paused || inProgress
        ? `<p class="now"><b>${esc(rationale)}</b>${progress}</p>`
        : launching ? '<p class="now">launching the controller…</p>' : ''

    // Idle: the ONE-TIME explainer + budget fields. Done: budget fields (to re-run). Otherwise neither.
    const b = (data.state && data.state.budget) || {}
    const field = (id, label, val, help) =>
      `<label class="expl-field"><span class="lbl" data-help="${esc(help)}">${esc(label)}</span><input type="number" min="0" id="${id}" value="${val != null ? esc(val) : ''}" placeholder="auto"></label>`
    const budgetFields =
      mode === 'idle' || mode === 'done'
        ? `<div class="expl-controls" style="margin-top:12px">${field('expl-maxruns', 'run budget', b.maxRuns, BUDGET_HELP.maxRuns)}${field('expl-target', 'target objective', data.state && data.state.targetObjective, BUDGET_HELP.targetObjective)}</div>`
        : ''
    const ledeHtml =
      mode === 'idle'
        ? `<p class="expl-lede">The autopilot searches this project's config space — screening which levers matter, finding every basin, then climbing each to its peak. Press Start; the coverage heatmap below fills in as runs land.</p>`
        : ''

    const logHtml = log.length
      ? `<div class="expl-logwrap" data-expl-log><ol class="expl-log">${log
          .map((e) => `<li><span class="st">${esc(e.stage)}</span><span class="ms">${esc(e.rationale)}</span><span class="rn">${e.spentRuns} runs</span></li>`)
          .join('')}</ol></div>`
      : ''

    return `<div class="expl-status ${working || launching ? 'live' : ''}">
      <div class="row"><span class="title">${head}</span><div style="flex:1"></div><div class="expl-ctrls">${ctrls}</div></div>
      ${readoutsHtml}
      ${stagesHtml}
      ${nowLine}
      ${ledeHtml}
      ${budgetFields}
      ${logHtml}
    </div>`
  }

  function mapControlsHtml(a) {
    if (!a.rankedKeys.length) return ''
    const opt = (sel, exclude) =>
      a.ranked
        .filter((r) => r.lever !== exclude)
        .map((r) => `<option value="${esc(r.lever)}" ${r.lever === sel ? 'selected' : ''}>${esc(r.lever)}${r.kind === 'cat' ? ' ▪' : ''}</option>`)
        .join('')
    const axisSel = `
      <div class="expl-sel">X <select data-expl-axis="x">${opt(a.vs.axisX, a.vs.axisY)}</select></div>
      <div class="expl-sel">Y <select data-expl-axis="y">${opt(a.vs.axisY, a.vs.axisX)}</select></div>`
    // peg controls for every ranked lever not currently an axis
    const pegSel = a.ranked
      .filter((r) => r.lever !== a.vs.axisX && r.lever !== a.vs.axisY)
      .slice(0, 4)
      .map((r) => {
        const vals = [...new Set(a.runs.map((run) => String(run.config[r.lever])).filter((v) => v && v !== 'undefined'))].sort()
        const cur = a.vs.pegs[r.lever]
        const opts = ['<option value="">any</option>']
          .concat(vals.map((v) => `<option value="${esc(v)}" ${String(cur) === v ? 'selected' : ''}>${esc(v)}</option>`))
          .join('')
        return `<div class="expl-sel">${esc(r.lever)} <select data-expl-peg="${esc(r.lever)}">${opts}</select></div>`
      })
      .join('')
    const zoom = `<div class="expl-zoom"><button data-expl-zoom="out" title="Zoom out">−</button><button data-expl-zoom="reset" title="Reset zoom">${Math.round((a.vs.zoom || 1) * 100)}%</button><button data-expl-zoom="in" title="Zoom in">+</button></div>`
    return `<div class="expl-mapbar">${axisSel}${pegSel ? '<span style="color:var(--e-faint);font-family:var(--e-mono);font-size:11px">peg:</span>' + pegSel : ''}${zoom}</div>`
  }

  function render(container, data, actions) {
    injectStyles()
    ctx = { container, data, actions }
    const a = analyze(data)
    const oMinL = fmt(a.oMin, 1)
    const oMaxL = fmt(a.oMax, 1)
    const cmapId = 'expl-cmap'
    const hasMap = a.rankedKeys.length >= 2

    const basinRows = a.basins.length
      ? a.basins.map((b, bi) => {
          const isG = data.state && b.id === data.state.declaredBasinId
          const center = b.centerConfig && a.vs.axisX && a.vs.axisY
            ? `${esc(a.vs.axisX)}=${esc(fmtVal(b.centerConfig[a.vs.axisX]))}, ${esc(a.vs.axisY)}=${esc(fmtVal(b.centerConfig[a.vs.axisY]))}`
            : '—'
          const nRuns = (b.memberRunKeys && b.memberRunKeys.length) || 0
          const openBtn = nRuns
            ? `<button class="expl-open" data-basin-open="${bi}" title="Open this basin's ${nRuns} run${nRuns === 1 ? '' : 's'} in the Runs tab">Runs ↗</button>`
            : ''
          const tier = isG ? '<span class="expl-badge">global</span>' : '<span style="color:var(--e-faint)">local</span>'
          return `<tr><td><span class="dot" style="background:${isG ? '#f4b740' : '#8fd6ff'}"></span>${esc(basinLabel(b, a))}</td>
            <td style="font-weight:600">${fmt(b.peakObjective, 1)}</td><td>${center}</td>
            <td>${b.peakSeeds || 1}</td><td>${b.plateaued ? 'yes' : '—'}</td>
            <td style="text-align:right;white-space:nowrap">${tier}${openBtn ? ' ' + openBtn : ''}</td></tr>`
        }).join('')
      : `<tr><td colspan="6" style="color:var(--e-faint);padding:14px 0">No basins yet — run the exploration to enumerate maxima.</td></tr>`

    container.innerHTML = `<div class="expl">
      ${statusHtml(data, a)}
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin:16px 0 6px">
        <h3 style="margin:0">Search-space coverage</h3>
        <div class="expl-legend"><span>objective</span><canvas id="${cmapId}" width="170" height="12"></canvas><span>${oMinL}</span><span>${oMaxL}</span></div>
      </div>
      ${
        hasMap
          ? `${mapControlsHtml(a)}
             <div class="expl-mapcard"><div class="expl-maprow">
               <div class="expl-axis-y">${esc(a.vs.axisY || '')} ↑</div>
               <div class="expl-mapmain"><div class="expl-mapscroll" data-expl-map><canvas class="expl-mapcanvas"></canvas></div>
               <div class="expl-axis-x">${esc(a.vs.axisX || '')} →</div></div>
             </div></div>`
          : `<div class="expl-empty">Coverage heatmap needs two levers with more than one value tried. Run the exploration (or a few experiments) and it will appear.</div>`
      }
      <div class="expl-grid2">
        <div class="expl-card"><h3>Convergence — best objective vs runs spent</h3><canvas data-expl-regret></canvas>
          <p class="cap">The strategist stops once no basin improves beyond the seed-noise floor${data.state && data.state.noiseFloor != null ? ' (±' + fmt(data.state.noiseFloor, 2) + ')' : ''}.</p></div>
        <div class="expl-card"><h3>Maxima found</h3>
          <table class="expl-basins"><thead><tr><th>basin</th><th>peak</th><th>center</th><th>seeds</th><th>plateaued</th><th></th></tr></thead><tbody>${basinRows}</tbody></table>
          <p class="cap">A basin beats the trivial baseline; the <b>global max</b> is the best basin that is robust across seeds and locally plateaued.</p></div>
      </div>
    </div>`

    const cm = container.querySelector('#' + cmapId)
    if (cm) {
      const cx = cm.getContext('2d')
      for (let i = 0; i < cm.width; i++) { cx.fillStyle = magma(i / (cm.width - 1)); cx.fillRect(i, 0, 1, cm.height) }
    }
    if (hasMap) drawHeatmap(container, a)
    drawRegret(container, a)
    stickLog(container)
    wire(container, data, actions, a)
  }

  // Terminal-log behaviour: after each re-render, snap to the newest line if the user was already at the
  // bottom; a scroll listener releases the snap the moment they scroll up to read history, and re-arms it
  // when they return to the bottom — so live updates never yank them away from what they're reading.
  function stickLog(container) {
    const box = container.querySelector('[data-expl-log]')
    if (!box) return
    if (logStick) box.scrollTop = box.scrollHeight
    box.addEventListener('scroll', () => {
      logStick = box.scrollHeight - box.scrollTop - box.clientHeight < 8
    })
  }

  function fmtVal(v) {
    if (typeof v === 'number') return fmt(v, v < 1 && v !== 0 ? 3 : v < 100 ? 2 : 0)
    return v
  }

  function wire(container, data, actions, a) {
    const rerender = () => render(ctx.container, ctx.data, ctx.actions)
    container.querySelectorAll('[data-expl-axis]').forEach((sel) => {
      sel.addEventListener('change', () => {
        const which = sel.getAttribute('data-expl-axis')
        if (which === 'x') a.vs.axisX = sel.value
        else a.vs.axisY = sel.value
        if (a.vs.axisX === a.vs.axisY) a.vs.axisY = a.rankedKeys.find((k) => k !== a.vs.axisX) || a.vs.axisY
        rerender()
      })
    })
    container.querySelectorAll('[data-expl-peg]').forEach((sel) => {
      sel.addEventListener('change', () => {
        const lever = sel.getAttribute('data-expl-peg')
        a.vs.pegs[lever] = sel.value === '' ? null : sel.value
        rerender()
      })
    })
    container.querySelectorAll('[data-expl-zoom]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const dir = btn.getAttribute('data-expl-zoom')
        if (dir === 'in') a.vs.zoom = Math.min(4, (a.vs.zoom || 1) + 0.5)
        else if (dir === 'out') a.vs.zoom = Math.max(0.5, (a.vs.zoom || 1) - 0.5)
        else a.vs.zoom = 1
        rerender()
      })
    })
    const onAct = (act, fn) => {
      const btn = container.querySelector(`[data-act="${act}"]`)
      if (btn) btn.addEventListener('click', fn)
    }
    const readBudget = () => {
      const g = (id) => {
        const el = container.querySelector('#' + id)
        const v = el ? parseInt(el.value, 10) : NaN
        return Number.isFinite(v) && v > 0 ? v : undefined
      }
      return { maxRuns: g('expl-maxruns'), targetObjective: g('expl-target') }
    }
    onAct('launch', () => actions.onLaunch && actions.onLaunch(readBudget()))
    // "Explore more" continues a converged search (reopen + escalate) instead of wiping it to start over.
    onAct('explore-more', () => actions.onExploreMore && actions.onExploreMore(readBudget()))
    onAct('pause', () => actions.onPause && actions.onPause())
    onAct('resume', () => actions.onResume && actions.onResume())
    onAct('abort', () => actions.onAbort && actions.onAbort())
    container.querySelectorAll('[data-basin-open]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const bi = parseInt(btn.getAttribute('data-basin-open'), 10)
        const basin = a.basins[bi]
        if (basin && basin.memberRunKeys && actions.onOpenRuns) actions.onOpenRuns(basin.memberRunKeys)
      })
    })
  }

  function basinLabel(b, a) {
    if (b.region) {
      const keys = Object.keys(b.region)
      if (keys.length) return keys.map((k) => `${k}=${fmtVal(b.region[k])}`).join(', ')
    }
    return b.id ? String(b.id).replace(/[[\]"]/g, '').slice(0, 24) || 'basin' : 'basin'
  }

  const Exploration = { render, analyze, magma, rankLevers, heatmapCells, makeAxis }
  if (typeof module !== 'undefined' && module.exports) module.exports = Exploration
  if (root) root.Exploration = Exploration
})(typeof window !== 'undefined' ? window : null)
