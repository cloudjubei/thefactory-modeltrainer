/*
 * Exploration view — the config-space search map, in-app and always available per project.
 *
 * Renders the `{recordType}-exploration` autopilot state (stage, basins, regret, convergence) plus a
 * live COVERAGE HEATMAP binned from the project's actual runs: for the two most-searched numeric levers
 * (per categorical basin, when there is one), each grid cell is coloured by the best objective observed
 * in it and empty cells read as unexplored gaps — so "how well are we covering the space" is legible at
 * a glance. Pure DOM/canvas over data the host hands in; app.js owns the bridge reads/launches.
 *
 * window.Exploration.render(container, data, actions)
 *   data    = { manifest, state|null, runs:[{config,objective}], activity:{status}|null }
 *   actions = { onLaunch(budget), onPause(), onResume(), onAbort() }
 *
 * Exposed as window.Exploration in the browser and module.exports under CommonJS (so the pure
 * `analyze`/`magma` logic is unit-tested the same way the other viewer modules are).
 */
;(function (root) {
  'use strict'
  const STAGES = ['calibrate', 'screen', 'global', 'local', 'converged']
  const GRID = 26
  // perceptual magma-ish ramp — the value scale itself (dark → magenta → orange → pale yellow)
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
    String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c])
  const num = (v) => (typeof v === 'number' && isFinite(v) ? v : NaN)
  const fmt = (v, d = 0) => (isFinite(v) ? Number(v).toFixed(d) : '—')

  let stylesInjected = false
  function injectStyles() {
    if (stylesInjected) return
    stylesInjected = true
    const css = `
    .expl{--e-line:#e2e7ef;--e-panel:#fff;--e-panel2:#f6f8fb;--e-text:#141d29;--e-muted:#5c6a7e;--e-faint:#8a97a9;--e-gold:#c8891a;--e-ice:#2b7fb8;--e-ok:#12936a;--e-mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;color:var(--e-text)}
    @media (prefers-color-scheme:dark){.expl{--e-line:#1f2a3b;--e-panel:#111725;--e-panel2:#0d121d;--e-text:#dbe4f1;--e-muted:#8492a8;--e-faint:#5c6a80;--e-gold:#f4b740;--e-ice:#8fd6ff;--e-ok:#4bd4a0}}
    .expl{padding:20px 22px 60px;font-feature-settings:"tnum" 1}
    .expl h2{font-size:17px;margin:0 0 4px;font-weight:640}
    .expl h3{font-size:12.5px;margin:0 0 12px;font-weight:620}
    .expl .lede{color:var(--e-muted);font-size:13px;max-width:74ch;margin:0 0 18px}
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
    .expl-legend{display:flex;align-items:center;gap:8px;font-family:var(--e-mono);font-size:11px;color:var(--e-muted)}
    .expl-legend canvas{border-radius:3px;border:1px solid var(--e-line);display:block}
    .expl-maps{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px;margin:6px 0 8px}
    .expl-mapcard{background:var(--e-panel);border:1px solid var(--e-line);border-radius:11px;padding:14px}
    .expl-mapcard .mh{display:flex;align-items:baseline;justify-content:space-between;gap:8px;margin-bottom:10px;font-family:var(--e-mono);font-size:12.5px}
    .expl-mapcard .mh b{font-weight:600}
    .expl-mapcard .mh .peak{color:var(--e-muted)}
    .expl-badge{font-family:var(--e-mono);font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;padding:2px 6px;border-radius:20px;background:color-mix(in srgb,var(--e-gold) 20%,transparent);color:var(--e-gold);border:1px solid color-mix(in srgb,var(--e-gold) 45%,transparent)}
    .expl-mapwrap{position:relative;aspect-ratio:1/1;border-radius:7px;overflow:hidden;border:1px solid var(--e-line)}
    .expl-mapwrap canvas{width:100%;height:100%;display:block}
    .expl-axis{display:flex;justify-content:space-between;font-family:var(--e-mono);font-size:10px;color:var(--e-faint);margin-top:5px}
    .expl-grid2{display:grid;grid-template-columns:1.1fr 1fr;gap:18px;margin-top:22px}
    @media (max-width:820px){.expl-grid2{grid-template-columns:1fr}}
    .expl-card{background:var(--e-panel);border:1px solid var(--e-line);border-radius:11px;padding:16px 18px}
    .expl-card canvas{width:100%;height:auto;display:block}
    .expl-card .cap{color:var(--e-muted);font-size:12px;margin:10px 0 0}
    table.expl-basins{width:100%;border-collapse:collapse;font-family:var(--e-mono);font-size:12px}
    table.expl-basins th{text-align:left;font-weight:500;color:var(--e-faint);font-size:10px;letter-spacing:.06em;text-transform:uppercase;padding:0 8px 8px 0;border-bottom:1px solid var(--e-line)}
    table.expl-basins td{padding:9px 8px 9px 0;border-bottom:1px solid var(--e-line);white-space:nowrap}
    table.expl-basins tr:last-child td{border-bottom:none}
    table.expl-basins .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}
    .expl-controls{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end;background:var(--e-panel2);border:1px solid var(--e-line);border-radius:11px;padding:14px 16px;margin-bottom:20px}
    .expl-field{display:flex;flex-direction:column;gap:4px;font-family:var(--e-mono);font-size:11px;color:var(--e-muted)}
    .expl-field input{width:110px;padding:7px 9px;border:1px solid var(--e-line);border-radius:7px;background:var(--e-panel);color:var(--e-text);font-family:var(--e-mono);font-size:12.5px}
    .expl-btn{font:inherit;font-family:var(--e-mono);font-size:12.5px;padding:8px 15px;border-radius:8px;border:1px solid var(--e-line);background:var(--e-panel);color:var(--e-text);cursor:pointer}
    .expl-btn.primary{background:var(--e-gold);border-color:transparent;color:#1a1204;font-weight:600}
    .expl-btn:disabled{opacity:.5;cursor:default}
    .expl-btn:focus-visible{outline:2px solid var(--e-ice);outline-offset:2px}
    .expl-empty{text-align:center;color:var(--e-muted);padding:40px 0;font-size:13.5px}
    .expl-note{font-family:var(--e-mono);font-size:11.5px;color:var(--e-faint);margin-top:6px}
    `
    const s = document.createElement('style')
    s.id = 'exploration-styles'
    s.textContent = css
    document.head.appendChild(s)
  }

  // Derive the render model from the state + runs + manifest.
  function analyze(data) {
    const manifest = data.manifest || { levers: {}, objective: { direction: 'max' } }
    const dir = (manifest.objective && manifest.objective.direction) || 'max'
    const levers = manifest.levers || {}
    const state = data.state || null
    const runs = (data.runs || []).filter((r) => r && r.config && isFinite(num(r.objective)))

    const modelLevers = Object.keys(levers).filter(
      (l) => l !== 'seed' && ((levers[l].scope || 'model') === 'model'),
    )
    const isNumeric = (l) => levers[l] && levers[l].type === 'number'
    const active = (state && state.activeLevers && state.activeLevers.length ? state.activeLevers : modelLevers)
    const numericActive = active.filter(isNumeric)
    const categoricalActive = active.filter((l) => !isNumeric(l))
    // choose the two most-varied numeric levers as the heatmap axes
    const varOf = (l) => {
      const vals = runs.map((r) => num(r.config[l])).filter((v) => isFinite(v))
      if (vals.length < 2) return 0
      const m = vals.reduce((a, b) => a + b, 0) / vals.length
      return vals.reduce((a, b) => a + (b - m) ** 2, 0) / vals.length
    }
    const axes = numericActive.slice().sort((a, b) => varOf(b) - varOf(a)).slice(0, 2)
    const regionAxis = categoricalActive[0] || null

    // objective orientation → color scale
    const objs = runs.map((r) => num(r.objective)).filter(isFinite)
    const oMin = objs.length ? Math.min(...objs) : 0
    const oMax = objs.length ? Math.max(...objs) : 1
    const nrm = (v) => {
      if (oMax === oMin) return 0.5
      const t = (v - oMin) / (oMax - oMin)
      return dir === 'max' ? t : 1 - t
    }
    const rangeOf = (l) => {
      const spec = levers[l] || {}
      if (Array.isArray(spec.range)) return spec.range.slice()
      const vals = runs.map((r) => num(r.config[l])).filter(isFinite)
      return vals.length ? [Math.min(...vals), Math.max(...vals)] : [0, 1]
    }

    const basins = (state && state.basins) || []
    return { manifest, dir, levers, state, runs, axes, regionAxis, basins, nrm, oMin, oMax, rangeOf, isNumeric }
  }

  function drawHeatmaps(container, a) {
    if (a.axes.length < 2) return
    const [lx, ly] = a.axes
    const rx = a.rangeOf(lx)
    const ry = a.rangeOf(ly)
    const cards = container.querySelectorAll('[data-expl-map]')
    cards.forEach((wrap) => {
      const region = wrap.getAttribute('data-region') // '' = whole space, else the categorical value
      const runs = a.runs.filter((r) => (region === '' ? true : String(r.config[a.regionAxis]) === region))
      const canvas = wrap.querySelector('canvas')
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      const box = wrap.getBoundingClientRect()
      const W = Math.max(150, box.width)
      const H = W
      canvas.width = W * dpr
      canvas.height = H * dpr
      const x = canvas.getContext('2d')
      x.scale(dpr, dpr)
      // bin into a grid — cell = best (oriented) objective observed
      const best = new Array(GRID * GRID).fill(null)
      const toi = (v, r) => Math.max(0, Math.min(GRID - 1, Math.floor(((v - r[0]) / (r[1] - r[0] || 1)) * GRID)))
      for (const r of runs) {
        const vx = num(r.config[lx])
        const vy = num(r.config[ly])
        if (!isFinite(vx) || !isFinite(vy)) continue
        const gi = toi(vx, rx)
        const gj = toi(vy, ry)
        const k = gj * GRID + gi
        const t = a.nrm(num(r.objective))
        if (best[k] === null || t > best[k]) best[k] = t
      }
      const cw = W / GRID
      const ch = H / GRID
      const emptyCol = getComputedStyle(container.querySelector('.expl')).getPropertyValue('--e-panel2') || '#0d121d'
      for (let j = 0; j < GRID; j++) {
        for (let i = 0; i < GRID; i++) {
          const t = best[(GRID - 1 - j) * GRID + i]
          x.fillStyle = t === null ? emptyCol : magma(t)
          x.fillRect(i * cw, j * ch, cw + 0.5, ch + 0.5)
        }
      }
      // faint grid so unexplored gaps read as a lattice
      x.strokeStyle = 'rgba(128,140,160,.14)'
      x.lineWidth = 1
      for (let g = 1; g < GRID; g++) {
        x.beginPath(); x.moveTo(g * cw, 0); x.lineTo(g * cw, H); x.stroke()
        x.beginPath(); x.moveTo(0, g * ch); x.lineTo(W, g * ch); x.stroke()
      }
      // peak crosshair for this region's basin
      const basin = a.basins.find((b) =>
        region === '' ? true : String(b.region && b.region[a.regionAxis]) === region,
      )
      if (basin && basin.centerConfig) {
        const px = ((num(basin.centerConfig[lx]) - rx[0]) / (rx[1] - rx[0] || 1)) * W
        const py = (1 - (num(basin.centerConfig[ly]) - ry[0]) / (ry[1] - ry[0] || 1)) * H
        const isG = a.state && basin.id === a.state.declaredBasinId
        const col = isG ? '#f4b740' : '#8fd6ff'
        if (isFinite(px) && isFinite(py)) {
          x.strokeStyle = col
          x.lineWidth = 2
          x.beginPath(); x.arc(px, py, 7, 0, 7); x.stroke()
          x.beginPath(); x.moveTo(px - 11, py); x.lineTo(px + 11, py); x.moveTo(px, py - 11); x.lineTo(px, py + 11); x.lineWidth = 1.3; x.stroke()
        }
      }
    })
  }

  function drawRegret(container, a) {
    const c = container.querySelector('[data-expl-regret]')
    if (!c) return
    const pts = (a.state && a.state.regret) || []
    const dpr = Math.min(2, window.devicePixelRatio || 1)
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

  function controlsHtml(data) {
    const st = data.activity && data.activity.status
    const running = st === 'running' || st === 'starting' || st === 'queued'
    const paused = !!(data.state && data.state.paused)
    const b = (data.state && data.state.budget) || {}
    const runsField = (id, label, val) =>
      `<label class="expl-field">${label}<input type="number" min="0" id="${id}" value="${val != null ? val : ''}" placeholder="auto"></label>`
    let ctrl = ''
    if (running && !paused) ctrl += `<button class="expl-btn" data-act="pause">Pause</button>`
    if (running && paused) ctrl += `<button class="expl-btn primary" data-act="resume">Resume</button>`
    if (running) ctrl += `<button class="expl-btn" data-act="abort">Stop</button>`
    if (!running) ctrl += `<button class="expl-btn primary" data-act="launch">Start exploration</button>`
    return `<div class="expl-controls">
      ${runsField('expl-maxruns', 'run budget', b.maxRuns)}
      ${runsField('expl-conc', 'max concurrent', b.maxConcurrent)}
      ${runsField('expl-target', 'target objective', (data.state && data.state.targetObjective))}
      <div style="flex:1"></div>${ctrl}
    </div>`
  }

  function render(container, data, actions) {
    injectStyles()
    const a = analyze(data)
    const dir = a.dir
    const declared = a.basins.find((x) => a.state && x.id === a.state.declaredBasinId)
    const bestObj = a.state && a.state.regret && a.state.regret.length
      ? a.state.regret[a.state.regret.length - 1].bestObjective
      : (a.runs.length ? (dir === 'max' ? a.oMax : a.oMin) : NaN)
    const stage = (a.state && a.state.stage) || (data.activity ? 'running' : null)

    const readouts = [
      ['stage', stage || 'idle', a.state && a.state.done ? 'ok' : ''],
      ['runs spent', a.state ? (a.state.budget && a.state.budget.spentRuns) || a.runs.length : a.runs.length, ''],
      ['basins', a.basins.length, ''],
      ['best objective', isFinite(bestObj) ? fmt(bestObj, 2) : '—', 'gold'],
    ]
    if (declared) readouts.push(['global max', 'basin ' + basinLabel(declared, a), 'gold'])

    const stageBar = STAGES.map((s, i) => {
      const reachedIdx = a.state ? STAGES.indexOf(a.state.stage) : -1
      const done = reachedIdx >= i
      const cur = a.state && a.state.stage === s && !a.state.done
      return `<li class="${done ? 'done' : ''} ${cur ? 'cur' : ''}"><span class="n">${i + 1}</span><span>${s}</span></li>`
    }).join('')

    // heatmap cards: one per basin region (categorical axis), else one for the whole space
    let mapsHtml = ''
    if (a.axes.length >= 2) {
      const regions = a.regionAxis
        ? [...new Set(a.runs.map((r) => String(r.config[a.regionAxis])))].filter((v) => v && v !== 'undefined')
        : ['']
      // prefer basin regions ordered by peak; cap to keep it readable
      const ordered = a.regionAxis
        ? regions.sort((r1, r2) => peakOfRegion(r2, a) - peakOfRegion(r1, a)).slice(0, 6)
        : ['']
      mapsHtml = ordered.map((region) => {
        const basin = a.basins.find((b) => (region === '' ? true : String(b.region && b.region[a.regionAxis]) === region))
        const isG = basin && a.state && basin.id === a.state.declaredBasinId
        const title = a.regionAxis ? `${a.regionAxis} = ${esc(region)}` : 'whole space'
        const peak = basin ? `peak <b>${fmt(basin.peakObjective, 1)}</b>` : ''
        return `<div class="expl-mapcard">
          <div class="mh"><span><b>${title}</b> ${isG ? '<span class="expl-badge">global</span>' : ''}</span><span class="peak">${peak}</span></div>
          <div class="expl-mapwrap" data-expl-map data-region="${esc(region)}"><canvas></canvas></div>
          <div class="expl-axis"><span>${esc(a.axes[0])} →</span><span>↑ ${esc(a.axes[1])}</span></div>
        </div>`
      }).join('')
    } else {
      mapsHtml = `<div class="expl-empty">The heatmap needs two searched numeric levers. This project's active search has ${a.axes.length}; the basins &amp; convergence below still apply.</div>`
    }

    const cmapId = 'expl-cmap-' + Math.floor(a.oMin)
    const basinRows = a.basins.length
      ? a.basins.map((b) => {
          const isG = a.state && b.id === a.state.declaredBasinId
          const center = a.axes.length >= 2 && b.centerConfig
            ? `(${fmt(num(b.centerConfig[a.axes[0]]), 2)}, ${fmt(num(b.centerConfig[a.axes[1]]), 2)})`
            : '—'
          return `<tr><td><span class="dot" style="background:${isG ? '#f4b740' : '#8fd6ff'}"></span>${esc(basinLabel(b, a))}</td>
            <td style="font-weight:600">${fmt(b.peakObjective, 1)}</td><td>${center}</td>
            <td>${b.peakSeeds || 1}</td><td>${b.plateaued ? 'yes' : '—'}</td>
            <td>${isG ? '<span class="expl-badge">global</span>' : '<span style="color:var(--e-faint)">local</span>'}</td></tr>`
        }).join('')
      : `<tr><td colspan="6" style="color:var(--e-faint);padding:14px 0">No basins yet — run the exploration to enumerate maxima.</td></tr>`

    container.innerHTML = `<div class="expl">
      <h2>Exploration${data.manifest ? ' — ' + esc(data.manifest.recordType || '') : ''}</h2>
      <p class="lede">The autopilot searches this project's config space — screening which levers matter, finding every basin, then climbing each to its peak. The heatmaps show where the search has spent its budget.</p>
      ${controlsHtml(data)}
      <div class="expl-readouts">${readouts.map((r) => `<div class="expl-ro"><span class="k">${r[0]}</span><span class="v ${r[2]}">${esc(r[1])}</span></div>`).join('')}</div>
      <ol class="expl-stages">${stageBar}</ol>
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:8px">
        <h3 style="margin:0">Search-space coverage</h3>
        <div class="expl-legend"><span>objective</span><canvas id="${cmapId}" width="180" height="12"></canvas><span>${fmt(a.oMin, 1)}</span><span>${fmt(a.oMax, 1)}</span></div>
      </div>
      <div class="expl-maps">${mapsHtml}</div>
      <div class="expl-grid2">
        <div class="expl-card"><h3>Convergence — best objective vs runs spent</h3><canvas data-expl-regret></canvas>
          <p class="cap">The strategist stops once no basin improves beyond the seed-noise floor${a.state && a.state.noiseFloor != null ? ' (±' + fmt(a.state.noiseFloor, 2) + ')' : ''}.</p></div>
        <div class="expl-card"><h3>Maxima found</h3>
          <table class="expl-basins"><thead><tr><th>basin</th><th>peak</th><th>center</th><th>seeds</th><th>plateaued</th><th></th></tr></thead><tbody>${basinRows}</tbody></table>
          <p class="cap">A basin beats the trivial baseline; the <b>global max</b> is the best basin that is robust across seeds and locally plateaued.</p></div>
      </div>
      ${a.state ? '' : '<p class="expl-note">No exploration has run for this project yet — set a budget and press Start.</p>'}
    </div>`

    // paint the colormap legend
    const cm = container.querySelector('#' + CSS.escape(cmapId))
    if (cm) {
      const cx = cm.getContext('2d')
      for (let i = 0; i < cm.width; i++) { cx.fillStyle = magma(i / (cm.width - 1)); cx.fillRect(i, 0, 1, cm.height) }
    }
    drawHeatmaps(container, a)
    drawRegret(container, a)
    wire(container, data, actions)
  }

  function wire(container, data, actions) {
    const onClick = (act, fn) => {
      const btn = container.querySelector(`[data-act="${act}"]`)
      if (btn) btn.addEventListener('click', fn)
    }
    onClick('launch', () => {
      const g = (id) => {
        const v = parseInt((container.querySelector('#' + id) || {}).value, 10)
        return Number.isFinite(v) && v > 0 ? v : undefined
      }
      actions.onLaunch && actions.onLaunch({
        maxRuns: g('expl-maxruns'),
        maxConcurrent: g('expl-conc'),
        targetObjective: g('expl-target'),
      })
    })
    onClick('pause', () => actions.onPause && actions.onPause())
    onClick('resume', () => actions.onResume && actions.onResume())
    onClick('abort', () => actions.onAbort && actions.onAbort())
  }

  function basinLabel(b, a) {
    if (a.regionAxis && b.region && b.region[a.regionAxis] != null) return `${a.regionAxis}=${b.region[a.regionAxis]}`
    return b.id ? String(b.id).replace(/[[\]"]/g, '').slice(0, 24) : 'basin'
  }
  function peakOfRegion(region, a) {
    const basin = a.basins.find((b) => String(b.region && b.region[a.regionAxis]) === region)
    return basin ? (a.dir === 'max' ? basin.peakObjective : -basin.peakObjective) : -Infinity
  }

  const Exploration = { render, analyze, magma }
  if (typeof module !== 'undefined' && module.exports) module.exports = Exploration
  if (root) root.Exploration = Exploration
})(typeof window !== 'undefined' ? window : null)
