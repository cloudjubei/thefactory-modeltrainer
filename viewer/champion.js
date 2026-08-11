/*
 * Champion-training autopilot panel — the "keep looking for the best model" launcher for a LEARNED core.
 *
 * Distinct from the config-space Exploration autopilot: this loops warm-started training generations (each
 * continues the champion + trains vs the league) and promotes stronger nets until strength plateaus / a
 * target is met / the generation budget is spent. It launches the `train-champion` activity and shows the
 * champion ladder (each generation: promoted?, win-rate vs the strong-mcts yardstick + vs the champion).
 *
 * window.Champion.render(hostEl, { state, launching, onLaunch(config) })
 */
;(function (root) {
  'use strict'

  var CSS =
    '.champion{margin:0 0 14px;padding:14px;border:1px solid var(--border);border-radius:11px;background:var(--surface)}' +
    '.champion .ch-head{font-size:14px;font-weight:600;color:var(--text)}' +
    '.champion .ch-sub{font-size:12px;color:var(--muted);margin:2px 0 10px}' +
    '.champion .ch-form{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end}' +
    '.champion .ch-field{display:flex;flex-direction:column;gap:3px}' +
    '.champion .ch-field label{font-size:11px;color:var(--muted)}' +
    '.champion .ch-field input,.champion .ch-field select{padding:5px 8px;border-radius:6px;background:var(--surface-subtle);' +
    'color:var(--text);border:1px solid var(--border);font-size:13px;width:96px}' +
    '.champion .ch-btn{padding:7px 14px;border-radius:7px;background:var(--accent);color:#fff;border:0;font-size:13px;' +
    'font-weight:600;cursor:pointer}' +
    '.champion .ch-btn:disabled{opacity:.55;cursor:default}' +
    '.champion .ch-status{font-size:12px;color:var(--muted);margin-top:10px}' +
    '.champion table{border-collapse:collapse;margin-top:10px;font-size:12px;width:100%;max-width:520px}' +
    '.champion th,.champion td{text-align:left;padding:4px 10px;border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums}' +
    '.champion th{color:var(--muted);font-weight:500}' +
    '.champion .ch-promoted{color:var(--ok);font-weight:600}'

  function ensureCss() {
    if (typeof document === 'undefined' || document.getElementById('champion-css')) return
    var st = document.createElement('style')
    st.id = 'champion-css'
    st.textContent = CSS
    document.head.appendChild(st)
  }

  function num(v, d) {
    var n = parseFloat(v)
    return isFinite(n) ? n : d
  }

  function pct(v) {
    return v === undefined || v === null ? '—' : Math.round(v * 100) + '%'
  }

  function field(label, node) {
    var f = document.createElement('div')
    f.className = 'ch-field'
    var l = document.createElement('label')
    l.textContent = label
    f.appendChild(l)
    f.appendChild(node)
    return f
  }

  function numInput(value) {
    var i = document.createElement('input')
    i.type = 'number'
    i.value = String(value)
    return i
  }

  function render(host, opts) {
    if (!host) return
    ensureCss()
    opts = opts || {}
    var state = opts.state
    host.className = 'champion'
    host.innerHTML = ''

    var head = document.createElement('div')
    head.className = 'ch-head'
    head.textContent = 'Champion training'
    var sub = document.createElement('div')
    sub.className = 'ch-sub'
    sub.textContent =
      'Keep training warm-started generations against the league, promoting the stronger net, until strength plateaus.'
    host.appendChild(head)
    host.appendChild(sub)

    var form = document.createElement('div')
    form.className = 'ch-form'
    var genIn = numInput(8)
    var simsIn = numInput(100)
    var targetIn = numInput(90)
    var evalIn = numInput(20)
    var oppSel = document.createElement('select')
    ;['mcts', 'heuristic', 'random'].forEach(function (o) {
      var op = document.createElement('option')
      op.value = o
      op.textContent = o
      oppSel.appendChild(op)
    })
    form.appendChild(field('Generations', genIn))
    form.appendChild(field('Search (az_sims)', simsIn))
    form.appendChild(field('Target vs mcts %', targetIn))
    form.appendChild(field('Eval games', evalIn))
    form.appendChild(field('Yardstick', oppSel))

    var btn = document.createElement('button')
    btn.className = 'ch-btn'
    btn.type = 'button'
    btn.textContent = opts.launching ? 'Starting…' : 'Train champion'
    btn.disabled = !!opts.launching
    btn.addEventListener('click', function () {
      if (typeof opts.onLaunch !== 'function') return
      opts.onLaunch({
        maxGenerations: num(genIn.value, 8),
        targetStrength: Math.max(0, Math.min(1, num(targetIn.value, 90) / 100)),
        opponent: oppSel.value,
        evalGames: num(evalIn.value, 20),
        hyperparams: { az_sims: num(simsIn.value, 100) },
      })
    })
    form.appendChild(field(' ', btn))
    host.appendChild(form)

    if (state) {
      var status = document.createElement('div')
      status.className = 'ch-status'
      var best = pct(state.bestVsStrongMcts)
      status.textContent =
        'Champion generation ' +
        (state.generation || 0) +
        ' · best vs strong mcts ' +
        best +
        (state.stage === 'converged' ? ' · stopped (' + (state.stopReason || 'done') + ')' : ' · training…')
      host.appendChild(status)

      var history = Array.isArray(state.history) ? state.history.slice(-12) : []
      if (history.length) {
        var table = document.createElement('table')
        table.innerHTML =
          '<tr><th>Gen</th><th>Promoted</th><th>vs strong mcts</th><th>vs champion</th></tr>' +
          history
            .map(function (h) {
              return (
                '<tr><td>' +
                h.generation +
                '</td><td>' +
                (h.promoted ? '<span class="ch-promoted">✔ crowned</span>' : '—') +
                '</td><td>' +
                pct(h.winRateVsStrongMcts) +
                '</td><td>' +
                pct(h.winRateVsChampion) +
                '</td></tr>'
              )
            })
            .join('')
        host.appendChild(table)
      }
    }
  }

  root.Champion = { render: render }
})(typeof window !== 'undefined' ? window : this)
