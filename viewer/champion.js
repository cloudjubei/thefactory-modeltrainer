/*
 * "Improve your model" — the one-button self-improvement loop for a LEARNED core.
 *
 * Trains a stronger warm-started generation, plays it against the league of best models, keeps the winner, and
 * repeats until strength plateaus / a target is met / the generation budget is spent. Each generation is rated
 * on the ONE fixed scale, so the leaderboard below climbs live. It launches the `train-champion` activity (an
 * Experiment), so progress also shows in the Experiments tab. Advanced knobs stay collapsed behind "Options".
 *
 * window.Champion.render(hostEl, { state, launching, onLaunch(config) })
 */
;(function (root) {
  'use strict'

  var CSS =
    '.champion{margin:0 0 14px;padding:16px;border:1px solid var(--border);border-radius:11px;background:var(--surface)}' +
    '.champion .ch-head{font-size:15px;font-weight:600;color:var(--text)}' +
    '.champion .ch-sub{font-size:12px;color:var(--muted);margin:3px 0 12px;max-width:640px;line-height:1.5}' +
    '.champion .ch-cta{display:flex;align-items:center;gap:12px;flex-wrap:wrap}' +
    '.champion .ch-btn{padding:9px 20px;border-radius:8px;background:var(--accent);color:#fff;border:0;font-size:14px;' +
    'font-weight:600;cursor:pointer}' +
    '.champion .ch-btn:disabled{opacity:.55;cursor:default}' +
    '.champion .ch-toggle{background:none;border:0;color:var(--muted);font-size:12px;cursor:pointer;padding:4px 2px;' +
    'text-decoration:underline;text-underline-offset:2px}' +
    '.champion .ch-form{display:none;flex-wrap:wrap;gap:10px;align-items:flex-end;margin-top:12px;padding-top:12px;' +
    'border-top:1px dashed var(--border)}' +
    '.champion .ch-form.open{display:flex}' +
    '.champion .ch-field{display:flex;flex-direction:column;gap:3px}' +
    '.champion .ch-field label{font-size:11px;color:var(--muted)}' +
    '.champion .ch-field input,.champion .ch-field select{padding:5px 8px;border-radius:6px;background:var(--surface-subtle);' +
    'color:var(--text);border:1px solid var(--border);font-size:13px;width:104px}' +
    '.champion .ch-status{font-size:13px;color:var(--text);margin-top:14px;font-weight:500}' +
    '.champion .ch-hint{font-size:12px;color:var(--muted);margin-top:5px;line-height:1.5;max-width:640px}' +
    '.champion table{border-collapse:collapse;margin-top:12px;font-size:12px;width:100%;max-width:520px}' +
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

  // The "what do I do next" line — the whole point of the one-button flow is that the user is never guessing.
  function nextStepHint(state) {
    if (!state || !state.generation) {
      return 'Press Improve to start. It runs as an experiment (watch it in the Experiments tab); the leaderboard below climbs after every generation.'
    }
    if (state.stage === 'converged') {
      var reason = state.stopReason || 'done'
      if (reason === 'reached-target') return 'Target strength reached. Press Improve again to push the bar higher, or search new architectures below.'
      if (reason === 'plateau') return 'Strength plateaued — more of the same stopped helping. Press Improve with more Search (Options), or search new architectures below.'
      return 'Generation budget spent. Press Improve to run another batch, or search new architectures below.'
    }
    return 'Improving… each generation warm-starts from the current champion. The leaderboard below updates as it climbs.'
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
    head.textContent = 'Improve your model'
    var sub = document.createElement('div')
    sub.className = 'ch-sub'
    sub.textContent =
      'Trains a stronger version, plays it against a league of your best models, and keeps the winner — over and over, until it stops getting better. Every generation is scored on one fixed scale, so the leaderboard below shows real progress.'
    host.appendChild(head)
    host.appendChild(sub)

    // Advanced knobs — created up-front (the Improve button reads their values) but collapsed by default.
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

    var cta = document.createElement('div')
    cta.className = 'ch-cta'
    var btn = document.createElement('button')
    btn.className = 'ch-btn'
    btn.type = 'button'
    btn.textContent = opts.launching ? 'Starting…' : 'Improve'
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
    var toggle = document.createElement('button')
    toggle.className = 'ch-toggle'
    toggle.type = 'button'
    toggle.textContent = '⚙ Options'
    cta.appendChild(btn)
    cta.appendChild(toggle)
    host.appendChild(cta)

    var form = document.createElement('div')
    form.className = 'ch-form'
    form.appendChild(field('Generations', genIn))
    form.appendChild(field('Search (az_sims)', simsIn))
    form.appendChild(field('Target vs mcts %', targetIn))
    form.appendChild(field('Eval games', evalIn))
    form.appendChild(field('Yardstick', oppSel))
    host.appendChild(form)
    toggle.addEventListener('click', function () {
      form.classList.toggle('open')
    })

    if (state) {
      var status = document.createElement('div')
      status.className = 'ch-status'
      var best = pct(state.bestVsStrongMcts)
      status.textContent =
        'Generation ' +
        (state.generation || 0) +
        ' · best vs strong mcts ' +
        best +
        (state.stage === 'converged' ? ' · stopped (' + (state.stopReason || 'done') + ')' : ' · improving…')
      host.appendChild(status)
    }

    var hint = document.createElement('div')
    hint.className = 'ch-hint'
    hint.textContent = nextStepHint(state)
    host.appendChild(hint)

    if (state) {
      var history = Array.isArray(state.history) ? state.history.slice(-12) : []
      if (history.length) {
        var table = document.createElement('table')
        table.innerHTML =
          '<tr><th>Gen</th><th>Kept</th><th>vs strong mcts</th><th>vs champion</th></tr>' +
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
