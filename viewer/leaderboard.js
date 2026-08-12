/*
 * Model leaderboard — the COMPARABLE strength ranking. Every model is rated on ONE scale (a gauntlet vs a
 * fixed reference spine), so "win-rate vs whatever opponent the lever picked" stops mattering: a run that
 * only beat `random` can't outrank one that beat a strong `mcts`. Ranked by the conservative LOWER BOUND, so
 * a saturated 100% against a weak opponent (flagged `weak-opponent-only`) never floats to the top.
 *
 * window.Leaderboard.render(hostEl, { record, rating, onRate(config) })
 */
;(function (root) {
  'use strict'

  var CSS =
    '.leaderboard{margin:14px 0 8px;padding:14px;border:1px solid var(--border);border-radius:11px;background:var(--surface)}' +
    '.leaderboard .lb-head{font-size:14px;font-weight:600;color:var(--text)}' +
    '.leaderboard .lb-sub{font-size:12px;color:var(--muted);margin:2px 0 10px}' +
    '.leaderboard .lb-row{display:flex;align-items:flex-end;gap:8px;flex-wrap:wrap;margin-bottom:10px}' +
    '.leaderboard .lb-field{display:flex;flex-direction:column;gap:3px}' +
    '.leaderboard .lb-field label{font-size:11px;color:var(--muted)}' +
    '.leaderboard .lb-field input{padding:5px 8px;border-radius:6px;background:var(--surface-subtle);color:var(--text);border:1px solid var(--border);font-size:13px;width:84px}' +
    '.leaderboard .lb-btn{padding:7px 14px;border-radius:7px;background:var(--accent);color:#fff;border:0;font-size:13px;font-weight:600;cursor:pointer}' +
    '.leaderboard .lb-btn:disabled{opacity:.55;cursor:default}' +
    '.leaderboard table{border-collapse:collapse;font-size:12px;width:100%;overflow-x:auto;display:block}' +
    '.leaderboard th,.leaderboard td{text-align:left;padding:5px 10px;border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums;white-space:nowrap}' +
    '.leaderboard th{color:var(--muted);font-weight:500}' +
    '.leaderboard .lb-rating{font-weight:600;color:var(--text)}' +
    '.leaderboard .lb-flag{color:var(--warn);font-size:11px}' +
    '.leaderboard .lb-climb{color:var(--muted);font-size:11px}' +
    '.leaderboard .lb-empty{font-size:12px;color:var(--muted)}' +
    '.leaderboard .lb-basis{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;' +
    'margin:0 0 10px;border:1px solid var(--border)}' +
    '.leaderboard .lb-basis.verified{color:var(--ok);border-color:var(--ok)}' +
    '.leaderboard .lb-basis.estimate{color:var(--muted)}' +
    '.leaderboard .lb-btn.secondary{background:var(--surface-subtle);color:var(--text);border:1px solid var(--border)}'

  function ensureCss() {
    if (typeof document === 'undefined' || document.getElementById('leaderboard-css')) return
    var st = document.createElement('style')
    st.id = 'leaderboard-css'
    st.textContent = CSS
    document.head.appendChild(st)
  }

  function esc(s) {
    return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  }

  function num(v, d) {
    var n = parseFloat(v)
    return isFinite(n) ? n : d
  }

  function render(host, opts) {
    if (!host) return
    ensureCss()
    opts = opts || {}
    var record = opts.record
    host.className = 'leaderboard'
    host.innerHTML = ''

    var head = document.createElement('div')
    head.className = 'lb-head'
    head.textContent = 'Model leaderboard'
    var sub = document.createElement('div')
    sub.className = 'lb-sub'
    sub.textContent =
      'Comparable strength on ONE scale. Ranked by the confident lower bound, so a 100% only against a weak opponent cannot rank high. It updates automatically as you Improve; play the full gauntlet to confirm with fresh games.'
    host.appendChild(head)
    host.appendChild(sub)

    var verified = record && record.basis === 'gauntlet'
    if (record) {
      var badge = document.createElement('span')
      badge.className = 'lb-basis ' + (verified ? 'verified' : 'estimate')
      badge.textContent = verified
        ? '✔ Verified — played the full reference gauntlet'
        : '~ Live estimate — from games already recorded'
      host.appendChild(badge)
    }

    var row = document.createElement('div')
    row.className = 'lb-row'
    var gamesIn = document.createElement('input')
    gamesIn.type = 'number'
    gamesIn.value = String((record && record.gamesPerRung) || 40)
    var field = document.createElement('div')
    field.className = 'lb-field'
    var lbl = document.createElement('label')
    lbl.textContent = 'Games / rung'
    field.appendChild(lbl)
    field.appendChild(gamesIn)
    var btn = document.createElement('button')
    // Secondary once a board exists — Improve keeps it live, so this is the optional "confirm with real games" step.
    btn.className = 'lb-btn' + (record ? ' secondary' : '')
    btn.type = 'button'
    btn.textContent = opts.rating ? 'Playing gauntlet…' : verified ? 'Re-verify (gauntlet)' : 'Verify on full gauntlet'
    btn.disabled = !!opts.rating
    btn.addEventListener('click', function () {
      if (typeof opts.onRate === 'function') opts.onRate({ gamesPerRung: num(gamesIn.value, 40) })
    })
    row.appendChild(field)
    row.appendChild(btn)
    host.appendChild(row)

    var entries = record && Array.isArray(record.entries) ? record.entries : []
    if (!entries.length) {
      var empty = document.createElement('div')
      empty.className = 'lb-empty'
      empty.textContent = opts.rating
        ? 'Playing the gauntlet…'
        : 'No leaderboard yet — press Improve above to start (the board fills after the first generation), or play the full gauntlet now.'
      host.appendChild(empty)
      return
    }

    var table = document.createElement('table')
    var rows = entries
      .map(function (e, i) {
        var climb = (e.pairings || [])
          .map(function (p) {
            return esc(p.opponent) + ':' + (typeof p.score === 'number' ? p.score.toFixed(2) : '?')
          })
          .join(' ')
        var flags = (e.flags || []).length ? '<span class="lb-flag">⚠ ' + esc(e.flags.join(', ')) + '</span>' : ''
        return (
          '<tr><td>' +
          (i + 1) +
          '</td><td>' +
          esc(e.modelName || '') +
          ' <code>' +
          esc(String(e.runKey).slice(0, 8)) +
          '</code></td><td class="lb-rating">' +
          esc(e.rating) +
          '</td><td>' +
          esc(e.lowerBound) +
          '</td><td>[' +
          esc(e.ciLow) +
          ', ' +
          esc(e.ciHigh) +
          ']</td><td>' +
          esc(e.games) +
          '</td><td>' +
          flags +
          '</td><td class="lb-climb">' +
          climb +
          '</td></tr>'
        )
      })
      .join('')
    table.innerHTML =
      '<tr><th>#</th><th>Model</th><th>Rating</th><th>Lower bound</th><th>95% CI</th><th>Games</th><th>Flags</th><th>Climbed</th></tr>' +
      rows
    host.appendChild(table)
  }

  root.Leaderboard = { render: render }
})(typeof window !== 'undefined' ? window : this)
