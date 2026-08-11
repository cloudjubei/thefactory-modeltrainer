/*
 * Game replay view — step through a run's SAMPLED GAME move by move, in-app.
 *
 * A board-game run captures one game under its summary's `sample_game` ({ model_seat, winner, opponent,
 * moves:[{player,action,label}], frames:[ascii,...] }) where frames[0] is the initial board and frames[i+1]
 * is the board AFTER moves[i]. This module builds an interactive stepper (first/prev/next/last, play/pause,
 * a scrubber, and a move caption) into a host element — self-contained (its own scoped CSS + listeners), so
 * the main app only has to hand it the host div and the sample_game.
 *
 * window.Game.render(hostEl, sampleGame)
 */
;(function (root) {
  'use strict'

  var CSS =
    '.game-replay{display:flex;flex-direction:column;gap:8px;margin:6px 0 8px}' +
    '.game-replay .g-head{font-size:12px;color:var(--muted)}' +
    '.game-replay .g-board{color:var(--text);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;' +
    'font-size:15px;line-height:1.4;white-space:pre;overflow-x:auto;margin:0;padding:12px;border-radius:9px;' +
    'background:var(--surface-subtle);border:1px solid var(--border)}' +
    '.game-replay .g-cap{font-size:13px;min-height:1.2em;color:var(--text)}' +
    '.game-replay .g-cap .g-win{font-weight:600}' +
    '.game-replay .g-controls{display:flex;align-items:center;gap:6px;flex-wrap:wrap}' +
    '.game-replay .g-btn{min-width:34px;min-height:30px;padding:2px 8px;border-radius:6px;cursor:pointer;' +
    'background:var(--surface);color:var(--text);border:1px solid var(--border);font-size:14px}' +
    '.game-replay .g-btn:hover{border-color:var(--accent)}' +
    '.game-replay .g-btn:disabled{opacity:.4;cursor:default}' +
    '.game-replay .g-range{flex:1;min-width:120px;accent-color:var(--accent)}' +
    '.game-replay .g-count{font-variant-numeric:tabular-nums;font-size:12px;color:var(--muted);min-width:74px;text-align:right}'

  function ensureCss() {
    if (typeof document === 'undefined' || document.getElementById('game-replay-css')) return
    var st = document.createElement('style')
    st.id = 'game-replay-css'
    st.textContent = CSS
    document.head.appendChild(st)
  }

  function seatLabel(seat, modelSeat, opponent) {
    return seat === modelSeat ? 'model' : opponent || 'opponent'
  }

  function resultText(winner, modelSeat, opponent) {
    if (winner === null || winner === undefined) return 'Draw'
    var who = winner === modelSeat ? 'model' : opponent || 'opponent'
    return who + ' wins'
  }

  function captionFor(idx, moves, modelSeat, opponent) {
    if (idx <= 0) return 'Initial position'
    var mv = moves[idx - 1]
    if (!mv) return ''
    var who = seatLabel(mv.player, modelSeat, opponent)
    var label = mv.label != null ? mv.label : 'action ' + mv.action
    return 'Move ' + idx + ': ' + who + ' plays ' + label
  }

  function render(host, sample) {
    if (!host || !sample) return
    var frames = Array.isArray(sample.frames) ? sample.frames : []
    if (!frames.length) return
    ensureCss()
    var moves = Array.isArray(sample.moves) ? sample.moves : []
    var modelSeat = typeof sample.model_seat === 'number' ? sample.model_seat : 0
    var opponent = sample.opponent || 'opponent'
    var winner = sample.winner === undefined ? null : sample.winner
    var last = frames.length - 1
    var idx = 0
    var timer = null

    host.className = 'game-replay'
    host.setAttribute('tabindex', '0')
    host.innerHTML = ''

    var head = document.createElement('div')
    head.className = 'g-head'
    head.textContent =
      'Sampled evaluation game — model is seat ' +
      modelSeat +
      ' vs ' +
      opponent +
      '  ·  ' +
      resultText(winner, modelSeat, opponent)

    var board = document.createElement('pre')
    board.className = 'g-board'

    var cap = document.createElement('div')
    cap.className = 'g-cap'

    var controls = document.createElement('div')
    controls.className = 'g-controls'

    function mkBtn(label, title) {
      var b = document.createElement('button')
      b.className = 'g-btn'
      b.type = 'button'
      b.textContent = label
      b.title = title
      return b
    }
    var bFirst = mkBtn('⏮', 'First')
    var bPrev = mkBtn('◀', 'Previous')
    var bPlay = mkBtn('▶', 'Play')
    var bNext = mkBtn('▶', 'Next')
    var bLast = mkBtn('⏭', 'Last')
    bNext.textContent = '▶'
    bPrev.textContent = '◀'

    var range = document.createElement('input')
    range.className = 'g-range'
    range.type = 'range'
    range.min = '0'
    range.max = String(last)
    range.step = '1'
    range.value = '0'

    var count = document.createElement('span')
    count.className = 'g-count'

    controls.appendChild(bFirst)
    controls.appendChild(bPrev)
    controls.appendChild(bPlay)
    controls.appendChild(bNext)
    controls.appendChild(bLast)
    controls.appendChild(range)
    controls.appendChild(count)

    host.appendChild(head)
    host.appendChild(board)
    host.appendChild(cap)
    host.appendChild(controls)

    function stop() {
      if (timer) {
        clearInterval(timer)
        timer = null
      }
      bPlay.textContent = '▶'
      bPlay.title = 'Play'
    }

    function update() {
      if (idx < 0) idx = 0
      if (idx > last) idx = last
      board.textContent = frames[idx]
      range.value = String(idx)
      count.textContent = 'frame ' + idx + ' / ' + last
      var text = captionFor(idx, moves, modelSeat, opponent)
      if (idx === last) {
        cap.innerHTML =
          (text ? escapeHtml(text) + '  ·  ' : '') +
          '<span class="g-win">' +
          escapeHtml(resultText(winner, modelSeat, opponent)) +
          '</span>'
      } else {
        cap.textContent = text
      }
      bFirst.disabled = bPrev.disabled = idx <= 0
      bLast.disabled = bNext.disabled = idx >= last
    }

    function go(to) {
      stop()
      idx = to
      update()
    }

    bFirst.addEventListener('click', function () {
      go(0)
    })
    bPrev.addEventListener('click', function () {
      go(idx - 1)
    })
    bNext.addEventListener('click', function () {
      go(idx + 1)
    })
    bLast.addEventListener('click', function () {
      go(last)
    })
    range.addEventListener('input', function () {
      go(parseInt(range.value, 10) || 0)
    })
    bPlay.addEventListener('click', function () {
      if (timer) {
        stop()
        return
      }
      if (idx >= last) idx = 0
      bPlay.textContent = '⏸'
      bPlay.title = 'Pause'
      timer = setInterval(function () {
        if (idx >= last) {
          stop()
          return
        }
        idx += 1
        update()
      }, 700)
    })
    host.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowLeft') {
        go(idx - 1)
        e.preventDefault()
      } else if (e.key === 'ArrowRight') {
        go(idx + 1)
        e.preventDefault()
      }
    })

    update()
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
  }

  root.Game = { render: render }
})(typeof window !== 'undefined' ? window : this)
