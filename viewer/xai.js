// The xAI config-effect engine — a faithful browser mirror of src/xaiUtils.ts (the viewer is sandboxed
// static JS and can't import the built engine). The TS engine stays the tested source of truth; a node
// parity harness (scripts/xaiParity check) asserts this mirror produces identical results. Deterministic:
// bootstrap intervals are seeded from the data so re-running analysis never drifts. Exposed as window.Xai.
;(function (root) {
  'use strict'

  var BOOTSTRAP_ITERATIONS = 2000
  var CI_LEVEL = 0.95
  var FDR_ALPHA = 0.1
  var MIN_SEEDS = 5
  var TOP_SETUPS_FOR_SEED_REC = 3
  var MAX_MISSING_CELL_RECS = 12

  function canonicalConfigString(value) {
    if (Array.isArray(value)) return '[' + value.map(canonicalConfigString).join(',') + ']'
    if (value && typeof value === 'object') {
      var entries = Object.entries(value)
        .sort(function (a, b) {
          return a[0] < b[0] ? -1 : 1
        })
        .map(function (e) {
          return JSON.stringify(e[0]) + ':' + canonicalConfigString(e[1])
        })
      return '{' + entries.join(',') + '}'
    }
    return JSON.stringify(value)
  }

  function makeRng(seed) {
    var s = seed >>> 0
    return function () {
      s = (s + 0x6d2b79f5) >>> 0
      var t = Math.imul(s ^ (s >>> 15), 1 | s)
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    }
  }

  function seedFrom(values) {
    var h = 2166136261
    var text = values.join(',')
    for (var i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619)
    return h >>> 0
  }

  function meanOf(values) {
    return values.length
      ? values.reduce(function (a, b) {
          return a + b
        }, 0) / values.length
      : 0
  }

  function medianOf(values) {
    if (!values.length) return 0
    var s = values.slice().sort(function (a, b) {
      return a - b
    })
    var m = Math.floor(s.length / 2)
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
  }

  function stdOf(values) {
    if (values.length < 2) return 0
    var m = meanOf(values)
    return Math.sqrt(
      values.reduce(function (a, b) {
        return a + (b - m) * (b - m)
      }, 0) /
        (values.length - 1),
    )
  }

  function iqm(values) {
    if (!values.length) return 0
    var sorted = values.slice().sort(function (a, b) {
      return a - b
    })
    var trim = Math.floor(sorted.length * 0.25)
    return meanOf(sorted.slice(trim, sorted.length - trim))
  }

  function percentileOf(sorted, p) {
    var idx = Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))
    return sorted[idx]
  }

  function resample(values, rng) {
    return values.map(function () {
      return values[Math.floor(rng() * values.length)]
    })
  }

  function aggregateRunValues(values) {
    var n = values.length
    if (!n) return { n: 0, mean: 0, iqm: 0, median: 0, std: 0, min: 0, max: 0, ci: [0, 0] }
    var point = iqm(values)
    var ci = [point, point]
    if (n > 1) {
      var rng = makeRng(seedFrom(values))
      var boot = []
      for (var i = 0; i < BOOTSTRAP_ITERATIONS; i++) boot.push(iqm(resample(values, rng)))
      boot.sort(function (a, b) {
        return a - b
      })
      var lo = (1 - CI_LEVEL) / 2
      ci = [percentileOf(boot, lo), percentileOf(boot, 1 - lo)]
    }
    return {
      n: n,
      mean: meanOf(values),
      iqm: point,
      median: medianOf(values),
      std: stdOf(values),
      min: Math.min.apply(null, values),
      max: Math.max.apply(null, values),
      ci: ci,
    }
  }

  function criterionValueOf(run, criterion) {
    var v
    if (criterion.key === 'objective') v = run.objective
    else if (criterion.key === 'durationMs') v = run.durationMs
    else v = run.metrics && run.metrics[criterion.key]
    return typeof v === 'number' && isFinite(v) ? v : undefined
  }

  function datasetSigOf(run) {
    var d = run.dataset
    if (!d || typeof d !== 'object') return ''
    return ['asset', 'timeframe', 'candles', 'from', 'to']
      .filter(function (k) {
        return d[k] !== undefined && d[k] !== null
      })
      .map(function (k) {
        return k + '=' + d[k]
      })
      .join('|')
  }

  function configWithout(config, omit) {
    var out = {}
    Object.keys(config).forEach(function (k) {
      if (omit.indexOf(k) < 0) out[k] = config[k]
    })
    return out
  }

  function controlSignatureOf(run, lever) {
    return (
      canonicalConfigString(configWithout(run.config, [lever, 'seed'])) + '||' + datasetSigOf(run)
    )
  }

  function orientedBetterFirst(direction) {
    return direction === 'max'
      ? function (a, b) {
          return b - a
        }
      : function (a, b) {
          return a - b
        }
  }

  function bootstrapDiff(toValues, fromValues, direction) {
    var orient = function (a, b) {
      return direction === 'max' ? a - b : b - a
    }
    var delta = orient(iqm(toValues), iqm(fromValues))
    if (toValues.length < 2 || fromValues.length < 2)
      return { ci: [delta, delta], pValue: 1, delta: delta }
    var rng = makeRng(
      seedFrom(
        toValues.concat([NaN], fromValues).map(function (v) {
          return isNaN(v) ? 0 : v
        }),
      ),
    )
    var diffs = []
    for (var i = 0; i < BOOTSTRAP_ITERATIONS; i++) {
      diffs.push(orient(iqm(resample(toValues, rng)), iqm(resample(fromValues, rng))))
    }
    diffs.sort(function (a, b) {
      return a - b
    })
    var lo = (1 - CI_LEVEL) / 2
    var ci = [percentileOf(diffs, lo), percentileOf(diffs, 1 - lo)]
    var below =
      diffs.filter(function (d) {
        return d <= 0
      }).length / diffs.length
    return { ci: ci, pValue: Math.min(1, 2 * Math.min(below, 1 - below)), delta: delta }
  }

  function benjaminiHochberg(pValues, alpha) {
    var m = pValues.length
    var rejected = new Array(m).fill(false)
    if (!m) return rejected
    var order = pValues
      .map(function (p, i) {
        return { p: p, i: i }
      })
      .sort(function (a, b) {
        return a.p - b.p
      })
    var maxK = -1
    for (var k = 0; k < m; k++) if (order[k].p <= ((k + 1) / m) * alpha) maxK = k
    for (var j = 0; j <= maxK; j++) rejected[order[j].i] = true
    return rejected
  }

  function distinctValues(runs, lever) {
    var out = new Map()
    runs.forEach(function (r) {
      if (lever in r.config) out.set(String(r.config[lever]), r.config[lever])
    })
    return out
  }

  function validRunsFor(runs, criterion) {
    return runs.filter(function (r) {
      return r.status === 'completed' && criterionValueOf(r, criterion) !== undefined
    })
  }

  function leversOf(runs) {
    var keys = new Set()
    runs.forEach(function (r) {
      Object.keys(r.config).forEach(function (k) {
        if (k !== 'seed') keys.add(k)
      })
    })
    return Array.from(keys)
  }

  function groupBy(items, keyFn) {
    var map = new Map()
    items.forEach(function (it) {
      var k = keyFn(it)
      var g = map.get(k)
      if (g) g.push(it)
      else map.set(k, [it])
    })
    return map
  }

  function ofatContrasts(runs, lever, criterion) {
    var valid = validRunsFor(runs, criterion).filter(function (r) {
      return lever in r.config
    })
    var groups = groupBy(valid, function (r) {
      return controlSignatureOf(r, lever)
    })
    var analyses = []
    groups.forEach(function (group, controlSignature) {
      var byValue = groupBy(group, function (r) {
        return String(r.config[lever])
      })
      if (byValue.size < 2) return
      var levelValues = new Map()
      var levels = []
      byValue.forEach(function (vruns, value) {
        var values = vruns.map(function (r) {
          return criterionValueOf(r, criterion)
        })
        levelValues.set(value, values)
        levels.push({
          value: value,
          runKeys: vruns.map(function (r) {
            return r.key
          }),
          seeds: new Set(
            vruns.map(function (r) {
              return r.seed == null ? 0 : r.seed
            }),
          ).size,
          aggregate: aggregateRunValues(values),
        })
      })
      levels.sort(function (a, b) {
        return orientedBetterFirst(criterion.direction)(a.aggregate.iqm, b.aggregate.iqm)
      })
      var baseline = levels[levels.length - 1]
      var baselineValues = levelValues.get(baseline.value)
      var effects = levels.slice(0, -1).map(function (level) {
        var d = bootstrapDiff(levelValues.get(level.value), baselineValues, criterion.direction)
        return {
          from: baseline.value,
          to: level.value,
          delta: d.delta,
          diffCi: d.ci,
          significant: false,
          pValue: d.pValue,
        }
      })
      var rejected = benjaminiHochberg(
        effects.map(function (e) {
          return e.pValue
        }),
        FDR_ALPHA,
      )
      effects.forEach(function (e, i) {
        e.significant = rejected[i] && (e.diffCi[0] > 0 || e.diffCi[1] < 0)
      })
      analyses.push({
        lever: lever,
        criterion: criterion,
        controlSignature: controlSignature,
        levels: levels,
        effects: effects,
      })
    })
    return analyses
  }

  function leverImportances(runs, criterion) {
    var valid = validRunsFor(runs, criterion)
    var raw = []
    leversOf(valid).forEach(function (lever) {
      var byValue = new Map()
      valid.forEach(function (r) {
        if (!(lever in r.config)) return
        var v = String(r.config[lever])
        var cv = criterionValueOf(r, criterion)
        var b = byValue.get(v)
        if (b) b.push(cv)
        else byValue.set(v, [cv])
      })
      if (byValue.size < 2) return
      var marginals = Array.from(byValue.entries()).map(function (e) {
        return { value: e[0], iqm: iqm(e[1]) }
      })
      var m = meanOf(
        marginals.map(function (x) {
          return x.iqm
        }),
      )
      var variance = meanOf(
        marginals.map(function (x) {
          return (x.iqm - m) * (x.iqm - m)
        }),
      )
      var sorted = marginals.slice().sort(function (a, b) {
        return orientedBetterFirst(criterion.direction)(a.iqm, b.iqm)
      })
      raw.push({
        lever: lever,
        variance: variance,
        values: byValue.size,
        best: sorted[0].value,
        worst: sorted[sorted.length - 1].value,
        minRuns: Math.min.apply(
          null,
          Array.from(byValue.values()).map(function (vals) {
            return vals.length
          }),
        ),
      })
    })
    var total =
      raw.reduce(function (a, b) {
        return a + b.variance
      }, 0) || 1
    return raw
      .map(function (r) {
        return {
          lever: r.lever,
          importance: r.variance / total,
          values: r.values,
          bestValue: r.best,
          worstValue: r.worst,
          minRuns: r.minRuns,
          confident: r.minRuns >= MIN_SEEDS,
        }
      })
      .sort(function (a, b) {
        return b.importance - a.importance
      })
  }

  function setupSignatureOf(run) {
    return canonicalConfigString(configWithout(run.config, ['seed'])) + '||' + datasetSigOf(run)
  }

  function freshSeeds(used, need) {
    var out = []
    var s = 0
    while (out.length < need) {
      if (!used.has(s)) out.push(s)
      s++
    }
    return out
  }

  function thinSeedRecommendations(valid, criterion) {
    var setups = groupBy(valid, setupSignatureOf)
    var ranked = Array.from(setups.values())
      .map(function (rs) {
        return {
          rs: rs,
          score: iqm(
            rs.map(function (r) {
              return criterionValueOf(r, criterion)
            }),
          ),
        }
      })
      .sort(function (a, b) {
        return orientedBetterFirst(criterion.direction)(a.score, b.score)
      })
    var recs = []
    ranked.slice(0, TOP_SETUPS_FOR_SEED_REC).forEach(function (entry) {
      var used = new Set(
        entry.rs.map(function (r) {
          return r.seed == null ? 0 : r.seed
        }),
      )
      if (used.size >= MIN_SEEDS) return
      var need = MIN_SEEDS - used.size
      recs.push({
        kind: 'thin-seeds',
        reason:
          'Top setup has ' +
          used.size +
          ' seed(s) — run ' +
          need +
          ' more for a trustworthy interval (≥' +
          MIN_SEEDS +
          ').',
        runCount: need,
        spec: { fixed: configWithout(entry.rs[0].config, ['seed']), seeds: freshSeeds(used, need) },
        priority: 100 - used.size,
      })
    })
    return recs
  }

  function missingCellRecommendations(valid, criterion) {
    var swept = leversOf(valid).filter(function (l) {
      return distinctValues(valid, l).size >= 2
    })
    var recs = []
    for (var i = 0; i < swept.length && recs.length < MAX_MISSING_CELL_RECS; i++) {
      for (var j = i + 1; j < swept.length && recs.length < MAX_MISSING_CELL_RECS; j++) {
        var lA = swept[i]
        var lB = swept[j]
        var contexts = new Map()
        valid.forEach(function (r) {
          if (!(lA in r.config) || !(lB in r.config)) return
          var sig =
            canonicalConfigString(configWithout(r.config, [lA, lB, 'seed'])) +
            '||' +
            datasetSigOf(r)
          var g = contexts.get(sig)
          if (g) g.push(r)
          else contexts.set(sig, [r])
        })
        contexts.forEach(function (group) {
          var valsA = distinctValues(group, lA)
          var valsB = distinctValues(group, lB)
          if (valsA.size < 2 || valsB.size < 2) return
          var observed = new Set(
            group.map(function (r) {
              return r.config[lA] + ' ' + r.config[lB]
            }),
          )
          var context = configWithout(group[0].config, [lA, lB, 'seed'])
          valsA.forEach(function (origA) {
            valsB.forEach(function (origB) {
              if (recs.length >= MAX_MISSING_CELL_RECS) return
              if (observed.has(origA + ' ' + origB)) return
              var fixed = Object.assign({}, context)
              fixed[lA] = origA
              fixed[lB] = origB
              recs.push({
                kind: 'missing-cell',
                reason:
                  'Untested cell: ' +
                  lA +
                  '=' +
                  origA +
                  ' × ' +
                  lB +
                  '=' +
                  origB +
                  ' (the rest of this grid was run).',
                runCount: 1,
                spec: { fixed: fixed, seeds: [0] },
                priority: 40,
              })
            })
          })
        })
      }
    }
    return recs
  }

  function recommendExperiments(runs, criterion) {
    var valid = validRunsFor(runs, criterion)
    if (!valid.length) return []
    return thinSeedRecommendations(valid, criterion)
      .concat(missingCellRecommendations(valid, criterion))
      .sort(function (a, b) {
        return b.priority - a.priority
      })
  }

  // --- Phase 3 mirror: seeded RF surrogate + ablation/fANOVA/interaction (must match xaiUtils.ts exactly) ---
  var SURROGATE_TREES = 64
  var SURROGATE_MAX_DEPTH = 8
  var SURROGATE_MIN_LEAF = 2

  function varianceOf(values) {
    if (values.length < 2) return 0
    var m = meanOf(values)
    return (
      values.reduce(function (a, b) {
        return a + (b - m) * (b - m)
      }, 0) / values.length
    )
  }

  function leverKindsOf(runs) {
    return leversOf(runs).map(function (name) {
      var allNumeric = runs.every(function (r) {
        return !(name in r.config) || typeof r.config[name] === 'number'
      })
      return { name: name, kind: allNumeric ? 'num' : 'cat' }
    })
  }

  function sampleSubset(items, k, rng) {
    var pool = items.slice()
    var out = []
    for (var i = 0; i < k && pool.length; i++)
      out.push(pool.splice(Math.floor(rng() * pool.length), 1)[0])
    return out
  }

  function buildSurrogateTree(rows, levers, rng, depth) {
    var ys = rows.map(function (r) {
      return r.y
    })
    var parentVar = varianceOf(ys)
    if (depth >= SURROGATE_MAX_DEPTH || rows.length <= SURROGATE_MIN_LEAF || parentVar === 0) {
      return { leaf: meanOf(ys) }
    }
    var tried = sampleSubset(levers, Math.max(1, Math.round(Math.sqrt(levers.length))), rng)
    var best
    var consider = function (lever, pick, extra) {
      var left = rows.filter(pick)
      var right = rows.filter(function (r) {
        return !pick(r)
      })
      if (!left.length || !right.length) return
      var score =
        parentVar -
        (left.length *
          varianceOf(
            left.map(function (r) {
              return r.y
            }),
          ) +
          right.length *
            varianceOf(
              right.map(function (r) {
                return r.y
              }),
            )) /
          rows.length
      if (!best || score > best.score) {
        best = {
          lever: lever.name,
          kind: lever.kind,
          threshold: extra.threshold,
          value: extra.value,
          left: left,
          right: right,
          score: score,
        }
      }
    }
    tried.forEach(function (lever) {
      if (lever.kind === 'num') {
        var nums = Array.from(
          new Set(
            rows
              .map(function (r) {
                return Number(r.config[lever.name])
              })
              .filter(Number.isFinite),
          ),
        ).sort(function (a, b) {
          return a - b
        })
        for (var i = 0; i + 1 < nums.length; i++) {
          ;(function (threshold) {
            consider(
              lever,
              function (r) {
                return Number(r.config[lever.name]) <= threshold
              },
              { threshold: threshold },
            )
          })((nums[i] + nums[i + 1]) / 2)
        }
      } else {
        new Set(
          rows.map(function (r) {
            return String(r.config[lever.name])
          }),
        ).forEach(function (value) {
          consider(
            lever,
            function (r) {
              return String(r.config[lever.name]) === value
            },
            { value: value },
          )
        })
      }
    })
    if (!best || best.score <= 0) return { leaf: meanOf(ys) }
    var left = buildSurrogateTree(best.left, levers, rng, depth + 1)
    var right = buildSurrogateTree(best.right, levers, rng, depth + 1)
    return best.kind === 'num'
      ? { lever: best.lever, kind: 'num', threshold: best.threshold, left: left, right: right }
      : { lever: best.lever, kind: 'cat', value: best.value, left: left, right: right }
  }

  function fitConfigSurrogate(runs, criterion) {
    var valid = validRunsFor(runs, criterion)
    var rows = valid.map(function (r) {
      return { config: r.config, y: criterionValueOf(r, criterion) }
    })
    var levers = leverKindsOf(valid)
    var mean = rows.length
      ? meanOf(
          rows.map(function (r) {
            return r.y
          }),
        )
      : 0
    if (rows.length < 2 || !levers.length) return { trees: [], levers: levers, mean: mean }
    var rng = makeRng(
      seedFrom(
        rows.map(function (r) {
          return r.y
        }),
      ),
    )
    var trees = []
    for (var t = 0; t < SURROGATE_TREES; t++) {
      var sample = rows.map(function () {
        return rows[Math.floor(rng() * rows.length)]
      })
      trees.push(buildSurrogateTree(sample, levers, rng, 0))
    }
    return { trees: trees, levers: levers, mean: mean }
  }

  function predictSurrogateTree(node, config) {
    var cur = node
    while (!('leaf' in cur)) {
      if (cur.kind === 'num') {
        var v = Number(config[cur.lever])
        cur = (Number.isFinite(v) ? v : cur.threshold) <= cur.threshold ? cur.left : cur.right
      } else {
        cur = String(config[cur.lever]) === cur.value ? cur.left : cur.right
      }
    }
    return cur.leaf
  }

  function predictConfig(surrogate, config) {
    var trees = surrogate.trees
    if (!trees.length) return surrogate.mean
    return meanOf(
      trees.map(function (t) {
        return predictSurrogateTree(t, config)
      }),
    )
  }

  function fanovaImportances(surrogate, runs, criterion) {
    var valid = validRunsFor(runs, criterion)
    var configs = valid.map(function (r) {
      return r.config
    })
    if (configs.length < 2) return []
    var totalVar =
      varianceOf(
        configs.map(function (c) {
          return predictConfig(surrogate, c)
        }),
      ) || 1
    var out = []
    surrogate.levers.forEach(function (lv) {
      var values = Array.from(distinctValues(valid, lv.name).values())
      if (values.length < 2) return
      var marginals = values.map(function (v) {
        return meanOf(
          configs.map(function (c) {
            var next = Object.assign({}, c)
            next[lv.name] = v
            return predictConfig(surrogate, next)
          }),
        )
      })
      out.push({
        lever: lv.name,
        importance: varianceOf(marginals) / totalVar,
        values: values.length,
        // The distinct observed values themselves, so the viewer can link each to its runs.
        valueList: values,
      })
    })
    return out.sort(function (a, b) {
      return b.importance - a.importance
    })
  }

  function ablationPath(surrogate, runs, criterion) {
    var valid = validRunsFor(runs, criterion)
    if (valid.length < 2) return undefined
    var sorted = valid.slice().sort(function (a, b) {
      return orientedBetterFirst(criterion.direction)(
        criterionValueOf(a, criterion),
        criterionValueOf(b, criterion),
      )
    })
    var incumbent = sorted[0].config
    var baseline = sorted[sorted.length - 1].config
    var orient = function (a, b) {
      return criterion.direction === 'max' ? a - b : b - a
    }
    var diffLevers = surrogate.levers
      .map(function (l) {
        return l.name
      })
      .filter(function (l) {
        return String(baseline[l]) !== String(incumbent[l])
      })
    if (!diffLevers.length) return undefined
    var baselinePredicted = predictConfig(surrogate, baseline)
    var steps = []
    var current = Object.assign({}, baseline)
    var prev = baselinePredicted
    var remaining = new Set(diffLevers)
    while (remaining.size) {
      var pick = undefined // reset each iteration — var is function-scoped, so a stale pick would loop forever
      remaining.forEach(function (lever) {
        var cand = Object.assign({}, current)
        cand[lever] = incumbent[lever]
        var predicted = predictConfig(surrogate, cand)
        var gain = orient(predicted, prev)
        if (!pick || gain > pick.gain) pick = { lever: lever, predicted: predicted, gain: gain }
      })
      if (!pick) break
      current = Object.assign({}, current)
      current[pick.lever] = incumbent[pick.lever]
      steps.push({
        lever: pick.lever,
        from: String(baseline[pick.lever]),
        to: String(incumbent[pick.lever]),
        predicted: pick.predicted,
        gain: pick.gain,
      })
      prev = pick.predicted
      remaining.delete(pick.lever)
    }
    return {
      baseline: baseline,
      incumbent: incumbent,
      baselinePredicted: baselinePredicted,
      incumbentPredicted: predictConfig(surrogate, incumbent),
      steps: steps,
    }
  }

  function interactionGrid(surrogate, runs, criterion, leverA, leverB) {
    var valid = validRunsFor(runs, criterion)
    var configs = valid.map(function (r) {
      return r.config
    })
    var valsA = distinctValues(valid, leverA)
    var valsB = distinctValues(valid, leverB)
    if (valsA.size < 2 || valsB.size < 2 || !configs.length) return undefined
    var valuesA = Array.from(valsA.keys())
    var valuesB = Array.from(valsB.keys())
    var cells = []
    valuesA.forEach(function (a) {
      valuesB.forEach(function (b) {
        cells.push(
          meanOf(
            configs.map(function (c) {
              var next = Object.assign({}, c)
              next[leverA] = valsA.get(a)
              next[leverB] = valsB.get(b)
              return predictConfig(surrogate, next)
            }),
          ),
        )
      })
    })
    return { leverA: leverA, leverB: leverB, valuesA: valuesA, valuesB: valuesB, cells: cells }
  }

  var Xai = {
    iqm: iqm,
    aggregateRunValues: aggregateRunValues,
    criterionValueOf: criterionValueOf,
    ofatContrasts: ofatContrasts,
    leverImportances: leverImportances,
    fitConfigSurrogate: fitConfigSurrogate,
    predictConfig: predictConfig,
    fanovaImportances: fanovaImportances,
    ablationPath: ablationPath,
    interactionGrid: interactionGrid,
    recommendExperiments: recommendExperiments,
    // Exposed so the viewer can recompute a setup key with the SAME canonicalisation the engine uses.
    canonicalConfigString: canonicalConfigString,
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = Xai
  if (root) root.Xai = Xai
})(typeof window !== 'undefined' ? window : null)
