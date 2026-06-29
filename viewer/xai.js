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
  function observedValues(runs, lever) {
    // Exclude the doesn't-apply sentinel — a conditional lever is swept only over the values it really takes.
    var values = distinctValues(runs, lever)
    values.delete('n/a')
    return values
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
      return lever in r.config && String(r.config[lever]) !== 'n/a'
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
        if (v === 'n/a') return // a conditional lever is scored only where it actually applies
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
    var all = acquisitionRecommendations(valid, criterion)
      .concat(thinSeedRecommendations(valid, criterion))
      .concat(missingCellRecommendations(valid, criterion))
      .sort(function (a, b) {
        return b.priority - a.priority
      })
    // Dedup by the launched config (acquisition can coincide with a missing-cell gap); keep the higher-priority.
    var seen = {}
    return all.filter(function (rec) {
      var key = canonicalConfigString((rec.spec && rec.spec.fixed) || {})
      if (seen[key]) return false
      seen[key] = true
      return true
    })
  }

  // --- Phase 3 mirror: seeded RF surrogate + ablation/fANOVA/interaction (must match xaiUtils.ts exactly) ---
  var SURROGATE_TREES = 64
  var SURROGATE_MAX_DEPTH = 8
  var SURROGATE_MIN_LEAF = 2
  // Phase 2 acquisition (mirror of xaiUtils.ts).
  var MAX_ACQUISITION_CANDIDATES = 2000
  var TOP_ACQUISITION_RECS = 5

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

  // Forest mean + epistemic std (tree disagreement) at a config — the explore signal EI balances.
  function predictConfigStats(surrogate, config) {
    var trees = surrogate.trees
    if (!trees.length) return { mean: surrogate.mean, std: 0 }
    var preds = trees.map(function (t) {
      return predictSurrogateTree(t, config)
    })
    return { mean: meanOf(preds), std: Math.sqrt(varianceOf(preds)) }
  }

  function erf(x) {
    var sign = x < 0 ? -1 : 1
    var ax = Math.abs(x)
    var t = 1 / (1 + 0.3275911 * ax)
    var y =
      1 -
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
        t *
        Math.exp(-ax * ax)
    return sign * y
  }
  function normalCdf(z) {
    return 0.5 * (1 + erf(z / Math.SQRT2))
  }
  function normalPdf(z) {
    return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI)
  }

  // Expected Improvement over the incumbent `best` (direction-aware), closed form under a Gaussian.
  function expectedImprovement(mean, std, best, direction) {
    var improve = direction === 'min' ? best - mean : mean - best
    if (std <= 1e-9) return Math.max(0, improve)
    var z = improve / std
    return improve * normalCdf(z) + std * normalPdf(z)
  }

  function cappedCartesian(lists, cap, rng) {
    var total = lists.reduce(function (a, l) {
      return a * Math.max(1, l.length)
    }, 1)
    if (total <= cap) {
      var acc = [[]]
      for (var i = 0; i < lists.length; i++) {
        var next = []
        for (var j = 0; j < acc.length; j++)
          for (var k = 0; k < lists[i].length; k++) next.push(acc[j].concat([lists[i][k]]))
        acc = next
      }
      return acc
    }
    var out = []
    var seen = {}
    for (var attempts = 0; out.length < cap && attempts < cap * 4; attempts++) {
      var combo = lists.map(function (l) {
        return l[Math.floor(rng() * l.length)]
      })
      var key = combo.map(String).join('')
      if (seen[key]) continue
      seen[key] = true
      out.push(combo)
    }
    return out
  }

  function fmtAcq(x) {
    if (!Number.isFinite(x)) return 'n/a'
    return Math.abs(x) >= 100 ? x.toFixed(1) : Number(x.toPrecision(3)).toString()
  }

  // Score every UNRUN config in the explored grid by Expected Improvement, surface the top few — the
  // "which way to explore" climb toward the optimum. Deterministic (seeded surrogate + sampler).
  function acquisitionRecommendations(valid, criterion) {
    var surrogate = fitConfigSurrogate(valid, criterion)
    if (!surrogate.trees.length || surrogate.levers.length === 0) return []
    var ys = valid.map(function (r) {
      return criterionValueOf(r, criterion)
    })
    var best = criterion.direction === 'min' ? Math.min.apply(null, ys) : Math.max.apply(null, ys)
    var observed = {}
    valid.forEach(function (r) {
      observed[canonicalConfigString(configWithout(r.config, 'seed'))] = true
    })
    var rng = makeRng(seedFrom(ys))
    var valueLists = surrogate.levers.map(function (l) {
      return Array.from(distinctValues(valid, l.name).values())
    })
    var scored = cappedCartesian(valueLists, MAX_ACQUISITION_CANDIDATES, rng)
      .map(function (combo) {
        var config = {}
        surrogate.levers.forEach(function (l, i) {
          config[l.name] = combo[i]
        })
        return config
      })
      .filter(function (config) {
        return !observed[canonicalConfigString(config)]
      })
      .map(function (config) {
        var st = predictConfigStats(surrogate, config)
        return {
          config: config,
          mean: st.mean,
          std: st.std,
          ei: expectedImprovement(st.mean, st.std, best, criterion.direction),
        }
      })
      .filter(function (s) {
        return s.ei > 0
      })
      .sort(function (a, b) {
        return b.ei - a.ei
      })
    return scored.slice(0, TOP_ACQUISITION_RECS).map(function (s, i) {
      return {
        kind: 'acquisition',
        reason:
          'Surrogate predicts ' +
          fmtAcq(s.mean) +
          ' ± ' +
          fmtAcq(s.std) +
          ' (best so far ' +
          fmtAcq(best) +
          ') — expected improvement ' +
          fmtAcq(s.ei) +
          '; the strongest unrun config toward the optimum.',
        runCount: 1,
        spec: { fixed: s.config, seeds: [0] },
        priority: 90 - i,
      }
    })
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
      var values = Array.from(observedValues(valid, lv.name).values())
      if (values.length < 2) return
      // Score the lever only over the configs where it APPLIES (its value isn't the sentinel).
      var applicable = configs.filter(function (c) {
        return String(c[lv.name]) !== 'n/a'
      })
      if (!applicable.length) return
      var marginals = values.map(function (v) {
        return meanOf(
          applicable.map(function (c) {
            var next = Object.assign({}, c)
            next[lv.name] = v
            return predictConfig(surrogate, next)
          }),
        )
      })
      // TOTAL effect: per config, the variance from sweeping THIS lever (interactions count), averaged.
      var perConfigVar = applicable.map(function (c) {
        return varianceOf(
          values.map(function (v) {
            var next = Object.assign({}, c)
            next[lv.name] = v
            return predictConfig(surrogate, next)
          }),
        )
      })
      out.push({
        lever: lv.name,
        importance: varianceOf(marginals) / totalVar,
        total: meanOf(perConfigVar) / totalVar,
        values: values.length,
        // The distinct observed values themselves, so the viewer can link each to its runs.
        valueList: values,
      })
    })
    return out.sort(function (a, b) {
      return b.importance - a.importance
    })
  }

  // Pairwise coupling strength (2-way ANOVA interaction term / total variance), sorted strongest-first.
  function leverCouplings(surrogate, runs, criterion) {
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
    var grand = meanOf(
      configs.map(function (c) {
        return predictConfig(surrogate, c)
      }),
    )
    var swept = surrogate.levers
      .map(function (l) {
        return l.name
      })
      .filter(function (n) {
        return observedValues(valid, n).size >= 2
      })
    var mainEffect = function (lever) {
      var m = new Map()
      observedValues(valid, lever).forEach(function (v, k) {
        m.set(
          k,
          meanOf(
            configs.map(function (c) {
              var next = Object.assign({}, c)
              next[lever] = v
              return predictConfig(surrogate, next)
            }),
          ),
        )
      })
      return m
    }
    var out = []
    for (var i = 0; i < swept.length; i++) {
      for (var j = i + 1; j < swept.length; j++) {
        var lA = swept[i]
        var lB = swept[j]
        var valsA = observedValues(valid, lA)
        var valsB = observedValues(valid, lB)
        var mainA = mainEffect(lA)
        var mainB = mainEffect(lB)
        var residuals = []
        valsA.forEach(function (va, ka) {
          valsB.forEach(function (vb, kb) {
            var joint = meanOf(
              configs.map(function (c) {
                var next = Object.assign({}, c)
                next[lA] = va
                next[lB] = vb
                return predictConfig(surrogate, next)
              }),
            )
            residuals.push(joint - mainA.get(ka) - mainB.get(kb) + grand)
          })
        })
        out.push({ leverA: lA, leverB: lB, strength: varianceOf(residuals) / totalVar })
      }
    }
    return out.sort(function (a, b) {
      return b.strength - a.strength
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

  function interactionGrid(surrogate, runs, criterion, leverA, leverB, appliesWhen) {
    var valid = validRunsFor(runs, criterion)
    var configs = valid.map(function (r) {
      return r.config
    })
    var valsA = observedValues(valid, leverA)
    var valsB = observedValues(valid, leverB)
    if (valsA.size < 2 || valsB.size < 2 || !configs.length) return undefined
    var valuesA = Array.from(valsA.keys())
    var valuesB = Array.from(valsB.keys())
    var conditionApplies = function (cfg, conds) {
      for (var k in conds) {
        if (conds[k].map(String).indexOf(String(cfg[k])) === -1) return false
      }
      return true
    }
    var normAW = function (cfg) {
      var out = cfg
      var changed = true
      while (changed) {
        changed = false
        for (var lever in appliesWhen) {
          if (lever in out && out[lever] !== 'n/a' && !conditionApplies(out, appliesWhen[lever])) {
            if (out === cfg) out = Object.assign({}, cfg)
            out[lever] = 'n/a'
            changed = true
          }
        }
      }
      return out
    }
    // A cell whose (leverA=a, leverB=b) combination can't occur (a conditional lever gets pinned to 'n/a'
    // once the what-if config is normalized) is left null, not a misleading surrogate extrapolation.
    var cells = []
    valuesA.forEach(function (a) {
      valuesB.forEach(function (b) {
        var av = valsA.get(a)
        var bv = valsB.get(b)
        var preds = []
        configs.forEach(function (c) {
          var next = Object.assign({}, c)
          next[leverA] = av
          next[leverB] = bv
          if (appliesWhen) {
            next = normAW(next)
            if (String(next[leverA]) !== String(av) || String(next[leverB]) !== String(bv)) return
          }
          preds.push(predictConfig(surrogate, next))
        })
        cells.push(preds.length ? meanOf(preds) : null)
      })
    })
    return { leverA: leverA, leverB: leverB, valuesA: valuesA, valuesB: valuesB, cells: cells }
  }

  // Phase 4 mirror: deterministic PCA projection (one point per setup, coloured by criterion).
  function dotVec(a, b) {
    var s = 0
    for (var i = 0; i < a.length; i++) s += a[i] * b[i]
    return s
  }
  function normalizeVec(v) {
    var norm = Math.sqrt(dotVec(v, v)) || 1
    return v.map(function (x) {
      return x / norm
    })
  }
  function topEigenpair(m, d) {
    var v = normalizeVec(
      Array.from({ length: d }, function (_, i) {
        return Math.sin(i + 1) + 1e-3
      }),
    )
    var value = 0
    for (var iter = 0; iter < 200; iter++) {
      var mv = m.map(function (row) {
        return dotVec(row, v)
      })
      value = dotVec(v, mv)
      v = normalizeVec(mv)
    }
    return { vector: v, value: value }
  }
  function pcaProjection(runs, criterion) {
    var valid = validRunsFor(runs, criterion)
    if (valid.length < 3) return null
    var bySetup = new Map()
    valid.forEach(function (r) {
      var sig = setupSignatureOf(r)
      var g = bySetup.get(sig)
      if (g) g.push(r)
      else bySetup.set(sig, [r])
    })
    var rows = Array.from(bySetup.values()).map(function (rs) {
      return {
        config: configWithout(rs[0].config, 'seed'),
        value: iqm(
          rs.map(function (r) {
            return criterionValueOf(r, criterion)
          }),
        ),
        runKeys: rs.map(function (r) {
          return r.key
        }),
      }
    })
    if (rows.length < 3) return null
    var columns = []
    leverKindsOf(valid).forEach(function (l) {
      if (l.kind === 'num') {
        columns.push(function (c) {
          var v = Number(c[l.name])
          return Number.isFinite(v) ? v : 0
        })
      } else {
        var vals = new Set(
          rows.map(function (r) {
            return String(r.config[l.name])
          }),
        )
        vals.forEach(function (val) {
          columns.push(function (c) {
            return String(c[l.name]) === val ? 1 : 0
          })
        })
      }
    })
    var d = columns.length
    if (d === 0) return null
    var raw = rows.map(function (r) {
      return columns.map(function (col) {
        return col(r.config)
      })
    })
    var n = raw.length
    var means = []
    var stds = []
    for (var j = 0; j < d; j++) {
      var colv = raw.map(function (row) {
        return row[j]
      })
      means.push(meanOf(colv))
      stds.push(Math.sqrt(varianceOf(colv)) || 1)
    }
    var X = raw.map(function (row) {
      return row.map(function (v, jj) {
        return (v - means[jj]) / stds[jj]
      })
    })
    var C = Array.from({ length: d }, function () {
      return new Array(d).fill(0)
    })
    for (var a = 0; a < d; a++) {
      for (var b = a; b < d; b++) {
        var s = 0
        for (var i = 0; i < n; i++) s += X[i][a] * X[i][b]
        C[a][b] = C[b][a] = s / n
      }
    }
    var trace =
      C.reduce(function (acc, row, ii) {
        return acc + row[ii]
      }, 0) || 1
    var pc1 = topEigenpair(C, d)
    var C2 = C.map(function (row, aa) {
      return row.map(function (val, bb) {
        return val - pc1.value * pc1.vector[aa] * pc1.vector[bb]
      })
    })
    var pc2 = d >= 2 ? topEigenpair(C2, d) : { vector: new Array(d).fill(0), value: 0 }
    var points = rows.map(function (r, ii) {
      return {
        x: dotVec(X[ii], pc1.vector),
        y: dotVec(X[ii], pc2.vector),
        value: r.value,
        key: r.runKeys[0],
        runKeys: r.runKeys,
      }
    })
    return {
      points: points,
      explainedVariance: [Math.max(0, pc1.value) / trace, Math.max(0, pc2.value) / trace],
      features: d,
    }
  }

  // Indices of the non-dominated points — the Pareto frontier. directions[k]: 'max' higher-better, 'min'
  // lower-better. A point dominates another when it's >= on every axis and strictly better on one.
  function paretoFrontier(points, directions) {
    var atLeast = function (x, y, dir) {
      return dir === 'min' ? x <= y : x >= y
    }
    var better = function (x, y, dir) {
      return dir === 'min' ? x < y : x > y
    }
    var dominates = function (a, b) {
      var strictly = false
      for (var k = 0; k < directions.length; k++) {
        if (!atLeast(a[k], b[k], directions[k])) return false
        if (better(a[k], b[k], directions[k])) strictly = true
      }
      return strictly
    }
    var out = []
    for (var i = 0; i < points.length; i++) {
      var dominated = false
      for (var j = 0; j < points.length; j++) {
        if (i !== j && dominates(points[j], points[i])) {
          dominated = true
          break
        }
      }
      if (!dominated) out.push(i)
    }
    return out
  }

  // A conditional lever (one with `appliesWhen`) applies to a config only when EVERY named control lever
  // currently holds one of its allowed values (AND). Mirrors the launch form's `leverApplies`.
  function leverConfigApplies(spec, config) {
    if (!spec || !spec.appliesWhen) return true
    for (var k in spec.appliesWhen) {
      var allowed = spec.appliesWhen[k]
      var arr = (Array.isArray(allowed) ? allowed : [allowed]).map(String)
      if (arr.indexOf(String(config[k])) === -1) return false
    }
    return true
  }

  // Pin each conditional model lever to the 'n/a' sentinel on configs where it doesn't apply (e.g.
  // forward_horizon on a PPO run), so it reads n/a in compare/detail and is excluded from importance —
  // matching the server-side config-space analysis. Returns a COPY; never mutates the input.
  function normalizeConditionalConfig(config, levers) {
    if (!config || !levers) return config || {}
    var out = {}
    for (var key in config) out[key] = config[key]
    // Fixpoint cascade (see normalizeConditionalLevers in xaiUtils.ts): a conditional lever controlled by
    // another conditional lever pins to 'n/a' once its controller does, regardless of key order.
    var changed = true
    while (changed) {
      changed = false
      for (var name in levers) {
        var spec = levers[name]
        if (name in out && out[name] !== 'n/a' && spec && spec.appliesWhen && !leverConfigApplies(spec, out)) {
          out[name] = 'n/a'
          changed = true
        }
      }
    }
    return out
  }

  var Xai = {
    iqm: iqm,
    normalizeConditionalConfig: normalizeConditionalConfig,
    paretoFrontier: paretoFrontier,
    aggregateRunValues: aggregateRunValues,
    criterionValueOf: criterionValueOf,
    ofatContrasts: ofatContrasts,
    leverImportances: leverImportances,
    fitConfigSurrogate: fitConfigSurrogate,
    predictConfig: predictConfig,
    predictConfigStats: predictConfigStats,
    expectedImprovement: expectedImprovement,
    fanovaImportances: fanovaImportances,
    leverCouplings: leverCouplings,
    ablationPath: ablationPath,
    interactionGrid: interactionGrid,
    pcaProjection: pcaProjection,
    recommendExperiments: recommendExperiments,
    // Exposed so the viewer can recompute a setup key with the SAME canonicalisation the engine uses.
    canonicalConfigString: canonicalConfigString,
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = Xai
  if (root) root.Xai = Xai
})(typeof window !== 'undefined' ? window : null)
