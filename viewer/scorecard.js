// Pure scorecard logic for the viewer — the browser twin of the engine's `computeScorecard`
// (src/modelTrainerUtils.ts). A run's SCORECARD (gates + fitness) is the machine-readable definition of
// SUCCESS, computed from the run summary INDEPENDENTLY of the reward/objective: the run table sorts by it
// (accepted-first, then primary fitness) and filters on acceptance. Kept byte-for-byte behaviourally equal
// to the engine version (same cases in src/scorecardViewer.test.ts) so the viewer and the server agree on
// which runs are "good". Pure + dual-loaded (browser `window.Scorecard` + node `module.exports`).
// A `manifest` is `{ objective:{name,direction}, gates?:[{metric,op,value,label}], fitness?:[{metric,direction}] }`;
// a `run` is `{ objective:number, metrics?:{...} }` (a RunRow's `summary`).
;(function (root) {
  'use strict'

  function readMetric(run, key) {
    var raw = key === 'objective' ? run && run.objective : run && run.metrics && run.metrics[key]
    return typeof raw === 'number' && isFinite(raw) ? raw : NaN
  }

  function applyOp(actual, op, bound) {
    if (!isFinite(actual) || !isFinite(bound)) return false
    switch (op) {
      case '>':
        return actual > bound
      case '>=':
        return actual >= bound
      case '<':
        return actual < bound
      case '<=':
        return actual <= bound
      case '==':
        return actual === bound
      case '!=':
        return actual !== bound
      default:
        return false
    }
  }

  // A run's scorecard: { gates:[{label,metric,op,bound,actual,pass}], accepted, fitness:[{metric,direction,value}] }.
  function computeScorecard(manifest, run) {
    manifest = manifest || {}
    var objective = manifest.objective || { name: 'objective', direction: 'max' }
    var gateSpecs = manifest.gates || []
    var gates = gateSpecs.map(function (g) {
      var actual = readMetric(run, g.metric)
      var isRef = g.value && typeof g.value === 'object'
      var bound = isRef ? readMetric(run, g.value.metric) : g.value
      var rendered = isRef ? g.value.metric : String(g.value)
      return {
        label: g.label || g.metric + ' ' + g.op + ' ' + rendered,
        metric: g.metric,
        op: g.op,
        bound: bound,
        actual: actual,
        pass: applyOp(actual, g.op, bound),
      }
    })
    var fitnessSpec =
      manifest.fitness && manifest.fitness.length
        ? manifest.fitness
        : [{ metric: 'objective', direction: objective.direction }]
    var fitness = fitnessSpec.map(function (f) {
      return { metric: f.metric, direction: f.direction, value: readMetric(run, f.metric) }
    })
    return {
      gates: gates,
      accepted: gates.every(function (g) {
        return g.pass
      }),
      fitness: fitness,
    }
  }

  // The scalar ranking criterion (first fitness objective, else the objective) — matches the engine.
  function primaryFitnessCriterion(manifest) {
    manifest = manifest || {}
    var primary = manifest.fitness && manifest.fitness[0]
    if (primary) return { key: primary.metric, direction: primary.direction }
    var dir = (manifest.objective && manifest.objective.direction) || 'max'
    return { key: 'objective', direction: dir }
  }

  // Primary fitness oriented so HIGHER is better; -Infinity when unrankable.
  function scorecardRankValue(scorecard) {
    var primary = scorecard && scorecard.fitness && scorecard.fitness[0]
    if (!primary || !isFinite(primary.value)) return -Infinity
    return primary.direction === 'min' ? -primary.value : primary.value
  }

  // Best-first comparator: accepted before rejected, then by oriented primary fitness (higher first).
  function compareScorecards(a, b) {
    if (a.accepted !== b.accepted) return a.accepted ? -1 : 1
    return scorecardRankValue(b) - scorecardRankValue(a)
  }

  // A single sortable scalar (higher = better) for the numeric run-table column: accepted runs always
  // outrank rejected ones, then by primary fitness. atan/π maps the unbounded fitness (incl. the ±Infinity
  // unrankable sentinel) into (-0.5, 0.5) while staying order-preserving across realistic metric ranges
  // (tanh would saturate by |x|~20); the +3 band then lifts every accepted run clear of every rejected one.
  // This is the scalar form of compareScorecards for a column framework that sorts by numbers, not comparators.
  function scorecardSortValue(scorecard) {
    return (scorecard.accepted ? 3 : 0) + Math.atan(scorecardRankValue(scorecard)) / Math.PI
  }

  // True iff the project declares a scorecard (gates or fitness) — i.e. success ≠ the raw objective.
  function hasScorecard(manifest) {
    return !!(manifest && ((manifest.gates && manifest.gates.length) || (manifest.fitness && manifest.fitness.length)))
  }

  var Scorecard = {
    computeScorecard: computeScorecard,
    primaryFitnessCriterion: primaryFitnessCriterion,
    scorecardRankValue: scorecardRankValue,
    compareScorecards: compareScorecards,
    scorecardSortValue: scorecardSortValue,
    hasScorecard: hasScorecard,
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = Scorecard
  if (root) root.Scorecard = Scorecard
})(typeof window !== 'undefined' ? window : null)
