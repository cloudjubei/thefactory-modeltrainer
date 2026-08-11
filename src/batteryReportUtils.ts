/**
 * Publication-grade evidence-battery report — PURE (a project's hypothesis/verdict trail → a self-contained,
 * shareable static HTML page). Domain-agnostic: the caller supplies the family labels and the narrative
 * sections; this only structures the trail and renders the shell + tables. No I/O, no DOM, no external assets.
 */
import type {
  BatteryBuildOptions,
  BatteryHypothesis,
  BatteryReportOptions,
  ExperimentBattery,
  ExperimentBatteryFamily,
} from './modelTrainerTypes.js'

const escapeHtml = (value: unknown): string =>
  String(value ?? '').replace(/[&<>"']/g, (ch) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch] as string,
  )

/** The number of backtested cells a spec's sweep declares (product of the swept-lever lengths; 1 for none). */
function cellCount(spec?: BatteryHypothesis['spec']): number {
  const sweep = spec?.sweep ?? {}
  return Object.values(sweep).reduce<number>(
    (n, v) => n * (Array.isArray(v) ? Math.max(v.length, 1) : 1),
    1,
  )
}

const gateLabel = (gate?: BatteryHypothesis['gate']): string =>
  !gate?.kind ? '' : gate.metric ? `${gate.kind}(${gate.metric})` : gate.kind

/** Fold trail records into families (signal classes) with roll-up cell counts + a status tally. */
export function buildBattery(hypotheses: BatteryHypothesis[], opts: BatteryBuildOptions = {}): ExperimentBattery {
  const familyOf = opts.familyOf ?? ((t: string) => t)
  const byFamily = new Map<string, ExperimentBatteryFamily>()
  for (const h of hypotheses) {
    const family = familyOf(h.type)
    let fam = byFamily.get(family)
    if (!fam) {
      fam = { type: h.type, family, probes: [], cells: 0 }
      byFamily.set(family, fam)
    }
    const cells = cellCount(h.spec)
    fam.probes.push({
      id: h.id,
      title: h.title ?? h.id,
      status: h.status ?? 'untested',
      gate: gateLabel(h.gate),
      cells,
    })
    fam.cells += cells
  }
  let families = [...byFamily.values()]
  if (opts.familyOrder) {
    const order = new Map(opts.familyOrder.map((f, i) => [f, i]))
    families = families.sort((a, b) => (order.get(a.family) ?? Number.MAX_SAFE_INTEGER) - (order.get(b.family) ?? Number.MAX_SAFE_INTEGER))
  }
  const byStatus: Record<string, number> = {}
  for (const f of families) for (const p of f.probes) byStatus[p.status] = (byStatus[p.status] ?? 0) + 1
  return {
    families,
    stats: {
      probes: families.reduce((n, f) => n + f.probes.length, 0),
      families: families.length,
      cellsRun: families.reduce((n, f) => n + f.cells, 0),
      byStatus,
    },
  }
}

// A completed null (disproved) is the desired result for a no-edge battery; inconclusive is an open thread; any
// other status (a surviving/passing hypothesis) is the one to highlight.
const statusClass = (status: string): string =>
  status === 'disproved' ? 'null' : status === 'inconclusive' ? 'open' : status === 'untested' ? 'pending' : 'surviving'

const STYLE = `
:root{--bg:#ffffff;--fg:#1b1f24;--muted:#5b6672;--line:#e3e7ec;--card:#f6f8fa;--accent:#2b6cb0;
--null:#2f855a;--null-bg:#e7f4ec;--open:#b7791f;--open-bg:#fbf3e2;--surviving:#c53030;--surviving-bg:#fdecec;--pending:#5b6672;--pending-bg:#eef1f4;}
@media(prefers-color-scheme:dark){:root{--bg:#0f1216;--fg:#e6e9ee;--muted:#9aa4b0;--line:#232a32;--card:#161b21;--accent:#63b3ed;
--null:#68d391;--null-bg:#16261d;--open:#f6c76b;--open-bg:#2a2313;--surviving:#fc8181;--surviving-bg:#2a1616;--pending:#9aa4b0;--pending-bg:#1a2029;}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:960px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:1.9rem;line-height:1.2;margin:0 0 6px}h2{font-size:1.3rem;margin:2.2em 0 .6em;padding-bottom:.3em;border-bottom:1px solid var(--line)}
h3{font-size:1.05rem;margin:1.4em 0 .5em}p{margin:.7em 0}.subtitle{color:var(--muted);font-size:1.05rem;margin:.2em 0 .2em}
.note{color:var(--muted);font-size:.85rem}a{color:var(--accent)}code{background:var(--card);padding:.1em .4em;border-radius:4px;font-size:.9em}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0 8px}
.chip{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:96px}
.chip .n{font-size:1.5rem;font-weight:700;display:block}.chip .l{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.chip.null .n{color:var(--null)}.chip.open .n{color:var(--open)}.chip.surviving .n{color:var(--surviving)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;margin:.4em 0 1.4em}
table{border-collapse:collapse;width:100%;font-size:.92rem}th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.03em}tr:last-child td{border-bottom:0}
td.id code{white-space:nowrap}td.cells{text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:.78rem;font-weight:600}
.badge.null{color:var(--null);background:var(--null-bg)}.badge.open{color:var(--open);background:var(--open-bg)}
.badge.surviving{color:var(--surviving);background:var(--surviving-bg)}.badge.pending{color:var(--pending);background:var(--pending-bg)}
.famcount{color:var(--muted);font-weight:400;font-size:.85rem}
footer{margin-top:2.4em;padding-top:1em;border-top:1px solid var(--line);color:var(--muted);font-size:.85rem}
`.trim()

function renderFamily(fam: ExperimentBatteryFamily): string {
  const rows = fam.probes
    .map(
      (p) =>
        `<tr><td class="id"><code>${escapeHtml(p.id)}</code></td>` +
        `<td><span class="badge ${statusClass(p.status)}">${escapeHtml(p.status)}</span></td>` +
        `<td>${escapeHtml(p.title)}</td>` +
        `<td><code>${escapeHtml(p.gate)}</code></td>` +
        `<td class="cells">${p.cells}</td></tr>`,
    )
    .join('')
  return (
    `<h3>${escapeHtml(fam.family)} <span class="famcount">(${fam.probes.length} probes · ${fam.cells} cells)</span></h3>` +
    `<div class="tablewrap"><table><thead><tr><th>Probe</th><th>Status</th><th>Hypothesis</th><th>Gate</th><th>Cells</th></tr></thead>` +
    `<tbody>${rows}</tbody></table></div>`
  )
}

/** Render the battery as a single self-contained, shareable static HTML page. */
export function renderBatteryHtml(battery: ExperimentBattery, opts: BatteryReportOptions): string {
  const s = battery.stats
  const chip = (n: number, label: string, cls = '') =>
    `<div class="chip ${cls}"><span class="n">${n.toLocaleString()}</span><span class="l">${escapeHtml(label)}</span></div>`
  const surviving = Object.entries(s.byStatus)
    .filter(([k]) => statusClass(k) === 'surviving')
    .reduce((n, [, v]) => n + v, 0)
  const chips = [
    chip(s.probes, 'probes'),
    chip(s.families, 'families'),
    chip(s.cellsRun, 'cells'),
    chip(s.byStatus.disproved ?? 0, 'disproved', 'null'),
    chip(s.byStatus.inconclusive ?? 0, 'inconclusive', 'open'),
    chip(surviving, 'surviving', 'surviving'),
  ].join('')
  const sections = (opts.sections ?? [])
    .map((sec) => `<section><h2>${escapeHtml(sec.heading)}</h2>${sec.html}</section>`)
    .join('')
  const families = battery.families.map(renderFamily).join('')
  const head =
    `<h1>${escapeHtml(opts.title)}</h1>` +
    (opts.subtitle ? `<p class="subtitle">${escapeHtml(opts.subtitle)}</p>` : '') +
    (opts.generatedNote ? `<p class="note">${escapeHtml(opts.generatedNote)}</p>` : '')
  return (
    '<!doctype html>\n' +
    `<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">` +
    `<title>${escapeHtml(opts.title)}</title><style>${STYLE}</style></head><body><main>` +
    head +
    `<div class="stats">${chips}</div>` +
    sections +
    `<section><h2>The battery</h2>${families}</section>` +
    (opts.footerHtml ? `<footer>${opts.footerHtml}</footer>` : '') +
    '</main></body></html>\n'
  )
}
