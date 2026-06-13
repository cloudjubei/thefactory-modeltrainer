// One-off: import BlackSwan's pre-hub experiment log (results_hour.ods) into the hub ledger as
// `blackswan-run` records, so historical experiments count as "tried" and are not re-run blindly.
//
// It computes the SAME trade-aware objective the live trainer now emits — traded_return =
// total_return_pct * min(1, n_trades / MIN_TRADES_FOR_FULL_CREDIT) — so historical and live runs sit
// on one axis (a 1-trade ~ buy-and-hold row gates to ~0). The config parsed from the spreadsheet's
// `Name` field is best-effort (reward names contain underscores; custom net-arch is descriptive
// text), so the raw Name/Data/Env strings are preserved on the record and each row is keyed by a
// stable hash of those strings (idempotent re-import). Runs are tagged ranBy='ledger-import' +
// thesis='historical (pre-hub)' so they form a distinct, identifiable cohort in the by-experiment view.
//
// Dry-run by default — parses, builds records, writes them to /tmp/blackswan-ledger-records.json and
// prints a summary. NOTHING is written to the hub. To write into a FILE-backed hub store:
//   node scripts/import-blackswan-ledger.mjs --write --scope <projectId> --data-dir <overseerRepoPath>/.factory/data
// (If the hub uses thefactory-db rather than file storage, import through the backend instead.)
//
// Requires python3 on PATH (used only to read the .ods spreadsheet).

import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { writeFileSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { setupKeyOf } from '../dist/modelTrainerHelpers.js'

// Mirror of trainer/summary.py — keep in sync if the live objective gate changes.
const MIN_TRADES_FOR_FULL_CREDIT = 20
const DEGENERATE_TRADE_COUNT = 2
const RECORD_TYPE = 'blackswan-run'

function arg(name, fallback) {
  const i = process.argv.indexOf(name)
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback
}
const ODS = arg('--ods', '/Users/cloud/Documents/Work/BlackSwan/results_hour.ods')
const WRITE = process.argv.includes('--write')
const SCOPE = arg('--scope', null)
const DATA_DIR = arg('--data-dir', null)

// Read the .ods (a zip of XML) via a tiny python helper — robust across cell repeats/spans.
function readOdsRows(odsPath) {
  const dir = mkdtempSync(join(tmpdir(), 'odsparse-'))
  const py = join(dir, 'parse.py')
  writeFileSync(
    py,
    `import sys, json, zipfile, xml.etree.ElementTree as ET
NS={'t':'urn:oasis:names:tc:opendocument:xmlns:table:1.0','x':'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}
root=ET.fromstring(zipfile.ZipFile(sys.argv[1]).read('content.xml'))
def text(c): return ''.join(''.join(p.itertext()) for p in c.findall('x:p',NS))
rows=[]
for tbl in root.iter('{%s}table'%NS['t']):
    for row in tbl.iter('{%s}table-row'%NS['t']):
        cells=[]
        for c in row.findall('t:table-cell',NS):
            rep=int(c.get('{%s}number-columns-repeated'%NS['t'],'1'))
            cells.extend([text(c)]*min(rep,40))
        rows.append(cells)
    break
print(json.dumps(rows))
`,
  )
  const out = execFileSync('python3', [py, odsPath], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 })
  return JSON.parse(out)
}

const numRe = /^\d*\.?\d+$/
function num(v) {
  const n = Number(String(v).replace(/[%,]/g, '').trim())
  return Number.isFinite(n) ? n : 0
}

// Best-effort config from the `Name` field. Returns null for non-run rows (notes/blank).
function parseName(name) {
  const n = (name || '').trim()
  if (!n) return null
  if (n.toLowerCase() === 'hodl') return { model_name: 'hodl', historical_name: n }
  if (!n.startsWith('rl_')) return null
  const toks = n.split('_')
  const cfg = { model_name: toks[1], historical_name: n }
  let end = toks.length
  let ranTs = null
  if (/^\d+\.\d+$/.test(toks[end - 1])) {
    ranTs = parseFloat(toks[end - 1])
    end -= 1
  }
  if (end > 2 && /^\d+$/.test(toks[end - 1])) {
    cfg.seed = parseInt(toks[end - 1], 10)
    end -= 1
  }
  let fn = 2
  while (fn < end && !numRe.test(toks[fn])) fn++
  if (fn > 2) cfg.reward_model = toks.slice(2, fn).join('_')
  const nums = []
  let i = fn
  while (i < end && numRe.test(toks[i]) && nums.length < 5) nums.push(toks[i++])
  const keys = ['learning_rate', 'learning_starts', 'batch_size', 'buffer_size', 'gamma']
  nums.forEach((v, k) => (cfg[keys[k]] = numRe.test(v) ? Number(v) : v))
  if (toks[i] && /^[A-Za-z]/.test(toks[i])) cfg.optimizer_class = toks[i++]
  if (toks[i] && /^[A-Za-z]/.test(toks[i])) cfg.activation_fn = toks[i++]
  if (toks[i] && /\d/.test(toks[i]) && toks[i].includes('|')) cfg.net_arch = toks[i++]
  return { cfg, ranTs }
}

const rows = readOdsRows(ODS)
const header = rows[0].map((h) => h.trim())
const col = (name) => header.indexOf(name)
const idx = {
  data: col('Data'),
  env: col('Env'),
  name: col('Name'),
  pct: col('%'),
  trades: col('#trades'),
  win: col('Win%'),
  profit: col('$'),
  tradeUsd: col('Trade$'),
  fees: col('Fees$'),
  volume: col('Volume$'),
  sls: col('SLs'),
  wins: col('Wins'),
  losses: col('Losses'),
}

const records = []
const skipped = []
for (let r = 1; r < rows.length; r++) {
  const row = rows[r]
  const name = (row[idx.name] || '').trim()
  const parsed = parseName(name)
  if (!parsed) {
    if (name) skipped.push(name)
    continue
  }
  const cfgParsed = parsed.cfg || parsed
  const ranTs = parsed.ranTs ?? null
  const data = (row[idx.data] || '').trim()
  const env = (row[idx.env] || '').trim()
  const isHodl = cfgParsed.model_name === 'hodl'
  const timeframe = data.includes('_1h') ? '1h' : data.includes('_1d') ? '1d' : '1h'

  const totalReturnPct = num(row[idx.pct]) * 100
  const nTrades = num(row[idx.trades])
  const tradeGate = MIN_TRADES_FOR_FULL_CREDIT > 0 ? Math.min(1, nTrades / MIN_TRADES_FOR_FULL_CREDIT) : 1
  const tradedReturn = totalReturnPct * tradeGate

  const config = {
    asset: 'BTCUSDT',
    timeframe,
    ...cfgParsed,
    historical_data: data,
    historical_env: env,
  }

  const flags = []
  if (!isHodl && nTrades === 0) flags.push('zero_trades')
  else if (!isHodl && nTrades <= DEGENERATE_TRADE_COUNT) flags.push('few_trades')

  const content = {
    objective: tradedReturn,
    metrics: {
      traded_return: tradedReturn,
      total_return_pct: totalReturnPct,
      win_pct: num(row[idx.win]),
      n_trades: nTrades,
      trade_gate: tradeGate,
      profit_usd: num(row[idx.profit]),
      trade_usd: num(row[idx.tradeUsd]),
      fees_usd: num(row[idx.fees]),
      volume_usd: num(row[idx.volume]),
      wins: num(row[idx.wins]),
      losses: num(row[idx.losses]),
      stop_losses: num(row[idx.sls]),
    },
    health: { status: flags.length ? 'degenerate' : 'ok', flags },
    config,
    setupKey: setupKeyOf(config),
    status: 'completed',
    ranAt: ranTs ? new Date(ranTs * 1000).toISOString() : '2024-07-01T00:00:00.000Z',
    ranBy: 'ledger-import',
    thesis: 'historical (pre-hub)',
    dataset: { asset: 'BTCUSDT', timeframe },
    durationMs: 0,
  }
  const key = createHash('sha256').update(`${name}|${data}|${env}`).digest('hex').slice(0, 12)
  records.push({ key, content })
}

// Summary
const objs = records.map((r) => r.content.objective).filter(Number.isFinite).sort((a, b) => a - b)
const med = objs.length ? objs[Math.floor(objs.length / 2)] : NaN
const degenerate = records.filter((r) => r.content.health.flags.length).length
const setupKeys = new Set(records.map((r) => r.content.setupKey))
console.log(`Parsed ${rows.length - 1} data rows -> ${records.length} run records (${skipped.length} non-run rows skipped)`)
console.log(`  distinct setups: ${setupKeys.size}   degenerate/low-trade: ${degenerate}`)
console.log(`  traded_return: min ${objs[0]?.toFixed(1)}  median ${med?.toFixed(1)}  max ${objs[objs.length - 1]?.toFixed(1)}`)
if (skipped.length) console.log(`  skipped sample: ${skipped.slice(0, 4).map((s) => JSON.stringify(s.slice(0, 40))).join(', ')}`)
console.log('  sample records:')
for (const rec of records.slice(0, 3)) {
  const m = rec.content.metrics
  console.log(
    `    ${rec.key}  ${rec.content.config.model_name}/${rec.content.config.reward_model ?? '-'}  ` +
      `traded_return=${m.traded_return.toFixed(1)} return=${m.total_return_pct.toFixed(1)}% trades=${m.n_trades} win=${m.win_pct}% gate=${m.trade_gate.toFixed(2)}`,
  )
}

const outFile = join(tmpdir(), 'blackswan-ledger-records.json')
writeFileSync(outFile, JSON.stringify(records, null, 2))
console.log(`\nRecords written for inspection: ${outFile}`)

if (!WRITE) {
  console.log('\nDRY-RUN (no hub write). To import into a file-backed hub store:')
  console.log('  node scripts/import-blackswan-ledger.mjs --write --scope <projectId> --data-dir <overseerRepoPath>/.factory/data')
  process.exit(0)
}

if (!SCOPE || !DATA_DIR) {
  console.error('\n--write requires --scope <projectId> and --data-dir <path>. Aborting (nothing written).')
  process.exit(1)
}
const { FileDataStorage } = await import('thefactory-tools')
const storage = new FileDataStorage(DATA_DIR)
let n = 0
for (const rec of records) {
  await storage.upsertRecord({ scope: SCOPE, type: RECORD_TYPE, key: rec.key, content: rec.content })
  n++
}
console.log(`\nWrote ${n} ${RECORD_TYPE} records to scope='${SCOPE}' under ${DATA_DIR}`)
