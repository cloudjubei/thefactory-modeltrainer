#!/usr/bin/env node
// The thefactory compute runner agent: pair once with a PIN, then long-poll
// the backend for compute jobs (training runs), execute them locally, and
// stream logs + results back. See runner/README.md.
//
//   node runner/agent.mjs pair --backend http://host:7001 [--name my-box] [--pin 123456]
//   node runner/agent.mjs run
//
// Config (backend URL + runner token) persists at ~/.thefactory-runner/config.json
// (override with --config <path>). The data cache lives next to it.
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { createInterface } from 'node:readline/promises'
import { ContentAddressedDataCache, spawnStreamingCommand } from 'thefactory-tools'
import { splitCommandLine, substituteCommandTemplate } from 'thefactory-tools/utils'

const POLL_WAIT_MS = 20000
const EVENT_FLUSH_MS = 1500
const LOG_TAIL_MAX = 200

function parseArgs(argv) {
  const [command, ...rest] = argv
  const flags = {}
  for (let i = 0; i < rest.length; i += 1) {
    if (rest[i].startsWith('--')) {
      flags[rest[i].slice(2)] = rest[i + 1] && !rest[i + 1].startsWith('--') ? rest[++i] : 'true'
    }
  }
  return { command, flags }
}

function configPath(flags) {
  return resolve(flags.config ?? join(homedir(), '.thefactory-runner', 'config.json'))
}

async function readConfig(flags) {
  try {
    return JSON.parse(await readFile(configPath(flags), 'utf8'))
  } catch {
    return undefined
  }
}

async function api(config, path, body) {
  const res = await fetch(`${config.backend}${path}`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      ...(config.token ? { authorization: `Bearer ${config.token}` } : {}),
    },
    body: JSON.stringify(body ?? {}),
  })
  if (!res.ok) {
    throw new Error(`${path} -> ${res.status} ${await res.text().catch(() => '')}`)
  }
  return res.json()
}

async function pair(flags) {
  const backend = (flags.backend ?? '').replace(/\/$/, '')
  if (!backend) {
    console.error('pair requires --backend http://host:port')
    process.exit(1)
  }
  let pin = flags.pin
  let name = flags.name
  if (!pin || !name) {
    const rl = createInterface({ input: process.stdin, output: process.stdout })
    name = name || (await rl.question('Runner name: '))
    pin = pin || (await rl.question('Pairing PIN (from Overseer → Settings → Compute Runners): '))
    rl.close()
  }
  const paired = await api({ backend }, '/api/v1/runners/pair', {
    pin: pin.trim(),
    name: name.trim(),
  })
  const file = configPath(flags)
  await mkdir(dirname(file), { recursive: true })
  await writeFile(
    file,
    JSON.stringify(
      { backend, runnerId: paired.runnerId, name: paired.name, token: paired.token },
      null,
      2,
    ),
  )
  console.log(`Paired as "${paired.name}" (${paired.runnerId}). Config saved to ${file}`)
  console.log('Start working with: node runner/agent.mjs run')
}

async function materialiseWorkspace(config, job, runnerHome) {
  if (job.repoRef?.kind === 'local') {
    if (!existsSync(job.repoRef.localPath)) {
      throw new Error(`local repo path not found on this runner: ${job.repoRef.localPath}`)
    }
    return job.repoRef.localPath
  }
  if (job.repoRef?.kind === 'git') {
    const slug = job.repoRef.gitUrl.replace(/[^a-zA-Z0-9]+/g, '-').slice(-60)
    const workspace = join(runnerHome, 'workspaces', `${slug}-${job.repoRef.commit}`)
    if (!existsSync(workspace)) {
      await mkdir(dirname(workspace), { recursive: true })
      await runOnce('git', ['clone', job.repoRef.gitUrl, workspace])
      await runOnce('git', ['-C', workspace, 'checkout', job.repoRef.commit])
    }
    return workspace
  }
  throw new Error(`unsupported repoRef: ${JSON.stringify(job.repoRef)}`)
}

function runOnce(cmd, args) {
  const { done } = spawnStreamingCommand({ argv: [cmd, ...args], cwd: process.cwd() })
  return done.then((exit) => {
    if (exit.exitCode !== 0) throw new Error(`${cmd} ${args.join(' ')} exited ${exit.exitCode}`)
  })
}

async function executeJob(config, job, dataCache, runnerHome, onAbortRegister) {
  const startedAt = Date.now()
  const logTail = []
  let pendingLogs = []
  const flush = async () => {
    if (pendingLogs.length === 0) return
    const logs = pendingLogs.splice(0)
    await api(config, `/api/v1/runners/channel/jobs/${job.jobId}/events`, { logs }).catch(() => {})
  }
  const flushTimer = setInterval(() => void flush(), EVENT_FLUSH_MS)

  try {
    const cwd = await materialiseWorkspace(config, job, runnerHome)
    if (job.dataFiles?.length) {
      await dataCache.ensureDataFiles({ files: job.dataFiles, targetDir: cwd })
    }
    const workDir = join(runnerHome, 'jobs', job.jobId)
    await mkdir(workDir, { recursive: true })
    const configFile = join(workDir, 'config.json')
    const summaryFile = join(workDir, 'summary.json')
    await writeFile(configFile, JSON.stringify(job.config ?? {}))
    const argv = splitCommandLine(
      substituteCommandTemplate(job.commandTemplate, {
        configPath: configFile,
        summaryOut: summaryFile,
      }),
    )
    const child = spawnStreamingCommand({
      argv,
      cwd,
      timeoutMs: job.timeoutMs,
      onLine: (line) => {
        pendingLogs.push(line)
        logTail.push(line)
        if (logTail.length > LOG_TAIL_MAX) logTail.shift()
      },
    })
    onAbortRegister(() => child.kill())
    const exit = await child.done
    clearInterval(flushTimer)
    await flush()

    let summary
    try {
      summary = JSON.parse(await readFile(summaryFile, 'utf8'))
    } catch {
      summary = undefined
    }
    await rm(workDir, { recursive: true, force: true })
    const durationMs = Date.now() - startedAt
    if (exit.aborted) {
      return { status: 'aborted', exitCode: exit.exitCode, durationMs, logTail }
    }
    if (exit.exitCode === 0 && summary === undefined) {
      return {
        status: 'failed',
        exitCode: exit.exitCode,
        error: 'missing summary',
        durationMs,
        logTail,
      }
    }
    return {
      status: exit.exitCode === 0 ? 'completed' : 'failed',
      exitCode: exit.exitCode,
      summary,
      ...(exit.exitCode !== 0
        ? { error: exit.timedOut ? 'timeout' : `exit ${exit.exitCode}` }
        : {}),
      durationMs,
      logTail,
    }
  } catch (err) {
    clearInterval(flushTimer)
    await flush()
    return {
      status: 'failed',
      exitCode: null,
      error: err instanceof Error ? err.message : String(err),
      durationMs: Date.now() - startedAt,
      logTail,
    }
  }
}

async function run(flags) {
  const config = await readConfig(flags)
  if (!config?.token) {
    console.error('Not paired. Run: node runner/agent.mjs pair --backend http://host:7001')
    process.exit(1)
  }
  const runnerHome = dirname(configPath(flags))
  const dataCache = new ContentAddressedDataCache({ cacheRoot: join(runnerHome, 'cache') })
  console.log(`Runner "${config.name}" polling ${config.backend} ...`)
  let currentAbort
  let currentJobId
  for (;;) {
    let poll
    try {
      poll = await api(config, '/api/v1/runners/channel/poll', { waitMs: POLL_WAIT_MS })
    } catch (err) {
      console.error(`poll failed (${err.message}); retrying in 5s`)
      await new Promise((r) => setTimeout(r, 5000))
      continue
    }
    if (poll.abort?.length && currentJobId && poll.abort.includes(currentJobId)) {
      console.log(`abort requested for ${currentJobId}`)
      currentAbort?.()
    }
    if (!poll.job) continue
    const job = poll.job
    currentJobId = job.jobId
    console.log(`job ${job.jobId}: ${job.commandTemplate}`)
    const result = await executeJob(config, job, dataCache, runnerHome, (abort) => {
      currentAbort = abort
    })
    currentAbort = undefined
    currentJobId = undefined
    console.log(`job ${job.jobId}: ${result.status} (${Math.round(result.durationMs / 1000)}s)`)
    await api(config, `/api/v1/runners/channel/jobs/${job.jobId}/result`, result).catch((err) =>
      console.error(`result post failed: ${err.message}`),
    )
  }
}

const { command, flags } = parseArgs(process.argv.slice(2))
if (command === 'pair') await pair(flags)
else if (command === 'run') await run(flags)
else {
  console.log(
    'Usage: node runner/agent.mjs <pair|run> [--backend URL] [--name NAME] [--pin PIN] [--config PATH]',
  )
  process.exit(command ? 1 : 0)
}
