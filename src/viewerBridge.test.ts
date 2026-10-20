import { readFileSync } from 'fs'
import { dirname, join } from 'path'
import { fileURLToPath } from 'url'
import * as vm from 'vm'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const here = dirname(fileURLToPath(import.meta.url))
const SOURCE = readFileSync(join(here, '..', 'viewer', 'bridge.js'), 'utf8')

type Posted = { type: string; id: string; payload: unknown }

function loadBridge() {
  const posted: Posted[] = []
  const listeners: Array<(e: { data: unknown }) => void> = []
  const win: Record<string, unknown> = {
    addEventListener: (_type: string, fn: (e: { data: unknown }) => void) => listeners.push(fn),
  }
  win.parent = { postMessage: (m: Posted) => posted.push(m) }
  vm.runInNewContext(SOURCE, { window: win, setTimeout, clearTimeout })
  const bridge = win.OverseerBridge as Record<string, (...a: unknown[]) => Promise<unknown>>
  const answer = (id: string, result: unknown) =>
    listeners.forEach((fn) => fn({ data: { overseerBridgeResponse: true, id, ok: true, result } }))
  const last = (type: string) => posted.filter((m) => m.type === type).at(-1)!
  return { bridge, answer, last }
}

beforeEach(() => vi.useFakeTimers())
afterEach(() => vi.useRealTimers())

describe('OverseerBridge.stopActivity', () => {
  it('sends the person’s stop as activities.stop and answers what the host observed', async () => {
    const { bridge, answer, last } = loadBridge()
    const stopping = bridge.stopActivity('act-1')
    const sent = last('overseer:activities.stop')
    expect(sent.payload).toEqual({ activityId: 'act-1' })
    answer(sent.id, { activityId: 'act-1', outcome: 'stopped' })
    await expect(stopping).resolves.toEqual({ activityId: 'act-1', outcome: 'stopped' })
  })

  it('carries a reason when one is given', () => {
    const { bridge, last } = loadBridge()
    void bridge.stopActivity('act-1', 'Wrong market').catch(() => {})
    expect(last('overseer:activities.stop').payload).toEqual({
      activityId: 'act-1',
      reason: 'Wrong market',
    })
  })

  it('waits past the default guard while the host awaits the teardown', async () => {
    const { bridge, answer, last } = loadBridge()
    const stopping = bridge.stopActivity('act-1')
    let settled = false
    void stopping.then(
      () => (settled = true),
      () => (settled = true),
    )
    await vi.advanceTimersByTimeAsync(5 * 60_000)
    expect(settled).toBe(false)
    answer(last('overseer:activities.stop').id, { activityId: 'act-1', outcome: 'stopped' })
    await expect(stopping).resolves.toMatchObject({ outcome: 'stopped' })
  })

  it('gives up waiting only after the stop guard', async () => {
    const { bridge } = loadBridge()
    const failed = expect(bridge.stopActivity('act-1')).rejects.toThrow(/timeout: activities\.stop/)
    await vi.advanceTimersByTimeAsync(11 * 60_000)
    await failed
  })

  it('keeps Pause on activities.abort under the default guard', async () => {
    const { bridge, last } = loadBridge()
    const pausing = bridge.abortActivity('act-1')
    expect(last('overseer:activities.abort').payload).toEqual({ activityId: 'act-1' })
    const failed = expect(pausing).rejects.toThrow(/timeout: activities\.abort/)
    await vi.advanceTimersByTimeAsync(8_000)
    await failed
  })
})
