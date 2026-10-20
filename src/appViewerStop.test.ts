import { readFileSync } from 'fs'
import { dirname, join } from 'path'
import { fileURLToPath } from 'url'
import { describe, expect, it } from 'vitest'

const here = dirname(fileURLToPath(import.meta.url))
const appJs = readFileSync(join(here, '..', 'viewer', 'app.js'), 'utf8')

function body(signature: string): string {
  const start = appJs.indexOf(signature)
  expect(start, `${signature} not found in viewer/app.js`).toBeGreaterThan(-1)
  const next = appJs.indexOf('\nasync function ', start + signature.length)
  const nextSync = appJs.indexOf('\nfunction ', start + signature.length)
  const ends = [next, nextSync].filter((i) => i > -1)
  return appJs.slice(start, ends.length ? Math.min(...ends) : undefined)
}

describe('viewer Stop / Discard go through activities.stop, confirmed first', () => {
  it('stops one activity for good through the host’s stop, never the resumable abort', () => {
    const stop = body('async function stopActivityById(activityId, opts)')
    expect(stop).toContain('window.OverseerBridge.stopActivity(activityId)')
    expect(stop).not.toContain('abortActivity')
    expect(stop).toContain('stopOutcomeNote(')
    expect(appJs).not.toContain('abortActivityById')
  })

  it('keeps Pause on the resumable abort', () => {
    expect(body('async function pauseActivityById(activityId)')).toContain(
      'window.OverseerBridge.abortActivity(activityId)',
    )
  })

  it('asks in the app before an Abort, a Discard or a queue removal stops anything', () => {
    const block = body('function activityBlockHtml(entry)')
    expect(block).toContain('stopPromptId === entry.activityId')
    expect(block).toContain('stopConfirmHtml(id,')
    expect(body('function stopConfirmHtml(id, question, verb)')).toContain('data-stop-confirm=')
    expect(body('function queueSectionHtml(entries, title)')).toContain('data-stop-confirm=')
    expect(appJs).toMatch(
      /closest\('button\[data-abort\]'\)\s*\n\s*if \(abortBtn\) \{\s*\n\s*stopPromptId = /,
    )
    expect(body('async function onQueueRemove(activityId)')).not.toContain('stopActivityById')
    expect(appJs).toMatch(/data-stop-confirm\][\s\S]{0,200}stopActivityById\(/)
  })

  it('asks before Find the best model’s Stop, then stops each of its activities through the host’s stop', () => {
    const panel = body('async function renderAutopilotPanel()')
    expect(panel).toContain('autopilotStopConfirm')
    expect(panel).toContain('autopilot-stop-confirm')
    expect(panel).toContain('stopActivityById(id')
    expect(panel).not.toContain('abortActivityById')
  })

  it('asks before the exploration’s Stop, then stops the controller and its child through the host’s stop', () => {
    const start = appJs.indexOf('    onAbort: async () => {')
    const end = appJs.indexOf('    onPause: async () => {', start)
    const stop = appJs.slice(start, end)
    expect(stop).toContain('explorationStopConfirm = true')
    expect(stop).toContain('onAbortConfirm: async () => {')
    expect(stop).toContain('window.OverseerBridge.stopActivity(')
    expect(stop).not.toContain('abortActivity')
    expect(appJs).toContain('confirmingStop: explorationStopConfirm')
  })
})
