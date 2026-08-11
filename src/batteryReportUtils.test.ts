import { describe, it, expect } from 'vitest'

import { buildBattery, renderBatteryHtml } from './batteryReportUtils.js'

describe('buildBattery', () => {
  it('groups by family, counts swept cells, and tallies status', () => {
    const hyps = [
      { id: 'p1', type: 't-a', status: 'disproved', gate: { kind: 'deflated-sharpe', metric: 'oos_sharpe' }, spec: { sweep: { x: [1, 2, 3], w: ['a', 'b'] } } }, // 6
      { id: 'p2', type: 't-a', status: 'inconclusive', spec: { sweep: { x: [1, 2] } } }, // 2
      { id: 'p3', type: 't-b', status: 'disproved', spec: {} }, // empty sweep -> 1
    ]
    const b = buildBattery(hyps, { familyOf: (t) => (t === 't-a' ? 'Family A' : 'Family B'), familyOrder: ['Family A', 'Family B'] })
    expect(b.stats.probes).toBe(3)
    expect(b.stats.families).toBe(2)
    expect(b.stats.cellsRun).toBe(9) // 6 + 2 + 1
    expect(b.stats.byStatus).toEqual({ disproved: 2, inconclusive: 1 })
    expect(b.families.map((f) => f.family)).toEqual(['Family A', 'Family B'])
    expect(b.families[0].cells).toBe(8)
    expect(b.families[0].probes[0].gate).toBe('deflated-sharpe(oos_sharpe)')
    expect(b.families[1].probes[0].cells).toBe(1)
  })

  it('defaults the family label to the raw type and the title to the id', () => {
    const b = buildBattery([{ id: 'probe-x', type: 'blackswan-cot-hypothesis', status: 'inconclusive' }], {})
    expect(b.families[0].family).toBe('blackswan-cot-hypothesis')
    expect(b.families[0].probes[0].title).toBe('probe-x')
    expect(b.families[0].probes[0].status).toBe('inconclusive')
  })
})

describe('renderBatteryHtml', () => {
  const battery = buildBattery(
    [{ id: 'p<x>', type: 't', title: 'A <b>bold</b> claim', status: 'disproved', gate: { kind: 'deflated-sharpe', metric: 'oos_sharpe' }, spec: { sweep: { x: [1, 2] } } }],
    {},
  )
  const html = renderBatteryHtml(battery, {
    title: 'My <report>',
    subtitle: 'a subtitle',
    sections: [{ heading: 'Thesis', html: '<p>trusted narrative</p>' }],
  })

  it('is a self-contained static HTML document (inline CSS, no scripts, no external assets)', () => {
    expect(html.startsWith('<!doctype html>')).toBe(true)
    expect(html.trimEnd().endsWith('</html>')).toBe(true)
    expect(html).toContain('<style>')
    expect(html).not.toContain('<script') // pure static, no JS
    expect(html).not.toContain('<link') // no external stylesheet
    expect(html).not.toMatch(/(?:src|href)="https?:/) // no external asset loads
  })

  it('escapes dynamic probe/report text but passes author-controlled section HTML through', () => {
    expect(html).toContain('My &lt;report&gt;') // report title escaped
    expect(html).toContain('A &lt;b&gt;bold&lt;/b&gt; claim') // probe title escaped
    expect(html).toContain('p&lt;x&gt;') // probe id escaped
    expect(html).toContain('<p>trusted narrative</p>') // section html trusted (not escaped)
  })

  it('renders the stats, the family table and the status', () => {
    expect(html).toContain('My &lt;report&gt;')
    expect(html).toContain('disproved')
    expect(html).toContain('deflated-sharpe(oos_sharpe)')
    expect(html).toContain('a subtitle')
  })
})
