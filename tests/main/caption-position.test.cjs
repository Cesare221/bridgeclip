const test = require('node:test')
const assert = require('node:assert/strict')
const { loadMain } = require('../zernio/support/load-main.cjs')
const fixture = require('../fixtures/editor/project.json')
const schema = loadMain("export * from './src/shared/clip-editor'; export * from './src/renderer/lib/caption-preview'")

test('caption position is validated, saved in render edits and reset without changing review evidence', () => {
  const old = schema.parseEditorProject(structuredClone(fixture)).candidates[0]
  assert.equal(old.caption_y, null)
  for (const y of [NaN, Infinity, -.1, .09, .91, true, '0.5', {}]) {
    assert.throws(() => schema.parseCandidateEdit({ ...old, caption_y: y }, fixture.duration_ms))
  }
  const ready = { ...old, status: 'ready' }
  const placed = schema.refineEdit(ready, { caption_y: .25 })
  assert.equal(placed.status, 'refining')
  assert.equal(schema.candidateEdit(placed).caption_y, .25)
  assert.equal(schema.editSignature(placed), schema.editSignature(ready))
  assert.notEqual(schema.renderEditKey(placed), schema.renderEditKey(ready))
  const reset = schema.parseCandidateEdit(schema.candidateEdit({ ...placed, caption_y: null }), fixture.duration_ms)
  assert.equal(reset.caption_y, null)
})

test('automatic caption anchors follow portrait layouts and landscape; manual placement overrides them', () => {
  const project = schema.parseEditorProject(structuredClone(fixture)), c = project.candidates[0]
  c.scenes = [{ at_ms: 0, layout: 'fill', crops: [[0, 0, 1, 1]] }]
  assert.deepEqual(schema.captionAnchor(project, c, 2000), { y: 1340 / 1920, bottom: false })
  c.scenes[0].layout = 'split'
  assert.deepEqual(schema.captionAnchor(project, c, 2000), { y: .5, bottom: false })
  c.scenes[0].layout = 'fit'
  assert.equal(schema.captionAnchor(project, c, 2000).bottom, true)
  project.aspect_ratio = '16:9'
  assert.deepEqual(schema.captionAnchor(project, c, 2000), { y: 1 - 100 / 1080, bottom: true })
  c.caption_y = .2
  assert.deepEqual(schema.captionAnchor(project, c, 2000), { y: .2, bottom: false })
})

test('timed caption preview follows words, cut-compressed time, linger and caption suppression', () => {
  const raw = structuredClone(fixture)
  raw.transcript = [
    { start_ms: 1000, end_ms: 5000, text: 'Stay here, please.',
      words: [{ text: 'Stay', start_ms: 4800, end_ms: 4900 }] },
    { start_ms: 7000, end_ms: 9000, text: 'Together now.',
      words: [{ text: 'together', start_ms: 7000, end_ms: 7100 }, { text: 'now.', start_ms: 7200, end_ms: 7300 }] }
  ]
  const project = schema.parseEditorProject(raw), candidate = project.candidates[0]
  const timeline = schema.captionPreviewTimeline(project, candidate, 3)
  const atCut = schema.captionPreviewAt(timeline, 7000)
  assert.deepEqual(atCut.words, [
    { text: 'Stay', state: 'past' }, { text: 'together', state: 'active' }, { text: 'now.', state: 'future' }
  ])
  assert.equal(schema.captionPreviewAt(timeline, 5500), null, 'gaps removed by the cut never show a caption')
  assert.ok(schema.captionPreviewAt(timeline, 7999), 'the last word lingers for 700ms on the compact cut timeline')
  assert.equal(schema.captionPreviewAt(timeline, 8000), null, 'captions clear after the export linger')
  candidate.caption_suppression_ranges = [[7000, 7300]]
  assert.equal(schema.captionPreviewAt(schema.captionPreviewTimeline(project, candidate), 7000), null)
  candidate.caption_suppression_ranges = []
  candidate.captions = false
  assert.equal(schema.captionPreviewAt(schema.captionPreviewTimeline(project, candidate), 7000), null)
})

test('timed caption preview groups punctuation and applies caption edits to existing timings', () => {
  const raw = structuredClone(fixture)
  raw.transcript = [{ start_ms: 1000, end_ms: 2000, text: 'One two three.',
    words: [
      { text: 'One', start_ms: 1000, end_ms: 1100 },
      { text: 'two,', start_ms: 1200, end_ms: 1300 },
      { text: 'three.', start_ms: 1400, end_ms: 1500 }
    ] }]
  const project = schema.parseEditorProject(raw), candidate = project.candidates[0]
  const timeline = schema.captionPreviewTimeline(project, candidate, 3)
  assert.deepEqual(schema.captionPreviewAt(timeline, 1250).words, [
    { text: 'One', state: 'past' }, { text: 'two,', state: 'active' }
  ], 'a comma ends the current group after two words')
  assert.deepEqual(schema.captionPreviewAt(timeline, 1450).words, [
    { text: 'three.', state: 'active' }
  ], 'sentence punctuation starts the next group')

  candidate.caption_edits = [{ segment: 0, text: 'Alpha Beta' }]
  const corrected = schema.captionPreviewTimeline(project, candidate, 3)
  assert.deepEqual(schema.captionPreviewAt(corrected, 1050).words, [
    { text: 'Alpha', state: 'active' }, { text: 'Beta', state: 'future' }
  ])
  assert.deepEqual(schema.captionPreviewAt(corrected, 1250).words, [
    { text: 'Alpha', state: 'past' }, { text: 'Beta', state: 'active' }
  ], 'same-count corrections keep word timing')

  candidate.caption_edits = [{ segment: 0, text: 'One longer corrected line' }]
  assert.deepEqual(schema.captionPreviewAt(schema.captionPreviewTimeline(project, candidate, 4), 1450).words, [
    { text: 'One', state: 'past' }, { text: 'longer', state: 'past' },
    { text: 'corrected', state: 'past' }, { text: 'line', state: 'active' }
  ], 'changed word counts distribute over the original spoken interval')
  candidate.caption_edits = [{ segment: 0, text: '   ' }]
  assert.equal(schema.captionPreviewAt(schema.captionPreviewTimeline(project, candidate), 1050), null)
})

test('legacy projects preview corrected segment text without word highlighting', () => {
  const raw = structuredClone(fixture)
  raw.transcript = [{ start_ms: 1000, end_ms: 3000, text: 'Old transcript line.' }]
  const oldProject = schema.parseEditorProject(raw), oldCandidate = oldProject.candidates[0]
  assert.equal(Object.hasOwn(oldProject.transcript[0], 'words'), false)
  assert.deepEqual(schema.captionPreviewAt(schema.captionPreviewTimeline(oldProject, oldCandidate), 1500), {
    words: [{ text: 'Old transcript line.', state: 'active' }], legacy: true
  })
  oldCandidate.caption_edits = [{ segment: 0, text: 'Corrected legacy text' }]
  assert.equal(schema.captionPreviewAt(schema.captionPreviewTimeline(oldProject, oldCandidate), 1500).words[0].text,
    'Corrected legacy text')
  assert.equal(schema.captionPreviewAt(schema.captionPreviewTimeline(oldProject, oldCandidate), 3500), null)
})

test('editor project accepts optional word timings and rejects invalid timed transcript data', () => {
  const raw = structuredClone(fixture)
  raw.transcript = [{ start_ms: 1000, end_ms: 3000, text: 'Timed words.',
    words: [{ text: 'Timed', start_ms: 1100, end_ms: 1600 }, { text: 'words.', start_ms: 1700, end_ms: 2500 }] }]
  assert.deepEqual(schema.parseEditorProject(raw).transcript[0].words, raw.transcript[0].words)
  for (const words of [
    [{ text: 'bad', start_ms: 1000, end_ms: 0 }],
    [{ text: 'bad', start_ms: 1000, end_ms: 12001 }],
    [{ text: 'late', start_ms: 2000, end_ms: 2200 }, { text: 'early', start_ms: 1500, end_ms: 1800 }],
    [{ text: 10, start_ms: 1000, end_ms: 1200 }]
  ]) {
    const invalid = structuredClone(raw)
    invalid.transcript[0].words = words
    assert.throws(() => schema.parseEditorProject(invalid))
  }
})

test('timed caption preview renderer uses actual text, active word styling and preset positioning styles', () => {
  const React = require('react')
  const { renderToStaticMarkup } = require('react-dom/server')
  const { CaptionPreviewWords } = loadMain(
    "export { CaptionPreviewWords } from './src/renderer/components/EditorCaptionPreview'",
    { react: React, 'react/jsx-runtime': require('react/jsx-runtime') })
  const render = (props) => {
    const previousReact = global.React
    global.React = React
    try { return renderToStaticMarkup(React.createElement(CaptionPreviewWords, props)) }
    finally {
      if (previousReact === undefined) delete global.React
      else global.React = previousReact
    }
  }
  const html = render({
      line: { legacy: false, words: [
        { text: 'AS', state: 'past' }, { text: 'MUDANÇAS', state: 'active' }, { text: 'CLIMÁTICAS', state: 'future' }
      ] }, presetId: 'pop', shadow: '0 1px 3px #000', size: 24
    })
  assert.match(html, /AS/)
  assert.match(html, /MUDANÇAS/)
  assert.match(html, /data-caption-word="active"[^>]*color:#FFE234/)
  assert.match(html, /data-caption-word="future"/)
  assert.match(html, /editor-caption-words/)
  const impactHtml = render({
    line: { legacy: false, words: [{ text: 'First', state: 'active' }, { text: 'hidden', state: 'future' }] },
    presetId: 'impact', shadow: 'none', size: 40
  })
  assert.match(impactHtml, /data-caption-word="future"[^>]*visibility:hidden/)
})
