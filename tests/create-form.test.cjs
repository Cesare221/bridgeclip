const assert = require('node:assert/strict')
const { test } = require('node:test')
const path = require('node:path')
const { buildSync } = require('esbuild')
const React = require('react')
const { renderToStaticMarkup } = require('react-dom/server')

const bundled = buildSync({
  stdin: {
    contents: `export { FormatStep, ClipsStep, CaptionsStep, JobForm, buildJobRequest, parseTrimRange } from './src/renderer/components/JobForm';
      export { JobProgress } from './src/renderer/components/JobProgress';
      export { SetupCard } from './src/renderer/components/SetupCard';
      export { useSettingsStore, requiredToolsAvailable } from './src/renderer/store/use-settings-store';
      export { useDraftStore } from './src/renderer/store/use-draft-store';
      export { SourcePicker, isValidSourceLink } from './src/renderer/components/SourcePicker';
      export { framingProblem, sourceAnalysisNotice } from './src/renderer/components/ClipList';
      export { parseJobOutput } from './src/shared/job-output';
      export { parseEditorProject } from './src/shared/clip-editor';
      export { twitchVodId, normalizeVideoSource } from './src/shared/video-source';`,
    resolveDir: path.resolve(__dirname, '..'),
    loader: 'ts'
  },
  bundle: true,
  platform: 'node',
  format: 'cjs',
  packages: 'external',
  loader: { '.css': 'empty' },
  define: { __APP_VERSION__: JSON.stringify(require('../package.json').version) },
  jsx: 'automatic',
  write: false
}).outputFiles[0].text

const form = { exports: {} }
new Function('module', 'exports', 'require', bundled)(form, form.exports, require)
const { FormatStep, JobForm, SourcePicker, isValidSourceLink, parseTrimRange, framingProblem, sourceAnalysisNotice, parseJobOutput } = form.exports

test('setup needs only OpenRouter for clipping', () => {
  const { SetupCard, useSettingsStore } = form.exports
  useSettingsStore.setState({ openrouterConfigured: false })
  const setup = renderToStaticMarkup(React.createElement(SetupCard, { onOpenSettings() {} }))
  assert.match(setup, /OpenRouter/)
  assert.doesNotMatch(setup, /ElevenLabs/)
  const localSetup = renderToStaticMarkup(React.createElement(SetupCard, { onOpenSettings() {}, localMode: true }))
  assert.doesNotMatch(localSetup, /Connect OpenRouter/)
  useSettingsStore.setState({ openrouterConfigured: false, toolStatus: null })
})

test('local setup does not require yt-dlp, which is only used for remote downloads', () => {
  const { requiredToolsAvailable } = form.exports
  const tools = {
    python: true, pythonDeps: true, pythonPath: 'python', pythonError: null,
    ffmpeg: true, ffmpegCaptions: true, ffprobe: true, ytdlp: false,
    engine: true, enginePath: 'engine', bridgeRunner: true, bridgePath: 'bridge'
  }
  assert.equal(requiredToolsAvailable(tools, true), true)
  assert.equal(requiredToolsAvailable(tools, false), false)
})

test('source picker accepts full HTTP(S) links and rejects malformed or credentialed links', () => {
  assert.equal(isValidSourceLink(' https://www.youtube.com/watch?v=abc '), true)
  assert.equal(isValidSourceLink('http://example.com/video.mp4'), true)
  assert.equal(isValidSourceLink('https://'), false)
  assert.equal(isValidSourceLink('https://user:pass@example.com/video'), false)
  assert.equal(isValidSourceLink('file:///tmp/video.mp4'), false)
})

test('trim validation matches main process bounds, including an end at zero', () => {
  assert.equal(parseTrimRange(true, '', '0').error, 'End must be after the start.')
  assert.equal(parseTrimRange(true, '1:30', '90').error, 'End must be after the start.')
  assert.deepEqual(parseTrimRange(true, '', '1:30'), { start: null, end: 90, error: null })
  assert.deepEqual(parseTrimRange(false, 'oops', '0'), { start: null, end: null, error: null })
})

test('the wizard opens on the video step with the steps listed in order', () => {
  const html = renderToStaticMarkup(React.createElement(JobForm, { onSubmit() {} }))
  const labels = [...html.matchAll(/aria-label="Create steps">(.*?)<\/nav>/gs)][0]?.[1] ?? ''
  const order = ['Video', 'Format', 'Clips', 'Captions', 'Review'].map((label) => labels.indexOf(label))
  assert.ok(order.every((index, i) => index > -1 && (i === 0 || index > order[i - 1])))
  assert.match(html, /aria-current="step"[^>]*>.*?Video/s)
  assert.match(html, /Choose a video/)
  assert.equal(form.exports.useDraftStore.getState().workflow, null)
  const radios = html.match(/<button[^>]*role="radio"[^>]*>/g) ?? []
  assert.equal(radios.length, 2)
  assert.ok(radios.every((radio) => radio.includes('aria-checked="false"')))
  assert.equal(radios.filter((radio) => radio.includes('tabindex="0"')).length, 1)
  assert.match(html, /Beginner friendly/)
  assert.match(html, /For advanced users/)
})

test('a job requires an explicit workflow and each new video resets that choice', () => {
  const { useDraftStore, buildJobRequest } = form.exports
  const original = useDraftStore.getState()
  const trim = { start: null, end: null }
  try {
    original.update({ source: 'https://example.com/video', workflow: null })
    assert.throws(() => buildJobRequest(useDraftStore.getState(), trim), /Choose a workflow/)
    for (const workflow of ['automatic', 'review']) {
      original.update({ workflow })
      original.setStep('format')
      original.setStep('video')
      assert.equal(useDraftStore.getState().workflow, workflow)
      assert.equal(buildJobRequest(useDraftStore.getState(), trim).workflow, workflow)
      original.startAnother()
      assert.equal(useDraftStore.getState().workflow, null)
    }
  } finally { useDraftStore.setState(original) }
})

test('clipping mode is selectable and economy disables paid vision in the submitted request', () => {
  const { ClipsStep, buildJobRequest } = form.exports
  const draft = {
    workflow: 'automatic', source: 'https://example.com/video', clippingMode: 'economy', aspectRatio: '9:16', layoutStyle: 'auto',
    layoutVision: true, pacing: 'tight', durations: ['short'], autoClipCount: true, maxClips: 5,
    includeCaptions: true, captionPreset: 'pop'
  }
  const html = renderToStaticMarkup(React.createElement(ClipsStep, { draft, update() {} }))
  assert.match(html, /aria-label="Clipping mode"/)
  assert.match(html, /Economy/)
  const request = buildJobRequest(draft, { start: null, end: null })
  assert.equal(request.clippingMode, 'economy')
  assert.equal(Object.hasOwn(request, 'debugCapture'), false)
  assert.equal(request.layoutVision, false)
  assert.equal(buildJobRequest({ ...draft, clippingMode: 'quality' }, { start: null, end: null }).layoutVision, true)
})

test('local mode is offline-only, vertical, keyless, and capped at two candidates', () => {
  const { ClipsStep, buildJobRequest } = form.exports
  const draft = {
    workflow: 'review', source: 'C:\\Users\\me\\Videos\\source.mp4', clippingMode: 'local', aspectRatio: '16:9',
    layoutStyle: 'auto', layoutVision: true, pacing: 'tight', durations: ['short'], autoClipCount: false,
    maxClips: 8, includeCaptions: true, captionPreset: 'pop'
  }
  const html = renderToStaticMarkup(React.createElement(ClipsStep, { draft, update() {} }))
  const request = buildJobRequest(draft, { start: null, end: null })
  assert.match(html, /Faster-Whisper \+ Ollama/)
  assert.match(html, /zero, one, or two complete non-overlapping clips/)
  assert.match(html, /aria-label="Maximum clips"[^>]*value="2"/)
  assert.equal(request.clippingMode, 'local')
  assert.equal(request.videoUrl, draft.source)
  assert.equal(request.aspectRatio, '9:16')
  assert.equal(request.layoutVision, false)
  assert.equal(request.maxClips, 2)

  const localPicker = renderToStaticMarkup(React.createElement(SourcePicker, {
    value: '', onChange() {}, localOnly: true
  }))
  assert.match(localPicker, /Browse local video/)
  assert.match(localPicker, /never sends it to an API/)
  assert.doesNotMatch(localPicker, /aria-label="Video link"/)
})

test('the title card is shown by default and can be turned off for automatic runs', () => {
  const { CaptionsStep, useDraftStore, buildJobRequest } = form.exports
  const original = useDraftStore.getState()
  try {
    assert.equal(original.includeTitle, true)
    original.update({ workflow: 'automatic', source: 'https://example.com/video' })
    const automatic = renderToStaticMarkup(React.createElement(CaptionsStep, { draft: useDraftStore.getState(), update() {} }))
    assert.match(automatic, /Show title at the top/)
    assert.equal(buildJobRequest(useDraftStore.getState(), { start: null, end: null }).includeTitle, true)
    original.update({ includeTitle: false })
    assert.equal(buildJobRequest(useDraftStore.getState(), { start: null, end: null }).includeTitle, false)
    original.startAnother()
    assert.equal(useDraftStore.getState().includeTitle, false)
    // Review exports never draw a title card, so the switch is not offered there.
    const review = renderToStaticMarkup(React.createElement(CaptionsStep, { draft: { ...useDraftStore.getState(), workflow: 'review' }, update() {} }))
    assert.doesNotMatch(review, /Show title at the top/)
  } finally { useDraftStore.setState(original) }
})

test('what to clip is optional, trimmed into the request and cleared for the next video', () => {
  const { ClipsStep, useDraftStore, buildJobRequest } = form.exports
  const original = useDraftStore.getState()
  try {
    assert.equal(original.clipRequest, '')
    original.update({ workflow: 'automatic', source: 'https://example.com/video' })
    const html = renderToStaticMarkup(React.createElement(ClipsStep, { draft: useDraftStore.getState(), update() {} }))
    assert.match(html, /aria-label="What to clip"/)
    assert.match(html, /maxLength="1000"/)
    assert.match(html, /aria-describedby="clip-request-help"/)
    assert.match(html, /id="clip-request-help"[^>]*>Only matching moments are clipped/)
    assert.equal(Object.hasOwn(buildJobRequest(useDraftStore.getState(), { start: null, end: null }), 'clipRequest'), false)
    original.update({ clipRequest: '   ' })
    assert.equal(Object.hasOwn(buildJobRequest(useDraftStore.getState(), { start: null, end: null }), 'clipRequest'), false)
    original.update({ clipRequest: '  every time they talk about pricing \n' })
    assert.equal(buildJobRequest(useDraftStore.getState(), { start: null, end: null }).clipRequest, 'every time they talk about pricing')
    original.startAnother()
    assert.equal(useDraftStore.getState().clipRequest, '')
  } finally { useDraftStore.setState(original) }
})

test('a running job shows what the user asked to clip', () => {
  const { JobProgress } = form.exports
  const job = {
    id: 'job', revision: 1, status: 'planning', percent: 40, step: 'Finding moments', clipsDone: 0, clipsTotal: 0,
    queuedAt: Date.now(), startedAt: Date.now(),
    request: { videoUrl: 'https://www.youtube.com/watch?v=abc123def45', workflow: 'automatic', clipRequest: ' every time they talk about pricing ' }
  }
  const html = renderToStaticMarkup(React.createElement(JobProgress, { job, onCancel() {} }))
  assert.match(html, /title="every time they talk about pricing"><span class="sr-only">What to clip: <\/span>every time they talk about pricing</)
  const plain = renderToStaticMarkup(React.createElement(JobProgress, { job: { ...job, request: { ...job.request, clipRequest: undefined } }, onCancel() {} }))
  assert.doesNotMatch(plain, /What to clip/)
})

test('format and framing radio groups each expose one keyboard tab stop', () => {
  const draft = { aspectRatio: '9:16', layoutStyle: 'auto', layoutVision: true, pacing: 'tight' }
  const html = renderToStaticMarkup(React.createElement(FormatStep, { draft, update() {} }))
  for (const label of ['Format', 'Framing', 'Video speed']) {
    const group = html.match(new RegExp(`role="radiogroup" aria-label="${label}"[^>]*>(.*?)<\\/div>`, 's'))?.[1]
    assert.ok(group)
    assert.equal((group.match(/tabindex="0"/g) ?? []).length, 1)
  }
})

test('speed survives navigation and another job, and appears in the submitted request', () => {
  const { useDraftStore, buildJobRequest, ClipsStep } = form.exports
  const original = useDraftStore.getState()
  try {
    assert.equal(original.videoSpeed, 1)
    original.update({ workflow: 'automatic', source: 'https://example.com/video', videoSpeed: 1.5 })
    original.setStep('review')
    assert.equal(useDraftStore.getState().step, 'review')
    const lengths = renderToStaticMarkup(React.createElement(ClipsStep, { draft: useDraftStore.getState(), update() {} }))
    assert.match(lengths, /60 seconds becomes about 40 seconds/)
    assert.equal(buildJobRequest(useDraftStore.getState(), { start: 10, end: 70 }).videoSpeed, 1.5)
    original.startAnother()
    assert.equal(useDraftStore.getState().videoSpeed, 1.5)
    assert.equal(useDraftStore.getState().step, 'video')
  } finally { useDraftStore.setState(original) }
})

test('saved run speed is retained while invalid speed metadata is discarded', () => {
  assert.equal(parseJobOutput({ clips: [], metrics: { requested_settings: { video_speed: 1.5 } } }).metrics.requested_settings.video_speed, 1.5)
  for (const video_speed of ['2', null, Infinity, 0, 3]) {
    assert.equal(parseJobOutput({ clips: [], metrics: { requested_settings: { video_speed } } }).metrics.requested_settings.video_speed, undefined)
  }
})

test('editor project parsing preserves the optional local provider marker', () => {
  const { parseEditorProject } = form.exports
  const project = {
    version: 1, revision: 0, provider: 'local', title: 'Source', duration_ms: 60000,
    width: 1920, height: 1080, aspect_ratio: '9:16',
    transcript: [{ start_ms: 0, end_ms: 35000, text: 'A complete idea.' }],
    candidates: [{
      id: 'candidate-1', title: 'A complete idea.', ranges: [[0, 35000]],
      scenes: [{ at_ms: 0, layout: 'fit', crops: [[0, 0, 1, 1]] }],
      captions: true, caption_preset: 'pop', video_speed: 1, status: 'refining',
      caption_edits: [], caption_suppression_ranges: [], score: 93,
      local_selection: { decision: 'narrative_not_closed', narrative_closed: false },
      reason: 'The explanation reaches a clear conclusion.', review: null, exports: []
    }]
  }
  const parsed = parseEditorProject(project)
  assert.equal(parsed.provider, 'local')
  assert.equal(parsed.candidates[0].reason, 'The explanation reaches a clear conclusion.')
  assert.deepEqual(parsed.candidates[0].local_selection, { decision: 'narrative_not_closed', narrative_closed: false })
  assert.equal(parseEditorProject({ ...project, provider: undefined }).provider, undefined)
  assert.throws(() => parseEditorProject({ ...project, provider: 'unknown' }))
  const invalidSelection = structuredClone(project)
  invalidSelection.candidates[0].local_selection.narrative_closed = 'false'
  assert.throws(() => parseEditorProject(invalidSelection))
})

test('local provider is retained in sanitized run metadata', () => {
  const local = parseJobOutput({ clips: [], metrics: { requested_settings: { clipping_mode: 'local' } } })
  assert.equal(local.metrics.requested_settings.clipping_mode, 'local')
})

test('advanced selections travel with the job while presets ignore retained custom choices', () => {
  const { ClipsStep, buildJobRequest } = form.exports
  const draft = {
    workflow: 'automatic', source: 'https://example.com/video', clippingMode: 'advanced',
    plannerModel: 'provider/planning', transcriptionModel: 'provider/speech',
    aspectRatio: '9:16', layoutStyle: 'auto', layoutVision: true, pacing: 'tight',
    durations: ['short'], autoClipCount: true, maxClips: 5, includeCaptions: true, captionPreset: 'pop'
  }
  const html = renderToStaticMarkup(React.createElement(ClipsStep, { draft, update() {} }))
  assert.match(html, /Clip planning model/)
  assert.match(html, /Transcription model/)
  assert.equal((html.match(/role="combobox"/g) ?? []).length, 2)
  const request = buildJobRequest(draft, { start: null, end: null })
  assert.equal(request.plannerModel, 'provider/planning')
  assert.equal(request.transcriptionModel, 'provider/speech')
  assert.equal(request.layoutVision, true)
  for (const clippingMode of ['quality', 'economy']) {
    const preset = buildJobRequest({ ...draft, clippingMode }, { start: null, end: null })
    assert.equal(preset.plannerModel, undefined)
    assert.equal(preset.transcriptionModel, undefined)
  }
})

test('clip list explains when smart framing intentionally keeps the whole frame', () => {
  const clip = (index) => ({
    clip_index: index, s3_url: `/tmp/clip-${index}.mp4`, duration_ms: 5000,
    start_time_ms: index * 5000, end_time_ms: (index + 1) * 5000, virality_score: 0.5
  })
  const output = parseJobOutput({
    clips: [clip(0), clip(1)],
    metrics: { clip_layouts: [
      { clip_index: 0, framing_status: 'whole_frame_auto' },
      { clip_index: 0, framing_status: 'whole_frame_auto' },
      { clip_index: 9, framing_status: 'whole_frame_auto' }
    ] }
  })
  assert.ok(output)
  assert.match(framingProblem(output, true), /whole frame for clip 1/)
  assert.equal(framingProblem(output, false), null)
  const classic = parseJobOutput({ clips: [clip(0)], metrics: {
    smart_framing_available: false, requested_settings: { aspect_ratio: '9:16', layout_style: 'fit' }
  } })
  assert.ok(classic)
  assert.equal(framingProblem(classic, true), null)
})

test('visual-only runs disclose unavailable captions and preserve analysis status', () => {
  const output = parseJobOutput({ clips: [], metrics: {
    transcription_status: 'no_speech', planning_source: 'visual', visual_frame_count: 12,
    captions_status: 'unavailable_without_transcript'
  } })
  assert.ok(output)
  assert.equal(output.metrics.visual_frame_count, 12)
  assert.equal(output.metrics.captions_status, 'unavailable_without_transcript')
  assert.match(sourceAnalysisNotice(output), /No speech was detected/)
})

test('Twitch VOD links canonicalize while other Twitch pages are rejected', () => {
  const { normalizeVideoSource, twitchVodId, SourcePicker } = form.exports
  for (const host of ['twitch.tv', 'www.twitch.tv', 'm.twitch.tv', 'go.twitch.tv']) {
    const source = `https://${host}/videos/12345/?t=1h&tracking=secret`
    assert.equal(isValidSourceLink(source), true)
    assert.equal(normalizeVideoSource(source), 'https://www.twitch.tv/videos/12345')
    assert.equal(twitchVodId(source), '12345')
  }
  for (const source of ['https://twitch.tv/channel', 'https://clips.twitch.tv/Clip', 'https://player.twitch.tv/?video=123', 'https://twitch.tv/videos/nope', 'https://twitch.tv:8443/videos/123']) assert.equal(isValidSourceLink(source), false)
  assert.equal(twitchVodId('https://twitch.tv.evil.test/videos/123'), null)
  const html = renderToStaticMarkup(React.createElement(SourcePicker, { value: 'https://www.twitch.tv/videos/12345', onChange() {} }))
  assert.match(html, /Twitch VOD/)
  assert.match(html, /Public, completed videos only/)
  assert.doesNotMatch(html, /<img/)
})


test('Review & edit shows required Jev review even when automatic review is off', () => {
  const { ClipsStep, useDraftStore, useSettingsStore } = form.exports
  const original = useSettingsStore.getState()
  const initialState = useSettingsStore.getInitialState()
  const initialJev = initialState.jevEnabled
  try {
    useSettingsStore.setState({ jevEnabled: 'off' })
    initialState.jevEnabled = 'off'
    const draft = useDraftStore.getState()
    const review = renderToStaticMarkup(React.createElement(ClipsStep, { draft: { ...draft, workflow: 'review' }, update() {} }))
    const automatic = renderToStaticMarkup(React.createElement(ClipsStep, { draft: { ...draft, workflow: 'automatic' }, update() {} }))
    assert.match(review, /Jev review required/)
    assert.doesNotMatch(review, /Jev review &amp; repairs off/)
    assert.match(automatic, /Jev review &amp; repairs off/)
    assert.equal(useSettingsStore.getState().jevEnabled, 'off')
  } finally { initialState.jevEnabled = initialJev; useSettingsStore.setState(original) }
})
