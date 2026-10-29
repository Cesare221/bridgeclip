const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('typescript')

function loadJobStart(mocks) {
  const source = fs.readFileSync(path.join(__dirname, '../../src/main/job-start.ts'), 'utf8')
  const js = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
  }).outputText
  const module = { exports: {} }
  vm.runInNewContext(js, {
    module,
    exports: module.exports,
    require: (id) => mocks[id] ?? require(id),
    URL,
    Set,
    Map,
    process,
    console
  })
  return module.exports
}

function createJobStart(settings) {
  const calls = []
  const jobStart = loadJobStart({
    crypto: { randomUUID: () => 'local-job-id' },
    './settings-store': { loadSettings: () => settings },
    './file-manager': { ensureOutputDir: (...args) => calls.push(['output', ...args]) },
    './pipeline-runner': {
      getBridgeRunnerPath: () => 'bridge_runner.py',
      getEnginePath: () => 'engine',
      preflightCheck: () => ({ ok: true }),
      resolvePythonPath: () => 'python',
      validatePython: async () => ({ ok: true })
    },
    './run-history': {
      createRunRecord: (...args) => calls.push(['record', ...args]),
      finishRunRecord: () => {}
    },
    './job-manager': {
      enqueueJob: (...args) => {
        calls.push(['enqueue', ...args])
        return { status: 'pending' }
      }
    },
    './logger': { logger: { warn() {}, error() {}, info() {} } },
    './security': {
      assertMediaPath: (videoPath) => calls.push(['media', videoPath]),
      assertSelectedVideoPath: (videoPath) => calls.push(['selected-video', videoPath]),
      isWebUrl: (videoPath) => /^https?:/.test(videoPath)
    },
    './network-policy': { assertPublicWebUrl: async () => {} },
    './validation': { validateJobConfig: (config) => config },
    './openrouter-models': { resolveAdvancedModels: async () => ({}) },
    './tools': { supportsCaptionFilter: async () => true }
  })
  return { jobStart, calls }
}

test('local jobs start without an OpenRouter key while other modes still require one', async () => {
  const settings = { openrouterApiKey: '', outputDirectory: 'output', pythonPath: 'python' }
  const { jobStart, calls } = createJobStart(settings)
  const localResult = await jobStart.startClipJobRequest({
    videoUrl: 'C:\\Users\\me\\Videos\\source.mp4',
    clippingMode: 'local',
    includeCaptions: false
  })

  assert.equal(localResult.jobId, 'local-job-id')
  assert.equal(localResult.queued, false)
  assert.ok(calls.some(([kind]) => kind === 'selected-video'))
  assert.ok(calls.some(([kind]) => kind === 'enqueue'))

  const { jobStart: cloudJobStart, calls: cloudCalls } = createJobStart(settings)
  const cloudResult = await cloudJobStart.startClipJobRequest({
    videoUrl: 'C:\\Users\\me\\Videos\\source.mp4',
    clippingMode: 'quality',
    includeCaptions: false
  })
  assert.match(cloudResult.error, /OpenRouter API key is required/)
  assert.equal(cloudCalls.some(([kind]) => kind === 'enqueue'), false)
})
