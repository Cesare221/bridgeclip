import { randomUUID } from 'crypto'
import { loadSettings } from './settings-store'
import { ensureOutputDir } from './file-manager'
import { getBridgeRunnerPath, getEnginePath, preflightCheck, resolvePythonPath, validatePython, type ClipJobConfig } from './pipeline-runner'
import { createRunRecord, finishRunRecord } from './run-history'
import { enqueueJob } from './job-manager'
import { logger } from './logger'
import { assertMediaPath, assertSelectedVideoPath, isWebUrl } from './security'
import { assertPublicWebUrl } from './network-policy'
import { validateJobConfig } from './validation'
import { resolveAdvancedModels } from './openrouter-models'
import { supportsCaptionFilter } from './tools'
import type { JobSnapshot } from '../shared/jobs'

/** A passing engine check is reused briefly, so queuing several videos stays quick. */
const ENGINE_CHECK_TTL_MS = 5 * 60 * 1000
let lastEngineCheck: { key: string; at: number } | null = null

export type StartJobResult = { jobId: string; queued: boolean; job: JobSnapshot } | { error: string }

/**
 * Validate a clipping request, check the engine and keys, create its run
 * folder and queue it. Used by the Create page and the assistant alike; it
 * needs no window, so jobs can start while every window is closed.
 */
export async function startClipJobRequest(request: unknown): Promise<StartJobResult> {
  let config: ClipJobConfig
  try {
    config = validateJobConfig(request)
    if (config.clippingMode === 'advanced') {
      config.plannerCapabilities = await resolveAdvancedModels(config.plannerModel!, config.transcriptionModel!)
    }
    if (isWebUrl(config.videoUrl)) await assertPublicWebUrl(config.videoUrl)
    else if (config.clippingMode === 'local') assertSelectedVideoPath(config.videoUrl)
    else assertMediaPath(config.videoUrl, loadSettings().outputDirectory)
    if (config.bannerChannelUrl) await assertPublicWebUrl(config.bannerChannelUrl)
  } catch (error) { return { error: error instanceof Error ? error.message : 'Invalid job options' } }
  const settings = loadSettings()

  if (config.clippingMode !== 'local' && !settings.openrouterApiKey) {
    logger.warn('job.start.missingKey', { key: 'OPENROUTER_API_KEY' })
    return { error: 'OpenRouter API key is required for AI clip planning. Go to Settings to add it.' }
  }

  const enginePath = getEnginePath()
  const bridgePath = getBridgeRunnerPath()
  const pythonPath = resolvePythonPath(enginePath, settings.pythonPath)

  const preflight = preflightCheck({ pythonPath, bridgePath, enginePath })
  if (!preflight.ok) {
    const message = preflight.hint
      ? `${preflight.error}\n\n${preflight.hint}`
      : preflight.error!
    logger.error('job.start.preflight.failed', {
      error: preflight.error,
      hint: preflight.hint,
      pythonPath,
      bridgePath,
      enginePath
    })
    return { error: message }
  }
  const engineKey = `${pythonPath}\0${enginePath}`
  if (!lastEngineCheck || lastEngineCheck.key !== engineKey || Date.now() - lastEngineCheck.at > ENGINE_CHECK_TTL_MS) {
    const pythonValidation = await validatePython(pythonPath, enginePath)
    if (!pythonValidation.ok) {
      lastEngineCheck = null
      return { error: 'The clipping engine is incomplete or incompatible. Open Settings → System check, then repair the BridgeClip installation before starting.' }
    }
    lastEngineCheck = { key: engineKey, at: Date.now() }
  }
  if (config.includeCaptions && !(await supportsCaptionFilter())) {
    return { error: 'FFmpeg cannot render captions because its ass filter is missing. Install an FFmpeg build with libass, or turn captions off.' }
  }

  ensureOutputDir(settings.outputDirectory)

  const jobId = randomUUID()
  logger.info('job.start.request', { jobId, sourceType: isWebUrl(config.videoUrl) ? 'remote' : 'local', aspectRatio: config.aspectRatio })
  try {
    createRunRecord(settings.outputDirectory, jobId, config.videoUrl)
  } catch {
    try { finishRunRecord(settings.outputDirectory, jobId, 'failed', 'Could not start this run.') } catch { /* Output folder may be unavailable. */ }
    return { error: 'Could not create the clipping run. Check the output folder and retry.' }
  }
  // Starts now when a slot is free; otherwise waits its turn in the queue.
  const job = enqueueJob(jobId, config, settings.outputDirectory)
  return { jobId, queued: job.status === 'queued', job }
}
