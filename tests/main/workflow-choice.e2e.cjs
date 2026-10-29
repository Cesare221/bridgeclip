const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { buildApp, launchApp } = require('../zernio/support/electron-app.cjs')

test('each new video requires a workflow, with keyboard selection and guarded navigation and submission', { timeout: 90000 }, async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'bridgeclip-workflow-e2e-'))
  const session = await launchApp({ appDir: buildApp(path.join(root, 'app')), userDataDir: path.join(root, 'user-data') })
  t.after(async () => { await session.close(); fs.rmSync(root, { recursive: true, force: true }) })
  const { app, page } = session
  const errors = []
  page.on('pageerror', (error) => errors.push(error.message))
  await app.evaluate(({ ipcMain }, root) => {
    globalThis.workflowTest = []
    ipcMain.removeHandler('settings:load')
    ipcMain.handle('settings:load', () => ({ openrouterConfigured: true, zernioConfigured: false, outputDirectory: root, pythonPath: '', customVocabulary: '' }))
    ipcMain.removeHandler('system:checkTools')
    ipcMain.handle('system:checkTools', () => ({ python: true, pythonDeps: true, ffmpeg: true, ffmpegCaptions: true, ffprobe: true, ytdlp: true, engine: true, bridgeRunner: true }))
    ipcMain.removeHandler('job:start')
    ipcMain.handle('job:start', (_event, request) => {
      globalThis.workflowTest.push(request)
      const id = `workflow-${globalThis.workflowTest.length}`
      const now = new Date().toISOString()
      return { jobId: id, queued: false, job: {
        id, revision: 0, request, status: 'running', percent: 0, step: 'Queued',
        clipsDone: 0, clipsTotal: 0, error: null, errorHint: null, output: null,
        outputDir: root, queuedAt: now, startedAt: now, finishedAt: null
      } }
    })
  }, root)
  await page.reload()
  await page.emulateMedia({ reducedMotion: 'no-preference' })
  const group = page.getByRole('radiogroup', { name: 'Workflow', exact: true })
  const automatic = group.getByRole('radio', { name: 'Automatic', exact: true })
  const review = group.getByRole('radio', { name: 'Review & edit', exact: true })
  const next = page.getByRole('button', { name: 'Next: Format', exact: true })
  const steps = page.getByRole('navigation', { name: 'Create steps' })
  const shortcut = process.platform === 'darwin' ? 'Meta+Enter' : 'Control+Enter'
  const assertSelectionMotion = async (radio) => {
    const motion = await radio.locator('svg.workflow-illustration').evaluate(async (svg) => {
      const animations = svg.getAnimations({ subtree: true })
      const finite = animations.every((animation) => {
        const timing = animation.effect.getTiming()
        return timing.iterations === 1 && timing.duration + timing.delay < 1000
      })
      // Inspect an intermediate frame to prove the illustration actually moves.
      for (const animation of animations) { animation.pause(); animation.currentTime = 300 }
      const moving = [...svg.querySelectorAll('*')].some((el) => getComputedStyle(el).transform !== 'none')
      for (const animation of animations) animation.finish()
      await Promise.all(animations.map((animation) => animation.finished))
      return { count: animations.length, finite, moving }
    })
    assert.ok(motion.count > 0)
    assert.equal(motion.finite, true)
    assert.equal(motion.moving, true)
  }
  const addSource = async () => {
    await page.getByPlaceholder('YouTube, Twitch VOD or direct video link').fill('https://example.com/video.mp4')
    await page.getByRole('button', { name: 'Use link', exact: true }).click()
  }
  const assertUnselected = async () => {
    assert.equal(await automatic.getAttribute('aria-checked'), 'false')
    assert.equal(await review.getAttribute('aria-checked'), 'false')
    assert.equal(await automatic.getAttribute('tabindex'), '0')
    assert.equal(await next.isDisabled(), true)
    assert.equal(await steps.getByRole('button', { name: /Format/ }).isDisabled(), true)
    assert.equal(await steps.getByRole('button', { name: /Review/ }).isDisabled(), true)
  }
  await assertUnselected()
  await group.getByText('Beginner friendly', { exact: true }).waitFor()
  await group.getByText('For advanced users', { exact: true }).waitFor()
  const sourceHeading = await page.getByRole('heading', { name: 'Choose a video', exact: true }).boundingBox()
  const workflowBounds = await group.boundingBox()
  assert.ok(sourceHeading.y > workflowBounds.y + workflowBounds.height, 'video heading belongs below the workflow cards')
  const artifacts = process.env.BRIDGECLIP_E2E_SHOTS
  if (artifacts) {
    fs.mkdirSync(artifacts, { recursive: true })
    await page.screenshot({ path: path.join(artifacts, 'workflow-unselected.png') })
  }
  await automatic.click()
  await assertSelectionMotion(automatic)
  await automatic.click()
  await assertSelectionMotion(automatic)
  assert.equal(await next.isDisabled(), true, 'choosing a workflow still requires a video')
  await page.reload()
  await addSource()
  await assertUnselected()
  await page.keyboard.press(shortcut)
  assert.deepEqual(await app.evaluate(() => globalThis.workflowTest), [])
  // Tab into an unselected group without silently making a choice.
  await steps.getByRole('button', { name: /Video/ }).focus()
  await page.keyboard.press('Tab')
  assert.equal(await automatic.evaluate((el) => el === document.activeElement), true)
  assert.equal(await automatic.getAttribute('aria-checked'), 'false')
  await page.keyboard.press('ArrowRight')
  assert.equal(await review.getAttribute('aria-checked'), 'true')
  await assertSelectionMotion(review)
  assert.equal(await next.isEnabled(), true)
  await next.click()
  assert.equal(await page.getByRole('switch', { name: 'Capture framing diagnostics', exact: true }).count(), 0)
  await page.getByRole('button', { name: 'Back', exact: true }).click()
  assert.equal(await review.getAttribute('aria-checked'), 'true')
  assert.equal(await review.locator('svg.workflow-illustration').evaluate((svg) => svg.getAnimations({ subtree: true }).length), 0)
  await page.getByRole('button', { name: 'Jobs', exact: true }).click()
  await page.getByRole('button', { name: 'Create', exact: true }).click()
  assert.equal(await review.getAttribute('aria-checked'), 'true')
  await automatic.click()
  await page.keyboard.press(shortcut)
  await page.getByRole('button', { name: 'Clip another video' }).waitFor()
  assert.equal(await app.evaluate(() => globalThis.workflowTest[0].workflow), 'automatic')
  await page.getByRole('button', { name: 'Clip another video' }).click()
  await addSource()
  await assertUnselected()
  await page.keyboard.press(shortcut)
  assert.equal(await app.evaluate(() => globalThis.workflowTest.length), 1)
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await automatic.click()
  assert.equal(await automatic.locator('svg.workflow-illustration').evaluate((svg) => svg.getAnimations({ subtree: true }).length), 0)
  await review.click()
  assert.equal(await review.locator('svg.workflow-illustration').evaluate((svg) => svg.getAnimations({ subtree: true }).length), 0)
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setContentSize(720, 700))
  await page.waitForFunction(() => window.innerWidth === 720)
  if (artifacts) await page.screenshot({ path: path.join(artifacts, 'workflow-compact.png') })
  await page.waitForFunction(() => document.documentElement.scrollWidth <= window.innerWidth, null, { timeout: 5000 })
  // Compact windows hide inactive step labels but keep the five step buttons.
  await steps.getByRole('button').nth(4).click()
  await page.getByRole('button', { name: 'Find candidates', exact: true }).click()
  await steps.waitFor({ state: 'hidden' })
  await page.getByRole('button', { name: 'Create', exact: true }).click()
  await page.getByRole('button', { name: 'Clip another video' }).waitFor()
  assert.equal(await app.evaluate(() => globalThis.workflowTest[1].workflow), 'review')
  assert.deepEqual(errors, [])
})
