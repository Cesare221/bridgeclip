const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { buildApp, launchApp } = require('../zernio/support/electron-app.cjs')

test('Local is available on entry, survives navigation and needs no paid setup', { timeout: 120000 }, async (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'bridgeclip-local-create-'))
  const session = await launchApp({ appDir: buildApp(path.join(root, 'app')), userDataDir: path.join(root, 'user') })
  t.after(async () => { await session.close(); fs.rmSync(root, { recursive: true, force: true }) })
  const { app, page } = session
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await app.evaluate(({ app, ipcMain }) => {
    globalThis.sessionCache = app.getPath('sessionData')
    ipcMain.removeHandler('system:checkTools')
    ipcMain.handle('system:checkTools', () => ({ python: true, pythonDeps: true, ffmpeg: true, ffmpegCaptions: true, ffprobe: true, ytdlp: false, engine: true, bridgeRunner: true }))
    ipcMain.removeHandler('dialog:selectVideo')
    ipcMain.handle('dialog:selectVideo', () => 'C:\\Videos\\source.mp4')
  })
  await page.reload()
  const modes = page.getByRole('radiogroup', { name: 'Clipping mode', exact: true })
  assert.equal(await modes.getByRole('radio').count(), 4)
  const local = modes.getByRole('radio', { name: /^Local/ })
  await local.click()
  assert.equal(await page.getByText('Connect OpenRouter', { exact: true }).count(), 0)
  await page.getByRole('radio', { name: 'Review & edit', exact: true }).click()
  // No inference is submitted: this test covers navigation with tools available.
  await page.getByRole('button', { name: /Browse local video/ }).click()
  for (const step of ['Format', 'Clips', 'Captions', 'Review']) {
    await page.getByRole('button', { name: `Next: ${step}`, exact: true }).click()
    assert.equal(await local.getAttribute('aria-checked'), 'true')
    assert.equal(await page.getByText('Check tricky shots with AI vision', { exact: true }).count(), 0)
  }
  assert.equal(await page.getByRole('button', { name: 'Find candidates', exact: true }).isEnabled(), true)
  await page.getByRole('navigation', { name: 'Create steps' }).getByRole('button', { name: /Video/ }).click()
  await page.getByRole('radio', { name: 'Automatic', exact: true }).click()
  assert.equal(await local.getAttribute('aria-checked'), 'true')
  await page.getByRole('navigation', { name: 'Create steps' }).getByRole('button', { name: /Review/ }).click()
  assert.equal(await page.getByRole('button', { name: 'Generate clips', exact: true }).isEnabled(), true)
  await modes.getByRole('radio', { name: /^Quality/ }).click()
  assert.equal(await page.getByRole('button', { name: 'Generate clips', exact: true }).isDisabled(), true)
  await local.click()
  assert.match(await app.evaluate(() => globalThis.sessionCache), /chromium-session-dev$/)
  assert.deepEqual(errors, [])
  if (process.env.BRIDGECLIP_E2E_SHOTS) {
    fs.mkdirSync(process.env.BRIDGECLIP_E2E_SHOTS, { recursive: true })
    await page.screenshot({ path: path.join(process.env.BRIDGECLIP_E2E_SHOTS, 'local-review.png') })
  }
})
