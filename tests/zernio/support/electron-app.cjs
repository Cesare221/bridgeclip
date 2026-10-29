'use strict'
// Builds BridgeClip into a scratch folder and launches it as an isolated
// Electron instance for end-to-end tests. The instance gets its own userData
// (and single-instance lock), talks to a mock Zernio, uses a scripted browser
// instead of the real one, a mock keychain, and a hidden window. It never
// touches the repo's out/ folder or a running `npm run dev`.
//
// Uses the pinned playwright-core dev dependency. BRIDGECLIP_E2E_TOOLS can
// point to an isolated installation when running against a local checkout.

const { execFileSync } = require('node:child_process')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '../../..')

function playwright() {
  for (const base of [process.env.BRIDGECLIP_E2E_TOOLS, ROOT].filter(Boolean)) {
    try {
      return require(require.resolve('playwright-core', { paths: [base] }))
    } catch { /* try the next place */ }
  }
  throw new Error('playwright-core not found. Install it outside the repo and set BRIDGECLIP_E2E_TOOLS to that folder.')
}

/**
 * Production-builds the app into `appDir/out` and makes `appDir` launchable
 * (package.json + a node_modules link for externalised dependencies).
 */
function buildApp(appDir = process.env.BRIDGECLIP_E2E_APP_DIR || path.join(os.tmpdir(), 'bridgeclip-e2e-app'), { skipBuild = process.env.BRIDGECLIP_E2E_SKIP_BUILD === '1' } = {}) {
  fs.mkdirSync(appDir, { recursive: true })
  if (!skipBuild || !fs.existsSync(path.join(appDir, 'out/main/index.js'))) {
    const electronViteCli = path.resolve(path.dirname(require.resolve('electron-vite', { paths: [ROOT] })), '../bin/electron-vite.js')
    execFileSync(process.execPath, [electronViteCli, 'build', '--outDir', path.join(appDir, 'out')], { cwd: ROOT, stdio: 'inherit' })
  }
  const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'))
  fs.writeFileSync(path.join(appDir, 'package.json'), JSON.stringify({ name: pkg.name, productName: pkg.productName, version: pkg.version, main: 'out/main/index.js' }, null, 2))
  const modules = path.join(appDir, 'node_modules')
  if (!fs.existsSync(modules)) fs.symlinkSync(path.join(ROOT, 'node_modules'), modules, 'junction')
  return appDir
}

/**
 * @param {{ appDir: string, userDataDir: string, mock?: { apiUrl: string, browserUrl: string }, apiUrl?: string, env?: object }} options
 * @returns {Promise<{ app: import('playwright-core').ElectronApplication, page: import('playwright-core').Page, close: () => Promise<void> }>}
 */
async function launchApp({ appDir, userDataDir, mock, apiUrl, env = {} }) {
  // Settings migrate (and delete) pre-rename files under the real appData and
  // home. Only a build whose isolation hook also moves those may be launched.
  const main = fs.readFileSync(path.join(appDir, 'out/main/index.js'), 'utf8')
  if (!main.includes('isolated-appData') && !main.includes('isolated-${')) {
    throw new Error('This build predates full userData isolation; rebuild before launching it.')
  }
  // Settings for a fresh run, with no keys and an output folder inside the
  // isolated dir (the app's default is the real ~/BridgeClip).
  const settingsFile = path.join(userDataDir, 'settings.json')
  const outputDirectory = path.join(userDataDir, 'BridgeClip')
  if (!fs.existsSync(settingsFile)) {
    fs.mkdirSync(userDataDir, { recursive: true })
    fs.writeFileSync(settingsFile, JSON.stringify({ version: 6, openrouterApiKey: '', zernioApiKey: '', outputDirectory, pythonPath: 'python3',  }), { mode: 0o600 })
  }
  const { _electron } = playwright()
  const electronPath = require(path.join(ROOT, 'node_modules/electron'))
  const cleanEnv = { ...process.env }
  for (const name of ['ELECTRON_RUN_AS_NODE', 'ELECTRON_RENDERER_URL', 'BRIDGECLIP_ZERNIO_API_URL', 'BRIDGECLIP_E2E_BROWSER_URL']) delete cleanEnv[name]
  const app = await _electron.launch({
    executablePath: electronPath,
    // A mock keychain keeps safeStorage off the developer's real keychain.
    args: [appDir, '--use-mock-keychain'],
    env: {
      ...cleanEnv,
      ...env,
      ...(mock || apiUrl ? { BRIDGECLIP_ZERNIO_API_URL: apiUrl ?? mock.apiUrl } : {}),
      ...(mock ? { BRIDGECLIP_E2E_BROWSER_URL: mock.browserUrl } : {}),
      // Isolation (own userData, hidden window, no real browser) can't be overridden.
      BRIDGECLIP_USER_DATA_DIR: userDataDir,
      BRIDGECLIP_E2E: '1'
    },
    timeout: 60_000
  })
  const paths = await app.evaluate(({ app: electronApp }) => ['userData', 'appData', 'home', 'logs'].map((name) => [name, electronApp.getPath(name)]))
  const real = (p) => { try { return fs.realpathSync(p) } catch { return path.resolve(p) } }
  const root = real(userDataDir)
  const outside = paths.filter(([, p]) => !real(p).startsWith(root)).map(([name, p]) => `${name}=${p}`)
  if (outside.length > 0) {
    await app.close().catch(() => {})
    throw new Error(`The test app is not isolated from ${userDataDir}: ${outside.join(', ')}`)
  }
  const page = await app.firstWindow()
  if (process.platform === 'linux' && process.env.GITHUB_ACTIONS === 'true') {
    // Linux CI runs inside Xvfb. Map the window onto that virtual display so
    // Chromium advances media, requestAnimationFrame and CSS animations.
    // Local test runs still stay hidden on the developer's desktop.
    await app.evaluate(({ BrowserWindow }) => {
      const window = BrowserWindow.getAllWindows()[0]
      window.webContents.setBackgroundThrottling(false)
      window.show()
    })
  }
  await page.waitForLoadState('domcontentloaded')
  const settings = await page.evaluate(() => window.bridgeclip.settings.load())
  if (!real(settings.outputDirectory).startsWith(root) && !path.resolve(settings.outputDirectory).startsWith(path.resolve(userDataDir))) {
    await app.close().catch(() => {})
    throw new Error(`The test app's output folder is outside ${userDataDir}.`)
  }
  return { app, page, close: () => app.close().catch(() => {}) }
}

module.exports = { buildApp, launchApp, ROOT }
