import { cpSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync, unlinkSync } from 'fs'
import { join } from 'path'

/** Chromium caches must not compete with legacy caches or persistent app data. */
export function prepareSessionCache(userData: string, temporary: string, packaged: boolean): { directory: string; fallback: boolean } {
  let directory = join(userData, packaged ? 'chromium-session' : 'chromium-session-dev')
  let fallback = false
  try {
    mkdirSync(directory, { recursive: true })
    const probe = join(directory, `.write-probe-${process.pid}`)
    writeFileSync(probe, '', { flag: 'wx' })
    unlinkSync(probe)
  } catch {
    // Leave inaccessible caches intact; a fresh temporary session remains writable.
    directory = mkdtempSync(join(temporary, 'bridgeclip-session-'))
    fallback = true
  }
  // Preserve sidebar/profile preferences; never move or delete the old database.
  // A copy can fail on Windows when a file is briefly locked. The legacy store
  // stays untouched and the next launch retries instead of blocking startup.
  const storage = join(userData, 'Local Storage')
  const destination = join(directory, 'Local Storage')
  if (existsSync(storage) && !existsSync(destination)) {
    try {
      cpSync(storage, destination, { recursive: true, errorOnExist: true, force: false })
    } catch {
      rmSync(destination, { recursive: true, force: true })
    }
  }
  return { directory, fallback }
}
