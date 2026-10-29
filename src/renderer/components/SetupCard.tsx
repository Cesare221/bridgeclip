import { KeyRound, ShieldCheck, Wrench } from 'lucide-react'
import { useSettingsStore, useSetupState } from '../store/use-settings-store'
import { useApiKeyDrafts } from '../hooks/use-api-key-drafts'
import { PROVIDER_LINKS } from '../config/brand'
import { cn } from '../lib/utils'
import { ApiKeyInput } from './ApiKeyInput'
import { Button } from './ui/Button'
import { IconTile } from './ui/IconTile'

/**
 * First-run setup, inline on Create: paid modes need an OpenRouter key; Local does not.
 */
export function SetupCard({ onOpenSettings, className, localMode = false }: { onOpenSettings: () => void; className?: string; localMode?: boolean }): React.JSX.Element | null {
  const { missingKeys, toolsOk } = useSetupState(localMode)
  const toolStatus = useSettingsStore((s) => s.toolStatus)
  const toolError = useSettingsStore((s) => s.toolError)
  const { drafts, setDraft, persist, error } = useApiKeyDrafts()
  const missingTools = toolStatus
    ? [
        !toolStatus.python && 'Python',
        !toolStatus.pythonDeps && 'Clipping dependencies or smart framing',
        !toolStatus.ffmpeg && 'FFmpeg',
        !toolStatus.ffprobe && 'FFprobe',
        !localMode && !toolStatus.ytdlp && 'yt-dlp',
        !toolStatus.engine && 'BridgeClip clipping engine',
        !toolStatus.bridgeRunner && 'Bridge runner'
      ].filter(Boolean).join(', ')
    : ''

  if (missingKeys.length > 0 && !localMode) {
    return (
      <section
        className={cn(
          'glass relative overflow-hidden rounded-3xl bg-accent/[0.05] animate-fade-in',
          'shadow-[inset_0_1px_0_rgb(255_255_255/0.08),inset_0_0_0_1px_rgb(var(--accent)/0.22)]',
          className
        )}
        aria-label="Finish setup"
      >
        <div className="relative flex items-start gap-3 p-3.5 pb-0">
          <IconTile tone="accent">
            <KeyRound />
          </IconTile>
          <div className="min-w-0">
            <p className="eyebrow text-accent">One-time setup</p>
            <h2 className="mt-0.5 text-base font-semibold text-ink">Connect OpenRouter</h2>
            <p className="mt-0.5 max-w-2xl text-xs text-ink-muted">
              BridgeClip has no account and no server. One OpenRouter key covers transcription with MAI Transcribe 2 and clip selection.
            </p>
          </div>
        </div>
        <div className="relative m-3 grid gap-3 rounded-xl bg-black/15 p-3 shadow-[inset_0_0_0_1px_rgb(255_255_255/0.06)]">
          <ApiKeyInput
            label="OpenRouter"
            value={drafts.openrouterApiKey}
            onChange={(v) => setDraft('openrouterApiKey', v)}
            onBlur={() => void persist()}
            placeholder="sk-or-…"
            getKeyUrl={PROVIDER_LINKS.openrouter}
          />
        </div>
        <p className="relative -mt-1 flex items-center gap-1.5 px-3.5 pb-3 text-2xs text-ink-subtle">
          <ShieldCheck className="h-3.5 w-3.5 text-success/80" />
          Your key is encrypted on this computer with your system keychain.
        </p>
        {error && <p role="alert" className="relative -mt-2 px-4 pb-4 text-xs text-danger">{error}</p>}
      </section>
    )
  }

  if (toolsOk === false) {
    return (
      <section
        className={cn(
          'flex items-center gap-3 rounded-3xl bg-danger/[0.07] p-3 pr-4 animate-fade-in',
          'shadow-[inset_0_0_0_1px_rgb(var(--danger)/0.26),inset_0_1px_0_rgb(255_255_255/0.06)]',
          className
        )}
      >
        <IconTile tone="danger">
          <Wrench />
        </IconTile>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-ink">{toolError ? 'Could not check required tools' : 'Some required tools are missing'}</p>
          <p className="mt-0.5 text-xs text-ink-muted">{toolError ? 'Open Settings to retry the system check.' : `Missing: ${missingTools}. Open Settings for details.`}</p>
        </div>
        <Button onClick={onOpenSettings}>See what’s missing</Button>
      </section>
    )
  }

  return null
}
