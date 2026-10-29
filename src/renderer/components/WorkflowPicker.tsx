import { Check } from 'lucide-react'
import { useState } from 'react'
import type { ClipDraft } from '../store/use-draft-store'
import { cn } from '../lib/utils'
import { onRadioKeyDown } from './ui/Segmented'
import './workflow-picker.css'

const WORKFLOWS = [
  {
    id: 'automatic', title: 'Automatic', level: 'Beginner friendly', outcome: 'Go straight to finished clips',
    description: 'AI finds the moments, checks the cuts and exports your clips for you.',
    steps: 'Find → Check → Export'
  },
  {
    id: 'review', title: 'Review & edit', level: 'For advanced users', outcome: 'Make the final cut yourself',
    description: 'Review candidate moments, adjust cuts and framing, then approve each clip before export.',
    steps: 'Find → You edit → Export'
  }
] as const

export function WorkflowPicker({ value, onChange, disabled }: {
  value: ClipDraft['workflow']
  onChange: (value: NonNullable<ClipDraft['workflow']>) => void
  disabled?: boolean
}): React.JSX.Element {
  // Remount only the illustration to replay on click or keyboard selection.
  // Returning to an existing draft leaves its illustration still.
  const [selectionRevision, setSelectionRevision] = useState(0)
  return (
    <section aria-labelledby="workflow-heading" className="space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h2 id="workflow-heading" className="text-sm font-semibold text-ink">How would you like to create?</h2>
        <p id="workflow-help" className="text-2xs text-ink-muted">Choose one to continue</p>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2" role="radiogroup" aria-label="Workflow" aria-required="true" aria-describedby="workflow-help">
        {WORKFLOWS.map((option, index) => {
          const selected = value === option.id
          return (
            <button
              key={option.id} type="button" role="radio" aria-checked={selected}
              aria-labelledby={`workflow-${option.id}-title`} aria-describedby={`workflow-${option.id}-level workflow-${option.id}-description`}
              tabIndex={selected || (value === null && index === 0) ? 0 : -1}
              onKeyDown={onRadioKeyDown} disabled={disabled} onClick={() => {
                setSelectionRevision((revision) => revision + 1)
                onChange(option.id)
              }}
              className={cn('glass-tile glass-tile-hover flex flex-col rounded-xl p-3 text-left disabled:cursor-not-allowed disabled:opacity-50', selected && 'glass-selected')}
            >
              <span className="flex w-full items-center justify-between gap-2">
                <span id={`workflow-${option.id}-title`} className="text-sm font-semibold text-ink">{option.title}</span>
                <span aria-hidden="true" className={cn('flex h-4 w-4 items-center justify-center rounded-full border', selected ? 'border-accent bg-accent text-accent-ink' : 'border-ink-subtle')}>
                  {selected && <Check className="h-3 w-3" strokeWidth={3} />}
                </span>
              </span>
              <span id={`workflow-${option.id}-level`} className="mt-2 inline-flex self-start rounded-full bg-white/[0.06] px-2 py-0.5 text-2xs font-medium text-ink-muted">{option.level}</span>
              <WorkflowIllustration key={`${option.id}-${selected ? selectionRevision : 0}`} workflow={option.id} selected={selected} animate={selected && selectionRevision > 0} />
              <span id={`workflow-${option.id}-description`} className="block flex-1">
                <span className="block text-xs font-medium text-ink">{option.outcome}</span>
                <span className="mt-1 block text-xs leading-relaxed text-ink-muted">{option.description}</span>
              </span>
              <span aria-hidden="true" className="mt-3 block text-2xs text-ink-muted">{option.steps}</span>
            </button>
          )
        })}
      </div>
    </section>
  )
}

/** A little clip conveyor for Automatic; a hands-on timeline for Review. */
function WorkflowIllustration({ workflow, selected, animate }: {
  workflow: NonNullable<ClipDraft['workflow']>; selected: boolean; animate: boolean
}): React.JSX.Element {
  return (
    <svg viewBox="0 0 240 80" fill="none" aria-hidden="true" focusable="false" className={cn('workflow-illustration my-2 h-16 w-full', animate && 'workflow-illustration-selected', selected ? 'text-accent' : 'text-ink-subtle')}>
      {workflow === 'automatic' ? <>
        <g className="workflow-source">
          <rect x="14" y="22" width="54" height="36" rx="7" fill="currentColor" fillOpacity=".08" stroke="currentColor" strokeWidth="1.5" />
          <path d="m36 32 13 8-13 8Z" fill="currentColor" />
        </g>
        <path d="M77 40h25m-5-5 5 5-5 5m36 0h25m-5-5 5 5-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path className="workflow-spark" d="m120 24 4 12 12 4-12 4-4 12-4-12-12-4 12-4Z" fill="currentColor" fillOpacity=".18" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
        <path className="workflow-twinkles" d="M137 15v8m-4-4h8M97 59v6m-3-3h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <g className="workflow-clips">
          <rect x="184" y="15" width="29" height="46" rx="5" transform="rotate(12 184 15)" fill="currentColor" fillOpacity=".08" stroke="currentColor" strokeWidth="1.5" />
          <rect x="169" y="23" width="29" height="46" rx="5" className="fill-raised" stroke="currentColor" strokeWidth="1.5" />
          <path d="m180 34 9 6-9 6Z" fill="currentColor" />
          <path d="M176 54h15m-12 5h9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </g>
        <g className="workflow-done">
          <circle cx="211" cy="58" r="10" className="fill-raised" stroke="currentColor" strokeWidth="1.5" />
          <path d="m206 58 3 3 6-6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </g>
      </> : <>
        <rect x="29" y="9" width="180" height="62" rx="8" fill="currentColor" fillOpacity=".06" stroke="currentColor" strokeWidth="1.5" />
        <rect x="40" y="17" width="46" height="28" rx="4" fill="currentColor" fillOpacity=".12" stroke="currentColor" />
        <path d="m59 24 10 7-10 7Z" fill="currentColor" />
        <path d="M100 23h34m-34 8h23m-23 8h40" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity=".6" />
        <path d="M41 57h155" stroke="currentColor" strokeOpacity=".3" strokeWidth="9" strokeLinecap="round" />
        <g className="workflow-trim">
          <rect x="72" y="51" width="88" height="12" rx="3" fill="currentColor" fillOpacity=".25" stroke="currentColor" strokeWidth="1.5" />
          <path d="M77 54v6m78-6v6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </g>
        <path className="workflow-playhead" d="M108 48v18" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <g className="workflow-cursor">
          <path d="m174 25 5 21 4-7 8-2Z" className="fill-ink" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
          <path className="workflow-click" d="M169 18v-4m-7 10h-4m6-6-3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </g>
      </>}
    </svg>
  )
}
