import { sceneAt, type CandidateEdit, type EditorProject, type EditorRange, type EditorTranscriptWord } from '../../shared/clip-editor'

export interface CaptionPreviewWord {
  text: string
  state: 'past' | 'active' | 'future'
}
export interface CaptionPreviewLine {
  words: CaptionPreviewWord[]
  /** Older projects have only segment timing, so no word-level highlight is available. */
  legacy: boolean
}

interface TimedWord extends EditorTranscriptWord {
  output_start_ms: number
  output_end_ms: number
}
interface LegacyCaption {
  start_ms: number; end_ms: number; text: string
}
export interface CaptionPreviewTimeline {
  enabled: boolean
  ranges: EditorRange[]
  suppressed: EditorRange[]
  groups: TimedWord[][]
  legacy: LegacyCaption[]
}

const CAPTION_LINGER_MS = 700
const SENTENCE_END = /[.!?]$/

function displayWord(value: string): string {
  return value.replace(/[{}\\]/g, '').trim().split(/\s+/).filter(Boolean).join(' ')
}

function candidateTime(time: number, ranges: EditorRange[]): { output_ms: number } | null {
  let output_ms = 0
  for (const [start, end] of ranges) {
    if (time >= start && time < end) return { output_ms: output_ms + time - start }
    output_ms += end - start
  }
  return null
}

function correctedWords(project: EditorProject, candidate: CandidateEdit): { timed: TimedWord[]; legacy: LegacyCaption[] } {
  const overrides = new Map((candidate.caption_edits ?? []).map(edit => [edit.segment, edit.text]))
  const timed: TimedWord[] = []
  const legacy: LegacyCaption[] = []
  const rangeOffsets = candidate.ranges.map((range, i) =>
    candidate.ranges.slice(0, i).reduce((sum, [a, b]) => sum + b - a, 0))
  for (const [index, segment] of project.transcript.entries()) {
    const hasEdit = overrides.has(index), editText = hasEdit ? overrides.get(index)!.trim() : segment.text
    const tokens = hasEdit ? editText.split(/\s+/).filter(Boolean) : null
    if (hasEdit && !tokens?.length) continue
    const sourceWords = segment.words ?? []
    let words: EditorTranscriptWord[]
    if (!sourceWords.length) {
      if (editText.trim()) legacy.push({ start_ms: segment.start_ms, end_ms: segment.end_ms, text: editText })
      continue
    }
    if (!hasEdit) words = sourceWords
    else if (tokens!.length === sourceWords.length) {
      words = sourceWords.map((word, i) => ({ ...word, text: tokens![i] }))
    } else {
      const start = sourceWords[0].start_ms, end = sourceWords[sourceWords.length - 1].end_ms
      const span = Math.max(1, end - start)
      words = tokens!.map((text, i) => ({
        text,
        start_ms: Math.round(start + i * span / tokens!.length),
        end_ms: Math.round(start + (i + 1) * span / tokens!.length)
      }))
    }
    for (const word of words) {
      if (!displayWord(word.text)) continue
      for (const [rangeIndex, [rangeStart, rangeEnd]] of candidate.ranges.entries()) {
        const start = Math.max(word.start_ms, rangeStart), end = Math.min(word.end_ms, rangeEnd)
        if (end <= start) continue
        timed.push({ ...word, output_start_ms: rangeOffsets[rangeIndex] + start - rangeStart,
          output_end_ms: rangeOffsets[rangeIndex] + end - rangeStart })
      }
    }
  }
  timed.sort((a, b) => a.output_start_ms - b.output_start_ms || a.output_end_ms - b.output_end_ms)
  return { timed, legacy }
}

function groupedWords(words: TimedWord[], maxWords: number): TimedWord[][] {
  const groups: TimedWord[][] = []
  let current: TimedWord[] = []
  for (const word of words) {
    current.push(word)
    const stripped = displayWord(word.text)
    if (current.length >= maxWords || SENTENCE_END.test(stripped) ||
        (current.length >= 2 && stripped.endsWith(','))) {
      groups.push(current)
      current = []
    }
  }
  if (current.length) groups.push(current)
  return groups
}

/** Build the expensive word groups once; playback only resolves the active group. */
export function captionPreviewTimeline(project: EditorProject, candidate: CandidateEdit, maxWords = 3): CaptionPreviewTimeline {
  const { timed, legacy } = correctedWords(project, candidate)
  return {
    enabled: candidate.captions,
    ranges: candidate.ranges,
    suppressed: candidate.caption_suppression_ranges ?? [],
    groups: groupedWords(timed, Math.max(1, maxWords)),
    legacy
  }
}

/** Resolve the caption line rendered for this source-time playhead. */
export function captionPreviewAt(timeline: CaptionPreviewTimeline, time: number): CaptionPreviewLine | null {
  if (!timeline.enabled || timeline.suppressed.some(([a, b]) => a <= time && time < b)) return null
  const position = candidateTime(time, timeline.ranges)
  if (!position) return null
  const fallback = timeline.legacy.find(line => line.start_ms <= time && time < line.end_ms)
  if (fallback) return { words: [{ text: fallback.text, state: 'active' }], legacy: true }

  const groups = timeline.groups
  for (let groupIndex = 0; groupIndex < groups.length; groupIndex++) {
    const group = groups[groupIndex]
    const nextStart = groups[groupIndex + 1]?.[0].output_start_ms
    const groupEnd = Math.min(group[group.length - 1].output_end_ms + CAPTION_LINGER_MS, nextStart ?? Infinity)
    if (position.output_ms < group[0].output_start_ms || position.output_ms >= groupEnd) continue
    let active = group.length - 1
    for (let i = 0; i < group.length; i++) {
      const nextWord = group[i + 1]?.output_start_ms ?? groupEnd
      if (position.output_ms < nextWord) { active = i; break }
    }
    return { words: group.map((word, i) => ({
      text: displayWord(word.text),
      state: i < active ? 'past' : i === active ? 'active' : 'future'
    })), legacy: false }
  }
  return null
}

/** Manual exports use these same layout anchors (layout_renderer.caption_anchor). */
export function captionAnchor(project: EditorProject, candidate: CandidateEdit, time: number): { y: number; bottom: boolean } {
  if (candidate.caption_y != null) return { y: candidate.caption_y, bottom: false }
  if (project.aspect_ratio === '16:9') return { y: 1 - 100 / 1080, bottom: true }
  const scene = sceneAt(candidate, time)
  if (scene.layout === 'split') return { y: .5, bottom: false }
  if (scene.layout === 'fit') {
    const height = Math.min(1920, 1080 * project.height / project.width)
    const bar = (1920 - height) / 2
    if (bar >= 200) return { y: (1920 - Math.floor(bar * .55)) / 1920, bottom: true }
  }
  return { y: 1340 / 1920, bottom: false }
}
