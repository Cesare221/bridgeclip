# Timed editor caption preview

## Context

The editor project currently carries transcript segment text and segment
timestamps, while the saved local transcript also contains Faster-Whisper word
timestamps. `EditorCaptionPreview` currently draws a static positioning guide.
The export path already groups timestamped words, applies caption edits, and
generates ASS captions; it is out of scope to change that path.

## Design

Extend each editor-project transcript segment with an optional bounded list of
words containing text and source-time start/end milliseconds. Populate it from
the existing `TranscriptSegment.words` when creating a project. Validate the
optional values when parsing the project. Its absence remains valid for saved
projects that predate this change.

Keep the existing editor overlay and its drag/keyboard position controls. In
the Captions and Transcript tabs, select the caption group for the player’s
current source time, using the existing preset’s maximum words, sentence and
comma boundaries, and the renderer’s 700 ms linger. Highlight the active word
according to the selected preset. Reuse preset styling and the existing
`captionAnchor`; natural browser wrapping is an approximation of ASS layout.

Only display captions when they are enabled, the playhead lies within one of
the candidate’s retained ranges, and it is not in a caption-suppression range.
Apply per-candidate transcript corrections. When a correction preserves word
count, retain original word timings; otherwise distribute words over the
spoken interval, matching the existing export correction behavior. Empty
corrections hide the corresponding caption. For old projects without word
timings, display the corrected segment text over the segment interval without
word-by-word highlighting.

Do not add a transcript IPC or external dependency. Do not modify the ASS
generator, FFmpeg graph, or export behavior.

## Validation

- Unit-test preview timing and grouping: active word, preset group limits,
  punctuation boundaries, linger, cut bounds, suppression, disabled captions,
  text corrections, and legacy segment-only data.
- Test rendered preview content and preset presentation, including selected
  text, active-word styling, position, and caption-off behavior.
- Run focused renderer/shared tests, typecheck, and relevant caption tests.
  Avoid a heavy E2E run; do not start F2.
