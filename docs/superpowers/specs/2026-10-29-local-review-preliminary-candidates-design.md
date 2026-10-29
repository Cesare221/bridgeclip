# Local review candidates and render validation

## Context

Local Automatic currently selects only candidates Ollama marks as narratively
closed. On `video-teste-02.mp4`, a replay of the saved local transcript produced
36 valid candidates and an eight-candidate shortlist. All eight structured
evaluations were valid, their scores were 70–85, and all had
`narrative_closed=false`. Score is a ranking field, not an approval threshold.
The original run did not persist per-candidate results, so its exact scores and
justifications cannot be recovered.

## Design

- Keep the default local planner response unchanged: Automatic receives only
  structurally valid, closed-narrative, non-overlapping selections. Do not lower
  the editorial criteria or force a clip.
- In Review & edit only, expose the structurally valid evaluated shortlist,
  including candidates Ollama did not approve. Carry the decision, score, and
  bounded rationale into the local editor project so each candidate can be
  clearly identified as preliminary.
- Review candidates remain unexportable while refining. The existing explicit
  “Mark ready” action is the human approval; export remains gated by the
  server-side `ready` state. Validate local candidate duration and source bounds
  and prevent a local export from overlapping an already exported local clip.
- Persist bounded local diagnostics per candidate (ID, source interval,
  duration, structural validity, score, narrative decision, and fixed rejection
  category). Do not persist transcript text or rationale in diagnostic logs.
- Add a Windows smoke script that creates a deterministic synthetic source with
  FFmpeg, renders a known 30–60 second cut through `RenderingService` with audio
  and captions, validates streams/dimensions/duration with `ffprobe`, and keeps
  the MP4 and a caption inspection frame in a caller-selected output folder.
  The smoke script must not invoke a planner or external service.

## Data flow

The local planner evaluates one validated shortlist and retains both its
approved non-overlapping selection and the individual evaluated candidates.
The pipeline uses only the approved selection for Automatic. Review & edit uses
the evaluated shortlist to create a local editor project; candidate approval
and export remain explicit editor actions.

## Validation

- Unit tests cover the eight-candidate rejection case, safe diagnostic shape,
  unchanged Automatic filtering, review-only preliminary visibility, and
  approval/export gates including bounds, duration, and overlap.
- A local render smoke run validates the actual renderer and FFmpeg independently
  of Ollama and Faster-Whisper.
- Re-run the real video in local Review & edit and confirm a preliminary editor
  project is created without exporting a rejected clip. An actual human
  approval is not simulated or claimed by the unattended E2E.
