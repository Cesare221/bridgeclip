# Local AI mode

[Create clips](usage.md#create-your-first-clips) · [Transcription](transcription.md) · [Editor](editor.md)

Local mode runs speech transcription with Faster-Whisper Base and candidate selection with Ollama's `qwen3:1.7b`. It does not require an OpenRouter key, send media or transcripts to a paid API, upload artifacts, or publish clips. During a job, the only permitted network destination is the local Ollama service at `127.0.0.1:11434`.

## One-time setup on Windows

The BridgeClip engine environment must have its pinned dependencies installed. From the repository root, initialize the local models:

```powershell
ollama pull qwen3:1.7b
ollama list
& '.\engine\.venv\Scripts\python.exe' -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"
```

The Faster-Whisper command downloads Base once if it is not cached. Local jobs use `local_files_only=True` and fail with setup guidance rather than downloading a model during processing. Ollama must be running before a Local job; on Windows, the Ollama desktop app normally starts its service. Confirm `qwen3:1.7b` appears in `ollama list`.

For a source checkout, start the desktop app with:

```powershell
npm run dev
```

FFmpeg and FFprobe must be available to BridgeClip, and the FFmpeg build must include the ASS subtitle filter when captions are enabled.

## Create and export

1. In **Create**, select **Local** under **Clipping mode**, above the wizard steps.
2. Select **Automatic** or **Review & edit**.
3. Use the native picker to select a video on this computer. Remote links are unavailable in Local mode.
4. Choose caption and duration options, then start the job. Local output is always vertical 9:16; no more than two non-overlapping candidates are selected. The default length is 30–60 seconds. The result can contain zero, one, or two clips.
5. Automatic exports are saved under the configured output folder. Review & edit opens the local candidates, their editorial score and selection rationale; adjust cut ranges, framing, and captions in the existing editor, then mark the clip ready and export.

An empty result is a completed job, not a render failure. It means the transcript did not contain a complete candidate within the requested bounds; the engine does not force a cut. Try a video with more speech, choose a different duration range, or adjust the source range.

Local mode disables Jev review, paid framing vision, source web research, and automatic uploads. It does not change the behavior of Quality, Economy, or Advanced jobs.

## Optional Windows render smoke test

The opt-in harness runs the real bridge with credentials and proxy variables removed, then checks the rendered MP4 with FFprobe for non-empty video and audio streams, 1080×1920 dimensions, and a valid duration. It extracts a frame for manual caption inspection; narrative closure and caption legibility still require watching the MP4.

```powershell
.\scripts\test-local-e2e.ps1 -VideoPath 'C:\path\to\local\video.mp4'
```

The harness requires `engine\.venv`, a cached Faster-Whisper Base model, a running Ollama service with `qwen3:1.7b`, and `ffmpeg`/`ffprobe` on `PATH`. Outputs and caption-check frames remain in a unique folder under `%TEMP%`. Nothing is uploaded or committed. If no suitable candidate exists, the smoke test stops without manufacturing an MP4.
