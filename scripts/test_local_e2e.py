#!/usr/bin/env python3
"""Opt-in Windows smoke test for local transcription, planning, and MP4 rendering."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from threading import Thread
from typing import Any


LOCAL_CREDENTIALS = (
    "OPENROUTER_API_KEY",
    "ZERNIO_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)
PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="Local source video to process")
    parser.add_argument("--output", type=Path, help="Output folder (defaults to a unique folder under TEMP)")
    args = parser.parse_args()

    video_path = args.video.expanduser().resolve(strict=True)
    if not video_path.is_file():
        parser.error("--video must be a local file")

    repository = Path(__file__).resolve().parent.parent
    engine_path = repository / "engine"
    python_path = engine_path / ".venv" / "Scripts" / "python.exe"
    bridge_path = repository / "bridge" / "bridge_runner.py"
    if not python_path.is_file():
        print(f"Engine Python environment not found: {python_path}", file=sys.stderr)
        return 2
    if not bridge_path.is_file():
        print("Bridge runner is missing from this checkout.", file=sys.stderr)
        return 2

    run_id = f"local-e2e-{uuid.uuid4().hex}"
    output_root = (args.output or Path(tempfile.gettempdir()) / f"bridgeclip-{run_id}").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root = output_root / ".work"
    work_root.mkdir(mode=0o700, exist_ok=True)

    config = {
        "contract_version": 4,
        "job_id": run_id,
        "video_url": str(video_path),
        "workflow": "automatic",
        "clipping_mode": "local",
        "max_clips": 2,
        "auto_clip_count": True,
        "duration_ranges": ["short"],
        "aspect_ratio": "9:16",
        "layout_style": "auto",
        "layout_vision_enabled": False,
        "pacing": "natural",
        "video_speed": 1.0,
        "include_captions": True,
        "caption_preset": "pop",
        "include_title": True,
        "start_time_seconds": None,
        "end_time_seconds": None,
        "banner_platform": None,
        "banner_channel_url": None,
        "output_dir": str(output_root),
    }

    environment = os.environ.copy()
    for name in (*LOCAL_CREDENTIALS, *PROXY_VARIABLES):
        environment.pop(name, None)
    environment.update({
        "CLIPPING_MODE": "local",
        "LOCAL_MODE": "true",
        "LOCAL_OUTPUT_DIR": str(output_root),
        "BRIDGECLIP_WORK_ROOT": str(work_root),
        "PYTHONPATH": str(engine_path),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "YTDLP_PROXIES": "",
        "YTDLP_PROXY": "",
    })

    stderr_parts: list[str] = []
    try:
        process = subprocess.Popen(
            [str(python_path), str(bridge_path)],
            cwd=engine_path,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        print(f"Could not start the local BridgeClip engine: {error}", file=sys.stderr)
        return 2

    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stderr_reader = Thread(target=lambda: stderr_parts.append(process.stderr.read()), daemon=True)
    stderr_reader.start()
    process.stdin.write(json.dumps(config) + "\n")
    process.stdin.close()

    result: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    for line in process.stdout:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "progress":
            step = event.get("step")
            status = event.get("status")
            if isinstance(step, str) and isinstance(status, str):
                print(f"[{status}] {step}")
        elif event.get("type") == "result":
            result = event
        elif event.get("type") == "error":
            failure = event

    exit_code = process.wait()
    stderr_reader.join(timeout=5)
    if exit_code != 0 or result is None:
        message = failure.get("message") if failure else "The local job did not return a result."
        hint = failure.get("hint") if failure else None
        print(message, file=sys.stderr)
        if hint:
            print(hint, file=sys.stderr)
        if failure:
            stage = failure.get("stage")
            code = failure.get("code")
            if stage in {"setup", "download", "transcription", "planning", "rendering", "saving", "uploading"}:
                details = [f"stage={stage}"]
                if isinstance(code, str) and len(code) <= 64 and re.fullmatch(r"[a-z]+(?:[._][a-z]+)*", code):
                    details.append(f"code={code}")
                print(f"Diagnostic: {' '.join(details)}", file=sys.stderr)
        return 1

    output = result.get("output")
    clips = output.get("clips") if isinstance(output, dict) else None
    if not isinstance(clips, list) or not clips:
        print("The local run completed, but selected no complete 30–60 second clips.")
        print("Try a video with more speech and complete ideas; no clip is forced.")
        return 1

    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        print("FFmpeg and FFprobe must be available on PATH to validate the rendered MP4.", file=sys.stderr)
        return 2

    for index, artifact in enumerate(clips, start=1):
        clip_path = _local_artifact_path(artifact.get("s3_url"))
        if clip_path is None or not clip_path.is_file() or clip_path.stat().st_size == 0:
            print(f"Rendered clip {index} is missing or empty.", file=sys.stderr)
            return 1
        probe = subprocess.run(
            [
                ffprobe, "-v", "error", "-show_entries",
                "stream=codec_type,width,height:format=duration", "-of", "json", str(clip_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            print(f"FFprobe could not read rendered clip {index}.", file=sys.stderr)
            return 1
        try:
            media = json.loads(probe.stdout)
            streams = media["streams"]
            duration = float(media["format"]["duration"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            print(f"FFprobe returned invalid media details for clip {index}.", file=sys.stderr)
            return 1
        video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
        audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
        if (
            video_stream is None
            or audio_stream is None
            or video_stream.get("width") != 1080
            or video_stream.get("height") != 1920
            or not 30 <= duration <= 60
        ):
            print(
                f"Clip {index} must have audio, 1080×1920 video, and a duration between 30 and 60 seconds.",
                file=sys.stderr,
            )
            return 1

        frame_path = output_root / f"clip-{index:02d}-caption-check.png"
        frame = subprocess.run(
            [
                ffmpeg, "-v", "error", "-y", "-ss", "1", "-i", str(clip_path),
                "-frames:v", "1", str(frame_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if frame.returncode != 0 or not frame_path.is_file() or frame_path.stat().st_size == 0:
            print(f"Could not extract a caption-inspection frame for clip {index}.", file=sys.stderr)
            return 1
        print(f"Validated {clip_path} ({duration:.1f}s, 1080x1920, video+audio)")
        print(f"Inspect burned-in captions in {frame_path}")

    print(f"Local E2E passed. Outputs remain in {output_root}")
    print("Manually inspect each caption frame and the MP4 for caption legibility and narrative closure.")
    return 0


def _local_artifact_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value.startswith("file://"):
        return None
    path = value[len("file://"):]
    if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


if __name__ == "__main__":
    raise SystemExit(main())
