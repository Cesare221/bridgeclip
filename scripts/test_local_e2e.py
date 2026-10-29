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
    parser.add_argument(
        "--workflow",
        choices=("automatic", "review"),
        default="automatic",
        help="Run automatic selection or prepare candidates for Review & edit",
    )
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
        "workflow": args.workflow,
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
    if args.workflow == "review":
        if not isinstance(output, dict) or output.get("editor_project") is not True:
            print("The local review run did not create an editor project.", file=sys.stderr)
            return 1
        project_path = output_root / run_id / "editor-project.json"
        diagnostics_path = output_root / run_id / "selection_diagnostics.json"
        try:
            project = json.loads(project_path.read_text(encoding="utf-8"))
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print("The local review project or selection diagnostics are missing or invalid.", file=sys.stderr)
            return 1
        candidates = project.get("candidates") if isinstance(project, dict) else None
        decisions = diagnostics.get("candidates") if isinstance(diagnostics, dict) else None
        source_duration = output.get("source_video_duration_seconds")
        if (
            not isinstance(candidates, list)
            or not candidates
            or not isinstance(decisions, list)
            or not isinstance(source_duration, (int, float))
            or not source_duration > 0
        ):
            print("The local review project contains no validated candidates.", file=sys.stderr)
            return 1
        approved = [
            item for item in decisions
            if isinstance(item, dict) and item.get("decision") == "selected"
        ]
        for item in decisions:
            if not isinstance(item, dict) or not isinstance(item.get("id"), int):
                print("Local selection diagnostics contain an invalid candidate.", file=sys.stderr)
                return 1
            if (
                set(item)
                != {
                    "id",
                    "start_time_ms",
                    "end_time_ms",
                    "duration_seconds",
                    "structural_valid",
                    "relevance_score",
                    "narrative_closed",
                    "decision",
                }
                or item.get("structural_valid") is not True
                or type(item.get("narrative_closed")) is not bool
                or type(item.get("relevance_score")) is not int
                or item.get("decision") not in {
                    "selected",
                    "narrative_not_closed",
                    "overlaps_selected_candidate",
                    "selection_limit",
                }
            ):
                print("Local selection diagnostics contain an unsafe or invalid record.", file=sys.stderr)
                return 1
            print(
                f"[candidate] id={item['id']} decision={item['decision']} "
                f"narrative_closed={str(item['narrative_closed']).lower()} "
                f"score={item['relevance_score']}"
            )
        for index, left in enumerate(approved):
            for right in approved[index + 1:]:
                if left["start_time_ms"] < right["end_time_ms"] and right["start_time_ms"] < left["end_time_ms"]:
                    print("Approved local candidate intervals overlap.", file=sys.stderr)
                    return 1
        preliminary_count = 0
        for item in candidates:
            local_selection = item.get("local_selection")
            ranges = item.get("ranges")
            if (
                not isinstance(local_selection, dict)
                or not isinstance(ranges, list)
                or not ranges
                or item.get("status") != "refining"
                or item.get("exports") != []
            ):
                print("The local review project has an invalid candidate.", file=sys.stderr)
                return 1
            duration_ms = 0
            for interval in ranges:
                if (
                    not isinstance(interval, list)
                    or len(interval) != 2
                    or type(interval[0]) is not int
                    or type(interval[1]) is not int
                ):
                    print("The local review project has an invalid time range.", file=sys.stderr)
                    return 1
                start, end = interval
                duration_ms += end - start
                if start < 0 or end > source_duration * 1000 or end <= start:
                    print("A local review candidate is outside source bounds.", file=sys.stderr)
                    return 1
            if not 30_000 <= duration_ms <= 60_000:
                print("A local review candidate is outside the 30–60 second duration range.", file=sys.stderr)
                return 1
            if local_selection.get("decision") != "selected":
                preliminary_count += 1
        if not preliminary_count:
            print("The review project contains no preliminary candidates.", file=sys.stderr)
            return 1
        if output.get("clips") != []:
            print("Review preparation unexpectedly exported a clip.", file=sys.stderr)
            return 1
        print(
            f"Local Review E2E passed. {len(candidates)} candidates are available; "
            f"{preliminary_count} require human review. No clip was exported."
        )
        print(f"Review project: {project_path}")
        print(f"Safe candidate diagnostics: {diagnostics_path}")
        return 0

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
