#!/usr/bin/env python3
"""Render and validate a deterministic local 9:16 captioned clip with FFmpeg."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ENGINE_PATH = Path(__file__).resolve().parents[1] / "engine"
if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))

from clip_engine.config import get_caption_preset
from clip_engine.services.layout_analyzer import ClipLayoutPlan, LayoutType, ShotLayout
from clip_engine.services.rendering_service import RenderRequest, RenderingService
from clip_engine.services.transcription_service import TranscriptSegment, TranscriptWord

SOURCE_START_MS = 5_000
SOURCE_END_MS = 40_000
EXPECTED_DURATION_SECONDS = 35


def _run(command: list[str], failure_message: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(failure_message)
    return result


def _transcript() -> list[TranscriptSegment]:
    segments = []
    for index in range(7):
        start = SOURCE_START_MS + index * 5_000 + 500
        words = [
            TranscriptWord(text, start + word_index * 500, start + (word_index + 1) * 500)
            for word_index, text in enumerate(
                ("Local", "caption", "render", f"check{index + 1}.")
            )
        ]
        segments.append(
            TranscriptSegment(
                start_time_ms=words[0].start_time_ms,
                end_time_ms=words[-1].end_time_ms,
                text=" ".join(word.word for word in words),
                words=words,
            )
        )
    return segments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Persistent folder for the rendered MP4 and inspection frame")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        print("FFmpeg and FFprobe must be available on PATH.", file=sys.stderr)
        return 2

    run_id = uuid.uuid4().hex[:12]
    output_path = output / f"local-render-smoke-{run_id}.mp4"
    frame_path = output / f"local-render-caption-check-{run_id}.png"
    os.environ["CLIPPING_MODE"] = "local"
    os.environ["LOCAL_MODE"] = "true"
    os.environ["LOCAL_OUTPUT_DIR"] = str(output)
    os.environ["OPENROUTER_API_KEY"] = ""

    try:
        with tempfile.TemporaryDirectory(prefix=".local-render-smoke-", dir=output) as work_dir:
            source_path = Path(work_dir) / "synthetic-source.mp4"
            _run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=640x360:rate=30:duration=45",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000:duration=45",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "mpeg4",
                    "-q:v",
                    "3",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(source_path),
                ],
                "FFmpeg could not create the synthetic render-smoke source.",
            )

            probe = _run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_type,width,height:format=duration",
                    "-of",
                    "json",
                    str(source_path),
                ],
                "FFprobe could not inspect the synthetic render-smoke source.",
            )
            source_media = json.loads(probe.stdout)
            source_video = next(
                stream
                for stream in source_media["streams"]
                if stream.get("codec_type") == "video"
            )
            renderer = RenderingService()
            layout = ClipLayoutPlan(
                [
                    ShotLayout(
                        0,
                        SOURCE_END_MS - SOURCE_START_MS,
                        LayoutType.SCREEN,
                        source="deterministic-smoke",
                    )
                ],
                int(source_video["width"]),
                int(source_video["height"]),
            )
            result = asyncio.run(
                renderer.render_clip(
                    RenderRequest(
                        video_path=str(source_path),
                        output_path=str(output_path),
                        start_time_ms=SOURCE_START_MS,
                        end_time_ms=SOURCE_END_MS,
                        source_width=int(source_video["width"]),
                        source_height=int(source_video["height"]),
                        transcript_segments=_transcript(),
                        include_captions=True,
                        caption_style=get_caption_preset("pop"),
                        apply_padding=False,
                        aspect_ratio="9:16",
                        layout_style="fit",
                        pacing="natural",
                        manual_ranges_ms=[(SOURCE_START_MS, SOURCE_END_MS)],
                        manual_plan=layout,
                    )
                )
            )
            if result.duration_ms != EXPECTED_DURATION_SECONDS * 1000:
                raise RuntimeError("RenderingService returned an unexpected clip duration.")

        probe = _run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,width,height:format=duration",
                "-of",
                "json",
                str(output_path),
            ],
            "FFprobe could not inspect the rendered smoke-test MP4.",
        )
        media = json.loads(probe.stdout)
        video_stream = next(
            (stream for stream in media["streams"] if stream.get("codec_type") == "video"),
            None,
        )
        audio_stream = next(
            (stream for stream in media["streams"] if stream.get("codec_type") == "audio"),
            None,
        )
        duration = float(media["format"]["duration"])
        if (
            video_stream is None
            or audio_stream is None
            or video_stream.get("width") != 1080
            or video_stream.get("height") != 1920
            or not 30 <= duration <= 60
        ):
            raise RuntimeError("Rendered MP4 failed resolution, duration, or audio validation.")

        _run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-ss",
                "2.5",
                "-i",
                str(output_path),
                "-frames:v",
                "1",
                str(frame_path),
            ],
            "FFmpeg could not extract the caption-inspection frame.",
        )
        if not frame_path.is_file() or frame_path.stat().st_size == 0:
            raise RuntimeError("The caption-inspection frame is missing or empty.")
    except (OSError, KeyError, StopIteration, TypeError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(
        f"Render smoke passed: {output_path} "
        f"({duration:.1f}s, 1080x1920, video+audio, captions enabled)"
    )
    print(f"Inspect burned-in captions: {frame_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
