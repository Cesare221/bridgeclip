"""Offline Faster-Whisper adapter for the engine transcription contract."""

import asyncio
import logging
import math
import os
import threading
from dataclasses import dataclass
from numbers import Real
from typing import Callable, ClassVar, Iterable, Optional, Protocol

from clip_engine.services.transcription_service import (
    TranscriptionError,
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
    normalize_keyterms,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "base"
MODEL_DEVICE = "cpu"
MODEL_COMPUTE_TYPE = "int8"
MODEL_TIMESTAMP_TOLERANCE_SECONDS = 0.05


class _WordLike(Protocol):
    start: float
    end: float
    word: str


class _SegmentLike(Protocol):
    text: str
    words: Optional[list[_WordLike]]


class _InfoLike(Protocol):
    duration: float
    language: Optional[str]


class _WhisperModelLike(Protocol):
    def transcribe(
        self,
        audio: str,
        *,
        language: Optional[str],
        word_timestamps: bool,
        hotwords: Optional[str],
    ) -> tuple[Iterable[_SegmentLike], _InfoLike]: ...


@dataclass
class _ModelCacheEntry:
    model: _WhisperModelLike
    inference_lock: threading.Lock


class FasterWhisperTranscriptionService:
    """Transcribe local audio without credentials or network model downloads."""

    _model_cache: ClassVar[dict[str, _ModelCacheEntry]] = {}
    _model_cache_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self.progress_callback: Optional[Callable[[str], None]] = None

    def _progress(self, message: str) -> None:
        callback = self.progress_callback
        if callback:
            try:
                callback(message)
            except Exception:
                logger.warning("Could not report local transcription progress")

    async def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        keyterms: Optional[list[str]] = None,
        timeline_offset_seconds: float = 0.0,
        source_duration_seconds: Optional[float] = None,
    ) -> TranscriptionResult:
        """Return word-level timestamps on the source timeline."""
        if not os.path.isfile(audio_path):
            raise TranscriptionError("Audio file not found", reason="audio_missing")
        if (
            type(timeline_offset_seconds) not in (int, float)
            or not math.isfinite(timeline_offset_seconds)
            or timeline_offset_seconds < 0
        ):
            raise TranscriptionError("Invalid transcription timeline offset", reason="invalid_timeline_offset")
        if source_duration_seconds is not None and (
            not _is_finite_number(source_duration_seconds) or source_duration_seconds <= 0
        ):
            raise TranscriptionError(
                "Transcription source duration is invalid",
                reason="invalid_source_duration",
            )

        normalized_language = None if language in (None, "", "auto") else language
        hotwords = ", ".join(normalize_keyterms(keyterms)) or None
        self._progress("Transcribing audio locally...")
        result = await asyncio.to_thread(
            self._transcribe_sync,
            audio_path,
            normalized_language,
            hotwords,
            float(timeline_offset_seconds),
            source_duration_seconds,
        )
        self._progress("Local transcription complete.")
        return result

    @classmethod
    def _get_model_entry(cls) -> _ModelCacheEntry:
        with cls._model_cache_lock:
            entry = cls._model_cache.get(MODEL_NAME)
            if entry is None:
                entry = _ModelCacheEntry(cls._load_model(), threading.Lock())
                cls._model_cache[MODEL_NAME] = entry
            return entry

    @staticmethod
    def _load_model() -> _WhisperModelLike:
        from faster_whisper import WhisperModel
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            return WhisperModel(
                MODEL_NAME,
                device=MODEL_DEVICE,
                compute_type=MODEL_COMPUTE_TYPE,
                local_files_only=True,
            )
        except (FileNotFoundError, LocalEntryNotFoundError) as error:
            raise TranscriptionError(
                "The Faster-Whisper base model is not cached locally. "
                "Install it once before using Local mode by running: "
                "python -c \"from faster_whisper import WhisperModel; "
                "WhisperModel('base', device='cpu', compute_type='int8')\"",
                reason="model_not_installed",
            ) from error

    def _transcribe_sync(
        self,
        audio_path: str,
        language: Optional[str],
        hotwords: Optional[str],
        timeline_offset_seconds: float,
        source_duration_seconds: Optional[float],
    ) -> TranscriptionResult:
        entry = self._get_model_entry()
        with entry.inference_lock:
            raw_segments, info = entry.model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                hotwords=hotwords,
            )
            segments = list(raw_segments)
        return self._convert_result(
            segments,
            info,
            timeline_offset_seconds,
            source_duration_seconds,
        )

    @staticmethod
    def _convert_result(
        raw_segments: list[_SegmentLike],
        info: _InfoLike,
        timeline_offset_seconds: float,
        source_duration_seconds: Optional[float] = None,
    ) -> TranscriptionResult:
        duration = info.duration
        if not _is_finite_number(duration) or duration <= 0:
            raise TranscriptionError("Transcription returned an invalid audio duration", reason="invalid_response")
        if source_duration_seconds is not None and (
            not _is_finite_number(source_duration_seconds) or source_duration_seconds <= 0
        ):
            raise TranscriptionError(
                "Transcription source duration is invalid",
                reason="invalid_source_duration",
            )

        converted_segments: list[TranscriptSegment] = []
        transcript_text: list[str] = []
        previous_word_start = -1.0
        for raw_segment in raw_segments:
            segment_text = " ".join(raw_segment.text.split())
            if segment_text:
                transcript_text.append(segment_text)
            raw_words = raw_segment.words or []
            if segment_text and not raw_words:
                raise TranscriptionError(
                    "Transcription response did not include word timestamps",
                    reason="missing_word_timestamps",
                )

            words: list[TranscriptWord] = []
            for raw_word in raw_words:
                word_text = " ".join(raw_word.word.split())
                start = raw_word.start
                end = raw_word.end
                if (
                    not word_text
                    or not _is_finite_number(start)
                    or not _is_finite_number(end)
                    or start < 0
                    or end <= start
                    or start < previous_word_start
                    or end > duration + MODEL_TIMESTAMP_TOLERANCE_SECONDS
                    or (
                        source_duration_seconds is not None
                        and end + timeline_offset_seconds
                        > source_duration_seconds + MODEL_TIMESTAMP_TOLERANCE_SECONDS
                    )
                ):
                    raise TranscriptionError(
                        "Transcription returned invalid word timestamps",
                        reason="invalid_response",
                    )
                previous_word_start = start
                offset_ms = round(timeline_offset_seconds * 1000)
                words.append(
                    TranscriptWord(
                        word=word_text,
                        start_time_ms=round(start * 1000) + offset_ms,
                        end_time_ms=round(end * 1000) + offset_ms,
                    )
                )

            if words:
                converted_segments.append(
                    TranscriptSegment(
                        start_time_ms=words[0].start_time_ms,
                        end_time_ms=words[-1].end_time_ms,
                        text=" ".join(word.word for word in words),
                        words=words,
                    )
                )

        return TranscriptionResult(
            segments=converted_segments,
            full_text=" ".join(transcript_text) or " ".join(
                segment.text for segment in converted_segments
            ),
            language=info.language if isinstance(info.language, str) else None,
            duration_seconds=float(duration),
            provider="local",
            model="faster-whisper/base",
            api_costs=None,
        )


def _is_finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
