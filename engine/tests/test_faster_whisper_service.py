import asyncio
import wave
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from clip_engine.services.faster_whisper_service import (
    FasterWhisperTranscriptionService,
    MODEL_COMPUTE_TYPE,
    MODEL_DEVICE,
    MODEL_NAME,
)
from clip_engine.services.transcription_service import TranscriptionError


class FakeModel:
    def __init__(self, segments, info):
        self.segments = segments
        self.info = info
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter(self.segments), self.info


@pytest.fixture(autouse=True)
def clear_model_cache(monkeypatch):
    monkeypatch.setattr(FasterWhisperTranscriptionService, "_model_cache", {})


def make_word(word, start, end):
    return SimpleNamespace(word=word, start=start, end=end)


def make_segment(text, words):
    return SimpleNamespace(text=text, words=words)


def test_transcribe_audio_converts_word_times_and_source_offset(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.touch()
    model = FakeModel(
        [
            make_segment(
                " Hello   there.",
                [make_word(" Hello", 0.125, 0.375), make_word("there.", 0.5, 0.875)],
            )
        ],
        SimpleNamespace(duration=2.0, language="en"),
    )
    monkeypatch.setattr(FasterWhisperTranscriptionService, "_load_model", staticmethod(lambda: model))
    progress = []
    service = FasterWhisperTranscriptionService()
    service.progress_callback = progress.append

    result = asyncio.run(
        service.transcribe_audio(
            str(audio),
            language="auto",
            keyterms=["BridgeClip", "local model"],
            timeline_offset_seconds=4.0,
            source_duration_seconds=5.0,
        )
    )

    assert result.full_text == "Hello there."
    assert result.provider == "local"
    assert result.model == "faster-whisper/base"
    assert result.api_costs is None
    assert result.language == "en"
    assert result.duration_seconds == 2.0
    assert result.segments[0].text == "Hello there."
    assert [
        (word.word, word.start_time_ms, word.end_time_ms)
        for word in result.segments[0].words
    ] == [("Hello", 4125, 4375), ("there.", 4500, 4875)]
    assert result.segments[0].speaker_label is None
    assert model.calls == [
        (
            str(audio),
            {
                "language": None,
                "word_timestamps": True,
                "hotwords": "BridgeClip, local model",
            },
        )
    ]
    assert progress == ["Transcribing audio locally...", "Local transcription complete."]


def test_accepts_numpy_word_timestamps():
    import numpy as np

    result = FasterWhisperTranscriptionService._convert_result(
        [make_segment("Hi.", [make_word("Hi.", np.float64(0.0), np.float64(0.25))])],
        SimpleNamespace(duration=np.float64(1.0), language="en"),
        0.0,
    )

    assert result.segments[0].words[0].start_time_ms == 0
    assert result.segments[0].words[0].end_time_ms == 250


def test_model_is_cached_across_service_instances(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.touch()
    model = FakeModel([], SimpleNamespace(duration=2.0, language="en"))
    loads = []

    def load_model():
        loads.append((MODEL_NAME, MODEL_DEVICE, MODEL_COMPUTE_TYPE))
        return model

    monkeypatch.setattr(FasterWhisperTranscriptionService, "_load_model", staticmethod(load_model))
    for _ in range(2):
        result = asyncio.run(FasterWhisperTranscriptionService().transcribe_audio(str(audio)))
        assert result.segments == []

    assert loads == [("base", "cpu", "int8")]
    assert len(model.calls) == 2


@pytest.mark.parametrize(
    "word",
    [
        make_word("bad", float("nan"), 1.0),
        make_word("bad", 1.0, 0.5),
        make_word("bad", 0.0, 2.1),
        make_word("bad", 1.0, 1.1),
    ],
)
def test_rejects_invalid_or_out_of_order_word_timestamps(word):
    with pytest.raises(TranscriptionError) as error:
        FasterWhisperTranscriptionService._convert_result(
            [make_segment("bad", [make_word("later", 1.5, 1.75), word])],
            SimpleNamespace(duration=2.0, language="en"),
            0.0,
        )
    assert error.value.reason == "invalid_response"


def test_rejects_text_without_word_timestamps():
    with pytest.raises(TranscriptionError) as error:
        FasterWhisperTranscriptionService._convert_result(
            [make_segment("spoken words", None)],
            SimpleNamespace(duration=2.0, language="en"),
            0.0,
        )
    assert error.value.reason == "missing_word_timestamps"


def test_rejects_timestamps_past_source_duration():
    with pytest.raises(TranscriptionError) as error:
        FasterWhisperTranscriptionService._convert_result(
            [make_segment("Hi.", [make_word("Hi.", 0.5, 1.0)])],
            SimpleNamespace(duration=2.0, language="en"),
            3.0,
            source_duration_seconds=3.5,
        )
    assert error.value.reason == "invalid_response"


def test_rejects_invalid_timeline_offset(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.touch()
    with pytest.raises(TranscriptionError) as error:
        asyncio.run(
            FasterWhisperTranscriptionService().transcribe_audio(
                str(audio),
                timeline_offset_seconds=float("inf"),
            )
        )
    assert error.value.reason == "invalid_timeline_offset"


def test_model_loader_uses_local_base_cpu_int8(monkeypatch):
    import faster_whisper

    model = FakeModel([], SimpleNamespace(duration=2.0, language="en"))
    model_factory = Mock(return_value=model)
    monkeypatch.setattr(faster_whisper, "WhisperModel", model_factory)

    assert FasterWhisperTranscriptionService._load_model() is model

    model_factory.assert_called_once_with(
        "base",
        device="cpu",
        compute_type="int8",
        local_files_only=True,
    )


def test_missing_local_model_has_actionable_error(monkeypatch):
    import faster_whisper
    from huggingface_hub.errors import LocalEntryNotFoundError

    model_factory = Mock(side_effect=LocalEntryNotFoundError("not in cache"))
    monkeypatch.setattr(faster_whisper, "WhisperModel", model_factory)

    with pytest.raises(TranscriptionError, match="not cached locally") as error:
        FasterWhisperTranscriptionService._load_model()
    assert error.value.reason == "model_not_installed"


def test_faster_whisper_decodes_wav_with_pyav_metadata_errors_ignored(tmp_path):
    from faster_whisper.audio import decode_audio

    audio = tmp_path / "metadata.wav"
    with wave.open(str(audio), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(b"\0\0" * 1600)

    decoded = decode_audio(str(audio), sampling_rate=16000)

    assert len(decoded) == 1600
