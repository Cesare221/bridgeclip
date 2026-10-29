import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from clip_engine.services import ai_clipping_pipeline as pipeline_module
from clip_engine.services.ai_clipping_pipeline import AIClippingPipeline, ClippingJobRequest, JobStatus
from clip_engine.services.intelligence_planner import ClipPlanResponse, ClipPlanSegment
from clip_engine.services.transcription_service import TranscriptSegment, TranscriptionResult


class FakeDownloader:
    def __init__(self, source_path):
        self.progress_callback = None
        self.result = SimpleNamespace(
            video_path=str(source_path),
            metadata=SimpleNamespace(
                title="Local source",
                duration_seconds=60,
                description="",
                uploader=None,
            ),
            source_type="local",
        )

    async def download_video(self, **kwargs):
        return self.result


class FakeTranscriber:
    progress_callback = None
    detail_callback = None

    def __init__(self, result):
        self.result = result
        self.arguments = None

    async def transcribe(self, **kwargs):
        self.arguments = kwargs
        return self.result


class FakePlanner:
    progress_callback = None

    def __init__(self, response):
        self.response = response
        self.arguments = None
        self.closed = False
        self.audit = {"requests": []}

    async def plan_clips(self, **kwargs):
        self.arguments = kwargs
        return self.response

    async def close(self):
        self.closed = True


class FakeJev:
    instances = []

    def __init__(self, *args, **kwargs):
        self.estimated_cost_usd = 0
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.enabled = False
        self.__class__.instances.append(self)

    @classmethod
    def from_settings(cls, *args, **kwargs):
        raise AssertionError("Local mode must not initialize a paid reviewer")

    async def evaluate(self, *args, **kwargs):
        raise AssertionError("Local mode must not make reviewer requests")


class FakeRenderer:
    async def _get_video_dimensions(self, video_path):
        return 1920, 1080

    async def capture_framing_source(self, source, destination, **kwargs):
        Path(destination).write_bytes(b"preview")


class FakeContextService:
    calls = 0

    async def build(self, metadata):
        self.__class__.calls += 1
        raise AssertionError("Local mode must not research source context")


class FakeS3Service:
    calls = 0

    async def upload_json_artifact(self, **kwargs):
        self.__class__.calls += 1
        raise AssertionError("Local mode must not upload artifacts")


def build_pipeline(monkeypatch, tmp_path, transcription, planned_segments):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"local video")
    work_root = tmp_path / "work"
    output_root = tmp_path / "output"
    settings = SimpleNamespace(
        clipping_mode="local",
        local_mode=False,
        layout_vision_enabled=True,
        jev_enabled=True,
        jev_visual_context=True,
        source_context_web_research=True,
        temp_directory=str(work_root),
        local_output_dir=str(output_root),
        max_clips_absolute=2,
        jev_threshold=0.75,
        jev_self_contained_threshold=0.70,
        jev_faithful_to_source_threshold=0.65,
        jev_title_supported_threshold=0.70,
        jev_sponsor_threshold=0.80,
        jev_evidence_threshold=0.50,
        jev_cut_threshold=0.95,
    )
    planner = FakePlanner(ClipPlanResponse(
        segments=planned_segments,
        total_clips=len(planned_segments),
        insights="Local plan",
    ))
    transcriber = FakeTranscriber(transcription)
    downloader = FakeDownloader(source)
    webhook = SimpleNamespace(clear_job_tracking=lambda job_id: None)

    monkeypatch.setattr(pipeline_module, "get_settings", lambda: settings)
    monkeypatch.setattr(pipeline_module, "VideoDownloaderService", lambda: downloader)
    monkeypatch.setattr(pipeline_module, "FasterWhisperTranscriptionService", lambda: transcriber)
    monkeypatch.setattr(pipeline_module, "TranscriptionService", lambda: (_ for _ in ()).throw(
        AssertionError("Local mode must not initialize the OpenRouter transcriber")
    ))
    monkeypatch.setattr(pipeline_module, "SourceContextService", lambda _settings: FakeContextService())
    monkeypatch.setattr(pipeline_module, "LocalPlanningService", lambda: planner)
    monkeypatch.setattr(pipeline_module, "IntelligencePlannerService", lambda: (_ for _ in ()).throw(
        AssertionError("Local mode must not initialize the OpenRouter planner")
    ))
    monkeypatch.setattr(pipeline_module, "RenderingService", FakeRenderer)
    monkeypatch.setattr(pipeline_module, "S3UploadService", FakeS3Service)
    monkeypatch.setattr(pipeline_module, "JevService", FakeJev)
    return AIClippingPipeline(webhook_service=webhook), settings, planner, transcriber


def test_local_pipeline_completes_without_candidates_or_external_calls(monkeypatch, tmp_path):
    FakeContextService.calls = 0
    FakeS3Service.calls = 0
    FakeJev.instances = []
    pipeline, settings, planner, transcriber = build_pipeline(
        monkeypatch,
        tmp_path,
        TranscriptionResult(segments=[], full_text="", provider="faster-whisper", model="base"),
        [],
    )
    progress = []
    pipeline.progress_callback = progress.append
    request = ClippingJobRequest(
        video_url=str(tmp_path / "source.mp4"),
        job_id="local-empty",
    )

    result = asyncio.run(pipeline.process_video(request))

    assert result.status == JobStatus.COMPLETED
    assert result.output.total_clips == 0
    assert result.output.metrics["api_costs"] == {
        "total_estimated_cost_usd": 0.0,
        "cost_incomplete": False,
    }
    assert planner.arguments["max_clips"] == 2
    assert planner.arguments["include_preliminary"] is False
    assert planner.arguments["jev_enabled"] is False
    assert planner.arguments["frames"] == []
    assert transcriber.arguments["source_duration_seconds"] == 60
    assert settings.local_mode is True
    assert settings.layout_vision_enabled is False
    assert settings.jev_enabled is False
    assert settings.jev_visual_context is False
    assert settings.source_context_web_research is False
    assert planner.closed is True
    assert FakeContextService.calls == 0
    assert FakeS3Service.calls == 0
    assert progress[-1].status == JobStatus.COMPLETED
    assert progress[-1].progress_percent == 100
    output_dir = tmp_path / "output" / "local-empty"
    assert (output_dir / "transcript.json").is_file()
    assert (output_dir / "plan.json").is_file()
    assert (output_dir / "job_output.json").is_file()
    assert not (output_dir / "editor-project.json").exists()


def test_local_review_pipeline_creates_unreviewed_local_editor_project(monkeypatch, tmp_path):
    FakeContextService.calls = 0
    FakeS3Service.calls = 0
    FakeJev.instances = []
    transcript = TranscriptSegment(0, 35_000, "A complete local story.")
    candidate = ClipPlanSegment(0, 35_000, 0.9, summary="A complete local story.")
    candidate.editorial = {
        "provider": "ollama",
        "narrative_closed": False,
        "selection_decision": "narrative_not_closed",
        "relevance_score": 90,
        "rationale": "Preliminary candidate for human review.",
    }
    pipeline, _, planner, _ = build_pipeline(
        monkeypatch,
        tmp_path,
        TranscriptionResult(
            segments=[transcript],
            full_text=transcript.text,
            provider="faster-whisper",
            model="base",
        ),
        [candidate],
    )
    request = ClippingJobRequest(
        video_url=str(tmp_path / "source.mp4"),
        job_id="local-review",
        workflow="review",
        layout_style="fit",
    )

    result = asyncio.run(pipeline.process_video(request))

    assert result.status == JobStatus.COMPLETED
    assert result.output.editor_project is True
    assert planner.arguments["include_preliminary"] is True
    assert planner.arguments["jev_enabled"] is False
    assert len(FakeJev.instances) == 2
    assert all(service.requests == 0 for service in FakeJev.instances)
    assert FakeContextService.calls == 0
    assert FakeS3Service.calls == 0
    project = json.loads(
        (tmp_path / "output" / "local-review" / "editor-project.json").read_text(
            encoding="utf-8"
        )
    )
    assert project["candidates"][0]["local_selection"] == {
        "decision": "narrative_not_closed",
        "narrative_closed": False,
    }
    project_path = tmp_path / "output" / "local-review" / "editor-project.json"
    assert project_path.is_file()
    project = json.loads(project_path.read_text(encoding="utf-8"))
    assert project["provider"] == "local"
    assert len(project["candidates"]) == 1
    assert project["candidates"][0]["review"] is None
