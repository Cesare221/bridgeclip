"""Representative climate explanation; synthetic timings, no video required."""

import asyncio
import json

import httpx
import pytest

from clip_engine.services import local_planning
from clip_engine.services.local_planning import (
    LocalPlanningError,
    LocalPlanningService,
    _build_ollama_payload,
    expansion_variants,
    generate_candidates,
)
from clip_engine.services.transcription_service import TranscriptionResult


@pytest.fixture
def climate_transcript():
    sentences = [
        (0, 20, "O aquecimento global ocorre quando gases emitidos pelas atividades humanas se acumulam na atmosfera e alteram o equilíbrio natural da temperatura da Terra."),
        (20, 36, "O efeito estufa mantém nosso planeta aquecido e permite a vida mas a emissão adicional de gases aumenta a retenção de calor na atmosfera."),
        (36, 56, "Assim a energia que deveria escapar permanece no planeta elevando a temperatura média e provocando mudanças no clima por isso reduzir essas emissões limita o aquecimento."),
        (56, 60, "Essa é a relação entre emissões e aquecimento."),
    ]
    segments = []
    for start, end, text in sentences:
        tokens = text.split()
        words = [{"word": token,
                  "start_time_ms": round(1000 * (start + (end - start) * index / len(tokens))),
                  "end_time_ms": round(1000 * (start + (end - start) * (index + 1) / len(tokens)))}
                 for index, token in enumerate(tokens)]
        segments.append({"text": text, "words": words})
    return segments


def run_selection(candidates, *, review=False, close_at=56_000, invalid_expansion=False, **kwargs):
    requests = []

    async def handle(request):
        payload = json.loads(request.content)
        data = json.loads(payload["messages"][1]["content"])
        requests.append(payload)
        if invalid_expansion and len(requests) > 1:
            return httpx.Response(200, json={"message": {"content": '{"evaluations": []}'}})
        by_id = {candidate["id"]: candidate for candidate in candidates}
        evaluations = [{"id": item["id"],
                        "narrative_closed": by_id[item["id"]]["end_time_ms"] >= close_at
                        and by_id[item["id"]]["start_time_ms"] == 0,
                        "relevance_score": 90,
                        "rationale": "Complete explanation." if by_id[item["id"]]["end_time_ms"] >= close_at else "Missing conclusion."}
                       for item in data["candidates"]]
        return httpx.Response(200, json={"message": {"content": json.dumps({"evaluations": evaluations})}})

    async def run():
        async with httpx.AsyncClient(base_url="http://127.0.0.1:11434", transport=httpx.MockTransport(handle)) as client:
            planner = LocalPlanningService(http_client=client)
            result = await planner.select_candidates(candidates, include_preliminary=review, **kwargs)
            return result, planner.selection_diagnostics

    result, diagnostics = asyncio.run(run())
    return result, diagnostics, requests


def test_expands_36_to_56_and_prefers_shortest_complete(climate_transcript):
    candidates = generate_candidates(climate_transcript, source_duration_seconds=60)
    assert candidates[0]["end_time_ms"] == 36_000
    selected, diagnostics, requests = run_selection(candidates)
    assert [(item["start_time_ms"], item["end_time_ms"]) for item in selected] == [(0, 56_000)]
    assert selected[0]["expansion"]["original_end_time_ms"] == 36_000
    assert selected[0]["narrative_closed"] is True
    assert len(requests) == 2
    assert all(payload["think"] is False for payload in requests)
    assert all("transcript" not in item for item in diagnostics)


@pytest.mark.parametrize("review", [False, True])
def test_no_artificial_approval_when_conclusion_outside_limit(climate_transcript, review):
    climate_transcript[2]["words"][-1]["end_time_ms"] = 65_000
    candidates = generate_candidates(climate_transcript[:3], source_duration_seconds=65)
    selected, _, _ = run_selection(candidates, review=review, close_at=65_000, source_duration_seconds=65)
    if review:
        assert len(selected) == 1
        assert selected[0]["selection_decision"] == "narrative_not_closed"
        assert selected[0]["narrative_closed"] is False
        assert selected[0]["end_time_ms"] == 36_000
    else:
        assert selected == []


def test_expansion_respects_requested_range_and_video(climate_transcript):
    candidates = generate_candidates(climate_transcript, source_duration_seconds=60, end_time_seconds=54)
    selected, _, _ = run_selection(candidates, end_time_seconds=54, source_duration_seconds=60)
    assert selected == []
    assert all(item["end_time_ms"] <= 54_000 for item in candidates)


def test_expansion_rejects_invalid_ollama_json(climate_transcript):
    candidates = generate_candidates(climate_transcript, source_duration_seconds=60)
    with pytest.raises(LocalPlanningError) as error:
        run_selection(candidates, invalid_expansion=True)
    assert error.value.reason == "invalid_ollama_response"


def test_variants_cover_start_end_and_both_with_bounded_budget():
    original = {"id": 0, "start_time_ms": 10_000, "end_time_ms": 46_000, "duration_seconds": 36}
    variants = [{"id": index, "start_time_ms": start, "end_time_ms": end, "duration_seconds": (end - start) / 1000}
                for index, (start, end) in enumerate([(10_000, 50_000), (0, 46_000), (0, 56_000), (10_000, 56_000), (0, 60_000)], 1)]
    result = expansion_variants(original, [original, *variants])
    assert len(result) == 4
    assert {1, 2, 3}.issubset({item["id"] for item in result})


def test_review_and_automatic_keep_two_disjoint_expansions(climate_transcript):
    candidates = generate_candidates(climate_transcript, source_duration_seconds=60)
    shifted = [{**item, "id": item["id"] + 100,
                "start_time_ms": item["start_time_ms"] + 70_000,
                "end_time_ms": item["end_time_ms"] + 70_000} for item in candidates]
    for review in (False, True):
        selected, _, _ = run_selection(candidates + shifted, review=review, source_duration_seconds=130)
        assert len(selected) <= local_planning.MAX_SELECTED_CLIPS
        assert all(left["end_time_ms"] <= right["start_time_ms"] for left, right in zip(selected, selected[1:]))
        assert all(30 <= item["duration_seconds"] <= 60 for item in selected)


def test_prompt_preserves_conclusion_beyond_old_truncation(climate_transcript):
    candidate = generate_candidates(climate_transcript)[1]
    candidate["transcript"] = "context " * 200 + "Therefore the explanation is complete."
    payload = _build_ollama_payload([candidate], None)
    text = json.loads(payload["messages"][1]["content"])["candidates"][0]["transcript"]
    assert text.endswith("Therefore the explanation is complete.")


def test_global_expansion_budget_and_no_repeated_evaluations(climate_transcript):
    base = generate_candidates(climate_transcript)
    candidates = [{**item, "id": item["id"] + group * 100,
                   "start_time_ms": item["start_time_ms"] + group * 70_000,
                   "end_time_ms": item["end_time_ms"] + group * 70_000}
                  for group in range(12) for item in base]
    _, _, requests = run_selection(candidates, close_at=1_000_000)
    batches = [json.loads(payload["messages"][1]["content"])["candidates"] for payload in requests]
    ids = [item["id"] for batch in batches for item in batch]
    assert len(ids) == len(set(ids))
    assert 12 < len(ids) <= (
        local_planning.MAX_CANDIDATES_FOR_OLLAMA
        + local_planning.MAX_EXPANSIONS_FOR_OLLAMA
    )
    assert len(requests) <= local_planning.MAX_OLLAMA_REQUESTS
    assert all(len(batch) <= 3 for batch in batches)


@pytest.mark.parametrize("review", [False, True])
def test_expanded_plan_preserves_contract_and_editorial_metadata(climate_transcript, monkeypatch, review):
    async def evaluate(self, candidates, clip_request):
        return {item["id"]: {"narrative_closed": item["start_time_ms"] == 0 and item["end_time_ms"] >= 56_000,
                             "relevance_score": 90, "rationale": "Narrative evaluation."} for item in candidates}

    monkeypatch.setattr(LocalPlanningService, "_evaluate_candidates", evaluate)
    planner = LocalPlanningService()
    plan = asyncio.run(planner.plan_clips(
        TranscriptionResult(segments=climate_transcript, full_text="", duration_seconds=60),
        include_preliminary=review,
    ))
    assert plan.total_clips == 1
    assert plan.segments[0].start_time_ms == 0
    assert plan.segments[0].end_time_ms == 56_000
    assert plan.segments[0].editorial["expansion"]["original_end_time_ms"] == 36_000
    assert plan.segments[0].editorial["selection_decision"] == "selected"
    assert plan.api_costs is None


@pytest.mark.parametrize("original_start,original_end", [(20_000, 56_000), (20_000, 50_000)])
def test_selection_can_expand_before_or_both_sides(monkeypatch, original_start, original_end):
    candidates = [{"id": index, "start_time_ms": start, "end_time_ms": end,
                   "duration_seconds": (end - start) / 1000, "transcript": "A timed explanation.", "context_after": ""}
                  for index, (start, end) in enumerate([(original_start, original_end), (0, 56_000), (0, 60_000)])]
    monkeypatch.setattr(local_planning, "shortlist_candidates", lambda items, **kwargs: items[:1])
    selected, _, _ = run_selection(candidates)
    assert [(item["start_time_ms"], item["end_time_ms"]) for item in selected] == [(0, 56_000)]
