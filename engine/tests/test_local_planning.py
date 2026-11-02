import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from clip_engine.services.local_planning import (
    MAX_OLLAMA_REQUESTS,
    MAX_SELECTED_CLIPS,
    LocalPlanningError,
    LocalPlanningService,
    _build_ollama_payload,
    generate_candidates,
    shortlist_candidates,
    select_non_overlapping_candidates,
)
from clip_engine.services.transcription_service import (
    TranscriptSegment,
    TranscriptWord,
    TranscriptionResult,
)


def transcript_segments(sentence_count=3):
    segments = []
    timestamp = 0
    for sentence in range(sentence_count):
        words = []
        for word_index in range(20):
            text = f"word{sentence}_{word_index}"
            if word_index == 19:
                text += "."
            words.append(TranscriptWord(text, timestamp, timestamp + 1000))
            timestamp += 1000
        segments.append(
            TranscriptSegment(
                start_time_ms=words[0].start_time_ms,
                end_time_ms=words[-1].end_time_ms,
                text=" ".join(word.word for word in words),
                words=words,
            )
        )
    return segments


def longform_segments(topic_count):
    segments = []
    for topic in range(topic_count):
        start = topic * 60_000
        words = []
        for word_index in range(45):
            text = f"topic{topic}_word{word_index}"
            if word_index == 44:
                text += "."
            words.append(TranscriptWord(text, start + word_index * 1000, start + (word_index + 1) * 1000))
        segments.append(TranscriptSegment(
            start_time_ms=words[0].start_time_ms,
            end_time_ms=words[-1].end_time_ms,
            text=" ".join(word.word for word in words),
            words=words,
        ))
    return segments


def run_longform_plan(topic_count, duration_seconds, max_clips=MAX_SELECTED_CLIPS, monkeypatch_budget=None):
    requests = []

    async def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        prompt = json.loads(payload["messages"][1]["content"])
        return httpx.Response(200, json=response_for([
            evaluation(candidate["id"], score=90 - candidate["id"] % 5)
            for candidate in prompt["candidates"]
        ]))

    async def run():
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:11434",
            transport=httpx.MockTransport(handle),
        ) as client:
            planner = LocalPlanningService(http_client=client)
            result = await planner.plan_clips(
                TranscriptionResult(
                    segments=longform_segments(topic_count),
                    full_text="",
                    duration_seconds=duration_seconds,
                ),
                video_metadata=SimpleNamespace(duration_seconds=duration_seconds),
                max_clips=max_clips,
            )
            return result, dict(planner.planning_metrics), requests

    return asyncio.run(run())


def response_for(evaluations):
    return {"message": {"content": json.dumps({"evaluations": evaluations})}}


def evaluation(candidate_id, *, closed=True, score=80, rationale="Complete idea."):
    return {
        "id": candidate_id,
        "narrative_closed": closed,
        "relevance_score": score,
        "rationale": rationale,
    }


def fake_client(responses, captured=None):
    queue = list(responses)
    requests = captured if captured is not None else []

    async def handle(request):
        payload = json.loads(request.content)
        requests.append((str(request.url), payload))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            status, body = item
            return httpx.Response(status, json=body)
        return httpx.Response(200, json=item)

    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handle),
    )
    return client, requests


async def select_and_close(planner, candidates, **kwargs):
    try:
        return await planner.select_candidates(candidates, **kwargs)
    finally:
        await planner.close()
        if planner._http_client and not planner._http_client.is_closed:
            await planner._http_client.aclose()


def make_candidates():
    return generate_candidates(
        transcript_segments(),
        source_duration_seconds=60,
        min_duration_seconds=30,
        max_duration_seconds=60,
    )


def test_candidate_generation_uses_real_word_boundaries_and_post_cut_context():
    candidates = make_candidates()

    assert len(candidates) == 3
    assert candidates[0]["id"] == 0
    assert candidates[0]["start_time_ms"] == 0
    assert candidates[0]["end_time_ms"] == 40_000
    assert candidates[0]["duration_seconds"] == 40.0
    assert len(candidates[0]["transcript"].split()) == 40
    assert candidates[0]["context_after"].startswith("word2_0")
    assert all(30 <= candidate["duration_seconds"] <= 60 for candidate in candidates)


def test_candidate_generation_honors_selected_time_range():
    candidates = generate_candidates(
        transcript_segments(),
        source_duration_seconds=60,
        start_time_seconds=20,
        end_time_seconds=60,
        min_duration_seconds=30,
        max_duration_seconds=60,
    )

    assert candidates
    assert all(candidate["start_time_ms"] >= 20_000 for candidate in candidates)
    assert all(candidate["end_time_ms"] <= 60_000 for candidate in candidates)


def test_candidate_generation_rejects_transcript_without_word_timestamps():
    with pytest.raises(LocalPlanningError) as error:
        generate_candidates([TranscriptSegment(0, 40_000, "spoken words")])
    assert error.value.reason == "missing_word_timestamps"


def test_candidate_generation_rejects_words_outside_video_duration():
    segments = transcript_segments(1)
    segments[0].words[-1].end_time_ms = 25_000

    with pytest.raises(LocalPlanningError) as error:
        generate_candidates(segments, source_duration_seconds=20)
    assert error.value.reason == "invalid_transcript_timestamps"


def test_candidate_generation_caps_dense_windows_and_reports_pruned_intervals():
    segments = []
    for sentence in range(60):
        start = sentence * 3000
        words = [
            TranscriptWord(
                f"sentence{sentence}_word{index}" + ("." if index == 9 else ""),
                start + index * 300,
                start + (index + 1) * 300,
            )
            for index in range(10)
        ]
        segments.append(TranscriptSegment(
            start_time_ms=words[0].start_time_ms,
            end_time_ms=words[-1].end_time_ms,
            text=" ".join(word.word for word in words),
            words=words,
        ))
    stats = {}

    candidates = generate_candidates(
        segments,
        source_duration_seconds=180,
        stats=stats,
    )

    assert stats["candidate_intervals_generated"] > len(candidates)
    assert stats["candidates_discarded_by_window_budget"] > 0
    assert len(candidates) <= 2 * 16


def test_shortlisting_keeps_candidates_across_overlapping_window_edges():
    candidates = [
        {
            "id": index,
            "start_time_ms": start,
            "end_time_ms": start + 40_000,
            "duration_seconds": 40,
        }
        for index, start in enumerate((119_000, 121_000))
    ]
    stats = {}

    result = shortlist_candidates(candidates, limit=10, stats=stats)

    assert [candidate["id"] for candidate in result] == [0, 1]
    assert stats["windows_analyzed"] == 2


def test_ollama_payload_is_local_structured_and_includes_context():
    candidates = make_candidates()[:2]
    payload = _build_ollama_payload(candidates, "only the pricing section")

    assert payload["model"] == "qwen3:1.7b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["format"]["type"] == "object"
    assert payload["format"]["additionalProperties"] is False
    assert payload["format"]["properties"]["evaluations"]["items"]["properties"]["id"]["enum"] == [0, 1]
    user_data = json.loads(payload["messages"][1]["content"])
    assert user_data["clip_request"] == "only the pricing section"
    assert user_data["candidates"][0]["context_after"].startswith("word2_0")
    assert "context_after" in payload["messages"][0]["content"]


def test_selection_retries_one_malformed_response_and_keeps_only_closed_candidates():
    candidates = make_candidates()
    captured = []
    client, requests = fake_client(
        [
            {"message": {"content": '{"evaluations": []}'}},
            response_for([
                evaluation(0, closed=True, score=90),
                evaluation(1, closed=False, score=70),
                evaluation(2, closed=False, score=100),
            ]),
        ],
        captured,
    )
    planner = LocalPlanningService(http_client=client)

    selected = asyncio.run(select_and_close(planner, candidates))

    assert len(requests) == 2
    assert [candidate["id"] for candidate in selected] == [0]
    assert selected[0]["relevance_score"] == 90
    assert selected[0]["narrative_closed"] is True
    assert all(url == "http://127.0.0.1:11434/api/chat" for url, _ in requests)


def test_review_can_include_preliminary_candidates_without_changing_automatic_selection():
    candidates = make_candidates()
    evaluations = [
        evaluation(0, closed=False, score=90, rationale="Preliminary."),
        evaluation(1, closed=False, score=70, rationale="Expansion candidate."),
        evaluation(2, closed=True, score=80, rationale="Closed narrative."),
    ]

    automatic_client, _ = fake_client([response_for(evaluations)])
    automatic = LocalPlanningService(http_client=automatic_client)
    selected = asyncio.run(select_and_close(automatic, candidates))

    review_client, _ = fake_client([response_for(evaluations)])
    review = LocalPlanningService(http_client=review_client)
    preliminary = asyncio.run(
        select_and_close(review, candidates, include_preliminary=True)
    )

    assert [candidate["id"] for candidate in selected] == [2]
    assert [candidate["id"] for candidate in preliminary] == [2]
    assert [candidate["selection_decision"] for candidate in preliminary] == [
        "selected",
    ]
    assert all(candidate["structural_valid"] for candidate in preliminary)
    assert [
        candidate["decision"] for candidate in review.selection_diagnostics
    ] == [
        "narrative_not_closed",
        "narrative_not_closed",
        "selected",
    ]
    assert all(
        set(candidate)
        == {
            "id",
            "start_time_ms",
            "end_time_ms",
            "duration_seconds",
            "structural_valid",
            "relevance_score",
            "narrative_closed",
            "decision",
        }
        for candidate in review.selection_diagnostics
    )


@pytest.mark.parametrize(
    "invalid_evaluations",
    [
        [evaluation(0), evaluation(999)],
        [evaluation(0), evaluation(0)],
    ],
    ids=["unknown-id", "duplicate-id"],
)
def test_selection_rejects_unknown_or_duplicate_ids_after_one_retry(invalid_evaluations):
    candidates = make_candidates()
    invalid = response_for(invalid_evaluations)
    client, _ = fake_client([invalid, invalid])
    planner = LocalPlanningService(http_client=client)

    with pytest.raises(LocalPlanningError) as error:
        asyncio.run(select_and_close(planner, candidates))
    assert error.value.reason == "invalid_ollama_response"


def test_unavailable_ollama_has_an_actionable_local_error():
    client, _ = fake_client([httpx.ConnectError("not connected")])
    planner = LocalPlanningService(http_client=client)

    with pytest.raises(LocalPlanningError, match="Start Ollama") as error:
        asyncio.run(select_and_close(planner, make_candidates()))
    assert error.value.reason == "ollama_unavailable"
    assert planner.planning_metrics["evaluation_failure_reason"] == "ollama_unavailable"
    assert planner.planning_metrics["ollama_calls"] == 1


def test_ollama_timeout_is_reported_as_a_local_planning_failure():
    client, _ = fake_client([httpx.ReadTimeout("slow model")])
    planner = LocalPlanningService(http_client=client)

    with pytest.raises(LocalPlanningError) as error:
        asyncio.run(select_and_close(planner, make_candidates()))

    assert error.value.reason == "ollama_timeout"
    assert planner.planning_metrics["evaluation_failure_reason"] == "ollama_timeout"


def test_local_planning_propagates_cancellation():
    planner = LocalPlanningService()

    async def cancel_request(*_args, **_kwargs):
        raise asyncio.CancelledError

    planner._evaluate_candidates = cancel_request

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(select_and_close(planner, make_candidates()))


def test_missing_ollama_model_has_an_install_instruction():
    client, _ = fake_client([(404, {"error": "model not found"})])
    planner = LocalPlanningService(http_client=client)

    with pytest.raises(LocalPlanningError, match="ollama pull qwen3:1.7b") as error:
        asyncio.run(select_and_close(planner, make_candidates()))
    assert error.value.reason == "ollama_model_unavailable"


def test_empty_candidates_do_not_call_ollama():
    client, requests = fake_client([])
    planner = LocalPlanningService(http_client=client)

    selected = asyncio.run(select_and_close(planner, []))

    assert selected == []
    assert requests == []


def test_overlap_optimization_prefers_the_best_non_overlapping_pair():
    candidates = [
        {"id": 0, "start_time_ms": 0, "end_time_ms": 40_000, "narrative_closed": True, "relevance_score": 100},
        {"id": 1, "start_time_ms": 20_000, "end_time_ms": 60_000, "narrative_closed": True, "relevance_score": 95},
        {"id": 2, "start_time_ms": 60_000, "end_time_ms": 100_000, "narrative_closed": True, "relevance_score": 80},
    ]

    selected = select_non_overlapping_candidates(candidates)

    assert [candidate["id"] for candidate in selected] == [0, 2]


def test_overlap_selection_allows_one_or_zero_results():
    candidates = [
        {"id": 0, "start_time_ms": 0, "end_time_ms": 40_000, "narrative_closed": True, "relevance_score": 80},
        {"id": 1, "start_time_ms": 10_000, "end_time_ms": 50_000, "narrative_closed": False, "relevance_score": 100},
    ]

    assert [candidate["id"] for candidate in select_non_overlapping_candidates(candidates)] == [0]
    assert select_non_overlapping_candidates(
        [{**candidate, "narrative_closed": False} for candidate in candidates]
    ) == []


def test_interval_selection_honors_each_manual_maximum_without_filling():
    candidates = [
        {
            "id": index,
            "start_time_ms": index * 60_000,
            "end_time_ms": index * 60_000 + 45_000,
            "narrative_closed": True,
            "relevance_score": 90,
        }
        for index in range(15)
    ]

    for maximum in (1, 2, 5, 10, 15):
        selected = select_non_overlapping_candidates(candidates, max_clips=maximum)
        assert len(selected) == maximum
        assert all(
            left["end_time_ms"] <= right["start_time_ms"]
            for left, right in zip(selected, selected[1:])
        )

    assert len(select_non_overlapping_candidates(candidates[:1], max_clips=15)) == 1


def test_three_minute_video_returns_only_existing_complete_candidates():
    result, metrics, _ = run_longform_plan(topic_count=1, duration_seconds=180)

    assert result.total_clips == 1
    assert metrics["candidates_discovered"] == 1
    assert metrics["selected_candidates"] == 1


def test_nineteen_minute_video_can_select_more_than_two_disjoint_candidates():
    result, metrics, _ = run_longform_plan(topic_count=19, duration_seconds=19 * 60)

    assert result.total_clips > 2
    assert result.total_clips <= MAX_SELECTED_CLIPS
    assert [segment.start_time_ms for segment in result.segments] == sorted(
        segment.start_time_ms for segment in result.segments
    )
    assert all(
        left.end_time_ms <= right.start_time_ms
        for left, right in zip(result.segments, result.segments[1:])
    )
    assert metrics["windows_analyzed"] > 2
    assert metrics["ollama_calls"] <= metrics["ollama_request_budget"]


def test_sixty_minute_video_keeps_model_work_bounded_and_context_local():
    result, metrics, requests = run_longform_plan(topic_count=60, duration_seconds=60 * 60)

    assert result.total_clips == MAX_SELECTED_CLIPS
    assert metrics["candidates_discovered"] == 60
    assert metrics["candidates_shortlisted"] <= 48
    assert metrics["ollama_calls"] <= metrics["ollama_request_budget"] <= MAX_OLLAMA_REQUESTS
    assert all(
        len(json.loads(payload["messages"][1]["content"])["candidates"]) <= 3
        for payload in requests
    )
    assert all(
        "transcript" not in json.loads(payload["messages"][1]["content"])
        for payload in requests
    )


def test_exhausted_ollama_budget_returns_only_evaluated_candidates(monkeypatch):
    monkeypatch.setattr("clip_engine.services.local_planning._ollama_request_budget", lambda *_: 1)

    result, metrics, requests = run_longform_plan(topic_count=15, duration_seconds=15 * 60)

    assert result.total_clips == 3
    assert metrics["ollama_calls"] == 1
    assert metrics["budget_exhausted"] is True
    assert len(requests) == 1


def test_time_budget_exhaustion_returns_without_model_requests(monkeypatch):
    monkeypatch.setattr("clip_engine.services.local_planning.MAX_PLANNING_SECONDS", 0)

    result, metrics, requests = run_longform_plan(topic_count=5, duration_seconds=5 * 60)

    assert result.total_clips == 0
    assert metrics["budget_exhausted"] is True
    assert metrics["budget_exhaustion_reason"] == "time_limit"
    assert requests == []


def test_plan_clips_returns_shared_clip_plan_without_api_costs():
    candidates = make_candidates()
    client, _ = fake_client([
        response_for([
            evaluation(0, score=90),
            evaluation(1, closed=False, score=75),
            evaluation(2, closed=False, score=80),
        ])
    ])
    planner = LocalPlanningService(http_client=client)
    result = asyncio.run(
        planner.plan_clips(
            TranscriptionResult(
                segments=transcript_segments(),
                full_text="",
                duration_seconds=60,
            ),
            video_metadata=SimpleNamespace(duration_seconds=60),
            max_clips=2,
        )
    )
    asyncio.run(planner.close())
    asyncio.run(client.aclose())

    assert result.total_clips == 1
    assert len(result.segments) == 1
    assert result.segments[0].start_time_ms == candidates[0]["start_time_ms"]
    assert result.segments[0].editorial["relevance_score"] == 90
    assert result.api_costs is None


def test_plan_clips_returns_zero_without_request_when_no_candidates_fit():
    client, requests = fake_client([])
    planner = LocalPlanningService(http_client=client)
    result = asyncio.run(
        planner.plan_clips(
            TranscriptionResult(
                segments=transcript_segments(sentence_count=1),
                full_text="",
                duration_seconds=20,
            ),
            video_metadata=SimpleNamespace(duration_seconds=20),
            min_duration_seconds=None,
            max_duration_seconds=None,
        )
    )
    asyncio.run(planner.close())
    asyncio.run(client.aclose())

    assert result.total_clips == 0
    assert result.segments == []
    assert "No self-contained clips" in result.insights
    assert requests == []
