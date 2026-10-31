"""Manual review retains failed candidates and exports exactly the selected edit."""
import asyncio
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from clip_engine.services.manual_editor import prepare_project, review_candidate, manual_plan, validate_candidate, run_editor, signature, scene_motion, caption_transcript
from clip_engine.services.transcription_service import TranscriptSegment, TranscriptWord
from clip_engine.services.coherence_review import CLIP_QUESTIONS, CUT_QUESTIONS
from clip_engine.services.layout_renderer import build_layout_graph, shot_views
from clip_engine.services.intelligence_planner import ClipPlanSegment
from clip_engine.services.rendering_service import RenderRequest, RenderingService, RenderingError
from tests.test_coherence_review import reviewer, transcript


def candidate():
    return {'id': 'candidate-1', 'title': 'The result', 'ranges': [[1000, 5000], [7000, 10000]],
        'scenes': [{'at_ms': 0, 'layout': 'fill', 'crops': [[.1, 0, .31640625, 1]]},
                   {'at_ms': 8000, 'layout': 'split', 'crops': [[0, .1, .4, .35555556], [.5, .3, .4, .35555556]]}],
        'captions': True, 'caption_preset': 'pop', 'video_speed': 1.25, 'score': .8, 'reason': '', 'review': None, 'exports': []}


def test_all_jev_questions_and_cut_questions_are_preserved_without_repair():
    gate, calls = reviewer(lambda state, q: False)
    gate.repair = AsyncMock()
    c = candidate()
    original = copy.deepcopy(c)
    asyncio.run(review_candidate(c, gate))
    assert c['ranges'] == original['ranges'] and c['title'] == original['title']
    assert c['review']['decision'] == 'needs_attention'
    assert {q['id'] for q in c['review']['questions']} == set(CLIP_QUESTIONS)
    assert {q['id'] for q in c['review']['cuts'][0]['questions']} == set(CUT_QUESTIONS)
    assert c['review']['cuts'][0]['interval'] == [5000, 7000]
    assert c['review']['signature'] == signature(c)
    assert next(q for q in c['review']['questions'] if q['id'] == 'faithful_to_source')['threshold'] == .65
    gate.repair.assert_not_called()


def test_unavailable_jev_is_not_a_pass_and_still_shows_every_question():
    gate, _ = reviewer()
    gate.service._api_key = ''
    c = candidate()
    asyncio.run(review_candidate(c, gate))
    assert c['review']['decision'] == 'needs_attention'
    assert len(c['review']['questions']) == 8
    assert all(q['probability'] is None for q in c['review']['questions'])


def test_prepare_retains_rejected_candidates_and_does_not_render(tmp_path):
    gate, _ = reviewer(lambda state, q: False)
    source = tmp_path / 'original.mov'
    source.write_bytes(b'original source')
    out = tmp_path / 'run'; out.mkdir()
    async def preview(src, dest, **kwargs): Path(dest).write_bytes(b'preview')
    renderer = SimpleNamespace(_get_video_dimensions=AsyncMock(return_value=(1920, 1080)),
        capture_framing_source=AsyncMock(side_effect=preview), render_clip=AsyncMock())
    request = SimpleNamespace(aspect_ratio='9:16', layout_style='fit', include_captions=True, caption_preset='pop', video_speed=1)
    segments = [ClipPlanSegment(0, 5000, .9, summary='First'), ClipPlanSegment(6000, 11000, .7, summary='Second')]
    project = asyncio.run(prepare_project(request, segments, transcript(), SimpleNamespace(video_path=str(source),
        metadata=SimpleNamespace(title='Original', duration_seconds=12)), renderer, gate, str(out), lambda *_: None))
    assert len(project['candidates']) == 2
    assert project['transcript'][0]['words'] == [
        {'text': 'Here is the setup.', 'start_ms': 0, 'end_ms': 2000}
    ]
    assert all(c['status'] == 'refining' and c['caption_edits'] == [] for c in project['candidates'])
    assert all(c['review']['decision'] == 'needs_attention' for c in project['candidates'])
    assert project['candidates'][0]['ranges'] == [[0, 5000]]
    assert (out / 'editor-source.mp4').read_bytes() == b'original source'
    source.write_bytes(b'changed elsewhere')
    assert (out / 'editor-source.mp4').read_bytes() == b'original source'
    renderer.render_clip.assert_not_called()
    assert json.loads((out / 'editor-project.json').read_text())['version'] == 1


def test_local_prepare_saves_unreviewed_candidate_without_jev(tmp_path):
    source = tmp_path / 'original.mov'
    source.write_bytes(b'original source')
    out = tmp_path / 'run'; out.mkdir()
    async def preview(src, dest, **kwargs): Path(dest).write_bytes(b'preview')
    renderer = SimpleNamespace(_get_video_dimensions=AsyncMock(return_value=(1920, 1080)),
        capture_framing_source=AsyncMock(side_effect=preview), render_clip=AsyncMock())
    reviewer = SimpleNamespace(judge=AsyncMock())
    request = SimpleNamespace(aspect_ratio='9:16', layout_style='fit', include_captions=True,
        caption_preset='pop', video_speed=1)
    segment = ClipPlanSegment(0, 5000, .93, summary='A complete idea.')
    segment.editorial = {
        'provider': 'ollama',
        'relevance_score': 93,
        'rationale': 'The explanation reaches a clear conclusion.',
    }

    project = asyncio.run(prepare_project(
        request, [segment], transcript(),
        SimpleNamespace(video_path=str(source), source_type='local',
            metadata=SimpleNamespace(title='Original', duration_seconds=12)),
        renderer, reviewer, str(out), lambda *_: None, provider='local',
    ))

    candidate_data = project['candidates'][0]
    assert project['provider'] == 'local'
    assert candidate_data['review'] is None
    assert candidate_data['status'] == 'refining'
    assert candidate_data['local_selection'] == {
        'decision': 'selected',
        'narrative_closed': True,
    }
    assert candidate_data['reason'] == 'The explanation reaches a clear conclusion.'
    assert candidate_data['score'] == .93
    reviewer.judge.assert_not_called()
    renderer.render_clip.assert_not_called()


def test_local_export_requires_human_approval_and_rejects_short_ranges(monkeypatch, tmp_path):
    from dataclasses import asdict
    from clip_engine.services.rendering_service import RenderResult

    c = {
        **candidate(),
        'ranges': [[0, 35_000]],
        'status': 'refining',
        'local_selection': {
            'decision': 'narrative_not_closed',
            'narrative_closed': False,
        },
    }
    project = {
        'version': 1,
        'revision': 0,
        'provider': 'local',
        'width': 1920,
        'height': 1080,
        'duration_ms': 70_000,
        'aspect_ratio': '9:16',
        'candidates': [c],
        'transcript': [asdict(segment) for segment in transcript()],
    }
    (tmp_path / 'editor-project.json').write_text(json.dumps(project))
    (tmp_path / 'editor-source.mp4').write_bytes(b'original')
    (tmp_path / 'transcript.json').write_text(json.dumps({'segments': [asdict(s) for s in transcript()]}))
    (tmp_path / 'job_output.json').write_text(json.dumps({'clips': [], 'total_clips': 0, 'editor_project': True}))
    monkeypatch.setattr(RenderingService, '_verify_ffmpeg', lambda _: None)
    render = AsyncMock()
    monkeypatch.setattr(RenderingService, 'render_clip', render)
    config = {'run': str(tmp_path), 'revision': 0, 'candidate_id': c['id'], 'action': 'export'}

    with pytest.raises(ValueError, match='Mark this clip ready') as error:
        asyncio.run(run_editor(config))
    assert error.value.editor_code == 'not_ready'
    render.assert_not_called()

    project['candidates'][0]['status'] = 'ready'
    project['candidates'][0]['ranges'] = [[0, 29_999]]
    (tmp_path / 'editor-project.json').write_text(json.dumps(project))
    with pytest.raises(ValueError, match='30 and 60 seconds') as error:
        asyncio.run(run_editor(config))
    assert error.value.editor_code == 'invalid_edit'
    render.assert_not_called()

    project['candidates'][0]['ranges'] = [[0, 35_000]]
    (tmp_path / 'editor-project.json').write_text(json.dumps(project))

    async def successful_render(self, request):
        Path(request.output_path).write_bytes(b'final clip')
        return RenderResult(request.output_path, 10, 35_000)

    monkeypatch.setattr(RenderingService, 'render_clip', successful_render)
    asyncio.run(run_editor(config))
    assert (tmp_path / 'clip_00.mp4').read_bytes() == b'final clip'


def test_local_export_rejects_overlap_with_previous_candidate(monkeypatch, tmp_path):
    from dataclasses import asdict

    c = {
        **candidate(),
        'ranges': [[20_000, 55_000]],
        'status': 'ready',
        'local_selection': {
            'decision': 'narrative_not_closed',
            'narrative_closed': False,
        },
    }
    project = {
        'version': 1,
        'revision': 0,
        'provider': 'local',
        'width': 1920,
        'height': 1080,
        'duration_ms': 70_000,
        'aspect_ratio': '9:16',
        'candidates': [c],
        'transcript': [asdict(segment) for segment in transcript()],
    }
    (tmp_path / 'editor-project.json').write_text(json.dumps(project))
    (tmp_path / 'editor-source.mp4').write_bytes(b'original')
    (tmp_path / 'transcript.json').write_text(json.dumps({'segments': [asdict(s) for s in transcript()]}))
    (tmp_path / 'job_output.json').write_text(json.dumps({
        'clips': [{'editor_candidate': 'previous', 'start_time_ms': 0, 'end_time_ms': 35_000}],
        'total_clips': 1,
        'editor_project': True,
    }))
    monkeypatch.setattr(RenderingService, '_verify_ffmpeg', lambda _: None)
    render = AsyncMock()
    monkeypatch.setattr(RenderingService, 'render_clip', render)

    with pytest.raises(ValueError, match='cannot overlap') as error:
        asyncio.run(run_editor({
            'run': str(tmp_path),
            'revision': 0,
            'candidate_id': c['id'],
            'action': 'export',
        }))

    assert error.value.editor_code == 'invalid_edit'
    render.assert_not_called()


def test_local_editor_rejects_jev_review_action(tmp_path):
    run = tmp_path / 'local-run'
    run.mkdir()
    (run / 'editor-project.json').write_text(json.dumps({
        'version': 1,
        'revision': 0,
        'provider': 'local',
        'media_freed': False,
    }))

    with pytest.raises(ValueError) as error:
        asyncio.run(run_editor({'run': str(run), 'action': 'review', 'revision': 0}))

    assert error.value.editor_code == 'review_unavailable'


def test_manual_crop_geometry_and_layout_switches_survive_cut_mapping():
    c = candidate()
    p = manual_plan({'width': 1920, 'height': 1080}, c)
    assert [(s.start_ms, s.end_ms) for s in p.shots] == [(0, 7000), (7000, 9000)]
    assert shot_views(p.shots[0], 0, 1920, 1080, 1080, 1920) == [((192, 0, 606, 1080), (0, 0, 1080, 1920))]
    graph = build_layout_graph(p, 1080, 1920, [(0, 4000), (6000, 9000)], True)
    assert 'crop=606:1080:192:0' in graph
    assert 'vstack=inputs=2' in graph and 'atrim=start=6.000:end=9.000' in graph


@pytest.mark.parametrize('duration', [-1, 99, 5001, float('inf'), float('nan'), '600', 600.5, None, True])
def test_invalid_movement_fails_before_rendering(duration):
    c = candidate()
    c['scenes'] = [c['scenes'][0], {**copy.deepcopy(c['scenes'][0]), 'at_ms': 2000, 'transition_ms': duration}]
    with pytest.raises(ValueError, match='movement'): validate_candidate(c, 12000)


def test_movement_is_continuous_when_interrupted_and_rejects_incompatible_layouts():
    c = candidate()
    c['scenes'] = [
        {'at_ms': 0, 'layout': 'fill', 'crops': [[0, 0, .4, 1]]},
        {'at_ms': 2000, 'layout': 'fill', 'crops': [[.6, .5, .2, .5]], 'transition_ms': 1000},
        {'at_ms': 2500, 'layout': 'fill', 'crops': [[0, 0, .4, 1]], 'transition_ms': 1000},
    ]
    validate_candidate(c, 12000)
    assert list(scene_motion(c))[2][1][0] == pytest.approx([.3, .25, .3, .75])
    p = manual_plan({'width': 1920, 'height': 1080}, c)
    shot = next(s for s in p.shots if s.start_ms <= 2000 < s.end_ms)
    assert shot_views(shot, 2000, 1920, 1080, 1080, 1920)[0][0] == pytest.approx([288, 134, 672, 944], abs=2)
    animated = signature(c)
    c['scenes'][1]['transition_ms'] = 600
    assert signature(c) != animated
    c['scenes'][0]['layout'] = 'fit'
    with pytest.raises(ValueError, match='movement'): validate_candidate(c, 12000)


@pytest.mark.parametrize('layout', ['fill', 'split'])
@pytest.mark.parametrize('fps', ['30', '30000/1001'])
def test_real_render_moves_and_zooms_on_the_source_clock_across_trim_and_removed_gap(layout, fps):
    """Decoded pixels verify start/middle/end crops, not just filter syntax."""
    import shutil
    import subprocess
    from fractions import Fraction
    import numpy as np
    from clip_engine.services.layout_renderer import video_frame_pieces
    bundled = Path(__file__).resolve().parents[2] / 'engine-bin/ffmpeg'
    ffmpeg = str(bundled) if bundled.exists() else shutil.which('ffmpeg')
    if not ffmpeg:
        pytest.skip('FFmpeg is needed for the actual motion render check')
    c = candidate()
    c['ranges'] = [[1000, 1600], [2000, 2600]]
    origins, targets = [[0, 0, .5, 1]], [[.5, .5, .25, .5]]
    if layout == 'split':
        origins += [[.5, .5, .25, .5]]; targets += [[0, 0, .5, 1]]
    c['scenes'] = [{'at_ms': 0, 'layout': layout, 'crops': origins},
        {'at_ms': 800, 'layout': layout, 'crops': targets, 'transition_ms': 1600}]
    validate_candidate(c, 12000)
    plan = manual_plan({'width': 320, 'height': 180}, c)
    keeps = [(0, 600), (1000, 1600)]
    graph = build_layout_graph(plan, 80, 120, keeps, fps=fps)
    source = np.zeros((180, 320, 3), dtype=np.uint8)
    source[:, :, 0] = np.arange(320)[None, :] * 255 / 320
    source[:, :, 1] = np.arange(180)[:, None] * 255 / 180
    result = subprocess.run([ffmpeg, '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '320x180',
        '-r', fps, '-i', 'pipe:0', '-filter_complex', graph, '-map', '[base]', '-pix_fmt', 'rgb24',
        '-f', 'rawvideo', 'pipe:1'], input=source.tobytes() * 60, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode()
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 120, 80, 3)
    pieces = video_frame_pieces(plan, keeps, fps)
    assert len(frames) == sum(count for _, _, count in pieces)
    output_index = 0
    for _, first, count in pieces:
        for frame in range(count):
            source_ms = 1000 + float(Fraction(first + frame, 1) / Fraction(fps) * 1000)
            t = max(0, min(1, (source_ms - 800) / 1600)); ease = t * t * (3 - 2 * t)
            for panel, (origin, target) in enumerate(zip(origins, targets)):
                x, y, w, h = [a + (b - a) * ease for a, b in zip(origin, target)]
                panel_h = 120 // len(origins)
                # Two distinct points also detect incorrect zoom/aspect, not just pan.
                for u, v in [(.25, .25), (.75, .75)]:
                    pixel = frames[output_index, panel * panel_h + int(panel_h * v), int(80 * u)]
                    assert pixel[:2] == pytest.approx([(x + w * u) * 255, (y + h * v) * 255], abs=5), (source_ms, panel, pixel)
            output_index += 1


@pytest.mark.parametrize('patch', [
    {'ranges': [[2000, 1000]]}, {'ranges': [[0, 3000], [2000, 4000]]}, {'ranges': [[0, float('nan')]]},
    {'video_speed': 0}, {'scenes': [{'at_ms': 0, 'layout': 'fill', 'crops': [[.8, 0, .4, 1]]}]},
    {'scenes': [{'at_ms': 1, 'layout': 'fill', 'crops': [[0, 0, 1, 1]]}]}
])
def test_invalid_manual_edits_fail_before_rendering(patch):
    c = {**candidate(), **patch}
    with pytest.raises(ValueError): validate_candidate(c, 12000)


@pytest.mark.parametrize('patch', [
    {'status': 'published'}, {'status': None}, {'caption_edits': None},
    {'caption_edits': [{'segment': 4, 'text': 'outside source'}]},
    {'caption_edits': [{'segment': True, 'text': 'not an index'}]},
    {'caption_edits': [{'segment': 0, 'text': 'a'}, {'segment': 0, 'text': 'b'}]},
    {'caption_edits': [{'segment': 0, 'text': 'x' * 2001}]},
    {'caption_edits': [{'segment': 0, 'text': 'bad\x00text'}]},
    {'caption_edits': [{'segment': i, 'text': ''} for i in range(2001)]},
])
def test_invalid_caption_edits_and_states_are_rejected(patch):
    with pytest.raises(ValueError): validate_candidate({**candidate(), **patch}, 12000, 4)


def test_caption_corrections_preserve_word_timing_without_mutating_source():
    source = [TranscriptSegment(0, 2000, 'The event happened.', words=[
        TranscriptWord('The', 100, 400), TranscriptWord('event', 500, 1000), TranscriptWord('happened.', 1200, 1900)
    ], speaker_label='Speaker 1'), TranscriptSegment(3000, 4000, 'Keep this.', words=[TranscriptWord('Keep this.', 3000, 4000)])]
    original = copy.deepcopy(source)
    changed = caption_transcript(source, [{'segment': 0, 'text': '  A\n  correction happened.  '}])
    assert changed[0].text == 'A correction happened.'
    assert [w.word for w in changed[0].words] == ['A', 'correction', 'happened.']
    assert [(w.start_time_ms, w.end_time_ms) for w in changed[0].words] == [(100, 400), (500, 1000), (1200, 1900)]
    assert changed[0].speaker_label == 'Speaker 1'
    changed[1].words[0].word = 'A renderer mutation'
    assert source == original


def test_caption_word_insertions_deletions_and_hidden_lines_use_original_spoken_interval():
    source = [TranscriptSegment(0, 2500, 'Original.', words=[TranscriptWord('Original.', 500, 2000)]),
        TranscriptSegment(3000, 4000, 'No words available.')]
    changed = caption_transcript(source, [{'segment': 0, 'text': 'Café is great!'}, {'segment': 1, 'text': '你好 again'}])
    assert [(w.word, w.start_time_ms, w.end_time_ms) for w in changed[0].words] == [('Café', 500, 1000), ('is', 1000, 1500), ('great!', 1500, 2000)]
    assert [(w.start_time_ms, w.end_time_ms) for w in changed[1].words] == [(3000, 3500), (3500, 4000)]
    fewer = caption_transcript(changed, [{'segment': 0, 'text': 'Great!'}])
    assert [(w.word, w.start_time_ms, w.end_time_ms) for w in fewer[0].words] == [('Great!', 500, 2000)]
    hidden = caption_transcript(source, [{'segment': 0, 'text': ' \n '}])
    assert len(hidden) == 1 and hidden[0].start_time_ms == 3000
    assert len(source) == 2 and source[0].text == 'Original.'


@pytest.mark.parametrize('ranges,speed', [
    ([[1000, 5000], [7000, 10000]], 1.25),
    ([[1000, 1200], [5000, 5200]], 1),  # Two short keeps must not restore the 3.8s gap.
    ([[1000, 1100], [2000, 5000], [6000, 6100]], 1),
    ([[1000, 2000], [3000, 3100], [5000, 6000]], 2),
    ([[1000, 1100]], 1),
])
def test_manual_render_never_replans_or_restores_user_cuts(monkeypatch, tmp_path, ranges, speed):
    monkeypatch.setattr(RenderingService, '_verify_ffmpeg', lambda _: None)
    renderer = RenderingService()
    renderer._get_video_dimensions = AsyncMock(return_value=(1920, 1080))
    renderer._plan_layout = AsyncMock(side_effect=AssertionError('No automatic layout'))
    renderer._keep_intervals = lambda *args: pytest.fail('No automatic pacing for manual exports')
    renderer._write_subtitles = AsyncMock(return_value=None)
    c = {**candidate(), 'ranges': ranges, 'video_speed': speed}
    validate_candidate(c, 12000)
    plan = manual_plan({'width': 1920, 'height': 1080}, c)
    calls = []
    async def render(request, plan, time_map, *args):
        calls.append(time_map.keeps)
        Path(request.output_path).write_bytes(b'fixture')
    renderer._render_edit = render
    request = RenderRequest(video_path='source.mp4', output_path=str(tmp_path / 'clip.mp4'), start_time_ms=ranges[0][0],
        end_time_ms=ranges[-1][1], source_width=1920, source_height=1080, apply_padding=False, pacing='natural',
        transcript_segments=transcript(), manual_plan=plan, manual_ranges_ms=ranges, video_speed=speed)
    result = asyncio.run(renderer.render_clip(request))
    assert calls == [[(a - ranges[0][0], b - ranges[0][0]) for a, b in ranges]]
    assert result.duration_ms == round(sum(b - a for a, b in ranges) / speed)
    renderer._render_edit = AsyncMock(side_effect=RenderingError('fixture failure'))
    with pytest.raises(RenderingError): asyncio.run(renderer.render_clip(request))
    assert renderer._render_edit.await_count == 1


def test_export_appends_library_clip_and_preserves_source_and_previous_exports(monkeypatch, tmp_path):
    from dataclasses import asdict
    from clip_engine.services.rendering_service import RenderResult
    c = {**candidate(), 'status': 'ready', 'caption_edits': [{'segment': 1, 'text': 'The corrected event happened.'}], 'caption_suppression_ranges': [[2000, 4000]], 'caption_y': .25}
    project = {'version': 1, 'revision': 3, 'width': 1920, 'height': 1080, 'duration_ms': 12000, 'aspect_ratio': '9:16', 'candidates': [c],
        'transcript': [{'start_ms': s.start_time_ms, 'end_ms': s.end_time_ms, 'text': s.text} for s in transcript()]}
    (tmp_path / 'editor-project.json').write_text(json.dumps(project))
    (tmp_path / 'editor-source.mp4').write_bytes(b'original')
    (tmp_path / 'transcript.json').write_text(json.dumps({'segments': [asdict(s) for s in transcript()]}))
    (tmp_path / 'job_output.json').write_text(json.dumps({'clips': [], 'total_clips': 0, 'editor_project': True}))
    monkeypatch.setattr(RenderingService, '_verify_ffmpeg', lambda _: None)
    async def render(self, request):
        assert request.apply_padding is False and request.pacing == 'natural'
        assert request.manual_ranges_ms == [(1000, 5000), (7000, 10000)] and request.include_captions
        assert request.caption_suppression_ranges_ms == [(2000, 4000)]
        assert request.caption_y == .25
        assert request.transcript_segments[1].text == 'The corrected event happened.'
        assert [w.word for w in request.transcript_segments[1].words] == ['The', 'corrected', 'event', 'happened.']
        Path(request.output_path).write_bytes(b'final clip')
        return RenderResult(request.output_path, 10, 5600, layout_type='two_shot')
    monkeypatch.setattr(RenderingService, 'render_clip', render)
    config = {'run': str(tmp_path), 'revision': 3, 'candidate_id': 'candidate-1', 'action': 'export'}
    asyncio.run(run_editor(config))
    output = json.loads((tmp_path / 'job_output.json').read_text())
    assert output['total_clips'] == 1 and output['clips'][0]['duration_ms'] == 5600
    assert (tmp_path / 'clip_00.mp4').read_bytes() == b'final clip'
    saved = json.loads((tmp_path / 'editor-project.json').read_text())
    assert saved['candidates'][0]['exports'] == [0] and saved['candidates'][0]['status'] == 'baked'
    assert saved['transcript'] == project['transcript']
    assert json.loads((tmp_path / 'transcript.json').read_text())['segments'][1]['text'] == transcript()[1].text
    with pytest.raises(ValueError, match='changed'): asyncio.run(run_editor(config))
    config['revision'] = 4
    with pytest.raises(ValueError, match='Mark this clip ready'): asyncio.run(run_editor(config))
    saved['candidates'][0]['status'] = 'ready'
    (tmp_path / 'editor-project.json').write_text(json.dumps(saved))
    asyncio.run(run_editor(config))
    assert (tmp_path / 'clip_01.mp4').exists() and (tmp_path / 'editor-source.mp4').read_bytes() == b'original'
    # Deleting exports must not recycle IDs referenced by posts or bank copies.
    for path in tmp_path.glob('clip_*.mp4'):
        path.unlink()
    output = json.loads((tmp_path / 'job_output.json').read_text())
    output.update(clips=[], total_clips=0, next_clip_index=2)
    (tmp_path / 'job_output.json').write_text(json.dumps(output))
    saved = json.loads((tmp_path / 'editor-project.json').read_text())
    saved['candidates'][0].update(exports=[], status='ready')
    (tmp_path / 'editor-project.json').write_text(json.dumps(saved))
    config['revision'] = saved['revision']
    asyncio.run(run_editor(config))
    assert (tmp_path / 'clip_02.mp4').exists() and not (tmp_path / 'clip_00.mp4').exists()
    assert json.loads((tmp_path / 'job_output.json').read_text())['clips'][0]['clip_index'] == 2


def test_short_manual_export_contains_only_the_selected_frames(monkeypatch, tmp_path):
    """Exercise the full export path: six red frames, six blue, no deleted green gap."""
    import shutil
    import subprocess
    from dataclasses import asdict
    import numpy as np
    from clip_engine.services import rendering_service
    from tests.test_camera_scan import encoder_args

    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg and FFprobe are needed for the actual export check')
    renderer = RenderingService()
    monkeypatch.setattr(renderer.settings, 'local_mode', True)
    renderer._verify_ffmpeg()
    monkeypatch.setattr(rendering_service, 'get_output_dimensions', lambda _: (80, 120))
    frames = np.zeros((180, 90, 160, 3), dtype=np.uint8)
    frames[:36, :, :, 0] = 255
    frames[36:150, :, :, 1] = 255
    frames[150:, :, :, 2] = 255
    source = tmp_path / 'editor-source.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '160x90', '-r', '30',
        '-i', 'pipe:0', *renderer._video_codec_args(160, 90), '-pix_fmt', 'yuv420p', str(source)],
        input=frames.tobytes(), capture_output=True, check=True, timeout=30)
    c = {**candidate(), 'ranges': [[1000, 1200], [5000, 5200]], 'status': 'ready', 'captions': False, 'video_speed': 1}
    project = {'version': 1, 'revision': 0, 'width': 160, 'height': 90, 'duration_ms': 12000, 'aspect_ratio': '9:16',
        'candidates': [c], 'transcript': [asdict(s) for s in transcript()]}
    (tmp_path / 'editor-project.json').write_text(json.dumps(project))
    (tmp_path / 'transcript.json').write_text(json.dumps({'segments': [asdict(s) for s in transcript()]}))
    (tmp_path / 'job_output.json').write_text(json.dumps({'clips': [], 'total_clips': 0, 'editor_project': True}))
    asyncio.run(run_editor({'run': str(tmp_path), 'revision': 0, 'candidate_id': c['id'], 'action': 'export'}))
    output = json.loads((tmp_path / 'job_output.json').read_text())['clips'][0]
    assert output['duration_ms'] == 400
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', output['s3_url'], '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
        capture_output=True, check=True, timeout=30)
    pictures = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(-1, 120, 80, 3)
    assert len(pictures) == 12
    assert np.all(pictures[:6, 60, 40, 0] > 220)
    assert np.all(pictures[6:, 60, 40, 2] > 220)
    assert np.all(pictures[:, 60, 40, 1] < 30)


@pytest.mark.parametrize('status', ['refining', 'discarded', 'ready'])
def test_unready_or_failed_export_never_marks_baked(monkeypatch, tmp_path, status):
    from dataclasses import asdict
    c = {**candidate(), 'status': status}
    project = {'version': 1, 'revision': 0, 'width': 1920, 'height': 1080, 'duration_ms': 12000, 'aspect_ratio': '9:16',
        'candidates': [c], 'transcript': [asdict(s) for s in transcript()]}
    (tmp_path / 'editor-project.json').write_text(json.dumps(project))
    (tmp_path / 'editor-source.mp4').write_bytes(b'original')
    (tmp_path / 'transcript.json').write_text(json.dumps({'segments': [asdict(s) for s in transcript()]}))
    (tmp_path / 'job_output.json').write_text(json.dumps({'clips': [], 'total_clips': 0}))
    monkeypatch.setattr(RenderingService, '_verify_ffmpeg', lambda _: None)
    render = AsyncMock(side_effect=RenderingError('fixture render failure'))
    monkeypatch.setattr(RenderingService, 'render_clip', render)
    with pytest.raises((ValueError, RenderingError), match='fixture render failure|Mark this clip ready'):
        asyncio.run(run_editor({'run': str(tmp_path), 'revision': 0, 'candidate_id': c['id'], 'action': 'export'}))
    assert render.await_count == (1 if status == 'ready' else 0)
    assert json.loads((tmp_path / 'editor-project.json').read_text()) == project
    assert json.loads((tmp_path / 'job_output.json').read_text())['clips'] == []
    assert not list(tmp_path.glob('clip_*.mp4'))


@pytest.mark.parametrize('jev_enabled', [True, False])
@pytest.mark.parametrize('available', [True, False])
def test_pipeline_review_stops_before_automatic_repairs_and_render(monkeypatch, tmp_path, jev_enabled, available):
    from clip_engine.services import ai_clipping_pipeline as module
    from clip_engine.services.ai_clipping_pipeline import AIClippingPipeline, ClippingJobRequest, JobStatus
    from clip_engine.services.intelligence_planner import ClipPlanResponse
    from clip_engine.services.transcription_service import TranscriptionResult
    settings = module.get_settings()
    monkeypatch.setattr(settings, 'local_mode', True)
    monkeypatch.setattr(settings, 'openrouter_api_key', 'fixture')
    monkeypatch.setattr(settings, 'jev_enabled', jev_enabled)
    from clip_engine.services.jev_service import JevService
    from tests.test_editorial_context import response
    calls = []
    async def evaluate(self, state, questions):
        assert self.enabled
        calls.append(questions)
        return {'status': 'success' if available else 'unavailable', 'questions': questions,
                'answers': response(questions)['answers'] if available else {}}
    monkeypatch.setattr(JevService, 'evaluate', evaluate)
    monkeypatch.setattr(settings, 'local_output_dir', str(tmp_path / 'out'))
    monkeypatch.setattr(settings.__class__, 'temp_directory', property(lambda self: str(tmp_path / 'work')))
    monkeypatch.setattr(RenderingService, '_verify_ffmpeg', lambda _: None)
    pipeline = AIClippingPipeline()
    source = tmp_path / 'original.mp4'; source.write_bytes(b'source')
    pipeline.source_context_service.build = AsyncMock(return_value={'status': 'metadata_only', 'source': {}, 'brief': None, 'research_status': 'not_applicable', 'citations': [], 'cost_usd': 0, 'cost_incomplete': False, 'requests': []})
    pipeline.video_downloader.download_video = AsyncMock(return_value=SimpleNamespace(video_path=str(source), file_size_bytes=6,
        metadata=SimpleNamespace(title='A manual source', duration_seconds=12, width=1920, height=1080)))
    pipeline.transcription_service.transcribe = AsyncMock(return_value=TranscriptionResult(segments=transcript(), full_text='Original source'))
    pipeline.intelligence_planner.plan_clips = AsyncMock(return_value=ClipPlanResponse(segments=[ClipPlanSegment(0, 5000, .8, summary='First'), ClipPlanSegment(6000, 11000, .7, summary='Second')], total_clips=2))
    pipeline.rendering_service._get_video_dimensions = AsyncMock(return_value=(1920, 1080))
    async def preview(src, dest, **kwargs): Path(dest).write_bytes(b'preview')
    pipeline.rendering_service.capture_framing_source = AsyncMock(side_effect=preview)
    pipeline.rendering_service.render_clip = AsyncMock(side_effect=AssertionError('Review must not render'))
    monkeypatch.setattr(module.CoherenceReviewer, 'prepare', AsyncMock(side_effect=AssertionError('Review must not repair')))
    result = asyncio.run(pipeline.process_video(ClippingJobRequest(video_url=str(source), job_id='review-run', workflow='review', layout_style='fit')))
    assert result.status == JobStatus.COMPLETED, result.error
    assert result.output.editor_project and result.output.total_clips == 0
    project = json.loads((tmp_path / 'out/review-run/editor-project.json').read_text())
    assert len(project['candidates']) == 2
    assert all(len(c['review']['questions']) == 8 for c in project['candidates'])
    assert calls
    assert settings.jev_enabled is jev_enabled
    assert all(c['review']['decision'] == ('passes' if available else 'needs_attention') for c in project['candidates'])
    audit = json.loads((tmp_path / 'out/review-run/edit_audit.json').read_text())
    assert audit['jev_enabled'] is True
    assert source.exists() and not (tmp_path / 'work/review-run').exists()
    pipeline.rendering_service.render_clip.assert_not_called()


@pytest.mark.parametrize('ranges', [None, 'bad', [[0]], [[0, 100, 200]], [[-1, 1000]], [[0, 12001]],
    [[0, 99]], [[1000, 1000]], [[2000, 3000], [1000, 2000]], [[0, 2000], [1000, 3000]],
    [[float('nan'), 1000]], [[0, float('inf')]], [[True, 1000]], [[i * 100, i * 100 + 100] for i in range(201)]])
def test_invalid_caption_suppression_is_rejected(ranges):
    with pytest.raises(ValueError, match='suppression'):
        validate_candidate({**candidate(), 'caption_suppression_ranges': ranges}, 12000)


@pytest.mark.parametrize('speed', [1, 2])
def test_caption_suppression_pixels_follow_source_cuts_and_speed(monkeypatch, tmp_path, speed):
    """Actual captions vanish mid-event, including linger; source pixels survive."""
    import shutil
    import subprocess
    import numpy as np
    from clip_engine.config import get_caption_preset
    from clip_engine.services import rendering_service
    from tests.test_camera_scan import encoder_args

    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg and FFprobe are needed for the caption suppression render check')
    renderer = RenderingService()
    monkeypatch.setattr(renderer.settings, 'local_mode', True)
    renderer._verify_ffmpeg()
    monkeypatch.setattr(rendering_service, 'get_output_dimensions', lambda _: (160, 240))
    source = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
        'color=c=black:s=160x240:r=30:d=6,drawbox=x=10:y=10:w=30:h=10:color=yellow:t=fill',
        *encoder_args(), '-pix_fmt', 'yuv420p', str(source)], check=True, capture_output=True, timeout=30)
    c = {**candidate(), 'ranges': [[1000, 3000], [4000, 6000]],
        'scenes': [{'at_ms': 0, 'layout': 'fill', 'crops': [[0, 0, 1, 1]]}],
        'caption_suppression_ranges': [[0, 1500], [2500, 4500], [5500, 6000]]}
    validate_candidate(c, 6000)
    style = get_caption_preset('sweep')
    style.font_size = 24
    transcript = [TranscriptSegment(0, 6000, 'Hello world', words=[
        TranscriptWord('Hello', 0, 2000), TranscriptWord('world', 2000, 6000)])]
    request = RenderRequest(video_path=str(source), output_path=str(tmp_path / 'out.mp4'),
        start_time_ms=1000, end_time_ms=6000, source_width=160, source_height=240,
        transcript_segments=transcript, caption_style=style, apply_padding=False, video_speed=speed,
        manual_ranges_ms=c['ranges'], manual_plan=manual_plan({'width': 160, 'height': 240}, c),
        caption_suppression_ranges_ms=c['caption_suppression_ranges'])
    result = asyncio.run(renderer.render_clip(request))
    assert result.duration_ms == 4000 // speed
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', result.output_path, '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', 'pipe:1'], check=True, capture_output=True, timeout=30)
    frames = np.frombuffer(raw.stdout, dtype=np.uint8).reshape(-1, 240, 160, 3)
    for ms, hidden in [(200, True), (800, False), (1800, True), (2300, True), (2800, False), (3800, True)]:
        frame = frames[round(ms / speed / 1000 * 30)]
        # A yellow source mark represents captions already baked into the source.
        assert frame[15, 20, 0] > 200 and frame[15, 20, 1] > 200 and frame[15, 20, 2] < 50
        caption_pixels = np.count_nonzero(frame[40:, :, :].max(axis=2) > 100)
        assert (caption_pixels == 0) if hidden else (caption_pixels > 5), (ms, hidden, caption_pixels)


@pytest.mark.parametrize('count', [99, 100, 200])
def test_caption_suppression_many_sections_render_with_ffmpeg(tmp_path, count):
    """Exercise the parser limit and every accepted section using the actual graph."""
    import shutil
    import subprocess
    import numpy as np
    from clip_engine.services.clip_editor import TimeMap

    if not shutil.which('ffmpeg'):
        pytest.skip('FFmpeg is needed for the caption suppression render check')
    duration_ms = count * 200
    intervals = [[i * 200, i * 200 + 100] for i in range(count)]
    c = {**candidate(), 'ranges': [[0, duration_ms]],
        'scenes': [{'at_ms': 0, 'layout': 'fill', 'crops': [[0, 0, 1, 1]]}],
        'caption_suppression_ranges': intervals}
    validate_candidate(c, duration_ms)
    captions = tmp_path / 'captions.ass'
    captions.write_text('''[Script Info]
ScriptType: v4.00+
PlayResX: 160
PlayResY: 240
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, Alignment
Style: Default,Arial,24,&H00FFFFFF,2
[Events]
Format: Layer, Start, End, Style, Text
Dialogue: 0,0:00:00.00,0:01:00.00,Default,CAPTIONS
''')
    renderer = RenderingService()
    graph = '[0:v]null[base]' + renderer._caption_graph(
        str(captions), intervals, 0, TimeMap([(0, duration_ms)]))
    result = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
        f'color=c=black:s=160x240:r=20:d={duration_ms / 1000},drawbox=x=10:y=10:w=30:h=10:color=yellow:t=fill',
        '-filter_complex', graph, '-map', '[captioned]', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
        capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 240, 160, 3)
    assert len(frames) == count * 4
    for i in range(count):
        # Sample halfway through each exclusion and the following visible gap,
        # including across expression-group boundaries and the last section.
        for offset, hidden in [(1, True), (3, False)]:
            frame = frames[i * 4 + offset]
            assert frame[15, 20, 0] > 200 and frame[15, 20, 1] > 200 and frame[15, 20, 2] < 50
            caption_pixels = np.count_nonzero(frame[40:, :, :].max(axis=2) > 100)
            assert (caption_pixels == 0) if hidden else (caption_pixels > 5), (i, hidden, caption_pixels)


@pytest.mark.parametrize('y', [True, '0.5', float('nan'), float('inf'), .09, .91, {}])
def test_invalid_caption_position_is_rejected(y):
    with pytest.raises(ValueError, match='caption position'):
        validate_candidate({**candidate(), 'caption_y': y}, 12000)


def export_fixture(tmp_path, monkeypatch, render):
    from dataclasses import asdict
    c = {**candidate(), 'status': 'ready', 'baked_hash': 'a' * 64}
    project = {'version': 1, 'revision': 5, 'width': 1920, 'height': 1080, 'duration_ms': 12000, 'aspect_ratio': '9:16',
        'candidates': [c], 'transcript': [asdict(s) for s in transcript()]}
    (tmp_path / 'editor-project.json').write_text(json.dumps(project))
    (tmp_path / 'editor-source.mp4').write_bytes(b'original')
    (tmp_path / 'transcript.json').write_text(json.dumps({'segments': [asdict(s) for s in transcript()]}))
    (tmp_path / 'job_output.json').write_text(json.dumps({'clips': [], 'total_clips': 0, 'editor_project': True}))
    monkeypatch.setattr(RenderingService, '_verify_ffmpeg', lambda _: None)
    monkeypatch.setattr(RenderingService, 'render_clip', render)
    return {'run': str(tmp_path), 'revision': 5, 'candidate_id': c['id'], 'action': 'export'}


def test_export_killed_between_library_and_project_commits_finishes_without_a_duplicate(monkeypatch, tmp_path):
    from clip_engine.services import manual_editor
    from clip_engine.services.rendering_service import RenderResult
    async def render(self, request):
        Path(request.output_path).write_bytes(b'final clip')
        return RenderResult(request.output_path, 10, 4000, layout_type='two_shot')
    render = AsyncMock(side_effect=render)
    config = export_fixture(tmp_path, monkeypatch, lambda self, request: render(self, request))
    real = manual_editor.atomic_json
    def killed(path, value):
        if path.name == 'editor-project.json':
            raise KeyboardInterrupt('SIGKILL stand-in')
        real(path, value)
    monkeypatch.setattr(manual_editor, 'atomic_json', killed)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(run_editor(config))
    assert json.loads((tmp_path / 'editor-project.json').read_text())['candidates'][0]['status'] == 'ready'
    monkeypatch.setattr(manual_editor, 'atomic_json', real)
    asyncio.run(run_editor(config))
    assert render.await_count == 1
    output = json.loads((tmp_path / 'job_output.json').read_text())
    assert [x['clip_index'] for x in output['clips']] == [0] and not (tmp_path / 'clip_01.mp4').exists()
    saved = json.loads((tmp_path / 'editor-project.json').read_text())
    c = saved['candidates'][0]
    # A new bake replaces the render that undoing "Refine again" could restore.
    assert (c['status'], c['exports'], saved['revision']) == ('baked', [0], 6) and 'baked_hash' not in c
    # A later, changed edit renders again rather than reusing that export.
    c['status'] = 'ready'
    (tmp_path / 'editor-project.json').write_text(json.dumps(saved))
    asyncio.run(run_editor({**config, 'revision': 6}))
    assert render.await_count == 2 and json.loads((tmp_path / 'editor-project.json').read_text())['candidates'][0]['exports'] == [0, 1]


def test_failures_carry_fixed_editor_codes(monkeypatch, tmp_path):
    config = export_fixture(tmp_path, monkeypatch, AsyncMock(side_effect=RenderingError('FFmpeg failed: /private/path')))
    def code(**patch):
        with pytest.raises(BaseException) as error:
            asyncio.run(run_editor({**config, **patch}))
        return getattr(error.value, 'editor_code', None)
    assert code() == 'render_failed'
    assert code(revision=4) == 'project_changed'
    assert code(candidate_id='missing') == 'project_changed'
    project = json.loads((tmp_path / 'editor-project.json').read_text())
    project['candidates'][0]['ranges'] = [[5000, 1000]]
    (tmp_path / 'editor-project.json').write_text(json.dumps(project))
    assert code() == 'invalid_edit'
    project['candidates'][0].update(ranges=candidate()['ranges'], status='refining')
    (tmp_path / 'editor-project.json').write_text(json.dumps(project))
    assert code() == 'not_ready'
    (tmp_path / 'editor-source.mp4').unlink()
    assert code() == 'source_missing'


@pytest.mark.parametrize('source_type', ['youtube', 'local'])
def test_downloads_move_into_the_project_and_local_files_are_copied(tmp_path, source_type):
    gate, _ = reviewer(lambda state, q: False)
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'downloaded')
    out = tmp_path / 'run'; out.mkdir()
    async def preview(src, dest, **kwargs): Path(dest).write_bytes(b'preview')
    renderer = SimpleNamespace(_get_video_dimensions=AsyncMock(return_value=(1920, 1080)),
        capture_framing_source=AsyncMock(side_effect=preview))
    request = SimpleNamespace(aspect_ratio='9:16', layout_style='fit', include_captions=True, caption_preset='pop', video_speed=1)
    asyncio.run(prepare_project(request, [ClipPlanSegment(0, 5000, .9, summary='First')], transcript(),
        SimpleNamespace(video_path=str(source), source_type=source_type, metadata=SimpleNamespace(title='T', duration_seconds=12)),
        renderer, gate, str(out), lambda *_: None))
    assert (out / 'editor-source.mp4').read_bytes() == b'downloaded'
    assert source.exists() == (source_type == 'local')
    # Windows reports DOS attributes here, not POSIX owner/group permissions.
    if os.name != 'nt':
        assert (out / 'editor-source.mp4').stat().st_mode & 0o777 == 0o600


def test_editor_preview_is_encoded_near_3_mbps_at_720p(monkeypatch, tmp_path):
    commands = []
    renderer = RenderingService.__new__(RenderingService)
    renderer.settings = SimpleNamespace(local_mode=True, ffmpeg_preset='fast', ffmpeg_crf=20)
    renderer._local_cpu_encoder = 'libopenh264'
    for fps, expected in (('30', '3M'), ('60', '4.5M')):
        monkeypatch.setattr(renderer, '_get_video_dimensions', AsyncMock(return_value=(3840, 2160)), raising=False)
        monkeypatch.setattr(renderer, '_probe_fps', AsyncMock(return_value=fps), raising=False)
        async def run(cmd, **_):
            commands.append(cmd)
            Path(cmd[-1]).write_bytes(b'')
        monkeypatch.setattr(renderer, '_run_cmd', run, raising=False)
        asyncio.run(renderer.capture_framing_source('in.mp4', str(tmp_path / f'preview-{fps}.mp4')))
        cmd = commands[-1]
        assert 'scale=1280:720,setsar=1' in cmd and cmd[cmd.index('-b:v') + 1] == expected

def test_titles_and_reasons_truncate_by_utf16_units_without_splitting_emoji(tmp_path):
    """The UI counts UTF-16 units; one emoji at the limit must not break the project."""
    from clip_engine.services.manual_editor import utf16_prefix
    assert utf16_prefix('ab😀', 3) == 'ab' and utf16_prefix('ab😀', 4) == 'ab😀' and utf16_prefix('', 5) == ''
    gate, _ = reviewer(lambda state, q: False)
    source = tmp_path / 'original.mov'
    source.write_bytes(b'source')
    out = tmp_path / 'run'; out.mkdir()
    async def preview(src, dest, **kwargs): Path(dest).write_bytes(b'preview')
    renderer = SimpleNamespace(_get_video_dimensions=AsyncMock(return_value=(1920, 1080)),
        capture_framing_source=AsyncMock(side_effect=preview))
    request = SimpleNamespace(aspect_ratio='9:16', layout_style='fit', include_captions=True, caption_preset='pop', video_speed=1)
    segments = [ClipPlanSegment(0, 5000, .9, summary='a' * 199 + '😀')]
    segments[0].reasoning = 'r' * 3999 + '👍🏽'
    project = asyncio.run(prepare_project(request, segments, transcript(), SimpleNamespace(video_path=str(source),
        metadata=SimpleNamespace(title='t' * 1023 + '😀', duration_seconds=12)), renderer, gate, str(out), lambda *_: None))
    c = project['candidates'][0]
    assert c['title'] == 'a' * 199 and c['reason'] == 'r' * 3999 and project['title'] == 't' * 1023
    validate_candidate(c, 12000)



def test_freed_editor_media_refuses_every_operation(monkeypatch, tmp_path):
    render = AsyncMock()
    config = export_fixture(tmp_path, monkeypatch, render)
    project = json.loads((tmp_path / 'editor-project.json').read_text())
    project['media_freed'] = True
    (tmp_path / 'editor-project.json').write_text(json.dumps(project))
    for action in ('export', 'scan-cameras', 'review', 'replace-source'):
        with pytest.raises(ValueError) as error:
            asyncio.run(run_editor({**config, 'action': action, 'source_id': 'a' * 32}))
        assert error.value.editor_code == 'source_missing'
    render.assert_not_called()
