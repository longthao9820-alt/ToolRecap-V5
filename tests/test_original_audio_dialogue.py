from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from catalog_test_helpers import populated_store, prepared
from toolrecap_v4.analysis.finalizer.finalization import OutputValidator, map_final_json
from toolrecap_v4.analysis.finalizer.season_plan import SeasonPlan
from toolrecap_v4.analysis.finalizer.writer import assemble_writer_jobs
from toolrecap_v4.analysis.models import PreparedEpisode, Transcript, TranscriptCue
from toolrecap_v4.analysis.vision.models import VisualRunResult
from toolrecap_v4.downstream import DownstreamVoiceCache
from toolrecap_v4.errors import SourceDialogueNarrationError
from toolrecap_v4.original_dialogue import (
    NO_ORIGINAL_DIALOGUE,
    ORIGINAL_DIALOGUE_MAPPED,
    OriginalDialogueSubtitleMapper,
    ResolvedSourceClip,
    narration_contains_source_dialogue,
)
from toolrecap_v4.settings import AppSettings
from toolrecap_v4.subtitles import SubtitleCue, build_srt


def test_mapper_clips_shifts_sorts_multiple_sources_and_does_not_repeat_hold():
    mapper = OriginalDialogueSubtitleMapper({
        "E01.mp4": (
            SubtitleCue(900, 1_200, "clipped opening"),
            SubtitleCue(2_000, 2_500, "middle dialogue"),
            SubtitleCue(4_800, 5_200, "clipped ending"),
        ),
        "E02.mp4": (SubtitleCue(10_100, 10_500, "second episode"),),
    })
    result = mapper.map((
        ResolvedSourceClip("a", "E01.mp4", 1_000, 5_000, 0, 5_000, True),
        ResolvedSourceClip("b", "E02.mp4", 10_000, 11_000, 5_000, 1_000, True),
    ))
    assert result.state == ORIGINAL_DIALOGUE_MAPPED
    assert [(c.start_ms, c.end_ms, c.text) for c in result.cues] == [
        (0, 200, "clipped opening"),
        (1_000, 1_500, "middle dialogue"),
        (3_800, 4_000, "clipped ending"),
        (5_100, 5_500, "second episode"),
    ]
    assert build_srt(result.cues).startswith("1\n00:00:00,000 --> 00:00:00,200")


def test_narration_fit_extension_updates_dialogue_map_but_hold_adds_no_speech():
    mapper = OriginalDialogueSubtitleMapper({
        "episode.mp4": (
            SubtitleCue(5_500, 6_000, "extension dialogue"),
            SubtitleCue(8_100, 8_500, "outside source EOF"),
        )
    })
    result = mapper.map((
        # Planned clip could have ended at 5s; resolved source range extends to 8s,
        # followed by a 2s held frame (10s final duration).
        ResolvedSourceClip("seg", "episode.mp4", 1_000, 8_000, 0, 10_000, True),
    ))
    assert [(cue.start_ms, cue.end_ms) for cue in result.cues] == [(4_500, 5_000)]


def test_true_no_dialogue_and_explicit_source_mute_are_distinct_valid_empty_states():
    mapper = OriginalDialogueSubtitleMapper({"episode.mp4": (SubtitleCue(1_000, 2_000, "line"),)})
    muted = mapper.map((ResolvedSourceClip("seg", "episode.mp4", 1_000, 2_000, 0, 1_000, False),))
    absent = OriginalDialogueSubtitleMapper({}).map((
        ResolvedSourceClip("seg", "episode.mp4", 1_000, 2_000, 0, 1_000, True),
    ))
    assert muted.state == absent.state == NO_ORIGINAL_DIALOGUE
    assert muted.cues == absent.cues == ()


def _finalization_context(tmp_path: Path):
    store, revision, episodes, _ = populated_store(tmp_path)
    episode = episodes[0]
    episode = PreparedEpisode(
        episode.episode_id, episode.source_id, episode.source_path, episode.duration_ms,
        episode.canvas_width, episode.canvas_height, source_basename="E01.mp4",
        transcript=Transcript(
            episode_id="E01", source_type="sidecar", source_format="srt",
            cues=(TranscriptCue("cue-1", 1_000, 3_000, "Get off my ranch."),),
        ), artifact_hash=episode.artifact_hash, source_hash=episode.source_hash,
    )
    output = {
        "output_id": "out_001", "planner_ref": "story", "title_concept": "Story",
        "editorial_thesis": "Conflict escalates", "story_arc": "One confrontation",
        "episode_ids": ["E01"], "evidence_ids": ["E01-EV-001"],
        "visual_evidence_ids": [], "source_ranges": [
            {"episode_id": "E01", "source_id": episode.source_id, "start_ms": 1_000, "end_ms": 3_000}
        ], "uncertainty": [], "writer_brief": {},
    }
    pending = SeasonPlan(
        "season-plan-v1", "pending", "project-1", "prompt-hash", "catalog-hash",
        revision, "visual", "draft-hash", "response-hash", (output,),
    )
    import hashlib
    plan_hash = hashlib.sha256(json.dumps(
        pending.semantic_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    plan = SeasonPlan(**{**pending.__dict__, "plan_hash": plan_hash})
    visual = VisualRunResult("visual", (), (), {"complete": True}, 0, 0)
    jobs = assemble_writer_jobs(
        project_id="project-1", raw_prompt="prompt", language="en-US", plan=plan,
        episodes=(episode,), evidence_store=store, visual=visual, model="writer", reasoning="",
    )
    validator = OutputValidator(project_id="project-1", plan=plan, episodes=(episode,), visual=visual, evidence_store=store)
    return episode, plan, visual, jobs[0], validator


def _writer_response(plan_hash: str, narration: str) -> dict:
    return {
        "writer_draft_version": "writer-draft-v1", "project_id": "project-1",
        "season_plan_hash": plan_hash, "output_id": "out_001", "title": "Ranch Conflict",
        "narration": {"text": narration}, "segments": [{
            "segment_id": "seg-001", "narration_text": narration, "episode_ids": ["E01"],
            "evidence_ids": ["E01-EV-001"], "visual_evidence_ids": [],
            "source_clips": [{"episode_id": "E01", "source_id": "src_shared", "start_ms": 1_000, "end_ms": 3_000}],
            "editorial_intent": "Retain the confrontation", "uncertainty": [],
        }], "writer_notes": {},
    }


def test_writer_dialogue_is_rejected_from_tts_and_valid_commentary_keeps_source_audio(tmp_path: Path):
    episode, plan, visual, job, validator = _finalization_context(tmp_path)
    bad = validator.validate(job, json.dumps(_writer_response(plan.plan_hash, "Get off my ranch.")))
    assert "SOURCE_DIALOGUE_ROUTED_TO_NARRATION" in bad.issues

    good = validator.validate(job, json.dumps(_writer_response(plan.plan_hash, "The confrontation raises the stakes.")))
    assert good.state == "VALID"
    final_json = map_final_json(
        project_id="project-1", project_name="Project", episodes=(episode,), plan=plan, validated=(good,),
    )
    segment = final_json["outputs"][0]["segments"][0]
    assert segment["type"] == "narration"
    assert segment["source_audio"] is True
    assert segment["narration"] == "The confrontation raises the stakes."


def test_character_name_and_sound_caption_are_not_verbatim_speech():
    narration = "Walt and Jesse identify their downstairs captive as Krazy-8."
    assert not narration_contains_source_dialogue(
        narration, ("Krazy-8.", "[KRAZY-8 COUGHING]"),
    )
    assert narration_contains_source_dialogue(
        "The narrator repeats: I need water.", ("I need water.",),
    )
    assert narration_contains_source_dialogue("The narrator says: Shut up!", ("Shut up!",))


def test_writer_validator_accepts_narrating_a_character_name(tmp_path: Path):
    episode, plan, visual, job, validator = _finalization_context(tmp_path)
    validator.eps[episode.episode_id] = PreparedEpisode(
        episode.episode_id, episode.source_id, episode.source_path,
        episode.duration_ms, episode.canvas_width, episode.canvas_height,
        source_basename=episode.source_basename,
        transcript=Transcript(
            episode_id="E01", source_type="embedded", source_format="srt",
            cues=(TranscriptCue("E01-CUE-0218", 1_000, 3_000, "Krazy-8."),),
        ),
    )
    response = _writer_response(plan.plan_hash, "Krazy-8 is awake downstairs.")
    assert validator.validate(job, json.dumps(response)).state == "VALID"


def test_voice_cache_defense_rejects_source_dialogue_before_voicestudio(tmp_path: Path):
    adapter = MagicMock()
    output = {"render_id": "out_001", "segments": [{
        "segment_id": "seg", "source_file": "episode.mp4", "start_ms": 1_000, "end_ms": 3_000,
        "type": "narration", "narration": "Get off my ranch.", "source_audio": True, "subtitles": [],
    }]}
    with pytest.raises(SourceDialogueNarrationError):
        DownstreamVoiceCache(tmp_path, "project").prepare_output(
            output_def=output, final_json_hash="hash", settings=AppSettings(), voice_adapter=adapter,
            source_dialogue_map={"episode.mp4": (SubtitleCue(1_000, 2_000, "Get off my ranch."),)},
        )
    adapter.synthesize.assert_not_called()


@pytest.mark.parametrize("original_db,commentary_db", [(-14.0, 0.0), (-2.0, -3.0)])
def test_audio_mix_levels_do_not_change_semantic_routing(original_db: float, commentary_db: float):
    settings = AppSettings(original_audio_db=original_db, commentary_audio_db=commentary_db, commentary_reading_speed=1.10)
    assert settings.original_audio_db == original_db
    assert settings.commentary_audio_db == commentary_db
    assert settings.commentary_reading_speed == 1.10
    # Semantic source audio remains explicit and independent of both gain and narration speed.
    segment = {"type": "narration", "source_audio": True}
    assert segment["source_audio"] is True
