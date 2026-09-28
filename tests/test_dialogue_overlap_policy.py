"""Cross-title regressions for speech detection: no character-name allowlist."""
import json

import pytest

from toolrecap_v4.original_dialogue import narration_contains_source_dialogue, source_dialogue_overlaps
from test_original_audio_dialogue import _finalization_context, _writer_response
from toolrecap_v4.analysis.finalizer.finalization_service import FinalizationConfig, FinalizationService
from toolrecap_v4.analysis.finalizer.writer import WriterArtifact
from toolrecap_v4.gateway import GatewayResult
from toolrecap_v4.analysis.finalizer.writer_store import WriterStore
from toolrecap_v4.analysis.finalizer.writer_service import WriterConfig, WriterService
from toolrecap_v4.validator import sanitize_publication_title, unique_publication_title


@pytest.mark.parametrize("caption,narration", [
    ("Mrs. Driscoll", "Mrs. Driscoll's damaged supplies give Nancy a lead."),
    ("Driscoll's rat.", "Nancy proposes examining Driscoll's rat despite the order."),
    ("Dr. Ada Chen", "Dr. Ada Chen discovers that the report was changed."),
    ("Jean-Luc Picard", "Jean-Luc Picard returns to the bridge."),
    ("Agent María del Río", "Agent María del Río searches the empty office."),
    ("THE SECRET KEY", "The secret key opens a door to another room."),
    ("the next morning", "The next morning, a new witness arrives."),
    ("her father", "She searches for her father throughout the town."),
    ("I think", "The note contains the words I think but the handwriting is unclear."),
    ("New York", "Location: New York."),
    ("Agent Nathan Cole", 'The report identifies him as "Agent Nathan Cole".'),
])
def test_incidental_names_and_short_phrases_do_not_block(caption, narration):
    assert not narration_contains_source_dialogue(narration, [caption])


@pytest.mark.parametrize("caption,narration", [
    ("Get off my ranch.", "Get off my ranch."),
    ("Shut up!", "The narrator says: Shut up!"),
    ("We must leave.", 'He says, "We must leave."'),
    ("I don't know why you came back to this place.", "I don’t know why you came back to this place."),
    ("You have no idea how much trouble we are in.", "The voice repeats you have no idea how much trouble we are in before moving on."),
])
def test_explicit_or_substantial_copied_speech_still_blocks(caption, narration):
    matches = source_dialogue_overlaps(narration, [caption])
    assert any(m["blocking"] for m in matches)
    for match in matches:
        assert narration[match["start_char"]:match["end_char"]] == match["matched_text"]


def test_repair_request_includes_exact_source_span_and_field(tmp_path):
    episode, plan, visual, job, validator = _finalization_context(tmp_path)
    raw = json.dumps(_writer_response(plan.plan_hash, "Get off my ranch."))
    validation = validator.validate(job, raw)
    seen = []
    class Gateway:
        def submit_text_chat(self, **kwargs):
            seen.append(json.loads(kwargs["prompt"]))
            return GatewayResult(json.dumps(_writer_response(plan.plan_hash, "The confrontation raises the stakes.")))
    artifact = WriterArtifact(job.output_id, job.dependency_digest, "responsehash", raw,
                              "PARSED", json.loads(raw), 100, {})
    service = FinalizationService(Gateway(), tmp_path, FinalizationConfig("writer-model"))
    repaired = service._repair(job, artifact, validation, validator, tmp_path/"repair", None)
    assert repaired.state == "VALID"
    match = seen[0]["dialogue_matches"][0]
    assert match["source_text"] == "Get off my ranch."
    assert match["field_path"] == "$.segments[0].narration_text"
    assert match["matched_text"] == "Get off my ranch"
    assert any("Get off my ranch" in item for item in seen[0]["validation_field_paths"])


def test_old_rejection_is_revalidated_without_ai_or_rewriting_raw(tmp_path):
    episode, plan, visual, job, validator = _finalization_context(tmp_path)
    raw = json.dumps(_writer_response(plan.plan_hash, "The confrontation raises the stakes."))
    store = WriterStore(tmp_path, "project-1", plan.plan_hash)
    store.save(job, raw, "PARSED", json.loads(raw), 100, {}, 100000, None,
               "INVALID_REPAIRABLE", ("SOURCE_DIALOGUE_ROUTED_TO_NARRATION",))
    class NoGateway:
        def submit_text_chat(self, **kwargs):
            pytest.fail("Revalidation of a durable response must not spend another AI request")
    artifact, reused, valid = WriterService(NoGateway(), tmp_path, WriterConfig("m"))._run_one(
        job, store, None, validator)
    assert valid and reused and artifact.raw_response == raw
    assert store.load(job) is not None


def test_publication_titles_are_cleaned_without_editorial_ai_and_collisions_stay_unique():
    used = set()
    assert sanitize_publication_title('Machine: Joyce/Hopper?') == "Machine - Joyce - Hopper"
    assert unique_publication_title('Machine: Joyce/Hopper?', used, 1) == "Machine - Joyce - Hopper"
    assert unique_publication_title('Machine? Joyce/Hopper?', used, 2) == "Machine - Joyce - Hopper (2)"
