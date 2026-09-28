"""General final Planner protocol and strict current-project identity tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from catalog_test_helpers import evidence, prepared
from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.analysis.finalizer.catalog import CatalogBuilder
from toolrecap_v4.analysis.finalizer.planner import PlannerDraft
from toolrecap_v4.analysis.finalizer.season_plan import (
    FINAL_PLAN_PROTOCOL, SeasonPlanService, final_response_contract, parse_final_planner_response,
)
from toolrecap_v4.analysis.vision.models import VisualEvidence, VisualRunResult
from toolrecap_v4.errors import FinalPlannerValidationError
from toolrecap_v4.gateway import GatewayResult


def _case(root: Path, episode_count: int, *, visual_count: int = 0):
    project_id = "synthetic-project"
    revision = "evr-generic"
    store = EvidenceStore(root, project_id)
    episodes = []
    for index in range(1, episode_count + 1):
        episode_id = f"E{index:02d}"
        source_id = f"src_{index:03d}"
        episodes.append(prepared(episode_id, source_id, empty=index % 3 == 0))
        items = () if index % 3 == 0 else (evidence(
            f"{episode_id}-EV-001", source_id, 1000, 3000, f"Synthetic event {index}."
        ),)
        store.save_episode(revision, episode_id, items, dependency_digest=f"dep-{episode_id}")
    store.commit_revision(revision, [episode.episode_id for episode in episodes], dependency_signature={"synthetic": True})
    catalog = CatalogBuilder().build(
        project_id=project_id, evidence_revision=revision,
        ordered_episodes=episodes, evidence_store=store,
    )
    visuals = tuple(VisualEvidence(
        f"E01-VIS-{index:03d}", project_id, "E01", episodes[0].source_id,
        "VR-E01-001", 1000, 3000, (f"E01-FR-{index:04d}",), (1000,),
        "A synthetic visible action.", (), (), (), (), ("E01-EV-001",), {},
    ) for index in range(1, visual_count + 1))
    visual = VisualRunResult("vis-generic", (), visuals, {"complete": True, "failed": 0, "canceled": 0}, 0, 0)
    draft = PlannerDraft("planner-draft-v1", "Synthetic rationale.", 0, (), ())
    return project_id, catalog, draft, visual


def _response(project_id, catalog, visual, outputs):
    return {
        "protocol_version": FINAL_PLAN_PROTOCOL,
        "action": "FINAL_SEASON_PLAN_DRAFT",
        "project_id": project_id,
        "catalog_hash": catalog.catalog_hash,
        "evidence_revision": catalog.evidence_revision,
        "visual_revision": visual.visual_revision,
        "outputs": outputs,
    }


def _output(index, episodes, catalog, visual, *, cross=False):
    selected = [episodes[0], episodes[-1]] if cross else [episodes[0]]
    selected = list(dict.fromkeys(selected))
    mapping = {episode.episode_id: episode.source_id for episode in catalog.ordered_episodes}
    allowed = {item.evidence_id for item in catalog.items}
    evidence_ids = [f"{episode_id}-EV-001" for episode_id in selected if f"{episode_id}-EV-001" in allowed]
    visual_ids = [item.visual_evidence_id for item in visual.evidence] if "E01" in selected else []
    return {
        "planner_ref": f"draft_output_{index:03d}", "title_concept": f"Concept {index}",
        "editorial_thesis": "Stay factual.", "story_arc": "An ordered synthetic story.",
        "episode_ids": selected, "evidence_ids": evidence_ids,
        "visual_evidence_ids": visual_ids,
        "source_ranges": [
            {"episode_id": episode_id, "source_id": mapping[episode_id], "start_ms": 1000, "end_ms": 2000}
            for episode_id in selected
        ],
        "uncertainty": [], "writer_brief": {"angle": "grounded"},
    }


class Gateway:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def submit_text_chat(self, **kwargs):
        self.calls.append(kwargs)
        assert "images" not in kwargs
        response = self.responses.pop(0)
        raw = response if isinstance(response, str) else json.dumps(response)
        return GatewayResult(raw_response=raw, bytes_sent=len(kwargs["prompt"].encode()), metadata={"status_code": 200})


@pytest.mark.parametrize("episode_count,output_count", [(1, 0), (1, 1), (3, 2), (10, 6), (21, 2)])
def test_current_project_counts_and_app_owned_order(tmp_path, episode_count, output_count):
    project_id, catalog, draft, visual = _case(tmp_path, episode_count)
    episode_ids = [episode.episode_id for episode in catalog.ordered_episodes]
    outputs = [_output(index, episode_ids, catalog, visual, cross=episode_count > 1 and index == 2)
               for index in range(1, output_count + 1)]
    gateway = Gateway([_response(project_id, catalog, visual, outputs)])
    result = SeasonPlanService(gateway, tmp_path, "model").run(
        project_id=project_id, raw_recap_prompt="Raw  prompt 🙂", catalog=catalog, draft=draft, visual=visual,
    )
    assert [item["output_id"] for item in result.plan.outputs] == [f"out_{index:03d}" for index in range(1, output_count + 1)]
    assert [item["title_concept"] for item in result.plan.outputs] == [f"Concept {index}" for index in range(1, output_count + 1)]
    assert len(final_response_contract(catalog, visual)["allowed_episode_source_mapping"]) == episode_count
    assert result.path.is_file()
    assert SeasonPlanService(gateway, tmp_path, "model").run(
        project_id=project_id, raw_recap_prompt="Raw  prompt 🙂", catalog=catalog, draft=draft, visual=visual,
    ).reused


def test_zero_evidence_episode_and_multiple_visual_ids(tmp_path):
    project_id, catalog, _draft, visual = _case(tmp_path, 3, visual_count=2)
    service = SeasonPlanService(Gateway([]), tmp_path, "model")
    output = _output(1, ["E01", "E03"], catalog, visual, cross=True)
    output["episode_ids"] = ["E01", "E03"]
    response = _response(project_id, catalog, visual, [output])
    assert service._validate(response, project_id, catalog, visual)[0]["visual_evidence_ids"] == ["E01-VIS-001", "E01-VIS-002"]
    assert len(catalog.ordered_episodes) == 3
    assert catalog.ordered_episodes[2].evidence_count == 0


@pytest.mark.parametrize("mutation,issue", [
    (lambda out: out["evidence_ids"].append("E99-EV-999"), "outputs[0].evidence_ids"),
    (lambda out: out["visual_evidence_ids"].append("E99-VIS-999"), "outputs[0].visual_evidence_ids"),
    (lambda out: out["source_ranges"][0].update(source_id="invented-source"), "outputs[0].source_ranges[0].source_id"),
    (lambda out: out["episode_ids"].append("E99"), "outputs[0].episode_ids"),
    (lambda out: out["source_ranges"][0].update(end_ms=100_001), "outputs[0].source_ranges[0].bounds"),
])
def test_invented_identity_and_range_are_rejected(tmp_path, mutation, issue):
    project_id, catalog, _draft, visual = _case(tmp_path, 3, visual_count=2)
    output = _output(1, ["E01"], catalog, visual)
    mutation(output)
    with pytest.raises(FinalPlannerValidationError) as error:
        SeasonPlanService(Gateway([]), tmp_path, "model")._validate(
            _response(project_id, catalog, visual, [output]), project_id, catalog, visual,
        )
    assert issue in error.value.issue_codes


def test_real_failure_structure_is_rejected_then_bounded_repair_succeeds(tmp_path):
    project_id, catalog, draft, visual = _case(tmp_path, 1)
    output = _output(1, ["E01"], catalog, visual)
    wrong_structure = {
        "protocol_version": "planner-draft-v1", "action": "FINAL_SEASON_PLAN_DRAFT",
        "project_id": project_id,
        "catalog_identity": {"catalog_hash": catalog.catalog_hash, "evidence_revision": catalog.evidence_revision},
        "final_plan": {"concepts": [output]},
    }
    valid = _response(project_id, catalog, visual, [output])
    gateway = Gateway([wrong_structure, valid])
    service = SeasonPlanService(gateway, tmp_path, "model", repair_attempts=1)
    result = service.run(project_id=project_id, raw_recap_prompt="raw", catalog=catalog, draft=draft, visual=visual)
    assert result.plan.outputs[0]["output_id"] == "out_001"
    assert [call["phase"] for call in gateway.calls] == ["final_planner_refinement", "final_planner_refinement_repair"]
    repair_payload = json.loads(gateway.calls[1]["prompt"])
    assert "root.fields" in repair_payload["validation_issue_codes"]
    assert repair_payload["response_contract"]["allowed_evidence_ids"] == ["E01-EV-001"]
    raw_files = sorted(result.path.parent.glob("attempt-*.raw.txt"))
    assert len(raw_files) == 2
    for raw_file in raw_files:
        manifest = json.loads(raw_file.with_name(raw_file.name.replace(".raw.txt", ".raw.manifest.json")).read_text())
        assert manifest["response_hash"] == hashlib.sha256(raw_file.read_bytes()).hexdigest()


def test_transport_normalization_does_not_change_semantic_validation(tmp_path):
    project_id, catalog, _draft, visual = _case(tmp_path, 1)
    valid = _response(project_id, catalog, visual, [_output(1, ["E01"], catalog, visual)])
    fenced = "```json\n" + json.dumps(valid) + "\n```"
    wrapped = {"choices": [{"message": {"content": json.dumps(valid)}}]}
    assert parse_final_planner_response(fenced) == valid
    assert parse_final_planner_response(json.dumps(wrapped)) == valid
    invalid = dict(valid)
    invalid["catalog_hash"] = "invented"
    with pytest.raises(FinalPlannerValidationError) as error:
        SeasonPlanService(Gateway([]), tmp_path, "model")._validate(
            parse_final_planner_response(json.dumps({"choices": [{"message": {"content": json.dumps(invalid)}}]})),
            project_id, catalog, visual,
        )
    assert "root.catalog_hash" in error.value.issue_codes
    with pytest.raises(FinalPlannerValidationError) as duplicate_error:
        parse_final_planner_response('{"project_id":"one","project_id":"two"}')
    assert "duplicate_json_key" in duplicate_error.value.issue_codes
