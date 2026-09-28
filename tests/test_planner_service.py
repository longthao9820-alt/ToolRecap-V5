from __future__ import annotations

import json
from pathlib import Path

import pytest

from catalog_test_helpers import populated_store
from test_planner_protocol import draft_response
from toolrecap_v4.analysis.finalizer.catalog import CatalogBuilder
from toolrecap_v4.analysis.finalizer.planner import PLANNER_PROTOCOL_VERSION
from toolrecap_v4.analysis.finalizer.planner_service import PlannerConfig, PlannerService
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError, PlannerCapacityError, PlannerRepairExhaustedError
from toolrecap_v4.gateway import GatewayResult


def request_response(catalog, round_id, request_index=1):
    return {
        "protocol_version": PLANNER_PROTOCOL_VERSION, "action": "REQUEST_EVIDENCE",
        "project_id": "project-1", "catalog_hash": catalog.catalog_hash,
        "evidence_revision": catalog.evidence_revision, "round_id": round_id,
        "requests": [{
            "request_id": f"{round_id}-request-{request_index:03d}",
            "type": "evidence_ids", "evidence_ids": ["E01-EV-001", "E01-EV-002"],
        }],
    }


class FakeGateway:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def submit_text_chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected Planner call")
        response = self.responses.pop(0)
        raw = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)
        return GatewayResult(
            raw_response=raw, bytes_sent=len(kwargs["prompt"].encode("utf-8")),
            metadata={"status_code": 200, "duration_ms": 2.5},
        )


@pytest.fixture
def env(tmp_path):
    store, revision, episodes, _ = populated_store(tmp_path)
    catalog = CatalogBuilder().build(
        project_id="project-1", evidence_revision=revision,
        ordered_episodes=episodes, evidence_store=store,
    )
    return tmp_path, catalog


def test_planner_can_return_draft_immediately_without_extra_round(env):
    root, catalog = env
    gateway = FakeGateway([draft_response(catalog)])
    result = PlannerService(gateway, root, PlannerConfig("route-id")).run(
        project_id="project-1", raw_recap_prompt="Raw prompt", catalog=catalog,
    )
    assert result.round_count == 1
    assert result.draft.proposed_output_count == 1
    assert result.round_count == 1
    assert len(gateway.calls) == 1
    assert result.request_measurements[0]["capacity_status"] == "UNKNOWN"
    assert result.request_measurements[0]["request_bytes"] > 0


def test_planner_requests_exact_evidence_then_returns_draft(env):
    root, catalog = env
    gateway = FakeGateway([
        request_response(catalog, "round-001"),
        draft_response(catalog, round_id="round-002"),
    ])
    raw_prompt = 'Keep  spacing\nUnicode Việt 🙂\n"quote"'
    result = PlannerService(gateway, root, PlannerConfig("route-id", reasoning="high")).run(
        project_id="project-1", raw_recap_prompt=raw_prompt, catalog=catalog,
    )
    assert result.round_count == 2
    assert len(gateway.calls) == 2
    initial = json.loads(gateway.calls[0]["prompt"])
    followup = json.loads(gateway.calls[1]["prompt"])
    assert initial["raw_recap_prompt"] == raw_prompt
    assert followup["raw_recap_prompt"] == raw_prompt
    assert followup["exact_full_evidence_fetch"]["completeness"]["complete"] is True
    assert len(followup["exact_full_evidence_fetch"]["items"]) == 2
    assert followup["catalog_transport"]["complete_packed_catalog"]["item_count"] == len(catalog.items)
    assert all(call["phase"] == "season_planner" and "images" not in call for call in gateway.calls)


def test_multiple_evidence_rounds_and_round_limit(env):
    root, catalog = env
    gateway = FakeGateway([
        request_response(catalog, "round-001", 1),
        request_response(catalog, "round-002", 2),
        draft_response(catalog, round_id="round-003"),
    ])
    result = PlannerService(gateway, root, PlannerConfig("route-id", max_rounds=3)).run(
        project_id="project-1", raw_recap_prompt="Prompt", catalog=catalog,
    )
    assert result.round_count == 3
    limited_gateway = FakeGateway([
        request_response(catalog, "round-001", 1),
        draft_response(catalog, round_id="round-002"),
    ])
    finalized = PlannerService(limited_gateway, root, PlannerConfig("limited-route", max_rounds=1)).run(
        project_id="project-1", raw_recap_prompt="Prompt", catalog=catalog,
    )
    assert finalized.round_count == 2
    final_prompt = json.loads(limited_gateway.calls[1]["prompt"])
    assert final_prompt["allowed_actions"] == ["PLANNER_DRAFT"]
    assert "REQUEST_EVIDENCE" not in final_prompt["response_contract"]


def test_planner_carries_all_fetched_evidence_and_resumes_after_repeated_final_request(env):
    root, catalog = env
    first = request_response(catalog, "round-001")
    first["requests"][0]["evidence_ids"] = ["E01-EV-001"]
    second = request_response(catalog, "round-002")
    second["requests"][0]["evidence_ids"] = ["E01-EV-002"]
    third = request_response(catalog, "round-003")
    third["requests"][0]["evidence_ids"] = ["E01-EV-001"]

    class CrashBeforeSynthesis(FakeGateway):
        def submit_text_chat(self, **kwargs):
            if len(self.calls) == 3:
                raise RuntimeError("interrupted before final synthesis")
            return super().submit_text_chat(**kwargs)

    first_gateway = CrashBeforeSynthesis([first, second, third])
    service = PlannerService(first_gateway, root, PlannerConfig("resume-route", max_rounds=3))
    with pytest.raises(RuntimeError, match="interrupted before final synthesis"):
        service.run(project_id="project-1", raw_recap_prompt="Prompt", catalog=catalog)
    third_prompt = json.loads(first_gateway.calls[2]["prompt"])
    assert third_prompt["already_fetched_evidence_ids"] == ["E01-EV-001", "E01-EV-002"]

    resumed_gateway = FakeGateway([draft_response(catalog, round_id="round-004")])
    resumed = PlannerService(resumed_gateway, root, PlannerConfig("resume-route", max_rounds=3)).run(
        project_id="project-1", raw_recap_prompt="Prompt", catalog=catalog,
    )
    assert resumed.round_count == 4
    assert len(resumed_gateway.calls) == 1
    final_prompt = json.loads(resumed_gateway.calls[0]["prompt"])
    assert final_prompt["allowed_actions"] == ["PLANNER_DRAFT"]
    assert final_prompt["evidence_rounds_remaining"] == 0
    assert final_prompt["exact_full_evidence_fetch"]["source_round_ids"] == [
        "round-001", "round-002", "round-003",
    ]
    assert [item["evidence_id"] for item in final_prompt["exact_full_evidence_fetch"]["items"]] == [
        "E01-EV-001", "E01-EV-002",
    ]


def test_final_synthesis_request_is_repaired_before_round_limit(env):
    root, catalog = env
    gateway = FakeGateway([
        request_response(catalog, "round-001"),
        request_response(catalog, "round-002"),
        draft_response(catalog, round_id="round-002"),
    ])
    result = PlannerService(
        gateway, root, PlannerConfig("draft-only-route", max_rounds=1, repair_attempts=1),
    ).run(project_id="project-1", raw_recap_prompt="Prompt", catalog=catalog)
    assert result.round_count == 2
    assert [call["phase"] for call in gateway.calls] == [
        "season_planner", "season_planner", "season_planner_repair",
    ]
    repair = json.loads(gateway.calls[2]["prompt"])
    assert "final_draft_required" in repair["validation_errors"]
    assert "REQUEST_EVIDENCE" not in repair["response_contract"]


def test_final_synthesis_refusal_remains_bounded(env):
    root, catalog = env
    gateway = FakeGateway([
        request_response(catalog, "round-001"),
        request_response(catalog, "round-002"),
        request_response(catalog, "round-002"),
    ])
    with pytest.raises(PlannerRepairExhaustedError, match="final_draft_required"):
        PlannerService(
            gateway, root, PlannerConfig("bounded-route", max_rounds=1, repair_attempts=1),
        ).run(project_id="project-1", raw_recap_prompt="Prompt", catalog=catalog)
    assert len(gateway.calls) == 3


def test_invalid_response_gets_bounded_technical_repair(env):
    root, catalog = env
    gateway = FakeGateway(["{bad", draft_response(catalog)])
    result = PlannerService(gateway, root, PlannerConfig("route-id", repair_attempts=1)).run(
        project_id="project-1", raw_recap_prompt="Prompt", catalog=catalog,
    )
    assert result.draft.proposed_output_count == 1
    assert [call["phase"] for call in gateway.calls] == ["season_planner", "season_planner_repair"]
    repair = json.loads(gateway.calls[1]["prompt"])
    assert "technical protocol defects" in repair["task"]


def test_valid_draft_and_round_are_reused_after_restart(env):
    root, catalog = env
    gateway = FakeGateway([draft_response(catalog)])
    service = PlannerService(gateway, root, PlannerConfig("route-id"))
    first = service.run(project_id="project-1", raw_recap_prompt="Prompt", catalog=catalog)
    second = service.run(project_id="project-1", raw_recap_prompt="Prompt", catalog=catalog)
    assert first.reused is False and second.reused is True
    assert len(gateway.calls) == 1
    assert second.draft == first.draft


def test_raw_response_recovered_after_crash_without_second_ai_call(env, monkeypatch):
    import toolrecap_v4.analysis.finalizer.planner_store as store_module

    root, catalog = env
    gateway = FakeGateway([draft_response(catalog)])
    service = PlannerService(gateway, root, PlannerConfig("route-id"))
    original = store_module.PlannerStore.save_round
    crashed = {"done": False}

    def crash_once(self, *args, **kwargs):
        if not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("crash after raw response")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(store_module.PlannerStore, "save_round", crash_once)
    with pytest.raises(RuntimeError):
        service.run(project_id="project-1", raw_recap_prompt="Prompt", catalog=catalog)
    result = service.run(project_id="project-1", raw_recap_prompt="Prompt", catalog=catalog)
    assert result.draft.proposed_output_count == 1
    assert len(gateway.calls) == 1


def test_completed_evidence_fetch_round_reused_after_restart(env):
    root, catalog = env

    class CrashOnSecond(FakeGateway):
        def submit_text_chat(self, **kwargs):
            if len(self.calls) == 1:
                raise RuntimeError("crash before round 2 response")
            return super().submit_text_chat(**kwargs)

    first_gateway = CrashOnSecond([request_response(catalog, "round-001")])
    service = PlannerService(first_gateway, root, PlannerConfig("route-id"))
    with pytest.raises(RuntimeError, match="crash before round 2"):
        service.run(project_id="project-1", raw_recap_prompt="Resume prompt", catalog=catalog)
    second_gateway = FakeGateway([draft_response(catalog, round_id="round-002")])
    resumed = PlannerService(second_gateway, root, PlannerConfig("route-id")).run(
        project_id="project-1", raw_recap_prompt="Resume prompt", catalog=catalog,
    )
    assert resumed.round_count == 2
    assert len(second_gateway.calls) == 1
    followup = json.loads(second_gateway.calls[0]["prompt"])
    assert followup["prior_round_id"] == "round-001"
    assert followup["exact_full_evidence_fetch"]["completeness"]["complete"] is True


def test_prompt_model_and_reasoning_changes_invalidate_only_planner_session(env):
    root, catalog = env
    gateway = FakeGateway([
        draft_response(catalog),
        draft_response(catalog),
        draft_response(catalog),
        draft_response(catalog),
    ])
    PlannerService(gateway, root, PlannerConfig("route-a", reasoning="low")).run(
        project_id="project-1", raw_recap_prompt="Prompt A", catalog=catalog,
    )
    cached = PlannerService(gateway, root, PlannerConfig("route-a", reasoning="low")).run(
        project_id="project-1", raw_recap_prompt="Prompt A", catalog=catalog,
    )
    assert cached.reused is True
    PlannerService(gateway, root, PlannerConfig("route-a", reasoning="low")).run(
        project_id="project-1", raw_recap_prompt="Prompt B", catalog=catalog,
    )
    PlannerService(gateway, root, PlannerConfig("route-b", reasoning="low")).run(
        project_id="project-1", raw_recap_prompt="Prompt A", catalog=catalog,
    )
    PlannerService(gateway, root, PlannerConfig("route-a", reasoning="high")).run(
        project_id="project-1", raw_recap_prompt="Prompt A", catalog=catalog,
    )
    assert len(gateway.calls) == 4


def test_capacity_preflight_does_not_trim_catalog(env):
    root, catalog = env
    gateway = FakeGateway([draft_response(catalog)])
    with pytest.raises(PlannerCapacityError):
        PlannerService(gateway, root, PlannerConfig("any-model-name", max_request_bytes=1024)).run(
            project_id="project-1", raw_recap_prompt="Prompt", catalog=catalog,
        )
    assert gateway.calls == []


def test_cancellation_creates_no_ready_draft(env):
    root, catalog = env
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CancelledError):
        PlannerService(FakeGateway([]), root, PlannerConfig("route-id")).run(
            project_id="project-1", raw_recap_prompt="Prompt", catalog=catalog,
            cancellation_token=token,
        )
    assert not list((root / "projects" / "project-1" / "planning").rglob("planner_draft.json"))


def test_cancellation_before_evidence_retrieval_does_not_complete_round(env):
    root, catalog = env
    token = CancellationToken()

    class CancelAfterResponse(FakeGateway):
        def submit_text_chat(self, **kwargs):
            result = super().submit_text_chat(**kwargs)
            token.cancel()
            return result

    gateway = CancelAfterResponse([request_response(catalog, "round-001")])
    with pytest.raises(CancelledError):
        PlannerService(gateway, root, PlannerConfig("route-id")).run(
            project_id="project-1", raw_recap_prompt="Cancel fetch", catalog=catalog,
            cancellation_token=token,
        )
    manifests = list((root / "projects" / "project-1" / "planning").rglob("manifest.json"))
    assert all(json.loads(path.read_text())["status"] != "COMPLETE" for path in manifests)
    assert not list(root.rglob("planner_draft.json"))


def test_phase6_never_creates_locked_plan_or_canonical_output_ids(env):
    root, catalog = env
    result = PlannerService(FakeGateway([draft_response(catalog)]), root, PlannerConfig("route-id")).run(
        project_id="project-1", raw_recap_prompt="Prompt", catalog=catalog,
    )
    assert not list(root.rglob("season_plan.json"))
    assert all(not output.draft_ref.startswith("out_") for output in result.draft.proposed_outputs)
