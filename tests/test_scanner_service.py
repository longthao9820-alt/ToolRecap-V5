from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.analysis.models import PreparedEpisode, Transcript, TranscriptCue
from toolrecap_v4.analysis.scanner import ScannerChunkPolicy, ScannerConfig, ScannerService, compute_evidence_revision
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError, EmptyApiResponseError, ScannerRepairExhaustedError, ScannerResponseError
from toolrecap_v4.gateway import GatewayResult


def _episode(text_suffix: str = "", cue_count: int = 4) -> PreparedEpisode:
    cues = tuple(
        TranscriptCue(f"E01-CUE-{i:03d}", i * 10_000, i * 10_000 + 5000, f"Dialogue {i}{text_suffix}")
        for i in range(cue_count)
    )
    return PreparedEpisode(
        episode_id="E01", source_id="src_001", source_path=Path("C:/secret/path/episode01.mkv"),
        source_basename="episode01.mkv", duration_ms=100_000, canvas_width=1920, canvas_height=1080,
        transcript=Transcript(
            episode_id="E01", source_type="sidecar", source_format="srt", cues=cues,
            provenance_hash=f"transcript{text_suffix}",
        ), artifact_hash=f"prepared{text_suffix}", source_fingerprint="source",
    )


class FakeGateway:
    def __init__(
        self, *, fail_first_chunk: bool = False, always_invalid: bool = False,
        reverse_delays: bool = False, invalid_chunks: set[str] | None = None,
    ):
        self.fail_first_chunk = fail_first_chunk
        self.always_invalid = always_invalid
        self.reverse_delays = reverse_delays
        self.invalid_chunks = invalid_chunks or set()
        self.calls: list[dict] = []
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self.max_active = 0
        self.active = 0

    def submit_text_chat(self, **kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            outer = json.loads(kwargs["prompt"])
            request = outer.get("original_request", outer)
            chunk_id = request["chunk_id"]
            with self._lock:
                self.calls.append(kwargs)
                count = self._counts.get(chunk_id, 0)
                self._counts[chunk_id] = count + 1
            if self.reverse_delays:
                time.sleep(0.015 if chunk_id.endswith("001") else 0.001)
            if self.always_invalid or chunk_id in self.invalid_chunks or (self.fail_first_chunk and chunk_id.endswith("001") and count == 0):
                raw = json.dumps({"episode_id": "WRONG", "source_id": request["source_id"], "chunk_id": chunk_id, "observations": []})
            else:
                part = request["transcript_parts"][0]
                raw = json.dumps({
                    "episode_id": request["episode_id"], "source_id": request["source_id"], "chunk_id": chunk_id,
                    "observations": [{
                        "start_ms": part["start_ms"], "end_ms": part["end_ms"], "category": "dialogue",
                        "observation": f"Fact from {part['cue_id']}",
                        "cue_refs": [{"cue_id": part["cue_id"], "part_index": part["part_index"]}],
                        "entities": ["Minor Character"], "modality": "subtitle", "confidence": None,
                        "uncertainty": ["speaker label absent"],
                    }],
                })
            return GatewayResult(raw_response=raw, bytes_sent=len(kwargs["prompt"].encode()), metadata={"status_code": 200, "duration_ms": 1.5})
        finally:
            with self._lock:
                self.active -= 1


def _config(**changes) -> ScannerConfig:
    values = dict(
        model="provider-neutral-id", reasoning="medium", parallelism=2,
        chunk_policy=ScannerChunkPolicy(max_duration_ms=15_000, max_request_bytes=5000),
        repair_attempts=1,
    )
    values.update(changes)
    return ScannerConfig(**values)


def test_scan_text_only_contract_stable_ids_and_measurements(tmp_path):
    gateway = FakeGateway(reverse_delays=True)
    service = ScannerService(gateway, tmp_path, _config())
    result = service.scan_project("project-1", [_episode()])
    assert gateway.max_active <= 2
    assert result.requested_chunk_count > 1 and result.reused_chunk_count == 0
    assert all(m["phase"] == "scanner" and m["request_bytes"] > 0 for m in result.request_measurements)
    assert all(call["phase"] == "scanner" and "images" not in call for call in gateway.calls)
    bodies = "\n".join(call["prompt"] for call in gateway.calls)
    assert "C:/secret/path" not in bodies and "Recap Prompt" not in bodies and "EditorialPolicy" not in bodies
    store = EvidenceStore(tmp_path, "project-1")
    items = store.get_episode("E01", result.evidence_revision).items
    assert [item.evidence_id for item in items] == [f"E01-EV-{i:03d}" for i in range(1, len(items) + 1)]
    assert all(item.confidence is None and not item.visual_refs for item in items)
    assert all(item.dialogue and item.dialogue[0].text.startswith("Dialogue") for item in items)


def test_restart_reuses_completed_chunks_and_ids(tmp_path):
    gateway = FakeGateway()
    service = ScannerService(gateway, tmp_path, _config())
    first = service.scan_project("project-1", [_episode()])
    calls = len(gateway.calls)
    ids_before = [x.evidence_id for x in EvidenceStore(tmp_path, "project-1").get_episode("E01", first.evidence_revision).items]
    second = service.scan_project("project-1", [_episode()])
    ids_after = [x.evidence_id for x in EvidenceStore(tmp_path, "project-1").get_episode("E01", second.evidence_revision).items]
    assert len(gateway.calls) == calls
    assert second.requested_chunk_count == 0 and second.reused_chunk_count == first.requested_chunk_count
    assert ids_before == ids_after


def test_concurrent_completion_order_does_not_change_evidence_ids(tmp_path):
    episode = _episode()
    slow_first_root = tmp_path / "slow-first"
    normal_root = tmp_path / "normal"
    first = ScannerService(FakeGateway(reverse_delays=True), slow_first_root, _config()).scan_project("project-1", [episode])
    second = ScannerService(FakeGateway(reverse_delays=False), normal_root, _config()).scan_project("project-1", [episode])
    first_items = EvidenceStore(slow_first_root, "project-1").get_episode("E01", first.evidence_revision).items
    second_items = EvidenceStore(normal_root, "project-1").get_episode("E01", second.evidence_revision).items
    assert [(item.evidence_id, item.observation) for item in first_items] == [
        (item.evidence_id, item.observation) for item in second_items
    ]


def test_only_invalid_chunk_is_repaired(tmp_path):
    gateway = FakeGateway(fail_first_chunk=True)
    result = ScannerService(gateway, tmp_path, _config()).scan_project("project-1", [_episode()])
    ids = [json.loads(call["prompt"]).get("chunk_id") or json.loads(call["prompt"])["original_request"]["chunk_id"] for call in gateway.calls]
    assert ids.count("E01-CH-001") == 2
    assert all(ids.count(chunk_id) == 1 for chunk_id in set(ids) - {"E01-CH-001"})
    assert any(m["phase"] == "scanner_repair" for m in result.request_measurements)


def test_repair_exhaustion_preserves_other_valid_chunk_cache(tmp_path):
    gateway = FakeGateway(always_invalid=True)
    with pytest.raises(ScannerRepairExhaustedError):
        ScannerService(gateway, tmp_path, _config()).scan_project("project-1", [_episode()])
    manifests = list((tmp_path / "projects" / "project-1" / "scanner").rglob("manifest.json"))
    assert not list((tmp_path / "projects" / "project-1" / "evidence").rglob("manifest.json"))
    assert all(json.loads(path.read_text())["status"] == "RECEIVED" for path in manifests)


def test_failed_chunk_resumes_without_resending_valid_chunks(tmp_path):
    gateway = FakeGateway(invalid_chunks={"E01-CH-001"})
    service = ScannerService(gateway, tmp_path, _config())
    with pytest.raises(ScannerRepairExhaustedError):
        service.scan_project("project-1", [_episode()])
    first_counts = dict(gateway._counts)
    assert any(chunk != "E01-CH-001" for chunk in first_counts)
    gateway.invalid_chunks.clear()
    result = service.scan_project("project-1", [_episode()])
    assert gateway._counts["E01-CH-001"] == first_counts["E01-CH-001"] + 1
    for chunk_id, count in first_counts.items():
        if chunk_id != "E01-CH-001":
            assert gateway._counts[chunk_id] == count
    assert result.reused_chunk_count >= 1
    resumed_items = EvidenceStore(tmp_path, "project-1").get_episode("E01", result.evidence_revision).items
    baseline_root = tmp_path / "baseline"
    baseline = ScannerService(FakeGateway(), baseline_root, _config()).scan_project("project-1", [_episode()])
    baseline_items = EvidenceStore(baseline_root, "project-1").get_episode("E01", baseline.evidence_revision).items
    assert [(item.evidence_id, item.observation) for item in resumed_items] == [
        (item.evidence_id, item.observation) for item in baseline_items
    ]


def test_empty_completed_gateway_stream_is_retryable_transport_failure(tmp_path):
    class EmptyOnceGateway(FakeGateway):
        def __init__(self):
            super().__init__()
            self.attempts: list[tuple[str, str]] = []
            self.failed = False

        def submit_text_chat(self, **kwargs):
            prompt = json.loads(kwargs["prompt"])
            chunk_id = prompt["chunk_id"]
            self.attempts.append((kwargs["phase"], chunk_id))
            if chunk_id == "E01-CH-002" and not self.failed:
                self.failed = True
                raise EmptyApiResponseError("Gateway streaming response completed with empty content.", raw_response="")
            return super().submit_text_chat(**kwargs)

    gateway = EmptyOnceGateway()
    service = ScannerService(
        gateway, tmp_path,
        _config(parallelism=1, chunk_policy=ScannerChunkPolicy(15_000, 5000)),
    )
    with pytest.raises(ScannerResponseError, match="Gateway failed"):
        service.scan_project("project-1", [_episode()])
    assert gateway.attempts == [("scanner", "E01-CH-001"), ("scanner", "E01-CH-002")]
    assert not list((tmp_path / "projects" / "project-1" / "scanner").rglob("E01-CH-002/attempt-*.raw.txt"))

    result = service.scan_project("project-1", [_episode()])
    assert gateway.attempts[-1] == ("scanner", "E01-CH-002")
    assert len(gateway.attempts) == 3
    assert result.reused_chunk_count == 1 and result.requested_chunk_count == 1
    assert result.total_evidence_count == 2


def test_revision_invalidation_matrix():
    episode = _episode()
    base = _config()
    revision = compute_evidence_revision([episode], base)
    assert compute_evidence_revision([episode], _config(parallelism=7)) == revision
    assert compute_evidence_revision([episode], _config(model="different")) != revision
    assert compute_evidence_revision([episode], _config(reasoning="high")) != revision
    assert compute_evidence_revision([episode], _config(chunk_policy=ScannerChunkPolicy(20_000, 5000))) != revision
    assert compute_evidence_revision([_episode(" changed")], base) != revision


@pytest.mark.parametrize("version_name", [
    "SCANNER_PROMPT_VERSION",
    "SCANNER_SCHEMA_VERSION",
    "SCANNER_VALIDATION_VERSION",
    "SCANNER_NORMALIZATION_VERSION",
    "CHUNK_POLICY_VERSION",
])
def test_contract_version_changes_invalidate_revision(monkeypatch, version_name):
    import toolrecap_v4.analysis.scanner.service as service_module

    episode = _episode()
    config = _config()
    before = compute_evidence_revision([episode], config)
    monkeypatch.setattr(service_module, version_name, f"changed-{version_name}")
    assert compute_evidence_revision([episode], config) != before


def test_pre_cancel_does_not_create_complete_artifacts(tmp_path):
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CancelledError):
        ScannerService(FakeGateway(), tmp_path, _config()).scan_project("project-1", [_episode()], cancellation_token=token)
    assert not (tmp_path / "projects" / "project-1" / "evidence").exists()


def test_cancellation_during_chunk_persistence_never_marks_complete(tmp_path, monkeypatch):
    import toolrecap_v4.analysis.scanner.service as service_module

    token = CancellationToken()
    original_save = service_module._ScannerChunkCache.save

    def cancel_before_commit(self, *args, **kwargs):
        token.cancel()
        return original_save(self, *args, **kwargs)

    monkeypatch.setattr(service_module._ScannerChunkCache, "save", cancel_before_commit)
    service = ScannerService(
        FakeGateway(), tmp_path,
        _config(chunk_policy=ScannerChunkPolicy(100_000, 5000)),
    )
    with pytest.raises(CancelledError):
        service.scan_project("project-1", [_episode(cue_count=1)], cancellation_token=token)
    complete_manifests = [
        path for path in (tmp_path / "projects" / "project-1").rglob("manifest.json")
        if json.loads(path.read_text())["status"] == "COMPLETE"
    ]
    assert complete_manifests == []


def test_restart_recovers_successful_raw_response_before_validation(tmp_path, monkeypatch):
    import toolrecap_v4.analysis.scanner.service as service_module

    gateway = FakeGateway()
    service = ScannerService(gateway, tmp_path, _config(chunk_policy=ScannerChunkPolicy(100_000, 5000)))
    original_save = service_module._ScannerChunkCache.save
    crashed = {"done": False}

    def crash_once(self, *args, **kwargs):
        if not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("simulated crash after raw save")
        return original_save(self, *args, **kwargs)

    monkeypatch.setattr(service_module._ScannerChunkCache, "save", crash_once)
    with pytest.raises(RuntimeError, match="simulated crash"):
        service.scan_project("project-1", [_episode(cue_count=1)])
    assert len(gateway.calls) == 1
    result = service.scan_project("project-1", [_episode(cue_count=1)])
    assert len(gateway.calls) == 1
    assert result.total_evidence_count == 1
