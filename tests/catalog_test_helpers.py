from __future__ import annotations

from pathlib import Path

from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.analysis.models import DialogueReference, Evidence, PreparedEpisode, Transcript, TranscriptCue


def evidence(
    evidence_id: str,
    source_id: str,
    start_ms: int,
    end_ms: int,
    observation: str,
    *,
    category: str = "dialogue",
    entities: tuple[str, ...] = (),
    modality: str = "subtitle",
    uncertainty: tuple[str, ...] = (),
    confidence: float | None = None,
) -> Evidence:
    episode_id = evidence_id.split("-EV-")[0]
    return Evidence(
        evidence_id=evidence_id, episode_id=episode_id, source_id=source_id,
        start_ms=start_ms, end_ms=end_ms, category=category, observation=observation,
        dialogue=(DialogueReference(f"{episode_id}-CUE-001", 1, 1, observation),),
        entities=entities, modality=modality, confidence=confidence,
        uncertainty=uncertainty, visual_refs=(),
        provenance={"chunk_id": f"{episode_id}-CH-001", "artifact_hash": "chunk-hash"},
    )


def prepared(
    episode_id: str,
    source_id: str,
    *,
    duration_ms: int = 100_000,
    empty: bool = False,
    status: str = "ready",
) -> PreparedEpisode:
    cues = () if empty else (TranscriptCue(f"{episode_id}-CUE-001", 1000, 5000, "Dialogue"),)
    transcript = Transcript(
        episode_id=episode_id,
        source_type="empty" if empty else "sidecar",
        source_format="none" if empty else "srt",
        cues=cues,
        has_speech=not empty,
        provenance_hash=f"transcript-{episode_id}",
    )
    return PreparedEpisode(
        episode_id=episode_id, source_id=source_id,
        source_path=Path(f"C:/media/{episode_id}.mkv"), source_basename=f"{episode_id}.mkv",
        duration_ms=duration_ms, canvas_width=1920, canvas_height=1080,
        transcript=transcript, transcript_method=transcript.source_type,
        artifact_hash=f"prepared-{episode_id}", source_fingerprint=f"source-{episode_id}",
        status=status,
    )


def populated_store(root: Path, project_id: str = "project-1", revision: str = "evr-catalog-test"):
    store = EvidenceStore(root, project_id)
    e01 = (
        evidence(
            "E01-EV-001", "src_shared", 1000, 3000,
            "Beth says “héllo” — こんにちは.", entities=("Beth", "Jamie"),
            uncertainty=("speaker label absent",),
        ),
        evidence(
            "E01-EV-002", "src_shared", 4000, 4500,
            "A short event.", category="event", entities=("Minor Character",), modality="ocr",
        ),
        evidence(
            "E01-EV-003", "src_shared", 8000, 9000,
            "Repeated factual event " + ("long text " * 100), category="event", modality="stt",
            confidence=None,
        ),
        evidence(
            "E01-EV-004", "src_shared", 10_000, 11_000,
            "Repeated factual event " + ("long text " * 100), category="event", modality="stt",
            confidence=None,
        ),
    )
    store.save_episode(revision, "E01", e01, dependency_digest="dep-e01")
    store.save_episode(revision, "E02", (), dependency_digest="dep-e02")
    store.commit_revision(
        revision, ["E01", "E02"],
        dependency_signature={"scanner": "test", "episodes": ["E01", "E02"]},
    )
    episodes = (
        prepared("E01", "src_shared"),
        prepared("E02", "src_shared", empty=True),
    )
    return store, revision, episodes, e01
