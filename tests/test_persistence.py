"""Tests for durable atomic persistence in LOCALAPPDATA with secret prevention."""

import os
from pathlib import Path
import pytest

from toolrecap_v4.errors import PersistenceError, SecretExposureError
from toolrecap_v4.persistence import (
    DEFAULT_APP_DIR_NAME,
    ProjectPersistence,
    atomic_write_json,
    get_storage_root,
    read_json,
)
from toolrecap_v4.validator import check_for_secrets


def test_storage_root_in_localappdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify default storage root is strictly located in LOCALAPPDATA, never app dir."""
    fake_local_app_data = str(tmp_path / "AppData" / "Local")
    monkeypatch.setenv("LOCALAPPDATA", fake_local_app_data)

    root = get_storage_root()

    assert str(root).startswith(fake_local_app_data)
    assert root.name == DEFAULT_APP_DIR_NAME
    assert root.exists()


def test_atomic_project_state_roundtrip(tmp_path: Path) -> None:
    """Verify atomic project save and load roundtrip reproduces exact JSON."""
    persistence = ProjectPersistence(storage_root=tmp_path)

    project_data = {
        "schema_version": "3.0",
        "project_id": "proj-roundtrip-01",
        "project_name": "Test_Project",
        "sources": [{"source_file": "ep1.mp4"}],
        "outputs": [
            {
                "render_id": "render-01",
                "title": "Recap_1",
                "segments": [
                    {
                        "segment_id": "seg-1",
                        "source_file": "ep1.mp4",
                        "start_ms": 0,
                        "end_ms": 1000,
                        "type": "narration",
                        "narration": "Text",
                        "source_audio": False,
                        "subtitles": [],
                    }
                ],
            }
        ],
    }

    saved_path = persistence.save_project(project_data)
    assert saved_path.exists()
    assert saved_path.suffix == ".json"

    loaded_data = persistence.load_project("proj-roundtrip-01")
    assert loaded_data == project_data


def test_atomic_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Verify checkpoint save and load roundtrip."""
    persistence = ProjectPersistence(storage_root=tmp_path)

    checkpoint_data = {
        "checkpoint_id": "ckpt-01",
        "render_id": "render-01",
        "status": "in_progress",
        "progress_percent": 45,
    }

    saved_path = persistence.save_checkpoint("proj-01", "ckpt-01", checkpoint_data)
    assert saved_path.exists()

    loaded_data = persistence.load_checkpoint("proj-01", "ckpt-01")
    assert loaded_data == checkpoint_data


def test_delete_project_removes_private_checkpoints_but_keeps_publication(tmp_path: Path) -> None:
    persistence = ProjectPersistence(storage_root=tmp_path / "state")
    project_id = "delete-me"
    publication = tmp_path / "published" / "result.mp4"
    publication.parent.mkdir()
    publication.write_bytes(b"published-video")
    persistence.save_project({
        "project_id": project_id,
        "project_name": "Delete Me",
        "status": "cancelled",
        "output_dir": str(publication.parent),
    })
    persistence.save_checkpoint(project_id, "render", {"status": "failed"})
    persistence.save_final_json(project_id, {"outputs": []})
    persistence.save_prepared_episode(project_id, "E01", {"episode_id": "E01"})
    (persistence.projects_dir / project_id / "downstream" / "voice").mkdir(parents=True)
    (persistence.projects_dir / project_id / "downstream" / "voice" / "seg.wav").write_bytes(b"voice")

    removed = persistence.delete_project(project_id)

    assert removed
    assert not persistence._project_path(project_id).exists()
    assert not (persistence.checkpoints_dir / project_id).exists()
    assert not (persistence.prepared_dir / project_id).exists()
    assert not (persistence.projects_dir / project_id).exists()
    assert publication.read_bytes() == b"published-video"


def test_atomic_settings_roundtrip(tmp_path: Path) -> None:
    """Verify settings save and load roundtrip."""
    persistence = ProjectPersistence(storage_root=tmp_path)

    settings = {
        "theme": "dark",
        "default_recap_mode": "FULL_RECAP",
    }

    saved_path = persistence.save_settings(settings)
    assert saved_path.exists()

    loaded_settings = persistence.load_settings()
    assert loaded_settings == settings


def test_atomic_write_cleans_up_temp_file(tmp_path: Path) -> None:
    """Verify atomic write leaves no temporary files behind on success."""
    target_file = tmp_path / "data.json"
    data = {"key": "value"}

    atomic_write_json(target_file, data)

    assert target_file.exists()
    tmp_files = list(tmp_path.glob("*.tmp.*"))
    assert len(tmp_files) == 0


def test_secrets_never_persisted_in_project_json(tmp_path: Path) -> None:
    """Verify any secret keys are rejected before disk write, leaving no file."""
    persistence = ProjectPersistence(storage_root=tmp_path)

    project_data = {
        "project_id": "proj-secrets",
        "project_name": "Secret_Project",
        "api_key": "sk-secret-token-12345",
    }

    with pytest.raises(SecretExposureError, match="Forbidden secret key detected"):
        persistence.save_project(project_data)

    target_file = tmp_path / "projects" / "proj-secrets.json"
    assert not target_file.exists()


def test_nested_secrets_rejected_before_persistence(tmp_path: Path) -> None:
    """Verify nested secret keys are also detected and rejected."""
    persistence = ProjectPersistence(storage_root=tmp_path)

    checkpoint_data = {
        "checkpoint_id": "ckpt-secret",
        "render_id": "render-01",
        "auth": {
            "token": "bearer-123",
        },
    }

    with pytest.raises(SecretExposureError, match="Forbidden secret key detected"):
        persistence.save_checkpoint("proj-01", "ckpt-secret", checkpoint_data)


def test_security_words_in_media_identities_and_content_are_not_secrets() -> None:
    check_for_secrets({
        "source_fingerprints": {
            "Secret Invasion S01E01.mkv": {"path": r"D:\Shows\Secret Invasion S01E01.mkv"},
            "The Secret Key.mp4": {"path": r"D:\Shows\The Secret Key.mp4"},
            "Password Reset Episode.mkv": {"path": r"D:\Shows\Password Reset Episode.mkv"},
        },
        "selected_subtitle": "Token of Trust.en.srt",
        "title": "The API Key Mystery",
        "prompt": "Explain why the secret key matters to the story.",
    })


@pytest.mark.skipif(os.name != "nt", reason="Windows long-path regression")
def test_atomic_json_supports_path_beyond_legacy_max_path(tmp_path: Path) -> None:
    target = tmp_path / ("project-" + "x" * 80) / ("revision-" + "y" * 80) / ("output-" + "z" * 40) / "manifest.json"
    assert len(str(target)) > 260
    atomic_write_json(target, {"status": "COMPLETE"})
    assert read_json(target) == {"status": "COMPLETE"}


def test_prepared_artifacts_persistence_roundtrip(tmp_path: Path) -> None:
    """Verify prepared episode metadata and manifest persistence roundtrip."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    project_id = "proj-prep-01"

    ep1_data = {
        "episode_id": "E01",
        "source_file": "ep01.mp4",
        "duration_ms": 5000,
        "status": "ready",
        "transcript_method": "sidecar",
    }
    ep2_data = {
        "episode_id": "E02",
        "source_file": "ep02.mp4",
        "duration_ms": 6000,
        "status": "ready",
        "transcript_method": "stt",
    }

    assert not persistence.has_prepared_episode(project_id, "E01")
    path1 = persistence.save_prepared_episode(project_id, "E01", ep1_data)
    assert path1.exists()
    assert persistence.has_prepared_episode(project_id, "E01")
    loaded1 = persistence.load_prepared_episode(project_id, "E01")
    assert loaded1 == ep1_data

    persistence.save_prepared_episode(project_id, "E02", ep2_data)
    all_eps = persistence.list_prepared_episodes(project_id)
    assert len(all_eps) == 2
    assert [e["episode_id"] for e in all_eps] == ["E01", "E02"]

    assert not persistence.has_prepared_manifest(project_id)
    manifest = {
        "project_id": project_id,
        "status": "completed",
        "episodes": all_eps,
        "total_episodes": 2,
    }
    m_path = persistence.save_prepared_manifest(project_id, manifest)
    assert m_path.exists()
    assert persistence.has_prepared_manifest(project_id)
    loaded_m = persistence.load_prepared_manifest(project_id)
    assert loaded_m == manifest

    # Secrets rejected in prepared artifacts
    with pytest.raises(SecretExposureError, match="Forbidden secret key detected"):
        persistence.save_prepared_episode(project_id, "E03", {"api_key": "secret"})
    with pytest.raises(SecretExposureError, match="Forbidden secret key detected"):
        persistence.save_prepared_manifest(project_id, {"auth_token": "secret"})
