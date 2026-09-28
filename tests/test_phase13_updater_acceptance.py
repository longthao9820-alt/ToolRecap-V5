"""Synthetic end-to-end updater staging acceptance for Phase 13."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import httpx
import pytest

from toolrecap_v4.errors import ChecksumMismatchError, MaliciousArchiveError, PackageValidationError, UpdateCheckError, UpdateInProgressError
from toolrecap_v4.updater import UpdateCheckResult, UpdateManager, safe_extract_zip, validate_package


def _package_files(version: str = "4.1.0") -> dict[str, bytes]:
    schema_path = Path(__file__).parents[1] / "toolrecap_v4" / "schemas" / "recap_v3_schema.json"
    marker = {
        "package_format": "toolrecap-portable-v1",
        "schema_version": "3.0",
        "version": version,
        "required_files": [
            "ToolRecapV4.exe", "ffmpeg.exe", "ffprobe.exe",
            "FFMPEG_LICENSE.txt", "schemas/recap_v3_schema.json",
            "_internal/base_library.zip", "_internal/python312.dll",
        ],
    }
    return {
        "ToolRecapV4.exe": b"MZ-new-executable",
        "_internal/base_library.zip": b"python standard library",
        "_internal/python312.dll": b"python runtime",
        "ffmpeg.exe": b"MZ-ffmpeg",
        "ffprobe.exe": b"MZ-ffprobe",
        "FFMPEG_LICENSE.txt": b"license",
        "schemas/recap_v3_schema.json": schema_path.read_bytes(),
        "package_marker.json": json.dumps(marker).encode(),
    }


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _result() -> UpdateCheckResult:
    return UpdateCheckResult(
        status="update_available", current_version="4.0.0", latest_version="4.1.0",
        zip_url="https://updates.example/package.zip",
        sha256_url="https://updates.example/package.sha256.txt",
        zip_name="ToolRecapV4-v4.1.0-windows-portable.zip",
    )


def _client(zip_payload: bytes, checksum: str | None = None) -> httpx.Client:
    digest = checksum or hashlib.sha256(zip_payload).hexdigest()
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("sha256.txt"):
            return httpx.Response(200, text=f"{digest}  package.zip\n")
        return httpx.Response(200, content=zip_payload, headers={"content-length": str(len(zip_payload))})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_valid_newer_update_stages_complete_package_and_preserves_user_state(tmp_path):
    payload = _zip_bytes(_package_files())
    state = tmp_path / "LocalAppData" / "ToolRecapV4"
    project = state / "projects" / "project.json"
    secret = state / "secrets" / "credentials.dpapi"
    final_json = state / "final" / "project.json"
    narration = state / "projects" / "p" / "downstream" / "voice" / "n.wav"
    publication = tmp_path / "Outputs_S03" / "Recap.mp4"
    for path, content in ((project,b"project"),(secret,b"secret"),(final_json,b"final"),(narration,b"wav"),(publication,b"video")):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
    before = {path: path.read_bytes() for path in (project, secret, final_json, narration, publication)}
    manager = UpdateManager(storage_root=state, client=_client(payload), current_version="4.0.0")
    staged = manager.download_and_stage(_result())
    validate_package(staged, expected_version="4.1.0")
    assert all(path.read_bytes() == content for path, content in before.items())


def test_checksum_mismatch_and_malformed_package_clean_staging(tmp_path):
    state = tmp_path / "state"
    payload = _zip_bytes(_package_files())
    manager = UpdateManager(storage_root=state, client=_client(payload, "0" * 64), current_version="4.0.0")
    with pytest.raises(ChecksumMismatchError):
        manager.download_and_stage(_result())
    assert not any(manager.staging_dir.glob("*.zip"))
    assert not (manager.staging_dir / "v4.1.0").exists()

    malformed = _package_files()
    malformed.pop("schemas/recap_v3_schema.json")
    malformed_payload = _zip_bytes(malformed)
    manager.client = _client(malformed_payload)
    with pytest.raises(PackageValidationError, match="schema"):
        manager.download_and_stage(_result())
    assert not (manager.staging_dir / "v4.1.0").exists()


def test_active_project_guard_blocks_stage_and_apply_before_mutation(tmp_path):
    manager = UpdateManager(
        storage_root=tmp_path / "state", client=_client(_zip_bytes(_package_files())),
        current_version="4.0.0", activity_probe=lambda: True,
    )
    with pytest.raises(UpdateInProgressError, match="workflow is active"):
        manager.download_and_stage(_result())
    with pytest.raises(UpdateInProgressError, match="workflow is active"):
        manager.launch_apply_helper(tmp_path / "missing", tmp_path / "install")


def test_package_audit_rejects_user_state_and_requires_runtime_layout(tmp_path):
    package = tmp_path / "package"
    files = _package_files()
    for name, content in files.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    validate_package(package, expected_version="4.1.0")
    forbidden = package / "secrets" / "credentials.dpapi"
    forbidden.parent.mkdir()
    forbidden.write_bytes(b"must-not-ship")
    with pytest.raises(PackageValidationError, match="forbidden"):
        validate_package(package)


def test_package_without_python_runtime_is_rejected(tmp_path):
    package = tmp_path / "package"
    for name, content in _package_files().items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (package / "_internal" / "python312.dll").unlink()
    with pytest.raises(PackageValidationError, match="Python runtime DLL"):
        validate_package(package)


@pytest.mark.parametrize(("tag", "expected"), [
    ("v4.1.0", "update_available"),
    ("v4.0.0", "up_to_date"),
    ("v3.9.9", "up_to_date"),
    ("invalid-version", "error"),
])
def test_release_semver_statuses(tmp_path, tag, expected):
    release = {
        "tag_name": tag,
        "assets": [
            {"name": "ToolRecapV4-v4.1.0-windows-portable.zip", "browser_download_url": "https://example.test/package.zip"},
            {"name": "ToolRecapV4-v4.1.0-windows-portable.zip.sha256.txt", "browser_download_url": "https://example.test/package.sha256.txt"},
        ],
    }
    client = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=release)))
    manager = UpdateManager(storage_root=tmp_path / "state", client=client, current_version="4.0.0")
    assert manager.check_for_updates("example/toolrecap").status == expected


@pytest.mark.parametrize("entry", [
    "..\\evil.exe", "../evil.exe", "nested/..\\..\\evil.exe",
    "C:\\outside\\evil.exe", "/absolute/evil.exe",
])
def test_windows_archive_traversal_variants_are_rejected(tmp_path, entry):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(entry, b"malicious")
    target = tmp_path / "staging"
    with pytest.raises(MaliciousArchiveError, match="Path traversal"):
        safe_extract_zip(archive_path, target)
    assert not (tmp_path / "evil.exe").exists()


def test_incomplete_download_cleans_archive_and_stage(tmp_path):
    payload = _zip_bytes(_package_files())
    digest = hashlib.sha256(payload).hexdigest()
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("sha256.txt"):
            return httpx.Response(200, text=f"{digest}  package.zip\n")
        return httpx.Response(200, content=payload, headers={"content-length": str(len(payload) + 1)})
    manager = UpdateManager(
        storage_root=tmp_path / "state", client=httpx.Client(transport=httpx.MockTransport(handler)),
        current_version="4.0.0",
    )
    with pytest.raises((UpdateCheckError, httpx.RemoteProtocolError)):
        manager.download_and_stage(_result())
    assert not any(manager.staging_dir.glob("*.zip"))
    assert not (manager.staging_dir / "v4.1.0").exists()
