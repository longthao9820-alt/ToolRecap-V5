"""Focused Phase 13 runtime-resource and self-check acceptance."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import build_portable
from toolrecap_v4.media import BinaryNotFoundError, EncoderStatus, find_binary
from toolrecap_v4.runtime import application_root, find_resource, package_mode
from toolrecap_v4.schemas import schema as schema_module
from toolrecap_v4.secrets import OPTIONAL_ENTROPY
from toolrecap_v4.selfcheck import (
    FAIL,
    PASS,
    WARN,
    SelfCheckItem,
    SelfCheckResult,
    collect_selfcheck,
    run_selfcheck,
)


def test_runtime_resource_lookup_ignores_process_cwd(tmp_path, monkeypatch):
    fake = tmp_path / "toolrecap_v4" / "schemas" / "recap_v3_schema.json"
    fake.parent.mkdir(parents=True)
    fake.write_text('{"title":"malicious cwd resource"}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    schema_module._SCHEMA_CACHE = None
    loaded = schema_module.get_project_schema()
    assert loaded.get("title") != "malicious cwd resource"
    assert find_resource("toolrecap_v4/schemas/recap_v3_schema.json").is_file()
    assert application_root() != tmp_path
    assert package_mode() == "source"


def test_media_binary_lookup_does_not_trust_cwd(tmp_path, monkeypatch):
    (tmp_path / "phase13_fake.exe").write_bytes(b"not executable")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("toolrecap_v4.media.shutil.which", lambda _name: None)
    monkeypatch.delenv("TOOLRECAP_PHASE13_FAKE_PATH", raising=False)
    monkeypatch.delenv("PHASE13_FAKE_PATH", raising=False)
    with pytest.raises(BinaryNotFoundError):
        find_binary("phase13_fake")


def test_structured_selfcheck_is_zero_project_and_secret_redacted(tmp_path, monkeypatch):
    state = tmp_path / "LocalAppData" / "ToolRecapV4"
    settings_dir = state / "settings"
    settings_dir.mkdir(parents=True)
    settings_dir.joinpath("settings.json").write_text(json.dumps({
        "gateway_endpoint": "http://127.0.0.1:20128",
        "scanner_model": "scanner-model",
        "finalizer_model": "finalizer-model",
        "voice_mode": "remote",
        "voice_remote_url": "https://voice.example.test",
    }), encoding="utf-8")
    secrets_dir = state / "secrets"
    secrets_dir.mkdir()
    secrets_dir.joinpath("credentials.dpapi").write_bytes(b"TOP-SECRET-CIPHERTEXT")
    project = state / "projects" / "project.json"
    final_json = state / "final" / "project.json"
    narration = state / "projects" / "project" / "downstream" / "voice" / "segment.wav"
    for path, content in ((project, b"project state"), (final_json, b"final json"), (narration, b"voice cache")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    preserved = {path: path.read_bytes() for path in (project, final_json, narration)}
    fake_binary = tmp_path / "binary.exe"
    fake_binary.write_bytes(b"binary")
    monkeypatch.setattr("toolrecap_v4.selfcheck._binary_version", lambda _name: (fake_binary, "test version"))
    monkeypatch.setattr("toolrecap_v4.selfcheck.detect_gpu_encoder", lambda *a, **k: EncoderStatus(False, "No NVIDIA", "libx264", "CPU", "no GPU"))
    monkeypatch.setattr("toolrecap_v4.selfcheck.probe_encoder_usable", lambda *a, **k: True)

    result = collect_selfcheck(storage_root=state)
    payload = json.dumps(result.to_dict(), ensure_ascii=False)
    assert result.status in (PASS, WARN)
    assert "TOP-SECRET-CIPHERTEXT" not in payload
    assert '"api_key": "configured"' in payload
    assert all(path.read_bytes() == content for path, content in preserved.items())
    assert not (state / "outputs").exists()
    assert any(item.id == "resources.schema" and item.status == PASS for item in result.checks)
    assert any(item.id == "ocr.runtime" and item.status == PASS for item in result.checks)
    assert any(item.id == "stt.runtime" and item.status == PASS for item in result.checks)
    assert any(item.id == "output.resolver" and item.details["created"] is False for item in result.checks)


def test_selfcheck_exit_policy_and_json_report(tmp_path):
    warning_result = SelfCheckResult(
        "4.0.0", "source", str(tmp_path), str(tmp_path),
        (SelfCheckItem("optional", "GPU / ENCODER", WARN, "CPU fallback", False),),
    )
    report = tmp_path / "report.json"
    assert run_selfcheck(
        ["tool", "--selfcheck", "--selfcheck-json", str(report)],
        collector=lambda **_kwargs: warning_result,
    ) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == WARN

    fatal_result = SelfCheckResult(
        "4.0.0", "source", str(tmp_path), str(tmp_path),
        (SelfCheckItem("schema", "RESOURCES", FAIL, "missing", True),),
    )
    assert run_selfcheck(["tool", "--selfcheck"], collector=lambda **_kwargs: fatal_result) == 1


def test_headless_update_handshake_precedes_gui_import():
    result = subprocess.run(
        [sys.executable, "main.py", "--update-handshake"], cwd=application_root(),
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "TOOLRECAP_V4_HANDSHAKE_OK" in result.stdout


def test_build_command_is_onedir_and_collects_native_runtimes():
    command = build_portable.pyinstaller_command(
        application_root() / "toolrecap_v4" / "schemas" / "recap_v3_schema.json"
    )
    joined = " ".join(command)
    assert "--onedir" in command and "--windowed" in command
    assert "--onefile" not in command
    for package in ("onnxruntime", "rapidocr_onnxruntime", "ctranslate2", "faster_whisper", "huggingface_hub"):
        assert f"--collect-all={package}" in command
    assert "--exclude-module=pytest" in command
    assert "C:\\Users\\Long" not in Path(build_portable.__file__).read_text(encoding="utf-8")


def test_dpapi_compatibility_constant_is_unchanged():
    assert OPTIONAL_ENTROPY == b"ToolRecapV3_DPAPI_SecretStorage_v1"
