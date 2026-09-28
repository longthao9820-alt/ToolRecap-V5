"""Tests for settings defaults and persistence."""

from pathlib import Path

import pytest

from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.settings import AppSettings, SettingsManager


def test_settings_defaults():
    """Verify default settings match the exact brief requirements.
    
    Defaults:
    - Audio: original 0.0 dB, commentary 0.0 dB, duck OFF, -12.0 dB ducking amount,
      target -14.0 LUFS, peak -1.0 dBTP
    - GPU: True (GPU on)
    - Voice: alloy, language en-US, style configurable ("")
    - VoiceStudio: mode auto with configurable local/remote endpoints.
    - Gateway: endpoint http://127.0.0.1:20128, provider-neutral model IDs empty
    - Notifications: all ON
    """
    s = AppSettings()

    # Audio mix
    assert s.original_audio_db == 0.0
    assert s.commentary_audio_db == 0.0
    assert s.auto_duck is False
    assert s.ducking_amount_db == -12.0
    assert s.target_loudness_lufs == -14.0
    assert s.true_peak_db == -1.0

    # Hardware & GPU
    assert s.use_gpu is True
    assert s.canvas_width == 1920
    assert s.canvas_height == 1080
    assert s.canvas_fps == 30.0
    assert s.burn_subtitles is False
    assert s.output_format == "mp4"

    # Voice & VoiceStudio
    assert s.voice_id == "alloy"
    assert s.voice_language == "en-US"
    assert s.voice_style == ""
    assert s.voice_mode == "auto"
    assert s.voice_local_url == "http://127.0.0.1:3900"
    assert s.voice_remote_url == ""

    # Gateway fresh install is provider-neutral and requires model configuration.
    assert s.gateway_endpoint == "http://127.0.0.1:20128"
    assert s.gateway_sub_model == ""
    assert s.gateway_sub_reasoning == ""
    assert s.gateway_prime_model == ""
    assert s.gateway_prime_reasoning == ""
    assert s.gateway_model == ""

    # Notifications defaults ON
    assert s.notify_complete is True
    assert s.notify_error is True
    assert s.play_completion_sound is True
    assert s.flash_taskbar is True
    assert s.notify_update is True

    # Updater
    assert s.update_repo == "longthao9820-alt/ToolRecap-V5"


def test_settings_persistence_roundtrip(tmp_path: Path):
    """Verify settings can be saved and loaded accurately."""
    mgr = SettingsManager(storage_root=tmp_path)

    # Initial load returns defaults
    default_s = mgr.load()
    assert default_s.original_audio_db == 0.0

    # Modify and save
    custom_s = AppSettings(
        original_audio_db=-3.0,
        commentary_audio_db=2.0,
        auto_duck=True,
        voice_style="cinematic",
        voice_id="echo",
        gateway_sub_model="custom-sub-model",
        gateway_sub_reasoning="high",
        gateway_prime_model="custom-prime-model",
        gateway_prime_reasoning="low",
    )
    mgr.save(custom_s)

    # Reload
    loaded_s = mgr.load()
    assert loaded_s.original_audio_db == -3.0
    assert loaded_s.commentary_audio_db == 2.0
    assert loaded_s.auto_duck is True
    assert loaded_s.voice_style == "cinematic"
    assert loaded_s.voice_id == "echo"
    assert loaded_s.target_loudness_lufs == -14.0  # preserved default
    assert loaded_s.gateway_sub_model == "custom-sub-model"
    assert loaded_s.gateway_sub_reasoning == "high"
    assert loaded_s.gateway_prime_model == "custom-prime-model"
    assert loaded_s.gateway_prime_reasoning == "low"


def test_corrupt_settings_are_backed_up_before_defaults_can_be_saved(tmp_path: Path):
    manager = SettingsManager(storage_root=tmp_path)
    manager.save(AppSettings(gateway_endpoint="https://gateway.example", scanner_model="scanner-x"))
    settings_path = tmp_path / "settings" / "settings.json"
    corrupt_bytes = b"{broken settings"
    settings_path.write_bytes(corrupt_bytes)

    loaded = manager.load()
    assert loaded.gateway_endpoint == "http://127.0.0.1:20128"
    assert manager.last_load_warning
    assert manager.corrupt_backup_path is not None
    assert manager.corrupt_backup_path.read_bytes() == corrupt_bytes

    loaded.prompt = "new prompt"
    manager.save(loaded)
    assert manager.corrupt_backup_path.read_bytes() == corrupt_bytes
    assert manager.load().prompt == "new prompt"


def test_legacy_scanner_settings_migrate_provider_neutrally():
    settings = AppSettings.from_dict({
        "gateway_sub_model": "literal-custom-route",
        "gateway_sub_reasoning": "medium",
    })
    assert settings.scanner_model == "literal-custom-route"
    assert settings.scanner_reasoning == "medium"


def test_fresh_scanner_model_is_not_vendor_hardcoded():
    settings = AppSettings()
    assert settings.scanner_model == ""


def test_legacy_planner_settings_migrate_provider_neutrally():
    settings = AppSettings.from_dict({
        "gateway_prime_model": "literal-planner-route",
        "gateway_prime_reasoning": "high",
    })
    assert settings.planner_model == "literal-planner-route"
    assert settings.planner_reasoning == "high"


def test_fresh_planner_model_is_not_vendor_hardcoded():
    assert AppSettings().planner_model == ""
