"""Settings configuration and defaults for ToolRecap V4.

Defaults match the specified requirements:
- Audio: original 0.0 dB, commentary 0.0 dB, duck OFF, ducking amount -12.0 dB,
  target loudness -14.0 LUFS, true peak -1.0 dBTP.
- GPU: True (GPU on).
- Voice: alloy, language en-US, style configurable.
- VoiceStudio: mode auto, configurable local/remote endpoints.
- AI Gateway: endpoint http://127.0.0.1:20128; provider-neutral model IDs default empty.
- Notifications: all defaults ON.
- Secrets: strictly separated; never stored in settings JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any, Dict, Optional

from toolrecap_v4.persistence import ProjectPersistence, get_storage_root
from toolrecap_v4.endpoints import normalize_http_endpoint

REASONING_VALUES = ("", "low", "medium", "high")
COMMENTARY_SPEED_VALUES = tuple(round(0.80 + index * 0.05, 2) for index in range(11))


@dataclass(frozen=True)
class GatewayFormValues:
    endpoint: str
    api_key: str
    scanner_model: str
    scanner_reasoning: str
    scanner_parallelism: int
    scanner_chunk_duration_ms: int
    vision_model: str
    vision_reasoning: str
    finalizer_model: str
    finalizer_reasoning: str


class GatewaySettingsController:
    """UI-independent validation/save/diagnostic layer for the Gateway pane."""

    def __init__(self, settings_manager: SettingsManager, secret_store: Any, gateway_factory: Any = None) -> None:
        self.settings_manager = settings_manager
        self.secret_store = secret_store
        self.gateway_factory = gateway_factory

    @staticmethod
    def validate(form: GatewayFormValues) -> GatewayFormValues:
        endpoint = normalize_http_endpoint(form.endpoint, "Gateway Endpoint")
        if form.scanner_reasoning not in REASONING_VALUES or form.vision_reasoning not in REASONING_VALUES or form.finalizer_reasoning not in REASONING_VALUES:
            raise ValueError("Unsupported reasoning value.")
        if type(form.scanner_parallelism) is not int or not 1 <= form.scanner_parallelism <= 32:
            raise ValueError("Scanner parallelism must be an integer from 1 to 32.")
        if type(form.scanner_chunk_duration_ms) is not int or not 1_000 <= form.scanner_chunk_duration_ms <= 3_600_000:
            raise ValueError("Scanner chunk duration must be 1,000–3,600,000 ms.")
        return replace(form, endpoint=endpoint, scanner_model=form.scanner_model.strip(), vision_model=form.vision_model.strip(), finalizer_model=form.finalizer_model.strip())

    def save(self, base: AppSettings, form: GatewayFormValues, *, masked_placeholder: str) -> AppSettings:
        valid = self.validate(form)
        if not valid.finalizer_model and not valid.scanner_model and not any(
            (base.planner_model, base.writer_model, base.gateway_prime_model)
        ):
            raise ValueError(
                "Configure the V5 Analysis Model or a legacy Scanner Model fallback."
            )
        updated = replace(base)
        updated.gateway_endpoint = valid.endpoint
        updated.scanner_model = updated.gateway_sub_model = updated.gateway_model = valid.scanner_model
        updated.scanner_reasoning = updated.gateway_sub_reasoning = valid.scanner_reasoning
        updated.scanner_parallelism = valid.scanner_parallelism
        updated.scanner_chunk_duration_ms = valid.scanner_chunk_duration_ms
        updated.vision_model = valid.vision_model
        updated.vision_reasoning = valid.vision_reasoning
        # Blank Finalizer on a legacy split configuration means "preserve".
        # A non-empty explicit value is the only operation that unifies stages.
        if valid.finalizer_model:
            updated.apply_unified_finalizer(valid.finalizer_model, valid.finalizer_reasoning)
        updated.gateway_thinking = bool(valid.scanner_reasoning or valid.finalizer_reasoning)
        previous = self.secret_store.get_secret("gateway_api_key")
        secret_changed = valid.api_key != masked_placeholder
        try:
            if secret_changed:
                if valid.api_key:
                    self.secret_store.set_secret("gateway_api_key", valid.api_key.strip())
                else:
                    self.secret_store.delete_secret("gateway_api_key")
            self.settings_manager.save(updated)
        except Exception:
            if secret_changed:
                if previous is None:
                    self.secret_store.delete_secret("gateway_api_key")
                else:
                    self.secret_store.set_secret("gateway_api_key", previous)
            raise
        return updated

    def test(self, form: GatewayFormValues, role: str, *, masked_placeholder: str) -> dict[str, Any]:
        from toolrecap_v4.gateway import GatewayClient, sanitize_message
        valid = self.validate(form)
        model = valid.scanner_model if role == "scanner" else valid.finalizer_model
        reasoning = valid.scanner_reasoning if role == "scanner" else valid.finalizer_reasoning
        if not model:
            return {"ok": False, "role": role, "model": "", "message": f"{role.title()} Model is not configured"}
        key = self.secret_store.get_secret("gateway_api_key") if valid.api_key == masked_placeholder else valid.api_key
        factory = self.gateway_factory or GatewayClient
        try:
            client = factory(base_url=valid.endpoint, api_key=key or None, timeout=8.0, max_retries=0)
            client.submit_text_chat(
                prompt=f"Configuration diagnostic for {role}. Reply with OK.",
                model=model,
                reasoning_effort=reasoning or None,
                stream=False,
                expect_json=False,
                phase=f"settings_test_{role}",
            )
            return {"ok": True, "role": role, "model": model, "message": "Connection successful"}
        except Exception as exc:
            safe = sanitize_message(str(exc), [key])[:300]
            return {"ok": False, "role": role, "model": model, "message": safe}


@dataclass
class AppSettings:
    settings_schema_version: int = 13
    # Audio mix
    original_audio_db: float = 0.0
    commentary_audio_db: float = 0.0
    auto_duck: bool = False
    ducking_amount_db: float = -12.0
    target_loudness_lufs: float = -14.0
    true_peak_db: float = -1.0
    # Highlight never contains narration.  Preserve source loudness by default;
    # creators may explicitly opt into the publication loudness target.
    highlight_normalize_loudness: bool = False

    # Hardware & render
    use_gpu: bool = True
    quality: str = "high"
    video_codec: str = "h264"
    canvas_width: int = 1920
    canvas_height: int = 1080
    canvas_fps: float = 30.0
    canvas_auto: bool = True
    burn_subtitles: bool = False
    output_format: str = "mp4"
    output_dir: str = ""
    review_before_render: bool = False

    # Voice & VoiceStudio
    voice_mode: str = "auto"  # "auto", "local", "remote"
    voice_local_url: str = "http://127.0.0.1:3900"
    voice_remote_url: str = ""
    voice_id: str = "alloy"
    voice_language: str = "en-US"
    voice_style: str = ""
    voice_model: str = "omnivoice"
    commentary_reading_speed: float = 1.0

    # AI Gateway (Dual-Stage: Sub for video analysis, Prime for synthesis)
    gateway_endpoint: str = "http://127.0.0.1:20128"
    gateway_sub_model: str = ""
    gateway_sub_reasoning: str = ""
    gateway_prime_model: str = ""
    gateway_prime_reasoning: str = ""
    gateway_model: str = ""
    gateway_thinking: bool = False

    # Phase 4 factual Scanner. Empty on a fresh install until explicitly configured.
    # Existing V3/V4 settings are migrated from gateway_sub_* by from_dict().
    scanner_model: str = ""
    scanner_reasoning: str = ""
    scanner_parallelism: int = 3
    scanner_chunk_duration_ms: int = 300_000
    scanner_max_request_bytes: int = 131_072
    scanner_repair_attempts: int = 1

    # Phase 6 provider-neutral Season Planner. Empty until configured.
    planner_model: str = ""
    planner_reasoning: str = ""
    planner_max_rounds: int = 3
    planner_repair_attempts: int = 1
    planner_max_request_bytes: int | None = None

    # Phase 7 selective general still-image Vision.
    vision_model: str = ""
    vision_reasoning: str = ""
    vision_frames_per_range: int = 6
    vision_frames_per_episode: int = 24
    vision_hard_frame_cap: int = 256
    vision_repair_attempts: int = 1

    # Phase 8 independent provider-neutral per-output Writers.
    writer_model: str = ""
    writer_reasoning: str = ""
    writer_parallelism: int = 3
    writer_max_request_bytes: int | None = None
    writer_max_response_bytes: int = 10 * 1024 * 1024
    writer_repair_attempts: int = 2

    # Canonical Phase 10 user-facing creative-stage setting. Compatibility
    # stage values remain persisted independently until explicit unified Save.
    finalizer_model: str = ""
    finalizer_reasoning: str = ""

    # Notifications (all defaults ON)
    notify_complete: bool = True
    notify_error: bool = True
    play_completion_sound: bool = True
    flash_taskbar: bool = True
    notify_update: bool = True

    # Updater
    update_repo: str = "longthao9820-alt/ToolRecap-V5"

    # Prompt
    prompt: str = ""
    highlight_prompt: str = ""

    # Recap metadata (V2 compatible)
    recap_language: str = "en-US"
    recap_mode: str = "MAIN_STORIES"
    content_type: str = "US_TV_SHOW"
    source_rights_status: str = "UNVERIFIED"

    def __post_init__(self) -> None:
        if self.canvas_auto is None:
            self.canvas_auto = True
        speed=float(self.commentary_reading_speed)
        if not any(abs(speed-value)<1e-9 for value in COMMENTARY_SPEED_VALUES):
            raise ValueError("commentary_reading_speed must be a supported value from 0.80x to 1.30x")
        self.commentary_reading_speed=round(speed,2)
        if self.gateway_sub_model and (not self.gateway_model or self.gateway_model == "sub"):
            self.gateway_model = self.gateway_sub_model
        elif self.gateway_model and not self.gateway_sub_model:
            self.gateway_sub_model = self.gateway_model
        if self.scanner_parallelism < 1:
            raise ValueError("scanner_parallelism must be at least 1")
        if self.scanner_chunk_duration_ms < 1:
            raise ValueError("scanner_chunk_duration_ms must be positive")
        if self.scanner_max_request_bytes < 1024:
            raise ValueError("scanner_max_request_bytes must be at least 1024")
        if self.scanner_repair_attempts < 0:
            raise ValueError("scanner_repair_attempts must be non-negative")
        if self.planner_max_rounds < 1:
            raise ValueError("planner_max_rounds must be at least 1")
        if self.planner_repair_attempts < 0:
            raise ValueError("planner_repair_attempts must be non-negative")
        if self.planner_max_request_bytes is not None and self.planner_max_request_bytes < 1024:
            raise ValueError("planner_max_request_bytes must be null or at least 1024")
        if min(self.vision_frames_per_range, self.vision_frames_per_episode, self.vision_hard_frame_cap) < 1:
            raise ValueError("Vision frame budgets must be positive")
        if self.vision_repair_attempts < 0:
            raise ValueError("vision_repair_attempts must be non-negative")
        if self.writer_parallelism < 1:
            raise ValueError("writer_parallelism must be at least 1")
        if self.writer_max_request_bytes is not None and self.writer_max_request_bytes < 1024:
            raise ValueError("writer_max_request_bytes must be null or at least 1024")
        if self.writer_max_response_bytes < 1024:
            raise ValueError("writer_max_response_bytes must be at least 1024")
        if self.writer_repair_attempts < 0:
            raise ValueError("writer_repair_attempts must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)

    def finalizer_ui_values(self) -> tuple[str, str, bool]:
        """Return common creative settings, without mutating differing legacy stages."""
        if self.finalizer_model:
            return self.finalizer_model, self.finalizer_reasoning, True
        models = {v for v in (self.planner_model, self.writer_model, self.gateway_prime_model) if v}
        reasons = {v for v in (self.planner_reasoning, self.writer_reasoning, self.gateway_prime_reasoning) if v}
        return (next(iter(models)) if len(models) == 1 else "", next(iter(reasons)) if len(reasons) == 1 else "", len(models) <= 1 and len(reasons) <= 1)

    def apply_unified_finalizer(self, model: str, reasoning: str) -> None:
        """Explicit Save mapping for all Phase 6-9 creative AI stages."""
        self.finalizer_model = model
        self.finalizer_reasoning = reasoning
        self.gateway_prime_model = model
        self.gateway_prime_reasoning = reasoning
        self.planner_model = model
        self.planner_reasoning = reasoning
        self.writer_model = model
        self.writer_reasoning = reasoning

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppSettings:
        """Create AppSettings from dict, safely migrating legacy gateway_model and thinking if needed."""
        data_copy = dict(data)
        # Transitional Phase 3 names become provider-neutral Scanner configuration.
        # Preserve the literal user-selected ID; never infer a provider from it.
        if "scanner_model" not in data_copy:
            data_copy["scanner_model"] = str(data_copy.get("gateway_sub_model") or data_copy.get("gateway_model") or "")
        if "scanner_reasoning" not in data_copy:
            data_copy["scanner_reasoning"] = str(data_copy.get("gateway_sub_reasoning") or "")
        if "planner_model" not in data_copy:
            data_copy["planner_model"] = str(data_copy.get("gateway_prime_model") or "")
        if "planner_reasoning" not in data_copy:
            data_copy["planner_reasoning"] = str(data_copy.get("gateway_prime_reasoning") or "")
        if "vision_model" not in data_copy:
            data_copy["vision_model"] = str(data_copy.get("scanner_model") or data_copy.get("gateway_sub_model") or "")
        if "writer_model" not in data_copy:
            data_copy["writer_model"] = str(data_copy.get("planner_model") or data_copy.get("gateway_prime_model") or "")
        if "writer_reasoning" not in data_copy:
            data_copy["writer_reasoning"] = str(data_copy.get("planner_reasoning") or data_copy.get("gateway_prime_reasoning") or "")
        if "finalizer_model" not in data_copy:
            models = {str(data_copy.get(k) or "") for k in ("planner_model", "writer_model", "gateway_prime_model")} - {""}
            data_copy["finalizer_model"] = next(iter(models)) if len(models) == 1 else ""
        if "finalizer_reasoning" not in data_copy:
            reasons = {str(data_copy.get(k) or "") for k in ("planner_reasoning", "writer_reasoning", "gateway_prime_reasoning")} - {""}
            data_copy["finalizer_reasoning"] = next(iter(reasons)) if len(reasons) == 1 else ""
        # Migrate legacy single gateway_model to dual sub/prime defaults
        if "gateway_model" in data_copy and "gateway_sub_model" not in data_copy:
            old_model = data_copy.get("gateway_model")
            if old_model:
                data_copy["gateway_sub_model"] = old_model
        if "gateway_sub_model" in data_copy and "gateway_model" not in data_copy:
            data_copy["gateway_model"] = data_copy["gateway_sub_model"]
        if "gateway_thinking" in data_copy:
            if "gateway_sub_reasoning" not in data_copy and data_copy["gateway_thinking"]:
                data_copy["gateway_sub_reasoning"] = "high"
            if "gateway_prime_reasoning" not in data_copy and data_copy["gateway_thinking"]:
                data_copy["gateway_prime_reasoning"] = "high"
        # Migrate old settings without canvas_auto to auto default
        if "canvas_auto" not in data_copy or data_copy.get("canvas_auto") is None:
            data_copy["canvas_auto"] = True
        allowed = cls.__dataclass_fields__
        filtered = {k: v for k, v in data_copy.items() if k in allowed}
        return cls(**filtered)


class SettingsManager:
    """Manages reading and writing application settings in LOCALAPPDATA."""

    def __init__(self, persistence: Optional[ProjectPersistence] = None, storage_root: Optional[Path | str] = None) -> None:
        if persistence is not None:
            self.persistence = persistence
        else:
            self.persistence = ProjectPersistence(storage_root=storage_root)
        self.last_load_warning: str | None = None
        self.corrupt_backup_path: Path | None = None

    def _preserve_corrupt_settings(self, exc: Exception) -> None:
        source = self.persistence.settings_dir / "settings.json"
        if not source.is_file():
            return
        if self.corrupt_backup_path is None or not self.corrupt_backup_path.is_file():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = source.with_name(f"settings.corrupt.{stamp}.json")
            counter = 1
            while backup.exists():
                backup = source.with_name(f"settings.corrupt.{stamp}.{counter}.json")
                counter += 1
            shutil.copy2(source, backup)
            self.corrupt_backup_path = backup
        self.last_load_warning = (
            "Settings could not be read and defaults are being used. "
            f"The original file was preserved at '{self.corrupt_backup_path}': {exc}"
        )

    def load(self) -> AppSettings:
        """Load settings from persistence; returns default AppSettings if absent or corrupted."""
        settings_path = self.persistence.settings_dir / "settings.json"
        if not settings_path.is_file():
            self.last_load_warning = None
            return AppSettings()
        try:
            raw = self.persistence.load_settings()
            loaded = AppSettings.from_dict(raw)
            self.last_load_warning = None
            return loaded
        except Exception as exc:
            self._preserve_corrupt_settings(exc)
            return AppSettings()

    def save(self, settings: AppSettings) -> Path:
        """Persist settings to LOCALAPPDATA atomically."""
        merged: Dict[str, Any] = {}
        try:
            merged.update(self.persistence.load_settings())
        except Exception as exc:
            self._preserve_corrupt_settings(exc)
        merged.update(settings.to_dict())
        return self.persistence.save_settings(merged)
