"""Structured, zero-AI runtime diagnostics for source and portable builds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from toolrecap_v4.__version__ import __version__
from toolrecap_v4.media import detect_gpu_encoder, find_binary, probe_encoder_usable
from toolrecap_v4.output_paths import resolve_publication_root
from toolrecap_v4.persistence import DEFAULT_APP_DIR_NAME
from toolrecap_v4.runtime import application_root, bundled_runtime_root, package_mode
from toolrecap_v4.schemas.schema import get_project_schema, get_schema_validator
from toolrecap_v4.settings import AppSettings

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass(frozen=True)
class SelfCheckItem:
    id: str
    category: str
    status: str
    message: str
    fatal: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelfCheckResult:
    app_version: str
    mode: str
    application_root: str
    resource_root: str
    checks: tuple[SelfCheckItem, ...]

    @property
    def status(self) -> str:
        if any(item.status == FAIL and item.fatal for item in self.checks):
            return FAIL
        if any(item.status in (WARN, FAIL) for item in self.checks):
            return WARN
        return PASS

    @property
    def exit_code(self) -> int:
        return 1 if self.status == FAIL else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "selfcheck_version": "toolrecap-selfcheck-v1",
            "status": self.status,
            "app_version": self.app_version,
            "mode": self.mode,
            "application_root": self.application_root,
            "resource_root": self.resource_root,
            "checks": [asdict(item) for item in self.checks],
        }


def _managed_state_root(override: Path | str | None = None) -> Path:
    if override is not None:
        return Path(override).resolve()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return (Path(local_app_data) / DEFAULT_APP_DIR_NAME).resolve()
    return (Path.home() / "AppData" / "Local" / DEFAULT_APP_DIR_NAME).resolve()


def _read_settings_without_mutation(root: Path) -> tuple[AppSettings, str]:
    settings_path = root / "settings" / "settings.json"
    if not settings_path.is_file():
        return AppSettings(), "fresh defaults"
    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("settings root must be an object")
        return AppSettings.from_dict(raw), "existing settings loaded"
    except Exception as exc:
        return AppSettings(), f"existing settings unreadable; safe defaults used: {type(exc).__name__}"


def _module_check(module_names: tuple[str, ...]) -> tuple[bool, dict[str, str], str]:
    versions: dict[str, str] = {}
    try:
        for name in module_names:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "available"))
        return True, versions, "runtime modules imported"
    except Exception as exc:
        return False, versions, f"{type(exc).__name__}: {exc}"


def _binary_version(name: str) -> tuple[Path, str]:
    binary = find_binary(name)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    result = subprocess.run(
        [str(binary), "-version"], capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=15, creationflags=creationflags,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{name} exited with code {result.returncode}")
    lines = (result.stdout or result.stderr or "").splitlines()
    return binary, lines[0] if lines else "version output unavailable"


def collect_selfcheck(*, storage_root: Path | str | None = None) -> SelfCheckResult:
    """Collect diagnostics without projects, media, network, AI, voice, render, or update mutation."""
    checks: list[SelfCheckItem] = []
    state_root = _managed_state_root(storage_root)
    checks.append(SelfCheckItem(
        "application.runtime", "APPLICATION", PASS, "application runtime located", True,
        {"version": __version__, "mode": package_mode(), "root": str(application_root())},
    ))
    try:
        state_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="selfcheck_", dir=state_root, delete=False) as handle:
            probe_path = Path(handle.name)
        probe_path.unlink(missing_ok=True)
        checks.append(SelfCheckItem(
            "application.state_root", "APPLICATION", PASS, "managed state root is writable", True,
            {"root": str(state_root), "location": "LOCALAPPDATA"},
        ))
    except Exception as exc:
        checks.append(SelfCheckItem(
            "application.state_root", "APPLICATION", FAIL,
            f"managed state root is not writable: {type(exc).__name__}", True, {"root": str(state_root)},
        ))

    try:
        schema = get_project_schema()
        get_schema_validator()
        schema_version = schema.get("properties", {}).get("schema_version", {}).get("const")
        if schema_version != "3.0":
            raise ValueError(f"unexpected schema version {schema_version!r}")
        checks.append(SelfCheckItem(
            "resources.schema", "RESOURCES", PASS, "schema 3.0 loaded and validated", True,
            {"schema_version": schema_version},
        ))
    except Exception as exc:
        checks.append(SelfCheckItem(
            "resources.schema", "RESOURCES", FAIL,
            f"schema unavailable: {type(exc).__name__}: {exc}", True,
        ))

    binaries: dict[str, Path] = {}
    for name in ("ffmpeg", "ffprobe"):
        try:
            binary, version = _binary_version(name)
            binaries[name] = binary
            checks.append(SelfCheckItem(
                f"media.{name}", "MEDIA", PASS, f"{name} is available", True,
                {"path": str(binary), "version": version},
            ))
        except Exception as exc:
            checks.append(SelfCheckItem(
                f"media.{name}", "MEDIA", FAIL,
                f"{name} unavailable: {type(exc).__name__}: {exc}", True,
            ))

    if "ffmpeg" in binaries:
        gpu = detect_gpu_encoder(binaries["ffmpeg"], refresh=True)
        cpu_ok = probe_encoder_usable(binaries["ffmpeg"], "libx264")
        if gpu.available:
            checks.append(SelfCheckItem(
                "encoder.video", "GPU / ENCODER", PASS, f"working GPU encoder: {gpu.encoder}", False,
                {"gpu": gpu.gpu_name, "encoder": gpu.encoder, "cpu_fallback": cpu_ok},
            ))
        elif cpu_ok:
            checks.append(SelfCheckItem(
                "encoder.video", "GPU / ENCODER", WARN,
                "no working GPU encoder detected; CPU libx264 fallback is available", False,
                {"gpu": gpu.gpu_name, "encoder": "libx264", "reason": gpu.reason},
            ))
        else:
            checks.append(SelfCheckItem(
                "encoder.video", "GPU / ENCODER", FAIL,
                "neither a working GPU encoder nor CPU libx264 fallback is available", True,
                {"reason": gpu.reason},
            ))

    ocr_ok, ocr_versions, ocr_message = _module_check(("onnxruntime", "rapidocr_onnxruntime"))
    checks.append(SelfCheckItem(
        "ocr.runtime", "OCR / STT RUNTIME", PASS if ocr_ok else FAIL,
        ocr_message if ocr_ok else f"OCR runtime unavailable: {ocr_message}", True, ocr_versions,
    ))
    if ocr_ok:
        from toolrecap_v4.analysis.source_prep.subtitles.ocr import OcrAdapter, OcrModelManager
        ocr_models = (
            OcrModelManager(model_dir=state_root / "models" / "ocr").is_ready()
            or OcrAdapter.bundled_models_ready()
        )
        checks.append(SelfCheckItem(
            "ocr.models", "OCR / STT RUNTIME", PASS if ocr_models else WARN,
            "OCR models are available" if ocr_models else "OCR models are optional/downloadable and not currently present",
        ))

    stt_ok, stt_versions, stt_message = _module_check(("ctranslate2", "faster_whisper", "huggingface_hub"))
    checks.append(SelfCheckItem(
        "stt.runtime", "OCR / STT RUNTIME", PASS if stt_ok else FAIL,
        stt_message if stt_ok else f"STT runtime unavailable: {stt_message}", True, stt_versions,
    ))
    if stt_ok:
        from toolrecap_v4.analysis.source_prep.stt import SttModelManager
        stt_models = SttModelManager(model_dir=state_root / "models" / "stt" / "tiny").is_ready()
        checks.append(SelfCheckItem(
            "stt.models", "OCR / STT RUNTIME", PASS if stt_models else WARN,
            "STT model is available" if stt_models else "STT model is optional/downloadable and not currently present",
        ))

    settings, settings_message = _read_settings_without_mutation(state_root)
    checks.append(SelfCheckItem(
        "settings.load", "APPLICATION", WARN if "unreadable" in settings_message else PASS,
        settings_message, False, {"settings_path": str(state_root / "settings" / "settings.json")},
    ))
    endpoint = urlparse(settings.gateway_endpoint)
    endpoint_ok = endpoint.scheme in ("http", "https") and bool(endpoint.netloc) and not endpoint.username
    secret_present = (state_root / "secrets" / "credentials.dpapi").is_file()
    finalizer_model = settings.finalizer_ui_values()[0]
    config_missing = not settings.scanner_model or not finalizer_model or not secret_present
    checks.append(SelfCheckItem(
        "gateway.configuration", "AI GATEWAY CONFIG", WARN if config_missing or not endpoint_ok else PASS,
        "Gateway configuration is complete" if not config_missing and endpoint_ok else "Gateway configuration is incomplete; AI production stages may require Settings",
        False,
        {
            "endpoint_configured": endpoint_ok,
            "scanner_model_configured": bool(settings.scanner_model),
            "finalizer_model_configured": bool(finalizer_model),
            "api_key": "configured" if secret_present else "not configured",
        },
    ))

    voice_mode = settings.voice_mode if settings.voice_mode in ("auto", "local", "remote") else "invalid"
    local_ok = bool(urlparse(settings.voice_local_url).netloc)
    remote_ok = bool(urlparse(settings.voice_remote_url).netloc)
    voice_ok = voice_mode != "invalid" and (
        local_ok if voice_mode == "local" else remote_ok if voice_mode == "remote" else (local_ok or remote_ok)
    )
    checks.append(SelfCheckItem(
        "voice.configuration", "VOICESTUDIO", PASS if voice_ok else WARN,
        "VoiceStudio configuration is present" if voice_ok else "VoiceStudio configuration requires attention",
        False, {"mode": voice_mode, "local_configured": local_ok, "remote_configured": remote_ok},
    ))

    try:
        synthetic_working = state_root.parent / "ToolRecap SelfCheck Source"
        resolved = resolve_publication_root(manual_output_dir="", working_folder=synthetic_working)
        checks.append(SelfCheckItem(
            "output.resolver", "OUTPUT", PASS, "Phase 12 resolver is available without creating output", True,
            {"rule": "sibling Outputs_<working-folder-name>", "created": resolved.exists()},
        ))
    except Exception as exc:
        checks.append(SelfCheckItem(
            "output.resolver", "OUTPUT", FAIL, f"output resolver unavailable: {exc}", True,
        ))

    try:
        from toolrecap_v4.updater import safe_extract_zip, validate_package  # noqa: F401
        checks.append(SelfCheckItem(
            "updater.runtime", "UPDATER", PASS, "updater validation and safe extraction are available", True,
        ))
    except Exception as exc:
        checks.append(SelfCheckItem(
            "updater.runtime", "UPDATER", FAIL, f"updater runtime unavailable: {exc}", True,
        ))

    return SelfCheckResult(
        __version__, package_mode(), str(application_root()), str(bundled_runtime_root()), tuple(checks),
    )


def determine_report_path(argv: Optional[list[str]] = None) -> Optional[Path]:
    args = sys.argv if argv is None else argv
    for flag in ("--selfcheck-json", "--report", "--json-report"):
        if flag in args:
            index = args.index(flag)
            if index + 1 >= len(args):
                raise ValueError(f"{flag} requires a file path")
            return Path(args[index + 1]).resolve()
    if "--selfcheck" in args:
        index = args.index("--selfcheck")
        if index + 1 < len(args) and not args[index + 1].startswith("-"):
            return Path(args[index + 1]).resolve()
    return None


def run_selfcheck(
    argv: Optional[list[str]] = None,
    *,
    storage_root: Path | str | None = None,
    collector: Callable[..., SelfCheckResult] = collect_selfcheck,
) -> int:
    """Run self-check, optionally write JSON, and return a fatal-aware exit code."""
    try:
        report_path = determine_report_path(argv)
        result = collector(storage_root=storage_root)
    except Exception as exc:
        report_path = None
        result = SelfCheckResult(
            __version__, package_mode(), str(application_root()), str(bundled_runtime_root()),
            (SelfCheckItem("selfcheck.internal", "APPLICATION", FAIL, f"self-check failed: {type(exc).__name__}: {exc}", True),),
        )
    payload = result.to_dict()
    if report_path is not None:
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            if sys.stderr is not None:
                print(f"Unable to write self-check report: {exc}", file=sys.stderr)
            return 2
    if sys.stdout is not None:
        try:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception:
            pass
    return result.exit_code
