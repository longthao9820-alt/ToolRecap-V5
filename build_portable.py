"""Build and accept the ToolRecap V5 Windows one-folder portable distribution."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from toolrecap_v4.__version__ import __version__
from toolrecap_v4.updater.archive import parse_sha256_content, safe_extract_zip, verify_checksum
from toolrecap_v4.updater.validator import audit_package_contents, validate_package

PROJECT_ROOT = Path(__file__).resolve().parent
DIST_BASE = PROJECT_ROOT / "dist"
DIST_DIR = DIST_BASE / "ToolRecapV5"
BUILD_DIR = PROJECT_ROOT / "build" / "pyinstaller"
RELEASE_DIR = PROJECT_ROOT / "release"
RELEASE_ZIP = RELEASE_DIR / f"ToolRecapV5-v{__version__}-windows-portable.zip"
RELEASE_SHA = RELEASE_DIR / f"{RELEASE_ZIP.name}.sha256.txt"
PACKAGE_FORMAT = "toolrecap-portable-v1"


def calculate_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def locate_ffmpeg() -> tuple[Path, Path, Path]:
    """Locate build-time FFmpeg inputs without persisting a developer path."""
    configured = os.environ.get("TOOLRECAP_FFMPEG_DISTRIBUTION", "").strip()
    if configured:
        configured_path = Path(configured).resolve()
        bin_dir = configured_path / "bin" if configured_path.is_dir() else configured_path.parent
        root = configured_path if (configured_path / "bin").is_dir() else bin_dir.parent
    else:
        ffmpeg_match = shutil.which("ffmpeg")
        _require(bool(ffmpeg_match), "FFmpeg is required to build the portable package")
        bin_dir = Path(str(ffmpeg_match)).resolve().parent
        root = bin_dir.parent
    ffmpeg = bin_dir / "ffmpeg.exe"
    ffprobe = bin_dir / "ffprobe.exe"
    license_candidates = (root / "LICENSE", root / "LICENSE.txt", root / "COPYING.GPLv3")
    license_path = next((path for path in license_candidates if path.is_file()), None)
    _require(ffmpeg.is_file(), f"Missing build-time FFmpeg binary: {ffmpeg}")
    _require(ffprobe.is_file(), f"Missing build-time FFprobe binary: {ffprobe}")
    _require(license_path is not None, f"Missing FFmpeg license under: {root}")
    return ffmpeg, ffprobe, Path(license_path)


def pyinstaller_command(schema_source: Path) -> list[str]:
    """Return the deterministic one-folder PyInstaller command."""
    collect_packages = (
        "windows_toasts", "winrt", "onnxruntime", "rapidocr_onnxruntime",
        "ctranslate2", "faster_whisper", "huggingface_hub",
    )
    command = [
        sys.executable, "-m", "PyInstaller", "--onedir", "--windowed",
        "--name=ToolRecapV5", "--noconfirm", "--clean",
        f"--distpath={DIST_BASE}", f"--workpath={BUILD_DIR}", f"--specpath={BUILD_DIR}",
        f"--add-data={schema_source};toolrecap_v4/schemas",
        "--hidden-import=tkinter",
        "--hidden-import=toolrecap_v4.selfcheck",
        "--hidden-import=toolrecap_v4.update_helper",
        "--hidden-import=toolrecap_v4.updater.helper",
        "--hidden-import=toolrecap_v4.updater.validator",
        "--hidden-import=toolrecap_v4.updater.archive",
        "--hidden-import=toolrecap_v4.updater.manager",
        "--hidden-import=toolrecap_v4.schemas.schema",
        "--hidden-import=toolrecap_v4.v5.analyzer",
        "--hidden-import=toolrecap_v4.v5.workflow",
        "--exclude-module=pytest",
    ]
    for package in collect_packages:
        command.append(f"--collect-all={package}")
    command.append(str(PROJECT_ROOT / "main.py"))
    return command


def _clean_runtime_environment(local_app_data: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PATH"] = os.pathsep.join((
        str(Path(environment.get("SYSTEMROOT", r"C:\Windows")) / "System32"),
        environment.get("SYSTEMROOT", r"C:\Windows"),
    ))
    environment["LOCALAPPDATA"] = str(local_app_data)
    for key in list(environment):
        if key.startswith("PYTHON") or key.startswith("TOOLRECAP_FFMPEG") or key in ("FFMPEG_PATH", "FFPROBE_PATH"):
            environment.pop(key, None)
    return environment


def _check_by_id(report: dict, check_id: str) -> dict:
    for check in report.get("checks", []):
        if check.get("id") == check_id:
            return check
    raise RuntimeError(f"Packaged self-check omitted required check: {check_id}")


def run_packaged_selfcheck(executable: Path, *, cwd: Path, report_path: Path, environment: dict[str, str]) -> dict:
    report_path.unlink(missing_ok=True)
    result = subprocess.run(
        [str(executable), "--selfcheck", "--selfcheck-json", str(report_path)],
        cwd=cwd, env=environment, capture_output=True, text=True, timeout=180,
    )
    _require(result.returncode == 0, f"Packaged self-check failed ({result.returncode}): {result.stderr or result.stdout}")
    _require(report_path.is_file(), f"Packaged self-check did not create: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("status") in ("PASS", "WARN"), f"Packaged self-check reported fatal failure: {report}")
    _require(report.get("mode") == "frozen", "Self-check did not identify frozen package mode")
    for check_id in (
        "resources.schema", "media.ffmpeg", "media.ffprobe", "ocr.runtime", "ocr.models", "stt.runtime",
        "settings.load", "output.resolver", "updater.runtime",
    ):
        _require(_check_by_id(report, check_id).get("status") != "FAIL", f"Required packaged check failed: {check_id}")
    _require(_check_by_id(report, "ocr.models").get("status") == "PASS",
             "Bundled OCR models are unavailable in the frozen package")
    return report


def _create_release_zip() -> str:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    RELEASE_ZIP.unlink(missing_ok=True)
    RELEASE_SHA.unlink(missing_ok=True)
    with zipfile.ZipFile(RELEASE_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file_path in sorted(DIST_DIR.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(DIST_DIR))
    checksum = calculate_sha256(RELEASE_ZIP)
    RELEASE_SHA.write_text(f"{checksum}  {RELEASE_ZIP.name}\n", encoding="utf-8")
    _require(parse_sha256_content(RELEASE_SHA.read_text(encoding="utf-8")) == checksum, "Checksum file parse failed")
    _require(verify_checksum(RELEASE_ZIP, checksum), "Release checksum verification failed")
    return checksum


def build() -> None:
    print("[1/7] Verifying build prerequisites...")
    _require(sys.platform == "win32", "Portable build must run on Windows")
    _require(sys.version_info >= (3, 12), f"Requires Python 3.12+, got {sys.version}")
    importlib.import_module("PyInstaller")
    for module_name in ("onnxruntime", "rapidocr_onnxruntime", "ctranslate2", "faster_whisper", "huggingface_hub"):
        importlib.import_module(module_name)
    ffmpeg, ffprobe, ffmpeg_license = locate_ffmpeg()
    schema_source = PROJECT_ROOT / "toolrecap_v4" / "schemas" / "recap_v3_schema.json"
    _require(schema_source.is_file(), f"Missing schema: {schema_source}")

    print("[2/7] Cleaning exact ignored build targets and running PyInstaller one-folder build...")
    shutil.rmtree(DIST_DIR, ignore_errors=True)
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(pyinstaller_command(schema_source), cwd=PROJECT_ROOT, check=False)
    _require(result.returncode == 0, f"PyInstaller failed with exit code {result.returncode}")
    executable = DIST_DIR / "ToolRecapV5.exe"
    _require(executable.is_file(), f"Portable executable was not created: {executable}")
    ocr_model_files = (
        "ch_PP-OCRv4_det_infer.onnx",
        "ch_PP-OCRv4_rec_infer.onnx",
        "ch_ppocr_mobile_v2.0_cls_infer.onnx",
    )
    ocr_model_paths = [
        DIST_DIR / "_internal" / "rapidocr_onnxruntime" / "models" / name
        for name in ocr_model_files
    ]
    _require(all(path.is_file() and path.stat().st_size > 0 for path in ocr_model_paths),
             "Portable OCR runtime is missing one or more bundled ONNX models")

    print("[3/7] Adding required runtime resources without user state or model caches...")
    shutil.copy2(ffmpeg, DIST_DIR / "ffmpeg.exe")
    shutil.copy2(ffprobe, DIST_DIR / "ffprobe.exe")
    shutil.copy2(ffmpeg_license, DIST_DIR / "FFMPEG_LICENSE.txt")
    schema_destination = DIST_DIR / "schemas" / "recap_v3_schema.json"
    schema_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(schema_source, schema_destination)
    if (PROJECT_ROOT / "README.md").is_file():
        shutil.copy2(PROJECT_ROOT / "README.md", DIST_DIR / "README.md")
    for prompt_name in ("ToolRecapV4_Recap_Prompt.txt", "ToolRecapV4_Highlight_Prompt.txt"):
        shutil.copy2(PROJECT_ROOT / prompt_name, DIST_DIR / prompt_name)
    notes_name = f"RELEASE_NOTES_{__version__}.md"
    shutil.copy2(PROJECT_ROOT / notes_name, DIST_DIR / notes_name)
    marker = {
        "package_format": PACKAGE_FORMAT,
        "schema_version": "3.0",
        "version": __version__,
        "layout": "pyinstaller-onedir",
        "required_files": [
            "ToolRecapV5.exe", "ffmpeg.exe", "ffprobe.exe",
            "ToolRecapV4_Recap_Prompt.txt", "ToolRecapV4_Highlight_Prompt.txt",
            notes_name,
            "FFMPEG_LICENSE.txt", "schemas/recap_v3_schema.json",
            "_internal/base_library.zip",
            f"_internal/python{sys.version_info.major}{sys.version_info.minor}.dll",
            *(str(path.relative_to(DIST_DIR)).replace("\\", "/") for path in ocr_model_paths),
        ],
        "bundled_models": False,
        "bundled_ocr_models": True,
        "user_state_bundled": False,
    }
    (DIST_DIR / "package_marker.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")

    print("[4/7] Validating and auditing portable contents...")
    validate_package(DIST_DIR, expected_version=__version__)
    audit_package_contents(DIST_DIR)

    print("[5/7] Running packaged self-check with Python and system FFmpeg removed from PATH...")
    with tempfile.TemporaryDirectory(prefix="toolrecap_phase13_") as temporary:
        temp_root = Path(temporary)
        state_root = temp_root / "LocalAppData"
        environment = _clean_runtime_environment(state_root)
        normal_report = temp_root / "normal.json"
        run_packaged_selfcheck(executable, cwd=DIST_DIR, report_path=normal_report, environment=environment)

        print("[6/7] Relocating through spaces/Unicode and running from a different CWD...")
        relocated = temp_root / "ToolRecap Portable Test" / "Bản thử" / "ToolRecapV5"
        shutil.copytree(DIST_DIR, relocated)
        different_cwd = temp_root / "Different Current Working Directory"
        different_cwd.mkdir(parents=True)
        relocated_report = temp_root / "relocated.json"
        relocated_data = run_packaged_selfcheck(
            relocated / "ToolRecapV5.exe", cwd=different_cwd,
            report_path=relocated_report, environment=environment,
        )
        _require(Path(relocated_data["application_root"]).resolve() == relocated.resolve(), "Relocated application root was incorrect")

    print("[7/7] Creating/validating release archive...")
    checksum = _create_release_zip()
    with tempfile.TemporaryDirectory(prefix="toolrecap_zip_") as extracted:
        extracted_root = Path(extracted)
        safe_extract_zip(RELEASE_ZIP, extracted_root)
        validate_package(extracted_root, expected_version=__version__)

    total_size = sum(path.stat().st_size for path in DIST_DIR.rglob("*") if path.is_file())
    print(f"Portable root: {DIST_DIR}")
    print(f"Portable size: {total_size / (1024 * 1024):.2f} MiB")
    print(f"Release zip: {RELEASE_ZIP}")
    print(f"Release SHA256: {checksum}")
    print("PHASE 13 REAL PORTABLE ACCEPTANCE PASSED")


if __name__ == "__main__":
    build()
