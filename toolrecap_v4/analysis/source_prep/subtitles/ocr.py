"""Managed OCR model runtime manager, RapidOCR adapter, quality gate, and cropped Vision OCR fallback."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import io
import os
from pathlib import Path
import re
from typing import Any, Callable, Sequence
import urllib.request

from PIL import Image

from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError, ToolRecapError
from toolrecap_v4.persistence import get_storage_root
from .parsers import strip_formatting_tags


@dataclass(frozen=True)
class OcrModelInfo:
    """Explicit metadata and checksum identity for an OCR model file."""

    key: str
    filename: str
    url: str
    sha256: str
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Official ModelScope pinned models for RapidOCR PP-OCRv4
DEFAULT_OCR_MODELS: dict[str, OcrModelInfo] = {
    "det": OcrModelInfo(
        key="det",
        filename="ch_PP-OCRv4_det_mobile.onnx",
        url="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.onnx",
        sha256="d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9",
    ),
    "rec": OcrModelInfo(
        key="rec",
        filename="ch_PP-OCRv4_rec_mobile.onnx",
        url="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile.onnx",
        sha256="48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b",
    ),
    "cls": OcrModelInfo(
        key="cls",
        filename="ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        url="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        sha256="e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    ),
}


@dataclass(frozen=True)
class OcrModelManifest:
    """Explicit runtime and model identity/version/checksum manifest."""

    runtime: str = "rapidocr"
    version: str = "v3.9.2"
    models: dict[str, OcrModelInfo] = field(default_factory=lambda: dict(DEFAULT_OCR_MODELS))

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "version": self.version,
            "models": {k: m.to_dict() for k, m in self.models.items()},
        }

    def compute_manifest_hash(self) -> str:
        """Deterministic hash of runtime, version, and model checksums."""
        items = [f"runtime:{self.runtime}", f"version:{self.version}"]
        for k in sorted(self.models.keys()):
            m = self.models[k]
            items.append(f"{k}:{m.filename}:{m.sha256.lower()}")
        raw = "|".join(items)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


DEFAULT_OCR_MANIFEST = OcrModelManifest()


def compute_ocr_dependency_signature(
    manifest: OcrModelManifest,
    extra_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute content-addressable dependency signature for OCR runtime and models."""
    settings = dict(extra_settings or {})
    manifest_hash = manifest.compute_manifest_hash()
    settings_raw = repr(sorted(settings.items()))
    combined = f"ocr_manifest:{manifest_hash}|settings:{settings_raw}"
    ocr_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

    return {
        "type": "ocr",
        "runtime": manifest.runtime,
        "version": manifest.version,
        "manifest_hash": manifest_hash,
        "settings": settings,
        "ocr_hash": ocr_hash,
    }


@dataclass(frozen=True)
class OcrResult:
    """Result of OCR processing on a single subtitle image crop."""

    text: str
    confidence: float
    is_valid: bool
    reason: str = ""
    source: str = "rapidocr"  # 'rapidocr' | 'vision_ai' | 'rejected' | 'empty'
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def quality_gate(
    text: str,
    confidence: float,
    min_confidence: float = 0.50,
) -> tuple[bool, str]:
    """Adapt RapidOCR quality gate to filter empty, low-confidence, or garbage strings.
    
    Returns (is_valid, reason).
    """
    clean = strip_formatting_tags(text).strip()
    if not clean:
        return False, "Empty text after normalization"

    if confidence < min_confidence:
        return False, f"Low confidence ({confidence:.2f} < {min_confidence:.2f})"

    # Filter repetitive garbage sequences (e.g. '||||||' or '-------')
    # before the generic alphanumeric check so diagnostics retain the useful cause.
    stripped = clean.replace(" ", "")
    if len(stripped) >= 4 and len(set(stripped)) <= 2:
        return False, "Repetitive garbage sequence"

    # Check for presence of alphanumeric words (including Latin and Vietnamese Unicode)
    has_alnum = bool(re.search(r"[a-zA-Z0-9\u00C0-\u024F\u1EA0-\u1EF9]", clean))
    if not has_alnum:
        return False, "No valid alphanumeric characters"

    return True, "Valid"


def validate_cropped_image(
    image: Image.Image,
    max_width: int = 1900,
    max_height: int = 1000,
) -> None:
    """Ensure image is a cropped subtitle bitmap, rejecting full video frames.
    
    Invariant: OCR accepts only cropped bitmap images, never full 1080p/4K scenes.
    """
    width = getattr(image, "width", 0)
    height = getattr(image, "height", 0)
    if width >= max_width and height >= max_height:
        raise ValueError(
            f"Bounding box violation: image dimensions ({width}x{height}) resemble full video frame; "
            f"OCR accepts only cropped subtitle bounding boxes"
        )


class OcrModelManager:
    """Manages local RapidOCR model runtime with versioned manifests and atomic staging."""

    def __init__(
        self,
        model_dir: Path | str | None = None,
        manifest: OcrModelManifest | None = None,
        downloader: Callable[..., Any] | None = None,
    ) -> None:
        if model_dir is not None:
            self.model_dir = Path(model_dir).resolve()
        else:
            self.model_dir = (get_storage_root() / "models" / "ocr").resolve()
        self.manifest = manifest or DEFAULT_OCR_MANIFEST
        self.downloader = downloader

    def get_model_paths(self) -> dict[str, Path]:
        """Return mapping of model key to target file path."""
        return {
            key: self.model_dir / info.filename
            for key, info in self.manifest.models.items()
        }

    def are_models_available(self) -> bool:
        """Check whether all manifest model files exist on disk with non-zero size."""
        paths = self.get_model_paths()
        if not paths:
            return False
        return all(p.is_file() and p.stat().st_size > 0 for p in paths.values())

    def verify_hashes(self) -> bool:
        """Validate SHA-256 checksums of local model files. Never fake checksums."""
        paths = self.get_model_paths()
        for key, info in self.manifest.models.items():
            path = paths.get(key)
            if not path or not path.is_file():
                return False
            expected = info.sha256.strip().lower()
            if not expected:
                return False

            hasher = hashlib.sha256()
            with path.open("rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    hasher.update(chunk)
            if hasher.hexdigest().lower() != expected:
                return False

        return True

    def is_ready(self) -> bool:
        """Check whether model manager is ready for offline local operation."""
        return self.are_models_available() and self.verify_hashes()

    def download_models(
        self,
        cancellation_token: CancellationToken | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> bool:
        """Download missing or corrupt models atomically with .partial staging.
        
        Invariants:
        - Downloads to .partial files first.
        - Verifies checksum before atomic promotion.
        - Cancelled/failed download leaves NO ready manifest and cleans partial files.
        """
        if cancellation_token:
            cancellation_token.check_cancelled()

        self.model_dir.mkdir(parents=True, exist_ok=True)
        paths = self.get_model_paths()

        for key, info in self.manifest.models.items():
            if cancellation_token:
                cancellation_token.check_cancelled()

            dest = paths[key]
            expected_hash = info.sha256.strip().lower()

            # Cache hit check
            if dest.is_file():
                if self._verify_single_file_hash(dest, expected_hash):
                    continue
                # Corrupted: remove and redownload
                dest.unlink(missing_ok=True)

            partial_dest = dest.with_suffix(dest.suffix + ".partial")
            if partial_dest.is_file():
                partial_dest.unlink(missing_ok=True)

            try:
                if self.downloader is not None:
                    self.downloader(
                        url=info.url,
                        dest_path=partial_dest,
                        expected_sha256=expected_hash,
                        cancellation_token=cancellation_token,
                        progress_callback=progress_callback,
                    )
                else:
                    self._default_download(
                        url=info.url,
                        partial_dest=partial_dest,
                        cancellation_token=cancellation_token,
                        progress_callback=progress_callback,
                    )

                if cancellation_token:
                    cancellation_token.check_cancelled()

                # Verify checksum before promotion
                if not self._verify_single_file_hash(partial_dest, expected_hash):
                    partial_dest.unlink(missing_ok=True)
                    raise ToolRecapError(
                        f"Checksum mismatch for OCR model {info.filename}: expected {expected_hash}"
                    )

                # Atomic rename
                os.replace(partial_dest, dest)

            except Exception:
                if partial_dest.is_file():
                    partial_dest.unlink(missing_ok=True)
                raise

        # Write manifest file upon verified completion
        self._write_manifest_record()
        return self.is_ready()

    def _verify_single_file_hash(self, path: Path, expected_hash: str) -> bool:
        if not path.is_file() or not expected_hash:
            return False
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest().lower() == expected_hash.lower()

    def _default_download(
        self,
        url: str,
        partial_dest: Path,
        cancellation_token: CancellationToken | None,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> None:
        req = urllib.request.Request(url, headers={"User-Agent": "ToolRecap/4.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total_bytes = int(resp.headers.get("Content-Length") or 0)
            downloaded = 0
            with partial_dest.open("wb") as out_f:
                while True:
                    if cancellation_token:
                        cancellation_token.check_cancelled()
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_bytes, f"Downloading {partial_dest.name}")

    def _write_manifest_record(self) -> None:
        import json
        manifest_file = self.model_dir / "manifest.json"
        data = self.manifest.to_dict()
        data["manifest_hash"] = self.manifest.compute_manifest_hash()
        data["ready"] = True
        manifest_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


class VisionOcrAdapter:
    """Adapter for calling Phase 2 Gateway submit_image_chat solely for cropped subtitle OCR fallback.
    
    Invariants:
    - Bounded to cropped subtitle bounding box images only.
    - OCR-only prompt with zero editorial or scene hallucination.
    - Explicit disabled path when vision is disabled or unavailable.
    - Sends clean cropped PNG/JPEG bytes via gateway.
    """

    OCR_PROMPT = (
        "Read and transcribe the exact text appearing in this cropped subtitle image verbatim. "
        "Return only the transcribed text with no markdown formatting, commentary, or translation."
    )

    def __init__(
        self,
        gateway: Any | None = None,
        model: str = "gpt-4o-mini",
        enabled: bool = False,
        custom_caller: Callable[..., str | None] | None = None,
    ) -> None:
        self.gateway = gateway
        self.model = model
        self.enabled = enabled
        self.custom_caller = custom_caller

    def is_available(self) -> bool:
        """Check whether Vision OCR fallback is actively enabled and supported."""
        if not self.enabled:
            return False
        if self.custom_caller is not None:
            return True
        return self.gateway is not None

    def ocr_cropped_image(
        self,
        image: Image.Image,
        cancellation_token: CancellationToken | None = None,
    ) -> str | None:
        """Submit cropped subtitle image to Vision OCR.
        
        Returns transcribed text or None if disabled/failed.
        """
        if not self.is_available():
            return None

        if cancellation_token:
            cancellation_token.check_cancelled()

        # Enforce cropped image boundary
        validate_cropped_image(image)

        # Injected deterministic test caller
        if self.custom_caller is not None:
            return self.custom_caller(image, cancellation_token)

        # Re-encode cropped PIL image to clean PNG bytes
        bio = io.BytesIO()
        image.save(bio, format="PNG")
        png_bytes = bio.getvalue()

        try:
            result = self.gateway.submit_image_chat(
                prompt=self.OCR_PROMPT,
                images=[png_bytes],
                model=self.model,
                system_prompt="You are a high-precision OCR engine for subtitle bitmap crops.",
                stream=False,
                expect_json=False,
                cancellation_token=cancellation_token,
                phase="phase3_ocr",
            )
            if result and getattr(result, "raw_response", None):
                return str(result.raw_response).strip()
        except CancelledError:
            raise
        except Exception:
            return None

        return None


class OcrAdapter:
    """RapidOCR ONNX adapter with lazy engine loading, quality gating, and optional Vision fallback."""

    def __init__(
        self,
        model_manager: OcrModelManager | None = None,
        custom_engine: Any = None,
        vision_adapter: VisionOcrAdapter | None = None,
        min_confidence: float = 0.50,
    ) -> None:
        self.model_manager = model_manager or OcrModelManager()
        self._engine = custom_engine
        self._initialized = custom_engine is not None
        self.vision_adapter = vision_adapter or VisionOcrAdapter(enabled=False)
        self.min_confidence = min_confidence

    @classmethod
    def is_package_installed(cls) -> bool:
        """Check whether rapidocr or rapidocr_onnxruntime is available."""
        try:
            import rapidocr_onnxruntime  # noqa: F401
            return True
        except ImportError:
            try:
                import rapidocr  # noqa: F401
                return True
            except ImportError:
                return False

    @staticmethod
    def bundled_models_ready() -> bool:
        """The portable RapidOCR package includes its own offline ONNX models."""
        try:
            import rapidocr_onnxruntime
        except ImportError:
            return False
        model_dir = Path(rapidocr_onnxruntime.__file__).resolve().parent / "models"
        return all(
            (model_dir / name).is_file() and (model_dir / name).stat().st_size > 0
            for name in (
                "ch_PP-OCRv4_det_infer.onnx",
                "ch_PP-OCRv4_rec_infer.onnx",
                "ch_ppocr_mobile_v2.0_cls_infer.onnx",
            )
        )

    def is_engine_ready(self) -> bool:
        """Check whether local engine or custom engine is ready."""
        if self._initialized:
            return True
        if not self.is_package_installed():
            return False
        return self.model_manager.is_ready() or self.bundled_models_ready()

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                raise ToolRecapError(
                    "RapidOCR package not installed; local bitmap OCR runtime unavailable"
                ) from exc

        if self.model_manager.is_ready():
            paths = self.model_manager.get_model_paths()
            self._engine = RapidOCR(
                det_model_path=str(paths["det"]),
                rec_model_path=str(paths["rec"]),
                cls_model_path=str(paths["cls"]),
            )
        elif self.bundled_models_ready():
            self._engine = RapidOCR()
        else:
            raise ToolRecapError("OCR models are missing from both portable runtime and verified local cache")
        self._initialized = True
        return self._engine

    def ocr_image(
        self,
        image: Image.Image,
        cancellation_token: CancellationToken | None = None,
    ) -> OcrResult:
        """Perform OCR on a cropped subtitle bounding box image."""
        if cancellation_token:
            cancellation_token.check_cancelled()

        # Enforce bounding box guard
        validate_cropped_image(image)

        raw_text = ""
        avg_score = 0.0
        ocr_failed = False
        failure_reason = ""

        # 1. Attempt local RapidOCR
        try:
            engine = self._get_engine()
            import numpy as np

            img_arr = np.array(image.convert("RGB"))
            raw_output = engine(img_arr)

            texts: list[str] = []
            scores: list[float] = []

            if hasattr(raw_output, "txts") and hasattr(raw_output, "scores"):
                if raw_output.txts and raw_output.scores:
                    texts = [str(t) for t in raw_output.txts]
                    scores = [float(s) for s in raw_output.scores]
            else:
                items = raw_output[0] if isinstance(raw_output, tuple) else raw_output
                if items and isinstance(items, (list, tuple)):
                    for item in items:
                        if isinstance(item, (list, tuple)) and len(item) >= 3:
                            texts.append(str(item[1]))
                            scores.append(float(item[2]))

            if texts:
                raw_text = " ".join(texts)
                avg_score = sum(scores) / len(scores) if scores else 0.0
        except Exception as exc:
            ocr_failed = True
            failure_reason = str(exc)

        is_valid, reason = quality_gate(raw_text, avg_score, min_confidence=self.min_confidence)
        if is_valid and not ocr_failed:
            return OcrResult(
                text=strip_formatting_tags(raw_text),
                confidence=avg_score,
                is_valid=True,
                reason="Local RapidOCR passed",
                source="rapidocr",
                raw_text=raw_text,
            )

        # 2. Local OCR failed or rejected by quality gate: Optional Vision OCR fallback
        if cancellation_token:
            cancellation_token.check_cancelled()

        if self.vision_adapter.is_available():
            try:
                vision_text = self.vision_adapter.ocr_cropped_image(
                    image, cancellation_token=cancellation_token
                )
                if vision_text:
                    v_clean = strip_formatting_tags(vision_text).strip()
                    v_valid, v_reason = quality_gate(v_clean, 0.85, min_confidence=self.min_confidence)
                    if v_valid:
                        return OcrResult(
                            text=v_clean,
                            confidence=0.85,
                            is_valid=True,
                            reason="Vision OCR fallback passed",
                            source="vision_ai",
                            raw_text=vision_text,
                        )
            except CancelledError:
                raise
            except Exception:
                pass

        # 3. Explicit disabled/rejected path
        vision_state = "Vision OCR fallback disabled" if not self.vision_adapter.is_available() else "Vision OCR yielded no valid text"
        rejection_reason = f"Local OCR rejected ({reason or failure_reason}); {vision_state}"

        return OcrResult(
            text="",
            confidence=0.0,
            is_valid=False,
            reason=rejection_reason,
            source="rejected",
            raw_text=raw_text,
        )

    def ocr_batch(
        self,
        images: Sequence[Image.Image],
        cancellation_token: CancellationToken | None = None,
    ) -> list[OcrResult]:
        """Perform OCR sequentially over a bounded list of cropped images with cooperative cancellation."""
        results: list[OcrResult] = []
        for img in images:
            if cancellation_token:
                cancellation_token.check_cancelled()
            results.append(self.ocr_image(img, cancellation_token=cancellation_token))
        return results
