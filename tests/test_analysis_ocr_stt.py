"""Focused deterministic unit tests for ToolRecap V4 OCR and STT modules."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import struct
from unittest.mock import MagicMock
import wave

from PIL import Image
import pytest

from toolrecap_v4.analysis.dependencies import (
    compute_ocr_dependency_signature,
    compute_stt_dependency_signature,
    is_signature_valid,
)
from toolrecap_v4.analysis.models import AudioSelection
from toolrecap_v4.analysis.source_prep.audio import AudioStreamInfo
from toolrecap_v4.analysis.source_prep.stt import (
    DEFAULT_STT_MANIFEST,
    SttModelManifest,
    SttModelManager,
    SttResult,
    SttStatus,
    deduplicate_overlap_cues,
    extract_bounded_audio_window,
    plan_audio_windows,
    resolve_device_policy,
    transcribe_episode_stt,
)
from toolrecap_v4.analysis.source_prep.subtitles.models import SubtitleCue
from toolrecap_v4.analysis.source_prep.subtitles.ocr import (
    DEFAULT_OCR_MANIFEST,
    DEFAULT_OCR_MODELS,
    OcrAdapter,
    OcrModelInfo,
    OcrModelManifest,
    OcrModelManager,
    OcrResult,
    VisionOcrAdapter,
    quality_gate,
    validate_cropped_image,
)
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError, ToolRecapError
from toolrecap_v4.media import CommandResult


def _generate_synthetic_wav(path: Path, duration_sec: float = 1.0, amplitude: int = 5000) -> Path:
    """Generate 16-bit mono 16kHz PCM WAV for deterministic acoustic testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        if amplitude > 0:
            samples = [int(amplitude * (1 if (i // 16) % 2 == 0 else -1)) for i in range(num_samples)]
        else:
            samples = [0] * num_samples
        raw = struct.pack(f"<{len(samples)}h", *samples)
        wf.writeframes(raw)
    return path


# ============================================================================
# 1. OCR Quality Gate Tests
# ============================================================================


def test_ocr_quality_gate_valid_text():
    valid, reason = quality_gate("Hello world", confidence=0.92)
    assert valid is True
    assert reason == "Valid"


def test_ocr_quality_gate_strips_tags():
    valid, reason = quality_gate("<i><b>Episode 1</b></i>", confidence=0.88)
    assert valid is True
    assert reason == "Valid"


def test_ocr_quality_gate_rejects_empty():
    valid, reason = quality_gate("   ", confidence=0.95)
    assert valid is False
    assert "Empty" in reason


def test_ocr_quality_gate_rejects_low_confidence():
    valid, reason = quality_gate("Some subtitle text", confidence=0.35, min_confidence=0.50)
    assert valid is False
    assert "Low confidence" in reason


def test_ocr_quality_gate_rejects_non_alphanumeric():
    valid, reason = quality_gate("... -- ??? !!!", confidence=0.90)
    assert valid is False
    assert "alphanumeric" in reason.lower()


def test_ocr_quality_gate_rejects_repetitive_garbage():
    # Repetitive patterns like |||||| or -------
    valid, reason = quality_gate("||||||||", confidence=0.95)
    assert valid is False
    assert "garbage" in reason.lower()

    valid2, _ = quality_gate("- - - - - -", confidence=0.90)
    assert valid2 is False


def test_ocr_quality_gate_supports_multilingual_unicode():
    # Vietnamese and Latin accents
    valid, _ = quality_gate("Tập 1: Xin chào quý vị", confidence=0.85)
    assert valid is True

    valid_fr, _ = quality_gate("Café et résumé 100%", confidence=0.85)
    assert valid_fr is True


# ============================================================================
# 2. OCR Bounding Box / Cropped Image Guard Tests
# ============================================================================


def test_ocr_validate_cropped_image_accepts_crops():
    # Normal subtitle crops are small
    img = Image.new("RGB", (450, 80), color="white")
    validate_cropped_image(img)  # Should not raise


def test_ocr_validate_cropped_image_rejects_full_video_frames():
    # 1080p full frame must be rejected
    full_frame = Image.new("RGB", (1920, 1080), color="black")
    with pytest.raises(ValueError, match="Bounding box violation"):
        validate_cropped_image(full_frame)

    # 4K frame
    frame_4k = Image.new("RGB", (3840, 2160), color="black")
    with pytest.raises(ValueError, match="Bounding box violation"):
        validate_cropped_image(frame_4k)


def test_ocr_adapter_rejects_full_frame_before_engine():
    adapter = OcrAdapter()
    full_frame = Image.new("RGB", (1920, 1080), color="black")
    with pytest.raises(ValueError, match="Bounding box violation"):
        adapter.ocr_image(full_frame)


# ============================================================================
# 3. OCR Model Manager: Atomic Staging, Cancellation & Hash Verification
# ============================================================================


def test_ocr_manifest_hash_deterministic():
    manifest1 = OcrModelManifest(
        runtime="rapidocr",
        version="v3.9.2",
        models=dict(DEFAULT_OCR_MODELS),
    )
    manifest2 = OcrModelManifest(
        runtime="rapidocr",
        version="v3.9.2",
        models=dict(DEFAULT_OCR_MODELS),
    )
    assert manifest1.compute_manifest_hash() == manifest2.compute_manifest_hash()


def test_ocr_model_manager_missing_models(tmp_path):
    mgr = OcrModelManager(model_dir=tmp_path / "models" / "ocr")
    assert mgr.are_models_available() is False
    assert mgr.verify_hashes() is False
    assert mgr.is_ready() is False


def test_portable_ocr_uses_bundled_models_without_local_cache(tmp_path):
    adapter = OcrAdapter(model_manager=OcrModelManager(model_dir=tmp_path / "empty-cache"))
    assert adapter.model_manager.is_ready() is False
    assert adapter.bundled_models_ready() is True
    assert adapter.is_engine_ready() is True
    assert type(adapter._get_engine()).__name__ == "RapidOCR"
    assert not (tmp_path / "empty-cache").exists()


def test_ocr_model_manager_corrupted_hash(tmp_path):
    model_dir = tmp_path / "models" / "ocr"
    model_dir.mkdir(parents=True)

    # Create dummy files with bad content (wrong hash)
    for info in DEFAULT_OCR_MODELS.values():
        (model_dir / info.filename).write_bytes(b"bad model content")

    mgr = OcrModelManager(model_dir=model_dir)
    assert mgr.are_models_available() is True
    assert mgr.verify_hashes() is False
    assert mgr.is_ready() is False


def test_ocr_model_manager_injected_download_and_atomic_promotion(tmp_path):
    model_dir = tmp_path / "models" / "ocr"

    dummy_content = b"valid onnx weights for test"
    dummy_sha256 = hashlib.sha256(dummy_content).hexdigest()

    test_models = {
        "det": OcrModelInfo("det", "det.onnx", "http://test/det.onnx", dummy_sha256),
        "rec": OcrModelInfo("rec", "rec.onnx", "http://test/rec.onnx", dummy_sha256),
    }
    manifest = OcrModelManifest(runtime="rapidocr", version="v1.0", models=test_models)

    def injected_downloader(url, dest_path, expected_sha256, cancellation_token, progress_callback):
        dest_path.write_bytes(dummy_content)

    mgr = OcrModelManager(model_dir=model_dir, manifest=manifest, downloader=injected_downloader)
    assert mgr.is_ready() is False

    success = mgr.download_models()
    assert success is True
    assert mgr.is_ready() is True

    # Ensure no lingering .partial files
    assert not list(model_dir.glob("*.partial"))
    assert (model_dir / "det.onnx").is_file()
    assert (model_dir / "rec.onnx").is_file()
    assert (model_dir / "manifest.json").is_file()


def test_ocr_model_manager_checksum_mismatch_cleans_partial(tmp_path):
    model_dir = tmp_path / "models" / "ocr"
    test_models = {
        "det": OcrModelInfo("det", "det.onnx", "http://test/det.onnx", "0000000000000000000000000000000000000000000000000000000000000000"),
    }
    manifest = OcrModelManifest(runtime="rapidocr", version="v1.0", models=test_models)

    def bad_downloader(url, dest_path, expected_sha256, cancellation_token, progress_callback):
        dest_path.write_bytes(b"corrupted bytes")

    mgr = OcrModelManager(model_dir=model_dir, manifest=manifest, downloader=bad_downloader)
    with pytest.raises(ToolRecapError, match="Checksum mismatch"):
        mgr.download_models()

    assert mgr.is_ready() is False
    assert not (model_dir / "det.onnx").exists()
    assert not (model_dir / "det.onnx.partial").exists()


def test_ocr_model_manager_cancellation_cleans_partial(tmp_path):
    model_dir = tmp_path / "models" / "ocr"
    test_models = {
        "det": OcrModelInfo("det", "det.onnx", "http://test/det.onnx", "dummyhash"),
    }
    manifest = OcrModelManifest(runtime="rapidocr", version="v1.0", models=test_models)
    token = CancellationToken()

    def cancelling_downloader(url, dest_path, expected_sha256, cancellation_token, progress_callback):
        dest_path.write_bytes(b"partial bytes")
        token.cancel()

    mgr = OcrModelManager(model_dir=model_dir, manifest=manifest, downloader=cancelling_downloader)
    with pytest.raises(CancelledError):
        mgr.download_models(cancellation_token=token)

    assert mgr.is_ready() is False
    assert not (model_dir / "det.onnx").exists()
    assert not (model_dir / "det.onnx.partial").exists()


def test_ocr_dependency_signature_and_validation():
    manifest = DEFAULT_OCR_MANIFEST
    sig1 = compute_ocr_dependency_signature(manifest, {"engine": "onnx"})
    sig2 = compute_ocr_dependency_signature(manifest, {"engine": "onnx"})
    sig3 = compute_ocr_dependency_signature(manifest, {"engine": "openvino"})

    assert sig1["type"] == "ocr"
    assert sig1["ocr_hash"] == sig2["ocr_hash"]
    assert sig1["ocr_hash"] != sig3["ocr_hash"]
    assert is_signature_valid(sig1, sig2) is True
    assert is_signature_valid(sig1, sig3) is False


# ============================================================================
# 4. Vision OCR Fallback: Disabled Path vs Injected Image-Only Flow
# ============================================================================


def test_vision_ocr_explicit_disabled_path():
    # Adapter disabled by default
    adapter = VisionOcrAdapter(enabled=False)
    assert adapter.is_available() is False

    crop = Image.new("RGB", (300, 50), color="white")
    assert adapter.ocr_cropped_image(crop) is None

    # OcrAdapter with disabled vision fallback records explicit disabled reason
    custom_engine = MagicMock(return_value=[])  # Empty local result
    ocr_adapter = OcrAdapter(custom_engine=custom_engine, vision_adapter=adapter)
    result = ocr_adapter.ocr_image(crop)

    assert result.is_valid is False
    assert result.source == "rejected"
    assert "Vision OCR fallback disabled" in result.reason


def test_vision_ocr_image_only_fallback_success():
    crop = Image.new("RGB", (320, 60), color="white")

    # Injected vision caller simulating successful vision OCR
    def mock_vision_caller(image, cancellation_token):
        return "Transcribed by Vision AI"

    vision_adapter = VisionOcrAdapter(enabled=True, custom_caller=mock_vision_caller)
    assert vision_adapter.is_available() is True

    # Local engine returns low confidence to trigger fallback
    custom_engine = MagicMock(return_value=([None, "garbled", 0.20],))
    ocr_adapter = OcrAdapter(custom_engine=custom_engine, vision_adapter=vision_adapter)

    res = ocr_adapter.ocr_image(crop)
    assert res.is_valid is True
    assert res.source == "vision_ai"
    assert res.text == "Transcribed by Vision AI"
    assert res.confidence == 0.85


def test_vision_ocr_gateway_media_contract():
    crop = Image.new("RGB", (200, 40), color="white")

    mock_gateway = MagicMock()
    mock_res = MagicMock()
    mock_res.raw_response = "Exact Text From Crop"
    mock_gateway.submit_image_chat.return_value = mock_res

    vision_adapter = VisionOcrAdapter(gateway=mock_gateway, enabled=True, model="gpt-4o-mini")
    text = vision_adapter.ocr_cropped_image(crop)

    assert text == "Exact Text From Crop"
    assert mock_gateway.submit_image_chat.called
    kwargs = mock_gateway.submit_image_chat.call_args[1]

    # Verify OCR-only prompt, not scene description
    assert "Read and transcribe the exact text" in kwargs["prompt"]
    assert kwargs["phase"] == "phase3_ocr"
    assert kwargs["expect_json"] is False
    assert len(kwargs["images"]) == 1
    # Verify image is clean PNG bytes
    assert kwargs["images"][0].startswith(b"\x89PNG\r\n\x1a\n")


def test_vision_ocr_cancellation_propagates():
    token = CancellationToken()
    token.cancel()

    def mock_vision_caller(image, cancellation_token):
        if cancellation_token:
            cancellation_token.check_cancelled()
        return "text"

    vision_adapter = VisionOcrAdapter(enabled=True, custom_caller=mock_vision_caller)
    crop = Image.new("RGB", (200, 40), color="white")

    with pytest.raises(CancelledError):
        vision_adapter.ocr_cropped_image(crop, cancellation_token=token)


def test_vision_ocr_safety_disabled_zero_gateway():
    crop = Image.new("RGB", (300, 50), color="white")
    fake_gateway = MagicMock()
    vision_adapter = VisionOcrAdapter(gateway=fake_gateway, enabled=False)
    assert vision_adapter.is_available() is False
    assert vision_adapter.ocr_cropped_image(crop) is None
    fake_gateway.submit_image_chat.assert_not_called()

    # Local OCR fails (empty output), vision adapter disabled -> zero gateway calls
    local_engine = MagicMock(return_value=[])
    ocr_adapter = OcrAdapter(custom_engine=local_engine, vision_adapter=vision_adapter)
    res = ocr_adapter.ocr_image(crop)

    assert res.is_valid is False
    assert res.source == "rejected"
    assert "Vision OCR fallback disabled" in res.reason
    fake_gateway.submit_image_chat.assert_not_called()


def test_vision_ocr_safety_local_valid_zero_gateway():
    crop = Image.new("RGB", (300, 50), color="white")
    fake_gateway = MagicMock()
    # Vision adapter is ENABLED with fake gateway
    vision_adapter = VisionOcrAdapter(gateway=fake_gateway, enabled=True, model="gpt-4o-mini")
    assert vision_adapter.is_available() is True

    # Local OCR succeeds with high confidence
    local_engine = MagicMock(return_value=([[None, "Valid Subtitle Text", 0.95]], None))
    ocr_adapter = OcrAdapter(custom_engine=local_engine, vision_adapter=vision_adapter)
    res = ocr_adapter.ocr_image(crop)

    assert res.is_valid is True
    assert res.source == "rapidocr"
    assert res.text == "Valid Subtitle Text"
    assert res.confidence == 0.95
    # Strict safety invariant: zero Gateway calls when local OCR succeeds
    fake_gateway.submit_image_chat.assert_not_called()


def test_vision_ocr_safety_rejected_enabled_crop_only():
    crop = Image.new("RGB", (250, 45), color="white")
    fake_gateway = MagicMock()
    fake_res = MagicMock()
    fake_res.raw_response = "Vision Subtitle Fallback Text"
    fake_gateway.submit_image_chat.return_value = fake_res

    vision_adapter = VisionOcrAdapter(gateway=fake_gateway, enabled=True, model="gpt-4o-mini")

    # Local OCR returns low confidence (0.25) -> rejected by quality gate
    local_engine = MagicMock(return_value=([[None, "unreadable garbage", 0.25]], None))
    ocr_adapter = OcrAdapter(custom_engine=local_engine, vision_adapter=vision_adapter)
    res = ocr_adapter.ocr_image(crop)

    assert res.is_valid is True
    assert res.source == "vision_ai"
    assert res.text == "Vision Subtitle Fallback Text"
    assert res.confidence == 0.85

    # Verify Gateway boundary contract:
    fake_gateway.submit_image_chat.assert_called_once()
    call_kwargs = fake_gateway.submit_image_chat.call_args[1]
    assert call_kwargs["prompt"] == VisionOcrAdapter.OCR_PROMPT
    assert call_kwargs["phase"] == "phase3_ocr"
    assert call_kwargs["expect_json"] is False
    assert len(call_kwargs["images"]) == 1
    # Verify crop-only: clean PNG bytes
    assert call_kwargs["images"][0].startswith(b"\x89PNG\r\n\x1a\n")


def test_vision_ocr_safety_gateway_failure_rejection_path():
    crop = Image.new("RGB", (280, 50), color="white")
    fake_gateway = MagicMock()
    # Simulate gateway failure (e.g. 500 error / network failure)
    fake_gateway.submit_image_chat.side_effect = ToolRecapError("Gateway service 503 unavailable")

    vision_adapter = VisionOcrAdapter(gateway=fake_gateway, enabled=True)

    # Direct Vision adapter handles failure safely (returns None)
    assert vision_adapter.ocr_cropped_image(crop) is None

    # Local OCR fails and Vision gateway fails -> must NOT crash, returns rejected OcrResult
    local_engine = MagicMock(return_value=[])
    ocr_adapter = OcrAdapter(custom_engine=local_engine, vision_adapter=vision_adapter)
    res = ocr_adapter.ocr_image(crop)

    assert res.is_valid is False
    assert res.source == "rejected"
    assert "Vision OCR yielded no valid text" in res.reason

    # Also test gateway returning empty response (raw_response = "")
    fake_gateway_empty = MagicMock()
    fake_res = MagicMock()
    fake_res.raw_response = ""
    fake_gateway_empty.submit_image_chat.return_value = fake_res
    vision_adapter_empty = VisionOcrAdapter(gateway=fake_gateway_empty, enabled=True)
    ocr_adapter_empty = OcrAdapter(custom_engine=local_engine, vision_adapter=vision_adapter_empty)
    res_empty = ocr_adapter_empty.ocr_image(crop)
    assert res_empty.is_valid is False
    assert res_empty.source == "rejected"


def test_vision_ocr_safety_oversized_full_frame_rejected():
    full_frame = Image.new("RGB", (1920, 1080), color="black")
    fake_gateway = MagicMock()
    vision_adapter = VisionOcrAdapter(gateway=fake_gateway, enabled=True)

    # 1. Direct call on VisionOcrAdapter with full frame raises ValueError and makes ZERO gateway calls
    with pytest.raises(ValueError, match="Bounding box violation: image dimensions.*resemble full video frame"):
        vision_adapter.ocr_cropped_image(full_frame)
    fake_gateway.submit_image_chat.assert_not_called()

    # 2. OcrAdapter.ocr_image raises ValueError before engine or gateway is touched
    local_engine = MagicMock()
    ocr_adapter = OcrAdapter(custom_engine=local_engine, vision_adapter=vision_adapter)
    with pytest.raises(ValueError, match="Bounding box violation"):
        ocr_adapter.ocr_image(full_frame)
    local_engine.assert_not_called()
    fake_gateway.submit_image_chat.assert_not_called()


# ============================================================================
# 5. STT Device Policy & Safe CPU Fallback Tests
# ============================================================================


def test_stt_device_policy_cpu():
    dev, compute, diags = resolve_device_policy(policy="cpu")
    assert dev == "cpu"
    assert compute == "int8"
    assert any("CPU" in d for d in diags)


def test_stt_device_policy_cuda_available():
    dev, compute, diags = resolve_device_policy(policy="cuda", check_cuda_fn=lambda: True)
    assert dev == "cuda"
    assert compute == "float16"
    assert any("CUDA" in d for d in diags)


def test_stt_device_policy_cuda_unavailable_is_explicit_error():
    # Explicit CUDA request must fail clearly; only auto policy may choose CPU fallback.
    with pytest.raises(ToolRecapError, match="Requested STT device 'cuda' is unavailable"):
        resolve_device_policy(policy="cuda", check_cuda_fn=lambda: False)


def test_stt_device_policy_auto():
    # Auto with CUDA available
    dev1, comp1, _ = resolve_device_policy(policy="auto", check_cuda_fn=lambda: True)
    assert dev1 == "cuda"
    assert comp1 == "float16"

    # Auto without CUDA
    dev2, comp2, _ = resolve_device_policy(policy="auto", check_cuda_fn=lambda: False)
    assert dev2 == "cpu"
    assert comp2 == "int8"


# ============================================================================
# 6. STT Model Manager: Manifest, Staging, & Cancellation Tests
# ============================================================================


def test_stt_manifest_hash_deterministic():
    m1 = SttModelManifest(model_name="tiny", repo_id="Systran/faster-whisper-tiny")
    m2 = SttModelManifest(model_name="tiny", repo_id="Systran/faster-whisper-tiny")
    assert m1.compute_manifest_hash() == m2.compute_manifest_hash()


def test_stt_model_manager_missing_files(tmp_path):
    mgr = SttModelManager(model_dir=tmp_path / "whisper_tiny")
    assert mgr.is_ready() is False


def test_stt_model_manager_injected_download_success(tmp_path):
    model_dir = tmp_path / "whisper_tiny"

    def mock_downloader(repo_id, revision, target_dir, cancellation_token, progress_callback):
        # Create all expected files
        for fname in DEFAULT_STT_MANIFEST.expected_files:
            (target_dir / fname).write_text("model weights or config", encoding="utf-8")

    mgr = SttModelManager(model_dir=model_dir, downloader=mock_downloader)
    assert mgr.is_ready() is False

    success = mgr.download_models()
    assert success is True
    assert mgr.is_ready() is True
    assert (model_dir / "config.json").is_file()
    assert (model_dir / "model.bin").is_file()
    assert (model_dir / "manifest.json").is_file()
    # Ensure staging dir cleaned
    assert not (tmp_path / "whisper_tiny.partial").exists()


def test_stt_model_manager_cancellation_never_ready(tmp_path):
    model_dir = tmp_path / "whisper_tiny"
    token = CancellationToken()

    def cancelling_downloader(repo_id, revision, target_dir, cancellation_token, progress_callback):
        (target_dir / "config.json").write_text("partial", encoding="utf-8")
        token.cancel()

    mgr = SttModelManager(model_dir=model_dir, downloader=cancelling_downloader)
    with pytest.raises(CancelledError):
        mgr.download_models(cancellation_token=token)

    assert mgr.is_ready() is False
    assert not model_dir.exists()
    assert not (tmp_path / "whisper_tiny.partial").exists()


def test_stt_dependency_signature_and_validation():
    stream = AudioStreamInfo(index=2, codec="aac", channels=2, sample_rate=48000, language="eng")
    sel = AudioSelection(selected_stream=stream, global_index=2, audio_ordinal=0)
    manifest = DEFAULT_STT_MANIFEST

    sig1 = compute_stt_dependency_signature(sel, manifest, device_policy="cpu")
    sig2 = compute_stt_dependency_signature(sel, manifest, device_policy="cpu")
    sig3 = compute_stt_dependency_signature(sel, manifest, device_policy="cuda")

    assert sig1["type"] == "stt"
    assert sig1["audio_global_index"] == 2
    assert sig1["stt_hash"] == sig2["stt_hash"]
    assert sig1["stt_hash"] != sig3["stt_hash"]
    assert is_signature_valid(sig1, sig2) is True
    assert is_signature_valid(sig1, sig3) is False


# ============================================================================
# 7. STT Audio Planning & Bounded Window Extraction Tests
# ============================================================================


def test_stt_plan_audio_windows():
    # 125 seconds total, 60s windows, 2s overlap
    # step = 58s
    # w0: [0, 60]
    # w1: [58, 118]
    # w2: [116, 125] -> duration = 9s
    windows = plan_audio_windows(125.0, window_duration_sec=60.0, overlap_sec=2.0)
    assert len(windows) == 3
    assert windows[0] == (0.0, 60.0)
    assert windows[1] == (58.0, 60.0)
    assert windows[2] == (116.0, 9.0)

    # Short video (30s) fits into single window
    short_windows = plan_audio_windows(30.0, window_duration_sec=60.0, overlap_sec=2.0)
    assert len(short_windows) == 1
    assert short_windows[0] == (0.0, 30.0)

    # Zero duration
    assert plan_audio_windows(0.0) == []


def test_stt_extract_bounded_window_ffmpeg_command(tmp_path):
    stream = AudioStreamInfo(index=3, codec="aac", channels=2, sample_rate=48000)
    selection = AudioSelection(selected_stream=stream, global_index=3, audio_ordinal=1)

    captured_cmds = []

    def mock_runner(cmd, timeout, cancellation_token):
        captured_cmds.append(cmd)
        # Create output file
        out_file = Path(cmd[-1])
        out_file.write_bytes(b"RIFF" + b"\x00" * 40)
        return CommandResult(exit_code=0, stdout="", stderr="")

    out_wav = tmp_path / "win_0000.wav"
    extract_bounded_audio_window(
        video_path="video.mkv",
        selection=selection,
        start_sec=10.5,
        duration_sec=60.0,
        output_wav=out_wav,
        command_runner=mock_runner,
    )

    assert len(captured_cmds) == 1
    cmd = captured_cmds[0]

    # Invariants:
    # 1. -ss and -t specify bounded slice
    assert "-ss" in cmd and "10.500" in cmd
    assert "-t" in cmd and "60.000" in cmd
    # 2. Uses explicit container global stream index -map 0:3, NEVER 0:a:1
    assert "-map" in cmd
    map_idx = cmd.index("-map")
    assert cmd[map_idx + 1] == "0:3"
    # 3. Audio format 16kHz mono PCM
    assert cmd[cmd.index("-acodec") + 1] == "pcm_s16le"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-ar") + 1] == "16000"


# ============================================================================
# 8. STT Overlap Deduplication & Boundary Speech Tests
# ============================================================================


def test_stt_deduplicate_overlap_cues():
    existing = [
        SubtitleCue(start_ms=1000, end_ms=3000, text="Hello and welcome", source_type="stt", source_format="whisper"),
        SubtitleCue(start_ms=4000, end_ms=7000, text="Today we discuss recap", source_type="stt", source_format="whisper"),
    ]

    candidates = [
        # Exact duplicate in overlap window (starts at 4050ms, same text): must be deduplicated
        SubtitleCue(start_ms=4050, end_ms=7020, text="Today we discuss recap", source_type="stt", source_format="whisper"),
        # New boundary speech in overlap window: must be preserved!
        SubtitleCue(start_ms=7200, end_ms=9000, text="Let's begin part one", source_type="stt", source_format="whisper"),
    ]

    accepted = deduplicate_overlap_cues(existing, candidates, max_jitter_ms=1500)
    assert len(accepted) == 1
    assert accepted[0].text == "Let's begin part one"
    assert accepted[0].start_ms == 7200


# ============================================================================
# 9. STT End-to-End Orchestration: Status Distinctions, Cleanup & Cancel
# ============================================================================


@dataclass
class _MockWhisperSegment:
    start: float
    end: float
    text: str


class _MockWhisperModel:
    def __init__(self, segments: list[_MockWhisperSegment] | None = None) -> None:
        self.segments = segments or []

    def transcribe(self, audio_path: str, beam_size: int = 1, language: str = "en"):
        return iter(self.segments), MagicMock()


def test_stt_no_audio_status():
    no_audio_sel = AudioSelection(selected_stream=None, global_index=-1, audio_ordinal=-1)
    res = transcribe_episode_stt(
        video_path="dummy.mkv",
        selection=no_audio_sel,
        duration_ms=60000,
    )
    assert res.status == SttStatus.NO_AUDIO
    assert res.cue_count == 0


def test_stt_runtime_unavailable_status(tmp_path):
    stream = AudioStreamInfo(index=1, codec="aac", channels=2, sample_rate=48000)
    sel = AudioSelection(selected_stream=stream, global_index=1, audio_ordinal=0)

    # Manager pointing to empty dir (not ready)
    unready_mgr = SttModelManager(model_dir=tmp_path / "not_ready")

    res = transcribe_episode_stt(
        video_path="dummy.mkv",
        selection=sel,
        duration_ms=60000,
        model_manager=unready_mgr,
    )
    assert res.status == SttStatus.RUNTIME_UNAVAILABLE
    assert res.cue_count == 0


def test_stt_silent_status_and_window_cleanup(tmp_path):
    stream = AudioStreamInfo(index=1, codec="aac", channels=2, sample_rate=48000)
    sel = AudioSelection(selected_stream=stream, global_index=1, audio_ordinal=0)

    # Ready model manager
    model_dir = tmp_path / "ready_model"
    model_dir.mkdir(parents=True)
    for f in DEFAULT_STT_MANIFEST.expected_files:
        (model_dir / f).write_text("weights", encoding="utf-8")
    mgr = SttModelManager(model_dir=model_dir)

    temp_stt_dir = tmp_path / "temp_stt"

    # Command runner generating silent WAV (amplitude 0)
    def silent_runner(cmd, timeout, cancellation_token):
        out_file = Path(cmd[-1])
        _generate_synthetic_wav(out_file, duration_sec=1.0, amplitude=0)
        return CommandResult(exit_code=0, stdout="", stderr="")

    mock_model = _MockWhisperModel([])

    res = transcribe_episode_stt(
        video_path="dummy.mkv",
        selection=sel,
        duration_ms=60000,
        model_manager=mgr,
        command_runner=silent_runner,
        model_factory=lambda *a, **kw: mock_model,
        temp_dir=temp_stt_dir,
    )

    assert res.status == SttStatus.SILENT
    assert res.cue_count == 0
    # Invariant: temporary files and directory are cleaned up
    assert not temp_stt_dir.exists()


def test_stt_success_empty_status(tmp_path):
    stream = AudioStreamInfo(index=1, codec="aac", channels=2, sample_rate=48000)
    sel = AudioSelection(selected_stream=stream, global_index=1, audio_ordinal=0)

    model_dir = tmp_path / "ready_model"
    model_dir.mkdir(parents=True)
    for f in DEFAULT_STT_MANIFEST.expected_files:
        (model_dir / f).write_text("weights", encoding="utf-8")
    mgr = SttModelManager(model_dir=model_dir)

    # Signal present (amplitude 5000), but Whisper produces 0 cues
    def active_runner(cmd, timeout, cancellation_token):
        out_file = Path(cmd[-1])
        _generate_synthetic_wav(out_file, duration_sec=1.0, amplitude=5000)
        return CommandResult(exit_code=0, stdout="", stderr="")

    mock_model = _MockWhisperModel([])

    res = transcribe_episode_stt(
        video_path="dummy.mkv",
        selection=sel,
        duration_ms=60000,
        model_manager=mgr,
        command_runner=active_runner,
        model_factory=lambda *a, **kw: mock_model,
    )

    assert res.status == SttStatus.SUCCESS_EMPTY
    assert res.cue_count == 0


def test_stt_success_with_absolute_timestamp_offsets(tmp_path):
    stream = AudioStreamInfo(index=1, codec="aac", channels=2, sample_rate=48000)
    sel = AudioSelection(selected_stream=stream, global_index=1, audio_ordinal=0)

    model_dir = tmp_path / "ready_model"
    model_dir.mkdir(parents=True)
    for f in DEFAULT_STT_MANIFEST.expected_files:
        (model_dir / f).write_text("weights", encoding="utf-8")
    mgr = SttModelManager(model_dir=model_dir)

    def active_runner(cmd, timeout, cancellation_token):
        out_file = Path(cmd[-1])
        _generate_synthetic_wav(out_file, duration_sec=1.0, amplitude=5000)
        return CommandResult(exit_code=0, stdout="", stderr="")

    # 125s duration -> 3 windows (w0: 0-60s, w1: 58-118s, w2: 116-125s)
    # Mock model returns relative segments inside window
    segments = [
        _MockWhisperSegment(start=5.0, end=10.0, text="First dialogue cue"),
    ]
    mock_model = _MockWhisperModel(segments)

    res = transcribe_episode_stt(
        video_path="dummy.mkv",
        selection=sel,
        duration_ms=125000,
        model_manager=mgr,
        command_runner=active_runner,
        model_factory=lambda *a, **kw: mock_model,
        window_duration_sec=60.0,
        overlap_sec=2.0,
    )

    assert res.status == SttStatus.SUCCESS
    assert res.is_success is True
    assert res.cue_count > 0

    first_cue = res.cues[0]
    # In window 0 (start 0s): 5s -> 5000ms, 10s -> 10000ms
    assert first_cue.start_ms == 5000
    assert first_cue.end_ms == 10000
    assert first_cue.text == "First dialogue cue"
    assert first_cue.source_type == "stt"
    assert first_cue.source_format == "whisper"
    assert first_cue.stream_index == 1


def test_stt_cancellation_propagates_and_cleans_temp(tmp_path):
    stream = AudioStreamInfo(index=1, codec="aac", channels=2, sample_rate=48000)
    sel = AudioSelection(selected_stream=stream, global_index=1, audio_ordinal=0)

    model_dir = tmp_path / "ready_model"
    model_dir.mkdir(parents=True)
    for f in DEFAULT_STT_MANIFEST.expected_files:
        (model_dir / f).write_text("weights", encoding="utf-8")
    mgr = SttModelManager(model_dir=model_dir)

    temp_stt_dir = tmp_path / "stt_cancelling_temp"
    token = CancellationToken()

    def cancelling_runner(cmd, timeout, cancellation_token):
        out_file = Path(cmd[-1])
        _generate_synthetic_wav(out_file, duration_sec=1.0, amplitude=5000)
        # Cancel right after first window extraction
        token.cancel()
        return CommandResult(exit_code=0, stdout="", stderr="")

    mock_model = _MockWhisperModel([_MockWhisperSegment(start=1.0, end=2.0, text="Hi")])

    with pytest.raises(CancelledError):
        transcribe_episode_stt(
            video_path="dummy.mkv",
            selection=sel,
            duration_ms=120000,
            model_manager=mgr,
            command_runner=cancelling_runner,
            model_factory=lambda *a, **kw: mock_model,
            temp_dir=temp_stt_dir,
            cancellation_token=token,
        )

    # Bounded cleanup invariant: temp directory must be cleaned up on cancellation
    assert not temp_stt_dir.exists()
