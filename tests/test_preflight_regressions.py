"""Focused preflight regression tests:
- prompt EXACT whitespace saves twice
- thinking disabled in settings dialog
- preview input_text/bytes/style with mocked actual adapter (calls handler, not inspect string)
- inert dropdown widgets removed while preserving compatible variables and V2 sidebar layout
"""

import io
import json
import os
from pathlib import Path
import sys
import threading
from unittest.mock import MagicMock, patch
import wave

if sys.platform == "win32":
    tcl_path = os.path.join(sys.base_prefix, "tcl", "tcl8.6")
    tk_path = os.path.join(sys.base_prefix, "tcl", "tk8.6")
    if os.path.isdir(tcl_path) and "TCL_LIBRARY" not in os.environ:
        os.environ["TCL_LIBRARY"] = tcl_path
    if os.path.isdir(tk_path) and "TK_LIBRARY" not in os.environ:
        os.environ["TK_LIBRARY"] = tk_path

import httpx
import pytest

from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.settings import SettingsManager
from toolrecap_v4.ui.main_window import MainWindow
from toolrecap_v4.ui.settings_dialog import SettingsDialog
from toolrecap_v4.voice_studio import VoiceStudioAdapter


def make_dummy_wav(duration_s: float = 1.0, framerate: int = 24000, channels: int = 1) -> bytes:
    """Generate valid PCM WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(framerate)
        nframes = int(duration_s * framerate)
        w.writeframes(b"\x00\x00" * channels * nframes)
    return buf.getvalue()


def test_recap_inert_dropdowns_removed_and_variables_preserved(tmp_path: Path):
    """Verify _build_recap has no inert dropdown widgets (recap_mode_combo,
    content_type_combo, rights_combo, recap_lang_combo), while preserving
    compatible variables and V2 sidebar layout.
    """
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)
    try:
        app.update_idletasks()
        dialog = SettingsDialog(app, persistence=persistence)
        dialog.update_idletasks()

        # 1. Inert dropdown widgets removed from Recap pane
        assert not hasattr(dialog, "recap_mode_combo")
        assert not hasattr(dialog, "content_type_combo")
        assert not hasattr(dialog, "rights_combo")
        assert not hasattr(dialog, "recap_lang_combo")

        # 2. Compatible variables preserved
        assert hasattr(dialog, "recap_language_var")
        assert hasattr(dialog, "recap_mode_var")
        assert hasattr(dialog, "content_type_var")
        assert hasattr(dialog, "rights_var")

        # 3. Prompt text widget exists and is editable
        assert hasattr(dialog, "txt_prompt")
        assert dialog.txt_prompt is not None

        # 4. V2 sidebar layout preserved
        assert set(dialog._panes.keys()) == {"Recap", "Highlight", "AI Gateway", "Voice", "Render and Output"}
        assert dialog._current_pane == "Recap"

        dialog.destroy()
    finally:
        app.destroy()


def test_prompt_exact_whitespace_saves_twice(tmp_path: Path):
    """Verify prompt EXACT whitespace is preserved without stripping or alteration across two save cycles."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)
    try:
        app.update_idletasks()

        # Exact whitespace string with leading spaces, tabs, empty lines, and trailing spaces
        exact_prompt = "  Line 1 with leading spaces\n\n\tLine 2 tab indented\nLine 3 with trailing spaces   \n\n"

        # First session: open dialog, insert exact prompt, and save
        dialog1 = SettingsDialog(app, persistence=persistence)
        dialog1.update_idletasks()
        dialog1.txt_prompt.delete("1.0", "end")
        dialog1.txt_prompt.insert("1.0", exact_prompt)
        dialog1.var_gw_sub_model.set("scanner-route")
        dialog1.var_gw_prime_model.set("finalizer-route")

        # Save once
        dialog1._on_save()
        dialog1.destroy()

        # Verify first save preserved exact whitespace
        mgr = SettingsManager(persistence=persistence)
        loaded1 = mgr.load()
        assert loaded1.prompt == exact_prompt

        # Second session: open dialog again with saved settings, verify loaded text matches exactly
        dialog2 = SettingsDialog(app, persistence=persistence)
        dialog2.update_idletasks()
        retrieved_text = dialog2.txt_prompt.get("1.0", "end-1c")
        assert retrieved_text == exact_prompt
        dialog2.var_gw_sub_model.set("scanner-route")
        dialog2.var_gw_prime_model.set("finalizer-route")

        # Save twice
        dialog2._on_save()
        dialog2.destroy()

        # Verify second save preserved exact whitespace identically
        loaded2 = mgr.load()
        assert loaded2.prompt == exact_prompt

    finally:
        app.destroy()


def test_thinking_disabled_in_settings_dialog(tmp_path: Path):
    """Verify thinking option in AI Gateway settings is disabled and uneditable."""
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)
    try:
        app.update_idletasks()
        dialog = SettingsDialog(app, persistence=persistence)
        dialog.update_idletasks()

        # Thinking checkbutton state must be disabled
        assert str(dialog.chk_thinking["state"]) == "disabled"
        assert dialog.var_gw_thinking.get() is False
        dialog.var_gw_sub_model.set("scanner-route")
        dialog.var_gw_prime_model.set("finalizer-route")

        # Save settings and verify gateway_thinking is persisted as False
        dialog._on_save()
        mgr = SettingsManager(persistence=persistence)
        loaded = mgr.load()
        assert loaded.gateway_thinking is False

        dialog.destroy()
    finally:
        app.destroy()


def test_preview_voice_input_text_bytes_style_with_mocked_adapter(tmp_path: Path):
    """Verify preview voice handler invokes synthesis with exact input_text, bytes,
    and style using actual VoiceStudioAdapter with mocked HTTP transport.
    Test calls the actual handler, not inspecting code strings.
    """
    persistence = ProjectPersistence(storage_root=tmp_path)
    app = MainWindow(persistence=persistence)
    try:
        app.update_idletasks()
        dialog = SettingsDialog(app, persistence=persistence)
        dialog.update_idletasks()

        # Configure voice settings
        test_voice = "alloy"
        test_lang = "en-US"
        test_style = "cinematic-dramatic"
        dialog.var_voice_id.set(test_voice)
        dialog.var_voice_lang.set(test_lang)
        dialog.var_voice_style.set(test_style)

        # Generate expected dummy WAV bytes
        expected_wav_bytes = make_dummy_wav(duration_s=1.0)
        captured_requests = []

        def transport_handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            if "/health" in str(request.url):
                return httpx.Response(200, json={"status": "ok"})
            if "/v1/audio/speech" in str(request.url):
                body = json.loads(request.content.decode("utf-8"))
                # Verify input_text and style passed in payload
                assert "input" in body
                assert body["input"] == "Hello, this is a voice synthesis preview from ToolRecap V5."
                assert body["voice"] == test_voice
                assert body["description"] == test_style
                assert body["language"] == test_lang
                return httpx.Response(200, content=expected_wav_bytes)
            return httpx.Response(404)

        # Create actual VoiceStudioAdapter instance using MockTransport
        client = httpx.Client(transport=httpx.MockTransport(transport_handler))
        actual_adapter = VoiceStudioAdapter(mode="local", client=client)

        def run_sync_thread(target=None, *args, **kwargs):
            mock_t = MagicMock()
            mock_t.start = lambda: target() if target else None
            return mock_t

        with patch.object(dialog, "_get_voice_adapter", return_value=actual_adapter), \
             patch.object(dialog, "after", side_effect=lambda ms, cb=None, *a: cb(*a) if cb else None), \
             patch("threading.Thread", side_effect=run_sync_thread), \
             patch("winsound.PlaySound", return_value=None):

            # Call the actual preview handler (NOT inspecting strings)
            dialog._preview_voice()

        # 1. Verify HTTP request was made by the adapter
        assert len(captured_requests) >= 1

        # 2. Verify preview WAV file was written to disk with exact bytes
        preview_file = persistence.root / "cache" / "preview_sample.wav"
        assert preview_file.exists()
        assert preview_file.read_bytes() == expected_wav_bytes

        # 3. Verify progress and status updated
        assert dialog.voice_preview_prog["value"] == 100
        assert "✓" in dialog.voice_status_var.get()

        dialog.destroy()
    finally:
        app.destroy()
