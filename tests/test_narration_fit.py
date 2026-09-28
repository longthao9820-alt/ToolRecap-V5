from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest

from toolrecap_v4.downstream import apply_commentary_speed
from toolrecap_v4.narration_fit import FIT_READY, resolve_narration_fit
from toolrecap_v4.settings import AppSettings, COMMENTARY_SPEED_VALUES
from toolrecap_v4.settings import SettingsManager


def wav(seconds: float, rate: int = 8000) -> bytes:
    buf=io.BytesIO()
    with wave.open(buf,"wb") as handle:
        handle.setnchannels(1);handle.setsampwidth(2);handle.setframerate(rate);handle.writeframes(b"\0\0"*int(seconds*rate))
    return buf.getvalue()


def test_fixed_speed_fit_extends_then_holds_without_voice_speed_change() -> None:
    fit=resolve_narration_fit(segment_id="seg_008-02",start_ms=0,end_ms=5172,source_duration_ms=7000,narration_duration_ms=9400)
    assert fit.state==FIT_READY
    assert fit.narration_duration_ms==9400
    assert fit.render_duration_ms==9400
    assert fit.source_end_ms==7000
    assert fit.hold_ms==2400
    assert fit.strategy==("EXTEND_SAME_SOURCE","SOURCE_FRAME_HOLD")


def test_shorter_narration_trims_visual_selection_not_voice() -> None:
    fit=resolve_narration_fit(segment_id="s",start_ms=100,end_ms=5000,source_duration_ms=10000,narration_duration_ms=2000)
    assert fit.render_duration_ms==2000 and fit.source_end_ms==2100 and fit.hold_ms==0


def test_commentary_speed_is_fixed_and_validated() -> None:
    assert AppSettings().commentary_reading_speed==1.0
    assert COMMENTARY_SPEED_VALUES==(0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3)
    with pytest.raises(ValueError):AppSettings(commentary_reading_speed=1.01)


def test_global_tempo_changes_duration_without_per_segment_policy() -> None:
    normal=apply_commentary_speed(wav(1.0),1.0)
    faster=apply_commentary_speed(wav(1.0),1.1)
    with wave.open(io.BytesIO(normal),"rb") as a,wave.open(io.BytesIO(faster),"rb") as b:
        assert a.getnframes()/a.getframerate()==pytest.approx(1.0,abs=.02)
        assert b.getnframes()/b.getframerate()==pytest.approx(1/1.1,abs=.03)


def test_speed_setting_persists_and_legacy_defaults(tmp_path:Path)->None:
    manager=SettingsManager(storage_root=tmp_path);manager.save(AppSettings(commentary_reading_speed=.9))
    assert manager.load().commentary_reading_speed==.9
    legacy=AppSettings.from_dict({"settings_schema_version":10})
    assert legacy.commentary_reading_speed==1.0
