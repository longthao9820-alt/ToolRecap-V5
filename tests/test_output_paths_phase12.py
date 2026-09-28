from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import wave

import pytest

from toolrecap_v4.downstream import DownstreamVoiceCache
from toolrecap_v4.renderer import SourceCollisionError
from toolrecap_v4.gateway import GatewayClient
from toolrecap_v4.output_paths import (
    OutputDirectoryError,
    derive_working_folder,
    ensure_publication_root,
    resolve_publication_root,
)
from toolrecap_v4.persistence import ProjectPersistence
from toolrecap_v4.settings import AppSettings
from toolrecap_v4.workflow import ProjectStatus, ProjectWorkflow


def test_folder_and_single_file_auto_are_exact_sibling(tmp_path):
    working = tmp_path / "Yellowstone (2018)" / "S03"
    working.mkdir(parents=True)
    episode = working / "E01.mkv"; episode.write_bytes(b"x")
    expected = working.parent / "Outputs_S03"
    assert derive_working_folder(working) == working
    assert derive_working_folder(episode) == working
    assert resolve_publication_root(manual_output_dir="", working_folder=working) == expected


def test_unicode_spaces_same_and_different_working_folders(tmp_path):
    s3 = tmp_path / "Chương Trình Season 3"; s4 = tmp_path / "S04"
    s3.mkdir(); s4.mkdir()
    e1, e2 = s3 / "E01.mp4", s3 / "E02.mp4"; e1.write_bytes(b"1"); e2.write_bytes(b"2")
    expected = tmp_path / "Outputs_Chương Trình Season 3"
    assert derive_working_folder([e1, e2]) == s3
    assert resolve_publication_root(
        manual_output_dir="", working_folder=derive_working_folder([e1, e2])
    ) == expected
    assert resolve_publication_root(manual_output_dir=None, working_folder=derive_working_folder(e1)) == expected
    assert resolve_publication_root(manual_output_dir=None, working_folder=derive_working_folder(e2)) == expected
    assert resolve_publication_root(manual_output_dir=None, working_folder=s4) == tmp_path / "Outputs_S04"


def test_manual_exact_and_manual_auto_transitions(tmp_path):
    working = tmp_path / "S03"; working.mkdir(); manual = tmp_path / "Finished"
    assert resolve_publication_root(manual_output_dir=manual, working_folder=working) == manual
    assert resolve_publication_root(manual_output_dir="", working_folder=working) == tmp_path / "Outputs_S03"
    assert resolve_publication_root(manual_output_dir=manual, working_folder=working) == manual


def test_root_relative_and_multi_parent_edges_are_explicit(tmp_path):
    with pytest.raises(OutputDirectoryError, match="absolute"):
        resolve_publication_root(manual_output_dir="relative", working_folder=tmp_path)
    with pytest.raises(OutputDirectoryError, match="manual"):
        resolve_publication_root(manual_output_dir="", working_folder=Path(tmp_path.anchor))
    a, b = tmp_path / "a" / "1.mp4", tmp_path / "b" / "2.mp4"
    a.parent.mkdir(); b.parent.mkdir(); a.write_bytes(b"a"); b.write_bytes(b"b")
    with pytest.raises(OutputDirectoryError, match="multiple"):
        derive_working_folder([a, b])


def test_creation_failures_never_fallback(tmp_path, monkeypatch):
    file_target = tmp_path / "target"; file_target.write_text("unrelated")
    with pytest.raises(OutputDirectoryError, match="file"):
        ensure_publication_root(file_target)
    unavailable = tmp_path / "unavailable"; original = Path.mkdir
    def fail(self, *args, **kwargs):
        if self == unavailable: raise PermissionError("denied")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "mkdir", fail)
    with pytest.raises(OutputDirectoryError, match="Cannot create"):
        ensure_publication_root(unavailable)


def test_auto_resolution_not_persisted_or_created_during_project_creation(tmp_path, monkeypatch):
    source = tmp_path / "Show" / "S03" / "E01.mp4"; source.parent.mkdir(parents=True); source.write_bytes(b"source")
    persistence = ProjectPersistence(storage_root=tmp_path / "managed")
    monkeypatch.setattr("toolrecap_v4.workflow.probe_media", lambda *a, **k: SimpleNamespace(duration=1.0))
    state = ProjectWorkflow(persistence=persistence).create_project("project-1", "Project", source, settings=AppSettings(output_dir=""))
    assert state["manual_output_dir"] == ""
    assert Path(state["output_dir"]) == source.parent.parent / "Outputs_S03"
    assert not Path(state["output_dir"]).exists()
    assert SettingsLikeBlank().output_dir == ""
    other = tmp_path / "Other" / "S02" / "E01.mp4"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"other")
    second = ProjectWorkflow(persistence=persistence).create_project(
        "project-2", "Project2", other, settings=AppSettings(output_dir="")
    )
    assert second["manual_output_dir"] == ""
    assert Path(second["output_dir"]) == other.parent.parent / "Outputs_S02"


class SettingsLikeBlank:
    output_dir = ""


def _final(project_id, source_name, narration=False):
    return {"schema_version":"3.0","project_id":project_id,"project_name":"Project","sources":[{"source_file":source_name}],"outputs":[{"render_id":"out_001","title":"Human Title","segments":[{"segment_id":"seg-001","source_file":source_name,"start_ms":0,"end_ms":1000,"type":"narration" if narration else "original_dialogue","narration":"Narration" if narration else "","source_audio":not narration,"subtitles":[]}]}]}


def _render_result(output_dir):
    output_dir.mkdir(parents=True,exist_ok=True);video=output_dir/"Human Title.mp4";video.write_bytes(b"rendered")
    narration=output_dir/"Human Title.narration.srt";original=output_dir/"Human Title.original.srt"
    narration.write_text("",encoding="utf-8");original.write_text("",encoding="utf-8")
    return SimpleNamespace(output_path=video,narration_srt_path=narration,original_srt_path=original,duration=1.0,video_codec="h264",audio_codec="aac",width=640,height=480,fps=25.0,original_subtitle_state="NO_ORIGINAL_DIALOGUE",original_subtitle_cue_count=0)


def test_imported_final_json_auto_publication_is_clean_and_zero_ai(tmp_path, monkeypatch):
    source = tmp_path / "Show" / "S03" / "E01.mp4"; source.parent.mkdir(parents=True); source.write_bytes(b"source")
    persistence=ProjectPersistence(storage_root=tmp_path/"managed");gateway=MagicMock(spec=GatewayClient);workflow=ProjectWorkflow(persistence=persistence,gateway_client=gateway)
    monkeypatch.setattr("toolrecap_v4.workflow.probe_media",lambda *a,**k:SimpleNamespace(duration=1.0))
    expected=source.parent.parent/"Outputs_S03"
    workflow.import_project("project-1","Project",source,_final("project-1",source.name),settings=AppSettings(output_dir=""))
    monkeypatch.setattr("toolrecap_v4.workflow.render_output",lambda **kwargs:_render_result(Path(kwargs["output_dir"])))
    state=workflow.start_project("project-1")
    assert state["status"]==ProjectStatus.COMPLETED.value and Path(state["output_dir"])==expected
    assert gateway.submit_text_chat.call_count==gateway.submit_image_chat.call_count==0
    assert {p.name for p in expected.iterdir()}=={"Human Title.mp4","Human Title.narration.srt","Human Title.original.srt"}
    assert not any(p.suffix in {".json",".wav"} for p in expected.iterdir())


def test_manual_destination_change_reuses_voice_final_json_and_zero_ai(tmp_path, monkeypatch):
    source=tmp_path/"Show"/"S03"/"E01.mp4";source.parent.mkdir(parents=True);source.write_bytes(b"source")
    a,b=tmp_path/"A",tmp_path/"B";persistence=ProjectPersistence(storage_root=tmp_path/"managed");gateway=MagicMock(spec=GatewayClient)
    wav=io.BytesIO()
    with wave.open(wav,"wb") as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(8000);w.writeframes(b"\x00\x00"*800)
    voice=MagicMock();voice.synthesize.return_value=wav.getvalue();workflow=ProjectWorkflow(persistence=persistence,gateway_client=gateway,voice_adapter=voice)
    monkeypatch.setattr("toolrecap_v4.workflow.probe_media",lambda *a,**k:SimpleNamespace(duration=1.0))
    workflow.import_project("project-1","Project",source,_final("project-1",source.name,True),output_dir=a)
    render_calls=[]
    def render(**kwargs):
        render_calls.append(Path(kwargs["output_dir"]))
        return _render_result(Path(kwargs["output_dir"]))
    monkeypatch.setattr("toolrecap_v4.workflow.render_output",render)
    before=persistence._final_path("project-1").read_bytes();workflow.start_project("project-1",settings=AppSettings(output_dir=str(a)));assert voice.synthesize.call_count==1
    workflow.retry_project("project-1",settings=AppSettings(output_dir=str(b)))
    assert voice.synthesize.call_count==1 and (b/"Human Title.mp4").is_file() and persistence._final_path("project-1").read_bytes()==before
    assert render_calls == [a, b]
    assert gateway.submit_text_chat.call_count==gateway.submit_image_chat.call_count==0


def test_partial_three_output_resume_uses_resolved_destination_and_zero_ai(tmp_path, monkeypatch):
    source=tmp_path/"Show"/"S03"/"E01.mp4";source.parent.mkdir(parents=True);source.write_bytes(b"source")
    publication=tmp_path/"Published";persistence=ProjectPersistence(storage_root=tmp_path/"managed");gateway=MagicMock(spec=GatewayClient)
    final=_final("project-1",source.name)
    final["outputs"]=[
        {**final["outputs"][0],"render_id":f"out_{number:03d}","title":f"Part {number}"}
        for number in (1,2,3)
    ]
    workflow=ProjectWorkflow(persistence=persistence,gateway_client=gateway)
    monkeypatch.setattr("toolrecap_v4.workflow.probe_media",lambda *a,**k:SimpleNamespace(duration=1.0))
    workflow.import_project("project-1","Project",source,final,output_dir=publication)
    calls=[]
    def render(**kwargs):
        title=kwargs["output_def"]["title"];calls.append(title)
        output_dir=Path(kwargs["output_dir"]);output_dir.mkdir(parents=True,exist_ok=True)
        video=output_dir/f"{title}.mp4";video.write_bytes(title.encode())
        narration=output_dir/f"{title}.narration.srt";original=output_dir/f"{title}.original.srt"
        narration.write_text("",encoding="utf-8");original.write_text("",encoding="utf-8")
        return SimpleNamespace(output_path=video,narration_srt_path=narration,original_srt_path=original,duration=1.0,video_codec="h264",audio_codec="aac",width=640,height=480,fps=25.0,original_subtitle_state="NO_ORIGINAL_DIALOGUE",original_subtitle_cue_count=0)
    monkeypatch.setattr("toolrecap_v4.workflow.render_output",render)
    workflow.start_project("project-1")
    (publication/"Part 2.mp4").unlink()
    calls.clear()
    resumed=workflow.resume_project("project-1")
    assert calls == ["Part 2"]
    assert resumed["outputs"]["out_001"]["status"] == "skipped"
    assert resumed["outputs"]["out_002"]["status"] == "completed"
    assert resumed["outputs"]["out_003"]["status"] == "skipped"
    assert gateway.submit_text_chat.call_count==gateway.submit_image_chat.call_count==0


def test_unowned_publication_collision_is_not_overwritten(tmp_path, monkeypatch):
    source=tmp_path/"E01.mp4";source.write_bytes(b"source");manual=tmp_path/"pub";manual.mkdir();target=manual/"Human Title.mp4";target.write_bytes(b"unrelated")
    persistence=ProjectPersistence(storage_root=tmp_path/"managed");workflow=ProjectWorkflow(persistence=persistence)
    monkeypatch.setattr("toolrecap_v4.workflow.probe_media",lambda *a,**k:SimpleNamespace(duration=1.0))
    workflow.import_project("project-1","Project",source,_final("project-1",source.name),output_dir=manual)
    with pytest.raises(SourceCollisionError,match="without a matching project checkpoint"):
        workflow.start_project("project-1")
    assert target.read_bytes()==b"unrelated"
