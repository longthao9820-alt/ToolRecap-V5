from __future__ import annotations
import io,json,wave
from pathlib import Path
import pytest
from toolrecap_v4.downstream import DownstreamVoiceCache
from toolrecap_v4.errors import InvalidAudioError
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError
from toolrecap_v4.settings import AppSettings
from toolrecap_v4.workflow import compute_output_fingerprint

def wav_bytes(seconds=.1):
    buf=io.BytesIO()
    with wave.open(buf,"wb") as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(8000);w.writeframes(b"\x00\x00"*int(8000*seconds))
    return buf.getvalue()
class Voice:
    def __init__(self,data=None):self.calls=[];self.data=data if data is not None else wav_bytes()
    def synthesize(self,text,**kwargs):self.calls.append((text,kwargs));return self.data
def output():return {"render_id":"out_001","title":"Title","segments":[{"segment_id":"seg-001","source_file":"E01.mp4","start_ms":0,"end_ms":1000,"type":"narration","narration":"Hello world","source_audio":False,"subtitles":[]}]}

def test_voice_cache_reuse_and_downstream_invalidation_boundaries(tmp_path):
    cache=DownstreamVoiceCache(tmp_path,"project-1");voice=Voice();settings=AppSettings(voice_id="voice-a")
    first=cache.prepare_output(output_def=output(),final_json_hash="final-hash",settings=settings,voice_adapter=voice);assert first.synthesized_count==1 and len(voice.calls)==1
    mix_changed=AppSettings(voice_id="voice-a",auto_duck=True,original_audio_db=-10,quality="low",use_gpu=False)
    second=cache.prepare_output(output_def=output(),final_json_hash="final-hash",settings=mix_changed,voice_adapter=voice);assert second.reused_count==1 and len(voice.calls)==1
    voice_changed=AppSettings(voice_id="voice-b",auto_duck=True)
    third=cache.prepare_output(output_def=output(),final_json_hash="final-hash",settings=voice_changed,voice_adapter=voice);assert third.synthesized_count==1 and len(voice.calls)==2

def test_remote_voice_cache_does_not_require_or_depend_on_local_endpoint(tmp_path):
    cache=DownstreamVoiceCache(tmp_path,"project-1");voice=Voice();a=AppSettings(voice_mode="remote",voice_local_url="http://missing-local:3900",voice_remote_url="https://remote.example")
    cache.prepare_output(output_def=output(),final_json_hash="hash",settings=a,voice_adapter=voice)
    b=AppSettings(voice_mode="remote",voice_local_url="http://another-missing-local:9999",voice_remote_url="https://remote.example")
    result=cache.prepare_output(output_def=output(),final_json_hash="hash",settings=b,voice_adapter=voice)
    assert result.reused_count==1 and len(voice.calls)==1

@pytest.mark.parametrize("payload",[b"",b"not wav",b"RIFF\x00\x00\x00\x00WAVE"])
def test_invalid_voice_audio_never_becomes_cache_hit(tmp_path,payload):
    cache=DownstreamVoiceCache(tmp_path,"project-1");voice=Voice(payload)
    with pytest.raises(InvalidAudioError):cache.prepare_output(output_def=output(),final_json_hash="hash",settings=AppSettings(),voice_adapter=voice)
    assert not list((tmp_path/"projects"/"project-1"/"downstream"/"voice").rglob("manifest.json"))

def test_partial_voice_success_is_preserved_and_retry_only_synthesizes_missing(tmp_path):
    out=output();out["segments"].append({**out["segments"][0],"segment_id":"seg-002","narration":"Second"})
    class Partial(Voice):
        def synthesize(self,text,**kwargs):
            self.calls.append((text,kwargs))
            if text=="Second" and sum(1 for call in self.calls if call[0]=="Second")==1:raise RuntimeError("temporary VoiceStudio failure")
            return self.data
    voice=Partial();cache=DownstreamVoiceCache(tmp_path,"project-1")
    with pytest.raises(RuntimeError):cache.prepare_output(output_def=out,final_json_hash="hash",settings=AppSettings(),voice_adapter=voice)
    assert len(list((tmp_path/"projects"/"project-1"/"downstream"/"voice"/"out_001"/"seg-001").rglob("manifest.json")))==1
    result=cache.prepare_output(output_def=out,final_json_hash="hash",settings=AppSettings(),voice_adapter=voice)
    assert result.reused_count==1 and result.synthesized_count==1 and len(voice.calls)==3

def test_corrupt_cached_voice_is_resynthesized(tmp_path):
    cache=DownstreamVoiceCache(tmp_path,"project-1");voice=Voice();first=cache.prepare_output(output_def=output(),final_json_hash="hash",settings=AppSettings(),voice_adapter=voice);next(iter(first.narration_audio_map.values())).write_bytes(b"corrupt")
    second=cache.prepare_output(output_def=output(),final_json_hash="hash",settings=AppSettings(),voice_adapter=voice)
    assert second.synthesized_count==1 and len(voice.calls)==2

def test_voice_cache_cancellation_preserves_final_json_boundary(tmp_path):
    token=CancellationToken();token.cancel()
    with pytest.raises(CancelledError):DownstreamVoiceCache(tmp_path,"project-1").prepare_output(output_def=output(),final_json_hash="hash",settings=AppSettings(),voice_adapter=Voice(),cancellation_token=token)
    assert not list((tmp_path/"projects"/"project-1"/"downstream").rglob("manifest.json"))

def test_final_json_bytes_immutable_during_voice_cache(tmp_path):
    final={"schema_version":"3.0","project_id":"project-1","project_name":"P","sources":[{"source_file":"E01.mp4"}],"outputs":[output()]};path=tmp_path/"final.json";path.write_text(json.dumps(final,ensure_ascii=False));before=path.read_bytes()
    DownstreamVoiceCache(tmp_path,"project-1").prepare_output(output_def=output(),final_json_hash="hash",settings=AppSettings(),voice_adapter=Voice())
    assert path.read_bytes()==before

def test_output_fingerprint_excludes_ai_settings_but_tracks_downstream_settings():
    out=output();sources={"E01.mp4":{"sha256":"abc","size_bytes":1}}
    base=AppSettings(scanner_model="scan-a",vision_model="vis-a",finalizer_model="fin-a")
    fp=compute_output_fingerprint(out,sources,base)
    ai=AppSettings(scanner_model="scan-b",vision_model="vis-b",finalizer_model="fin-b")
    assert compute_output_fingerprint(out,sources,ai)==fp
    assert compute_output_fingerprint(out,sources,AppSettings(voice_id="different"))!=fp
    assert compute_output_fingerprint(out,sources,AppSettings(auto_duck=True))!=fp
    assert compute_output_fingerprint(out,sources,AppSettings(use_gpu=False))!=fp
