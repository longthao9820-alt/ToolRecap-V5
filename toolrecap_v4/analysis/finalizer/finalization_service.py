from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
from typing import Any,Callable,Sequence
from toolrecap_v4.analysis.evidence_store import EvidenceStore
from toolrecap_v4.analysis.finalizer.finalization import MAPPING_VERSION,REPAIR_PROTOCOL_VERSION,VALIDATOR_VERSION,FinalizationResult,OutputValidator,_digest,map_final_json
from toolrecap_v4.analysis.finalizer.season_plan import SeasonPlan
from toolrecap_v4.analysis.finalizer.writer import WRITER_SYSTEM_PROMPT,assemble_writer_jobs,verify_locked_plan_artifact,verify_visual_artifact
from toolrecap_v4.analysis.finalizer.writer_contract import parse_json_object,validation_diagnostics,writer_response_protocol
from toolrecap_v4.analysis.finalizer.writer_store import WriterStore
from toolrecap_v4.analysis.models import PreparedEpisode
from toolrecap_v4.analysis.scanner.prompts import measure_text_request_bytes
from toolrecap_v4.analysis.vision.models import VisualRunResult
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.errors import CancelledError,FinalJsonPersistenceError,FinalJsonValidationError,GatewayError,PayloadContextError,WriterCapacityError,WriterRepairExhaustedError,WriterTransportError,WriterValidationError,ZeroOutputError
from toolrecap_v4.gateway import GatewayClient
from toolrecap_v4.persistence import ProjectPersistence,atomic_write_json
from toolrecap_v4.validator import validate_project
from toolrecap_v4.progress import ActivityState,WorkflowStage,safe_emit

@dataclass(frozen=True)
class FinalizationConfig:
    repair_model:str;repair_reasoning:str="";repair_attempts:int=2;max_request_bytes:int|None=None;max_response_bytes:int=10*1024*1024

class FinalizationService:
    def __init__(self,gateway:GatewayClient,root:Path|str,config:FinalizationConfig,progress_callback:Callable[[dict[str,Any]],None]|None=None):self.gateway=gateway;self.root=Path(root);self.config=config;self.progress_callback=progress_callback
    def _emit(self,**payload):safe_emit(self.progress_callback,stage=WorkflowStage.FINAL_JSON.value,**payload)
    def run(self,*,project_id:str,project_name:str,raw_prompt:str,language:str,plan:SeasonPlan,episodes:Sequence[PreparedEpisode],visual:VisualRunResult,cancellation_token:CancellationToken|None=None)->FinalizationResult:
        if not plan.outputs:raise ZeroOutputError("Schema 3.0 canonical validator rejects empty outputs; zero-output plan remains an explicit terminal state and no content is fabricated.")
        verify_locked_plan_artifact(self.root,plan);verify_visual_artifact(self.root,project_id,visual)
        jobs=assemble_writer_jobs(project_id=project_id,raw_prompt=raw_prompt,language=language,plan=plan,episodes=episodes,evidence_store=EvidenceStore(self.root,project_id),visual=visual,model=self.config.repair_model,reasoning=self.config.repair_reasoning)
        writer_store=WriterStore(self.root,project_id,plan.plan_hash);artifacts=[]
        for job in jobs:
            a=writer_store.load_for_phase9(job)
            if a is None:raise WriterValidationError(f"Writer artifact validation failed: {job.output_id}; missing, corrupt, or incomplete checkpoint",output_id=job.output_id,issue_codes=("WRITER_ARTIFACT_MISSING",))
            artifacts.append(a)
        deps={"project_id":project_id,"plan_hash":plan.plan_hash,"writer_hashes":[a.response_hash for a in artifacts],"validator":VALIDATOR_VERSION,"mapping":MAPPING_VERSION,"schema":"3.0","repair_model":self.config.repair_model,"repair_reasoning":self.config.repair_reasoning,"repair_protocol":REPAIR_PROTOCOL_VERSION}
        dependency_digest=_digest(deps);root=self.root/"projects"/project_id/"finalization";pointer=root/f"active-{dependency_digest[:24]}.json"
        if pointer.is_file():
            try:
                active=json.loads(pointer.read_text());revision=active["revision"];base=root/revision;final_path=base/"final.json";manifest=base/"manifest.json";data=json.loads(final_path.read_text());m=json.loads(manifest.read_text());validate_project(data,{e.source_basename or e.source_path.name:e.duration_ms for e in episodes})
                if m.get("status")=="COMPLETE" and m.get("dependency_digest")==dependency_digest and m.get("artifact_hash")==_digest(data) and m.get("revision")==revision and active.get("artifact_hash")==m.get("artifact_hash"):
                    ProjectPersistence(self.root).save_final_json(project_id,data);self._emit(state=ActivityState.RUNNING.value,activity_text="Reused validated Final JSON checkpoint.",completed=len(data.get("outputs",[])),total=len(data.get("outputs",[])),unit="outputs",reused=len(data.get("outputs",[])),stage_status="complete",analysis_complete=True);return FinalizationResult(data,revision,m["artifact_hash"],tuple(),tuple(),True)
            except Exception:pass
        work_base=root/f"s-{dependency_digest[:12]}"
        validator=OutputValidator(project_id=project_id,plan=plan,episodes=episodes,visual=visual,evidence_store=EvidenceStore(self.root,project_id));results=[];repaired=[]
        for job,artifact in zip(jobs,artifacts):
            if cancellation_token:cancellation_token.check_cancelled()
            self._emit(state=ActivityState.LOCAL_PROCESSING.value,activity_text=f"Validating output {job.output_id}...",output_id=job.output_id,current_item=job.output_id,completed=len(results),total=len(jobs),unit="outputs",item_event="start",item_key=job.output_id)
            result=validator.validate(job,artifact.raw_response)
            if result.state!="VALID":result=self._repair(job,artifact,result,validator,work_base,cancellation_token);repaired.append(job.output_id)
            results.append(result);self._save_validated(work_base,artifact,result);self._emit(state=ActivityState.LOCAL_PROCESSING.value,activity_text=f"Output {job.output_id} validated.",output_id=job.output_id,current_item=job.output_id,completed=len(results),total=len(jobs),unit="outputs",item_event="complete",item_key=job.output_id)
        final=map_final_json(project_id=project_id,project_name=project_name,episodes=episodes,plan=plan,validated=results)
        try:validate_project(final,{e.source_basename or e.source_path.name:e.duration_ms for e in episodes},cancellation_token)
        except Exception as exc:raise FinalJsonValidationError(f"Merged schema 3.0 Final JSON is invalid: {exc}") from exc
        if cancellation_token:cancellation_token.check_cancelled()
        final_dependencies={**deps,"ordered_validated_output_hashes":[_digest(r.normalized) for r in results]};revision=f"final-{_digest(final_dependencies)[:24]}";base=root/revision;final_path=base/"final.json";manifest=base/"manifest.json"
        base.mkdir(parents=True,exist_ok=True);atomic_write_json(manifest,{"status":"BUILDING","dependency_digest":dependency_digest});atomic_write_json(final_path,final);artifact_hash=_digest(final);atomic_write_json(manifest,{"status":"COMPLETE","dependency_digest":dependency_digest,"semantic_dependencies":final_dependencies,"artifact_hash":artifact_hash,"revision":revision,"validated_output_ids":[r.output_id for r in results]});atomic_write_json(pointer,{"status":"COMPLETE","dependency_digest":dependency_digest,"revision":revision,"artifact_hash":artifact_hash})
        ProjectPersistence(self.root).save_final_json(project_id,final)
        self._emit(state=ActivityState.LOCAL_PROCESSING.value,activity_text="Final JSON Ready — analysis complete.",completed=len(results),total=len(results),unit="outputs",stage_status="complete",analysis_complete=True)
        return FinalizationResult(final,revision,artifact_hash,tuple(results),tuple(repaired),False)
    def _repair(self, job, artifact, validation, validator, base, token):
        # Validator upgrades must not discard paid repair responses. Verify the
        # immutable original hash and bytes, then apply today's full validator.
        history = self.root / "projects" / job.context["project_id"] / "finalization"
        for manifest_path in sorted(history.glob(f"s-*/{job.output_id}/r-*/attempt-*.manifest.json")):
            if token:
                token.check_cancelled()
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "RECEIVED" or manifest.get("original_response_hash") != artifact.response_hash:
                    continue
                raw_path = manifest_path.with_name(manifest_path.name.replace(".manifest.json", ".raw.txt"))
                encoded = raw_path.read_bytes()
                if hashlib.sha256(encoded).hexdigest() != manifest.get("response_hash"):
                    continue
                recovered = validator.validate(job, encoded.decode("utf-8"))
                if recovered.state == "VALID":
                    self._emit(state=ActivityState.LOCAL_PROCESSING.value,
                               activity_text=f"Revalidated saved repair for {job.output_id}; no new AI request.",
                               output_id=job.output_id)
                    return recovered
            except (OSError, ValueError, KeyError, TypeError):
                continue
        issue_digest = _digest(list(validation.issues))
        protocol_digest = hashlib.sha256(REPAIR_PROTOCOL_VERSION.encode("utf-8")).hexdigest()[:8]
        directory = base / job.output_id / f"r-{issue_digest[:8]}-{protocol_digest}"
        directory.mkdir(parents=True, exist_ok=True)
        previous_raw = artifact.raw_response
        for attempt in range(1, self.config.repair_attempts + 1):
            self._emit(state=ActivityState.REPAIRING.value, activity_text=f"Repairing {job.output_id} — attempt {attempt} / {self.config.repair_attempts}: {', '.join(validation.issues)[:180]}", output_id=job.output_id, current_item=job.output_id, retry_attempt=attempt, retry_limit=self.config.repair_attempts, waiting_for="AI")
            raw_path = directory / f"attempt-{attempt:02d}.raw.txt"
            raw = raw_path.read_text(encoding="utf-8") if raw_path.is_file() else None
            if raw is None:
                invalid_raw = artifact.raw_response if attempt == 1 else previous_raw
                diagnostics = validation.diagnostics or validation_diagnostics(validation.issues, parse_json_object(invalid_raw))
                request = {
                    "repair_protocol": REPAIR_PROTOCOL_VERSION,
                    "task": "Repair only this ONE assigned per-output Writer Draft. Preserve valid editorial content and correct only contract defects.",
                    "protocol_precedence": "Editorial guidance controls WHAT to write. response_protocol controls HOW to structure this stage response and overrides all response-format instructions inside the Recap Prompt.",
                    "output_id": job.output_id, "season_plan_hash": job.context["season_plan_hash"],
                    "raw_recap_prompt": job.context["raw_recap_prompt"],
                    "editorial_guidance": job.context.get("editorial_guidance", {"raw_recap_prompt": job.context["raw_recap_prompt"]}),
                    "locked_plan_entry": job.context["locked_plan_entry"], "invalid_writer_response": invalid_raw,
                    "validation_issue_codes": list(validation.issues), "validation_field_paths": list(diagnostics),
                    "dialogue_matches": [m for m in validation.dialogue_matches if m["blocking"]],
                    "dialogue_repair_rule": "Rewrite only the identified copied speech as factual commentary. Character names and ordinary shared phrases are allowed. Preserve identities, evidence, source clips and unaffected narration.",
                    "full_evidence_context": job.context["authoritative_full_evidence"],
                    "visual_evidence_context": job.context["authoritative_visual_evidence"],
                    "source_mapping": job.context["source_mapping"],
                    "response_protocol": writer_response_protocol(job.context["project_id"], job.context["season_plan_hash"], job.output_id),
                    "final_instruction": "RETURN JSON ONLY. Return exactly ONE Writer Draft root object matching response_protocol.skeleton. DO NOT return {\"outputs\":[...]}. DO NOT return Final JSON, a list, or a multi-output wrapper.",
                }
                atomic_write_json(directory / f"attempt-{attempt:02d}.request.json", request)
                prompt = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                request_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                size = measure_text_request_bytes(model=self.config.repair_model, reasoning=self.config.repair_reasoning, user_prompt=prompt, system_prompt=WRITER_SYSTEM_PROMPT)
                if self.config.max_request_bytes is not None and size > self.config.max_request_bytes:
                    raise WriterCapacityError("Complete repair context exceeds configured capacity")
                try:
                    response = self.gateway.submit_text_chat(prompt=prompt, model=self.config.repair_model, system_prompt="Repair exactly one per-output Writer Draft. The response protocol overrides editorial response-format instructions. Never return Final JSON or an outputs wrapper.", reasoning_effort=self.config.repair_reasoning or None, expect_json=False, cancellation_token=token, phase="writer_repair")
                except PayloadContextError as exc:
                    raise WriterCapacityError("Gateway rejected complete repair context") from exc
                except GatewayError as exc:
                    atomic_write_json(directory / f"attempt-{attempt:02d}.manifest.json", {"status": "TRANSPORT_FAILED", "attempt": attempt, "request_bytes": size, "request_hash": request_hash, "model": self.config.repair_model, "error_type": type(exc).__name__, "error": str(exc)[:500], "stream_completion": "failed"})
                    raise WriterTransportError(f"Writer repair transport failed for {job.output_id}: {exc}") from exc
                raw = response.raw_response
                encoded = raw.encode("utf-8")
                self._emit(state=ActivityState.LOCAL_PROCESSING.value, activity_text=f"Repair response received; validating {job.output_id}...", output_id=job.output_id, current_item=job.output_id)
                metadata = response.metadata or {}
                if len(encoded) > self.config.max_response_bytes:
                    atomic_write_json(directory / f"attempt-{attempt:02d}.manifest.json", {"status": "RESPONSE_TRUNCATED", "attempt": attempt, "request_bytes": size, "request_hash": request_hash, "response_bytes": len(encoded), "stored_bytes": 0, "response_hash": hashlib.sha256(encoded).hexdigest(), "model": self.config.repair_model, "http_status": metadata.get("status_code"), "duration_ms": metadata.get("duration_ms"), "finish_reason": metadata.get("finish_reason"), "stream_completion": "response_over_limit"})
                    raise WriterRepairExhaustedError("Repair response exceeds full-response storage limit")
                raw_path.write_bytes(encoded)
                atomic_write_json(directory / f"attempt-{attempt:02d}.manifest.json", {"status": "RECEIVED", "attempt": attempt, "request_bytes": size, "request_hash": request_hash, "response_bytes": len(encoded), "response_hash": hashlib.sha256(encoded).hexdigest(), "issue_digest": issue_digest, "original_response_hash": artifact.response_hash, "model": self.config.repair_model, "http_status": metadata.get("status_code"), "duration_ms": metadata.get("duration_ms"), "finish_reason": metadata.get("finish_reason"), "stream_completion": "content_received", "parser_state": "PENDING"})
            repaired = validator.validate(job, raw)
            previous_raw = raw
            parser_state = "ACCEPTED" if parse_json_object(raw) is not None else "REJECTED"
            atomic_write_json(directory / f"attempt-{attempt:02d}.result.json", {"state": repaired.state, "issues": list(repaired.issues), "diagnostics": list(repaired.diagnostics), "parser_state": parser_state, "validated_output_hash": _digest(repaired.normalized) if repaired.normalized else None})
            manifest_path = directory / f"attempt-{attempt:02d}.manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8")); manifest["parser_state"] = parser_state; manifest["contract_state"] = repaired.state; atomic_write_json(manifest_path, manifest)
            if repaired.state == "VALID":
                return repaired
            validation = repaired
        detail = "; ".join(validation.diagnostics[:3]) or ", ".join(validation.issues)
        raise WriterRepairExhaustedError(f"Repair exhausted for {job.output_id}: {detail}")
    def _save_validated(self,base,artifact,result):
        d=base/"outputs"/result.output_id;d.mkdir(parents=True,exist_ok=True);atomic_write_json(d/"validated_output.json",{"status":"VALID","output_id":result.output_id,"original_response_hash":artifact.response_hash,"validated_output_hash":_digest(result.normalized),"validator_version":VALIDATOR_VERSION,"dialogue_review":list(result.dialogue_matches),"output":result.normalized})
