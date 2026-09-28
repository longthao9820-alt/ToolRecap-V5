"""Single-boundary AI analysis for ToolRecap V5.

The desktop application prepares factual source material locally, then makes one
Gateway request.  The Gateway owns every editorial decision and returns the
canonical render JSON consumed by the existing renderer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from toolrecap_v4.analysis.models import PreparedEpisode
from toolrecap_v4.cancellation import CancellationToken
from toolrecap_v4.gateway import GatewayClient
from toolrecap_v4.schemas.schema import get_project_schema
from toolrecap_v4.settings import AppSettings

V5_ANALYSIS_CONTRACT_VERSION = "toolrecap-v5-gateway-render-json-v1"

V5_SYSTEM_PROMPT = """You are the sole editorial analysis engine for ToolRecap V5.
Analyse only the supplied source manifest and timestamped transcript according to
the user's prompt. Return exactly one JSON object matching the supplied JSON
Schema. Do not return Markdown, explanations, evidence objects, plans, drafts, or
intermediate analysis. Never invent a source filename or a timestamp outside a
source duration. All subtitle cue timestamps must be relative to their segment.
The desktop tool will validate and render your JSON without editorial rewriting."""


@dataclass(frozen=True)
class AnalysisResult:
    final_json: dict[str, Any]
    raw_response: str
    model: str
    request_digest: str


def select_analysis_model(settings: AppSettings) -> tuple[str, str]:
    """Use existing V4 settings without forcing a settings migration."""
    candidates = (
        (settings.finalizer_model, settings.finalizer_reasoning),
        (settings.gateway_prime_model, settings.gateway_prime_reasoning),
        (settings.planner_model, settings.planner_reasoning),
        (settings.writer_model, settings.writer_reasoning),
        (settings.scanner_model, settings.scanner_reasoning),
        (settings.gateway_model, settings.gateway_sub_reasoning),
        (settings.gateway_sub_model, settings.gateway_sub_reasoning),
    )
    for model, reasoning in candidates:
        if str(model).strip():
            return str(model).strip(), str(reasoning or "").strip()
    raise ValueError(
        "No AI Gateway model is configured. Configure the existing Finalizer or Scanner model."
    )


def _episode_payload(episode: PreparedEpisode) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "source_id": episode.source_id,
        "source_file": episode.source_basename or episode.source_path.name,
        "duration_ms": episode.duration_ms,
        "width": episode.canvas_width,
        "height": episode.canvas_height,
        "transcript_method": episode.transcript_method,
        "transcript_language": episode.transcript.language,
        "transcript": [
            {
                "start_ms": cue.start_ms,
                "end_ms": cue.end_ms,
                "text": cue.text,
            }
            for cue in episode.transcript.cues
        ],
    }


def _mode_rules(mode: str) -> list[str]:
    if mode == "HIGHLIGHT":
        return [
            "Select coherent highlight scenes according to the user's prompt.",
            "Use type=original_dialogue, narration='', source_audio=true, and audio_intent=ORIGINAL_AUDIO.",
            "Create separate outputs for unrelated storylines; do not collapse an entire season into one compilation.",
        ]
    return [
        "Create factual recap outputs according to the user's prompt.",
        "Use type=narration for commentary and type=original_dialogue only for intentionally preserved source dialogue.",
        "For narration segments set audio_intent=NARRATION_WITH_ORIGINAL_AUDIO unless the prompt clearly requires silent source audio.",
    ]


class GatewayRenderJsonAnalyzer:
    """Send one complete editorial request and accept one render JSON response."""

    def __init__(
        self,
        gateway: GatewayClient,
        settings: AppSettings,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.gateway = gateway
        self.settings = settings
        self.progress_callback = progress_callback

    def build_request(
        self,
        *,
        mode: str,
        project_id: str,
        project_name: str,
        editorial_prompt: str,
        episodes: Sequence[PreparedEpisode],
    ) -> dict[str, Any]:
        normalized_mode = str(mode).upper()
        if normalized_mode not in {"RECAP", "HIGHLIGHT"}:
            raise ValueError("V5 mode must be RECAP or HIGHLIGHT")
        return {
            "contract_version": V5_ANALYSIS_CONTRACT_VERSION,
            "task": "Analyse the supplied sources according to editorial_prompt and return final render JSON.",
            "mode": normalized_mode,
            "editorial_prompt": editorial_prompt,
            "project": {"project_id": project_id, "project_name": project_name},
            "sources": [_episode_payload(episode) for episode in episodes],
            "mode_rules": _mode_rules(normalized_mode),
            "render_json_schema": get_project_schema(),
            "required_envelope": {
                "schema_version": "3.0",
                "project_id": project_id,
                "project_name": project_name,
                "sources": [
                    {"source_file": episode.source_basename or episode.source_path.name,
                     "duration_ms": episode.duration_ms}
                    for episode in episodes
                ],
            },
        }

    def analyze(
        self,
        *,
        mode: str,
        project_id: str,
        project_name: str,
        editorial_prompt: str,
        episodes: Sequence[PreparedEpisode],
        cancellation_token: CancellationToken | None = None,
    ) -> AnalysisResult:
        request = self.build_request(
            mode=mode,
            project_id=project_id,
            project_name=project_name,
            editorial_prompt=editorial_prompt,
            episodes=episodes,
        )
        prompt = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        request_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        model, reasoning = select_analysis_model(self.settings)
        if self.progress_callback:
            self.progress_callback({
                "state": "WAITING FOR AI",
                "stage": "ai_gateway",
                "activity_text": "AI Gateway is analysing the prompt and composing Render JSON...",
                "waiting_for": "AI Gateway",
                "pipeline_mode": "V5",
            })
        result = self.gateway.submit_text_chat(
            prompt=prompt,
            system_prompt=V5_SYSTEM_PROMPT,
            model=model,
            reasoning_effort=reasoning or None,
            stream=True,
            expect_json=True,
            cancellation_token=cancellation_token,
            phase="v5_render_json",
        )
        parsed = result.parsed_json
        if not isinstance(parsed, dict):
            raise TypeError("AI Gateway must return one Render JSON object")
        return AnalysisResult(
            final_json=dict(parsed),
            raw_response=result.raw_response,
            model=result.model or model,
            request_digest=request_digest,
        )
