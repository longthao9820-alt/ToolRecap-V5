"""Readable plan review and revision-bound approval before rendering."""
from __future__ import annotations

import hashlib
import json
from typing import Any
from toolrecap_v4.persistence import atomic_write_json


def plan_data(state: dict[str, Any]) -> dict[str, Any] | None:
    if state.get("pipeline_version") == "5.0":
        return state.get("final_json")
    return state.get("highlight_json") if state.get("project_mode") == "HIGHLIGHT" else state.get("final_json")


def plan_digest(state: dict[str, Any]) -> str | None:
    value = plan_data(state)
    if not isinstance(value, dict):
        return None
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode()).hexdigest()


def _clock(ms: int) -> str:
    seconds, milliseconds = divmod(ms, 1000)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def parse_timecode(value: str) -> int:
    """Accept HH:MM:SS.mmm or MM:SS.mmm for creator cut adjustments."""
    pieces = value.strip().replace(",", ".").split(":")
    if len(pieces) not in (2, 3):
        raise ValueError("Use MM:SS.mmm or HH:MM:SS.mmm")
    try:
        if len(pieces) == 2:
            hours = 0
            minutes = int(pieces[0])
        else:
            hours, minutes = int(pieces[0]), int(pieces[1])
        seconds, _, fraction = pieces[-1].partition(".")
        second = int(seconds)
        if hours < 0 or minutes < 0 or second < 0 or second >= 60 or (len(pieces) == 3 and minutes >= 60):
            raise ValueError
        if fraction and (not fraction.isdigit() or len(fraction) > 3):
            raise ValueError
        ms = int(fraction.ljust(3, "0")) if fraction else 0
    except ValueError as exc:
        raise ValueError("Use a valid time such as 00:14:53.950") from exc
    return ((hours * 60 + minutes) * 60 + second) * 1000 + ms


def save_reviewed_plan(persistence, project_id: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Validate a creator's edits and update only that project's active plan."""
    state = persistence.load_project(project_id)
    if state.get("status") != "review_required":
        raise ValueError("Plan edits are available only before rendering")
    if state.get("project_mode") == "HIGHLIGHT" and state.get("pipeline_version") != "5.0":
        from toolrecap_v4.highlight.models import build_highlight_project
        source = state.get("highlight_json") or {}
        project = build_highlight_project(
            project_id=project_id, prompt=state.get("prompt", ""),
            dependency_revision=source.get("dependency_revision", ""),
            candidates=plan.get("outputs", []), sources=state["sources"],
        )
        if not project.outputs:
            raise ValueError("Keep at least one storyline in the reviewed plan")
        state["highlight_json"] = project.to_dict()
        checkpoint_path = persistence._checkpoint_path(project_id, "highlight_final")
        atomic_write_json(checkpoint_path, {
            "status": "COMPLETE", "dependency_revision": project.dependency_revision,
            "candidates": [item.to_dict() for item in project.outputs],
        })
    else:
        from toolrecap_v4.validator import validate_project
        durations = {item["source_file"]: item["duration_ms"] for item in state["sources"]}
        validate_project(plan, source_durations=durations)
        state["final_json"] = plan
        persistence.save_final_json(project_id, plan)
    state["review_plan_hash"] = plan_digest(state)
    state.pop("review_approved_hash", None)
    persistence.save_project(state)
    return state


def review_text(state: dict[str, Any]) -> str:
    value = plan_data(state)
    if not isinstance(value, dict):
        return "No completed editorial plan is available yet."
    mode = state.get("project_mode", "RECAP")
    lines = [f"{mode} — {state.get('project_name', state.get('project_id', 'Project'))}",
             f"Plan: {plan_digest(state)}", ""]
    if state.get("visual_coverage_note"):
        lines.extend([f"Visual coverage: {state['visual_coverage_note']}", ""])
    source_signals = {item.get("source_file"): item.get("structure_signals", [])
                      for item in state.get("sources", []) if isinstance(item, dict)}
    seen_ranges: list[tuple[str, int, int, str]] = []
    for index, output in enumerate(value.get("outputs", []), 1):
        lines.append(f"{index}. {output.get('title', 'Untitled')}")
        is_legacy_highlight = mode == "HIGHLIGHT" and "clips" in output
        clips = output.get("clips", []) if is_legacy_highlight else output.get("segments", [])
        if is_legacy_highlight:
            lines.append(f"   Story: {output.get('storyline', '')}")
        duration = sum(max(0, item.get("end_ms", 0) - item.get("start_ms", 0)) for item in clips)
        lines.append(f"   Selected footage: {_clock(duration)} ({len(clips)} clips)")
        for item in clips:
            source = item.get("source_file", "unknown source")
            episode = item.get("episode_id", "")
            lines.append(f"   • {episode + ' — ' if episode else ''}{source}: "
                         f"{_clock(item.get('start_ms', 0))} → {_clock(item.get('end_ms', 0))}")
            start, end = item.get("start_ms", 0), item.get("end_ms", 0)
            for signal in source_signals.get(source, []):
                if signal["start_ms"] < end and signal["end_ms"] > start:
                    lines.append(f"     REVIEW: overlaps {signal['kind']} marker; confirm actual scene boundaries.")
            for previous_source, previous_start, previous_end, previous_title in seen_ranges:
                if source == previous_source and start < previous_end and end > previous_start:
                    lines.append(f"     REVIEW: footage overlaps output '{previous_title}'.")
            seen_ranges.append((source, start, end, output.get("title", "")))
            if item.get("narration"):
                lines.append(f"     Narration: {item['narration']}")
        lines.append("")
    lines.append("Review the source moments and story grouping before approval. "
                 "The plan text alone cannot verify exact visual cut frames.")
    return "\n".join(lines)
