"""Shared editorial source rules and conservative transcript structure signals."""
import re

EDITORIAL_VERSION = "source-structure-v1"
SOURCE_RULES = """One output tells one evidenced story. Multiple related clips from one or several supplied episodes are allowed. Separate unrelated stories.
Previously-on recaps replay earlier events; never count a replay as a new occurrence. Prefer the original scene when supplied. Next-time previews do not prove events have happened. Exclude branding intros and credits from story clips, but preserve meaningful cold opens, flashbacks and post-credit scenes. Never trim fixed amounts from episode heads/tails.
Mark uncertain chronology, dreams, imagined events, repeated footage and uncertain speaker identities rather than inventing facts. Source structure signals are hints, not confirmed exclusion boundaries. Use only supplied evidence; sampled frames do not constitute watching the complete episode."""


def transcript_structure(cues):
    """Detect explicit recap/preview markers; do not guess the end of their range."""
    signals = []
    for cue in cues:
        text = cue.text.strip()
        kind = None
        if re.match(r"(?i)^\W*previously\s+on\b", text):
            kind = "possible_previous_episode_recap"
        elif re.match(r"(?i)^\W*(?:next\s+time\s+on|on\s+the\s+next\s+episode)\b", text):
            kind = "possible_next_episode_preview"
        if kind:
            signals.append({"kind": kind, "start_ms": cue.start_ms, "end_ms": cue.end_ms,
                            "text": text, "boundary_confirmed": False})
    return signals
