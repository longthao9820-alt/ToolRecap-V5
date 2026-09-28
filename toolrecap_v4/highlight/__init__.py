"""Original-audio Highlight Mode for ToolRecap V4."""

from .models import (
    HIGHLIGHT_SCHEMA_VERSION,
    HighlightClip,
    HighlightOutput,
    HighlightProject,
    HighlightSubtitleCue,
    build_highlight_project,
    validate_highlight_project,
)
from .renderer import HighlightRenderResult, render_highlight
from .service import HighlightWorkflow

__all__ = [
    "HIGHLIGHT_SCHEMA_VERSION", "HighlightClip", "HighlightOutput", "HighlightProject",
    "HighlightSubtitleCue", "build_highlight_project", "validate_highlight_project",
    "HighlightRenderResult", "render_highlight", "HighlightWorkflow",
]
