"""Field detection and goal region modules."""

from importlib import import_module
from typing import Any

__all__ = [
    "GoalRegion",
    "GoalRegionProvider",
    "HeuristicGoalRegionProvider",
    "GoalRegionDetector",
    "GoalRegionTracker",
    "estimate_frame_viewports",
    "normalize_tracks_to_field_view",
]

_SYMBOL_TO_MODULE = {
    "GoalRegion": "src.vision.field.goal_detector",
    "GoalRegionProvider": "src.vision.field.goal_detector",
    "HeuristicGoalRegionProvider": "src.vision.field.goal_detector",
    "GoalRegionDetector": "src.vision.field.goal_detector",
    "GoalRegionTracker": "src.vision.field.goal_detector",
    "estimate_frame_viewports": "src.vision.field.normalization",
    "normalize_tracks_to_field_view": "src.vision.field.normalization",
}


def __getattr__(name: str) -> Any:
    """Lazily import field modules that depend on OpenCV."""
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
