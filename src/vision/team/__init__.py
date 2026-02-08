"""Team identification and jersey color analysis."""

from importlib import import_module
from typing import Any

__all__ = [
    "extract_jersey_color",
    "extract_dominant_color_kmeans",
    "bgr_to_hsv",
    "color_distance",
    "is_similar_color",
    "TeamAssigner",
    "collect_track_colors",
]

_SYMBOL_TO_MODULE = {
    "extract_jersey_color": "src.vision.team.colors",
    "extract_dominant_color_kmeans": "src.vision.team.colors",
    "bgr_to_hsv": "src.vision.team.colors",
    "color_distance": "src.vision.team.colors",
    "is_similar_color": "src.vision.team.colors",
    "TeamAssigner": "src.vision.team.clustering",
    "collect_track_colors": "src.vision.team.clustering",
}


def __getattr__(name: str) -> Any:
    """Lazily import modules that depend on OpenCV/sklearn."""
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
