"""Object detection module for soccer video analysis."""

from importlib import import_module
from typing import Any

__all__ = [
    "ObjectDetector",
    "Detection",
    "YOLODetector",
    "BallSpecialistDetector",
    "DetectorEnsemble",
]

_SYMBOL_TO_MODULE = {
    "ObjectDetector": "src.vision.detect.base",
    "Detection": "src.vision.detect.yolo",
    "YOLODetector": "src.vision.detect.yolo",
    "BallSpecialistDetector": "src.vision.detect.ball_specialist",
    "DetectorEnsemble": "src.vision.detect.ensemble",
}


def __getattr__(name: str) -> Any:
    """Lazily import optional-heavy detector modules."""
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
