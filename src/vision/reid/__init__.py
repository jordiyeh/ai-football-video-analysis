"""ReID (Re-Identification) module for player embedding extraction."""

from importlib import import_module
from typing import Any

__all__ = ["ReIDExtractor", "OSNetExtractor", "CropExtractor", "PlayerCrop"]

_SYMBOL_TO_MODULE = {
    "ReIDExtractor": "src.vision.reid.base",
    "OSNetExtractor": "src.vision.reid.osnet",
    "CropExtractor": "src.vision.reid.crop",
    "PlayerCrop": "src.vision.reid.crop",
}


def __getattr__(name: str) -> Any:
    """Lazily import modules that may require torch/OpenCV."""
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
