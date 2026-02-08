"""Multi-object tracking."""

from importlib import import_module
from typing import Any

__all__ = ["ByteTracker", "Track"]

_SYMBOL_TO_MODULE = {
    "ByteTracker": "src.vision.track.bytetrack",
    "Track": "src.vision.track.bytetrack",
}


def __getattr__(name: str) -> Any:
    """Lazily import tracking modules that rely on optional deps."""
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
