"""Tests for lazy package exports that avoid optional dependency crashes."""

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    ("package_name", "heavy_submodule"),
    [
        ("src.events", "src.events.detection"),
        ("src.identity", "src.identity.profile_seed"),
        ("src.vision.detect", "src.vision.detect.yolo"),
        ("src.vision.field", "src.vision.field.goal_detector"),
        ("src.vision.reid", "src.vision.reid.osnet"),
        ("src.vision.team", "src.vision.team.colors"),
        ("src.vision.track", "src.vision.track.bytetrack"),
    ],
)
def test_package_import_does_not_eagerly_import_optional_submodules(
    package_name: str,
    heavy_submodule: str,
) -> None:
    """Importing package namespaces should not import optional-heavy modules."""
    sys.modules.pop(package_name, None)
    sys.modules.pop(heavy_submodule, None)

    importlib.import_module(package_name)

    assert heavy_submodule not in sys.modules
