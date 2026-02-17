"""Abstract base class for object detectors."""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


class ObjectDetector(ABC):
    """Abstract base class for object detectors.

    All detectors should inherit from this class and implement
    the required abstract methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the detector name for logging."""
        pass

    @property
    @abstractmethod
    def supported_types(self) -> list[Literal["player", "ball"]]:
        """Return list of object types this detector can detect."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list:
        """Detect objects in a single frame.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            List of Detection objects
        """
        pass

    @abstractmethod
    def detect_batch(self, frames: list[np.ndarray]) -> list[list]:
        """Detect objects in multiple frames (batch processing).

        Args:
            frames: List of input frames

        Returns:
            List of detection lists (one per frame)
        """
        pass
