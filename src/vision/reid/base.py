"""Base class for ReID (Re-Identification) extractors."""

from abc import ABC, abstractmethod

import numpy as np


class ReIDExtractor(ABC):
    """Abstract base class for person re-identification embedding extractors."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of extracted embeddings."""
        pass

    @abstractmethod
    def extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings from a batch of person crops.

        Args:
            crops: List of RGB images (H, W, 3) as numpy arrays.

        Returns:
            Embeddings array of shape (N, embedding_dim).
        """
        pass

    def extract_single(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single person crop.

        Args:
            crop: RGB image (H, W, 3) as numpy array.

        Returns:
            Embedding vector of shape (embedding_dim,).
        """
        embeddings = self.extract([crop])
        return embeddings[0]
