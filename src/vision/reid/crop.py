"""Player crop extraction and quality filtering for ReID."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PlayerCrop:
    """A player crop with metadata."""

    image: np.ndarray  # RGB image (H, W, 3)
    track_id: int
    frame_idx: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    team_id: int | None = None

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]


class CropExtractor:
    """
    Extract and filter player crops from video frames.

    Applies quality filtering to ensure crops are suitable for ReID:
    - Minimum size thresholds
    - Aspect ratio filtering
    - Optional confidence filtering
    """

    def __init__(
        self,
        min_height: int = 50,
        min_width: int = 25,
        min_aspect_ratio: float = 0.3,  # width/height
        max_aspect_ratio: float = 1.5,  # width/height
        min_confidence: float = 0.5,
        padding_ratio: float = 0.1,  # Add padding around bbox
    ):
        """
        Initialize crop extractor.

        Args:
            min_height: Minimum crop height in pixels.
            min_width: Minimum crop width in pixels.
            min_aspect_ratio: Minimum width/height ratio.
            max_aspect_ratio: Maximum width/height ratio.
            min_confidence: Minimum detection confidence.
            padding_ratio: Ratio of padding to add around bbox.
        """
        self.min_height = min_height
        self.min_width = min_width
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_confidence = min_confidence
        self.padding_ratio = padding_ratio

    def extract_crop(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        track_id: int,
        frame_idx: int,
        confidence: float = 1.0,
        team_id: int | None = None,
    ) -> PlayerCrop | None:
        """
        Extract a crop from a frame given a bounding box.

        Args:
            frame: RGB image (H, W, 3).
            bbox: Bounding box (x1, y1, x2, y2).
            track_id: Track identifier.
            frame_idx: Frame index.
            confidence: Detection confidence.
            team_id: Optional team identifier.

        Returns:
            PlayerCrop if valid, None if filtered out.
        """
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame.shape[:2]

        # Validate bbox
        if x1 >= x2 or y1 >= y2:
            return None

        # Calculate dimensions
        width = x2 - x1
        height = y2 - y1

        # Check minimum size
        if width < self.min_width or height < self.min_height:
            return None

        # Check aspect ratio
        aspect_ratio = width / height
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return None

        # Check confidence
        if confidence < self.min_confidence:
            return None

        # Add padding
        pad_w = width * self.padding_ratio
        pad_h = height * self.padding_ratio

        x1_padded = max(0, int(x1 - pad_w))
        y1_padded = max(0, int(y1 - pad_h))
        x2_padded = min(frame_w, int(x2 + pad_w))
        y2_padded = min(frame_h, int(y2 + pad_h))

        # Extract crop
        crop = frame[y1_padded:y2_padded, x1_padded:x2_padded].copy()

        if crop.size == 0:
            return None

        return PlayerCrop(
            image=crop,
            track_id=track_id,
            frame_idx=frame_idx,
            bbox=bbox,
            confidence=confidence,
            team_id=team_id,
        )

    def extract_crops_from_frame(
        self,
        frame: np.ndarray,
        tracks: list[dict],
        frame_idx: int,
    ) -> list[PlayerCrop]:
        """
        Extract all valid crops from a frame.

        Args:
            frame: RGB image (H, W, 3).
            tracks: List of track dictionaries with 'bbox', 'track_id', etc.
            frame_idx: Frame index.

        Returns:
            List of valid PlayerCrop objects.
        """
        crops = []

        for track in tracks:
            # Skip non-player tracks
            if track.get("object_type") != "player":
                continue

            bbox = track.get("bbox")
            if bbox is None:
                continue

            # Handle both list and tuple formats
            if isinstance(bbox, list):
                bbox = tuple(bbox)

            crop = self.extract_crop(
                frame=frame,
                bbox=bbox,
                track_id=track.get("track_id", -1),
                frame_idx=frame_idx,
                confidence=track.get("confidence", 1.0),
                team_id=track.get("team_id"),
            )

            if crop is not None:
                crops.append(crop)

        return crops

    def sample_crops_for_track(
        self,
        all_crops: list[PlayerCrop],
        track_id: int,
        n_samples: int = 10,
        strategy: str = "uniform",
    ) -> list[PlayerCrop]:
        """
        Sample a subset of crops for a specific track.

        Args:
            all_crops: All crops collected.
            track_id: Track to sample from.
            n_samples: Number of samples to take.
            strategy: Sampling strategy ('uniform', 'highest_conf', 'random').

        Returns:
            List of sampled crops.
        """
        track_crops = [c for c in all_crops if c.track_id == track_id]

        if len(track_crops) <= n_samples:
            return track_crops

        if strategy == "uniform":
            # Sample uniformly across frame indices
            track_crops_sorted = sorted(track_crops, key=lambda c: c.frame_idx)
            indices = np.linspace(0, len(track_crops_sorted) - 1, n_samples, dtype=int)
            return [track_crops_sorted[i] for i in indices]

        elif strategy == "highest_conf":
            # Take highest confidence crops
            track_crops_sorted = sorted(
                track_crops, key=lambda c: c.confidence, reverse=True
            )
            return track_crops_sorted[:n_samples]

        elif strategy == "random":
            # Random sampling
            indices = np.random.choice(len(track_crops), n_samples, replace=False)
            return [track_crops[i] for i in indices]

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
