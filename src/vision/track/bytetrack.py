"""ByteTrack multi-object tracking implementation."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.vision.track.kalman import BBoxKalmanFilter, bbox_to_measurement, measurement_to_bbox


def iou(bbox1: tuple[float, float, float, float], bbox2: tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.

    Args:
        bbox1: (x1, y1, x2, y2)
        bbox2: (x1, y1, x2, y2)

    Returns:
        IoU score [0, 1]
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


@dataclass
class Track:
    """Single object track."""

    track_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    object_type: Literal["player", "ball"]

    # Tracking state
    kf: BBoxKalmanFilter
    age: int = 0  # Frames since track started
    hits: int = 0  # Total number of detection matches
    time_since_update: int = 0  # Frames since last detection match
    state: Literal["tentative", "confirmed", "deleted"] = "tentative"

    def predict(self) -> None:
        """Predict next state using Kalman filter."""
        predicted = self.kf.predict()
        self.bbox = measurement_to_bbox(predicted)
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: tuple[float, float, float, float], confidence: float) -> None:
        """Update track with new detection."""
        measurement = bbox_to_measurement(bbox)
        self.kf.update(measurement)

        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0

        # Promote tentative tracks to confirmed after enough hits
        if self.state == "tentative" and self.hits >= 3:
            self.state = "confirmed"

    def mark_missed(self) -> None:
        """Mark track as missed (no detection match)."""
        if self.state == "tentative":
            self.state = "deleted"


class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Key idea: Use both high and low confidence detections for tracking.
    - High confidence detections are matched first
    - Low confidence detections are used to recover tracks
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_hits: int = 3,
    ):
        """
        Initialize ByteTracker.

        Args:
            track_thresh: Minimum confidence for track initialization
            track_buffer: Frames to keep lost track before deletion
            match_thresh: IoU threshold for matching
            min_hits: Minimum hits before track is confirmed
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_hits = min_hits

        self.tracks: list[Track] = []
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections: list[dict]) -> list[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detection dicts with keys:
                - bbox: (x1, y1, x2, y2)
                - confidence: float
                - object_type: "player" or "ball"

        Returns:
            List of confirmed tracks
        """
        self.frame_count += 1

        # Predict all track positions
        for track in self.tracks:
            track.predict()

        # Separate high and low confidence detections
        high_dets = [d for d in detections if d["confidence"] >= self.track_thresh]
        low_dets = [d for d in detections if d["confidence"] < self.track_thresh]

        # First association: confirmed tracks with high confidence detections
        confirmed_tracks = [t for t in self.tracks if t.state == "confirmed"]
        unmatched_tracks, unmatched_dets = self._match_tracks_detections(
            confirmed_tracks, high_dets
        )

        # Second association: unmatched tracks with low confidence detections
        if len(low_dets) > 0 and len(unmatched_tracks) > 0:
            unmatched_tracks, _ = self._match_tracks_detections(
                unmatched_tracks, low_dets
            )

        # Third association: tentative tracks with unmatched high confidence detections
        tentative_tracks = [t for t in self.tracks if t.state == "tentative"]
        if len(unmatched_dets) > 0 and len(tentative_tracks) > 0:
            _, unmatched_dets = self._match_tracks_detections(
                tentative_tracks, unmatched_dets
            )

        # Initialize new tracks from remaining unmatched detections
        for det in unmatched_dets:
            if det["confidence"] >= self.track_thresh:
                self._initiate_track(det)

        # Mark unmatched tracks
        for track in unmatched_tracks:
            track.mark_missed()

        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks
            if t.state != "deleted" and t.time_since_update < self.track_buffer
        ]

        # Return confirmed tracks only
        return [t for t in self.tracks if t.state == "confirmed"]

    def _match_tracks_detections(
        self, tracks: list[Track], detections: list[dict]
    ) -> tuple[list[Track], list[dict]]:
        """
        Match tracks to detections using IoU.

        Args:
            tracks: List of tracks
            detections: List of detections

        Returns:
            (unmatched_tracks, unmatched_detections)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return tracks, detections

        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # Only match same object type
                if track.object_type == det["object_type"]:
                    iou_matrix[i, j] = iou(track.bbox, det["bbox"])

        # Greedy matching (simple but effective)
        matched_tracks = set()
        matched_dets = set()

        # Sort matches by IoU (highest first)
        matches = []
        for i in range(len(tracks)):
            for j in range(len(detections)):
                if iou_matrix[i, j] >= self.match_thresh:
                    matches.append((i, j, iou_matrix[i, j]))

        matches.sort(key=lambda x: x[2], reverse=True)

        # Assign matches
        for track_idx, det_idx, _ in matches:
            if track_idx not in matched_tracks and det_idx not in matched_dets:
                tracks[track_idx].update(
                    detections[det_idx]["bbox"],
                    detections[det_idx]["confidence"]
                )
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)

        # Collect unmatched
        unmatched_tracks = [t for i, t in enumerate(tracks) if i not in matched_tracks]
        unmatched_dets = [d for i, d in enumerate(detections) if i not in matched_dets]

        return unmatched_tracks, unmatched_dets

    def _initiate_track(self, detection: dict) -> None:
        """Initialize new track from detection."""
        kf = BBoxKalmanFilter()
        measurement = bbox_to_measurement(detection["bbox"])
        kf.initiate(measurement)

        track = Track(
            track_id=self.next_id,
            bbox=detection["bbox"],
            confidence=detection["confidence"],
            object_type=detection["object_type"],
            kf=kf,
            hits=1,
        )

        self.tracks.append(track)
        self.next_id += 1
