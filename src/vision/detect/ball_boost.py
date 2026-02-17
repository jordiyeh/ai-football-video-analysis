"""Ball detection boosting module.

Provides temporal filtering, candidate tracking, and soft-NMS to improve
ball detection rates while minimizing false positives.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.events.kalman_filter import BallKalmanFilter
    from src.vision.detect.yolo import Detection


class CandidateState(Enum):
    """State machine for ball candidate tracking."""

    NEW = "new"
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


@dataclass
class BallCandidate:
    """A ball detection candidate being tracked."""

    candidate_id: int
    state: CandidateState = CandidateState.NEW
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    last_position: tuple[float, float] = (0.0, 0.0)
    last_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_confidence: float = 0.0
    velocity: tuple[float, float] = (0.0, 0.0)

    # Optional Kalman filter for prediction
    kalman: "BallKalmanFilter | None" = None


def compute_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union between two bounding boxes.

    Args:
        box1: (x1, y1, x2, y2) first box
        box2: (x1, y1, x2, y2) second box

    Returns:
        IoU value in [0, 1]
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def soft_nms(
    detections: list["Detection"],
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
) -> list["Detection"]:
    """
    Apply Soft Non-Maximum Suppression to detections.

    Instead of hard elimination, reduces confidence of overlapping detections
    based on IoU. Higher IoU = lower confidence.

    Args:
        detections: List of Detection objects
        iou_threshold: IoU threshold for applying soft penalty
        sigma: Gaussian parameter for confidence decay
        score_threshold: Minimum confidence to keep detection

    Returns:
        Filtered list of detections with adjusted confidences
    """
    if len(detections) == 0:
        return []

    # Sort by confidence (descending)
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

    # Track which detections to keep and their adjusted confidences
    keep = []
    confidences = [d.confidence for d in sorted_dets]

    while sorted_dets:
        # Take highest confidence detection
        best = sorted_dets.pop(0)
        best_conf = confidences.pop(0)

        if best_conf < score_threshold:
            continue

        # Create new detection with potentially adjusted confidence
        from src.vision.detect.yolo import Detection

        adjusted_det = Detection(
            object_type=best.object_type,
            bbox=best.bbox,
            confidence=best_conf,
            class_id=best.class_id,
        )
        keep.append(adjusted_det)

        # Apply soft penalty to remaining detections
        new_sorted = []
        new_confs = []
        for det, conf in zip(sorted_dets, confidences):
            iou = compute_iou(best.bbox, det.bbox)
            if iou > iou_threshold:
                # Gaussian decay: conf *= exp(-iou^2 / sigma)
                weight = np.exp(-(iou * iou) / sigma)
                conf = conf * weight

            if conf >= score_threshold:
                new_sorted.append(det)
                new_confs.append(conf)

        sorted_dets = new_sorted
        confidences = new_confs

    return keep


class BallTemporalFilter:
    """
    Sliding window temporal consistency filter for ball detections.

    Tracks ball detections over N frames and requires min_confirmations
    within the window plus max_displacement constraint.
    """

    def __init__(
        self,
        window_size: int = 5,
        min_confirmations: int = 2,
        max_displacement: float = 100.0,
    ):
        """
        Initialize temporal filter.

        Args:
            window_size: Number of frames to track
            min_confirmations: Minimum detections in window to confirm
            max_displacement: Maximum pixel displacement between frames
        """
        self.window_size = window_size
        self.min_confirmations = min_confirmations
        self.max_displacement = max_displacement

        # Sliding window: deque of (frame_idx, list of ball detections)
        self.history: deque[tuple[int, list["Detection"]]] = deque(maxlen=window_size)

    def reset(self) -> None:
        """Reset the temporal filter state."""
        self.history.clear()

    def filter(
        self,
        detections: list["Detection"],
        frame_idx: int,
    ) -> list["Detection"]:
        """
        Filter ball detections using temporal consistency.

        Args:
            detections: Ball detections for current frame
            frame_idx: Current frame index

        Returns:
            Filtered detections that pass temporal consistency check
        """
        # Add current detections to history
        self.history.append((frame_idx, detections))

        if len(detections) == 0:
            return []

        # Not enough history yet - be lenient
        if len(self.history) < self.min_confirmations:
            return detections

        # For each detection, check if it's temporally consistent
        consistent = []
        for det in detections:
            if self._is_consistent(det, frame_idx):
                consistent.append(det)

        return consistent

    def _is_consistent(self, detection: "Detection", current_frame: int) -> bool:
        """
        Check if a detection is temporally consistent with history.

        A detection is consistent if there are min_confirmations detections
        in the history that are within max_displacement pixels.

        Args:
            detection: Detection to check
            current_frame: Current frame index

        Returns:
            True if detection is consistent
        """
        cx, cy = detection.center
        confirmations = 0

        for hist_frame, hist_dets in self.history:
            if hist_frame == current_frame:
                continue

            # Check if any detection in this frame is close enough
            for hist_det in hist_dets:
                hx, hy = hist_det.center
                distance = np.sqrt((cx - hx) ** 2 + (cy - hy) ** 2)

                # Scale displacement by frame gap (allow larger displacement for bigger gaps)
                frame_gap = abs(current_frame - hist_frame)
                max_allowed = self.max_displacement * frame_gap

                if distance <= max_allowed:
                    confirmations += 1
                    break  # One confirmation per frame is enough

        return confirmations >= self.min_confirmations - 1  # -1 because current frame counts


class BallCandidateTracker:
    """
    Lightweight Kalman-based ball candidate tracker.

    Implements soft-tracking before committing to a detection.
    States: NEW -> TENTATIVE -> CONFIRMED -> LOST

    Uses BallKalmanFilter from src/events/kalman_filter.py for prediction.
    """

    def __init__(
        self,
        min_hits: int = 3,
        max_age: int = 5,
        iou_threshold: float = 0.3,
        use_kalman: bool = True,
    ):
        """
        Initialize candidate tracker.

        Args:
            min_hits: Consecutive hits required for CONFIRMED state
            max_age: Max frames without detection before LOST
            iou_threshold: IoU threshold for matching detections to candidates
            use_kalman: Whether to use Kalman filter for prediction
        """
        self.min_hits = min_hits
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.use_kalman = use_kalman

        self.candidates: list[BallCandidate] = []
        self.next_id = 0

    def reset(self) -> None:
        """Reset tracker state."""
        self.candidates.clear()
        self.next_id = 0

    def update(
        self,
        detections: list["Detection"],
        frame_idx: int,
    ) -> list["Detection"]:
        """
        Update tracker with new detections.

        Args:
            detections: Ball detections for current frame
            frame_idx: Current frame index

        Returns:
            Confirmed ball detections only
        """
        # Predict positions for existing candidates
        for candidate in self.candidates:
            candidate.age += 1
            candidate.time_since_update += 1

            if candidate.kalman is not None:
                pred_pos = candidate.kalman.predict(dt=1.0)
                candidate.last_position = pred_pos

        # Match detections to candidates
        matched_candidates: set[int] = set()
        matched_detections: set[int] = set()

        if len(detections) > 0 and len(self.candidates) > 0:
            # Build cost matrix
            cost_matrix = np.zeros((len(detections), len(self.candidates)))
            for i, det in enumerate(detections):
                for j, cand in enumerate(self.candidates):
                    iou = compute_iou(det.bbox, cand.last_bbox)
                    # Use negative IoU as cost (we want to maximize IoU)
                    cost_matrix[i, j] = 1.0 - iou

            # Greedy matching (simple approach, could use Hungarian algorithm)
            for _ in range(min(len(detections), len(self.candidates))):
                if cost_matrix.size == 0:
                    break

                # Find best match
                min_cost = np.min(cost_matrix)
                if min_cost > (1.0 - self.iou_threshold):
                    break  # No more good matches

                i, j = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)

                # Mark as matched
                matched_detections.add(i)
                matched_candidates.add(j)

                # Update candidate
                det = detections[i]
                cand = self.candidates[j]
                self._update_candidate(cand, det)

                # Remove from consideration
                cost_matrix[i, :] = np.inf
                cost_matrix[:, j] = np.inf

        # Create new candidates for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_detections:
                self._create_candidate(det)

        # Update states and remove lost candidates
        active_candidates = []
        for j, cand in enumerate(self.candidates):
            if j not in matched_candidates:
                # Not matched - age out
                if cand.time_since_update > self.max_age:
                    cand.state = CandidateState.LOST
                    continue  # Don't keep LOST candidates

            active_candidates.append(cand)

        self.candidates = active_candidates

        # Return only confirmed detections
        from src.vision.detect.yolo import Detection

        confirmed = []
        for cand in self.candidates:
            if cand.state == CandidateState.CONFIRMED:
                det = Detection(
                    object_type="ball",
                    bbox=cand.last_bbox,
                    confidence=cand.last_confidence,
                    class_id=32,  # COCO sports ball
                )
                confirmed.append(det)

        return confirmed

    def _create_candidate(self, detection: "Detection") -> BallCandidate:
        """Create a new candidate from a detection."""
        candidate = BallCandidate(
            candidate_id=self.next_id,
            state=CandidateState.NEW,
            hits=1,
            age=0,
            time_since_update=0,
            last_position=detection.center,
            last_bbox=detection.bbox,
            last_confidence=detection.confidence,
        )

        if self.use_kalman:
            from src.events.kalman_filter import BallKalmanFilter

            candidate.kalman = BallKalmanFilter()
            candidate.kalman.initialize(position=detection.center)

        self.candidates.append(candidate)
        self.next_id += 1

        return candidate

    def _update_candidate(self, candidate: BallCandidate, detection: "Detection") -> None:
        """Update a candidate with a new detection."""
        # Update Kalman filter
        if candidate.kalman is not None:
            candidate.kalman.update(detection.center)

        # Compute velocity
        old_x, old_y = candidate.last_position
        new_x, new_y = detection.center
        candidate.velocity = (new_x - old_x, new_y - old_y)

        # Update state
        candidate.last_position = detection.center
        candidate.last_bbox = detection.bbox
        candidate.last_confidence = detection.confidence
        candidate.hits += 1
        candidate.time_since_update = 0

        # State transitions
        if candidate.state == CandidateState.NEW:
            candidate.state = CandidateState.TENTATIVE

        if candidate.state == CandidateState.TENTATIVE and candidate.hits >= self.min_hits:
            candidate.state = CandidateState.CONFIRMED


def merge_multiscale_detections(
    all_detections: list[list["Detection"]],
    iou_threshold: float = 0.5,
) -> list["Detection"]:
    """
    Merge detections from multiple scales using soft-NMS.

    Args:
        all_detections: List of detection lists (one per scale)
        iou_threshold: IoU threshold for soft-NMS

    Returns:
        Merged list of detections
    """
    # Flatten all detections
    flat = []
    for scale_dets in all_detections:
        flat.extend(scale_dets)

    if len(flat) == 0:
        return []

    # Apply soft-NMS
    return soft_nms(flat, iou_threshold=iou_threshold)
