"""Detector ensemble with Weighted Box Fusion (WBF) for combining multiple detectors."""

from typing import Literal

import numpy as np

from src.vision.detect.base import ObjectDetector
from src.vision.detect.yolo import Detection


def compute_iou(box1: tuple[float, ...], box2: tuple[float, ...]) -> float:
    """Compute Intersection over Union between two boxes.

    Args:
        box1: First box (x1, y1, x2, y2)
        box2: Second box (x1, y1, x2, y2)

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


class DetectorEnsemble(ObjectDetector):
    """Ensemble of multiple detectors with Weighted Box Fusion.

    Combines detections from multiple detectors using WBF algorithm,
    which averages overlapping box coordinates weighted by confidence.
    When multiple detectors agree on a detection, confidence is boosted.
    """

    @property
    def name(self) -> str:
        """Return the detector name for logging."""
        return "ensemble"

    @property
    def supported_types(self) -> list[Literal["player", "ball"]]:
        """Return list of object types this ensemble can detect."""
        # Union of all detector types
        types = set()
        for detector in self.detectors.values():
            types.update(detector.supported_types)
        return list(types)

    def __init__(
        self,
        detectors: dict[str, ObjectDetector],
        weights: dict[str, float] | None = None,
        iou_threshold: float = 0.5,
        skip_box_threshold: float = 0.01,
        fusion_type: Literal["wbf", "nms", "soft_nms"] = "wbf",
    ):
        """Initialize detector ensemble.

        Args:
            detectors: Dictionary of detector name -> detector instance
            weights: Dictionary of detector name -> weight (default: 1.0 for all)
            iou_threshold: IoU threshold for clustering overlapping boxes
            skip_box_threshold: Skip boxes with confidence below this
            fusion_type: Fusion algorithm ("wbf", "nms", "soft_nms")
        """
        self.detectors = detectors
        self.weights = weights or {name: 1.0 for name in detectors}
        self.iou_threshold = iou_threshold
        self.skip_box_threshold = skip_box_threshold
        self.fusion_type = fusion_type

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.normalized_weights = {
            name: w / total_weight for name, w in self.weights.items()
        }

        print(f"DetectorEnsemble initialized with {len(detectors)} detectors:")
        for name, detector in detectors.items():
            print(f"  - {name}: types={detector.supported_types}, weight={self.weights.get(name, 1.0)}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects using all detectors and fuse results.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            List of fused Detection objects
        """
        # Collect detections from all detectors
        all_detections: list[tuple[Detection, str, float]] = []

        for detector_name, detector in self.detectors.items():
            weight = self.weights.get(detector_name, 1.0)
            detections = detector.detect(frame)

            for det in detections:
                if det.confidence >= self.skip_box_threshold:
                    all_detections.append((det, detector_name, weight))

        # Separate by object type (players and balls have very different sizes)
        player_detections = [
            (det, name, weight)
            for det, name, weight in all_detections
            if det.object_type == "player"
        ]
        ball_detections = [
            (det, name, weight)
            for det, name, weight in all_detections
            if det.object_type == "ball"
        ]

        # Fuse each type separately
        fused_players = self._fuse_detections(player_detections, "player")
        fused_balls = self._fuse_detections(ball_detections, "ball")

        return fused_players + fused_balls

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Detect objects in multiple frames using all detectors.

        Args:
            frames: List of input frames

        Returns:
            List of detection lists (one per frame)
        """
        if len(frames) == 0:
            return []

        # Collect batch detections from all detectors
        # Dict: frame_idx -> list of (detection, detector_name, weight)
        all_frame_detections: dict[int, list[tuple[Detection, str, float]]] = {
            i: [] for i in range(len(frames))
        }

        for detector_name, detector in self.detectors.items():
            weight = self.weights.get(detector_name, 1.0)
            batch_detections = detector.detect_batch(frames)

            for frame_idx, frame_dets in enumerate(batch_detections):
                for det in frame_dets:
                    if det.confidence >= self.skip_box_threshold:
                        all_frame_detections[frame_idx].append(
                            (det, detector_name, weight)
                        )

        # Fuse detections for each frame
        results = []
        for frame_idx in range(len(frames)):
            detections = all_frame_detections[frame_idx]

            # Separate by object type
            player_detections = [
                (det, name, weight)
                for det, name, weight in detections
                if det.object_type == "player"
            ]
            ball_detections = [
                (det, name, weight)
                for det, name, weight in detections
                if det.object_type == "ball"
            ]

            # Fuse each type
            fused_players = self._fuse_detections(player_detections, "player")
            fused_balls = self._fuse_detections(ball_detections, "ball")

            results.append(fused_players + fused_balls)

        return results

    def _fuse_detections(
        self,
        detections: list[tuple[Detection, str, float]],
        object_type: str,
    ) -> list[Detection]:
        """Fuse overlapping detections using the selected algorithm.

        Args:
            detections: List of (detection, detector_name, weight) tuples
            object_type: Type of objects being fused

        Returns:
            List of fused Detection objects
        """
        if len(detections) == 0:
            return []

        if self.fusion_type == "wbf":
            return self._weighted_box_fusion(detections, object_type)
        elif self.fusion_type == "nms":
            return self._non_max_suppression(detections)
        elif self.fusion_type == "soft_nms":
            return self._soft_nms(detections)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

    def _weighted_box_fusion(
        self,
        detections: list[tuple[Detection, str, float]],
        object_type: str,
    ) -> list[Detection]:
        """Weighted Box Fusion algorithm.

        Clusters overlapping boxes and averages their coordinates
        weighted by confidence * detector_weight.

        Args:
            detections: List of (detection, detector_name, weight) tuples
            object_type: Type of objects being fused

        Returns:
            List of fused Detection objects
        """
        if len(detections) == 0:
            return []

        # Sort by weighted confidence (descending)
        sorted_dets = sorted(
            detections,
            key=lambda x: x[0].confidence * x[2],
            reverse=True,
        )

        # Cluster overlapping boxes
        clusters: list[list[tuple[Detection, str, float]]] = []
        used = [False] * len(sorted_dets)

        for i, (det_i, name_i, weight_i) in enumerate(sorted_dets):
            if used[i]:
                continue

            # Start new cluster with this detection
            cluster = [(det_i, name_i, weight_i)]
            used[i] = True

            # Find all overlapping boxes
            for j, (det_j, name_j, weight_j) in enumerate(sorted_dets):
                if used[j]:
                    continue

                iou = compute_iou(det_i.bbox, det_j.bbox)
                if iou >= self.iou_threshold:
                    cluster.append((det_j, name_j, weight_j))
                    used[j] = True

            clusters.append(cluster)

        # Fuse each cluster
        fused = []
        for cluster in clusters:
            fused_det = self._fuse_cluster(cluster, object_type)
            fused.append(fused_det)

        return fused

    def _fuse_cluster(
        self,
        cluster: list[tuple[Detection, str, float]],
        object_type: str,
    ) -> Detection:
        """Fuse a cluster of overlapping detections.

        Computes weighted average of box coordinates and boosts
        confidence when multiple detectors agree.

        Args:
            cluster: List of (detection, detector_name, weight) tuples
            object_type: Type of objects being fused

        Returns:
            Single fused Detection
        """
        if len(cluster) == 1:
            det, _, _ = cluster[0]
            return Detection(
                object_type=det.object_type,
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id,
            )

        # Compute weighted average of box coordinates
        total_weight = 0.0
        weighted_x1 = 0.0
        weighted_y1 = 0.0
        weighted_x2 = 0.0
        weighted_y2 = 0.0

        max_confidence = 0.0
        contributing_detectors = set()

        for det, detector_name, det_weight in cluster:
            # Weight = confidence * detector_weight
            w = det.confidence * det_weight
            total_weight += w

            x1, y1, x2, y2 = det.bbox
            weighted_x1 += x1 * w
            weighted_y1 += y1 * w
            weighted_x2 += x2 * w
            weighted_y2 += y2 * w

            max_confidence = max(max_confidence, det.confidence)
            contributing_detectors.add(detector_name)

        # Compute weighted average coordinates
        fused_bbox = (
            weighted_x1 / total_weight,
            weighted_y1 / total_weight,
            weighted_x2 / total_weight,
            weighted_y2 / total_weight,
        )

        # Boost confidence when multiple detectors agree
        # Boost formula: conf * (1 + 0.1 * (n_detectors - 1))
        # Capped at 0.99 to avoid certainty
        n_detectors = len(contributing_detectors)
        confidence_boost = 1.0 + 0.1 * (n_detectors - 1)
        fused_confidence = min(0.99, max_confidence * confidence_boost)

        # Use class_id from highest confidence detection
        class_id = max(cluster, key=lambda x: x[0].confidence)[0].class_id

        return Detection(
            object_type=object_type,
            bbox=fused_bbox,
            confidence=fused_confidence,
            class_id=class_id,
        )

    def _non_max_suppression(
        self,
        detections: list[tuple[Detection, str, float]],
    ) -> list[Detection]:
        """Standard Non-Maximum Suppression.

        Keeps highest confidence box and removes overlapping boxes.
        """
        if len(detections) == 0:
            return []

        # Sort by confidence (descending)
        sorted_dets = sorted(
            detections,
            key=lambda x: x[0].confidence,
            reverse=True,
        )

        kept = []
        used = [False] * len(sorted_dets)

        for i, (det_i, _, _) in enumerate(sorted_dets):
            if used[i]:
                continue

            kept.append(det_i)
            used[i] = True

            # Suppress overlapping boxes
            for j, (det_j, _, _) in enumerate(sorted_dets):
                if used[j]:
                    continue

                iou = compute_iou(det_i.bbox, det_j.bbox)
                if iou >= self.iou_threshold:
                    used[j] = True

        return kept

    def _soft_nms(
        self,
        detections: list[tuple[Detection, str, float]],
        sigma: float = 0.5,
    ) -> list[Detection]:
        """Soft Non-Maximum Suppression.

        Reduces confidence of overlapping boxes instead of removing them.
        """
        if len(detections) == 0:
            return []

        # Sort by confidence (descending)
        sorted_dets = sorted(
            detections,
            key=lambda x: x[0].confidence,
            reverse=True,
        )

        # Work with copies to modify confidence
        results = []
        confidences = [det.confidence for det, _, _ in sorted_dets]

        for i in range(len(sorted_dets)):
            det_i, _, _ = sorted_dets[i]

            if confidences[i] < self.skip_box_threshold:
                continue

            # Keep this detection with current confidence
            results.append(
                Detection(
                    object_type=det_i.object_type,
                    bbox=det_i.bbox,
                    confidence=confidences[i],
                    class_id=det_i.class_id,
                )
            )

            # Decay confidence of overlapping boxes
            for j in range(i + 1, len(sorted_dets)):
                det_j, _, _ = sorted_dets[j]
                iou = compute_iou(det_i.bbox, det_j.bbox)

                if iou > 0:
                    # Gaussian decay
                    decay = np.exp(-(iou * iou) / sigma)
                    confidences[j] *= decay

        return results
