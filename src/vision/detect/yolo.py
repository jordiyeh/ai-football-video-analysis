"""YOLO-based player and ball detection."""

from typing import Literal

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.vision.detect.base import ObjectDetector


class Detection:
    """Single object detection result."""

    def __init__(
        self,
        object_type: Literal["player", "ball", "unknown"],
        bbox: tuple[float, float, float, float],  # x1, y1, x2, y2
        confidence: float,
        class_id: int,
    ):
        self.object_type = object_type
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id

    @property
    def center(self) -> tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        """Get width of bounding box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Get height of bounding box."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        """Get area of bounding box."""
        return self.width * self.height

    def to_dict(self) -> dict:
        """Convert detection to dictionary."""
        return {
            "object_type": self.object_type,
            "bbox": list(self.bbox),
            "center": list(self.center),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "width": self.width,
            "height": self.height,
            "area": self.area,
        }


class YOLODetector(ObjectDetector):
    """Player and ball detection using YOLOv8."""

    @property
    def name(self) -> str:
        """Return the detector name for logging."""
        return "yolo"

    @property
    def supported_types(self) -> list[Literal["player", "ball"]]:
        """Return list of object types this detector can detect."""
        return ["player", "ball"]

    def __init__(
        self,
        model_name: str = "yolov8x.pt",
        device: Literal["mps", "cpu", "cuda"] = "mps",
        player_class_id: int = 0,  # COCO person class
        ball_class_id: int = 32,  # COCO sports ball class
        confidence_threshold: float = 0.5,
        ball_confidence_threshold: float | None = None,  # Separate threshold for ball
        ball_max_size_ratio: float = 0.05,  # Max ball size as fraction of frame
    ):
        """
        Initialize YOLO detector.

        Args:
            model_name: YOLOv8 model name or path
            device: Device to run inference on
            player_class_id: COCO class ID for players
            ball_class_id: COCO class ID for ball
            confidence_threshold: Minimum confidence for player detections
            ball_confidence_threshold: Minimum confidence for ball (default: lower than players)
            ball_max_size_ratio: Maximum ball bbox dimension as fraction of frame
        """
        self.model_name = model_name
        self.player_class_id = player_class_id
        self.ball_class_id = ball_class_id
        self.confidence_threshold = confidence_threshold
        self.ball_confidence_threshold = ball_confidence_threshold or confidence_threshold * 0.5
        self.ball_max_size_ratio = ball_max_size_ratio

        # Check device availability
        self.device = self._select_device(device)

        # Load model
        self.model = YOLO(model_name)

        # Model will automatically use the correct device on first inference
        print(f"YOLODetector initialized with device: {self.device}")
        print(f"  Player confidence threshold: {self.confidence_threshold}")
        print(f"  Ball confidence threshold: {self.ball_confidence_threshold}")

    def _select_device(self, requested_device: str) -> str:
        """
        Select appropriate device based on availability.

        Args:
            requested_device: Requested device (mps, cuda, cpu)

        Returns:
            Selected device name
        """
        if requested_device == "mps" and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            return "cpu"
        elif requested_device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            return "cpu"
        return requested_device

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[Detection]:
        """
        Detect players and ball in a single frame.

        Args:
            frame: Input frame (BGR format from OpenCV)
            confidence_threshold: Override default confidence threshold

        Returns:
            List of Detection objects
        """
        player_threshold = confidence_threshold or self.confidence_threshold
        ball_threshold = self.ball_confidence_threshold

        # Get frame dimensions for ball size filtering
        frame_height, frame_width = frame.shape[:2]
        max_ball_dimension = max(frame_width, frame_height) * self.ball_max_size_ratio

        # Run inference with lower threshold to catch balls
        min_threshold = min(player_threshold, ball_threshold)
        results = self.model(frame, device=self.device, verbose=False, conf=min_threshold)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Extract bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1

                # Filter by class and apply class-specific thresholds
                if cls == self.player_class_id:
                    if conf < player_threshold:
                        continue
                    object_type = "player"
                elif cls == self.ball_class_id:
                    if conf < ball_threshold:
                        continue
                    # Ball size filtering: reject if too large (likely a misclassification)
                    if max(width, height) > max_ball_dimension:
                        continue
                    # Reject very elongated boxes (balls should be roughly square)
                    aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
                    if aspect_ratio > 3.0:
                        continue
                    object_type = "ball"
                else:
                    continue

                detection = Detection(
                    object_type=object_type,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls,
                )
                detections.append(detection)

        return detections

    def detect_batch(
        self,
        frames: list[np.ndarray],
        confidence_threshold: float | None = None,
    ) -> list[list[Detection]]:
        """
        Detect players and ball in multiple frames (batch processing).

        Args:
            frames: List of input frames
            confidence_threshold: Override default confidence threshold

        Returns:
            List of detection lists (one per frame)
        """
        player_threshold = confidence_threshold or self.confidence_threshold
        ball_threshold = self.ball_confidence_threshold

        # Get frame dimensions for ball size filtering (use first frame)
        if len(frames) > 0:
            frame_height, frame_width = frames[0].shape[:2]
            max_ball_dimension = max(frame_width, frame_height) * self.ball_max_size_ratio
        else:
            max_ball_dimension = float("inf")

        # Run batch inference with lower threshold
        min_threshold = min(player_threshold, ball_threshold)
        results = self.model(frames, device=self.device, verbose=False, conf=min_threshold)

        all_detections = []
        for result in results:
            frame_detections = []
            boxes = result.boxes

            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Extract bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1

                # Filter by class and apply class-specific thresholds
                if cls == self.player_class_id:
                    if conf < player_threshold:
                        continue
                    object_type = "player"
                elif cls == self.ball_class_id:
                    if conf < ball_threshold:
                        continue
                    # Ball size filtering: reject if too large
                    if max(width, height) > max_ball_dimension:
                        continue
                    # Reject very elongated boxes
                    aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
                    if aspect_ratio > 3.0:
                        continue
                    object_type = "ball"
                else:
                    continue

                detection = Detection(
                    object_type=object_type,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls,
                )
                frame_detections.append(detection)

            all_detections.append(frame_detections)

        return all_detections

    def detect_multiscale(
        self,
        frame: np.ndarray,
        scales: list[float] = [0.5, 1.0, 1.5],
        merge_iou_threshold: float = 0.5,
        ball_only: bool = False,
    ) -> list[Detection]:
        """
        Multi-scale detection for improved ball detection.

        Runs detection at multiple scales, transforms bboxes back to original
        coordinates, and merges using soft-NMS.

        Args:
            frame: Input frame (BGR format from OpenCV)
            scales: List of scale factors to use
            merge_iou_threshold: IoU threshold for soft-NMS merging
            ball_only: If True, only detect at multiple scales for ball

        Returns:
            List of merged Detection objects
        """
        from src.vision.detect.ball_boost import soft_nms

        frame_height, frame_width = frame.shape[:2]
        all_scale_detections: list[list[Detection]] = []

        for scale in scales:
            if scale == 1.0:
                # Use original frame
                scaled_frame = frame
                scale_h, scale_w = frame_height, frame_width
            else:
                # Resize frame
                scale_w = int(frame_width * scale)
                scale_h = int(frame_height * scale)

                # Ensure dimensions are at least 32 (YOLO requirement)
                scale_w = max(32, scale_w)
                scale_h = max(32, scale_h)

                interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
                scaled_frame = cv2.resize(frame, (scale_w, scale_h), interpolation=interpolation)

            # Run detection
            detections = self.detect(scaled_frame)

            # Transform bboxes back to original coordinates
            if scale != 1.0:
                scale_x = frame_width / scale_w
                scale_y = frame_height / scale_h

                transformed = []
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    new_bbox = (
                        x1 * scale_x,
                        y1 * scale_y,
                        x2 * scale_x,
                        y2 * scale_y,
                    )
                    transformed.append(
                        Detection(
                            object_type=det.object_type,
                            bbox=new_bbox,
                            confidence=det.confidence,
                            class_id=det.class_id,
                        )
                    )
                detections = transformed

            all_scale_detections.append(detections)

        # Separate ball and player detections
        ball_detections = []
        player_detections = []

        for scale_dets in all_scale_detections:
            for det in scale_dets:
                if det.object_type == "ball":
                    ball_detections.append(det)
                else:
                    player_detections.append(det)

        # Apply soft-NMS to ball detections (merge across scales)
        merged_ball = soft_nms(ball_detections, iou_threshold=merge_iou_threshold)

        # For players, just use the 1.0 scale (multiscale less helpful)
        if ball_only:
            # Return original player detections from 1.0 scale only
            scale_1_idx = scales.index(1.0) if 1.0 in scales else 0
            player_detections = [
                d for d in all_scale_detections[scale_1_idx] if d.object_type == "player"
            ]
        else:
            # Apply soft-NMS to player detections too
            merged_players = soft_nms(player_detections, iou_threshold=merge_iou_threshold)
            player_detections = merged_players

        return player_detections + merged_ball
