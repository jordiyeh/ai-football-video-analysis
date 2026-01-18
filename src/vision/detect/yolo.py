"""YOLO-based player and ball detection."""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from ultralytics import YOLO


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


class YOLODetector:
    """Player and ball detection using YOLOv8."""

    def __init__(
        self,
        model_name: str = "yolov8x.pt",
        device: Literal["mps", "cpu", "cuda"] = "mps",
        player_class_id: int = 0,  # COCO person class
        ball_class_id: int = 32,  # COCO sports ball class
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize YOLO detector.

        Args:
            model_name: YOLOv8 model name or path
            device: Device to run inference on
            player_class_id: COCO class ID for players
            ball_class_id: COCO class ID for ball
            confidence_threshold: Minimum confidence for detections
        """
        self.model_name = model_name
        self.player_class_id = player_class_id
        self.ball_class_id = ball_class_id
        self.confidence_threshold = confidence_threshold

        # Check device availability
        self.device = self._select_device(device)

        # Load model
        self.model = YOLO(model_name)

        # Model will automatically use the correct device on first inference
        print(f"YOLODetector initialized with device: {self.device}")

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
        threshold = confidence_threshold or self.confidence_threshold

        # Run inference
        results = self.model(frame, device=self.device, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < threshold:
                    continue

                # Filter to only players and ball
                if cls == self.player_class_id:
                    object_type = "player"
                elif cls == self.ball_class_id:
                    object_type = "ball"
                else:
                    continue

                # Extract bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()

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
        threshold = confidence_threshold or self.confidence_threshold

        # Run batch inference
        results = self.model(frames, device=self.device, verbose=False)

        all_detections = []
        for result in results:
            frame_detections = []
            boxes = result.boxes

            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < threshold:
                    continue

                # Filter to only players and ball
                if cls == self.player_class_id:
                    object_type = "player"
                elif cls == self.ball_class_id:
                    object_type = "ball"
                else:
                    continue

                # Extract bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detection = Detection(
                    object_type=object_type,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls,
                )
                frame_detections.append(detection)

            all_detections.append(frame_detections)

        return all_detections
