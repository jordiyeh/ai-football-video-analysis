"""Specialized ball detector using a soccer-trained YOLO model."""

import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.vision.detect.base import ObjectDetector
from src.vision.detect.yolo import Detection


class BallSpecialistDetector(ObjectDetector):
    """Specialized ball detector using a soccer-trained YOLO model.

    This detector uses a model specifically trained on soccer balls,
    which tends to have much better ball detection than general COCO models.
    """

    @property
    def name(self) -> str:
        """Return the detector name for logging."""
        return "ball_specialist"

    @property
    def supported_types(self) -> list[Literal["player", "ball"]]:
        """Return list of object types this detector can detect."""
        return ["ball"]

    def __init__(
        self,
        model_source: str = "keremberke/yolov8n-soccer-ball-detection",
        device: Literal["mps", "cpu", "cuda"] = "mps",
        confidence_threshold: float = 0.3,
        ball_class_id: int = 0,  # Class ID for ball in the specialized model
        max_size_ratio: float = 0.08,  # Max ball size as fraction of frame
        max_aspect_ratio: float = 3.0,  # Reject very elongated boxes
        cache_dir: str = "models",
    ):
        """Initialize ball specialist detector.

        Args:
            model_source: HuggingFace model ID or local path to weights
            device: Device to run inference on
            confidence_threshold: Minimum confidence for detections
            ball_class_id: Class ID for ball in the model
            max_size_ratio: Maximum ball bbox dimension as fraction of frame
            max_aspect_ratio: Maximum aspect ratio (reject elongated boxes)
            cache_dir: Directory to cache downloaded models
        """
        self.model_source = model_source
        self.confidence_threshold = confidence_threshold
        self.ball_class_id = ball_class_id
        self.max_size_ratio = max_size_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.cache_dir = Path(cache_dir)

        # Check device availability
        self.device = self._select_device(device)

        # Load model
        self.model = self._load_model()

        print(f"BallSpecialistDetector initialized with device: {self.device}")
        print(f"  Model: {model_source}")
        print(f"  Confidence threshold: {self.confidence_threshold}")

    def _select_device(self, requested_device: str) -> str:
        """Select appropriate device based on availability."""
        if requested_device == "mps" and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            return "cpu"
        elif requested_device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            return "cpu"
        return requested_device

    def _load_model(self):
        """Load the YOLO model from HuggingFace or local path."""
        from ultralytics import YOLO

        # Check if it's a local path
        if os.path.exists(self.model_source):
            print(f"Loading model from local path: {self.model_source}")
            return YOLO(self.model_source)

        # Try to load from HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download

            # Create cache directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Download model from HuggingFace
            print(f"Downloading model from HuggingFace: {self.model_source}")
            model_path = hf_hub_download(
                repo_id=self.model_source,
                filename="best.pt",
                cache_dir=str(self.cache_dir),
            )

            return YOLO(model_path)

        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download models from HuggingFace. "
                "Install with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {self.model_source}: {e}"
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect balls in a single frame.

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            List of Detection objects (ball only)
        """
        # Get frame dimensions for size filtering
        frame_height, frame_width = frame.shape[:2]
        max_ball_dimension = max(frame_width, frame_height) * self.max_size_ratio

        # Run inference
        results = self.model(
            frame,
            device=self.device,
            verbose=False,
            conf=self.confidence_threshold,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Only process ball class
                if cls != self.ball_class_id:
                    continue

                # Extract bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1

                # Size filtering: reject if too large
                if max(width, height) > max_ball_dimension:
                    continue

                # Aspect ratio filtering: reject very elongated boxes
                aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
                if aspect_ratio > self.max_aspect_ratio:
                    continue

                detection = Detection(
                    object_type="ball",
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls,
                )
                detections.append(detection)

        return detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Detect balls in multiple frames (batch processing).

        Args:
            frames: List of input frames

        Returns:
            List of detection lists (one per frame)
        """
        if len(frames) == 0:
            return []

        # Get frame dimensions for size filtering (use first frame)
        frame_height, frame_width = frames[0].shape[:2]
        max_ball_dimension = max(frame_width, frame_height) * self.max_size_ratio

        # Run batch inference
        results = self.model(
            frames,
            device=self.device,
            verbose=False,
            conf=self.confidence_threshold,
        )

        all_detections = []
        for result in results:
            frame_detections = []
            boxes = result.boxes

            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Only process ball class
                if cls != self.ball_class_id:
                    continue

                # Extract bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1

                # Size filtering
                if max(width, height) > max_ball_dimension:
                    continue

                # Aspect ratio filtering
                aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
                if aspect_ratio > self.max_aspect_ratio:
                    continue

                detection = Detection(
                    object_type="ball",
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls,
                )
                frame_detections.append(detection)

            all_detections.append(frame_detections)

        return all_detections
