"""Video overlay rendering for detections and tracks."""

from pathlib import Path

import cv2
import numpy as np

from src.config.schemas import OverlayConfig
from src.vision.detect.yolo import Detection


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """
    Convert hex color to BGR tuple for OpenCV.

    Args:
        hex_color: Hex color string (e.g., "#FF0000")

    Returns:
        BGR color tuple
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV uses BGR


class OverlayRenderer:
    """Render detection overlays on video frames."""

    def __init__(self, config: OverlayConfig):
        """
        Initialize overlay renderer.

        Args:
            config: Overlay configuration
        """
        self.config = config
        self.player_color_bgr = hex_to_bgr(config.player_color)
        self.ball_color_bgr = hex_to_bgr(config.ball_color)

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        track_ids: dict[Detection, int] | None = None,
        team_labels: dict[Detection, str] | None = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: Input frame
            detections: List of detections to draw
            track_ids: Optional mapping of detection to track ID
            team_labels: Optional mapping of detection to team label

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for detection in detections:
            # Skip detections with invalid bounding boxes
            if any(np.isnan(v) or np.isinf(v) for v in detection.bbox):
                continue

            x1, y1, x2, y2 = map(int, detection.bbox)
            confidence = detection.confidence
            obj_type = detection.object_type

            # Choose color based on object type
            if obj_type == "player":
                color = self.player_color_bgr
                # Override with team color if available
                if team_labels and detection in team_labels:
                    team = team_labels[detection]
                    if team == "ours":
                        color = (255, 0, 0)  # Blue
                    elif team == "opponent":
                        color = (0, 0, 255)  # Red
            elif obj_type == "ball":
                color = self.ball_color_bgr
            else:
                color = (128, 128, 128)  # Gray for unknown

            # Draw bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                self.config.bbox_thickness,
            )

            # Prepare label
            label_parts = [obj_type]
            if self.config.show_confidence:
                label_parts.append(f"{confidence:.2f}")
            if self.config.show_track_ids and track_ids and detection in track_ids:
                label_parts.append(f"ID:{track_ids[detection]}")

            label = " ".join(label_parts)

            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        return annotated

    def draw_tracks(
        self,
        frame: np.ndarray,
        track_history: dict[int, list[tuple[float, float]]],
    ) -> np.ndarray:
        """
        Draw track trails on frame.

        Args:
            frame: Input frame
            track_history: Mapping of track ID to list of center points

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for track_id, points in track_history.items():
            if len(points) < 2:
                continue

            # Draw trail
            trail_points = points[-self.config.trail_length :]

            # Filter out points with NaN values
            valid_points = []
            for pt in trail_points:
                if not any(np.isnan(v) or np.isinf(v) for v in pt):
                    valid_points.append(pt)

            if len(valid_points) < 2:
                continue

            for i in range(len(valid_points) - 1):
                pt1 = tuple(map(int, valid_points[i]))
                pt2 = tuple(map(int, valid_points[i + 1]))

                # Fade trail (older points are more transparent)
                alpha = (i + 1) / len(valid_points)
                thickness = max(1, int(3 * alpha))

                cv2.line(annotated, pt1, pt2, (0, 255, 255), thickness)

        return annotated


class VideoWriter:
    """Write annotated video to file."""

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ):
        """
        Initialize video writer.

        Args:
            output_path: Output video path
            fps: Frames per second
            width: Frame width
            height: Frame height
            codec: Video codec (fourcc code)
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not self.writer.isOpened():
            raise ValueError(f"Failed to open video writer: {output_path}")

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a single frame to video.

        Args:
            frame: Frame to write
        """
        self.writer.write(frame)

    def close(self) -> None:
        """Release video writer."""
        if self.writer is not None:
            self.writer.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()
