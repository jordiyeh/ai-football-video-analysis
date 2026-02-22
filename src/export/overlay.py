"""Video overlay rendering for detections and tracks."""

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.config.schemas import OverlayConfig
from src.vision.detect.yolo import Detection

logger = logging.getLogger(__name__)


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

    def draw_scoreboard(
        self,
        frame: np.ndarray,
        score: dict[str, int] | None = None,
        match_time: float | None = None,
        team_names: dict[str, str] | None = None,
    ) -> np.ndarray:
        """
        Draw scoreboard overlay showing current score and match time.

        Args:
            frame: Input frame
            score: Dict with team_id -> goals (e.g. {"ours": 1, "opponent": 0})
            match_time: Current timestamp in seconds
            team_names: Optional mapping of team_id -> display name

        Returns:
            Annotated frame
        """
        if score is None and match_time is None:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        team_names = team_names or {}
        score = score or {}

        # Format time as MM:SS
        time_str = ""
        if match_time is not None:
            minutes = int(match_time // 60)
            seconds = int(match_time % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"

        # Build score text
        teams = sorted(score.keys())
        if len(teams) >= 2:
            name_a = team_names.get(teams[0], teams[0]).upper()[:10]
            name_b = team_names.get(teams[1], teams[1]).upper()[:10]
            score_text = f"{name_a} {score[teams[0]]} - {score[teams[1]]} {name_b}"
        elif len(teams) == 1:
            name_a = team_names.get(teams[0], teams[0]).upper()[:10]
            score_text = f"{name_a} {score[teams[0]]}"
        else:
            score_text = ""

        # Combine
        display = f"{score_text}  {time_str}".strip() if score_text else time_str

        if not display:
            return annotated

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(1.2, w / 1200))
        thickness = max(1, int(font_scale * 2))

        (text_w, text_h), baseline = cv2.getTextSize(display, font, font_scale, thickness)

        # Position: top-center
        pad_x, pad_y = 16, 8
        box_w = text_w + pad_x * 2
        box_h = text_h + baseline + pad_y * 2
        x_start = (w - box_w) // 2
        y_start = 10

        # Semi-transparent background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x_start, y_start),
                      (x_start + box_w, y_start + box_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

        # Draw text
        text_x = x_start + pad_x
        text_y = y_start + pad_y + text_h
        cv2.putText(annotated, display, (text_x, text_y),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

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


class FfmpegVideoWriter:
    """Write video via ffmpeg subprocess for proper H.264 compression.

    Pipes raw BGR frames to ffmpeg's stdin for encoding with libx264.
    Produces dramatically smaller files than OpenCV's VideoWriter
    (e.g., 4-6 GB vs 37 GB for a 96-min 1080p@30fps match).

    Same interface as VideoWriter (write_frame, close, context manager).
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        crf: int = 23,
        preset: str = "medium",
    ):
        """
        Initialize ffmpeg-based video writer.

        Args:
            output_path: Output video path
            fps: Frames per second
            width: Frame width
            height: Frame height
            crf: Constant Rate Factor (0=lossless, 23=default, 28=smaller)
            preset: Encoding speed/quality tradeoff
                    (ultrafast/fast/medium/slow/veryslow)
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self._closed = False

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Verify ffmpeg is available
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError(
                "ffmpeg not found in PATH. Install with: brew install ffmpeg"
            )

        # Build ffmpeg command
        # -g sets keyframe interval (2 seconds) for clean HLS segment boundaries
        keyint = max(1, int(fps * 2))
        cmd = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            # Input: raw BGR frames from pipe
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            # Output: H.264 with CRF compression
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            "-g", str(keyint),
            "-keyint_min", str(keyint),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(self.output_path),
        ]

        logger.info(
            "FfmpegVideoWriter: %dx%d @ %.1ffps, crf=%d, preset=%s -> %s",
            width, height, fps, crf, preset, self.output_path,
        )

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a single frame to video.

        Args:
            frame: BGR frame (numpy array, shape HxWx3, dtype uint8)
        """
        if self._closed:
            raise RuntimeError("Cannot write to closed FfmpegVideoWriter")
        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            # ffmpeg process died; collect stderr for diagnostics
            _, stderr = self.proc.communicate()
            raise RuntimeError(
                f"ffmpeg process died unexpectedly: {stderr.decode(errors='replace')}"
            )

    def close(self) -> None:
        """Finalize and close the ffmpeg process."""
        if self._closed:
            return
        self._closed = True
        if self.proc is not None and self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            self.proc.wait()
            if self.proc.returncode != 0:
                stderr = self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else ""
                logger.error("ffmpeg exited with code %d: %s", self.proc.returncode, stderr)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


def package_hls(
    mp4_path: str | Path,
    hls_dir: str | Path,
    segment_seconds: int = 10,
) -> Path | None:
    """Split a compressed MP4 into HLS segments (stream copy, no re-encoding).

    Produces an HLS playlist (.m3u8) and fragmented MP4 segments (.m4s).
    This runs in seconds because it copies the bitstream without re-encoding.

    Args:
        mp4_path: Path to the source MP4 (must have keyframes at regular intervals)
        hls_dir: Output directory for HLS files
        segment_seconds: Target segment duration in seconds (actual may vary by keyframe)

    Returns:
        Path to the playlist.m3u8 file, or None if ffmpeg is not available / fails.
    """
    mp4_path = Path(mp4_path)
    hls_dir = Path(hls_dir)

    if not mp4_path.exists():
        logger.warning("package_hls: source MP4 not found: %s", mp4_path)
        return None

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        logger.warning("package_hls: ffmpeg not found, skipping HLS packaging")
        return None

    hls_dir.mkdir(parents=True, exist_ok=True)
    playlist_path = hls_dir / "playlist.m3u8"

    cmd = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(mp4_path),
        "-codec", "copy",
        "-hls_time", str(segment_seconds),
        "-hls_list_size", "0",
        "-hls_segment_type", "fmp4",
        "-hls_segment_filename", str(hls_dir / "seg_%04d.m4s"),
        "-f", "hls",
        str(playlist_path),
    ]

    logger.info("package_hls: splitting %s into %d-second segments -> %s", mp4_path.name, segment_seconds, hls_dir)

    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        # Count segments produced
        segments = list(hls_dir.glob("seg_*.m4s"))
        logger.info("package_hls: produced %d segments + playlist", len(segments))
        return playlist_path
    except subprocess.CalledProcessError as exc:
        logger.error(
            "package_hls failed (exit %d): %s",
            exc.returncode,
            exc.stderr.decode(errors="replace") if exc.stderr else "",
        )
        return None
    except FileNotFoundError:
        logger.error("package_hls: ffmpeg binary not found")
        return None
