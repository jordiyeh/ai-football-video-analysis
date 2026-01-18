"""Video reading and frame extraction utilities."""

import json
from pathlib import Path
from typing import Generator, Literal

import cv2
import numpy as np


class VideoMetadata:
    """Container for video metadata."""

    def __init__(
        self,
        fps: float,
        total_frames: int,
        width: int,
        height: int,
        duration: float,
        codec: str,
    ):
        self.fps = fps
        self.total_frames = total_frames
        self.width = width
        self.height = height
        self.duration = duration
        self.codec = codec

    def to_dict(self) -> dict:
        """Convert metadata to dictionary."""
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "codec": self.codec,
        }

    def save(self, path: str | Path) -> None:
        """Save metadata to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class VideoReader:
    """Read and validate soccer match videos."""

    def __init__(self, video_path: str | Path):
        """
        Initialize video reader.

        Args:
            video_path: Path to video file

        Raises:
            ValueError: If video cannot be opened
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise ValueError(f"Video file does not exist: {video_path}")

        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Extract metadata
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.codec = self._get_codec()

    def _get_codec(self) -> str:
        """Extract codec information."""
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        return codec

    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata."""
        return VideoMetadata(
            fps=self.fps,
            total_frames=self.total_frames,
            width=self.width,
            height=self.height,
            duration=self.duration,
            codec=self.codec,
        )

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """
        Read next frame from video.

        Returns:
            Tuple of (success, frame)
        """
        ret, frame = self.cap.read()
        return ret, frame if ret else None

    def seek(self, frame_idx: int) -> bool:
        """
        Seek to specific frame.

        Args:
            frame_idx: Frame index to seek to

        Returns:
            True if seek was successful
        """
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def frames(
        self,
        sampling_strategy: Literal["every_frame", "every_2nd", "every_nth"] = "every_frame",
        sampling_interval: int = 1,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        Iterate over frames with sampling strategy.

        Args:
            sampling_strategy: How to sample frames
            sampling_interval: Interval for "every_nth" strategy
            start_frame: Frame to start from
            end_frame: Frame to end at (None = end of video)

        Yields:
            Tuple of (frame_index, frame)
        """
        # Seek to start frame if needed
        if start_frame > 0:
            self.seek(start_frame)

        frame_idx = start_frame
        max_frame = end_frame if end_frame is not None else self.total_frames

        while frame_idx < max_frame:
            ret, frame = self.read_frame()
            if not ret:
                break

            # Apply sampling strategy
            should_yield = False
            if sampling_strategy == "every_frame":
                should_yield = True
            elif sampling_strategy == "every_2nd":
                should_yield = frame_idx % 2 == 0
            elif sampling_strategy == "every_nth":
                should_yield = frame_idx % sampling_interval == 0

            if should_yield:
                yield frame_idx, frame

            frame_idx += 1

    def get_frame_at(self, frame_idx: int) -> np.ndarray | None:
        """
        Get specific frame by index.

        Args:
            frame_idx: Frame index

        Returns:
            Frame array or None if failed
        """
        if self.seek(frame_idx):
            ret, frame = self.read_frame()
            return frame if ret else None
        return None

    def get_frame_at_time(self, time_seconds: float) -> np.ndarray | None:
        """
        Get frame at specific timestamp.

        Args:
            time_seconds: Time in seconds

        Returns:
            Frame array or None if failed
        """
        frame_idx = int(time_seconds * self.fps)
        return self.get_frame_at(frame_idx)

    def close(self) -> None:
        """Release video capture."""
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()
