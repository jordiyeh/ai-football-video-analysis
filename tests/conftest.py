"""Shared test fixtures and helpers for the test suite."""

import importlib.util
import numpy as np
import pytest
from dataclasses import dataclass
from typing import Generator
from pathlib import Path

_OPTIONAL_TEST_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "tests/golden/test_regression.py": ("filterpy",),
    "tests/integration/test_detection_tracking.py": ("filterpy", "cv2", "torch"),
    "tests/integration/test_event_pipeline.py": ("cv2",),
    "tests/integration/test_team_pipeline.py": ("cv2",),
    "tests/unit/test_ball_boost.py": ("cv2", "torch"),
    "tests/unit/test_bytetrack.py": ("filterpy",),
    "tests/unit/test_celebration_detection.py": ("cv2",),
    "tests/unit/test_colors.py": ("cv2",),
    "tests/unit/test_detection.py": ("cv2", "torch"),
    "tests/unit/test_embedding_generator.py": ("cv2",),
    "tests/unit/test_ensemble.py": ("cv2", "torch"),
    "tests/unit/test_event_detection.py": ("cv2",),
    "tests/unit/test_field_normalization.py": ("cv2",),
    "tests/unit/test_goal_detector.py": ("cv2",),
    "tests/unit/test_highlights.py": ("cv2",),
    "tests/unit/test_identity_db.py": ("cv2",),
    "tests/unit/test_identity_fusion.py": ("cv2",),
    "tests/unit/test_identity_multimodal.py": ("cv2",),
    "tests/unit/test_kalman.py": ("filterpy",),
    "tests/unit/test_kick_detection.py": ("cv2",),
    "tests/unit/test_kit_colors.py": ("cv2",),
    "tests/unit/test_overlay.py": ("cv2",),
    "tests/unit/test_player_analysis.py": ("cv2",),
    "tests/unit/test_player_reels.py": ("cv2",),
    "tests/unit/test_reid.py": ("torch",),
    "tests/unit/test_team_clustering.py": ("cv2",),
    "tests/unit/test_ui_server.py": ("fastapi",),
    "tests/unit/test_video_reader.py": ("cv2",),
}


def _module_available(name: str) -> bool:
    """Return True when an optional test dependency can be imported."""
    return importlib.util.find_spec(name) is not None


def _to_repo_relative(path_obj: object) -> str:
    """Normalize pytest path objects to repo-relative POSIX paths."""
    path = Path(str(path_obj)).resolve()
    repo_root = Path(__file__).resolve().parent.parent
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def pytest_ignore_collect(collection_path, config):  # type: ignore[no-untyped-def]
    """Skip optional-dependency test files when their dependencies are missing."""
    _ = config
    rel_path = _to_repo_relative(collection_path)
    requirements = _OPTIONAL_TEST_REQUIREMENTS.get(rel_path)
    if not requirements:
        return False

    return any(not _module_available(requirement) for requirement in requirements)


# -----------------------------------------------------------------------------
# Sample Data Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create a synthetic 1920x1080 BGR frame with simple colored regions."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Blue background
    frame[:, :] = [128, 64, 64]  # BGR
    # Red region in top-left (for team 1)
    frame[100:200, 100:200] = [0, 0, 200]  # Red in BGR
    # Green region in top-right (for team 2)
    frame[100:200, 1700:1800] = [0, 200, 0]  # Green in BGR
    return frame


@pytest.fixture
def sample_frame_720p() -> np.ndarray:
    """Create a synthetic 1280x720 BGR frame."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = [100, 100, 100]  # Gray
    return frame


@pytest.fixture
def sample_detections() -> list[dict]:
    """Sample detection dicts as returned by detectors."""
    return [
        {
            "bbox": (100.0, 100.0, 150.0, 200.0),
            "confidence": 0.95,
            "object_type": "player",
        },
        {
            "bbox": (200.0, 150.0, 250.0, 250.0),
            "confidence": 0.88,
            "object_type": "player",
        },
        {
            "bbox": (500.0, 400.0, 520.0, 420.0),
            "confidence": 0.75,
            "object_type": "ball",
        },
    ]


@pytest.fixture
def sample_ball_detection() -> dict:
    """Single ball detection dict."""
    return {
        "bbox": (960.0, 540.0, 980.0, 560.0),
        "confidence": 0.85,
        "object_type": "ball",
    }


@pytest.fixture
def sample_player_detections() -> list[dict]:
    """Multiple player detection dicts."""
    return [
        {"bbox": (100.0, 100.0, 150.0, 200.0), "confidence": 0.92, "object_type": "player"},
        {"bbox": (200.0, 150.0, 250.0, 250.0), "confidence": 0.88, "object_type": "player"},
        {"bbox": (300.0, 100.0, 350.0, 200.0), "confidence": 0.85, "object_type": "player"},
        {"bbox": (400.0, 150.0, 450.0, 250.0), "confidence": 0.78, "object_type": "player"},
    ]


@pytest.fixture
def default_config():
    """Default pipeline configuration dict."""
    return {
        "detection": {
            "model": "yolov8n",
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
        },
        "tracking": {
            "track_thresh": 0.5,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "min_hits": 3,
        },
        "team": {
            "n_teams": 2,
            "color_space": "hsv",
            "min_samples_per_track": 5,
        },
        "events": {
            "shot_velocity_threshold": 15.0,
            "goal_confidence_threshold": 0.6,
        },
    }


# -----------------------------------------------------------------------------
# Helper Functions for Creating Test Data
# -----------------------------------------------------------------------------

def make_detection(
    x1: float = 100.0,
    y1: float = 100.0,
    x2: float = 150.0,
    y2: float = 200.0,
    confidence: float = 0.9,
    object_type: str = "player",
) -> dict:
    """Create a detection dict with specified values."""
    return {
        "bbox": (x1, y1, x2, y2),
        "confidence": confidence,
        "object_type": object_type,
    }


def make_track_dict(
    track_id: int,
    bbox: tuple[float, float, float, float],
    confidence: float = 0.9,
    object_type: str = "player",
    team_id: int | None = None,
) -> dict:
    """Create a track dict as stored in tracks data."""
    d = {
        "track_id": track_id,
        "bbox": bbox,
        "confidence": confidence,
        "object_type": object_type,
    }
    if team_id is not None:
        d["team_id"] = team_id
    return d


def make_trajectory_point(
    frame_idx: int,
    x: float,
    y: float,
    confidence: float = 0.9,
    velocity: tuple[float, float] | None = None,
    speed: float | None = None,
    fps: float = 30.0,
) -> dict:
    """Create a trajectory point dict."""
    return {
        "frame_idx": frame_idx,
        "timestamp": frame_idx / fps,
        "position": (x, y),
        "velocity": velocity,
        "speed": speed,
        "confidence": confidence,
    }


def make_bbox(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    """Create a bbox tuple from center and dimensions."""
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    return (x1, y1, x2, y2)


def make_colored_frame(
    width: int = 1920,
    height: int = 1080,
    color: tuple[int, int, int] = (100, 100, 100),
) -> np.ndarray:
    """Create a solid colored frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def make_frame_with_region(
    width: int = 1920,
    height: int = 1080,
    bg_color: tuple[int, int, int] = (100, 100, 100),
    region_bbox: tuple[int, int, int, int] = (100, 100, 200, 200),
    region_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Create a frame with a colored region."""
    frame = make_colored_frame(width, height, bg_color)
    x1, y1, x2, y2 = region_bbox
    frame[y1:y2, x1:x2] = region_color
    return frame


# -----------------------------------------------------------------------------
# Mock Classes
# -----------------------------------------------------------------------------

class MockDetector:
    """Mock detector that returns predefined detections."""

    def __init__(self, detections_by_frame: dict[int, list[dict]] | None = None):
        """Initialize with optional frame->detections mapping."""
        self.detections_by_frame = detections_by_frame or {}
        self.detect_calls = []

    def detect(self, frames: list[np.ndarray], frame_indices: list[int] | None = None) -> list[list[dict]]:
        """Return predefined detections for each frame."""
        self.detect_calls.append((len(frames), frame_indices))
        results = []
        for i, frame in enumerate(frames):
            idx = frame_indices[i] if frame_indices else i
            detections = self.detections_by_frame.get(idx, [])
            results.append(detections)
        return results


class MockVideoReader:
    """Mock video reader that returns synthetic frames."""

    def __init__(
        self,
        total_frames: int = 100,
        fps: float = 30.0,
        width: int = 1920,
        height: int = 1080,
        frame_color: tuple[int, int, int] = (100, 100, 100),
    ):
        """Initialize with video parameters."""
        self.total_frames = total_frames
        self.fps = fps
        self.width = width
        self.height = height
        self.frame_color = frame_color
        self._current_frame = 0
        self._closed = False

    @property
    def metadata(self):
        """Return mock metadata."""
        return MockVideoMetadata(
            fps=self.fps,
            total_frames=self.total_frames,
            width=self.width,
            height=self.height,
            duration=self.total_frames / self.fps,
            codec="h264",
        )

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read next frame."""
        if self._current_frame >= self.total_frames or self._closed:
            return False, None
        frame = make_colored_frame(self.width, self.height, self.frame_color)
        self._current_frame += 1
        return True, frame

    def seek(self, frame_idx: int) -> bool:
        """Seek to frame."""
        if 0 <= frame_idx < self.total_frames:
            self._current_frame = frame_idx
            return True
        return False

    def get_frame_at(self, frame_idx: int) -> np.ndarray | None:
        """Get frame at specific index."""
        if 0 <= frame_idx < self.total_frames:
            return make_colored_frame(self.width, self.height, self.frame_color)
        return None

    def frames(
        self,
        sampling_strategy: str = "every_frame",
        sampling_interval: int = 1,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """Generate frames."""
        end = end_frame if end_frame is not None else self.total_frames
        step = sampling_interval if sampling_strategy == "every_nth" else 1
        if sampling_strategy == "every_2nd":
            step = 2
        for i in range(start_frame, end, step):
            yield i, make_colored_frame(self.width, self.height, self.frame_color)

    def close(self):
        """Close the reader."""
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass
class MockVideoMetadata:
    """Mock video metadata."""
    fps: float
    total_frames: int
    width: int
    height: int
    duration: float
    codec: str

    def to_dict(self) -> dict:
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "codec": self.codec,
        }


class MockTracker:
    """Mock tracker for testing."""

    def __init__(self, tracks_by_frame: dict[int, list[dict]] | None = None):
        """Initialize with optional frame->tracks mapping."""
        self.tracks_by_frame = tracks_by_frame or {}
        self._frame_idx = 0
        self.update_calls = []

    def update(self, detections: list[dict]) -> list[dict]:
        """Return predefined tracks."""
        self.update_calls.append(detections)
        tracks = self.tracks_by_frame.get(self._frame_idx, [])
        self._frame_idx += 1
        return tracks


class MockKalmanFilter:
    """Mock Kalman filter for testing tracker."""

    def __init__(self):
        self._state = np.zeros(8)
        self._measurement = np.zeros(4)

    def initiate(self, measurement: np.ndarray):
        """Initialize filter state."""
        self._measurement = measurement.copy()
        self._state[:4] = measurement

    def predict(self) -> np.ndarray:
        """Return predicted measurement."""
        return self._measurement.copy()

    def update(self, measurement: np.ndarray):
        """Update with measurement."""
        self._measurement = measurement.copy()
        self._state[:4] = measurement

    def get_state(self) -> np.ndarray:
        """Return current state."""
        return self._state.copy()


# -----------------------------------------------------------------------------
# Temporary File Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_video_path(tmp_path: Path) -> Path:
    """Create a path for a temporary video file (not actually created)."""
    return tmp_path / "test_video.mp4"


# -----------------------------------------------------------------------------
# Common Test Data
# -----------------------------------------------------------------------------

@pytest.fixture
def frame_sequence_with_ball() -> list[dict]:
    """A sequence of frames with ball positions for trajectory testing."""
    return [
        {"frame_idx": 0, "ball_position": (100, 100), "ball_confidence": 0.9},
        {"frame_idx": 1, "ball_position": (110, 102), "ball_confidence": 0.88},
        {"frame_idx": 2, "ball_position": (120, 104), "ball_confidence": 0.85},
        {"frame_idx": 3, "ball_position": None, "ball_confidence": 0.0},  # missing
        {"frame_idx": 4, "ball_position": (140, 108), "ball_confidence": 0.82},
        {"frame_idx": 5, "ball_position": (150, 110), "ball_confidence": 0.9},
    ]


@pytest.fixture
def tracks_single_player() -> list[list[dict]]:
    """Track data for a single player across 5 frames."""
    return [
        [make_track_dict(1, (100, 100, 150, 200), 0.9)],
        [make_track_dict(1, (105, 102, 155, 202), 0.88)],
        [make_track_dict(1, (110, 104, 160, 204), 0.87)],
        [make_track_dict(1, (115, 106, 165, 206), 0.85)],
        [make_track_dict(1, (120, 108, 170, 208), 0.86)],
    ]


@pytest.fixture
def tracks_two_teams() -> list[list[dict]]:
    """Track data for players from two teams."""
    return [
        [
            make_track_dict(1, (100, 100, 150, 200), 0.9, team_id=0),
            make_track_dict(2, (200, 100, 250, 200), 0.88, team_id=0),
            make_track_dict(3, (800, 100, 850, 200), 0.85, team_id=1),
            make_track_dict(4, (900, 100, 950, 200), 0.82, team_id=1),
        ],
    ]
