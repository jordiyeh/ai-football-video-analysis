"""Tests for video reading and frame extraction."""

import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.video.reader import VideoReader, VideoMetadata


# -----------------------------------------------------------------------------
# VideoMetadata Tests
# -----------------------------------------------------------------------------

class TestVideoMetadata:
    """Tests for VideoMetadata class."""

    @pytest.fixture
    def metadata(self):
        """Create sample metadata."""
        return VideoMetadata(
            fps=30.0,
            total_frames=1800,
            width=1920,
            height=1080,
            duration=60.0,
            codec="h264",
        )

    def test_metadata_creation(self, metadata):
        """Test metadata is created correctly."""
        assert metadata.fps == 30.0
        assert metadata.total_frames == 1800
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.duration == 60.0
        assert metadata.codec == "h264"

    def test_to_dict(self, metadata):
        """Test conversion to dictionary."""
        d = metadata.to_dict()

        assert d["fps"] == 30.0
        assert d["total_frames"] == 1800
        assert d["width"] == 1920
        assert d["height"] == 1080
        assert d["duration"] == 60.0
        assert d["codec"] == "h264"

    def test_to_dict_is_json_serializable(self, metadata):
        """Test that to_dict output is JSON serializable."""
        d = metadata.to_dict()

        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_save(self, metadata, tmp_path):
        """Test saving metadata to file."""
        output_path = tmp_path / "metadata.json"

        metadata.save(output_path)

        assert output_path.exists()

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["fps"] == 30.0
        assert loaded["total_frames"] == 1800


# -----------------------------------------------------------------------------
# VideoReader Tests with Mocking
# -----------------------------------------------------------------------------

class TestVideoReaderInit:
    """Tests for VideoReader initialization."""

    def test_nonexistent_file_raises(self, tmp_path):
        """Test that nonexistent file raises ValueError."""
        fake_path = tmp_path / "nonexistent.mp4"

        with pytest.raises(ValueError, match="does not exist"):
            VideoReader(fake_path)

    def test_unopenable_file_raises(self, tmp_path):
        """Test that unopenable file raises ValueError."""
        # Create an empty file (not a valid video)
        bad_file = tmp_path / "bad.mp4"
        bad_file.touch()

        with pytest.raises(ValueError, match="Cannot open video"):
            VideoReader(bad_file)


class TestVideoReaderWithMock:
    """Tests for VideoReader using mocked cv2."""

    @pytest.fixture
    def mock_cap(self):
        """Create a mock VideoCapture."""
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = self._mock_get
        cap.read.return_value = (True, np.zeros((1080, 1920, 3), dtype=np.uint8))
        cap.set.return_value = True
        return cap

    def _mock_get(self, prop):
        """Mock VideoCapture.get for various properties."""
        import cv2
        props = {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FOURCC: ord('h') | (ord('2') << 8) | (ord('6') << 16) | (ord('4') << 24),
        }
        return props.get(prop, 0)

    @patch('src.video.reader.cv2.VideoCapture')
    def test_init_extracts_metadata(self, mock_cv2_cap, mock_cap, tmp_path):
        """Test that init extracts video metadata."""
        mock_cv2_cap.return_value = mock_cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)

        assert reader.fps == 30.0
        assert reader.total_frames == 100
        assert reader.width == 1920
        assert reader.height == 1080

    @patch('src.video.reader.cv2.VideoCapture')
    def test_metadata_property(self, mock_cv2_cap, mock_cap, tmp_path):
        """Test metadata property."""
        mock_cv2_cap.return_value = mock_cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)
        meta = reader.metadata

        assert isinstance(meta, VideoMetadata)
        assert meta.fps == 30.0
        assert meta.total_frames == 100

    @patch('src.video.reader.cv2.VideoCapture')
    def test_read_frame(self, mock_cv2_cap, mock_cap, tmp_path):
        """Test read_frame method."""
        mock_cv2_cap.return_value = mock_cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)
        success, frame = reader.read_frame()

        assert success is True
        assert frame is not None
        assert frame.shape == (1080, 1920, 3)

    @patch('src.video.reader.cv2.VideoCapture')
    def test_read_frame_end_of_video(self, mock_cv2_cap, mock_cap, tmp_path):
        """Test read_frame at end of video."""
        mock_cap.read.return_value = (False, None)
        mock_cv2_cap.return_value = mock_cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)
        success, frame = reader.read_frame()

        assert success is False
        assert frame is None

    @patch('src.video.reader.cv2.VideoCapture')
    def test_seek(self, mock_cv2_cap, mock_cap, tmp_path):
        """Test seek method."""
        mock_cv2_cap.return_value = mock_cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)
        result = reader.seek(50)

        assert result is True
        mock_cap.set.assert_called()

    @patch('src.video.reader.cv2.VideoCapture')
    def test_get_frame_at(self, mock_cv2_cap, mock_cap, tmp_path):
        """Test get_frame_at method."""
        mock_cv2_cap.return_value = mock_cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)
        frame = reader.get_frame_at(25)

        assert frame is not None
        assert frame.shape == (1080, 1920, 3)

    @patch('src.video.reader.cv2.VideoCapture')
    def test_get_frame_at_time(self, mock_cv2_cap, mock_cap, tmp_path):
        """Test get_frame_at_time method."""
        mock_cv2_cap.return_value = mock_cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)
        frame = reader.get_frame_at_time(1.5)  # 1.5 seconds = frame 45 at 30fps

        assert frame is not None

    @patch('src.video.reader.cv2.VideoCapture')
    def test_close(self, mock_cv2_cap, mock_cap, tmp_path):
        """Test close method."""
        mock_cv2_cap.return_value = mock_cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)
        reader.close()

        mock_cap.release.assert_called_once()

    @patch('src.video.reader.cv2.VideoCapture')
    def test_context_manager(self, mock_cv2_cap, mock_cap, tmp_path):
        """Test context manager usage."""
        mock_cv2_cap.return_value = mock_cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        with VideoReader(video_path) as reader:
            success, frame = reader.read_frame()
            assert success is True

        mock_cap.release.assert_called()


class TestVideoReaderFrames:
    """Tests for frames generator method."""

    @pytest.fixture
    def mock_cap(self):
        """Create a mock VideoCapture that yields frames."""
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = self._mock_get

        # Track frame index for proper simulation
        self.frame_idx = 0
        self.max_frames = 100

        def mock_read():
            if self.frame_idx >= self.max_frames:
                return False, None
            self.frame_idx += 1
            return True, np.zeros((1080, 1920, 3), dtype=np.uint8)

        cap.read.side_effect = mock_read
        cap.set.side_effect = self._mock_set
        return cap

    def _mock_get(self, prop):
        """Mock VideoCapture.get."""
        import cv2
        return {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FOURCC: 0,
        }.get(prop, 0)

    def _mock_set(self, prop, value):
        """Mock VideoCapture.set for seeking."""
        import cv2
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.frame_idx = int(value)
            return True
        return True

    @patch('src.video.reader.cv2.VideoCapture')
    def test_frames_every_frame(self, mock_cv2_cap, mock_cap, tmp_path):
        """Test frames generator with every_frame strategy."""
        mock_cv2_cap.return_value = mock_cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)
        frames = list(reader.frames(sampling_strategy="every_frame", end_frame=10))

        assert len(frames) == 10
        # Check indices are sequential
        indices = [idx for idx, _ in frames]
        assert indices == list(range(10))

    @patch('src.video.reader.cv2.VideoCapture')
    def test_frames_every_2nd(self, mock_cv2_cap, mock_cap, tmp_path):
        """Test frames generator with every_2nd strategy."""
        mock_cv2_cap.return_value = mock_cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)
        frames = list(reader.frames(sampling_strategy="every_2nd", end_frame=10))

        # Should get frames 0, 2, 4, 6, 8
        indices = [idx for idx, _ in frames]
        assert indices == [0, 2, 4, 6, 8]

    @patch('src.video.reader.cv2.VideoCapture')
    def test_frames_every_nth(self, mock_cv2_cap, mock_cap, tmp_path):
        """Test frames generator with every_nth strategy."""
        mock_cv2_cap.return_value = mock_cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)
        frames = list(reader.frames(
            sampling_strategy="every_nth",
            sampling_interval=5,
            end_frame=20
        ))

        # Should get frames 0, 5, 10, 15
        indices = [idx for idx, _ in frames]
        assert indices == [0, 5, 10, 15]

    @patch('src.video.reader.cv2.VideoCapture')
    def test_frames_start_frame(self, mock_cv2_cap, mock_cap, tmp_path):
        """Test frames generator with start_frame."""
        mock_cv2_cap.return_value = mock_cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)
        frames = list(reader.frames(start_frame=5, end_frame=10))

        indices = [idx for idx, _ in frames]
        assert indices == [5, 6, 7, 8, 9]

    @patch('src.video.reader.cv2.VideoCapture')
    def test_frames_yields_tuples(self, mock_cv2_cap, mock_cap, tmp_path):
        """Test that frames yields (index, frame) tuples."""
        mock_cv2_cap.return_value = mock_cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)

        for frame_idx, frame in reader.frames(end_frame=3):
            assert isinstance(frame_idx, int)
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (1080, 1920, 3)
            break  # Just check first


class TestVideoReaderDuration:
    """Tests for duration calculation."""

    @patch('src.video.reader.cv2.VideoCapture')
    def test_duration_calculation(self, mock_cv2_cap, tmp_path):
        """Test duration is calculated correctly."""
        cap = MagicMock()
        cap.isOpened.return_value = True

        import cv2
        def mock_get(prop):
            return {
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_COUNT: 900,  # 900 frames at 30fps = 30 seconds
                cv2.CAP_PROP_FRAME_WIDTH: 1920,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                cv2.CAP_PROP_FOURCC: 0,
            }.get(prop, 0)

        cap.get.side_effect = mock_get
        mock_cv2_cap.return_value = cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)

        assert reader.duration == pytest.approx(30.0)

    @patch('src.video.reader.cv2.VideoCapture')
    def test_duration_with_zero_fps(self, mock_cv2_cap, tmp_path):
        """Test duration handling with zero fps."""
        cap = MagicMock()
        cap.isOpened.return_value = True

        import cv2
        def mock_get(prop):
            return {
                cv2.CAP_PROP_FPS: 0.0,  # Zero fps
                cv2.CAP_PROP_FRAME_COUNT: 100,
                cv2.CAP_PROP_FRAME_WIDTH: 1920,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                cv2.CAP_PROP_FOURCC: 0,
            }.get(prop, 0)

        cap.get.side_effect = mock_get
        mock_cv2_cap.return_value = cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)

        assert reader.duration == 0  # Should handle gracefully


class TestVideoReaderCodec:
    """Tests for codec extraction."""

    @patch('src.video.reader.cv2.VideoCapture')
    def test_codec_extraction(self, mock_cv2_cap, tmp_path):
        """Test codec is extracted correctly."""
        cap = MagicMock()
        cap.isOpened.return_value = True

        import cv2
        # FOURCC for "avc1"
        fourcc = ord('a') | (ord('v') << 8) | (ord('c') << 16) | (ord('1') << 24)

        def mock_get(prop):
            return {
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_COUNT: 100,
                cv2.CAP_PROP_FRAME_WIDTH: 1920,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                cv2.CAP_PROP_FOURCC: fourcc,
            }.get(prop, 0)

        cap.get.side_effect = mock_get
        mock_cv2_cap.return_value = cap

        video_path = tmp_path / "test.mp4"
        video_path.touch()

        reader = VideoReader(video_path)

        assert reader.codec == "avc1"
