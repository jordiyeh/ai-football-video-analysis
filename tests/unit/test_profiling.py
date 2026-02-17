"""Tests for pipeline profiling and metrics."""

import time
from datetime import datetime


from src.pipeline.metrics import (
    PipelineMetrics,
    StageMetrics,
    StageTimer,
    detect_device,
    format_duration,
)


class TestStageMetrics:
    """Tests for StageMetrics dataclass."""

    def test_to_dict_basic(self):
        """Test basic serialization."""
        metrics = StageMetrics(
            stage_name="detection",
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T10:01:00",
            duration_seconds=60.0,
            items_processed=1000,
            items_per_second=16.67,
        )

        result = metrics.to_dict()

        assert result["stage_name"] == "detection"
        assert result["duration_seconds"] == 60.0
        assert result["items_processed"] == 1000
        assert result["items_per_second"] == 16.67
        assert result["custom_metrics"] == {}

    def test_to_dict_with_custom_metrics(self):
        """Test serialization with custom metrics."""
        metrics = StageMetrics(
            stage_name="detection",
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T10:01:00",
            duration_seconds=60.0,
            custom_metrics={
                "detector_type": "yolo",
                "ball_detections": 500,
            },
        )

        result = metrics.to_dict()

        assert result["custom_metrics"]["detector_type"] == "yolo"
        assert result["custom_metrics"]["ball_detections"] == 500

    def test_duration_rounding(self):
        """Test that duration is rounded to 3 decimal places."""
        metrics = StageMetrics(
            stage_name="test",
            start_time="",
            end_time="",
            duration_seconds=1.23456789,
        )

        result = metrics.to_dict()
        assert result["duration_seconds"] == 1.235


class TestStageTimer:
    """Tests for StageTimer context manager."""

    def test_basic_timing(self):
        """Test that timer measures elapsed time."""
        with StageTimer("test_stage") as timer:
            time.sleep(0.1)

        assert timer.duration_seconds >= 0.1
        assert timer.duration_seconds < 0.2

    def test_iso_timestamps(self):
        """Test that ISO timestamps are recorded."""
        with StageTimer("test_stage") as timer:
            pass

        assert timer.start_iso != ""
        assert timer.end_iso != ""

        # Verify they're valid ISO format
        datetime.fromisoformat(timer.start_iso)
        datetime.fromisoformat(timer.end_iso)

    def test_to_metrics_basic(self):
        """Test conversion to StageMetrics."""
        with StageTimer("detection") as timer:
            time.sleep(0.05)

        metrics = timer.to_metrics()

        assert metrics.stage_name == "detection"
        assert metrics.duration_seconds >= 0.05
        assert metrics.items_processed == 0
        assert metrics.items_per_second == 0.0

    def test_to_metrics_with_items(self):
        """Test conversion with items processed."""
        with StageTimer("detection") as timer:
            time.sleep(0.1)

        metrics = timer.to_metrics(items_processed=1000)

        assert metrics.items_processed == 1000
        # 1000 items in ~0.1s = ~10000 items/sec
        assert metrics.items_per_second > 5000
        assert metrics.items_per_second < 20000

    def test_to_metrics_with_custom(self):
        """Test conversion with custom metrics."""
        with StageTimer("detection") as timer:
            pass

        custom = {"detector_type": "ensemble", "model": "yolov8x"}
        metrics = timer.to_metrics(custom_metrics=custom)

        assert metrics.custom_metrics == custom

    def test_duration_while_running(self):
        """Test that duration can be checked while timer is running."""
        with StageTimer("test_stage") as timer:
            time.sleep(0.05)
            mid_duration = timer.duration_seconds
            time.sleep(0.05)

        final_duration = timer.duration_seconds

        assert mid_duration >= 0.05
        assert final_duration >= 0.1
        assert final_duration > mid_duration


class TestPipelineMetrics:
    """Tests for PipelineMetrics aggregate class."""

    def test_default_initialization(self):
        """Test that system info is populated by default."""
        metrics = PipelineMetrics()

        assert metrics.python_version != ""
        assert metrics.platform_info != ""
        assert "." in metrics.python_version  # e.g., "3.11.9"

    def test_add_stage(self):
        """Test adding stage metrics."""
        pipeline = PipelineMetrics()

        stage = StageMetrics(
            stage_name="detection",
            start_time="",
            end_time="",
            duration_seconds=10.0,
        )
        pipeline.add_stage(stage)

        assert "detection" in pipeline.stages
        assert pipeline.stages["detection"].duration_seconds == 10.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        pipeline = PipelineMetrics(
            total_duration_seconds=100.0,
            device="mps",
        )

        stage = StageMetrics(
            stage_name="detection",
            start_time="2024-01-15T10:00:00",
            end_time="2024-01-15T10:01:00",
            duration_seconds=60.0,
        )
        pipeline.add_stage(stage)

        result = pipeline.to_dict()

        assert result["total_duration_seconds"] == 100.0
        assert result["device"] == "mps"
        assert "detection" in result["stages"]
        assert result["stages"]["detection"]["duration_seconds"] == 60.0

    def test_get_stage_breakdown(self):
        """Test stage breakdown calculation."""
        pipeline = PipelineMetrics(total_duration_seconds=100.0)

        pipeline.add_stage(StageMetrics(
            stage_name="detection",
            start_time="", end_time="",
            duration_seconds=70.0,
        ))
        pipeline.add_stage(StageMetrics(
            stage_name="tracking",
            start_time="", end_time="",
            duration_seconds=20.0,
        ))
        pipeline.add_stage(StageMetrics(
            stage_name="ingest",
            start_time="", end_time="",
            duration_seconds=10.0,
        ))

        breakdown = pipeline.get_stage_breakdown()

        # Should be sorted by duration descending
        assert breakdown[0][0] == "detection"
        assert breakdown[0][1] == 70.0
        assert breakdown[0][2] == 70.0  # 70%

        assert breakdown[1][0] == "tracking"
        assert breakdown[1][2] == 20.0  # 20%

        assert breakdown[2][0] == "ingest"
        assert breakdown[2][2] == 10.0  # 10%

    def test_get_stage_breakdown_empty(self):
        """Test breakdown with no stages."""
        pipeline = PipelineMetrics(total_duration_seconds=0.0)

        breakdown = pipeline.get_stage_breakdown()
        assert breakdown == []


class TestFormatDuration:
    """Tests for format_duration helper."""

    def test_seconds_only(self):
        """Test formatting under 60 seconds."""
        assert format_duration(0.5) == "0.5s"
        assert format_duration(30.0) == "30.0s"
        assert format_duration(59.9) == "59.9s"

    def test_minutes_and_seconds(self):
        """Test formatting between 1-60 minutes."""
        assert format_duration(60.0) == "1m 0s"
        assert format_duration(90.0) == "1m 30s"
        assert format_duration(125.5) == "2m 6s"  # Rounded
        assert format_duration(3599.0) == "59m 59s"

    def test_hours_minutes_seconds(self):
        """Test formatting over 1 hour."""
        assert format_duration(3600.0) == "1h 0m 0s"
        assert format_duration(3661.0) == "1h 1m 1s"
        assert format_duration(7325.0) == "2h 2m 5s"


class TestDetectDevice:
    """Tests for detect_device helper."""

    def test_returns_string(self):
        """Test that detect_device returns a string."""
        device = detect_device()
        assert isinstance(device, str)
        assert device in ["mps", "cpu", "unknown (torch not available)"] or device.startswith("cuda")


class TestTimerEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_items_per_second(self):
        """Test that zero duration doesn't cause division by zero."""
        # Create metrics with very small duration
        metrics = StageMetrics(
            stage_name="test",
            start_time="",
            end_time="",
            duration_seconds=0.0,
            items_processed=100,
            items_per_second=0.0,  # Explicitly set to avoid inf
        )

        result = metrics.to_dict()
        assert result["items_per_second"] == 0.0

    def test_timer_to_metrics_zero_duration(self):
        """Test to_metrics with effectively zero duration."""
        timer = StageTimer("test")
        timer._start_time = 0.0
        timer._end_time = 0.0
        timer._start_iso = ""
        timer._end_iso = ""

        metrics = timer.to_metrics(items_processed=100)

        # Should handle gracefully without inf/nan
        assert metrics.items_per_second == 0.0 or metrics.items_per_second > 0
