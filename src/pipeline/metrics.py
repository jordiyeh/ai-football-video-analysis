"""Pipeline performance metrics and timing infrastructure."""

import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""

    stage_name: str
    start_time: str  # ISO format
    end_time: str  # ISO format
    duration_seconds: float
    items_processed: int = 0
    items_per_second: float = 0.0
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage_name": self.stage_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": round(self.duration_seconds, 3),
            "items_processed": self.items_processed,
            "items_per_second": round(self.items_per_second, 2),
            "custom_metrics": self.custom_metrics,
        }


@dataclass
class PipelineMetrics:
    """Aggregate metrics for the entire pipeline run."""

    total_duration_seconds: float = 0.0
    stages: dict[str, StageMetrics] = field(default_factory=dict)
    device: str = ""
    python_version: str = ""
    platform_info: str = ""

    def __post_init__(self):
        """Initialize system info if not provided."""
        if not self.python_version:
            self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if not self.platform_info:
            self.platform_info = f"{platform.system()} {platform.release()}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "stages": {
                name: metrics.to_dict() for name, metrics in self.stages.items()
            },
            "device": self.device,
            "python_version": self.python_version,
            "platform_info": self.platform_info,
        }

    def add_stage(self, metrics: StageMetrics) -> None:
        """Add stage metrics."""
        self.stages[metrics.stage_name] = metrics

    def get_stage_breakdown(self) -> list[tuple[str, float, float]]:
        """
        Get breakdown of time spent in each stage.

        Returns:
            List of (stage_name, duration_seconds, percentage) tuples
        """
        if self.total_duration_seconds == 0:
            return []

        breakdown = []
        for name, metrics in self.stages.items():
            pct = (metrics.duration_seconds / self.total_duration_seconds) * 100
            breakdown.append((name, metrics.duration_seconds, pct))

        return sorted(breakdown, key=lambda x: x[1], reverse=True)


class StageTimer:
    """Context manager for timing stage execution."""

    def __init__(self, stage_name: str):
        """
        Initialize timer.

        Args:
            stage_name: Name of the stage being timed
        """
        self.stage_name = stage_name
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._start_iso: str = ""
        self._end_iso: str = ""

    def __enter__(self) -> "StageTimer":
        """Start timing."""
        self._start_time = time.perf_counter()
        self._start_iso = datetime.now().isoformat()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing."""
        self._end_time = time.perf_counter()
        self._end_iso = datetime.now().isoformat()

    @property
    def duration_seconds(self) -> float:
        """Get elapsed duration in seconds."""
        if self._end_time == 0:
            # Timer still running
            return time.perf_counter() - self._start_time
        return self._end_time - self._start_time

    @property
    def start_iso(self) -> str:
        """Get start time in ISO format."""
        return self._start_iso

    @property
    def end_iso(self) -> str:
        """Get end time in ISO format."""
        return self._end_iso

    def to_metrics(
        self,
        items_processed: int = 0,
        custom_metrics: dict[str, Any] | None = None,
    ) -> StageMetrics:
        """
        Convert timer to StageMetrics.

        Args:
            items_processed: Number of items processed during this stage
            custom_metrics: Additional metrics specific to this stage

        Returns:
            StageMetrics object
        """
        duration = self.duration_seconds
        items_per_second = items_processed / duration if duration > 0 else 0.0

        return StageMetrics(
            stage_name=self.stage_name,
            start_time=self._start_iso,
            end_time=self._end_iso,
            duration_seconds=duration,
            items_processed=items_processed,
            items_per_second=items_per_second,
            custom_metrics=custom_metrics or {},
        )


def detect_device() -> str:
    """
    Detect the compute device being used.

    Returns:
        Device string (e.g., "mps", "cuda", "cpu")
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
        else:
            return "cpu"
    except ImportError:
        return "unknown (torch not available)"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable form.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2h 30m 15s" or "45.2s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"
