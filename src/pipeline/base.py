"""Pipeline orchestration and stage management."""

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.config.schemas import PipelineConfig


class PipelineStage(ABC):
    """Base class for pipeline stages."""

    def __init__(self, name: str, config: PipelineConfig):
        """
        Initialize pipeline stage.

        Args:
            name: Stage name
            config: Pipeline configuration
        """
        self.name = name
        self.config = config
        self.console = Console()

    @abstractmethod
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute stage logic.

        Args:
            context: Pipeline context with results from previous stages

        Returns:
            Updated context with this stage's results
        """
        pass

    def get_cache_key(self, inputs: dict[str, Any]) -> str:
        """
        Generate cache key from inputs.

        Args:
            inputs: Input data to hash

        Returns:
            Cache key (hex string)
        """
        # Create deterministic JSON string
        input_str = json.dumps(inputs, sort_keys=True)
        return hashlib.sha256(input_str.encode()).hexdigest()[:16]

    def get_cache_path(self, cache_key: str, suffix: str = ".json") -> Path:
        """
        Get path to cached result.

        Args:
            cache_key: Cache key
            suffix: File suffix

        Returns:
            Path to cache file
        """
        cache_dir = Path(self.config.cache_dir) / self.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{cache_key}{suffix}"

    def load_from_cache(self, cache_key: str) -> Any | None:
        """
        Load result from cache if available.

        Args:
            cache_key: Cache key

        Returns:
            Cached result or None if not found
        """
        if not self.config.enable_cache:
            return None

        cache_path = self.get_cache_path(cache_key)
        if cache_path.exists():
            self.console.print(f"[yellow]Loading {self.name} from cache...[/yellow]")
            with open(cache_path, "r") as f:
                return json.load(f)
        return None

    def save_to_cache(self, cache_key: str, result: Any) -> None:
        """
        Save result to cache.

        Args:
            cache_key: Cache key
            result: Result to cache
        """
        if not self.config.enable_cache:
            return

        cache_path = self.get_cache_path(cache_key)
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)


class Pipeline:
    """Pipeline orchestrator."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.console = Console()
        self.stages: list[PipelineStage] = []

    def add_stage(self, stage: PipelineStage) -> None:
        """
        Add a stage to the pipeline.

        Args:
            stage: Stage to add
        """
        self.stages.append(stage)

    def run(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        resume: bool = False,
    ) -> dict[str, Any]:
        """
        Run the full pipeline.

        Args:
            video_path: Path to input video
            output_dir: Directory for outputs
            resume: Resume from existing outputs (skip completed stages)

        Returns:
            Pipeline results
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.console.print(f"\n[bold green]Starting pipeline for: {video_path.name}[/bold green]\n")

        if resume:
            self.console.print("[yellow]Resume mode enabled - using cached outputs where available[/yellow]\n")

        # Initialize context
        context = {
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "start_time": datetime.now().isoformat(),
            "resume": resume,
        }

        # Run stages sequentially
        for i, stage in enumerate(self.stages, 1):
            self.console.print(
                f"[bold cyan]Stage {i}/{len(self.stages)}: {stage.name}[/bold cyan]"
            )

            try:
                context = stage.run(context)
            except Exception as e:
                self.console.print(f"[bold red]Error in stage {stage.name}: {e}[/bold red]")
                raise

        # Save run manifest
        context["end_time"] = datetime.now().isoformat()
        manifest_path = output_dir / "run_manifest.json"
        self._save_manifest(context, manifest_path)

        self.console.print(f"\n[bold green]Pipeline complete! Output: {output_dir}[/bold green]\n")

        return context

    def _save_manifest(self, context: dict[str, Any], path: Path) -> None:
        """
        Save run manifest.

        Args:
            context: Pipeline context
            path: Output path
        """
        manifest = {
            "schema_version": "1.0",
            "video_path": context["video_path"],
            "output_dir": context["output_dir"],
            "start_time": context["start_time"],
            "end_time": context["end_time"],
            "config": self.config.model_dump(),
            "stages": [stage.name for stage in self.stages],
        }

        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)


def save_detections_to_parquet(
    detections: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save detections to Parquet file.

    Args:
        detections: List of detection dictionaries
        output_path: Output file path
    """
    df = pd.DataFrame(detections)
    df.to_parquet(output_path, index=False)


def save_detections_to_jsonl(
    detections: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save detections to JSONL file.

    Args:
        detections: List of detection dictionaries
        output_path: Output file path
    """
    with open(output_path, "w") as f:
        for detection in detections:
            f.write(json.dumps(detection) + "\n")
