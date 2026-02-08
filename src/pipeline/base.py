"""Pipeline orchestration and stage management."""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console

from src.config.schemas import PipelineConfig
from src.pipeline.metrics import (
    PipelineMetrics,
    StageTimer,
    detect_device,
    format_duration,
)


class PipelineCancelledError(Exception):
    """Raised when a pipeline stage detects that cancellation has been requested."""


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

    def check_cancelled(self, context: dict[str, Any]) -> None:
        """Check if cancellation has been requested and raise if so.

        Stages should call this inside long-running loops. When no callback
        is present in context (e.g. CLI mode), this is a no-op.
        """
        callback = context.get("_check_cancel")
        if callback is not None and callback():
            raise PipelineCancelledError("Pipeline cancelled by user")

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
        self.metrics = PipelineMetrics(device=detect_device())

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

        # Track pipeline start time
        pipeline_start = time.perf_counter()

        # Run stages sequentially with timing
        for i, stage in enumerate(self.stages, 1):
            self.console.print(
                f"[bold cyan]Stage {i}/{len(self.stages)}: {stage.name}[/bold cyan]"
            )

            try:
                with StageTimer(stage.name) as timer:
                    context = stage.run(context)

                # Get stage metrics from context
                items_key = f"{stage.name}_items_processed"
                custom_key = f"{stage.name}_custom_metrics"
                items_processed = context.get(items_key, 0)
                custom_metrics = context.get(custom_key, {})

                stage_metrics = timer.to_metrics(items_processed, custom_metrics)
                self.metrics.add_stage(stage_metrics)

                duration_str = format_duration(stage_metrics.duration_seconds)
                if items_processed > 0:
                    self.console.print(
                        f"  [green]Completed in {duration_str} "
                        f"({stage_metrics.items_per_second:.1f} items/sec)[/green]"
                    )
                else:
                    self.console.print(f"  [green]Completed in {duration_str}[/green]")

            except Exception as e:
                self.console.print(f"[bold red]Error in stage {stage.name}: {e}[/bold red]")
                raise

        # Calculate total duration
        self.metrics.total_duration_seconds = time.perf_counter() - pipeline_start

        # Save run manifest
        context["end_time"] = datetime.now().isoformat()
        manifest_path = output_dir / "run_manifest.json"
        self._save_manifest(context, manifest_path)
        summary_path = output_dir / "summary.json"
        self._save_summary(context, output_dir, summary_path)
        ui_index_path = output_dir / "ui_index.json"
        self._save_ui_index(context, output_dir, ui_index_path)

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
            "schema_version": "1.1",
            "video_path": context["video_path"],
            "original_video_path": context.get("original_video_path", context["video_path"]),
            "output_dir": context["output_dir"],
            "start_time": context["start_time"],
            "end_time": context["end_time"],
            "config": self.config.model_dump(),
            "stages": [stage.name for stage in self.stages],
            "metrics": self.metrics.to_dict(),
        }

        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Print timing summary
        self._print_timing_summary()

    def _collect_artifact_index(self, output_dir: Path) -> dict[str, str]:
        """
        Collect available run artifacts.

        Args:
            output_dir: Run output directory

        Returns:
            Mapping of artifact keys to run-relative paths
        """
        artifact_candidates: dict[str, list[str]] = {
            "run_manifest": ["run_manifest.json"],
            "video_metadata": ["video_metadata.json"],
            "detections": ["detections.parquet", "detections.jsonl", "detections.csv"],
            "tracks": ["tracks.parquet", "tracks.jsonl", "tracks.csv"],
            "teams": ["teams.json"],
            "field_normalization": ["field_normalization.json"],
            "field_viewports": ["field_viewports.parquet"],
            "team_analytics": ["team_analytics.json"],
            "player_analytics": ["player_analytics.json"],
            "match_stats": ["match_stats.json"],
            "team_possession_timeline": ["team_possession_timeline.csv"],
            "team_pass_network": ["team_pass_network.csv"],
            "team_pressing_timeline": ["team_pressing_timeline.csv"],
            "team_territory_zones": ["team_territory_zones.csv"],
            "cross_match_report": ["cross_match_report.json"],
            "cross_match_match_trends": ["cross_match_match_trends.csv"],
            "cross_match_player_trends": ["cross_match_player_trends.csv"],
            "coach_report_template": ["coach_report_template.md"],
            "player_report_templates": ["player_report_templates.md"],
            "profile_registry": ["profile_registry.json"],
            "profile_embeddings": ["profile_embeddings.parquet"],
            "profile_player_links": ["profile_player_links.json"],
            "player_assignments": ["player_assignments.json"],
            "events": ["events.jsonl"],
            "score_timeline": ["score_timeline.json"],
            "highlight_candidates": ["highlight_candidates.jsonl"],
            "highlights": ["highlights.json"],
            "highlights_csv": ["highlights.csv"],
            "highlights_manifest": ["highlights_manifest.json"],
            "player_highlights": ["player_highlights.json"],
            "player_highlights_csv": ["player_highlights.csv"],
            "player_highlights_manifest": ["player_highlights_manifest.json"],
            "clips_dir": ["clips"],
            "player_clips_dir": ["player_clips"],
            "overlay": ["overlay.mp4"],
            "summary": ["summary.json"],
            "ui_index": ["ui_index.json"],
        }

        artifact_index: dict[str, str] = {}
        for key, candidates in artifact_candidates.items():
            for candidate in candidates:
                candidate_path = output_dir / candidate
                if candidate_path.exists():
                    artifact_index[key] = candidate
                    break

        return artifact_index

    def _safe_len(self, value: Any) -> int:
        """
        Return len(value) if possible, otherwise 0.
        """
        if value is None:
            return 0
        try:
            return len(value)
        except Exception:
            return 0

    def _event_type_counts(self, events: Any) -> dict[str, int]:
        """Return per-event-type counts for event dicts/dataclass-like objects."""
        if not isinstance(events, list):
            return {}

        counts: dict[str, int] = {}
        for event in events:
            event_type: str | None = None
            if isinstance(event, dict):
                raw_type = event.get("event_type")
                if raw_type is not None:
                    event_type = str(raw_type)
            else:
                raw_type = getattr(event, "event_type", None)
                if raw_type is not None:
                    event_type = str(raw_type)

            if not event_type:
                continue
            counts[event_type] = counts.get(event_type, 0) + 1

        return counts

    def _save_summary(
        self,
        context: dict[str, Any],
        output_dir: Path,
        path: Path,
    ) -> None:
        """
        Save aggregated run summary.

        Args:
            context: Pipeline context
            output_dir: Run output directory
            path: Summary output path
        """
        detection_metrics = context.get("detection_custom_metrics", {})
        tracking_metrics = context.get("tracking_custom_metrics", {})
        event_metrics = context.get("event_detection_custom_metrics", {})
        highlight_metrics = context.get("highlight_generation_custom_metrics", {})
        player_reel_metrics = context.get("player_highlight_reels_custom_metrics", {})
        player_analytics_metrics = context.get("player_analytics_custom_metrics", {})
        team_metrics = context.get("team_analytics_custom_metrics", {})
        match_stats_metrics = context.get("match_stats_custom_metrics", {})
        cross_match_metrics = context.get("cross_match_reporting_custom_metrics", {})
        event_type_counts = self._event_type_counts(context.get("events"))
        set_piece_count = sum(
            event_type_counts.get(event_type, 0)
            for event_type in (
                "set_piece",
                "kickoff",
                "throw_in",
                "corner_kick",
                "free_kick",
                "goal_kick",
            )
        )
        tactical_count = sum(
            event_type_counts.get(event_type, 0)
            for event_type in (
                "build_up",
                "pressing",
                "defending",
                "transition",
            )
        )

        video_metadata = context.get("video_metadata", {})
        timeline_path = output_dir / "score_timeline.json"
        final_score = None
        timeline_goals = None
        if timeline_path.exists():
            try:
                with open(timeline_path) as f:
                    timeline_data = json.load(f)
                final_score = timeline_data.get("final_score")
                timeline_goals = timeline_data.get("goals")
            except Exception:
                final_score = None
                timeline_goals = None

        artifacts = self._collect_artifact_index(output_dir)
        artifacts["summary"] = "summary.json"
        artifacts["ui_index"] = "ui_index.json"

        summary_data = {
            "schema_version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "run_name": output_dir.name,
            "video_id": Path(str(context.get("video_path", output_dir.name))).stem,
            "video": {
                "input_path": context.get("video_path"),
                "original_path": context.get("original_video_path", context.get("video_path")),
                "fps": video_metadata.get("fps"),
                "duration_seconds": video_metadata.get("duration"),
                "total_frames": video_metadata.get("total_frames"),
                "resolution": {
                    "width": video_metadata.get("width"),
                    "height": video_metadata.get("height"),
                },
            },
            "counts": {
                "detections_total": detection_metrics.get(
                    "total_detections",
                    self._safe_len(context.get("detections")),
                ),
                "player_detections": detection_metrics.get("player_detections"),
                "ball_detections": detection_metrics.get("ball_detections"),
                "track_points": tracking_metrics.get(
                    "track_points",
                    self._safe_len(context.get("tracks")),
                ),
                "tracks_unique": tracking_metrics.get("unique_tracks"),
                "events_total": self._safe_len(context.get("events")),
                "shots": event_metrics.get("shots"),
                "goals": event_metrics.get("goals", timeline_goals),
                "passes": event_metrics.get("passes", event_type_counts.get("pass")),
                "set_pieces": event_metrics.get("set_pieces", set_piece_count),
                "tactical_events": event_metrics.get("tactical_events", tactical_count),
                "build_ups": event_metrics.get("tactical_build_ups", event_type_counts.get("build_up")),
                "pressing_events": event_metrics.get("tactical_pressing", event_type_counts.get("pressing")),
                "defending_events": event_metrics.get("tactical_defending", event_type_counts.get("defending")),
                "transition_events": event_metrics.get("tactical_transitions", event_type_counts.get("transition")),
                "kickoffs": event_metrics.get("kickoffs", event_type_counts.get("kickoff")),
                "throw_ins": event_metrics.get("throw_ins", event_type_counts.get("throw_in")),
                "corner_kicks": event_metrics.get("corner_kicks", event_type_counts.get("corner_kick")),
                "free_kicks": event_metrics.get("free_kicks", event_type_counts.get("free_kick")),
                "goal_kicks": event_metrics.get("goal_kicks", event_type_counts.get("goal_kick")),
                "highlights_segments": highlight_metrics.get("segments_selected"),
                "highlight_candidates": highlight_metrics.get("candidates"),
                "players_with_reels": player_reel_metrics.get("players_with_reels"),
                "player_reel_segments_total": player_reel_metrics.get("player_segments_total"),
                "player_analytics_players": player_analytics_metrics.get("players_detected"),
                "player_analytics_runs": player_analytics_metrics.get("runs_analyzed"),
                "player_analytics_events_total": player_analytics_metrics.get("events_total"),
                "player_analytics_sprints_total": player_analytics_metrics.get("sprints_total"),
                "possession_frames": team_metrics.get("frames_with_possession"),
                "passes_inferred": team_metrics.get("passes_inferred"),
                "cross_match_matches": cross_match_metrics.get("matches_analyzed"),
                "cross_match_players": cross_match_metrics.get("unique_players"),
                "match_stats_teams": match_stats_metrics.get("teams_detected"),
            },
            "score": {
                "final_score": final_score,
                "timeline_entries": self._safe_len(context.get("score_timeline")),
            },
            "stages": [stage.name for stage in self.stages],
            "timing": self.metrics.to_dict(),
            "artifacts": artifacts,
        }

        with open(path, "w") as f:
            json.dump(summary_data, f, indent=2)

        self.console.print(f"Saved summary to: {path}")

    def _save_ui_index(
        self,
        context: dict[str, Any],
        output_dir: Path,
        path: Path,
    ) -> None:
        """
        Save compact UI run index.

        Args:
            context: Pipeline context
            output_dir: Run output directory
            path: UI index output path
        """
        summary_path = output_dir / "summary.json"
        summary = {}
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
            except Exception:
                summary = {}

        artifacts = self._collect_artifact_index(output_dir)
        artifacts["ui_index"] = "ui_index.json"
        counts = summary.get("counts", {})

        ui_index = {
            "schema_version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "run_name": output_dir.name,
            "video_id": Path(str(context.get("video_path", output_dir.name))).stem,
            "artifacts": artifacts,
            "summary_path": artifacts.get("summary", "summary.json"),
            "quicklook": {
                "duration_seconds": summary.get("video", {}).get("duration_seconds"),
                "fps": summary.get("video", {}).get("fps"),
                "events_total": counts.get("events_total"),
                "shots": counts.get("shots"),
                "goals": counts.get("goals"),
                "highlights_segments": counts.get("highlights_segments"),
                "players_with_reels": counts.get("players_with_reels"),
            },
            "capabilities": {
                "has_overlay_video": "overlay" in artifacts,
                "has_events": "events" in artifacts,
                "has_score_timeline": "score_timeline" in artifacts,
                "has_highlights": "highlights" in artifacts,
                "has_player_reels": "player_highlights" in artifacts,
                "has_player_analytics": "player_analytics" in artifacts,
                "has_team_analytics": "team_analytics" in artifacts,
                "has_match_stats": "match_stats" in artifacts,
                "has_cross_match_report": "cross_match_report" in artifacts,
            },
            "preferred_video_artifact": "overlay.mp4"
            if "overlay" in artifacts else None,
        }

        with open(path, "w") as f:
            json.dump(ui_index, f, indent=2)

        self.console.print(f"Saved UI index to: {path}")

    def _print_timing_summary(self) -> None:
        """Print a summary of timing metrics."""
        total_str = format_duration(self.metrics.total_duration_seconds)
        self.console.print(f"\n[bold]Timing Summary[/bold] (total: {total_str})")

        breakdown = self.metrics.get_stage_breakdown()
        for stage_name, duration, pct in breakdown:
            duration_str = format_duration(duration)
            bar_len = int(pct / 5)  # Max ~20 chars for 100%
            bar = "[cyan]" + "=" * bar_len + "[/cyan]"
            self.console.print(f"  {stage_name:15} {bar} {duration_str} ({pct:.1f}%)")


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
