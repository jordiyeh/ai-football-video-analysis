"""Command-line interface for soccer video analysis."""

import json
from pathlib import Path
from typing import Any

import click
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from src.config.schemas import PipelineConfig
from src.export.overlay import OverlayRenderer, VideoWriter
from src.pipeline.base import Pipeline, PipelineStage, save_detections_to_parquet
from src.video.reader import VideoReader
from src.vision.detect.yolo import YOLODetector


class IngestStage(PipelineStage):
    """Stage A: Video ingestion and validation."""

    def __init__(self, config: PipelineConfig):
        super().__init__("ingest", config)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Extract video metadata and validate."""
        video_path = Path(context["video_path"])
        output_dir = Path(context["output_dir"])
        metadata_path = output_dir / "video_metadata.json"

        # Check for cached metadata
        if context.get("resume", False) and metadata_path.exists():
            self.console.print(f"[bold yellow]✓ Using cached video metadata from {metadata_path.name}[/bold yellow]")

            with open(metadata_path) as f:
                video_metadata = json.load(f)

            self.console.print(f"  {video_path.name} - {video_metadata['duration']:.2f}s @ {video_metadata['fps']:.2f}fps (skipped ingest stage)")
            context["video_metadata"] = video_metadata
            return context
        else:
            if context.get("resume", False):
                self.console.print(f"[dim]No cache found at {metadata_path.name}, reading video...[/dim]")

        with VideoReader(video_path) as reader:
            metadata = reader.metadata

            self.console.print(f"Video: {video_path.name}")
            self.console.print(f"  Duration: {metadata.duration:.2f}s")
            self.console.print(f"  FPS: {metadata.fps:.2f}")
            self.console.print(f"  Resolution: {metadata.width}x{metadata.height}")
            self.console.print(f"  Total frames: {metadata.total_frames}")

            # Save metadata
            metadata.save(metadata_path)

            context["video_metadata"] = metadata.to_dict()

        return context


class DetectionStage(PipelineStage):
    """Stage B: Detect players and ball."""

    def __init__(self, config: PipelineConfig):
        super().__init__("detection", config)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run detection on all frames."""
        video_path = Path(context["video_path"])
        output_dir = Path(context["output_dir"])

        # Check for cached detections
        if context.get("resume", False):
            if self.config.export.detections_format == "parquet":
                cache_path = output_dir / "detections.parquet"
            else:
                cache_path = output_dir / "detections.jsonl"

            if cache_path.exists():
                self.console.print(f"[bold yellow]✓ Using cached detections from {cache_path.name}[/bold yellow]")
                import pandas as pd

                if cache_path.suffix == ".parquet":
                    df = pd.read_parquet(cache_path)
                    all_detections = df.to_dict(orient="records")
                else:
                    all_detections = []
                    with open(cache_path) as f:
                        for line in f:
                            all_detections.append(json.loads(line))

                self.console.print(f"  Loaded {len(all_detections)} detections (skipped detection stage)")
                context["detections"] = all_detections
                return context
            else:
                self.console.print(f"[dim]No cache found at {cache_path.name}, running detection...[/dim]")

        # Initialize detector
        detector = YOLODetector(
            model_name=self.config.detection.model_name,
            device=self.config.detection.device,
            confidence_threshold=self.config.detection.confidence_threshold,
        )

        # Process video
        all_detections = []

        with VideoReader(video_path) as reader:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console,
            ) as progress:
                total_frames = reader.total_frames
                task = progress.add_task("Detecting players and ball...", total=total_frames)

                for frame_idx, frame in reader.frames(
                    sampling_strategy=self.config.video.sampling_strategy,
                    sampling_interval=self.config.video.sampling_interval,
                ):
                    # Run detection
                    detections = detector.detect(frame)

                    # Store detections
                    for detection in detections:
                        det_dict = detection.to_dict()
                        det_dict["frame_idx"] = frame_idx
                        det_dict["timestamp"] = frame_idx / reader.fps
                        all_detections.append(det_dict)

                    progress.update(task, advance=1)

        self.console.print(f"Total detections: {len(all_detections)}")

        # Save detections
        if self.config.export.save_detections:
            if self.config.export.detections_format == "parquet":
                output_path = output_dir / "detections.parquet"
                save_detections_to_parquet(all_detections, output_path)
            elif self.config.export.detections_format == "jsonl":
                output_path = output_dir / "detections.jsonl"
                with open(output_path, "w") as f:
                    for det in all_detections:
                        f.write(json.dumps(det) + "\n")

            self.console.print(f"Saved detections to: {output_path}")

        context["detections"] = all_detections
        return context


class TrackingStage(PipelineStage):
    """Stage C: Multi-object tracking."""

    def __init__(self, config: PipelineConfig):
        super().__init__("tracking", config)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run tracking on detections."""
        from src.vision.track import ByteTracker

        output_dir = Path(context["output_dir"])

        # Check for cached tracks
        if context.get("resume", False):
            if self.config.export.detections_format == "parquet":
                cache_path = output_dir / "tracks.parquet"
            else:
                cache_path = output_dir / "tracks.jsonl"

            if cache_path.exists():
                self.console.print(f"[bold yellow]✓ Using cached tracks from {cache_path.name}[/bold yellow]")
                import pandas as pd

                if cache_path.suffix == ".parquet":
                    df = pd.read_parquet(cache_path)
                    all_tracks = df.to_dict(orient="records")
                else:
                    all_tracks = []
                    with open(cache_path) as f:
                        for line in f:
                            all_tracks.append(json.loads(line))

                self.console.print(f"  Loaded {len(all_tracks)} track points (skipped tracking stage)")
                context["tracks"] = all_tracks
                return context
            else:
                self.console.print(f"[dim]No cache found at {cache_path.name}, running tracking...[/dim]")

        detections = context.get("detections", [])
        video_metadata = context["video_metadata"]
        fps = video_metadata["fps"]

        # Initialize tracker
        tracker = ByteTracker(
            track_thresh=0.5,  # Use confidence threshold from detections
            track_buffer=self.config.tracking.max_age,
            match_thresh=self.config.tracking.iou_threshold,
            min_hits=self.config.tracking.min_hits,
        )

        # Group detections by frame
        detections_by_frame = {}
        for det in detections:
            frame_idx = det["frame_idx"]
            if frame_idx not in detections_by_frame:
                detections_by_frame[frame_idx] = []
            detections_by_frame[frame_idx].append(det)

        # Run tracking frame by frame
        all_tracks = []
        frame_indices = sorted(detections_by_frame.keys())

        self.console.print(f"Tracking objects across {len(frame_indices)} frames...")

        for frame_idx in frame_indices:
            frame_dets = detections_by_frame[frame_idx]

            # Convert to tracker format
            tracker_dets = [
                {
                    "bbox": tuple(d["bbox"]),
                    "confidence": d["confidence"],
                    "object_type": d["object_type"],
                }
                for d in frame_dets
            ]

            # Update tracker
            tracks = tracker.update(tracker_dets)

            # Store track results
            timestamp = frame_idx / fps
            for track in tracks:
                track_dict = {
                    "track_id": track.track_id,
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "object_type": track.object_type,
                    "bbox": list(track.bbox),
                    "confidence": track.confidence,
                    "age": track.age,
                    "hits": track.hits,
                    "time_since_update": track.time_since_update,
                }
                all_tracks.append(track_dict)

        self.console.print(f"Total tracks: {len(set(t['track_id'] for t in all_tracks))}")
        self.console.print(f"Total track points: {len(all_tracks)}")

        # Save tracks
        if self.config.export.save_tracks:
            if self.config.export.detections_format == "parquet":
                output_path = output_dir / "tracks.parquet"
                save_detections_to_parquet(all_tracks, output_path)
            elif self.config.export.detections_format == "jsonl":
                output_path = output_dir / "tracks.jsonl"
                with open(output_path, "w") as f:
                    for track in all_tracks:
                        f.write(json.dumps(track) + "\n")

            self.console.print(f"Saved tracks to: {output_path}")

        context["tracks"] = all_tracks
        return context


class TeamAssignmentStage(PipelineStage):
    """Stage D: Team identification from jersey colors."""

    def __init__(self, config: PipelineConfig):
        super().__init__("team_assignment", config)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Assign teams based on jersey colors."""
        from src.vision.team import TeamAssigner, extract_jersey_color, collect_track_colors

        output_dir = Path(context["output_dir"])

        # Check for cached team assignments
        if context.get("resume", False):
            teams_path = output_dir / "teams.json"
            if self.config.export.detections_format == "parquet":
                tracks_path = output_dir / "tracks.parquet"
            else:
                tracks_path = output_dir / "tracks.jsonl"

            if teams_path.exists() and tracks_path.exists():
                self.console.print(f"[bold yellow]✓ Using cached team assignments from {teams_path.name}[/bold yellow]")

                # Load teams info
                with open(teams_path) as f:
                    team_info = json.load(f)

                # Reload tracks (they should have team assignments already)
                import pandas as pd
                if tracks_path.suffix == ".parquet":
                    df = pd.read_parquet(tracks_path)
                    tracks = df.to_dict(orient="records")
                else:
                    tracks = []
                    with open(tracks_path) as f:
                        for line in f:
                            tracks.append(json.loads(line))

                context["tracks"] = tracks
                context["team_info"] = team_info
                self.console.print(f"  Loaded {team_info['n_teams']} teams (skipped team assignment stage)")
                return context
            else:
                self.console.print(f"[dim]No cache found, running team assignment...[/dim]")

        video_path = Path(context["video_path"])
        tracks = context.get("tracks", [])

        if len(tracks) == 0:
            self.console.print("No tracks available, skipping team assignment")
            return context

        # Group tracks by frame
        tracks_by_frame = {}
        for track in tracks:
            frame_idx = track["frame_idx"]
            if frame_idx not in tracks_by_frame:
                tracks_by_frame[frame_idx] = []
            tracks_by_frame[frame_idx].append(track)

        # Sample frames for color extraction (use every Nth frame to speed up)
        sample_interval = 30  # Sample every 30 frames
        sampled_frames = sorted(tracks_by_frame.keys())[::sample_interval]

        self.console.print(f"Extracting jersey colors from {len(sampled_frames)} frames...")

        # Load sampled frames and extract colors
        frames_dict = {}
        with VideoReader(video_path) as reader:
            for frame_idx in sampled_frames:
                frame = reader.get_frame_at(frame_idx)
                if frame is not None:
                    frames_dict[frame_idx] = frame

        # Collect track colors
        track_colors = collect_track_colors(
            {idx: tracks_by_frame[idx] for idx in sampled_frames if idx in tracks_by_frame},
            frames_dict,
            extract_fn=lambda f, b: extract_jersey_color(f, b, sample_region="upper"),
        )

        self.console.print(f"Collected colors from {len(track_colors)} tracks")

        # Cluster into teams
        n_teams = self.config.team.n_clusters
        try:
            assigner = TeamAssigner(n_teams=n_teams, color_space="hsv", min_samples_per_track=3)
            assigner.fit(track_colors)

            self.console.print(f"Identified {n_teams} teams")

            # Assign team names
            team_names = assigner.assign_team_names()

            # Add team assignments to tracks
            for track in tracks:
                track_id = track["track_id"]
                team_id = assigner.get_team_label(track_id)
                if team_id is not None:
                    track["team_id"] = team_id
                    track["team_name"] = team_names.get(team_id, f"team_{team_id}")
                else:
                    track["team_id"] = -1  # Unknown
                    track["team_name"] = "unknown"

            # Get team colors for visualization
            team_colors_bgr = assigner.get_team_colors_bgr()

            # Save team assignment info
            team_info = {
                "n_teams": n_teams,
                "team_colors": {
                    int(team_id): color.tolist()
                    for team_id, color in team_colors_bgr.items()
                },
                "team_names": team_names,
                "track_assignments": {
                    int(track_id): int(team_id)
                    for track_id, team_id in assigner.team_labels.items()
                },
            }

            teams_path = output_dir / "teams.json"
            with open(teams_path, "w") as f:
                json.dump(team_info, f, indent=2)

            self.console.print(f"Saved team assignments to: {teams_path}")

            # Store in context
            context["team_assigner"] = assigner
            context["team_info"] = team_info

            # Re-save tracks with team assignments
            if self.config.export.save_tracks:
                if self.config.export.detections_format == "parquet":
                    output_path = output_dir / "tracks.parquet"
                    save_detections_to_parquet(tracks, output_path)
                elif self.config.export.detections_format == "jsonl":
                    output_path = output_dir / "tracks.jsonl"
                    with open(output_path, "w") as f:
                        for track in tracks:
                            f.write(json.dumps(track) + "\n")

                self.console.print(f"Updated tracks with team assignments: {output_path}")

        except Exception as e:
            self.console.print(f"[yellow]Warning: Team assignment failed: {e}[/yellow]")
            self.console.print("Continuing without team assignments...")

        return context


class EventDetectionStage(PipelineStage):
    """Stage E: Detect shots and goals."""

    def __init__(self, config: PipelineConfig):
        super().__init__("event_detection", config)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Detect shot and goal events from ball trajectory."""
        from src.events import BallTrajectory, EventDetector

        output_dir = Path(context["output_dir"])

        # Check for cached events
        if context.get("resume", False):
            events_path = output_dir / "events.jsonl"
            timeline_path = output_dir / "score_timeline.json"

            if events_path.exists() and timeline_path.exists():
                self.console.print(f"[bold yellow]✓ Using cached events from {events_path.name}[/bold yellow]")

                # Load events
                events = []
                with open(events_path) as f:
                    for line in f:
                        events.append(json.loads(line))

                # Load timeline
                with open(timeline_path) as f:
                    timeline_data = json.load(f)

                self.console.print(f"  Loaded {len(events)} events (skipped event detection stage)")
                context["events"] = events
                context["score_timeline"] = timeline_data.get("timeline", [])
                return context
            else:
                self.console.print(f"[dim]No cache found, running event detection...[/dim]")

        tracks = context.get("tracks", [])
        video_metadata = context["video_metadata"]

        if len(tracks) == 0:
            self.console.print("No tracks available, skipping event detection")
            return context

        # Extract ball tracks only
        ball_tracks = [t for t in tracks if t.get("object_type") == "ball"]

        if len(ball_tracks) == 0:
            self.console.print("No ball tracks found, skipping event detection")
            return context

        self.console.print(f"Analyzing {len(ball_tracks)} ball detections for events...")

        # Build ball trajectory
        trajectory = BallTrajectory(smoothing_window=3)
        trajectory.add_from_tracks(ball_tracks)

        self.console.print(f"Ball trajectory: {len(trajectory.points)} points")

        # Initialize event detector
        detector = EventDetector(
            frame_width=video_metadata["width"],
            frame_height=video_metadata["height"],
            shot_velocity_threshold=15.0,  # pixels/frame
            goal_confidence_threshold=0.6,
            fps=video_metadata["fps"],
        )

        # Detect shots
        shot_events = detector.detect_shots(trajectory)
        self.console.print(f"Detected {len(shot_events)} potential shots")

        # Detect goals
        goal_events = detector.detect_goals(trajectory, shot_events)
        self.console.print(f"Detected {len(goal_events)} potential goals")

        # Combine all events
        all_events = shot_events + goal_events
        all_events = sorted(all_events, key=lambda e: e.timestamp)

        # Save events to JSONL
        events_path = output_dir / "events.jsonl"
        with open(events_path, "w") as f:
            for event in all_events:
                event_dict = {
                    "event_type": event.event_type,
                    "frame_idx": event.frame_idx,
                    "timestamp": event.timestamp,
                    "confidence": event.confidence,
                    "location": list(event.location) if event.location else None,
                    "metadata": event.metadata,
                }
                f.write(json.dumps(event_dict) + "\n")

        self.console.print(f"Saved events to: {events_path}")

        # Create score timeline from goal events
        score_timeline = []
        current_score = {"team_a": 0, "team_b": 0}

        for event in goal_events:
            # For now, assign goals alternately (in future, use team info)
            # This is simplified - would need goal region -> team mapping
            goal_region = event.metadata.get("goal_region", "unknown")

            if goal_region == "top":
                current_score["team_a"] += 1
            elif goal_region == "bottom":
                current_score["team_b"] += 1

            score_timeline.append({
                "timestamp": event.timestamp,
                "frame_idx": event.frame_idx,
                "score": dict(current_score),
                "confidence": event.confidence,
                "goal_region": goal_region,
            })

        # Save score timeline
        timeline_path = output_dir / "score_timeline.json"
        with open(timeline_path, "w") as f:
            json.dump({
                "goals": len(goal_events),
                "final_score": current_score,
                "timeline": score_timeline,
            }, f, indent=2)

        self.console.print(f"Saved score timeline to: {timeline_path}")
        self.console.print(f"Final score: {current_score['team_a']} - {current_score['team_b']}")

        context["events"] = all_events
        context["score_timeline"] = score_timeline
        return context


class OverlayStage(PipelineStage):
    """Stage F: Render overlay video."""

    def __init__(self, config: PipelineConfig):
        super().__init__("overlay", config)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Render annotated video."""
        if not self.config.export.save_overlay_video:
            self.console.print("Skipping overlay video generation")
            return context

        video_path = Path(context["video_path"])
        output_dir = Path(context["output_dir"])

        # Use tracks if available, otherwise fall back to detections
        tracks = context.get("tracks", [])
        use_tracks = len(tracks) > 0

        if use_tracks:
            self.console.print("Rendering with tracks and trails...")
            data_by_frame = {}
            for track in tracks:
                frame_idx = track["frame_idx"]
                if frame_idx not in data_by_frame:
                    data_by_frame[frame_idx] = []
                data_by_frame[frame_idx].append(track)
        else:
            self.console.print("Rendering with detections only...")
            detections = context.get("detections", [])
            data_by_frame = {}
            for det in detections:
                frame_idx = det["frame_idx"]
                if frame_idx not in data_by_frame:
                    data_by_frame[frame_idx] = []
                data_by_frame[frame_idx].append(det)

        # Initialize renderer
        renderer = OverlayRenderer(self.config.overlay)

        # Track history for trails (track_id -> list of center points)
        track_history = {}

        # Initialize video writer
        metadata = context["video_metadata"]
        output_path = output_dir / "overlay.mp4"

        with VideoReader(video_path) as reader, VideoWriter(
            output_path=output_path,
            fps=metadata["fps"],
            width=metadata["width"],
            height=metadata["height"],
            codec=self.config.export.video_codec,
        ) as writer:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console,
            ) as progress:
                total_frames = reader.total_frames
                task = progress.add_task("Rendering overlay video...", total=total_frames)

                for frame_idx, frame in reader.frames(
                    sampling_strategy=self.config.video.sampling_strategy,
                    sampling_interval=self.config.video.sampling_interval,
                ):
                    frame_data = data_by_frame.get(frame_idx, [])

                    # Convert to Detection objects with track IDs
                    from src.vision.detect.yolo import Detection

                    detection_objects = []
                    track_id_map = {}

                    team_labels = {}  # Detection -> team label

                    for data_dict in frame_data:
                        det = Detection(
                            object_type=data_dict["object_type"],
                            bbox=tuple(data_dict["bbox"]),
                            confidence=data_dict["confidence"],
                            class_id=data_dict.get("class_id", 0),
                        )
                        detection_objects.append(det)

                        # Track ID mapping and history
                        if use_tracks and "track_id" in data_dict:
                            track_id = data_dict["track_id"]
                            track_id_map[det] = track_id

                            # Team label mapping
                            if "team_name" in data_dict:
                                team_labels[det] = data_dict["team_name"]

                            # Update track history
                            bbox = data_dict["bbox"]

                            # Skip tracks with NaN bounding boxes
                            if not any(np.isnan(v) or np.isinf(v) for v in bbox):
                                center_x = (bbox[0] + bbox[2]) / 2
                                center_y = (bbox[1] + bbox[3]) / 2

                                if track_id not in track_history:
                                    track_history[track_id] = []
                                track_history[track_id].append((center_x, center_y))

                    # Draw track trails first (below boxes)
                    if use_tracks:
                        annotated = renderer.draw_tracks(frame, track_history)
                    else:
                        annotated = frame.copy()

                    # Draw detections/tracks on top
                    annotated = renderer.draw_detections(
                        annotated,
                        detection_objects,
                        track_ids=track_id_map if use_tracks else None,
                        team_labels=team_labels if use_tracks else None
                    )

                    # Write frame
                    writer.write_frame(annotated)

                    progress.update(task, advance=1)

        self.console.print(f"Saved overlay video to: {output_path}")

        context["overlay_path"] = str(output_path)
        return context


@click.command()
@click.option(
    "--video",
    required=True,
    type=click.Path(exists=True),
    help="Path to input video file",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Output directory for results",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to configuration YAML file",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from existing outputs (skip completed stages)",
)
def main(video: str, output: str, config: str | None, resume: bool):
    """Analyze a soccer match video."""
    console = Console()
    console.print("[bold green]Veo-style Soccer Match Analysis[/bold green]\n")

    # Load configuration
    if config:
        pipeline_config = PipelineConfig.from_yaml(config)
        console.print(f"Loaded config from: {config}\n")
    else:
        pipeline_config = PipelineConfig()
        console.print("Using default configuration\n")

    # Create pipeline
    pipeline = Pipeline(pipeline_config)

    # Add stages
    pipeline.add_stage(IngestStage(pipeline_config))
    pipeline.add_stage(DetectionStage(pipeline_config))
    pipeline.add_stage(TrackingStage(pipeline_config))
    pipeline.add_stage(TeamAssignmentStage(pipeline_config))
    pipeline.add_stage(EventDetectionStage(pipeline_config))
    pipeline.add_stage(OverlayStage(pipeline_config))

    # Run pipeline
    try:
        result = pipeline.run(video_path=video, output_dir=output, resume=resume)

        # Print summary
        console.print("\n[bold green]Analysis Complete![/bold green]\n")
        console.print(f"Output directory: {output}")
        console.print(f"Total detections: {len(result.get('detections', []))}")
        if "overlay_path" in result:
            console.print(f"Overlay video: {result['overlay_path']}")

    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]\n")
        raise


if __name__ == "__main__":
    main()
