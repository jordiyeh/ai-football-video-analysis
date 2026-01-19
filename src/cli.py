"""Command-line interface for soccer video analysis."""

import json
from pathlib import Path
from typing import Any

import click
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

        with VideoReader(video_path) as reader:
            metadata = reader.metadata

            self.console.print(f"Video: {video_path.name}")
            self.console.print(f"  Duration: {metadata.duration:.2f}s")
            self.console.print(f"  FPS: {metadata.fps:.2f}")
            self.console.print(f"  Resolution: {metadata.width}x{metadata.height}")
            self.console.print(f"  Total frames: {metadata.total_frames}")

            # Save metadata
            output_dir = Path(context["output_dir"])
            metadata.save(output_dir / "video_metadata.json")

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

                            # Update track history
                            bbox = data_dict["bbox"]
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
                        track_ids=track_id_map if use_tracks else None
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
def main(video: str, output: str, config: str | None):
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
    pipeline.add_stage(OverlayStage(pipeline_config))

    # Run pipeline
    try:
        result = pipeline.run(video_path=video, output_dir=output)

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
