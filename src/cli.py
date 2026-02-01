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
from src.vision.detect.base import ObjectDetector
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

        # Store original video path in context for UI
        context["original_video_path"] = str(video_path.absolute())

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

        # Report metrics
        context["ingest_items_processed"] = 1
        context["ingest_custom_metrics"] = {
            "fps": context["video_metadata"]["fps"],
            "total_frames": context["video_metadata"]["total_frames"],
            "duration": context["video_metadata"]["duration"],
            "resolution": f"{context['video_metadata']['width']}x{context['video_metadata']['height']}",
        }

        return context


class DetectionStage(PipelineStage):
    """Stage B: Detect players and ball."""

    def __init__(self, config: PipelineConfig):
        super().__init__("detection", config)

    def _build_detector(self) -> ObjectDetector:
        """Build the appropriate detector based on configuration.

        Returns:
            ObjectDetector: Either a YOLODetector or DetectorEnsemble
        """
        ball_config = self.config.detection.ball
        specialist_config = self.config.detection.ball_specialist
        ensemble_config = self.config.detection.ensemble

        # Build base YOLO detector
        yolo_detector = YOLODetector(
            model_name=self.config.detection.model_name,
            device=self.config.detection.device,
            confidence_threshold=self.config.detection.confidence_threshold,
            ball_confidence_threshold=ball_config.confidence_threshold,
            ball_max_size_ratio=ball_config.max_size_ratio,
        )

        # If ensemble is not enabled, return just the YOLO detector
        if not ensemble_config.enabled:
            return yolo_detector

        # Build ensemble with additional detectors
        detectors: dict[str, ObjectDetector] = {"yolo": yolo_detector}

        # Add ball specialist if enabled
        if specialist_config.enabled:
            from src.vision.detect.ball_specialist import BallSpecialistDetector

            specialist = BallSpecialistDetector(
                model_source=specialist_config.model_source,
                device=self.config.detection.device,
                confidence_threshold=specialist_config.confidence_threshold,
                ball_class_id=specialist_config.ball_class_id,
                max_size_ratio=specialist_config.max_size_ratio,
                max_aspect_ratio=specialist_config.max_aspect_ratio,
                cache_dir=specialist_config.cache_dir,
            )
            detectors["ball_specialist"] = specialist

        # Create ensemble
        from src.vision.detect.ensemble import DetectorEnsemble

        ensemble = DetectorEnsemble(
            detectors=detectors,
            weights=ensemble_config.weights,
            iou_threshold=ensemble_config.iou_threshold,
            skip_box_threshold=ensemble_config.skip_box_threshold,
            fusion_type=ensemble_config.fusion_type,
        )

        self.console.print(f"Using detector ensemble with {len(detectors)} detectors")
        return ensemble

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

                # Report metrics for cached results
                ball_count = sum(1 for d in all_detections if d["object_type"] == "ball")
                player_count = sum(1 for d in all_detections if d["object_type"] == "player")
                context["detection_items_processed"] = len(all_detections)
                context["detection_custom_metrics"] = {
                    "cached": True,
                    "ball_detections": ball_count,
                    "player_detections": player_count,
                }
                return context
            else:
                self.console.print(f"[dim]No cache found at {cache_path.name}, running detection...[/dim]")

        # Get ball boost configuration
        ball_config = self.config.detection.ball

        # Build detector (either YOLO or ensemble based on config)
        detector = self._build_detector()

        # Check if we're using an ensemble (affects multiscale behavior)
        is_ensemble = self.config.detection.ensemble.enabled

        # Initialize ball boost components
        temporal_filter = None
        candidate_tracker = None

        if ball_config.enable_temporal_filter:
            from src.vision.detect.ball_boost import BallTemporalFilter
            temporal_filter = BallTemporalFilter(
                window_size=ball_config.temporal_window_size,
                min_confirmations=ball_config.min_temporal_confirmations,
                max_displacement=ball_config.max_frame_displacement,
            )
            self.console.print(f"  Ball temporal filter enabled (window={ball_config.temporal_window_size})")

        if ball_config.enable_candidate_tracking:
            from src.vision.detect.ball_boost import BallCandidateTracker
            candidate_tracker = BallCandidateTracker(
                min_hits=ball_config.candidate_min_hits,
                max_age=ball_config.candidate_max_age,
                iou_threshold=0.3,
            )
            self.console.print(f"  Ball candidate tracking enabled (min_hits={ball_config.candidate_min_hits})")

        # Multi-scale detection only works with YOLODetector, not ensemble
        # (ensemble already combines multiple detection methods)
        use_multiscale = ball_config.enable_multiscale and not is_ensemble
        if use_multiscale:
            self.console.print(f"  Multi-scale detection enabled (scales={ball_config.scales})")
        elif ball_config.enable_multiscale and is_ensemble:
            self.console.print(f"  Multi-scale disabled (using ensemble instead)")

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
                    # Run detection (multi-scale or standard)
                    # Multi-scale only available for YOLODetector (not ensemble)
                    if use_multiscale and hasattr(detector, 'detect_multiscale'):
                        detections = detector.detect_multiscale(
                            frame,
                            scales=ball_config.scales,
                            merge_iou_threshold=ball_config.merge_iou_threshold,
                            ball_only=True,  # Multi-scale mainly helps ball detection
                        )
                    else:
                        detections = detector.detect(frame)

                    # Separate ball and player detections
                    ball_detections = [d for d in detections if d.object_type == "ball"]
                    player_detections = [d for d in detections if d.object_type == "player"]

                    # Apply temporal filter to ball detections
                    if temporal_filter is not None:
                        ball_detections = temporal_filter.filter(ball_detections, frame_idx)

                    # Apply candidate tracking to ball detections
                    if candidate_tracker is not None:
                        ball_detections = candidate_tracker.update(ball_detections, frame_idx)

                    # Combine detections
                    final_detections = player_detections + ball_detections

                    # Store detections
                    for detection in final_detections:
                        det_dict = detection.to_dict()
                        det_dict["frame_idx"] = frame_idx
                        det_dict["timestamp"] = frame_idx / reader.fps
                        all_detections.append(det_dict)

                    progress.update(task, advance=1)

        # Count ball detections for reporting
        ball_count = sum(1 for d in all_detections if d["object_type"] == "ball")
        player_count = sum(1 for d in all_detections if d["object_type"] == "player")

        self.console.print(f"Total detections: {len(all_detections)} (players: {player_count}, balls: {ball_count})")

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

        # Report metrics
        is_ensemble = self.config.detection.ensemble.enabled
        context["detection_items_processed"] = total_frames
        context["detection_custom_metrics"] = {
            "detector_type": "ensemble" if is_ensemble else "yolo",
            "ball_detections": ball_count,
            "player_detections": player_count,
            "total_detections": len(all_detections),
            "temporal_filter": ball_config.enable_temporal_filter,
            "candidate_tracking": ball_config.enable_candidate_tracking,
            "multiscale": use_multiscale,
        }

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

                # Report metrics for cached results
                unique_tracks = len(set(t["track_id"] for t in all_tracks))
                context["tracking_items_processed"] = len(frame_indices) if "frame_indices" in dir() else len(all_tracks)
                context["tracking_custom_metrics"] = {
                    "cached": True,
                    "unique_tracks": unique_tracks,
                    "track_points": len(all_tracks),
                }
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

        # Report metrics
        unique_tracks = len(set(t["track_id"] for t in all_tracks))
        context["tracking_items_processed"] = len(frame_indices)
        context["tracking_custom_metrics"] = {
            "unique_tracks": unique_tracks,
            "track_points": len(all_tracks),
            "max_age": self.config.tracking.max_age,
            "iou_threshold": self.config.tracking.iou_threshold,
        }

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

                # Report metrics for cached results
                context["team_assignment_items_processed"] = len(team_info.get("track_assignments", {}))
                context["team_assignment_custom_metrics"] = {
                    "cached": True,
                    "n_teams": team_info["n_teams"],
                }
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

            # Report metrics
            context["team_assignment_items_processed"] = len(track_colors)
            context["team_assignment_custom_metrics"] = {
                "n_teams": n_teams,
                "sampled_frames": len(sampled_frames),
                "tracks_with_colors": len(track_colors),
            }

        except Exception as e:
            self.console.print(f"[yellow]Warning: Team assignment failed: {e}[/yellow]")
            self.console.print("Continuing without team assignments...")

            # Report metrics even on failure
            context["team_assignment_items_processed"] = 0
            context["team_assignment_custom_metrics"] = {"error": str(e)}

        return context


class PlayerIdentityStage(PipelineStage):
    """Stage D.5: Player identity persistence using ReID embeddings."""

    def __init__(self, config: PipelineConfig):
        super().__init__("player_identity", config)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Extract ReID embeddings and match to persistent player identities."""
        if not self.config.identity.enabled:
            self.console.print("Player identity disabled, skipping")
            return context

        output_dir = Path(context["output_dir"])

        # Check for cached assignments
        if context.get("resume", False):
            assignments_path = output_dir / "player_assignments.json"
            if assignments_path.exists():
                self.console.print(f"[bold yellow]✓ Using cached player assignments from {assignments_path.name}[/bold yellow]")
                with open(assignments_path) as f:
                    assignments_data = json.load(f)
                context["player_assignments"] = assignments_data
                context["player_identity_items_processed"] = len(assignments_data.get("assignments", []))
                context["player_identity_custom_metrics"] = {"cached": True}
                return context

        video_path = Path(context["video_path"])
        tracks = context.get("tracks", [])

        if len(tracks) == 0:
            self.console.print("No tracks available, skipping player identity")
            return context

        # Get player tracks only
        player_tracks = [t for t in tracks if t.get("object_type") == "player"]
        unique_track_ids = sorted(set(t["track_id"] for t in player_tracks))

        self.console.print(f"Processing {len(unique_track_ids)} player tracks for identity matching...")

        # Import ReID and identity modules
        try:
            from src.vision.reid import OSNetExtractor, CropExtractor
            from src.identity import PlayerDatabase, match_embedding_to_players, aggregate_embeddings
        except ImportError as e:
            self.console.print(f"[yellow]Warning: Could not import identity modules: {e}[/yellow]")
            return context

        # Initialize ReID extractor
        try:
            reid_extractor = OSNetExtractor(
                model_name=self.config.reid.model_name,
                device=self.config.reid.device,
                crop_size=self.config.reid.crop_size,
                batch_size=self.config.reid.batch_size,
                cache_dir=self.config.reid.cache_dir,
            )
            self.console.print(f"  Loaded ReID model: {self.config.reid.model_name} on {reid_extractor.device}")
        except Exception as e:
            self.console.print(f"[yellow]Warning: Failed to load ReID model: {e}[/yellow]")
            return context

        # Initialize crop extractor
        crop_extractor = CropExtractor(
            min_height=self.config.identity.min_crop_height,
            min_width=self.config.identity.min_crop_width,
        )

        # Initialize database
        db_path = Path(self.config.identity.database_path)
        if not db_path.is_absolute():
            db_path = Path(self.config.output_dir) / db_path

        db = PlayerDatabase(db_path)
        self.console.print(f"  Player database: {db_path}")

        # Generate video ID from video path
        video_id = video_path.stem

        # Group tracks by frame
        tracks_by_frame = {}
        for track in player_tracks:
            frame_idx = track["frame_idx"]
            if frame_idx not in tracks_by_frame:
                tracks_by_frame[frame_idx] = []
            tracks_by_frame[frame_idx].append(track)

        # Sample frames for embedding extraction
        # Take every Nth frame based on samples_per_track
        total_frames = len(tracks_by_frame)
        sample_interval = max(1, total_frames // (self.config.identity.samples_per_track * 2))
        sampled_frames = sorted(tracks_by_frame.keys())[::sample_interval]

        self.console.print(f"  Sampling {len(sampled_frames)} frames for ReID extraction...")

        # Collect crops per track
        from src.video.reader import VideoReader
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

        track_crops: dict[int, list] = {tid: [] for tid in unique_track_ids}

        with VideoReader(video_path) as reader:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("Extracting player crops...", total=len(sampled_frames))

                for frame_idx in sampled_frames:
                    frame = reader.get_frame_at(frame_idx)
                    if frame is None:
                        progress.update(task, advance=1)
                        continue

                    frame_tracks = tracks_by_frame.get(frame_idx, [])
                    crops = crop_extractor.extract_crops_from_frame(frame, frame_tracks, frame_idx)

                    for crop in crops:
                        if crop.track_id in track_crops:
                            track_crops[crop.track_id].append(crop)

                    progress.update(task, advance=1)

        # Extract embeddings and match
        self.console.print(f"  Extracting embeddings and matching...")

        # Get existing player centroids
        player_centroids = db.get_all_player_centroids()
        self.console.print(f"  Existing players in database: {len(player_centroids)}")

        assignments = []
        stats = {
            "total_tracks": len(unique_track_ids),
            "auto_matched": 0,
            "suggested": 0,
            "new_players": 0,
            "skipped": 0,
        }

        for track_id in unique_track_ids:
            crops = track_crops.get(track_id, [])

            # Sample crops for this track
            sampled_crops = crop_extractor.sample_crops_for_track(
                crops, track_id,
                n_samples=self.config.identity.samples_per_track,
                strategy="uniform",
            )

            if len(sampled_crops) < 2:
                # Not enough crops for reliable embedding
                stats["skipped"] += 1
                assignments.append({
                    "track_id": track_id,
                    "player_id": None,
                    "player_name": None,
                    "match_method": "skipped",
                    "confidence": 0.0,
                })
                continue

            # Extract embeddings
            crop_images = [c.image for c in sampled_crops]
            embeddings = reid_extractor.extract(crop_images)

            # Aggregate to single embedding
            track_embedding = aggregate_embeddings(list(embeddings))

            # Match to existing players
            match_result = match_embedding_to_players(
                track_embedding,
                player_centroids,
                auto_threshold=self.config.identity.auto_match_threshold,
                suggest_threshold=self.config.identity.suggest_threshold,
                new_player_threshold=self.config.identity.new_player_threshold,
            )

            # Get frame range for this track
            track_data = [t for t in player_tracks if t["track_id"] == track_id]
            frame_start = min(t["frame_idx"] for t in track_data)
            frame_end = max(t["frame_idx"] for t in track_data)

            if match_result.method == "new_player":
                # Create new player
                new_player = db.create_player(embedding=track_embedding)
                player_id = new_player.player_id
                player_name = None
                player_centroids[player_id] = track_embedding  # Update local cache
                stats["new_players"] += 1

            elif match_result.method == "auto":
                # Auto-assign to existing player
                player_id = match_result.player_id
                db.update_player_centroid(player_id, track_embedding)
                player = db.get_player(player_id)
                player_name = player.name if player else None
                stats["auto_matched"] += 1

            else:  # suggested
                player_id = match_result.player_id
                player = db.get_player(player_id) if player_id else None
                player_name = player.name if player else None
                stats["suggested"] += 1

            # Create appearance record
            db.create_appearance(
                video_id=video_id,
                run_name=output_dir.name,
                track_id=track_id,
                player_id=player_id,
                match_confidence=match_result.confidence,
                match_method=match_result.method,
                frame_start=frame_start,
                frame_end=frame_end,
                embedding=track_embedding,
            )

            assignments.append({
                "track_id": track_id,
                "player_id": player_id,
                "player_name": player_name,
                "match_method": match_result.method,
                "confidence": match_result.confidence,
            })

        db.close()

        # Save assignments
        assignments_data = {
            "schema_version": "1.0",
            "video_id": video_id,
            "assignments": assignments,
            "stats": stats,
        }

        assignments_path = output_dir / "player_assignments.json"
        with open(assignments_path, "w") as f:
            json.dump(assignments_data, f, indent=2)

        self.console.print(f"Saved player assignments to: {assignments_path}")
        self.console.print(f"  Auto-matched: {stats['auto_matched']}, Suggested: {stats['suggested']}, New: {stats['new_players']}, Skipped: {stats['skipped']}")

        context["player_assignments"] = assignments_data

        # Report metrics
        context["player_identity_items_processed"] = len(assignments)
        context["player_identity_custom_metrics"] = stats

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

                # Report metrics for cached results
                shots = [e for e in events if e.get("event_type") == "shot"]
                goals = [e for e in events if e.get("event_type") == "goal"]
                context["event_detection_items_processed"] = len(events)
                context["event_detection_custom_metrics"] = {
                    "cached": True,
                    "shots": len(shots),
                    "goals": len(goals),
                }
                return context
            else:
                self.console.print(f"[dim]No cache found, running event detection...[/dim]")

        tracks = context.get("tracks", [])
        video_metadata = context["video_metadata"]

        if len(tracks) == 0:
            self.console.print("No tracks available, skipping event detection")
            return context

        # Extract ball and player tracks
        ball_tracks = [t for t in tracks if t.get("object_type") == "ball"]
        player_tracks = [t for t in tracks if t.get("object_type") == "player"]

        self.console.print(f"Analyzing {len(ball_tracks)} ball detections for events...")
        self.console.print(f"  Player tracks available: {len(player_tracks)}")

        # Compute ball coverage
        total_frames = video_metadata.get("total_frames")
        if ball_tracks and total_frames:
            ball_frames = len(set(t["frame_idx"] for t in ball_tracks))
            coverage = ball_frames / total_frames
            self.console.print(f"  Ball coverage: {coverage:.1%} ({ball_frames}/{total_frames} frames)")
        else:
            coverage = 0.0

        # Build ball trajectory (even if sparse, for velocity-based detection)
        trajectory = BallTrajectory(smoothing_window=3)
        if ball_tracks:
            trajectory.add_from_tracks(ball_tracks)

        original_points = len(trajectory.points)
        self.console.print(f"Ball trajectory: {original_points} raw points")

        # Optionally interpolate gaps in trajectory
        if self.config.events.interpolate_ball and original_points >= 2:
            trajectory = trajectory.interpolate_gaps(
                max_gap_frames=self.config.events.max_interpolation_gap,
                fps=video_metadata["fps"],
            )
            self.console.print(f"After interpolation: {len(trajectory.points)} points (+{len(trajectory.points) - original_points})")

        # Initialize event detector with alternative detection config
        detector = EventDetector(
            frame_width=video_metadata["width"],
            frame_height=video_metadata["height"],
            shot_velocity_threshold=self.config.events.shot_velocity_threshold,
            goal_confidence_threshold=self.config.events.goal_confidence_threshold,
            fps=video_metadata["fps"],
            alternative_config=self.config.events.alternative_shot,
        )

        # Detect shots using combined method (velocity + alternative)
        shot_events = detector.detect_shots_all(
            trajectory,
            player_tracks,
            ball_tracks,
            total_frames=total_frames,
        )

        # Count shots by method
        velocity_shots = [e for e in shot_events if not (e.metadata and e.metadata.get("detection_method") == "alternative")]
        alt_shots = [e for e in shot_events if e.metadata and e.metadata.get("detection_method") == "alternative"]

        self.console.print(f"Detected {len(shot_events)} potential shots")
        if velocity_shots:
            self.console.print(f"  - Velocity-based: {len(velocity_shots)}")
        if alt_shots:
            self.console.print(f"  - Alternative (player behavior): {len(alt_shots)}")

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

        # Report metrics
        context["event_detection_items_processed"] = len(ball_tracks)
        context["event_detection_custom_metrics"] = {
            "shots": len(shot_events),
            "velocity_shots": len(velocity_shots),
            "alternative_shots": len(alt_shots),
            "goals": len(goal_events),
            "ball_coverage": coverage,
            "trajectory_points": len(trajectory.points) if trajectory else 0,
        }

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

        # Report metrics
        context["overlay_items_processed"] = total_frames
        context["overlay_custom_metrics"] = {
            "output_path": str(output_path),
            "use_tracks": use_tracks,
            "codec": self.config.export.video_codec,
        }

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
@click.option(
    "--no-overlay",
    is_flag=True,
    default=False,
    help="Skip overlay video generation (use dynamic rendering in UI instead)",
)
def main(video: str, output: str, config: str | None, resume: bool, no_overlay: bool):
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
    pipeline.add_stage(PlayerIdentityStage(pipeline_config))
    pipeline.add_stage(EventDetectionStage(pipeline_config))

    # Overlay stage is optional
    if not no_overlay:
        pipeline.add_stage(OverlayStage(pipeline_config))
    else:
        console.print("[yellow]Skipping overlay video generation (use dynamic rendering in UI)[/yellow]\n")

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
