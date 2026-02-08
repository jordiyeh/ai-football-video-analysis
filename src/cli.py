"""Command-line interface for soccer video analysis."""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from src.config.schemas import PipelineConfig
from src.export.overlay import OverlayRenderer, VideoWriter
from src.pipeline.base import Pipeline, PipelineCancelledError, PipelineStage, save_detections_to_parquet
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
            self.console.print("  Multi-scale disabled (using ensemble instead)")

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

                try:
                    for frame_idx, frame in reader.frames(
                        sampling_strategy=self.config.video.sampling_strategy,
                        sampling_interval=self.config.video.sampling_interval,
                    ):
                        self.check_cancelled(context)

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
                except PipelineCancelledError:
                    self.console.print(f"[yellow]Detection cancelled after {len(all_detections)} detections[/yellow]")
                    raise

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
                context["tracking_items_processed"] = len({t["frame_idx"] for t in all_tracks})
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
            self.check_cancelled(context)

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
                self.console.print("[dim]No cache found, running team assignment...[/dim]")

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
                self.check_cancelled(context)
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

            # ── Map clusters to persistent teams if pre-selected ──
            home_team_id = context.get("home_team_id") or self.config.team.home_team_id
            away_team_id = context.get("away_team_id") or self.config.team.away_team_id
            if home_team_id is not None and away_team_id is not None:
                try:
                    from src.identity.database import PlayerDatabase
                    from src.vision.team.colors import color_distance, bgr_to_hsv

                    db_path = Path(context.get("output_dir", ".")).parent / "players.db"
                    if not db_path.exists():
                        db_path = Path("players.db")

                    with PlayerDatabase(db_path) as db:
                        home_kit_type = context.get("home_kit") or self.config.team.home_kit
                        away_kit_type = context.get("away_kit") or self.config.team.away_kit
                        home_kit = db.get_kit(home_team_id, home_kit_type)
                        away_kit = db.get_kit(away_team_id, away_kit_type)
                        home_team = db.get_team(home_team_id)
                        away_team = db.get_team(away_team_id)

                        if home_kit and away_kit and home_kit.dominant_color_hsv and away_kit.dominant_color_hsv:
                            import numpy as _np
                            home_hsv = _np.array(home_kit.dominant_color_hsv, dtype=_np.float32)
                            away_hsv = _np.array(away_kit.dominant_color_hsv, dtype=_np.float32)

                            # Get cluster centroids in HSV
                            cluster_ids = sorted(team_colors_bgr.keys())
                            if len(cluster_ids) >= 2:
                                c0_hsv = bgr_to_hsv(team_colors_bgr[cluster_ids[0]])
                                c1_hsv = bgr_to_hsv(team_colors_bgr[cluster_ids[1]])

                                # Try both assignments, pick lower total distance
                                d_a = color_distance(c0_hsv, home_hsv, "hsv") + color_distance(c1_hsv, away_hsv, "hsv")
                                d_b = color_distance(c0_hsv, away_hsv, "hsv") + color_distance(c1_hsv, home_hsv, "hsv")

                                if d_a <= d_b:
                                    cluster_to_team = {cluster_ids[0]: "home", cluster_ids[1]: "away"}
                                else:
                                    cluster_to_team = {cluster_ids[0]: "away", cluster_ids[1]: "home"}

                                # Use real team names
                                home_name = home_team.name if home_team else f"team_{home_team_id}"
                                away_name = away_team.name if away_team else f"team_{away_team_id}"
                                for cid, role in cluster_to_team.items():
                                    team_names[cid] = home_name if role == "home" else away_name

                                # Update track team_name
                                for track in tracks:
                                    tid = track.get("team_id")
                                    if tid is not None and tid in team_names:
                                        track["team_name"] = team_names[tid]

                                team_info["team_names"] = team_names
                                team_info["db_team_ids"] = {
                                    "home": home_team_id,
                                    "away": away_team_id,
                                }
                                team_info["cluster_to_role"] = {int(k): v for k, v in cluster_to_team.items()}
                                team_info["cluster_to_team_confidence"] = {
                                    "distance_chosen": float(min(d_a, d_b)),
                                    "distance_rejected": float(max(d_a, d_b)),
                                }

                                # Update run_teams cluster mapping in DB
                                for cid, role in cluster_to_team.items():
                                    db.update_run_team_cluster(context.get("run_name", ""), role, int(cid))

                                self.console.print(
                                    f"Mapped clusters to teams: {home_name} (home), {away_name} (away)"
                                )

                except Exception as map_err:
                    self.console.print(f"[yellow]Warning: Team mapping failed: {map_err}[/yellow]")

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


class FieldNormalizationStage(PipelineStage):
    """Stage D.3: Compute zoom-aware normalized coordinates (`norm_xy`)."""

    def __init__(self, config: PipelineConfig):
        super().__init__("field_normalization", config)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Normalize track coordinates into field-view space."""
        import pandas as pd

        if not self.config.field.enabled:
            self.console.print("Field normalization disabled, skipping")
            context["field_normalization_items_processed"] = 0
            context["field_normalization_custom_metrics"] = {"enabled": False}
            return context

        output_dir = Path(context["output_dir"])
        tracks = context.get("tracks", [])

        if len(tracks) == 0:
            self.console.print("No tracks available, skipping field normalization")
            context["field_normalization_items_processed"] = 0
            context["field_normalization_custom_metrics"] = {
                "enabled": True,
                "skipped_reason": "no_tracks",
            }
            return context

        if self.config.export.detections_format == "parquet":
            tracks_path = output_dir / "tracks.parquet"
        else:
            tracks_path = output_dir / "tracks.jsonl"

        field_norm_path = output_dir / "field_normalization.json"
        viewports_path = output_dir / "field_viewports.parquet"

        # Check for cached normalization outputs.
        if context.get("resume", False) and field_norm_path.exists() and tracks_path.exists():
            try:
                if tracks_path.suffix == ".parquet":
                    df = pd.read_parquet(tracks_path)
                    cached_tracks = df.to_dict(orient="records")
                else:
                    cached_tracks = []
                    with open(tracks_path) as f:
                        for line in f:
                            cached_tracks.append(json.loads(line))

                has_norm = False
                if cached_tracks:
                    sample = cached_tracks[0]
                    has_norm = "norm_x" in sample and "norm_y" in sample

                if has_norm:
                    with open(field_norm_path) as f:
                        field_artifact = json.load(f)
                    self.console.print(
                        f"[bold yellow]✓ Using cached field normalization from {field_norm_path.name}[/bold yellow]"
                    )
                    context["tracks"] = cached_tracks
                    context["field_normalization"] = field_artifact
                    context["field_normalization_path"] = str(field_norm_path)
                    context["field_viewports_path"] = str(viewports_path)
                    context["field_normalization_items_processed"] = len(cached_tracks)
                    context["field_normalization_custom_metrics"] = {
                        "cached": True,
                        **(field_artifact.get("summary") if isinstance(field_artifact.get("summary"), dict) else {}),
                    }
                    return context
            except Exception:
                # Fall through and recompute when cached artifacts are invalid.
                pass

        from src.vision.field import normalize_tracks_to_field_view

        video_metadata = context.get("video_metadata", {})
        frame_width = int(video_metadata.get("width", 0) or 0)
        frame_height = int(video_metadata.get("height", 0) or 0)
        if frame_width <= 1 or frame_height <= 1:
            self.console.print("[yellow]Invalid video dimensions, skipping field normalization[/yellow]")
            context["field_normalization_items_processed"] = 0
            context["field_normalization_custom_metrics"] = {
                "enabled": True,
                "skipped_reason": "invalid_video_dimensions",
                "width": frame_width,
                "height": frame_height,
            }
            return context

        normalized_tracks, viewport_rows, summary = normalize_tracks_to_field_view(
            tracks=tracks,
            frame_width=frame_width,
            frame_height=frame_height,
            config=self.config.field,
        )

        # Persist viewport debug artifact (compact frame-level info).
        viewports_df = pd.DataFrame(viewport_rows)
        viewports_df.to_parquet(viewports_path, index=False)

        field_artifact = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "video_id": Path(context["video_path"]).stem,
            "config": self.config.field.model_dump(),
            "summary": summary,
            "outputs": {
                "tracks_path": str(tracks_path),
                "viewports_path": str(viewports_path),
            },
        }
        with open(field_norm_path, "w") as f:
            json.dump(field_artifact, f, indent=2)

        if self.config.export.save_tracks:
            if tracks_path.suffix == ".parquet":
                save_detections_to_parquet(normalized_tracks, tracks_path)
            else:
                with open(tracks_path, "w") as f:
                    for track in normalized_tracks:
                        f.write(json.dumps(track) + "\n")

        context["tracks"] = normalized_tracks
        context["field_normalization"] = field_artifact
        context["field_normalization_path"] = str(field_norm_path)
        context["field_viewports_path"] = str(viewports_path)
        context["field_normalization_items_processed"] = len(normalized_tracks)
        context["field_normalization_custom_metrics"] = {
            "enabled": True,
            **summary,
        }

        self.console.print(f"Saved field normalization artifact to: {field_norm_path}")
        self.console.print(f"Saved field viewport timeline to: {viewports_path}")
        self.console.print(
            "  Normalized track points: "
            f"{summary.get('track_points_normalized', 0)}/{summary.get('track_points_total', 0)}"
        )
        return context


class ProfileIngestionStage(PipelineStage):
    """Stage D.4: Ingest external player profile bundles."""

    def __init__(self, config: PipelineConfig):
        super().__init__("profile_ingestion", config)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Load profile folders (.pkl + photos) and export normalized artifacts."""
        profile_config = self.config.identity.profile_ingestion
        output_dir = Path(context["output_dir"])

        if not profile_config.enabled:
            self.console.print("Profile ingestion disabled, skipping")
            context["profile_ingestion_items_processed"] = 0
            context["profile_ingestion_custom_metrics"] = {"enabled": False}
            return context

        if not profile_config.profiles_root:
            self.console.print("[yellow]Profile ingestion enabled but profiles_root is not set, skipping[/yellow]")
            context["profile_ingestion_items_processed"] = 0
            context["profile_ingestion_custom_metrics"] = {
                "enabled": True,
                "skipped_reason": "profiles_root_not_set",
            }
            return context

        registry_path = output_dir / "profile_registry.json"
        embeddings_path = output_dir / "profile_embeddings.parquet"

        if context.get("resume", False) and registry_path.exists() and embeddings_path.exists():
            self.console.print(
                f"[bold yellow]✓ Using cached profile artifacts from {registry_path.name}[/bold yellow]"
            )
            with open(registry_path) as f:
                registry = json.load(f)

            context["profile_registry"] = registry
            context["profile_registry_path"] = str(registry_path)
            context["profile_embeddings_path"] = str(embeddings_path)
            context["profile_ingestion_items_processed"] = registry.get("summary", {}).get(
                "profiles_found", 0
            )
            context["profile_ingestion_custom_metrics"] = {
                "cached": True,
                "profiles_root": registry.get("profile_root"),
                **registry.get("summary", {}),
            }
            return context

        from src.identity import ingest_profiles
        import pandas as pd

        profiles_root = Path(profile_config.profiles_root).expanduser()
        self.console.print(f"Loading player profiles from: {profiles_root}")

        registry, embedding_rows = ingest_profiles(
            profile_root=profiles_root,
            recursive_image_scan=profile_config.recursive_image_scan,
            image_extensions=profile_config.image_extensions,
        )

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        embedding_columns = [
            "schema_version",
            "profile_id",
            "player_name",
            "jersey_number",
            "modality",
            "embedding_model",
            "embedding_source",
            "embedding_index",
            "embedding_dim",
            "embedding_norm",
            "embedding",
            "source_file",
            "source_image_path",
        ]
        if embedding_rows:
            embeddings_df = pd.DataFrame(embedding_rows)
        else:
            embeddings_df = pd.DataFrame(columns=embedding_columns)
        embeddings_df.to_parquet(embeddings_path, index=False)

        summary = registry.get("summary", {})
        self.console.print(
            "  Profiles: "
            f"{summary.get('profiles_found', 0)}, "
            f"with embeddings: {summary.get('profiles_with_embeddings', 0)}, "
            f"embeddings: {summary.get('embeddings_total', 0)}"
        )
        self.console.print(f"Saved profile registry to: {registry_path}")
        self.console.print(f"Saved profile embeddings to: {embeddings_path}")

        context["profile_registry"] = registry
        context["profile_registry_path"] = str(registry_path)
        context["profile_embeddings_path"] = str(embeddings_path)
        context["profile_ingestion_items_processed"] = summary.get("profiles_found", 0)
        context["profile_ingestion_custom_metrics"] = {
            "profiles_root": str(profiles_root),
            **summary,
        }
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
            from src.identity import (
                PlayerDatabase,
                match_embedding_to_players,
                aggregate_embeddings,
                fuse_identity_evidence,
                build_profile_signatures,
                seed_players_from_signatures,
                match_embedding_to_profile_links,
                build_profile_face_signatures,
                match_track_face_evidence,
                build_jersey_player_index,
                extract_jersey_ocr_evidence,
                apply_multimodal_evidence,
                apply_substitution_locks,
            )
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

        profile_cfg = self.config.identity.profile_ingestion
        multimodal_cfg = self.config.identity.multimodal
        profile_registry = context.get("profile_registry")
        profile_signatures = []
        profile_links: list[dict[str, Any]] = []
        profile_seed_summary = {
            "profiles_seen": 0,
            "profiles_with_signatures": 0,
            "images_used": 0,
            "images_failed": 0,
            "profiles_skipped": 0,
            "profiles_linked": 0,
        }
        face_signatures = []
        face_signature_summary = {
            "profiles_seen": 0,
            "profiles_linked": 0,
            "profiles_with_face_signatures": 0,
            "images_used": 0,
            "images_failed": 0,
        }
        multimodal_locking_summary = {
            "locks_applied": 0,
            "overlap_conflicts": 0,
            "substitution_unlocks": 0,
            "demoted_conflicts": 0,
        }

        # Optional: seed identity database from external profile pictures.
        if profile_cfg.enabled and profile_cfg.enable_body_embedding_seed:
            if profile_registry is None:
                registry_path = output_dir / "profile_registry.json"
                if registry_path.exists():
                    with open(registry_path) as f:
                        profile_registry = json.load(f)

            if profile_registry and profile_registry.get("profiles"):
                self.console.print("  Building profile-photo embedding signatures...")
                try:
                    profile_detector = YOLODetector(
                        model_name=self.config.detection.model_name,
                        device=self.config.detection.device,
                        confidence_threshold=profile_cfg.detector_confidence_threshold,
                        ball_confidence_threshold=self.config.detection.ball.confidence_threshold,
                        ball_max_size_ratio=self.config.detection.ball.max_size_ratio,
                    )

                    profile_signatures, profile_seed_summary = build_profile_signatures(
                        profile_registry=profile_registry,
                        detector=profile_detector,
                        crop_extractor=crop_extractor,
                        reid_extractor=reid_extractor,
                        max_images_per_profile=profile_cfg.max_images_per_profile_for_reid,
                        min_profile_crops_for_seed=profile_cfg.min_profile_crops_for_seed,
                        fallback_full_image=profile_cfg.fallback_full_image,
                    )

                    if profile_signatures:
                        profile_links, _ = seed_players_from_signatures(db, profile_signatures)
                        profile_seed_summary["profiles_linked"] = len(profile_links)

                        links_path = output_dir / "profile_player_links.json"
                        with open(links_path, "w") as f:
                            json.dump(
                                {
                                    "schema_version": "1.0",
                                    "video_id": video_path.stem,
                                    "generated_at": datetime.now(timezone.utc).isoformat(),
                                    "summary": profile_seed_summary,
                                    "links": profile_links,
                                },
                                f,
                                indent=2,
                            )
                        context["profile_player_links"] = profile_links
                        context["profile_player_links_path"] = str(links_path)
                        self.console.print(
                            f"  Seeded/linked {len(profile_links)} players from profile pictures"
                        )
                    else:
                        self.console.print("  No valid profile signatures were produced")
                except Exception as e:
                    self.console.print(f"[yellow]Warning: Profile-photo seeding failed: {e}[/yellow]")
            else:
                self.console.print("  No profile registry available for profile-photo seeding")

        # Optional: build profile-linked face-signature gallery for multimodal evidence.
        if multimodal_cfg.enabled and multimodal_cfg.face.enabled:
            if profile_registry and profile_links:
                face_signatures, face_signature_summary = build_profile_face_signatures(
                    profile_registry=profile_registry,
                    profile_links=profile_links,
                    max_images_per_profile=multimodal_cfg.face.max_images_per_profile,
                    min_face_images=multimodal_cfg.face.min_profile_face_images,
                )
                if face_signatures:
                    self.console.print(
                        "  Built face-signature gallery: "
                        f"{len(face_signatures)} profile-linked players"
                    )
                else:
                    self.console.print("  Face-signature gallery unavailable (no valid face descriptors)")
            else:
                self.console.print("  Face-signature gallery skipped (profile links unavailable)")

        jersey_player_index = build_jersey_player_index(db.list_players())

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
                    self.check_cancelled(context)

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
        self.console.print("  Extracting embeddings and matching...")

        # Get existing player centroids
        player_centroids = db.get_all_player_centroids()
        self.console.print(f"  Existing players in database: {len(player_centroids)}")

        track_ranges: dict[int, tuple[int | None, int | None]] = {}
        for row in player_tracks:
            tid = int(row["track_id"])
            frame_idx = int(row["frame_idx"])
            if tid not in track_ranges:
                track_ranges[tid] = (frame_idx, frame_idx)
            else:
                start, end = track_ranges[tid]
                track_ranges[tid] = (min(start, frame_idx), max(end, frame_idx))

        assignments = []
        track_embeddings: dict[int, np.ndarray] = {}
        stats = {
            "total_tracks": len(unique_track_ids),
            "auto_matched": 0,
            "suggested": 0,
            "new_players": 0,
            "skipped": 0,
            "fusion_body_only": 0,
            "fusion_profile_only": 0,
            "fusion_profile_override": 0,
            "fusion_agreement_boost": 0,
            "profile_signatures": len(profile_signatures),
            "profile_links": len(profile_links),
            "profile_seed_summary": profile_seed_summary,
            "face_signatures": len(face_signatures),
            "face_signature_summary": face_signature_summary,
            "face_evidence_tracks": 0,
            "jersey_ocr_tracks": 0,
            "jersey_ocr_supported_tracks": 0,
            "multimodal_face_overrides": 0,
            "multimodal_jersey_overrides": 0,
            "multimodal_face_agreements": 0,
            "multimodal_jersey_agreements": 0,
            "locking_locks_applied": 0,
            "locking_overlap_conflicts": 0,
            "locking_substitution_unlocks": 0,
            "locking_demoted_conflicts": 0,
        }

        for track_id in unique_track_ids:
            crops = track_crops.get(track_id, [])
            frame_start, frame_end = track_ranges.get(track_id, (None, None))

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
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "player_id": None,
                    "player_name": None,
                    "match_method": "skipped",
                    "confidence": 0.0,
                    "lock_state": "candidate",
                    "lock_reason": "insufficient_samples",
                    "lock_conflict_with_track_id": None,
                })
                continue

            # Extract embeddings
            crop_images = [c.image for c in sampled_crops]
            embeddings = reid_extractor.extract(crop_images)

            # Aggregate to single embedding
            track_embedding = aggregate_embeddings(list(embeddings))
            track_embeddings[track_id] = track_embedding

            # Match to existing players
            body_match = match_embedding_to_players(
                track_embedding,
                player_centroids,
                auto_threshold=self.config.identity.auto_match_threshold,
                suggest_threshold=self.config.identity.suggest_threshold,
                new_player_threshold=self.config.identity.new_player_threshold,
            )

            profile_evidence = None
            if profile_cfg.profile_match_enabled and profile_signatures and profile_links:
                profile_evidence = match_embedding_to_profile_links(
                    embedding=track_embedding,
                    signatures=profile_signatures,
                    profile_links=profile_links,
                    suggest_threshold=profile_cfg.profile_match_suggest_threshold,
                )

            player_id, final_confidence, final_method, fusion_metadata = fuse_identity_evidence(
                body_match=body_match,
                profile_evidence=profile_evidence,
                profile_auto_threshold=profile_cfg.profile_match_auto_threshold,
                profile_suggest_threshold=profile_cfg.profile_match_suggest_threshold,
                override_margin=profile_cfg.profile_override_margin,
                agreement_bonus=profile_cfg.profile_agreement_bonus,
            )

            strategy = fusion_metadata.get("strategy", "body_only")
            if strategy == "agreement_boost":
                stats["fusion_agreement_boost"] += 1
            elif strategy == "profile_override":
                stats["fusion_profile_override"] += 1
            elif strategy == "profile_only":
                stats["fusion_profile_only"] += 1
            else:
                stats["fusion_body_only"] += 1

            # Optional multimodal post-fusion evidence (face + jersey OCR).
            multimodal_metadata: dict[str, Any] = {
                "enabled": multimodal_cfg.enabled,
                "face": None,
                "jersey_ocr": None,
                "applied": [],
            }
            if multimodal_cfg.enabled:
                face_evidence = None
                if multimodal_cfg.face.enabled and face_signatures:
                    face_evidence = match_track_face_evidence(
                        crop_images=crop_images,
                        signatures=face_signatures,
                        suggest_threshold=multimodal_cfg.face.suggest_threshold,
                        min_support_frames=multimodal_cfg.face.min_track_support_frames,
                    )
                    if face_evidence is not None:
                        stats["face_evidence_tracks"] += 1

                jersey_evidence = None
                if multimodal_cfg.jersey_ocr.enabled and jersey_player_index:
                    jersey_evidence = extract_jersey_ocr_evidence(
                        crop_images=crop_images,
                        jersey_player_index=jersey_player_index,
                        min_ocr_confidence=multimodal_cfg.jersey_ocr.min_ocr_confidence,
                        min_support_frames=multimodal_cfg.jersey_ocr.min_track_support_frames,
                    )
                    if jersey_evidence is not None:
                        stats["jersey_ocr_tracks"] += 1
                        if jersey_evidence.player_id is not None:
                            stats["jersey_ocr_supported_tracks"] += 1

                (
                    player_id,
                    final_confidence,
                    final_method,
                    multimodal_metadata,
                ) = apply_multimodal_evidence(
                    base_player_id=player_id,
                    base_confidence=final_confidence,
                    base_method=final_method,
                    auto_threshold=self.config.identity.auto_match_threshold,
                    suggest_threshold=self.config.identity.suggest_threshold,
                    face_evidence=face_evidence,
                    jersey_evidence=jersey_evidence,
                    face_override_margin=multimodal_cfg.face.override_margin,
                    face_agreement_bonus=multimodal_cfg.face.agreement_bonus,
                    jersey_override_margin=multimodal_cfg.jersey_ocr.override_margin,
                    jersey_agreement_bonus=multimodal_cfg.jersey_ocr.agreement_bonus,
                )

                applied_signals = set(multimodal_metadata.get("applied", []))
                if "face_override" in applied_signals:
                    stats["multimodal_face_overrides"] += 1
                if "jersey_override" in applied_signals:
                    stats["multimodal_jersey_overrides"] += 1
                if "face_agreement_boost" in applied_signals:
                    stats["multimodal_face_agreements"] += 1
                if "jersey_agreement_boost" in applied_signals:
                    stats["multimodal_jersey_agreements"] += 1

            fusion_metadata["multimodal"] = multimodal_metadata

            if player_id is None or final_method == "new_player":
                # Create new player
                new_player = db.create_player(embedding=track_embedding)
                player_id = new_player.player_id
                player_name = None
                player_centroids[player_id] = track_embedding  # Update local cache
                stats["new_players"] += 1

            elif final_method == "auto":
                # Auto-assign to existing player
                db.update_player_centroid(player_id, track_embedding)
                player = db.get_player(player_id)
                player_name = player.name if player else None

            else:  # suggested
                player = db.get_player(player_id) if player_id else None
                player_name = player.name if player else None

            assignments.append({
                "track_id": track_id,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "player_id": player_id,
                "player_name": player_name,
                "match_method": final_method,
                "confidence": final_confidence,
                "fusion": fusion_metadata,
                "lock_state": "candidate",
                "lock_reason": None,
                "lock_conflict_with_track_id": None,
            })

        if multimodal_cfg.enabled and multimodal_cfg.locking.enabled:
            assignments, multimodal_locking_summary = apply_substitution_locks(
                assignments,
                lock_confidence_threshold=multimodal_cfg.locking.lock_confidence_threshold,
                overlap_conflict_frames=multimodal_cfg.locking.overlap_conflict_frames,
                substitution_gap_frames=multimodal_cfg.locking.substitution_gap_frames,
                demote_conflicting_auto=multimodal_cfg.locking.demote_conflicting_auto,
            )

        # Persist appearances after lock/unlock reconciliation so DB mirrors final methods.
        for assignment in assignments:
            track_id = int(assignment["track_id"])
            track_embedding = track_embeddings.get(track_id)
            if track_embedding is None:
                continue
            appearance_method = assignment.get("match_method")
            if appearance_method not in ("auto", "suggested"):
                appearance_method = "auto"
            db.create_appearance(
                video_id=video_id,
                run_name=output_dir.name,
                track_id=track_id,
                player_id=assignment.get("player_id"),
                match_confidence=float(assignment.get("confidence") or 0.0),
                match_method=appearance_method,  # type: ignore[arg-type]
                frame_start=assignment.get("frame_start"),
                frame_end=assignment.get("frame_end"),
                embedding=track_embedding,
            )

        # Recompute high-level method counters after multimodal and locking adjustments.
        stats["auto_matched"] = sum(1 for row in assignments if row.get("match_method") == "auto")
        stats["suggested"] = sum(1 for row in assignments if row.get("match_method") == "suggested")
        stats["locking_locks_applied"] = multimodal_locking_summary["locks_applied"]
        stats["locking_overlap_conflicts"] = multimodal_locking_summary["overlap_conflicts"]
        stats["locking_substitution_unlocks"] = multimodal_locking_summary["substitution_unlocks"]
        stats["locking_demoted_conflicts"] = multimodal_locking_summary["demoted_conflicts"]

        db.close()

        multimodal_summary = {
            "schema_version": "1.0",
            "video_id": video_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "enabled": multimodal_cfg.enabled,
            "face": {
                "enabled": multimodal_cfg.face.enabled,
                "signatures": len(face_signatures),
                "summary": face_signature_summary,
            },
            "jersey_ocr": {
                "enabled": multimodal_cfg.jersey_ocr.enabled,
                "tracks_with_evidence": stats["jersey_ocr_tracks"],
                "tracks_with_unique_player": stats["jersey_ocr_supported_tracks"],
                "jersey_player_index_size": len(jersey_player_index),
            },
            "locking": {
                "enabled": multimodal_cfg.locking.enabled,
                **multimodal_locking_summary,
            },
        }
        multimodal_summary_path = output_dir / "identity_multimodal_summary.json"
        with open(multimodal_summary_path, "w") as f:
            json.dump(multimodal_summary, f, indent=2)

        # Save assignments
        assignments_data = {
            "schema_version": "1.2",
            "video_id": video_id,
            "assignments": assignments,
            "stats": stats,
        }

        assignments_path = output_dir / "player_assignments.json"
        with open(assignments_path, "w") as f:
            json.dump(assignments_data, f, indent=2)

        self.console.print(f"Saved player assignments to: {assignments_path}")
        self.console.print(f"Saved multimodal identity summary to: {multimodal_summary_path}")
        self.console.print(f"  Auto-matched: {stats['auto_matched']}, Suggested: {stats['suggested']}, New: {stats['new_players']}, Skipped: {stats['skipped']}")
        if stats["profile_links"] > 0:
            self.console.print(
                "  Fusion: "
                f"body_only={stats['fusion_body_only']}, "
                f"profile_only={stats['fusion_profile_only']}, "
                f"profile_override={stats['fusion_profile_override']}, "
                f"agreement_boost={stats['fusion_agreement_boost']}"
            )
        if multimodal_cfg.enabled:
            self.console.print(
                "  Multimodal: "
                f"face_tracks={stats['face_evidence_tracks']}, "
                f"jersey_tracks={stats['jersey_ocr_tracks']}, "
                f"locks={stats['locking_locks_applied']}, "
                f"conflicts={stats['locking_overlap_conflicts']}"
            )

        context["player_assignments"] = assignments_data
        context["identity_multimodal_summary"] = multimodal_summary
        context["identity_multimodal_summary_path"] = str(multimodal_summary_path)

        # Report metrics
        context["player_identity_items_processed"] = len(assignments)
        context["player_identity_custom_metrics"] = stats

        return context


class TeamAnalyticsStage(PipelineStage):
    """Stage D.6: Team-level analytics (possession, territory, pass network, pressing)."""

    def __init__(self, config: PipelineConfig):
        super().__init__("team_analytics", config)

    @staticmethod
    def _write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
        """Write rows to CSV with stable empty-file schema."""
        import pandas as pd

        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
        else:
            pd.DataFrame(columns=columns).to_csv(path, index=False)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Compute tactical team analytics from tracks."""
        cfg = self.config.team_analytics
        output_dir = Path(context["output_dir"])

        if not cfg.enabled:
            self.console.print("Team analytics disabled, skipping")
            context["team_analytics_items_processed"] = 0
            context["team_analytics_custom_metrics"] = {"enabled": False}
            return context

        analytics_path = output_dir / "team_analytics.json"
        possession_csv_path = output_dir / "team_possession_timeline.csv"
        pass_network_csv_path = output_dir / "team_pass_network.csv"
        pressing_csv_path = output_dir / "team_pressing_timeline.csv"
        territory_csv_path = output_dir / "team_territory_zones.csv"

        if (
            context.get("resume", False)
            and analytics_path.exists()
            and possession_csv_path.exists()
            and pass_network_csv_path.exists()
            and pressing_csv_path.exists()
            and territory_csv_path.exists()
        ):
            self.console.print(
                f"[bold yellow]✓ Using cached team analytics from {analytics_path.name}[/bold yellow]"
            )
            with open(analytics_path) as f:
                analytics_artifact = json.load(f)

            summary = analytics_artifact.get("summary", {})
            context["team_analytics"] = analytics_artifact
            context["team_analytics_path"] = str(analytics_path)
            context["team_analytics_items_processed"] = int(summary.get("frames_with_ball", 0))
            context["team_analytics_custom_metrics"] = {
                "cached": True,
                "teams_detected": len(summary.get("teams_detected", [])),
                "passes_inferred": int(summary.get("passes_inferred", 0)),
                "pressing_evaluations": int(summary.get("pressing_evaluations", 0)),
                "territory_samples": int(summary.get("territory_samples", 0)),
            }
            return context

        tracks = context.get("tracks", [])
        if not tracks:
            self.console.print("No tracks available, skipping team analytics")
            context["team_analytics_items_processed"] = 0
            context["team_analytics_custom_metrics"] = {
                "enabled": True,
                "skipped_reason": "no_tracks",
            }
            return context

        assignments_data = context.get("player_assignments")
        assignments = []
        if isinstance(assignments_data, dict):
            assignments = list(assignments_data.get("assignments", []) or [])
        else:
            assignments_path = output_dir / "player_assignments.json"
            if assignments_path.exists():
                with open(assignments_path) as f:
                    assignments = list(json.load(f).get("assignments", []) or [])

        metadata = context.get("video_metadata", {})
        fps = float(metadata.get("fps", 30.0) or 30.0)
        frame_width = int(metadata.get("width", 0) or 0)
        frame_height = int(metadata.get("height", 0) or 0)

        from src.analytics import build_team_analytics

        analytics_data = build_team_analytics(
            tracks=tracks,
            assignments=assignments,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
            config=cfg,
        )

        possession_rows = list(analytics_data.pop("possession_timeline", []))
        pass_network_rows = list(analytics_data.pop("pass_network_edges", []))
        pressing_rows = list(analytics_data.pop("pressing_timeline", []))
        territory_rows = list(analytics_data.pop("territory_rows", []))

        self._write_csv(
            possession_rows,
            possession_csv_path,
            columns=[
                "frame_idx",
                "timestamp",
                "ball_track_id",
                "raw_owner_track_id",
                "owner_track_id",
                "owner_team",
                "owner_player_id",
                "owner_player_name",
                "nearest_distance_px",
                "owner_norm_x",
                "owner_norm_y",
                "available_players",
            ],
        )
        self._write_csv(
            pass_network_rows,
            pass_network_csv_path,
            columns=[
                "team",
                "from_track_id",
                "to_track_id",
                "from_player_id",
                "to_player_id",
                "from_player_name",
                "to_player_name",
                "pass_count",
                "avg_gap_seconds",
                "avg_distance_norm",
            ],
        )
        self._write_csv(
            pressing_rows,
            pressing_csv_path,
            columns=[
                "frame_idx",
                "timestamp",
                "attacking_team",
                "defending_team",
                "carrier_track_id",
                "carrier_player_id",
                "nearest_distance_norm",
                "defenders_within_radius",
                "pressure_score",
                "high_press",
            ],
        )
        self._write_csv(
            territory_rows,
            territory_csv_path,
            columns=[
                "team",
                "axis",
                "zone",
                "count",
                "team_ratio",
                "zone_control_share",
            ],
        )

        analytics_artifact = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "video_id": Path(context["video_path"]).stem,
            "config": cfg.model_dump(),
            "summary": analytics_data.get("summary", {}),
            "possession": analytics_data.get("possession", {}),
            "territory": analytics_data.get("territory", {}),
            "pass_network": analytics_data.get("pass_network", {}),
            "pressing": analytics_data.get("pressing", {}),
            "outputs": {
                "possession_timeline_csv": str(possession_csv_path),
                "pass_network_csv": str(pass_network_csv_path),
                "pressing_timeline_csv": str(pressing_csv_path),
                "territory_zones_csv": str(territory_csv_path),
            },
        }

        with open(analytics_path, "w") as f:
            json.dump(analytics_artifact, f, indent=2)

        summary = analytics_artifact.get("summary", {})
        self.console.print(f"Saved team analytics to: {analytics_path}")
        self.console.print(f"Saved possession timeline to: {possession_csv_path}")
        self.console.print(f"Saved pass network to: {pass_network_csv_path}")
        self.console.print(f"Saved pressing timeline to: {pressing_csv_path}")
        self.console.print(f"Saved territory zones to: {territory_csv_path}")
        self.console.print(
            "  Team analytics: "
            f"teams={len(summary.get('teams_detected', []))}, "
            f"possession_frames={int(summary.get('frames_with_possession', 0))}, "
            f"passes={int(summary.get('passes_inferred', 0))}, "
            f"pressing_evals={int(summary.get('pressing_evaluations', 0))}"
        )

        context["team_analytics"] = analytics_artifact
        context["team_analytics_path"] = str(analytics_path)
        context["team_analytics_items_processed"] = int(summary.get("frames_with_ball", 0))
        context["team_analytics_custom_metrics"] = {
            "teams_detected": len(summary.get("teams_detected", [])),
            "frames_with_possession": int(summary.get("frames_with_possession", 0)),
            "passes_inferred": int(summary.get("passes_inferred", 0)),
            "pressing_evaluations": int(summary.get("pressing_evaluations", 0)),
            "territory_samples": int(summary.get("territory_samples", 0)),
        }
        return context


class EventDetectionStage(PipelineStage):
    """Stage E: Detect match events (shots/goals/passes/set-pieces/tactical)."""

    _SET_PIECE_EVENT_TYPES = (
        "set_piece",
        "kickoff",
        "throw_in",
        "corner_kick",
        "free_kick",
        "goal_kick",
    )
    _TACTICAL_EVENT_TYPES = (
        "build_up",
        "pressing",
        "defending",
        "transition",
    )

    def __init__(self, config: PipelineConfig):
        super().__init__("event_detection", config)

    @staticmethod
    def _event_type_counts(events: list[Any]) -> dict[str, int]:
        """Aggregate event counts by event_type across dicts/dataclasses."""
        counts: Counter[str] = Counter()
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
            if event_type:
                counts[event_type] += 1
        return dict(counts)

    @classmethod
    def _set_piece_count(cls, event_type_counts: dict[str, int]) -> int:
        """Return total set-piece events including subtype event types."""
        return int(sum(event_type_counts.get(event_type, 0) for event_type in cls._SET_PIECE_EVENT_TYPES))

    @classmethod
    def _tactical_count(cls, event_type_counts: dict[str, int]) -> int:
        """Return total tactical events including all tactical subtypes."""
        return int(sum(event_type_counts.get(event_type, 0) for event_type in cls._TACTICAL_EVENT_TYPES))

    @staticmethod
    def _pass_inference_config(config: PipelineConfig) -> dict[str, Any]:
        """Build pass inference configuration from team analytics defaults."""
        team_cfg = config.team_analytics
        return {
            "possession_max_ball_distance_px": float(team_cfg.possession_max_ball_distance_px),
            "possession_smoothing_frames": int(team_cfg.possession_smoothing_frames),
            "possession_min_stable_frames": int(team_cfg.possession_min_stable_frames),
            "possession_min_segment_frames": int(team_cfg.possession_min_segment_frames),
            "pass_min_gap_seconds": float(team_cfg.pass_min_gap_seconds),
            "pass_max_gap_seconds": float(team_cfg.pass_max_gap_seconds),
        }

    @staticmethod
    def _tactical_inference_config(config: PipelineConfig, fps: float) -> dict[str, Any]:
        """Build tactical inference thresholds from team-analytics defaults."""
        team_cfg = config.team_analytics
        high_press_frames = max(1, int(team_cfg.high_press_min_frames))

        return {
            "build_up_min_frames": max(6, int(team_cfg.possession_min_segment_frames) * 2),
            "build_up_min_progress_norm": 0.10,
            "build_up_min_carrier_changes": 1,
            "pressing_min_frames": max(3, high_press_frames // 2),
            "pressing_min_pressure_score": float(team_cfg.high_press_threshold),
            "defending_min_frames": max(4, high_press_frames),
            "defending_max_nearest_distance_norm": float(team_cfg.pressure_radius_norm) * 1.35,
            "defending_min_defenders_within_radius": 1.0,
            "transition_max_gap_frames": max(2, int(round(max(1.0, fps) * 1.2))),
            "transition_min_displacement_norm": 0.08,
        }

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Detect match events from trajectory + tracking context."""
        from src.events import (
            BallTrajectory,
            EventDetector,
            infer_pass_events,
            infer_set_piece_events,
            infer_tactical_events,
        )

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
                event_type_counts = self._event_type_counts(events)
                context["event_detection_items_processed"] = len(events)
                context["event_detection_custom_metrics"] = {
                    "cached": True,
                    "shots": int(event_type_counts.get("shot", 0)),
                    "goals": int(event_type_counts.get("goal", 0)),
                    "passes": int(event_type_counts.get("pass", 0)),
                    "set_pieces": self._set_piece_count(event_type_counts),
                    "tactical_events": self._tactical_count(event_type_counts),
                    "tactical_build_ups": int(event_type_counts.get("build_up", 0)),
                    "tactical_pressing": int(event_type_counts.get("pressing", 0)),
                    "tactical_defending": int(event_type_counts.get("defending", 0)),
                    "tactical_transitions": int(event_type_counts.get("transition", 0)),
                    "kickoffs": int(event_type_counts.get("kickoff", 0)),
                    "throw_ins": int(event_type_counts.get("throw_in", 0)),
                    "corner_kicks": int(event_type_counts.get("corner_kick", 0)),
                    "free_kicks": int(event_type_counts.get("free_kick", 0)),
                    "goal_kicks": int(event_type_counts.get("goal_kick", 0)),
                }
                return context
            else:
                self.console.print("[dim]No cache found, running event detection...[/dim]")

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

        shot_events = []
        velocity_shots = []
        alt_shots = []
        if self.config.events.detect_shots:
            # Detect shots using combined method (velocity + alternative)
            shot_events = detector.detect_shots_all(
                trajectory,
                player_tracks,
                ball_tracks,
                total_frames=total_frames,
            )

            # Count shots by method
            velocity_shots = [
                e
                for e in shot_events
                if not (e.metadata and e.metadata.get("detection_method") == "alternative")
            ]
            alt_shots = [
                e
                for e in shot_events
                if e.metadata and e.metadata.get("detection_method") == "alternative"
            ]

            self.console.print(f"Detected {len(shot_events)} potential shots")
            if velocity_shots:
                self.console.print(f"  - Velocity-based: {len(velocity_shots)}")
            if alt_shots:
                self.console.print(f"  - Alternative (player behavior): {len(alt_shots)}")
        else:
            self.console.print("Shot detection disabled by config")

        goal_events = []
        if self.config.events.detect_goals:
            goal_events = detector.detect_goals(trajectory, shot_events)
            self.console.print(f"Detected {len(goal_events)} potential goals")
        else:
            self.console.print("Goal detection disabled by config")

        pass_events = []
        if self.config.events.detect_passes:
            pass_events = infer_pass_events(
                tracks=tracks,
                fps=video_metadata["fps"],
                config=self._pass_inference_config(self.config),
            )
            self.console.print(f"Detected {len(pass_events)} potential passes")

        set_piece_events = []
        if self.config.events.detect_set_pieces:
            set_piece_events = infer_set_piece_events(
                tracks=tracks,
                fps=video_metadata["fps"],
                frame_width=video_metadata.get("width"),
                frame_height=video_metadata.get("height"),
            )
            self.console.print(f"Detected {len(set_piece_events)} potential set-pieces")

        team_analytics = context.get("team_analytics")
        if not isinstance(team_analytics, dict):
            team_analytics_path = output_dir / "team_analytics.json"
            if team_analytics_path.exists():
                try:
                    with open(team_analytics_path) as f:
                        team_analytics = json.load(f)
                except Exception:
                    team_analytics = {}
            else:
                team_analytics = {}

        tactical_events = []
        if self.config.events.detect_tactical:
            tactical_events = infer_tactical_events(
                tracks=tracks,
                team_analytics=team_analytics,
                fps=video_metadata["fps"],
                config=self._tactical_inference_config(self.config, fps=video_metadata["fps"]),
            )
            tactical_counts = self._event_type_counts(tactical_events)
            self.console.print(
                "Detected "
                f"{len(tactical_events)} tactical events "
                f"(build_up={int(tactical_counts.get('build_up', 0))}, "
                f"pressing={int(tactical_counts.get('pressing', 0))}, "
                f"defending={int(tactical_counts.get('defending', 0))}, "
                f"transition={int(tactical_counts.get('transition', 0))})"
            )

        # Combine all events
        all_events = shot_events + goal_events + pass_events + set_piece_events + tactical_events
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

        event_type_counts = self._event_type_counts(all_events)

        # Report metrics
        context["event_detection_items_processed"] = len(ball_tracks)
        context["event_detection_custom_metrics"] = {
            "shots": int(event_type_counts.get("shot", len(shot_events))),
            "velocity_shots": len(velocity_shots),
            "alternative_shots": len(alt_shots),
            "goals": int(event_type_counts.get("goal", len(goal_events))),
            "passes": int(event_type_counts.get("pass", len(pass_events))),
            "set_pieces": self._set_piece_count(event_type_counts),
            "tactical_events": self._tactical_count(event_type_counts),
            "tactical_build_ups": int(event_type_counts.get("build_up", 0)),
            "tactical_pressing": int(event_type_counts.get("pressing", 0)),
            "tactical_defending": int(event_type_counts.get("defending", 0)),
            "tactical_transitions": int(event_type_counts.get("transition", 0)),
            "kickoffs": int(event_type_counts.get("kickoff", 0)),
            "throw_ins": int(event_type_counts.get("throw_in", 0)),
            "corner_kicks": int(event_type_counts.get("corner_kick", 0)),
            "free_kicks": int(event_type_counts.get("free_kick", 0)),
            "goal_kicks": int(event_type_counts.get("goal_kick", 0)),
            "ball_coverage": coverage,
            "trajectory_points": len(trajectory.points) if trajectory else 0,
        }

        return context


class MatchStatsStage(PipelineStage):
    """Stage E.2: Unified team-level match stats artifact."""

    def __init__(self, config: PipelineConfig):
        super().__init__("match_stats", config)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Build and persist match_stats.json from events and team analytics."""
        from src.analytics import build_match_stats

        output_dir = Path(context["output_dir"])
        match_stats_path = output_dir / "match_stats.json"

        if context.get("resume", False) and match_stats_path.exists():
            self.console.print(
                f"[bold yellow]✓ Using cached match stats from {match_stats_path.name}[/bold yellow]"
            )
            with open(match_stats_path) as f:
                match_stats = json.load(f)

            summary = match_stats.get("summary", {})
            totals = match_stats.get("totals", {})
            context["match_stats"] = match_stats
            context["match_stats_path"] = str(match_stats_path)
            context["match_stats_items_processed"] = int(summary.get("events_processed", 0))
            context["match_stats_custom_metrics"] = {
                "cached": True,
                "teams_detected": len(summary.get("teams_detected", [])),
                "shots": int(totals.get("shots", 0)),
                "goals": int(totals.get("goals", 0)),
                "passes": int(totals.get("passes", 0)),
                "set_pieces": int(totals.get("set_pieces", 0)),
                "events_without_team": int(summary.get("events_without_team", 0)),
            }
            return context

        events = context.get("events", [])
        tracks = context.get("tracks", [])

        team_analytics = context.get("team_analytics")
        if not isinstance(team_analytics, dict):
            team_analytics_path = output_dir / "team_analytics.json"
            if team_analytics_path.exists():
                try:
                    with open(team_analytics_path) as f:
                        team_analytics = json.load(f)
                except Exception:
                    team_analytics = {}
            else:
                team_analytics = {}

        video_metadata = context.get("video_metadata", {})
        fps = float(video_metadata.get("fps", 30.0) or 30.0)

        stats_payload = build_match_stats(
            events=events,
            team_analytics=team_analytics,
            tracks=tracks,
            fps=fps,
        )

        sources = {}
        if (output_dir / "events.jsonl").exists():
            sources["events"] = "events.jsonl"
        if (output_dir / "team_analytics.json").exists():
            sources["team_analytics"] = "team_analytics.json"
        if (output_dir / "tracks.parquet").exists():
            sources["tracks"] = "tracks.parquet"
        elif (output_dir / "tracks.jsonl").exists():
            sources["tracks"] = "tracks.jsonl"

        match_stats_artifact = {
            "schema_version": stats_payload.get("schema_version", "1.0"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "video_id": Path(context["video_path"]).stem,
            "summary": stats_payload.get("summary", {}),
            "teams": stats_payload.get("teams", {}),
            "totals": stats_payload.get("totals", {}),
            "sources": sources,
        }

        with open(match_stats_path, "w") as f:
            json.dump(match_stats_artifact, f, indent=2)

        summary = match_stats_artifact.get("summary", {})
        totals = match_stats_artifact.get("totals", {})

        self.console.print(f"Saved match stats to: {match_stats_path}")
        self.console.print(
            "  Match stats totals: "
            f"shots={int(totals.get('shots', 0))}, "
            f"goals={int(totals.get('goals', 0))}, "
            f"passes={int(totals.get('passes', 0))}, "
            f"set_pieces={int(totals.get('set_pieces', 0))}"
        )

        context["match_stats"] = match_stats_artifact
        context["match_stats_path"] = str(match_stats_path)
        context["match_stats_items_processed"] = int(summary.get("events_processed", 0))
        context["match_stats_custom_metrics"] = {
            "teams_detected": len(summary.get("teams_detected", [])),
            "shots": int(totals.get("shots", 0)),
            "goals": int(totals.get("goals", 0)),
            "passes": int(totals.get("passes", 0)),
            "set_pieces": int(totals.get("set_pieces", 0)),
            "events_without_team": int(summary.get("events_without_team", 0)),
        }
        return context


class HighlightGenerationStage(PipelineStage):
    """Stage E.5: Generate highlight segments from event/audio/action signals."""

    def __init__(self, config: PipelineConfig):
        super().__init__("highlight_generation", config)

    @staticmethod
    def _scale_candidates(candidates: list, scale: float) -> list:
        """Scale candidate scores by source weight."""
        scaled = []
        for candidate in candidates:
            candidate.score = max(0.0, min(1.5, candidate.score * scale))
            scaled.append(candidate)
        return scaled

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        """Convert numpy-heavy metadata into JSON-safe primitives."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): HighlightGenerationStage._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [HighlightGenerationStage._to_jsonable(v) for v in value]
        return str(value)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate highlight candidates, segments, and optional clips."""
        highlight_config = self.config.highlights
        output_dir = Path(context["output_dir"])

        if not highlight_config.enabled:
            self.console.print("Highlight generation disabled, skipping")
            context["highlight_generation_items_processed"] = 0
            context["highlight_generation_custom_metrics"] = {"enabled": False}
            return context

        candidates_path = output_dir / "highlight_candidates.jsonl"
        highlights_path = output_dir / "highlights.json"
        highlights_csv_path = output_dir / "highlights.csv"
        manifest_path = output_dir / "highlights_manifest.json"

        if (
            context.get("resume", False)
            and candidates_path.exists()
            and highlights_path.exists()
            and manifest_path.exists()
        ):
            self.console.print(
                f"[bold yellow]✓ Using cached highlights from {highlights_path.name}[/bold yellow]"
            )
            with open(highlights_path) as f:
                highlights_data = json.load(f)

            context["highlights"] = highlights_data
            context["highlights_path"] = str(highlights_path)
            segment_count = len(highlights_data.get("segments", []))
            context["highlight_generation_items_processed"] = segment_count
            context["highlight_generation_custom_metrics"] = {
                "cached": True,
                "segments": segment_count,
                "candidates": highlights_data.get("summary", {}).get("candidates_total", 0),
            }
            return context

        from src.events.highlights import (
            build_action_candidates,
            build_event_candidates,
            build_segments_from_candidates,
            extract_audio_energy_spikes,
            extract_clip,
            segment_to_dict,
            select_highlight_segments,
        )
        import pandas as pd

        video_path = Path(context["video_path"])
        metadata = context.get("video_metadata", {})
        duration = float(metadata.get("duration", 0.0))
        fps = float(metadata.get("fps", 30.0))
        events = context.get("events", [])
        tracks = context.get("tracks", [])

        self.console.print("Generating highlights from event/audio/action signals...")

        event_candidates = build_event_candidates(
            events=events,
            include_goals=highlight_config.event.include_goals,
            include_shots=highlight_config.event.include_shots,
            goal_weight=highlight_config.event.goal_weight,
            shot_weight=highlight_config.event.shot_weight,
            min_confidence=highlight_config.event.min_confidence,
        )

        audio_candidates = []
        if highlight_config.audio.enabled:
            audio_candidates = extract_audio_energy_spikes(
                video_path=video_path,
                sample_rate=highlight_config.audio.sample_rate,
                window_seconds=highlight_config.audio.window_seconds,
                hop_seconds=highlight_config.audio.hop_seconds,
                min_z_score=highlight_config.audio.min_z_score,
                min_abs_rms=highlight_config.audio.min_abs_rms,
                min_gap_seconds=highlight_config.audio.min_gap_seconds,
                max_spikes=highlight_config.audio.max_spikes,
            )
            audio_candidates = self._scale_candidates(audio_candidates, highlight_config.audio.weight)

        action_candidates = []
        if highlight_config.action.enabled:
            action_candidates = build_action_candidates(
                tracks=tracks,
                fps=fps,
                min_speed_pixels_per_sec=highlight_config.action.min_speed_pixels_per_sec,
                player_pressure_radius=highlight_config.action.player_pressure_radius,
                score_quantile=highlight_config.action.score_quantile,
                min_candidate_score=highlight_config.action.min_candidate_score,
                max_candidates=highlight_config.action.max_candidates,
            )
            action_candidates = self._scale_candidates(action_candidates, highlight_config.action.weight)

        all_candidates = sorted(
            event_candidates + audio_candidates + action_candidates,
            key=lambda c: (c.timestamp, -c.score),
        )

        with open(candidates_path, "w") as f:
            for idx, candidate in enumerate(all_candidates, start=1):
                row = {
                    "schema_version": "1.0",
                    "candidate_id": f"cand_{idx:05d}",
                    "timestamp": candidate.timestamp,
                    "frame_idx": candidate.frame_idx,
                    "score": candidate.score,
                    "source": candidate.source,
                    "reason": candidate.reason,
                    "must_include": candidate.must_include,
                    "metadata": self._to_jsonable(candidate.metadata),
                }
                f.write(json.dumps(row) + "\n")

        segments = build_segments_from_candidates(
            candidates=all_candidates,
            duration_seconds=duration,
            pre_roll_seconds=highlight_config.segment.pre_roll_seconds,
            post_roll_seconds=highlight_config.segment.post_roll_seconds,
            merge_gap_seconds=highlight_config.segment.merge_gap_seconds,
        )
        selected_segments = select_highlight_segments(
            segments=segments,
            top_n=highlight_config.segment.top_n,
            min_segment_score=highlight_config.segment.min_segment_score,
        )

        clips_dir = output_dir / highlight_config.export.clips_dir
        clip_failures = 0
        clip_success = 0
        segment_rows: list[dict[str, Any]] = []

        for idx, segment in enumerate(selected_segments, start=1):
            clip_path = None
            if highlight_config.export.save_clips:
                clip_file = clips_dir / f"highlight_{idx:03d}_{segment.start:.1f}_{segment.end:.1f}.mp4"
                ok, error = extract_clip(
                    video_path=video_path,
                    output_path=clip_file,
                    start_time=segment.start,
                    end_time=segment.end,
                    video_codec=highlight_config.export.clip_video_codec,
                    audio_codec=highlight_config.export.clip_audio_codec,
                )
                if ok:
                    clip_success += 1
                    clip_path = str(clip_file)
                else:
                    clip_failures += 1
                    clip_path = None
                    self.console.print(
                        f"[yellow]Warning: clip export failed for segment {idx}: {error}[/yellow]"
                    )

            segment_rows.append(
                segment_to_dict(
                    segment=segment,
                    segment_id=f"highlight_{idx:03d}",
                    clip_path=clip_path,
                )
            )

        if segment_rows:
            pd.DataFrame(segment_rows).to_csv(highlights_csv_path, index=False)
        else:
            pd.DataFrame(
                columns=[
                    "segment_id",
                    "start_time",
                    "end_time",
                    "duration",
                    "score",
                    "must_include",
                    "reasons",
                    "sources",
                    "candidate_count",
                    "clip_path",
                ]
            ).to_csv(highlights_csv_path, index=False)

        highlights_data = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "video_path": str(video_path),
            "segments": segment_rows,
            "summary": {
                "candidates_total": len(all_candidates),
                "event_candidates": len(event_candidates),
                "audio_candidates": len(audio_candidates),
                "action_candidates": len(action_candidates),
                "segments_selected": len(segment_rows),
                "clips_exported": clip_success,
                "clip_failures": clip_failures,
            },
        }

        with open(highlights_path, "w") as f:
            json.dump(highlights_data, f, indent=2)

        manifest_data = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": highlight_config.model_dump(),
            "inputs": {
                "video_path": str(video_path),
                "duration_seconds": duration,
                "fps": fps,
                "events_count": len(events),
                "tracks_count": len(tracks),
            },
            "outputs": {
                "highlight_candidates_path": str(candidates_path),
                "highlights_path": str(highlights_path),
                "highlights_csv_path": str(highlights_csv_path),
                "clips_dir": str(clips_dir),
            },
            "summary": highlights_data["summary"],
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)

        self.console.print(f"Saved highlight candidates to: {candidates_path}")
        self.console.print(f"Saved highlights to: {highlights_path}")
        self.console.print(f"Saved highlights CSV to: {highlights_csv_path}")
        if highlight_config.export.save_clips:
            self.console.print(f"Saved clips to: {clips_dir} ({clip_success} exported, {clip_failures} failed)")
        self.console.print(
            "  Candidates: "
            f"{len(all_candidates)} (event={len(event_candidates)}, audio={len(audio_candidates)}, action={len(action_candidates)}), "
            f"segments selected: {len(segment_rows)}"
        )

        context["highlights"] = highlights_data
        context["highlights_path"] = str(highlights_path)
        context["highlight_generation_items_processed"] = len(segment_rows)
        context["highlight_generation_custom_metrics"] = {
            "candidates": len(all_candidates),
            "event_candidates": len(event_candidates),
            "audio_candidates": len(audio_candidates),
            "action_candidates": len(action_candidates),
            "segments_selected": len(segment_rows),
            "clips_exported": clip_success,
            "clip_failures": clip_failures,
        }
        return context


class PlayerHighlightReelsStage(PipelineStage):
    """Stage E.6: Build per-player reels from fused assignments and highlight segments."""

    def __init__(self, config: PipelineConfig):
        super().__init__("player_highlight_reels", config)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate per-player highlight reels and optional clips."""
        reel_cfg = self.config.highlights.player_reels
        output_dir = Path(context["output_dir"])

        if not self.config.highlights.enabled or not reel_cfg.enabled:
            self.console.print("Player highlight reels disabled, skipping")
            context["player_highlight_reels_items_processed"] = 0
            context["player_highlight_reels_custom_metrics"] = {"enabled": False}
            return context

        reels_json_path = output_dir / "player_highlights.json"
        reels_csv_path = output_dir / "player_highlights.csv"
        reels_manifest_path = output_dir / "player_highlights_manifest.json"

        if (
            context.get("resume", False)
            and reels_json_path.exists()
            and reels_csv_path.exists()
            and reels_manifest_path.exists()
        ):
            self.console.print(
                f"[bold yellow]✓ Using cached player reels from {reels_json_path.name}[/bold yellow]"
            )
            with open(reels_json_path) as f:
                reels_data = json.load(f)
            context["player_highlights"] = reels_data
            count = reels_data.get("summary", {}).get("player_segments_total", 0)
            context["player_highlight_reels_items_processed"] = count
            context["player_highlight_reels_custom_metrics"] = {
                "cached": True,
                "players_with_reels": reels_data.get("summary", {}).get("players_with_reels", 0),
                "player_segments_total": count,
            }
            return context

        from src.events.highlights import extract_clip
        from src.events.player_reels import build_player_reels
        import pandas as pd

        highlights_data = context.get("highlights")
        if highlights_data is None:
            highlights_path = output_dir / "highlights.json"
            if highlights_path.exists():
                with open(highlights_path) as f:
                    highlights_data = json.load(f)

        assignments_data = context.get("player_assignments")
        if assignments_data is None:
            assignments_path = output_dir / "player_assignments.json"
            if assignments_path.exists():
                with open(assignments_path) as f:
                    assignments_data = json.load(f)

        tracks = context.get("tracks", [])
        video_metadata = context.get("video_metadata", {})
        fps = float(video_metadata.get("fps", 30.0))

        if not highlights_data or not assignments_data:
            self.console.print("[yellow]Highlights or player assignments missing, skipping player reels[/yellow]")
            context["player_highlight_reels_items_processed"] = 0
            context["player_highlight_reels_custom_metrics"] = {
                "skipped": True,
                "reason": "missing_inputs",
            }
            return context

        segments = list(highlights_data.get("segments", []))
        assignments = list(assignments_data.get("assignments", []))

        player_reels, summary = build_player_reels(
            segments=segments,
            tracks=tracks,
            assignments=assignments,
            fps=fps,
            max_segments_per_player=reel_cfg.max_segments_per_player,
            min_presence_seconds=reel_cfg.min_presence_seconds,
            min_player_segment_score=reel_cfg.min_player_segment_score,
            min_assignment_confidence=reel_cfg.min_assignment_confidence,
            include_suggested=reel_cfg.include_suggested_assignments,
        )

        # Optional per-player clip extraction.
        clip_success = 0
        clip_failures = 0
        if reel_cfg.save_clips and player_reels:
            video_path = Path(context["video_path"])
            clip_root = output_dir / reel_cfg.clips_dir

            for player in player_reels:
                player_id = player["player_id"]
                player_dir = clip_root / f"player_{player_id}"
                player_dir.mkdir(parents=True, exist_ok=True)

                for idx, segment in enumerate(player["segments"], start=1):
                    clip_name = (
                        f"{segment['segment_id']}_"
                        f"{segment['start_time']:.1f}_{segment['end_time']:.1f}.mp4"
                    )
                    clip_path = player_dir / clip_name

                    ok, error = extract_clip(
                        video_path=video_path,
                        output_path=clip_path,
                        start_time=float(segment["start_time"]),
                        end_time=float(segment["end_time"]),
                        video_codec=self.config.highlights.export.clip_video_codec,
                        audio_codec=self.config.highlights.export.clip_audio_codec,
                    )
                    if ok:
                        clip_success += 1
                        segment["clip_path"] = str(clip_path)
                    else:
                        clip_failures += 1
                        segment["clip_path"] = None
                        self.console.print(
                            f"[yellow]Warning: player clip export failed for player {player_id} segment {idx}: {error}[/yellow]"
                        )

        reels_data = {
            "schema_version": "1.0",
            "video_id": Path(context["video_path"]).stem,
            "players": player_reels,
            "summary": {
                **summary,
                "clips_exported": clip_success,
                "clip_failures": clip_failures,
            },
        }

        with open(reels_json_path, "w") as f:
            json.dump(reels_data, f, indent=2)

        flat_rows = []
        for player in player_reels:
            for segment in player["segments"]:
                flat_rows.append(
                    {
                        "player_id": player["player_id"],
                        "player_name": player.get("player_name"),
                        **segment,
                    }
                )
        if flat_rows:
            pd.DataFrame(flat_rows).to_csv(reels_csv_path, index=False)
        else:
            pd.DataFrame(
                columns=[
                    "player_id",
                    "player_name",
                    "segment_id",
                    "start_time",
                    "end_time",
                    "duration",
                    "base_segment_score",
                    "player_segment_score",
                    "presence_seconds",
                    "presence_ratio",
                    "track_ids",
                    "track_count",
                    "assignment_confidence_avg",
                    "activity_score",
                    "reasons",
                    "sources",
                    "clip_path",
                ]
            ).to_csv(reels_csv_path, index=False)

        manifest = {
            "schema_version": "1.0",
            "config": reel_cfg.model_dump(),
            "inputs": {
                "highlights_segments": len(segments),
                "assignments": len(assignments),
                "tracks": len(tracks),
                "fps": fps,
            },
            "outputs": {
                "player_highlights_json": str(reels_json_path),
                "player_highlights_csv": str(reels_csv_path),
            },
            "summary": reels_data["summary"],
        }
        with open(reels_manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self.console.print(f"Saved player highlights to: {reels_json_path}")
        self.console.print(f"Saved player highlights CSV to: {reels_csv_path}")
        self.console.print(
            "  Players with reels: "
            f"{summary['players_with_reels']}, "
            f"segments: {summary['player_segments_total']}"
        )
        if reel_cfg.save_clips:
            self.console.print(
                f"  Player clips exported: {clip_success}, failed: {clip_failures}"
            )

        context["player_highlights"] = reels_data
        context["player_highlight_reels_items_processed"] = summary["player_segments_total"]
        context["player_highlight_reels_custom_metrics"] = {
            **summary,
            "clips_exported": clip_success,
            "clip_failures": clip_failures,
        }
        return context


class CrossMatchReportingStage(PipelineStage):
    """Stage E.7: Cross-match season reporting and export templates."""

    def __init__(self, config: PipelineConfig):
        super().__init__("cross_match_reporting", config)

    @staticmethod
    def _write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
        """Write rows to CSV with stable schema for empty exports."""
        import pandas as pd

        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
        else:
            pd.DataFrame(columns=columns).to_csv(path, index=False)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Aggregate season trends across runs and export coach/player templates."""
        cfg = self.config.cross_match
        output_dir = Path(context["output_dir"])

        if not cfg.enabled:
            self.console.print("Cross-match reporting disabled, skipping")
            context["cross_match_reporting_items_processed"] = 0
            context["cross_match_reporting_custom_metrics"] = {"enabled": False}
            return context

        report_path = output_dir / "cross_match_report.json"
        matches_csv_path = output_dir / "cross_match_match_trends.csv"
        players_csv_path = output_dir / "cross_match_player_trends.csv"
        coach_template_path = output_dir / "coach_report_template.md"
        player_templates_path = output_dir / "player_report_templates.md"

        if (
            context.get("resume", False)
            and report_path.exists()
            and matches_csv_path.exists()
            and players_csv_path.exists()
            and coach_template_path.exists()
            and player_templates_path.exists()
        ):
            self.console.print(
                f"[bold yellow]✓ Using cached cross-match report from {report_path.name}[/bold yellow]"
            )
            with open(report_path) as f:
                report_artifact = json.load(f)

            summary = report_artifact.get("summary", {})
            context["cross_match_report"] = report_artifact
            context["cross_match_report_path"] = str(report_path)
            context["cross_match_reporting_items_processed"] = int(summary.get("matches_analyzed", 0))
            context["cross_match_reporting_custom_metrics"] = {
                "cached": True,
                "matches_analyzed": int(summary.get("matches_analyzed", 0)),
                "unique_players": int(summary.get("unique_players", 0)),
            }
            return context

        runs_root = (
            Path(cfg.runs_root).expanduser()
            if cfg.runs_root
            else output_dir.parent
        )
        if not runs_root.exists():
            self.console.print(
                f"[yellow]Cross-match runs root not found ({runs_root}), skipping stage[/yellow]"
            )
            context["cross_match_reporting_items_processed"] = 0
            context["cross_match_reporting_custom_metrics"] = {
                "enabled": True,
                "skipped_reason": "runs_root_not_found",
                "runs_root": str(runs_root),
            }
            return context

        from src.export import build_cross_match_report

        payload = build_cross_match_report(
            runs_root=runs_root,
            current_run=output_dir,
            config=cfg,
        )

        report_artifact = payload.get("report", {})
        match_rows = list(payload.get("match_rows", []) or [])
        player_rows = list(payload.get("player_rows", []) or [])
        coach_template = str(payload.get("coach_template", ""))
        player_templates = str(payload.get("player_templates", ""))

        with open(report_path, "w") as f:
            json.dump(report_artifact, f, indent=2)
        with open(coach_template_path, "w") as f:
            f.write(coach_template)
        with open(player_templates_path, "w") as f:
            f.write(player_templates)

        self._write_csv(
            match_rows,
            matches_csv_path,
            columns=[
                "schema_version",
                "run_name",
                "timestamp",
                "goals",
                "shots",
                "highlights_segments",
                "players_with_reels",
                "player_reel_segments_total",
                "passes_inferred",
                "possession_frames",
                "dominant_team",
                "possession_share_ours",
                "possession_share_opponent",
                "high_press_rate_ours",
                "high_press_rate_opponent",
            ],
        )
        self._write_csv(
            player_rows,
            players_csv_path,
            columns=[
                "schema_version",
                "player_id",
                "player_name",
                "matches_with_reels",
                "total_segments",
                "total_highlight_seconds",
                "avg_segment_score",
                "best_segment_score",
                "goal_tagged_segments",
                "shot_tagged_segments",
                "share_of_all_segments",
            ],
        )

        summary = report_artifact.get("summary", {})
        self.console.print(f"Saved cross-match report to: {report_path}")
        self.console.print(f"Saved match trends CSV to: {matches_csv_path}")
        self.console.print(f"Saved player trends CSV to: {players_csv_path}")
        self.console.print(f"Saved coach template to: {coach_template_path}")
        self.console.print(f"Saved player templates to: {player_templates_path}")
        self.console.print(
            "  Cross-match summary: "
            f"matches={int(summary.get('matches_analyzed', 0))}, "
            f"players={int(summary.get('unique_players', 0))}"
        )

        context["cross_match_report"] = report_artifact
        context["cross_match_report_path"] = str(report_path)
        context["cross_match_reporting_items_processed"] = int(summary.get("matches_analyzed", 0))
        context["cross_match_reporting_custom_metrics"] = {
            "matches_analyzed": int(summary.get("matches_analyzed", 0)),
            "unique_players": int(summary.get("unique_players", 0)),
            "top_players_exported": len(report_artifact.get("players", {}).get("top_players", [])),
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
                    self.check_cancelled(context)

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
    pipeline.add_stage(FieldNormalizationStage(pipeline_config))
    pipeline.add_stage(ProfileIngestionStage(pipeline_config))
    pipeline.add_stage(PlayerIdentityStage(pipeline_config))
    pipeline.add_stage(TeamAnalyticsStage(pipeline_config))
    pipeline.add_stage(EventDetectionStage(pipeline_config))
    pipeline.add_stage(MatchStatsStage(pipeline_config))
    pipeline.add_stage(HighlightGenerationStage(pipeline_config))
    pipeline.add_stage(PlayerHighlightReelsStage(pipeline_config))
    pipeline.add_stage(CrossMatchReportingStage(pipeline_config))

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
