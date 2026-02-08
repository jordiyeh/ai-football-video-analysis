"""Deterministic set-piece inference from ball restart patterns."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any

from src.events.detection import Event


UNKNOWN_TEAM = "unknown"
SET_PIECE_INFERENCE_ALGO_VERSION = "1.0"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely cast arbitrary values to float."""
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    """Safely cast arbitrary values to int."""
    try:
        return int(value)
    except Exception:
        return default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp value into an inclusive numeric range."""
    return max(lower, min(upper, value))


def _cfg_value(config: Any, key: str, default: Any) -> Any:
    """Read a config value from dict/object with fallback."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_team_label(track: dict[str, Any]) -> str:
    """Resolve canonical team label from a player track row."""
    team_name = track.get("team_name")
    if team_name is not None:
        team_text = str(team_name).strip()
        if team_text and team_text.lower() != UNKNOWN_TEAM:
            return team_text

    team_id = _safe_int(track.get("team_id"), default=None)
    if team_id is not None and team_id >= 0:
        return f"team_{team_id}"
    return UNKNOWN_TEAM


def _center_from_bbox(track: dict[str, Any]) -> tuple[float, float] | None:
    """Return bbox center for valid [x1, y1, x2, y2] boxes."""
    bbox = track.get("bbox")
    if not isinstance(bbox, list | tuple) or len(bbox) < 4:
        return None

    x1 = _safe_float(bbox[0], default=float("nan"))
    y1 = _safe_float(bbox[1], default=float("nan"))
    x2 = _safe_float(bbox[2], default=float("nan"))
    y2 = _safe_float(bbox[3], default=float("nan"))
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _resolve_image_xy(track: dict[str, Any]) -> tuple[float, float] | None:
    """Resolve image-space point from explicit fields or bbox center fallback."""
    image_x = track.get("image_x")
    image_y = track.get("image_y")
    if image_x is not None and image_y is not None:
        return (_safe_float(image_x), _safe_float(image_y))

    image_xy = track.get("image_xy")
    if isinstance(image_xy, list | tuple) and len(image_xy) >= 2:
        return (_safe_float(image_xy[0]), _safe_float(image_xy[1]))

    return _center_from_bbox(track)


def _score_band(distance: float, tolerance: float) -> float:
    """Return normalized closeness score (1 at center, 0 outside tolerance)."""
    if tolerance <= 1e-6:
        return 0.0
    return _clamp(1.0 - (distance / tolerance))


def _cosine_score(
    vector: tuple[float, float],
    expected: tuple[float, float],
) -> float:
    """Map cosine similarity into [0, 1] confidence score."""
    vx, vy = vector
    ex, ey = expected
    v_norm = hypot(vx, vy)
    e_norm = hypot(ex, ey)
    if v_norm <= 1e-6 or e_norm <= 1e-6:
        return 0.5
    cosine = ((vx / v_norm) * (ex / e_norm)) + ((vy / v_norm) * (ey / e_norm))
    return _clamp((cosine + 1.0) * 0.5)


@dataclass
class SetPieceInferenceConfig:
    """Configuration for deterministic set-piece heuristics."""

    # Restart candidate extraction (stationary -> kick).
    stationary_speed_px_per_frame: float = 6.0
    stationary_min_frames: int = 4
    restart_min_speed_px_per_frame: float = 12.0
    restart_min_displacement_px: float = 18.0
    max_transition_gap_frames: int = 2
    min_event_gap_seconds: float = 1.5

    # Actor/team assignment.
    actor_max_distance_px: float = 180.0
    actor_frame_search_radius: int = 2

    # Geometric priors.
    edge_margin_ratio_x: float = 0.08
    edge_margin_ratio_y: float = 0.10
    corner_margin_ratio_x: float = 0.10
    corner_margin_ratio_y: float = 0.12
    kickoff_center_radius_ratio_x: float = 0.10
    kickoff_center_radius_ratio_y: float = 0.08
    goal_kick_y_band_ratio: float = 0.12
    goal_kick_center_band_ratio_x: float = 0.22

    # Per-type acceptance thresholds.
    kickoff_min_confidence: float = 0.60
    throw_in_min_confidence: float = 0.56
    corner_kick_min_confidence: float = 0.60
    free_kick_min_confidence: float = 0.52
    goal_kick_min_confidence: float = 0.58

    @classmethod
    def from_any(cls, config: Any | None) -> "SetPieceInferenceConfig":
        """Build config from dict/object/defaults."""
        if isinstance(config, cls):
            return config

        return cls(
            stationary_speed_px_per_frame=float(
                _cfg_value(config, "stationary_speed_px_per_frame", cls.stationary_speed_px_per_frame)
            ),
            stationary_min_frames=max(
                2,
                int(_cfg_value(config, "stationary_min_frames", cls.stationary_min_frames)),
            ),
            restart_min_speed_px_per_frame=float(
                _cfg_value(config, "restart_min_speed_px_per_frame", cls.restart_min_speed_px_per_frame)
            ),
            restart_min_displacement_px=float(
                _cfg_value(config, "restart_min_displacement_px", cls.restart_min_displacement_px)
            ),
            max_transition_gap_frames=max(
                1,
                int(_cfg_value(config, "max_transition_gap_frames", cls.max_transition_gap_frames)),
            ),
            min_event_gap_seconds=float(
                _cfg_value(config, "min_event_gap_seconds", cls.min_event_gap_seconds)
            ),
            actor_max_distance_px=float(
                _cfg_value(config, "actor_max_distance_px", cls.actor_max_distance_px)
            ),
            actor_frame_search_radius=max(
                0,
                int(_cfg_value(config, "actor_frame_search_radius", cls.actor_frame_search_radius)),
            ),
            edge_margin_ratio_x=float(_cfg_value(config, "edge_margin_ratio_x", cls.edge_margin_ratio_x)),
            edge_margin_ratio_y=float(_cfg_value(config, "edge_margin_ratio_y", cls.edge_margin_ratio_y)),
            corner_margin_ratio_x=float(_cfg_value(config, "corner_margin_ratio_x", cls.corner_margin_ratio_x)),
            corner_margin_ratio_y=float(_cfg_value(config, "corner_margin_ratio_y", cls.corner_margin_ratio_y)),
            kickoff_center_radius_ratio_x=float(
                _cfg_value(config, "kickoff_center_radius_ratio_x", cls.kickoff_center_radius_ratio_x)
            ),
            kickoff_center_radius_ratio_y=float(
                _cfg_value(config, "kickoff_center_radius_ratio_y", cls.kickoff_center_radius_ratio_y)
            ),
            goal_kick_y_band_ratio=float(
                _cfg_value(config, "goal_kick_y_band_ratio", cls.goal_kick_y_band_ratio)
            ),
            goal_kick_center_band_ratio_x=float(
                _cfg_value(config, "goal_kick_center_band_ratio_x", cls.goal_kick_center_band_ratio_x)
            ),
            kickoff_min_confidence=float(
                _cfg_value(config, "kickoff_min_confidence", cls.kickoff_min_confidence)
            ),
            throw_in_min_confidence=float(
                _cfg_value(config, "throw_in_min_confidence", cls.throw_in_min_confidence)
            ),
            corner_kick_min_confidence=float(
                _cfg_value(config, "corner_kick_min_confidence", cls.corner_kick_min_confidence)
            ),
            free_kick_min_confidence=float(
                _cfg_value(config, "free_kick_min_confidence", cls.free_kick_min_confidence)
            ),
            goal_kick_min_confidence=float(
                _cfg_value(config, "goal_kick_min_confidence", cls.goal_kick_min_confidence)
            ),
        )


class SetPieceInferencer:
    """Infer set-piece events from restart patterns in tracking data."""

    def __init__(
        self,
        config: SetPieceInferenceConfig | dict[str, Any] | Any | None = None,
        frame_width: int | float | None = None,
        frame_height: int | float | None = None,
    ):
        self.config = SetPieceInferenceConfig.from_any(config)
        self._frame_width = _safe_float(frame_width, default=0.0) if frame_width is not None else 0.0
        self._frame_height = _safe_float(frame_height, default=0.0) if frame_height is not None else 0.0

    def infer(self, tracks: list[dict[str, Any]], fps: float = 30.0) -> list[Event]:
        """Infer kickoff/throw-in/corner/free-kick/goal-kick events from tracks."""
        if fps <= 0:
            fps = 30.0

        player_rows_by_frame: dict[int, list[dict[str, Any]]] = {}
        ball_rows_by_frame: dict[int, dict[str, Any]] = {}
        max_observed_x = 0.0
        max_observed_y = 0.0

        for track in tracks:
            frame_idx = _safe_int(track.get("frame_idx"), default=None)
            track_id = _safe_int(track.get("track_id"), default=None)
            if frame_idx is None or track_id is None:
                continue

            object_type = str(track.get("object_type", "")).strip().lower()
            image_xy = _resolve_image_xy(track)
            if image_xy is None:
                continue

            max_observed_x = max(max_observed_x, float(image_xy[0]))
            max_observed_y = max(max_observed_y, float(image_xy[1]))

            row = {
                "frame_idx": frame_idx,
                "track_id": track_id,
                "timestamp": _safe_float(track.get("timestamp"), default=frame_idx / fps),
                "confidence": _clamp(_safe_float(track.get("confidence"), default=0.0)),
                "image_xy": (float(image_xy[0]), float(image_xy[1])),
            }

            if object_type == "ball":
                existing = ball_rows_by_frame.get(frame_idx)
                if existing is None or row["confidence"] >= existing["confidence"]:
                    ball_rows_by_frame[frame_idx] = row
            elif object_type == "player":
                row["team"] = _resolve_team_label(track)
                player_rows_by_frame.setdefault(frame_idx, []).append(row)

        if not ball_rows_by_frame:
            return []

        frame_width = self._resolve_dimension(self._frame_width, max_observed_x)
        frame_height = self._resolve_dimension(self._frame_height, max_observed_y)

        timeline = [ball_rows_by_frame[idx] for idx in sorted(ball_rows_by_frame.keys())]
        if len(timeline) < (self.config.stationary_min_frames + 1):
            return []

        restart_candidates = self._extract_restart_candidates(timeline)
        if not restart_candidates:
            return []

        events: list[Event] = []
        for candidate in restart_candidates:
            classified = self._classify_candidate(candidate, frame_width, frame_height)
            if classified is None:
                continue

            event_type, confidence, classification_scores, type_factors = classified
            actor_track_id, actor_team, actor_distance = self._nearest_player_actor(
                player_rows_by_frame=player_rows_by_frame,
                frame_idx=int(candidate["restart_frame_idx"]),
                ball_xy=(float(candidate["restart_xy"][0]), float(candidate["restart_xy"][1])),
            )

            metadata = {
                "set_piece_type": event_type,
                "team_id": actor_team,
                "actor_track_id": actor_track_id,
                "actor_distance_px": actor_distance,
                "stationary_start_frame": int(candidate["stationary_start_frame"]),
                "stationary_end_frame": int(candidate["stationary_end_frame"]),
                "stationary_frames": int(candidate["stationary_frames"]),
                "stationary_duration_seconds": float(candidate["stationary_duration_seconds"]),
                "origin_xy": [
                    float(candidate["origin_xy"][0]),
                    float(candidate["origin_xy"][1]),
                ],
                "restart_xy": [
                    float(candidate["restart_xy"][0]),
                    float(candidate["restart_xy"][1]),
                ],
                "restart_speed_px_per_frame": float(candidate["restart_speed_px_per_frame"]),
                "restart_displacement_px": float(candidate["restart_displacement_px"]),
                "classification_scores": classification_scores,
                "confidence_factors": {
                    "generic_quality": float(candidate["generic_quality"]),
                    **type_factors,
                },
                "provenance": {
                    "detector": "set_piece_heuristics",
                    "algorithm_version": SET_PIECE_INFERENCE_ALGO_VERSION,
                    "source": "ball_stationary_restart",
                    "config": {
                        "stationary_speed_px_per_frame": float(self.config.stationary_speed_px_per_frame),
                        "stationary_min_frames": int(self.config.stationary_min_frames),
                        "restart_min_speed_px_per_frame": float(self.config.restart_min_speed_px_per_frame),
                        "restart_min_displacement_px": float(self.config.restart_min_displacement_px),
                        "thresholds": self._threshold_map(),
                    },
                },
            }

            event = Event(
                event_type=event_type,
                frame_idx=int(candidate["restart_frame_idx"]),
                timestamp=float(candidate["timestamp"]),
                confidence=float(confidence),
                location=(
                    float(candidate["origin_xy"][0]),
                    float(candidate["origin_xy"][1]),
                ),
                metadata=metadata,
            )

            if not events:
                events.append(event)
                continue

            previous = events[-1]
            time_gap = event.timestamp - previous.timestamp
            if time_gap < self.config.min_event_gap_seconds:
                if event.confidence > previous.confidence:
                    events[-1] = event
                continue

            events.append(event)

        return events

    def _extract_restart_candidates(self, timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract stationary->restart candidate windows from ball timeline."""
        stationary_segments: list[tuple[int, int]] = []
        segment_start: int | None = None
        segment_end: int | None = None

        for idx in range(1, len(timeline)):
            prev = timeline[idx - 1]
            curr = timeline[idx]
            frame_gap = int(curr["frame_idx"] - prev["frame_idx"])
            if frame_gap <= 0:
                continue

            move_px = hypot(
                float(curr["image_xy"][0]) - float(prev["image_xy"][0]),
                float(curr["image_xy"][1]) - float(prev["image_xy"][1]),
            )
            speed_px_per_frame = move_px / frame_gap
            is_stationary = (
                frame_gap <= self.config.max_transition_gap_frames
                and speed_px_per_frame <= self.config.stationary_speed_px_per_frame
            )

            if is_stationary:
                if segment_start is None:
                    segment_start = idx - 1
                segment_end = idx
                continue

            if segment_start is not None and segment_end is not None:
                stationary_segments.append((segment_start, segment_end))
                segment_start = None
                segment_end = None

        if segment_start is not None and segment_end is not None:
            stationary_segments.append((segment_start, segment_end))

        candidates: list[dict[str, Any]] = []
        for seg_start, seg_end in stationary_segments:
            segment_points = timeline[seg_start : seg_end + 1]
            stationary_frames = len(segment_points)
            if stationary_frames < self.config.stationary_min_frames:
                continue
            if seg_end + 1 >= len(timeline):
                continue

            restart_prev = timeline[seg_end]
            restart_curr = timeline[seg_end + 1]
            frame_gap = int(restart_curr["frame_idx"] - restart_prev["frame_idx"])
            if frame_gap <= 0 or frame_gap > self.config.max_transition_gap_frames:
                continue

            restart_move_px = hypot(
                float(restart_curr["image_xy"][0]) - float(restart_prev["image_xy"][0]),
                float(restart_curr["image_xy"][1]) - float(restart_prev["image_xy"][1]),
            )
            restart_speed = restart_move_px / frame_gap
            if restart_speed < self.config.restart_min_speed_px_per_frame:
                continue

            origin_x = sum(float(row["image_xy"][0]) for row in segment_points) / stationary_frames
            origin_y = sum(float(row["image_xy"][1]) for row in segment_points) / stationary_frames
            restart_x = float(restart_curr["image_xy"][0])
            restart_y = float(restart_curr["image_xy"][1])
            restart_displacement = hypot(restart_x - origin_x, restart_y - origin_y)
            if restart_displacement < self.config.restart_min_displacement_px:
                continue

            stationary_duration_seconds = max(
                0.0,
                float(segment_points[-1]["timestamp"]) - float(segment_points[0]["timestamp"]),
            )

            candidates.append(
                {
                    "restart_frame_idx": int(restart_curr["frame_idx"]),
                    "timestamp": float(restart_curr["timestamp"]),
                    "origin_xy": (origin_x, origin_y),
                    "restart_xy": (restart_x, restart_y),
                    "direction_xy": (restart_x - origin_x, restart_y - origin_y),
                    "stationary_start_frame": int(segment_points[0]["frame_idx"]),
                    "stationary_end_frame": int(segment_points[-1]["frame_idx"]),
                    "stationary_frames": int(stationary_frames),
                    "stationary_duration_seconds": float(stationary_duration_seconds),
                    "restart_speed_px_per_frame": float(restart_speed),
                    "restart_displacement_px": float(restart_displacement),
                    "generic_quality": self._generic_quality(
                        stationary_frames=stationary_frames,
                        restart_speed=restart_speed,
                        restart_displacement=restart_displacement,
                    ),
                }
            )

        return candidates

    def _generic_quality(
        self,
        *,
        stationary_frames: int,
        restart_speed: float,
        restart_displacement: float,
    ) -> float:
        """Compute generic restart quality independent of event subtype."""
        stationary_score = _clamp(
            (float(stationary_frames) - float(self.config.stationary_min_frames) + 1.0)
            / max(1.0, float(self.config.stationary_min_frames) * 2.0)
        )
        speed_score = _clamp(
            (float(restart_speed) - float(self.config.restart_min_speed_px_per_frame))
            / max(1e-6, float(self.config.restart_min_speed_px_per_frame) * 1.5)
        )
        displacement_score = _clamp(
            float(restart_displacement)
            / max(1e-6, float(self.config.restart_min_displacement_px) * 2.0)
        )
        return (0.35 * stationary_score) + (0.35 * speed_score) + (0.30 * displacement_score)

    def _classify_candidate(
        self,
        candidate: dict[str, Any],
        frame_width: float,
        frame_height: float,
    ) -> tuple[str, float, dict[str, float], dict[str, float]] | None:
        """Classify a restart candidate as a specific set-piece event."""
        origin_x = float(candidate["origin_xy"][0])
        origin_y = float(candidate["origin_xy"][1])
        direction = (
            float(candidate["direction_xy"][0]),
            float(candidate["direction_xy"][1]),
        )
        generic_quality = float(candidate["generic_quality"])

        x_norm = _clamp(origin_x / max(1e-6, frame_width))
        y_norm = _clamp(origin_y / max(1e-6, frame_height))

        left_edge = _score_band(x_norm, self.config.edge_margin_ratio_x)
        right_edge = _score_band(1.0 - x_norm, self.config.edge_margin_ratio_x)
        side_edge = max(left_edge, right_edge)

        left_corner = _score_band(x_norm, self.config.corner_margin_ratio_x)
        right_corner = _score_band(1.0 - x_norm, self.config.corner_margin_ratio_x)
        top_corner = _score_band(y_norm, self.config.corner_margin_ratio_y)
        bottom_corner = _score_band(1.0 - y_norm, self.config.corner_margin_ratio_y)

        corner_scores = {
            "top_left": min(left_corner, top_corner),
            "top_right": min(right_corner, top_corner),
            "bottom_left": min(left_corner, bottom_corner),
            "bottom_right": min(right_corner, bottom_corner),
        }
        dominant_corner = max(corner_scores, key=corner_scores.get)
        corner_location = float(corner_scores[dominant_corner])

        center_x = _score_band(abs(x_norm - 0.5), self.config.kickoff_center_radius_ratio_x)
        center_y = _score_band(abs(y_norm - 0.5), self.config.kickoff_center_radius_ratio_y)
        kickoff_location = center_x * center_y

        goal_top = _score_band(y_norm, self.config.goal_kick_y_band_ratio)
        goal_bottom = _score_band(1.0 - y_norm, self.config.goal_kick_y_band_ratio)
        goal_band = max(goal_top, goal_bottom)
        goal_center = _score_band(abs(x_norm - 0.5), self.config.goal_kick_center_band_ratio_x)
        goal_kick_location = goal_band * goal_center

        throw_in_location = side_edge * (1.0 - (corner_location * 0.8))

        free_kick_location = _clamp(
            1.0
            - max(
                kickoff_location,
                corner_location,
                goal_kick_location,
                side_edge * 0.85,
            )
        )

        corner_expected_vector = {
            "top_left": (1.0, 1.0),
            "top_right": (-1.0, 1.0),
            "bottom_left": (1.0, -1.0),
            "bottom_right": (-1.0, -1.0),
        }[dominant_corner]
        corner_inward = _cosine_score(direction, corner_expected_vector)

        throw_direction_expected = (1.0, 0.0) if left_edge >= right_edge else (-1.0, 0.0)
        throw_inward = _cosine_score(direction, throw_direction_expected)

        goal_direction_expected = (0.0, 1.0) if goal_top >= goal_bottom else (0.0, -1.0)
        goal_outward = _cosine_score(direction, goal_direction_expected)

        specialized_strength = max(
            kickoff_location,
            throw_in_location,
            corner_location,
            goal_kick_location,
        )
        free_kick_base = (0.25 * free_kick_location) + (0.75 * generic_quality)
        free_kick_score = free_kick_base * (1.0 - (0.55 * specialized_strength))

        classification_scores = {
            "kickoff": (0.60 * kickoff_location) + (0.40 * generic_quality),
            "throw_in": (0.50 * throw_in_location) + (0.20 * throw_inward) + (0.30 * generic_quality),
            "corner_kick": (0.55 * corner_location) + (0.20 * corner_inward) + (0.25 * generic_quality),
            "free_kick": free_kick_score,
            "goal_kick": (0.50 * goal_kick_location) + (0.20 * goal_outward) + (0.30 * generic_quality),
        }

        thresholds = self._threshold_map()
        eligible_scores = {
            event_type: float(score)
            for event_type, score in classification_scores.items()
            if float(score) >= float(thresholds[event_type])
        }
        if not eligible_scores:
            return None

        event_type = max(eligible_scores, key=eligible_scores.get)
        confidence = _clamp(eligible_scores[event_type])
        type_factors = {
            "location_score": float(
                {
                    "kickoff": kickoff_location,
                    "throw_in": throw_in_location,
                    "corner_kick": corner_location,
                    "free_kick": free_kick_location,
                    "goal_kick": goal_kick_location,
                }[event_type]
            ),
            "direction_score": float(
                {
                    "kickoff": 0.5,
                    "throw_in": throw_inward,
                    "corner_kick": corner_inward,
                    "free_kick": 0.5,
                    "goal_kick": goal_outward,
                }[event_type]
            ),
            "threshold": float(thresholds[event_type]),
        }
        return event_type, confidence, classification_scores, type_factors

    def _nearest_player_actor(
        self,
        *,
        player_rows_by_frame: dict[int, list[dict[str, Any]]],
        frame_idx: int,
        ball_xy: tuple[float, float],
    ) -> tuple[int | None, str, float | None]:
        """Find nearest player around frame and resolve likely acting team."""
        best_player: dict[str, Any] | None = None
        best_distance: float | None = None

        for offset in range(0, self.config.actor_frame_search_radius + 1):
            if offset == 0:
                candidate_frames = [frame_idx]
            else:
                candidate_frames = [frame_idx - offset, frame_idx + offset]

            for candidate_frame in candidate_frames:
                players = player_rows_by_frame.get(candidate_frame)
                if not players:
                    continue

                for player in players:
                    distance = hypot(
                        float(player["image_xy"][0]) - float(ball_xy[0]),
                        float(player["image_xy"][1]) - float(ball_xy[1]),
                    )
                    if best_distance is None or distance < best_distance:
                        best_player = player
                        best_distance = distance

            if best_player is not None:
                break

        if best_player is None:
            return None, UNKNOWN_TEAM, None

        actor_track_id = _safe_int(best_player.get("track_id"), default=None)
        actor_team = str(best_player.get("team", UNKNOWN_TEAM))
        actor_distance = float(best_distance) if best_distance is not None else None

        if actor_distance is not None and actor_distance > self.config.actor_max_distance_px:
            return actor_track_id, UNKNOWN_TEAM, actor_distance
        return actor_track_id, actor_team, actor_distance

    @staticmethod
    def _resolve_dimension(configured: float, observed_max: float) -> float:
        """Resolve usable frame dimension from explicit config or observed coords."""
        if configured > 1.0:
            return float(configured)
        if observed_max > 0.0:
            return float(observed_max * 1.05)
        return 1.0

    def _threshold_map(self) -> dict[str, float]:
        """Return current per-type confidence threshold map."""
        return {
            "kickoff": float(self.config.kickoff_min_confidence),
            "throw_in": float(self.config.throw_in_min_confidence),
            "corner_kick": float(self.config.corner_kick_min_confidence),
            "free_kick": float(self.config.free_kick_min_confidence),
            "goal_kick": float(self.config.goal_kick_min_confidence),
        }


def infer_set_piece_events(
    tracks: list[dict[str, Any]],
    fps: float = 30.0,
    config: SetPieceInferenceConfig | dict[str, Any] | Any | None = None,
    frame_width: int | float | None = None,
    frame_height: int | float | None = None,
) -> list[Event]:
    """Convenience wrapper for one-shot set-piece inference."""
    inferencer = SetPieceInferencer(
        config=config,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    return inferencer.infer(tracks=tracks, fps=fps)
