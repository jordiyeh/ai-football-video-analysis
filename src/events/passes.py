"""Deterministic pass inference from possession handoffs."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any

from src.events.detection import Event


UNKNOWN_TEAM = "unknown"
PASS_INFERENCE_ALGO_VERSION = "1.0"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely cast arbitrary value to float."""
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    """Safely cast arbitrary value to int."""
    try:
        return int(value)
    except Exception:
        return default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp numeric value into inclusive range."""
    return max(lower, min(upper, value))


def _cfg_value(config: Any, key: str, default: Any) -> Any:
    """Read config value from dict/object with fallback."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_team_label(track: dict[str, Any]) -> str:
    """Resolve canonical team label for a player track row."""
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
    """Return bbox center if bbox is valid."""
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
    """Resolve image-space center point from explicit fields or bbox fallback."""
    image_x = track.get("image_x")
    image_y = track.get("image_y")
    if image_x is not None and image_y is not None:
        return (_safe_float(image_x), _safe_float(image_y))

    image_xy = track.get("image_xy")
    if isinstance(image_xy, list | tuple) and len(image_xy) >= 2:
        return (_safe_float(image_xy[0]), _safe_float(image_xy[1]))

    return _center_from_bbox(track)


def _majority_smooth(values: list[int | None], window: int) -> list[int | None]:
    """Apply centered majority-vote smoothing to ownership track IDs."""
    if window <= 1 or len(values) <= 2:
        return list(values)

    radius = max(0, window // 2)
    smoothed: list[int | None] = []

    for idx, value in enumerate(values):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        candidate_counts: dict[int, int] = {}

        for candidate in values[start:end]:
            if candidate is None:
                continue
            candidate_counts[candidate] = candidate_counts.get(candidate, 0) + 1

        if not candidate_counts:
            smoothed.append(None)
            continue

        max_count = max(candidate_counts.values())
        tied = [candidate for candidate, count in candidate_counts.items() if count == max_count]
        smoothed.append(value if value in tied else tied[0])

    return smoothed


def _remove_short_runs(values: list[int | None], min_run: int) -> list[int | None]:
    """Replace short ownership runs with neighboring owner or unknown."""
    if min_run <= 1 or not values:
        return list(values)

    output = list(values)
    runs: list[tuple[int, int, int | None]] = []
    start = 0
    current = values[0]

    for idx in range(1, len(values)):
        if values[idx] != current:
            runs.append((start, idx - 1, current))
            start = idx
            current = values[idx]
    runs.append((start, len(values) - 1, current))

    for run_idx, (run_start, run_end, run_value) in enumerate(runs):
        if run_value is None:
            continue

        run_length = run_end - run_start + 1
        if run_length >= min_run:
            continue

        prev_value = runs[run_idx - 1][2] if run_idx > 0 else None
        next_value = runs[run_idx + 1][2] if run_idx + 1 < len(runs) else None
        replacement = prev_value if prev_value == next_value else None
        for idx in range(run_start, run_end + 1):
            output[idx] = replacement

    return output


@dataclass
class PassInferenceConfig:
    """Configuration for deterministic pass inference."""

    possession_max_ball_distance_px: float = 140.0
    possession_smoothing_frames: int = 3
    possession_min_stable_frames: int = 3
    possession_min_segment_frames: int = 4
    pass_min_gap_seconds: float = 0.15
    pass_max_gap_seconds: float = 2.5
    min_pass_confidence: float = 0.20
    max_pass_confidence: float = 0.98

    @classmethod
    def from_any(cls, config: Any | None) -> "PassInferenceConfig":
        """Build config from dict/object/defaults."""
        if isinstance(config, cls):
            return config

        return cls(
            possession_max_ball_distance_px=float(
                _cfg_value(config, "possession_max_ball_distance_px", cls.possession_max_ball_distance_px)
            ),
            possession_smoothing_frames=max(
                1,
                int(_cfg_value(config, "possession_smoothing_frames", cls.possession_smoothing_frames)),
            ),
            possession_min_stable_frames=max(
                1,
                int(_cfg_value(config, "possession_min_stable_frames", cls.possession_min_stable_frames)),
            ),
            possession_min_segment_frames=max(
                1,
                int(_cfg_value(config, "possession_min_segment_frames", cls.possession_min_segment_frames)),
            ),
            pass_min_gap_seconds=float(_cfg_value(config, "pass_min_gap_seconds", cls.pass_min_gap_seconds)),
            pass_max_gap_seconds=float(_cfg_value(config, "pass_max_gap_seconds", cls.pass_max_gap_seconds)),
            min_pass_confidence=float(_cfg_value(config, "min_pass_confidence", cls.min_pass_confidence)),
            max_pass_confidence=float(_cfg_value(config, "max_pass_confidence", cls.max_pass_confidence)),
        )


class PassInferencer:
    """Infer pass events from nearest-player possession handoffs."""

    def __init__(self, config: PassInferenceConfig | dict[str, Any] | Any | None = None):
        self.config = PassInferenceConfig.from_any(config)

    def infer(self, tracks: list[dict[str, Any]], fps: float = 30.0) -> list[Event]:
        """Infer pass events from tracking rows."""
        if fps <= 0:
            fps = 30.0

        player_rows_by_frame: dict[int, dict[int, dict[str, Any]]] = {}
        ball_rows_by_frame: dict[int, dict[str, Any]] = {}

        for track in tracks:
            frame_idx = _safe_int(track.get("frame_idx"), default=None)
            track_id = _safe_int(track.get("track_id"), default=None)
            if frame_idx is None or track_id is None:
                continue

            object_type = str(track.get("object_type", "")).strip().lower()
            image_xy = _resolve_image_xy(track)
            if image_xy is None:
                continue

            row = {
                "frame_idx": frame_idx,
                "track_id": track_id,
                "timestamp": _safe_float(track.get("timestamp"), default=frame_idx / fps),
                "confidence": _safe_float(track.get("confidence"), default=0.0),
                "image_xy": image_xy,
            }

            if object_type == "player":
                row["team"] = _resolve_team_label(track)
                frame_players = player_rows_by_frame.setdefault(frame_idx, {})
                existing = frame_players.get(track_id)
                if existing is None or row["confidence"] >= existing["confidence"]:
                    frame_players[track_id] = row
            elif object_type == "ball":
                existing = ball_rows_by_frame.get(frame_idx)
                if existing is None or row["confidence"] >= existing["confidence"]:
                    ball_rows_by_frame[frame_idx] = row

        if not ball_rows_by_frame or not player_rows_by_frame:
            return []

        possession_timeline: list[dict[str, Any]] = []

        for frame_idx in sorted(ball_rows_by_frame.keys()):
            ball = ball_rows_by_frame[frame_idx]
            players = player_rows_by_frame.get(frame_idx, {})
            if not players:
                continue

            nearest_player = None
            nearest_distance = None
            for player in players.values():
                distance_px = hypot(
                    player["image_xy"][0] - ball["image_xy"][0],
                    player["image_xy"][1] - ball["image_xy"][1],
                )
                if nearest_distance is None or distance_px < nearest_distance:
                    nearest_distance = distance_px
                    nearest_player = player

            raw_owner_track_id = None
            raw_owner_team = UNKNOWN_TEAM
            raw_owner_confidence = None
            if (
                nearest_player is not None
                and nearest_distance is not None
                and nearest_distance <= self.config.possession_max_ball_distance_px
            ):
                raw_owner_track_id = int(nearest_player["track_id"])
                raw_owner_team = str(nearest_player["team"])
                raw_owner_confidence = float(nearest_player["confidence"])

            possession_timeline.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": ball["timestamp"],
                    "ball_track_id": int(ball["track_id"]),
                    "ball_xy": ball["image_xy"],
                    "raw_owner_track_id": raw_owner_track_id,
                    "raw_owner_team": raw_owner_team,
                    "raw_owner_distance_px": nearest_distance,
                    "raw_owner_confidence": raw_owner_confidence,
                    "owner_track_id": None,
                    "owner_team": UNKNOWN_TEAM,
                    "owner_distance_px": None,
                    "owner_confidence": None,
                }
            )

        if not possession_timeline:
            return []

        owner_sequence = [row["raw_owner_track_id"] for row in possession_timeline]
        owner_sequence = _majority_smooth(owner_sequence, self.config.possession_smoothing_frames)
        owner_sequence = _remove_short_runs(owner_sequence, self.config.possession_min_stable_frames)

        for idx, owner_track_id in enumerate(owner_sequence):
            row = possession_timeline[idx]
            if owner_track_id is None:
                continue

            player = player_rows_by_frame.get(int(row["frame_idx"]), {}).get(int(owner_track_id))
            if player is None:
                continue

            row["owner_track_id"] = int(owner_track_id)
            row["owner_team"] = str(player["team"])
            row["owner_confidence"] = float(player["confidence"])
            row["owner_distance_px"] = hypot(
                player["image_xy"][0] - row["ball_xy"][0],
                player["image_xy"][1] - row["ball_xy"][1],
            )

        possession_segments: list[dict[str, Any]] = []
        current_segment: dict[str, Any] | None = None

        for row in possession_timeline:
            owner_track_id = row["owner_track_id"]
            owner_team = row["owner_team"]
            frame_idx = int(row["frame_idx"])

            if owner_track_id is None or owner_team == UNKNOWN_TEAM:
                if current_segment is not None:
                    segment_frames = current_segment["end_frame"] - current_segment["start_frame"] + 1
                    if segment_frames >= self.config.possession_min_segment_frames:
                        current_segment["frames"] = segment_frames
                        possession_segments.append(current_segment)
                current_segment = None
                continue

            if (
                current_segment is not None
                and current_segment["owner_track_id"] == owner_track_id
                and current_segment["owner_team"] == owner_team
                and frame_idx == current_segment["end_frame"] + 1
            ):
                current_segment["end_frame"] = frame_idx
                current_segment["end_time"] = float(row["timestamp"])
                current_segment["end_distance_px"] = row["owner_distance_px"]
                current_segment["end_owner_confidence"] = row["owner_confidence"]
                current_segment["end_ball_xy"] = row["ball_xy"]
            else:
                if current_segment is not None:
                    segment_frames = current_segment["end_frame"] - current_segment["start_frame"] + 1
                    if segment_frames >= self.config.possession_min_segment_frames:
                        current_segment["frames"] = segment_frames
                        possession_segments.append(current_segment)

                current_segment = {
                    "owner_team": owner_team,
                    "owner_track_id": int(owner_track_id),
                    "start_frame": frame_idx,
                    "end_frame": frame_idx,
                    "start_time": float(row["timestamp"]),
                    "end_time": float(row["timestamp"]),
                    "start_distance_px": row["owner_distance_px"],
                    "end_distance_px": row["owner_distance_px"],
                    "start_owner_confidence": row["owner_confidence"],
                    "end_owner_confidence": row["owner_confidence"],
                    "start_ball_xy": row["ball_xy"],
                    "end_ball_xy": row["ball_xy"],
                }

        if current_segment is not None:
            segment_frames = current_segment["end_frame"] - current_segment["start_frame"] + 1
            if segment_frames >= self.config.possession_min_segment_frames:
                current_segment["frames"] = segment_frames
                possession_segments.append(current_segment)

        if len(possession_segments) < 2:
            return []

        events: list[Event] = []
        for idx in range(1, len(possession_segments)):
            previous = possession_segments[idx - 1]
            current = possession_segments[idx]

            if previous["owner_team"] != current["owner_team"]:
                continue
            if previous["owner_track_id"] == current["owner_track_id"]:
                continue

            gap_frames = int(current["start_frame"] - previous["end_frame"] - 1)
            gap_seconds = float(current["start_time"] - previous["end_time"])
            if gap_seconds < self.config.pass_min_gap_seconds:
                continue
            if gap_seconds > self.config.pass_max_gap_seconds:
                continue

            confidence, confidence_factors = self._compute_pass_confidence(previous, current, gap_seconds)
            location = self._pass_location(previous, current)

            metadata = {
                "team_id": str(current["owner_team"]),
                "from_track_id": int(previous["owner_track_id"]),
                "to_track_id": int(current["owner_track_id"]),
                "from_frame_idx": int(previous["end_frame"]),
                "to_frame_idx": int(current["start_frame"]),
                "gap_frames": gap_frames,
                "gap_seconds": gap_seconds,
                "from_possession_frames": int(previous["frames"]),
                "to_possession_frames": int(current["frames"]),
                "confidence_factors": confidence_factors,
                "provenance": {
                    "detector": "possession_handoff",
                    "algorithm_version": PASS_INFERENCE_ALGO_VERSION,
                    "source": "ball_player_proximity",
                    "smoothing_window_frames": int(self.config.possession_smoothing_frames),
                    "min_stable_frames": int(self.config.possession_min_stable_frames),
                    "min_segment_frames": int(self.config.possession_min_segment_frames),
                    "pass_gap_window_seconds": [
                        float(self.config.pass_min_gap_seconds),
                        float(self.config.pass_max_gap_seconds),
                    ],
                },
            }

            events.append(
                Event(
                    event_type="pass",
                    frame_idx=int(current["start_frame"]),
                    timestamp=float(current["start_time"]),
                    confidence=confidence,
                    location=location,
                    metadata=metadata,
                )
            )

        return events

    def _compute_pass_confidence(
        self,
        previous_segment: dict[str, Any],
        current_segment: dict[str, Any],
        gap_seconds: float,
    ) -> tuple[float, dict[str, float]]:
        """Compute deterministic pass confidence with transparent factorization."""
        min_segment_target = max(1.0, float(self.config.possession_min_segment_frames) * 2.0)
        stability = min(
            1.0,
            min(float(previous_segment["frames"]), float(current_segment["frames"])) / min_segment_target,
        )

        distance_values = [
            value
            for value in (previous_segment.get("end_distance_px"), current_segment.get("start_distance_px"))
            if value is not None
        ]
        if distance_values:
            avg_distance = float(sum(distance_values) / len(distance_values))
            proximity = _clamp(
                1.0 - (avg_distance / max(1e-6, float(self.config.possession_max_ball_distance_px))),
            )
        else:
            proximity = 0.5

        gap_span = float(self.config.pass_max_gap_seconds - self.config.pass_min_gap_seconds)
        if gap_span <= 1e-6:
            gap_score = 1.0
        else:
            gap_score = _clamp(
                1.0 - ((gap_seconds - self.config.pass_min_gap_seconds) / gap_span),
            )

        owner_conf_values = [
            value
            for value in (
                previous_segment.get("end_owner_confidence"),
                current_segment.get("start_owner_confidence"),
            )
            if value is not None
        ]
        if owner_conf_values:
            owner_conf = _clamp(float(sum(owner_conf_values) / len(owner_conf_values)))
        else:
            owner_conf = 0.5

        raw_confidence = (
            (0.40 * proximity)
            + (0.30 * stability)
            + (0.20 * gap_score)
            + (0.10 * owner_conf)
        )
        confidence = _clamp(
            raw_confidence,
            lower=self.config.min_pass_confidence,
            upper=self.config.max_pass_confidence,
        )
        return confidence, {
            "proximity": proximity,
            "stability": stability,
            "gap": gap_score,
            "owner_confidence": owner_conf,
            "raw": raw_confidence,
        }

    @staticmethod
    def _pass_location(
        previous_segment: dict[str, Any],
        current_segment: dict[str, Any],
    ) -> tuple[float, float] | None:
        """Compute pass location from segment boundary ball locations."""
        prev_xy = previous_segment.get("end_ball_xy")
        curr_xy = current_segment.get("start_ball_xy")

        if prev_xy is not None and curr_xy is not None:
            return (
                (float(prev_xy[0]) + float(curr_xy[0])) * 0.5,
                (float(prev_xy[1]) + float(curr_xy[1])) * 0.5,
            )
        if curr_xy is not None:
            return (float(curr_xy[0]), float(curr_xy[1]))
        if prev_xy is not None:
            return (float(prev_xy[0]), float(prev_xy[1]))
        return None


def infer_pass_events(
    tracks: list[dict[str, Any]],
    fps: float = 30.0,
    config: PassInferenceConfig | dict[str, Any] | Any | None = None,
) -> list[Event]:
    """Convenience wrapper for one-shot pass inference."""
    inferencer = PassInferencer(config=config)
    return inferencer.infer(tracks=tracks, fps=fps)
