"""Deterministic tactical event inference from team-analytics timelines."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any

from src.events.detection import Event


UNKNOWN_TEAM = "unknown"
TACTICAL_INFERENCE_ALGO_VERSION = "1.0"


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
    """Read config value from dict/object with fallback."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_team_label(raw: Any) -> str:
    """Normalize team labels for timeline rows."""
    if raw is None:
        return UNKNOWN_TEAM
    text = str(raw).strip()
    if not text:
        return UNKNOWN_TEAM
    if text.lower() in {UNKNOWN_TEAM, "none", "null", "-1"}:
        return UNKNOWN_TEAM
    return text


def _resolve_norm_xy(row: dict[str, Any]) -> tuple[float, float] | None:
    """Resolve normalized xy from row fields."""
    norm_x = row.get("owner_norm_x")
    norm_y = row.get("owner_norm_y")
    if norm_x is not None and norm_y is not None:
        return (_safe_float(norm_x), _safe_float(norm_y))

    norm_xy = row.get("owner_norm_xy")
    if isinstance(norm_xy, list | tuple) and len(norm_xy) >= 2:
        return (_safe_float(norm_xy[0]), _safe_float(norm_xy[1]))

    return None


@dataclass
class TacticalInferenceConfig:
    """Configuration for deterministic tactical event classification."""

    build_up_min_frames: int = 16
    build_up_min_progress_norm: float = 0.10
    build_up_min_carrier_changes: int = 1

    pressing_min_frames: int = 6
    pressing_min_pressure_score: float = 0.62

    defending_min_frames: int = 8
    defending_max_nearest_distance_norm: float = 0.14
    defending_min_defenders_within_radius: float = 1.2

    transition_max_gap_frames: int = 12
    transition_min_displacement_norm: float = 0.10
    transition_min_previous_possession_frames: int = 2
    transition_min_new_possession_frames: int = 2

    min_event_separation_seconds: float = 1.2
    min_confidence: float = 0.20
    max_confidence: float = 0.98

    @classmethod
    def from_any(cls, config: Any | None) -> "TacticalInferenceConfig":
        """Build config from dict/object/defaults."""
        if isinstance(config, cls):
            return config

        return cls(
            build_up_min_frames=max(
                1,
                int(_cfg_value(config, "build_up_min_frames", cls.build_up_min_frames)),
            ),
            build_up_min_progress_norm=float(
                _cfg_value(config, "build_up_min_progress_norm", cls.build_up_min_progress_norm)
            ),
            build_up_min_carrier_changes=max(
                0,
                int(_cfg_value(config, "build_up_min_carrier_changes", cls.build_up_min_carrier_changes)),
            ),
            pressing_min_frames=max(
                1,
                int(_cfg_value(config, "pressing_min_frames", cls.pressing_min_frames)),
            ),
            pressing_min_pressure_score=float(
                _cfg_value(config, "pressing_min_pressure_score", cls.pressing_min_pressure_score)
            ),
            defending_min_frames=max(
                1,
                int(_cfg_value(config, "defending_min_frames", cls.defending_min_frames)),
            ),
            defending_max_nearest_distance_norm=float(
                _cfg_value(
                    config,
                    "defending_max_nearest_distance_norm",
                    cls.defending_max_nearest_distance_norm,
                )
            ),
            defending_min_defenders_within_radius=float(
                _cfg_value(
                    config,
                    "defending_min_defenders_within_radius",
                    cls.defending_min_defenders_within_radius,
                )
            ),
            transition_max_gap_frames=max(
                1,
                int(_cfg_value(config, "transition_max_gap_frames", cls.transition_max_gap_frames)),
            ),
            transition_min_displacement_norm=float(
                _cfg_value(
                    config,
                    "transition_min_displacement_norm",
                    cls.transition_min_displacement_norm,
                )
            ),
            transition_min_previous_possession_frames=max(
                1,
                int(
                    _cfg_value(
                        config,
                        "transition_min_previous_possession_frames",
                        cls.transition_min_previous_possession_frames,
                    )
                ),
            ),
            transition_min_new_possession_frames=max(
                1,
                int(
                    _cfg_value(
                        config,
                        "transition_min_new_possession_frames",
                        cls.transition_min_new_possession_frames,
                    )
                ),
            ),
            min_event_separation_seconds=float(
                _cfg_value(config, "min_event_separation_seconds", cls.min_event_separation_seconds)
            ),
            min_confidence=float(_cfg_value(config, "min_confidence", cls.min_confidence)),
            max_confidence=float(_cfg_value(config, "max_confidence", cls.max_confidence)),
        )


class TacticalInferencer:
    """Infer tactical events from possession and pressing timelines."""

    def __init__(self, config: TacticalInferenceConfig | dict[str, Any] | Any | None = None):
        self.config = TacticalInferenceConfig.from_any(config)

    def infer(
        self,
        tracks: list[dict[str, Any]],
        team_analytics: dict[str, Any] | None,
        fps: float = 30.0,
    ) -> list[Event]:
        """Infer tactical events from analytics payload (tracks kept for API parity)."""
        del tracks  # Tactical inference currently relies on team-analytics stage outputs.

        if fps <= 0:
            fps = 30.0

        possession_rows = self._extract_possession_rows(team_analytics, fps=fps)
        pressing_rows = self._extract_pressing_rows(team_analytics, fps=fps)

        events: list[Event] = []
        events.extend(self._infer_build_up_events(possession_rows=possession_rows, fps=fps))
        events.extend(self._infer_pressing_events(pressing_rows=pressing_rows, fps=fps))
        events.extend(self._infer_defending_events(pressing_rows=pressing_rows, fps=fps))
        events.extend(self._infer_transition_events(possession_rows=possession_rows, fps=fps))

        events.sort(key=lambda event: event.timestamp)
        return self._deduplicate_events(events)

    @staticmethod
    def _extract_possession_rows(
        team_analytics: dict[str, Any] | None,
        fps: float,
    ) -> list[dict[str, Any]]:
        """Extract possession timeline rows from team analytics."""
        if not isinstance(team_analytics, dict):
            return []

        rows = team_analytics.get("possession_timeline")
        if not isinstance(rows, list):
            return []

        parsed: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue

            frame_idx = _safe_int(row.get("frame_idx"), default=None)
            if frame_idx is None:
                continue

            owner_team = _resolve_team_label(row.get("owner_team"))
            if owner_team == UNKNOWN_TEAM:
                continue

            owner_track_id = _safe_int(row.get("owner_track_id"), default=None)
            timestamp = _safe_float(row.get("timestamp"), default=frame_idx / fps)
            norm_xy = _resolve_norm_xy(row)

            parsed.append(
                {
                    "frame_idx": int(frame_idx),
                    "timestamp": timestamp,
                    "owner_team": owner_team,
                    "owner_track_id": owner_track_id,
                    "owner_norm_xy": norm_xy,
                }
            )

        parsed.sort(key=lambda row: int(row["frame_idx"]))
        return parsed

    @staticmethod
    def _extract_pressing_rows(
        team_analytics: dict[str, Any] | None,
        fps: float,
    ) -> list[dict[str, Any]]:
        """Extract pressing timeline rows from team analytics."""
        if not isinstance(team_analytics, dict):
            return []

        rows = team_analytics.get("pressing_timeline")
        if not isinstance(rows, list):
            return []

        parsed: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue

            frame_idx = _safe_int(row.get("frame_idx"), default=None)
            if frame_idx is None:
                continue

            defending_team = _resolve_team_label(row.get("defending_team"))
            attacking_team = _resolve_team_label(row.get("attacking_team"))
            if defending_team == UNKNOWN_TEAM or attacking_team == UNKNOWN_TEAM:
                continue

            parsed.append(
                {
                    "frame_idx": int(frame_idx),
                    "timestamp": _safe_float(row.get("timestamp"), default=frame_idx / fps),
                    "defending_team": defending_team,
                    "attacking_team": attacking_team,
                    "carrier_track_id": _safe_int(row.get("carrier_track_id"), default=None),
                    "nearest_distance_norm": _safe_float(row.get("nearest_distance_norm"), default=1.0),
                    "defenders_within_radius": _safe_int(row.get("defenders_within_radius"), default=0) or 0,
                    "pressure_score": _safe_float(row.get("pressure_score"), default=0.0),
                    "high_press": bool(row.get("high_press", False)),
                }
            )

        parsed.sort(key=lambda row: int(row["frame_idx"]))
        return parsed

    def _infer_build_up_events(
        self,
        *,
        possession_rows: list[dict[str, Any]],
        fps: float,
    ) -> list[Event]:
        """Infer sustained team build-up events from possession segments."""
        if not possession_rows:
            return []

        segments: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []

        for row in possession_rows:
            if not current:
                current = [row]
                continue

            previous = current[-1]
            contiguous = int(row["frame_idx"]) == int(previous["frame_idx"]) + 1
            same_team = str(row["owner_team"]) == str(previous["owner_team"])

            if contiguous and same_team:
                current.append(row)
            else:
                segments.append(current)
                current = [row]

        if current:
            segments.append(current)

        events: list[Event] = []
        for segment in segments:
            frames = len(segment)
            if frames < self.config.build_up_min_frames:
                continue

            norm_points = [row["owner_norm_xy"] for row in segment if row["owner_norm_xy"] is not None]
            if len(norm_points) < 2:
                continue

            start_xy = norm_points[0]
            end_xy = norm_points[-1]
            dx = float(end_xy[0]) - float(start_xy[0])
            dy = float(end_xy[1]) - float(start_xy[1])
            progression_norm = max(abs(dx), abs(dy))
            displacement_norm = hypot(dx, dy)

            carrier_changes = 0
            previous_carrier = None
            carriers: set[int] = set()
            for row in segment:
                carrier = row.get("owner_track_id")
                if carrier is None:
                    continue
                carriers.add(int(carrier))
                if previous_carrier is not None and carrier != previous_carrier:
                    carrier_changes += 1
                previous_carrier = carrier

            if progression_norm < self.config.build_up_min_progress_norm:
                continue
            if (
                carrier_changes < self.config.build_up_min_carrier_changes
                and frames < (self.config.build_up_min_frames * 2)
            ):
                continue

            duration_score = _clamp(frames / max(1.0, float(self.config.build_up_min_frames) * 2.0))
            progression_score = _clamp(
                progression_norm / max(1e-6, float(self.config.build_up_min_progress_norm) * 2.0)
            )
            carrier_score = _clamp(float(carrier_changes) / max(1.0, float(self.config.build_up_min_carrier_changes + 1)))
            raw_confidence = (
                (0.42 * progression_score)
                + (0.33 * duration_score)
                + (0.25 * max(carrier_score, 0.25))
            )
            confidence = _clamp(
                raw_confidence,
                lower=self.config.min_confidence,
                upper=self.config.max_confidence,
            )

            start_row = segment[0]
            end_row = segment[-1]
            metadata = {
                "tactical_type": "build_up",
                "team_id": str(end_row["owner_team"]),
                "start_frame_idx": int(start_row["frame_idx"]),
                "end_frame_idx": int(end_row["frame_idx"]),
                "duration_frames": frames,
                "duration_seconds": frames / fps,
                "start_norm_xy": [float(start_xy[0]), float(start_xy[1])],
                "end_norm_xy": [float(end_xy[0]), float(end_xy[1])],
                "progression_norm": progression_norm,
                "displacement_norm": displacement_norm,
                "carrier_changes": carrier_changes,
                "carrier_track_ids": sorted(carriers),
                "confidence_factors": {
                    "duration": duration_score,
                    "progression": progression_score,
                    "carrier_changes": carrier_score,
                    "raw": raw_confidence,
                },
                "provenance": {
                    "detector": "tactical_phase_heuristics",
                    "algorithm_version": TACTICAL_INFERENCE_ALGO_VERSION,
                    "source": "team_analytics.possession_timeline",
                    "config": {
                        "build_up_min_frames": int(self.config.build_up_min_frames),
                        "build_up_min_progress_norm": float(self.config.build_up_min_progress_norm),
                        "build_up_min_carrier_changes": int(self.config.build_up_min_carrier_changes),
                    },
                },
            }

            events.append(
                Event(
                    event_type="build_up",
                    frame_idx=int(end_row["frame_idx"]),
                    timestamp=float(end_row["timestamp"]),
                    confidence=confidence,
                    metadata=metadata,
                )
            )

        return events

    def _infer_pressing_events(
        self,
        *,
        pressing_rows: list[dict[str, Any]],
        fps: float,
    ) -> list[Event]:
        """Infer high-intensity pressing runs from pressing timeline."""
        del fps  # Timestamp already embedded in timeline rows.

        if not pressing_rows:
            return []

        runs = self._group_pressing_runs(
            pressing_rows=pressing_rows,
            high_press=True,
        )

        events: list[Event] = []
        for run in runs:
            frames = len(run)
            if frames < self.config.pressing_min_frames:
                continue

            avg_pressure = sum(float(row["pressure_score"]) for row in run) / max(1, frames)
            if avg_pressure < self.config.pressing_min_pressure_score:
                continue

            avg_density = sum(int(row["defenders_within_radius"]) for row in run) / max(1, frames)
            duration_score = _clamp(frames / max(1.0, float(self.config.pressing_min_frames) * 2.0))
            pressure_score = _clamp(avg_pressure)
            density_score = _clamp(avg_density / 4.0)
            raw_confidence = (0.52 * pressure_score) + (0.28 * duration_score) + (0.20 * density_score)
            confidence = _clamp(
                raw_confidence,
                lower=self.config.min_confidence,
                upper=self.config.max_confidence,
            )

            start = run[0]
            end = run[-1]
            metadata = {
                "tactical_type": "pressing",
                "team_id": str(start["defending_team"]),
                "attacking_team": str(start["attacking_team"]),
                "start_frame_idx": int(start["frame_idx"]),
                "end_frame_idx": int(end["frame_idx"]),
                "duration_frames": frames,
                "duration_seconds": float(end["timestamp"]) - float(start["timestamp"]),
                "avg_pressure_score": avg_pressure,
                "avg_defenders_within_radius": avg_density,
                "confidence_factors": {
                    "pressure": pressure_score,
                    "duration": duration_score,
                    "density": density_score,
                    "raw": raw_confidence,
                },
                "provenance": {
                    "detector": "tactical_phase_heuristics",
                    "algorithm_version": TACTICAL_INFERENCE_ALGO_VERSION,
                    "source": "team_analytics.pressing_timeline",
                    "config": {
                        "pressing_min_frames": int(self.config.pressing_min_frames),
                        "pressing_min_pressure_score": float(self.config.pressing_min_pressure_score),
                    },
                },
            }

            events.append(
                Event(
                    event_type="pressing",
                    frame_idx=int(start["frame_idx"]),
                    timestamp=float(start["timestamp"]),
                    confidence=confidence,
                    metadata=metadata,
                )
            )

        return events

    def _infer_defending_events(
        self,
        *,
        pressing_rows: list[dict[str, Any]],
        fps: float,
    ) -> list[Event]:
        """Infer organized defending runs from non-high-press defensive states."""
        del fps

        if not pressing_rows:
            return []

        runs = self._group_pressing_runs(
            pressing_rows=pressing_rows,
            high_press=False,
        )

        events: list[Event] = []
        for run in runs:
            frames = len(run)
            if frames < self.config.defending_min_frames:
                continue

            avg_nearest = sum(float(row["nearest_distance_norm"]) for row in run) / max(1, frames)
            avg_density = sum(int(row["defenders_within_radius"]) for row in run) / max(1, frames)

            if avg_nearest > self.config.defending_max_nearest_distance_norm:
                continue
            if avg_density < self.config.defending_min_defenders_within_radius:
                continue

            compactness_score = _clamp(
                1.0 - (avg_nearest / max(1e-6, float(self.config.defending_max_nearest_distance_norm)))
            )
            density_score = _clamp(
                avg_density / max(1e-6, float(self.config.defending_min_defenders_within_radius) * 2.0)
            )
            duration_score = _clamp(
                frames / max(1.0, float(self.config.defending_min_frames) * 2.0)
            )
            raw_confidence = (0.42 * compactness_score) + (0.30 * density_score) + (0.28 * duration_score)
            confidence = _clamp(
                raw_confidence,
                lower=self.config.min_confidence,
                upper=self.config.max_confidence,
            )

            start = run[0]
            end = run[-1]
            metadata = {
                "tactical_type": "defending",
                "team_id": str(start["defending_team"]),
                "attacking_team": str(start["attacking_team"]),
                "start_frame_idx": int(start["frame_idx"]),
                "end_frame_idx": int(end["frame_idx"]),
                "duration_frames": frames,
                "duration_seconds": float(end["timestamp"]) - float(start["timestamp"]),
                "avg_nearest_distance_norm": avg_nearest,
                "avg_defenders_within_radius": avg_density,
                "confidence_factors": {
                    "compactness": compactness_score,
                    "density": density_score,
                    "duration": duration_score,
                    "raw": raw_confidence,
                },
                "provenance": {
                    "detector": "tactical_phase_heuristics",
                    "algorithm_version": TACTICAL_INFERENCE_ALGO_VERSION,
                    "source": "team_analytics.pressing_timeline",
                    "config": {
                        "defending_min_frames": int(self.config.defending_min_frames),
                        "defending_max_nearest_distance_norm": float(
                            self.config.defending_max_nearest_distance_norm
                        ),
                        "defending_min_defenders_within_radius": float(
                            self.config.defending_min_defenders_within_radius
                        ),
                    },
                },
            }

            events.append(
                Event(
                    event_type="defending",
                    frame_idx=int(start["frame_idx"]),
                    timestamp=float(start["timestamp"]),
                    confidence=confidence,
                    metadata=metadata,
                )
            )

        return events

    def _infer_transition_events(
        self,
        *,
        possession_rows: list[dict[str, Any]],
        fps: float,
    ) -> list[Event]:
        """Infer transition events from fast possession turnovers."""
        del fps

        if len(possession_rows) < 2:
            return []

        frames_by_idx = [int(row["frame_idx"]) for row in possession_rows]
        teams = [str(row["owner_team"]) for row in possession_rows]

        events: list[Event] = []
        for idx in range(1, len(possession_rows)):
            previous = possession_rows[idx - 1]
            current = possession_rows[idx]

            previous_team = str(previous["owner_team"])
            current_team = str(current["owner_team"])
            if previous_team == current_team:
                continue

            frame_gap = int(current["frame_idx"]) - int(previous["frame_idx"])
            if frame_gap <= 0 or frame_gap > self.config.transition_max_gap_frames:
                continue

            prev_xy = previous.get("owner_norm_xy")
            curr_xy = current.get("owner_norm_xy")
            if prev_xy is None or curr_xy is None:
                continue

            displacement_norm = hypot(
                float(curr_xy[0]) - float(prev_xy[0]),
                float(curr_xy[1]) - float(prev_xy[1]),
            )
            if displacement_norm < self.config.transition_min_displacement_norm:
                continue

            previous_streak = self._possession_streak_length(
                frames=frames_by_idx,
                teams=teams,
                index=idx - 1,
                direction=-1,
            )
            new_streak = self._possession_streak_length(
                frames=frames_by_idx,
                teams=teams,
                index=idx,
                direction=1,
            )
            if previous_streak < self.config.transition_min_previous_possession_frames:
                continue
            if new_streak < self.config.transition_min_new_possession_frames:
                continue

            speed_score = _clamp(
                1.0 - (
                    (frame_gap - 1)
                    / max(1.0, float(self.config.transition_max_gap_frames - 1))
                )
            )
            displacement_score = _clamp(
                displacement_norm / max(1e-6, float(self.config.transition_min_displacement_norm) * 2.0)
            )
            stability_score = _clamp(min(previous_streak, new_streak) / 8.0)
            raw_confidence = (0.42 * speed_score) + (0.42 * displacement_score) + (0.16 * stability_score)
            confidence = _clamp(
                raw_confidence,
                lower=self.config.min_confidence,
                upper=self.config.max_confidence,
            )

            metadata = {
                "tactical_type": "transition",
                "team_id": current_team,
                "from_team": previous_team,
                "to_team": current_team,
                "frame_gap": frame_gap,
                "displacement_norm": displacement_norm,
                "from_norm_xy": [float(prev_xy[0]), float(prev_xy[1])],
                "to_norm_xy": [float(curr_xy[0]), float(curr_xy[1])],
                "previous_possession_frames": previous_streak,
                "new_possession_frames": new_streak,
                "confidence_factors": {
                    "speed": speed_score,
                    "displacement": displacement_score,
                    "stability": stability_score,
                    "raw": raw_confidence,
                },
                "provenance": {
                    "detector": "tactical_phase_heuristics",
                    "algorithm_version": TACTICAL_INFERENCE_ALGO_VERSION,
                    "source": "team_analytics.possession_timeline",
                    "config": {
                        "transition_max_gap_frames": int(self.config.transition_max_gap_frames),
                        "transition_min_displacement_norm": float(
                            self.config.transition_min_displacement_norm
                        ),
                        "transition_min_previous_possession_frames": int(
                            self.config.transition_min_previous_possession_frames
                        ),
                        "transition_min_new_possession_frames": int(
                            self.config.transition_min_new_possession_frames
                        ),
                    },
                },
            }

            events.append(
                Event(
                    event_type="transition",
                    frame_idx=int(current["frame_idx"]),
                    timestamp=float(current["timestamp"]),
                    confidence=confidence,
                    metadata=metadata,
                )
            )

        return events

    @staticmethod
    def _possession_streak_length(
        *,
        frames: list[int],
        teams: list[str],
        index: int,
        direction: int,
    ) -> int:
        """Count contiguous same-team possession rows from a pivot index."""
        team = teams[index]
        streak = 0
        current = index

        while 0 <= current < len(frames):
            if teams[current] != team:
                break
            if streak > 0:
                previous = current - direction
                expected_gap = 1
                observed_gap = abs(frames[current] - frames[previous])
                if observed_gap != expected_gap:
                    break
            streak += 1
            current += direction

        return streak

    @staticmethod
    def _group_pressing_runs(
        *,
        pressing_rows: list[dict[str, Any]],
        high_press: bool,
    ) -> list[list[dict[str, Any]]]:
        """Group contiguous rows by defending team and high-press state."""
        runs: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []

        for row in pressing_rows:
            if bool(row["high_press"]) != high_press:
                if current:
                    runs.append(current)
                    current = []
                continue

            if not current:
                current = [row]
                continue

            previous = current[-1]
            contiguous = int(row["frame_idx"]) == int(previous["frame_idx"]) + 1
            same_team = str(row["defending_team"]) == str(previous["defending_team"])

            if contiguous and same_team:
                current.append(row)
            else:
                runs.append(current)
                current = [row]

        if current:
            runs.append(current)

        return runs

    def _deduplicate_events(self, events: list[Event]) -> list[Event]:
        """Deduplicate events per tactical type/team within a time window."""
        if not events:
            return []

        deduplicated: list[Event] = []
        for event in events:
            team_id = None
            if isinstance(event.metadata, dict):
                raw_team = event.metadata.get("team_id")
                if raw_team is not None:
                    team_id = str(raw_team)

            replace_idx = None
            for idx in range(len(deduplicated) - 1, -1, -1):
                existing = deduplicated[idx]
                if existing.event_type != event.event_type:
                    continue

                existing_team = None
                if isinstance(existing.metadata, dict):
                    raw_existing_team = existing.metadata.get("team_id")
                    if raw_existing_team is not None:
                        existing_team = str(raw_existing_team)

                if existing_team != team_id:
                    continue

                if event.timestamp - existing.timestamp <= self.config.min_event_separation_seconds:
                    if event.confidence > existing.confidence:
                        replace_idx = idx
                    break
                if existing.timestamp < event.timestamp - (self.config.min_event_separation_seconds * 1.5):
                    break

            if replace_idx is None:
                deduplicated.append(event)
            else:
                deduplicated[replace_idx] = event

        deduplicated.sort(key=lambda event: event.timestamp)
        return deduplicated


def infer_tactical_events(
    tracks: list[dict[str, Any]],
    team_analytics: dict[str, Any] | None,
    fps: float = 30.0,
    config: TacticalInferenceConfig | dict[str, Any] | Any | None = None,
) -> list[Event]:
    """Convenience wrapper for tactical-event inference."""
    inferencer = TacticalInferencer(config=config)
    return inferencer.infer(
        tracks=tracks,
        team_analytics=team_analytics,
        fps=fps,
    )
