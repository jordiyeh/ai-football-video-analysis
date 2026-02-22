"""Highlight generation from event, audio, and action signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Any, Literal

import numpy as np


@dataclass
class HighlightCandidate:
    """Single highlight trigger candidate."""

    timestamp: float
    score: float
    source: Literal["event", "audio", "action"]
    reason: str
    frame_idx: int | None = None
    must_include: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HighlightSegment:
    """Merged highlight segment composed of one or more candidates."""

    start: float
    end: float
    score: float
    reasons: list[str]
    sources: list[str]
    must_include: bool
    primary_candidate: HighlightCandidate
    candidates: list[HighlightCandidate]

    @property
    def duration(self) -> float:
        """Segment duration in seconds."""
        return max(0.0, self.end - self.start)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely coerce a numeric value to float."""
    try:
        parsed = float(value)
        if np.isnan(parsed) or np.isinf(parsed):
            return default
        return parsed
    except Exception:
        return default


def _event_to_dict(event: Any) -> dict[str, Any]:
    """Convert an event object or dict into a normalized dict."""
    if isinstance(event, dict):
        return event

    return {
        "event_type": getattr(event, "event_type", "other"),
        "frame_idx": getattr(event, "frame_idx", None),
        "timestamp": getattr(event, "timestamp", None),
        "confidence": getattr(event, "confidence", None),
        "location": getattr(event, "location", None),
        "metadata": getattr(event, "metadata", None),
    }


def build_event_candidates(
    events: list[Any],
    include_goals: bool = True,
    include_shots: bool = True,
    goal_weight: float = 1.0,
    shot_weight: float = 0.7,
    min_confidence: float = 0.2,
) -> list[HighlightCandidate]:
    """Create highlight candidates from shot/goal events."""
    candidates: list[HighlightCandidate] = []

    for raw_event in events:
        event = _event_to_dict(raw_event)
        event_type = str(event.get("event_type", "other"))
        confidence = _safe_float(event.get("confidence"), default=0.0)
        timestamp = _safe_float(event.get("timestamp"), default=-1.0)

        if timestamp < 0:
            continue
        if confidence < min_confidence:
            continue

        if event_type == "goal":
            if not include_goals:
                continue
            weight = goal_weight
            must_include = True
        elif event_type == "shot":
            if not include_shots:
                continue
            weight = shot_weight
            must_include = False
        else:
            continue

        score = max(0.0, min(1.5, weight * confidence))
        candidates.append(
            HighlightCandidate(
                timestamp=timestamp,
                score=score,
                source="event",
                reason=event_type,
                frame_idx=event.get("frame_idx"),
                must_include=must_include,
                metadata={
                    "event_type": event_type,
                    "confidence": confidence,
                    "weight": weight,
                    "metadata": event.get("metadata") or {},
                },
            )
        )

    return candidates


def _bbox_center(bbox: Any) -> tuple[float, float] | None:
    """Compute bbox center for (x1, y1, x2, y2)."""
    if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    values = [x1, y1, x2, y2]
    if any(np.isnan(v) or np.isinf(v) for v in values):
        return None
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def build_action_candidates(
    tracks: list[dict[str, Any]],
    fps: float,
    min_speed_pixels_per_sec: float = 220.0,
    player_pressure_radius: float = 120.0,
    score_quantile: float = 0.9,
    min_candidate_score: float = 0.45,
    max_candidates: int = 120,
) -> list[HighlightCandidate]:
    """
    Create highlight candidates from action intensity.

    Action score combines ball speed bursts and player pressure near the ball.
    """
    if not tracks:
        return []
    if fps <= 0:
        return []

    ball_center_by_frame: dict[int, tuple[float, float]] = {}
    player_centers_by_frame: dict[int, list[tuple[float, float]]] = {}

    for track in tracks:
        frame_idx = track.get("frame_idx")
        if frame_idx is None:
            continue
        center = _bbox_center(track.get("bbox"))
        if center is None:
            continue

        object_type = track.get("object_type")
        if object_type == "ball":
            if frame_idx in ball_center_by_frame:
                prev = ball_center_by_frame[frame_idx]
                ball_center_by_frame[frame_idx] = (
                    (prev[0] + center[0]) / 2.0,
                    (prev[1] + center[1]) / 2.0,
                )
            else:
                ball_center_by_frame[frame_idx] = center
        elif object_type == "player":
            player_centers_by_frame.setdefault(frame_idx, []).append(center)

    frame_indices = sorted(ball_center_by_frame.keys())
    if len(frame_indices) < 2:
        return []

    frame_scores: list[tuple[int, float, float, float]] = []
    speeds: list[float] = []
    pressures: list[float] = []

    for i in range(1, len(frame_indices)):
        prev_frame = frame_indices[i - 1]
        frame_idx = frame_indices[i]
        dt = (frame_idx - prev_frame) / fps
        if dt <= 0:
            continue

        prev_center = ball_center_by_frame[prev_frame]
        center = ball_center_by_frame[frame_idx]
        dist = float(np.hypot(center[0] - prev_center[0], center[1] - prev_center[1]))
        speed = dist / dt

        pressure = 0.0
        players = player_centers_by_frame.get(frame_idx, [])
        if players:
            pressure = float(
                sum(
                    1
                    for p in players
                    if np.hypot(center[0] - p[0], center[1] - p[1]) <= player_pressure_radius
                )
            )

        speeds.append(speed)
        pressures.append(pressure)
        frame_scores.append((frame_idx, speed, pressure, frame_idx / fps))

    if not frame_scores:
        return []

    speed_norm_base = max(min_speed_pixels_per_sec, float(np.percentile(speeds, 95)))
    pressure_norm_base = max(1.0, float(np.percentile(pressures, 95)))

    combined_scores: list[float] = []
    enriched_rows: list[tuple[int, float, float, float, float]] = []
    for frame_idx, speed, pressure, timestamp in frame_scores:
        speed_norm = min(1.0, speed / speed_norm_base)
        pressure_norm = min(1.0, pressure / pressure_norm_base)
        combined = 0.75 * speed_norm + 0.25 * pressure_norm
        combined_scores.append(combined)
        enriched_rows.append((frame_idx, speed, pressure, timestamp, combined))

    score_floor = max(min_candidate_score, float(np.quantile(combined_scores, score_quantile)))

    candidates: list[HighlightCandidate] = []
    for frame_idx, speed, pressure, timestamp, combined in enriched_rows:
        if combined < score_floor:
            continue
        candidates.append(
            HighlightCandidate(
                timestamp=timestamp,
                score=max(0.0, min(1.5, combined)),
                source="action",
                reason="high_action",
                frame_idx=frame_idx,
                must_include=False,
                metadata={
                    "ball_speed_pixels_per_sec": speed,
                    "nearby_players": int(pressure),
                    "combined_score": combined,
                    "threshold": score_floor,
                },
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


def extract_audio_energy_spikes(
    video_path: str | Path,
    sample_rate: int = 2000,
    window_seconds: float = 1.0,
    hop_seconds: float = 0.25,
    min_z_score: float = 2.0,
    min_abs_rms: float = 0.01,
    min_gap_seconds: float = 2.0,
    max_spikes: int = 120,
) -> list[HighlightCandidate]:
    """Extract crowd-excitement spikes from audio RMS envelope."""
    video_path = Path(video_path)
    if not video_path.exists():
        return []

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]

    try:
        proc = subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
    except Exception:
        return []

    if not proc.stdout:
        return []

    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    if samples.size == 0:
        return []

    window_size = max(1, int(sample_rate * window_seconds))
    hop_size = max(1, int(sample_rate * hop_seconds))

    if samples.size < window_size:
        return []

    rms_values: list[float] = []
    timestamps: list[float] = []

    for start in range(0, samples.size - window_size + 1, hop_size):
        chunk = samples[start : start + window_size]
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        ts = float(start + (window_size / 2)) / float(sample_rate)
        rms_values.append(rms)
        timestamps.append(ts)

    if not rms_values:
        return []

    rms_arr = np.asarray(rms_values, dtype=np.float32)
    baseline = np.median(rms_arr)
    mad = np.median(np.abs(rms_arr - baseline))
    scale = max(1e-6, 1.4826 * mad)
    z_scores = (rms_arr - baseline) / scale

    raw_spikes: list[HighlightCandidate] = []
    for ts, rms, z in zip(timestamps, rms_arr, z_scores):
        if rms < min_abs_rms:
            continue
        if z < min_z_score:
            continue
        raw_spikes.append(
            HighlightCandidate(
                timestamp=ts,
                score=max(0.0, min(1.5, float(z / 4.0))),
                source="audio",
                reason="crowd_spike",
                frame_idx=None,
                must_include=False,
                metadata={
                    "rms": float(rms),
                    "baseline_rms": float(baseline),
                    "z_score": float(z),
                },
            )
        )

    if not raw_spikes:
        return []

    # Keep strongest spikes with min temporal spacing.
    raw_spikes.sort(key=lambda c: c.score, reverse=True)
    selected: list[HighlightCandidate] = []
    for spike in raw_spikes:
        if any(abs(spike.timestamp - s.timestamp) < min_gap_seconds for s in selected):
            continue
        selected.append(spike)
        if len(selected) >= max_spikes:
            break

    selected.sort(key=lambda c: c.timestamp)
    return selected


def build_segments_from_candidates(
    candidates: list[HighlightCandidate],
    duration_seconds: float,
    pre_roll_seconds: float = 8.0,
    post_roll_seconds: float = 12.0,
    merge_gap_seconds: float = 4.0,
) -> list[HighlightSegment]:
    """Expand candidates into segments, then merge overlaps."""
    if not candidates:
        return []

    rows = sorted(candidates, key=lambda c: c.timestamp)
    merged: list[HighlightSegment] = []

    for candidate in rows:
        start = max(0.0, candidate.timestamp - pre_roll_seconds)
        end = min(duration_seconds, candidate.timestamp + post_roll_seconds)
        if end <= start:
            continue

        if not merged:
            merged.append(
                HighlightSegment(
                    start=start,
                    end=end,
                    score=candidate.score,
                    reasons=[candidate.reason],
                    sources=[candidate.source],
                    must_include=candidate.must_include,
                    primary_candidate=candidate,
                    candidates=[candidate],
                )
            )
            continue

        current = merged[-1]
        if start <= current.end + merge_gap_seconds:
            current.end = max(current.end, end)
            current.candidates.append(candidate)
            if candidate.reason not in current.reasons:
                current.reasons.append(candidate.reason)
            if candidate.source not in current.sources:
                current.sources.append(candidate.source)
            if candidate.score > current.primary_candidate.score:
                current.primary_candidate = candidate
            current.must_include = current.must_include or candidate.must_include
            base = current.primary_candidate.score
            bonus = 0.05 * max(0, len(current.candidates) - 1) + 0.1 * max(0, len(current.sources) - 1)
            current.score = max(0.0, min(1.5, base + bonus))
        else:
            merged.append(
                HighlightSegment(
                    start=start,
                    end=end,
                    score=candidate.score,
                    reasons=[candidate.reason],
                    sources=[candidate.source],
                    must_include=candidate.must_include,
                    primary_candidate=candidate,
                    candidates=[candidate],
                )
            )

    return merged


def select_highlight_segments(
    segments: list[HighlightSegment],
    top_n: int = 20,
    min_segment_score: float = 0.4,
) -> list[HighlightSegment]:
    """Select final highlight segments with must-include priority."""
    if not segments:
        return []

    must_include = [s for s in segments if s.must_include]
    optional = [s for s in segments if not s.must_include and s.score >= min_segment_score]

    must_include.sort(key=lambda s: s.start)
    optional.sort(key=lambda s: s.score, reverse=True)

    selected = list(must_include)
    if top_n > 0:
        remaining = max(0, top_n - len(selected))
        selected.extend(optional[:remaining])
    else:
        selected.extend(optional)

    # Deduplicate very similar segments by temporal overlap.
    selected.sort(key=lambda s: s.start)
    deduped: list[HighlightSegment] = []
    for segment in selected:
        if not deduped:
            deduped.append(segment)
            continue
        last = deduped[-1]
        if segment.start < last.end and segment.end > last.start:
            if segment.score > last.score:
                deduped[-1] = segment
        else:
            deduped.append(segment)

    return deduped


def segment_to_dict(segment: HighlightSegment, segment_id: str, clip_path: str | None = None) -> dict[str, Any]:
    """Convert segment to serialized dict."""
    return {
        "segment_id": segment_id,
        "start_time": segment.start,
        "end_time": segment.end,
        "duration": segment.duration,
        "score": segment.score,
        "must_include": segment.must_include,
        "reasons": segment.reasons,
        "sources": segment.sources,
        "primary_trigger": {
            "timestamp": segment.primary_candidate.timestamp,
            "score": segment.primary_candidate.score,
            "source": segment.primary_candidate.source,
            "reason": segment.primary_candidate.reason,
            "frame_idx": segment.primary_candidate.frame_idx,
        },
        "candidate_count": len(segment.candidates),
        "clip_path": clip_path,
    }


def extract_clip(
    video_path: str | Path,
    output_path: str | Path,
    start_time: float,
    end_time: float,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
) -> tuple[bool, str | None]:
    """Extract one highlight clip with ffmpeg."""
    if end_time <= start_time:
        return False, "invalid_time_window"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-to",
        f"{end_time:.3f}",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        video_codec,
        "-c:a",
        audio_codec,
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except Exception as exc:
        return False, str(exc)

    return True, None
