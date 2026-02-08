"""Event detection module for shots, goals, and match events."""

from importlib import import_module
from typing import Any

__all__ = [
    "BallTrajectory",
    "BallTrajectoryPoint",
    "CelebrationDetector",
    "CelebrationEvent",
    "Event",
    "EventDetector",
    "KickEvent",
    "KickEventDetector",
    "GoalAreaEntryEvent",
    "GoalAreaEntryDetector",
    "ShotCandidate",
    "ShotFusionEngine",
    "HighlightCandidate",
    "HighlightSegment",
    "build_event_candidates",
    "build_action_candidates",
    "extract_audio_energy_spikes",
    "build_segments_from_candidates",
    "select_highlight_segments",
    "segment_to_dict",
    "extract_clip",
    "build_player_reels",
    "ClusteringState",
    "PlayerClusteringAnalyzer",
    "GoalkeeperDiveEvent",
    "GoalkeeperAnalyzer",
    "PassInferenceConfig",
    "PassInferencer",
    "infer_pass_events",
    "SetPieceInferenceConfig",
    "SetPieceInferencer",
    "infer_set_piece_events",
    "TacticalInferenceConfig",
    "TacticalInferencer",
    "infer_tactical_events",
]

_SYMBOL_TO_MODULE = {
    "BallTrajectory": "src.events.ball_trajectory",
    "BallTrajectoryPoint": "src.events.ball_trajectory",
    "CelebrationDetector": "src.events.celebration_detection",
    "CelebrationEvent": "src.events.celebration_detection",
    "Event": "src.events.detection",
    "EventDetector": "src.events.detection",
    "KickEvent": "src.events.kick_detection",
    "KickEventDetector": "src.events.kick_detection",
    "GoalAreaEntryEvent": "src.events.kick_detection",
    "GoalAreaEntryDetector": "src.events.kick_detection",
    "ShotCandidate": "src.events.kick_detection",
    "ShotFusionEngine": "src.events.kick_detection",
    "HighlightCandidate": "src.events.highlights",
    "HighlightSegment": "src.events.highlights",
    "build_event_candidates": "src.events.highlights",
    "build_action_candidates": "src.events.highlights",
    "extract_audio_energy_spikes": "src.events.highlights",
    "build_segments_from_candidates": "src.events.highlights",
    "select_highlight_segments": "src.events.highlights",
    "segment_to_dict": "src.events.highlights",
    "extract_clip": "src.events.highlights",
    "build_player_reels": "src.events.player_reels",
    "ClusteringState": "src.events.player_analysis",
    "PlayerClusteringAnalyzer": "src.events.player_analysis",
    "GoalkeeperDiveEvent": "src.events.player_analysis",
    "GoalkeeperAnalyzer": "src.events.player_analysis",
    "PassInferenceConfig": "src.events.passes",
    "PassInferencer": "src.events.passes",
    "infer_pass_events": "src.events.passes",
    "SetPieceInferenceConfig": "src.events.set_pieces",
    "SetPieceInferencer": "src.events.set_pieces",
    "infer_set_piece_events": "src.events.set_pieces",
    "TacticalInferenceConfig": "src.events.tactical",
    "TacticalInferencer": "src.events.tactical",
    "infer_tactical_events": "src.events.tactical",
}


def __getattr__(name: str) -> Any:
    """Lazily import event modules that rely on optional runtime deps."""
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
