"""Fusion logic for combining body-ReID and profile-based identity evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.identity.matching import MatchResult


FusionMethod = Literal["auto", "suggested", "new_player"]


@dataclass
class ProfileEvidence:
    """Profile-match evidence in the same embedding space as track embeddings."""

    profile_id: str
    player_id: int
    confidence: float


def fuse_identity_evidence(
    body_match: MatchResult,
    profile_evidence: ProfileEvidence | None,
    profile_auto_threshold: float,
    profile_suggest_threshold: float,
    override_margin: float = 0.05,
    agreement_bonus: float = 0.05,
) -> tuple[int | None, float, FusionMethod, dict]:
    """
    Fuse body ReID match with optional profile evidence.

    Returns:
        (player_id, confidence, method, fusion_metadata)
    """
    player_id = body_match.player_id
    confidence = body_match.confidence
    method: FusionMethod = body_match.method
    strategy = "body_only"

    metadata = {
        "body_match": {
            "player_id": body_match.player_id,
            "confidence": body_match.confidence,
            "method": body_match.method,
        },
        "profile_match": None,
        "strategy": strategy,
    }

    if profile_evidence is None:
        return player_id, confidence, method, metadata

    metadata["profile_match"] = {
        "profile_id": profile_evidence.profile_id,
        "player_id": profile_evidence.player_id,
        "confidence": profile_evidence.confidence,
    }

    # If body is uncertain/new and profile is strong enough, use profile candidate.
    if player_id is None and profile_evidence.confidence >= profile_suggest_threshold:
        player_id = profile_evidence.player_id
        confidence = profile_evidence.confidence
        method = (
            "auto"
            if profile_evidence.confidence >= profile_auto_threshold
            else "suggested"
        )
        metadata["strategy"] = "profile_only"
        return player_id, confidence, method, metadata

    # Agreement boost when both sources point to the same player.
    if player_id == profile_evidence.player_id:
        boosted = max(confidence, profile_evidence.confidence) + agreement_bonus
        confidence = min(1.0, boosted)
        if confidence >= profile_auto_threshold:
            method = "auto"
        metadata["strategy"] = "agreement_boost"
        return player_id, confidence, method, metadata

    # Override when profile evidence is materially stronger.
    if (
        profile_evidence.confidence >= profile_suggest_threshold
        and profile_evidence.confidence > confidence + override_margin
    ):
        player_id = profile_evidence.player_id
        confidence = profile_evidence.confidence
        method = (
            "auto"
            if profile_evidence.confidence >= profile_auto_threshold
            else "suggested"
        )
        metadata["strategy"] = "profile_override"
        return player_id, confidence, method, metadata

    return player_id, confidence, method, metadata

