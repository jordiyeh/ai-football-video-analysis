"""Tests for multimodal identity evidence and lock/unlock behavior."""

from __future__ import annotations

import cv2
import numpy as np

from src.identity.multimodal import (
    FaceEvidence,
    apply_multimodal_evidence,
    apply_substitution_locks,
    build_profile_face_signatures,
    extract_jersey_ocr_evidence,
    match_track_face_evidence,
)


def _solid_image(color_bgr: tuple[int, int, int], h: int = 96, w: int = 64) -> np.ndarray:
    """Create a solid-color BGR test image."""
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :] = np.array(color_bgr, dtype=np.uint8)
    return image


def test_apply_multimodal_evidence_face_override():
    """Strong face evidence should override weaker base assignment."""
    face = FaceEvidence(profile_id="10_Nick", player_id=10, confidence=0.88, support_frames=3)
    player_id, confidence, method, metadata = apply_multimodal_evidence(
        base_player_id=3,
        base_confidence=0.71,
        base_method="suggested",
        auto_threshold=0.85,
        suggest_threshold=0.70,
        face_evidence=face,
        face_override_margin=0.08,
    )

    assert player_id == 10
    assert confidence == 0.88
    assert method == "auto"
    assert "face_override" in metadata["applied"]


def test_apply_multimodal_evidence_face_agreement_boost():
    """Agreement between base and face evidence should raise confidence."""
    face = FaceEvidence(profile_id="7_Alex", player_id=7, confidence=0.80, support_frames=2)
    player_id, confidence, method, metadata = apply_multimodal_evidence(
        base_player_id=7,
        base_confidence=0.78,
        base_method="suggested",
        auto_threshold=0.80,
        suggest_threshold=0.70,
        face_evidence=face,
        face_agreement_bonus=0.03,
    )

    assert player_id == 7
    assert confidence >= 0.80
    assert method == "auto"
    assert "face_agreement_boost" in metadata["applied"]


def test_extract_jersey_ocr_evidence_unique_mapping():
    """OCR evidence should map to a unique player when jersey number is unambiguous."""
    crops = [_solid_image((255, 255, 255)) for _ in range(3)]
    jersey_index = {10: [101], 9: [202]}

    responses = iter([("10", 0.82), ("10", 0.76), (None, 0.0)])

    def ocr_stub(_img):
        return next(responses)

    evidence = extract_jersey_ocr_evidence(
        crop_images=crops,
        jersey_player_index=jersey_index,
        min_ocr_confidence=0.5,
        min_support_frames=2,
        ocr_fn=ocr_stub,
    )

    assert evidence is not None
    assert evidence.jersey_number == 10
    assert evidence.player_id == 101
    assert evidence.ambiguous is False
    assert evidence.support_frames == 2


def test_apply_substitution_locks_demotes_overlap_conflict():
    """Overlapping confident assignments for same player should unlock and demote loser."""
    assignments = [
        {
            "track_id": 1,
            "player_id": 10,
            "match_method": "auto",
            "confidence": 0.92,
            "frame_start": 0,
            "frame_end": 320,
        },
        {
            "track_id": 2,
            "player_id": 10,
            "match_method": "auto",
            "confidence": 0.87,
            "frame_start": 120,
            "frame_end": 280,
        },
    ]

    updated, summary = apply_substitution_locks(
        assignments,
        lock_confidence_threshold=0.8,
        overlap_conflict_frames=45,
        substitution_gap_frames=150,
        demote_conflicting_auto=True,
    )

    by_track = {row["track_id"]: row for row in updated}
    assert by_track[1]["lock_state"] == "locked"
    assert by_track[2]["lock_state"] == "unlocked"
    assert by_track[2]["match_method"] == "suggested"
    assert summary["overlap_conflicts"] == 1
    assert summary["demoted_conflicts"] == 1


def test_apply_substitution_locks_gap_unlock_transition():
    """Large gap between tracks should be treated as substitution-style unlock transition."""
    assignments = [
        {
            "track_id": 1,
            "player_id": 4,
            "match_method": "auto",
            "confidence": 0.9,
            "frame_start": 0,
            "frame_end": 100,
        },
        {
            "track_id": 3,
            "player_id": 4,
            "match_method": "auto",
            "confidence": 0.88,
            "frame_start": 420,
            "frame_end": 520,
        },
    ]

    updated, summary = apply_substitution_locks(
        assignments,
        lock_confidence_threshold=0.8,
        overlap_conflict_frames=45,
        substitution_gap_frames=150,
    )

    by_track = {row["track_id"]: row for row in updated}
    assert by_track[1]["lock_state"] == "locked"
    assert by_track[3]["lock_state"] == "locked"
    assert by_track[3]["lock_reason"] == "substitution_gap_unlock"
    assert summary["substitution_unlocks"] == 1
    assert summary["locks_applied"] == 2


def test_profile_face_signatures_and_track_matching(tmp_path):
    """Face-signature building should produce a signature that can be matched from similar crops."""
    profile_dir = tmp_path / "10_Nick"
    profile_dir.mkdir()
    image_path = profile_dir / "img1.jpg"
    cv2.imwrite(str(image_path), _solid_image((10, 30, 220)))

    registry = {
        "schema_version": "1.0",
        "profiles": [
            {
                "profile_id": "10_Nick",
                "image_paths": [str(image_path)],
            }
        ],
    }
    links = [{"profile_id": "10_Nick", "player_id": 10}]

    signatures, summary = build_profile_face_signatures(
        profile_registry=registry,
        profile_links=links,
        max_images_per_profile=1,
        min_face_images=1,
    )

    assert len(signatures) == 1
    assert summary["profiles_with_face_signatures"] == 1

    crop_images = [_solid_image((12, 35, 215)), _solid_image((8, 28, 225))]
    evidence = match_track_face_evidence(
        crop_images=crop_images,
        signatures=signatures,
        suggest_threshold=-1.0,
        min_support_frames=1,
    )

    assert evidence is not None
    assert evidence.player_id == 10
    assert evidence.profile_id == "10_Nick"
    assert evidence.confidence >= -1.0
    assert evidence.backend in {"facenet512", "histogram_fallback", "unknown"}
