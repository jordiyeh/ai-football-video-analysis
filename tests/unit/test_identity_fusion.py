"""Tests for dynamic identity fusion and profile evidence matching."""

import numpy as np

from src.identity.fusion import ProfileEvidence, fuse_identity_evidence
from src.identity.matching import MatchResult
from src.identity.profile_seed import ProfileSignature, match_embedding_to_profile_links


class TestFuseIdentityEvidence:
    """Fusion behavior for body + profile evidence."""

    def test_body_only_when_no_profile(self):
        """Body result should pass through when no profile evidence exists."""
        body = MatchResult(player_id=3, confidence=0.81, method="suggested", all_scores={3: 0.81})
        player_id, confidence, method, metadata = fuse_identity_evidence(
            body_match=body,
            profile_evidence=None,
            profile_auto_threshold=0.82,
            profile_suggest_threshold=0.68,
        )

        assert player_id == 3
        assert confidence == 0.81
        assert method == "suggested"
        assert metadata["strategy"] == "body_only"

    def test_profile_only_when_body_new_player(self):
        """Strong profile evidence should rescue uncertain body result."""
        body = MatchResult(player_id=None, confidence=0.4, method="new_player", all_scores={})
        profile = ProfileEvidence(profile_id="10_Nick", player_id=10, confidence=0.78)
        player_id, confidence, method, metadata = fuse_identity_evidence(
            body_match=body,
            profile_evidence=profile,
            profile_auto_threshold=0.82,
            profile_suggest_threshold=0.68,
        )

        assert player_id == 10
        assert method == "suggested"
        assert confidence == 0.78
        assert metadata["strategy"] == "profile_only"

    def test_agreement_boost(self):
        """When both sources agree, confidence should be boosted."""
        body = MatchResult(player_id=4, confidence=0.79, method="suggested", all_scores={4: 0.79})
        profile = ProfileEvidence(profile_id="4_Alex", player_id=4, confidence=0.83)
        player_id, confidence, method, metadata = fuse_identity_evidence(
            body_match=body,
            profile_evidence=profile,
            profile_auto_threshold=0.82,
            profile_suggest_threshold=0.68,
            agreement_bonus=0.05,
        )

        assert player_id == 4
        assert method == "auto"
        assert confidence >= 0.82
        assert metadata["strategy"] == "agreement_boost"

    def test_profile_override(self):
        """Profile evidence should override body when materially stronger."""
        body = MatchResult(player_id=7, confidence=0.71, method="suggested", all_scores={7: 0.71})
        profile = ProfileEvidence(profile_id="12_Luca", player_id=12, confidence=0.82)
        player_id, confidence, method, metadata = fuse_identity_evidence(
            body_match=body,
            profile_evidence=profile,
            profile_auto_threshold=0.82,
            profile_suggest_threshold=0.68,
            override_margin=0.05,
        )

        assert player_id == 12
        assert method == "auto"
        assert confidence == 0.82
        assert metadata["strategy"] == "profile_override"


class TestMatchEmbeddingToProfileLinks:
    """Matching track embedding against seeded profile signatures."""

    def test_profile_signature_matching(self):
        """Best profile signature above threshold should be returned."""
        sig_a = ProfileSignature(
            profile_id="10_Nick",
            display_name="Nick",
            jersey_number=10,
            embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            images_used=3,
        )
        sig_b = ProfileSignature(
            profile_id="8_Sam",
            display_name="Sam",
            jersey_number=8,
            embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            images_used=3,
        )
        links = [
            {"profile_id": "10_Nick", "player_id": 101},
            {"profile_id": "8_Sam", "player_id": 108},
        ]

        query = np.array([0.98, 0.1, 0.0], dtype=np.float32)
        query = query / np.linalg.norm(query)

        evidence = match_embedding_to_profile_links(
            embedding=query,
            signatures=[sig_a, sig_b],
            profile_links=links,
            suggest_threshold=0.68,
        )

        assert evidence is not None
        assert evidence.profile_id == "10_Nick"
        assert evidence.player_id == 101
        assert evidence.confidence > 0.9

