"""Profile photo seeding and matching helpers for dynamic player tagging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.identity.fusion import ProfileEvidence
from src.identity.matching import aggregate_embeddings, cosine_similarity, cosine_similarity_batch


@dataclass
class ProfileSignature:
    """Profile embedding signature derived from profile photos."""

    profile_id: str
    display_name: str | None
    jersey_number: int | None
    embedding: np.ndarray
    images_used: int


def build_profile_signatures(
    profile_registry: dict[str, Any] | None,
    detector: Any,
    crop_extractor: Any,
    reid_extractor: Any,
    max_images_per_profile: int = 5,
    min_profile_crops_for_seed: int = 2,
    fallback_full_image: bool = True,
) -> tuple[list[ProfileSignature], dict[str, Any]]:
    """
    Build body-embedding signatures from profile images.

    Args:
        profile_registry: Loaded profile_registry.json content.
        detector: Detector with detect(image) method.
        crop_extractor: Crop extractor to crop detected players.
        reid_extractor: ReID extractor with extract(crops) method.

    Returns:
        (signatures, summary)
    """
    signatures: list[ProfileSignature] = []
    summary = {
        "profiles_seen": 0,
        "profiles_with_signatures": 0,
        "images_used": 0,
        "images_failed": 0,
        "profiles_skipped": 0,
    }

    if not profile_registry:
        return signatures, summary

    profiles = profile_registry.get("profiles", [])
    if not isinstance(profiles, list):
        return signatures, summary

    for profile in profiles:
        summary["profiles_seen"] += 1
        profile_id = str(profile.get("profile_id", "unknown_profile"))
        display_name = profile.get("display_name")
        jersey_number = profile.get("jersey_number")
        image_paths = profile.get("image_paths", [])
        if not isinstance(image_paths, list):
            image_paths = []

        crops: list[np.ndarray] = []
        selected_paths = image_paths[:max_images_per_profile]
        for image_path in selected_paths:
            img = cv2.imread(str(image_path))
            if img is None:
                summary["images_failed"] += 1
                continue

            best_crop = None
            try:
                detections = detector.detect(img)
                player_dets = [d for d in detections if getattr(d, "object_type", None) == "player"]
                if player_dets:
                    best = max(player_dets, key=lambda d: getattr(d, "confidence", 0.0))
                    crop = crop_extractor.extract_crop(
                        frame=img,
                        bbox=best.bbox,
                        track_id=-1,
                        frame_idx=0,
                        confidence=best.confidence,
                    )
                    if crop is not None:
                        best_crop = crop.image
            except Exception:
                # Fallback path below if enabled.
                best_crop = None

            if best_crop is not None:
                crops.append(best_crop)
                summary["images_used"] += 1
            elif fallback_full_image:
                crops.append(img.copy())
                summary["images_used"] += 1
            else:
                summary["images_failed"] += 1

        if len(crops) < min_profile_crops_for_seed:
            summary["profiles_skipped"] += 1
            continue

        embeddings = reid_extractor.extract(crops)
        centroid = aggregate_embeddings([emb for emb in embeddings])

        signatures.append(
            ProfileSignature(
                profile_id=profile_id,
                display_name=display_name,
                jersey_number=jersey_number,
                embedding=centroid,
                images_used=len(crops),
            )
        )
        summary["profiles_with_signatures"] += 1

    return signatures, summary


def seed_players_from_signatures(
    db: Any,
    signatures: list[ProfileSignature],
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    """
    Seed/update players from profile signatures.

    Returns:
        (profile_player_links, seeded_player_centroids)
    """
    links: list[dict[str, Any]] = []
    centroids: dict[int, np.ndarray] = {}

    for signature in signatures:
        player = db.find_player_by_name_and_number(
            name=signature.display_name,
            jersey_number=signature.jersey_number,
        )

        if player is None:
            created = db.create_player(
                name=signature.display_name,
                jersey_number=signature.jersey_number,
                embedding=signature.embedding,
            )
            player_id = created.player_id
            link_method = "created_from_profile"
            centroids[player_id] = signature.embedding
        else:
            player_id = player.player_id
            link_method = "linked_existing_player"

            # Backfill missing metadata.
            needs_update = (
                (player.name is None and signature.display_name is not None)
                or (player.jersey_number is None and signature.jersey_number is not None)
            )
            if needs_update:
                db.update_player(
                    player_id=player_id,
                    name=signature.display_name if player.name is None else None,
                    jersey_number=signature.jersey_number if player.jersey_number is None else None,
                )

            # Seed centroid if missing or weakly aligned.
            current = player.get_centroid_array()
            if current is None:
                db.update_player_centroid(player_id, signature.embedding)
            else:
                sim = cosine_similarity(current, signature.embedding)
                if sim < 0.98:
                    db.update_player_centroid(player_id, signature.embedding)
            centroids[player_id] = signature.embedding

        links.append(
            {
                "profile_id": signature.profile_id,
                "player_id": player_id,
                "display_name": signature.display_name,
                "jersey_number": signature.jersey_number,
                "images_used": signature.images_used,
                "link_method": link_method,
            }
        )

    return links, centroids


def match_embedding_to_profile_links(
    embedding: np.ndarray,
    signatures: list[ProfileSignature],
    profile_links: list[dict[str, Any]],
    suggest_threshold: float,
) -> ProfileEvidence | None:
    """
    Match track embedding against profile signatures and return best evidence.
    """
    if not signatures or not profile_links:
        return None

    profile_to_player = {
        str(link["profile_id"]): int(link["player_id"])
        for link in profile_links
        if link.get("profile_id") is not None and link.get("player_id") is not None
    }

    valid = [sig for sig in signatures if sig.profile_id in profile_to_player]
    if not valid:
        return None

    gallery = np.array([sig.embedding for sig in valid], dtype=np.float32)
    similarities = cosine_similarity_batch(embedding, gallery)
    if similarities.size == 0:
        return None

    best_idx = int(np.argmax(similarities))
    best_sig = valid[best_idx]
    best_conf = float(similarities[best_idx])
    if best_conf < suggest_threshold:
        return None

    player_id = profile_to_player[best_sig.profile_id]
    return ProfileEvidence(
        profile_id=best_sig.profile_id,
        player_id=player_id,
        confidence=best_conf,
    )

