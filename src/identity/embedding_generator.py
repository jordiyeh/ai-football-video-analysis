"""Generate face embeddings from training images using Facenet512."""

import logging
import pickle
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.identity.multimodal import (
    _extract_face_region,
    _compute_face_descriptor_facenet,
    _load_face_detector,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def generate_player_embeddings(
    player_id: int | str,
    image_paths: list[str | Path],
    player_name: str | None = None,
) -> dict:
    """Process images through Facenet512 and return pkl-compatible dict.

    Args:
        player_id: Player identifier.
        image_paths: Paths to training images.
        player_name: Optional player name for metadata.

    Returns:
        Dict matching existing pkl schema with encodings, averaged_encoding, stats.
    """
    detector = _load_face_detector()
    encodings = []
    failed = 0

    for img_path in image_paths:
        img_path = Path(img_path)
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Could not read image: %s", img_path)
            failed += 1
            continue

        region = _extract_face_region(image, detector)
        if region is None:
            logger.warning("No face region extracted from: %s", img_path)
            failed += 1
            continue

        embedding = _compute_face_descriptor_facenet(region)
        if embedding is None:
            logger.warning("Facenet512 embedding failed for: %s", img_path)
            failed += 1
            continue

        encodings.append({
            "encoding": embedding,
            "image_path": str(img_path),
            "image_name": img_path.name,
            "timestamp": datetime.utcnow().isoformat(),
            "facial_area": {},
            "model": "Facenet512",
        })

    # Compute averaged encoding (L2-normalized mean)
    averaged_encoding = None
    if encodings:
        all_vecs = np.stack([e["encoding"] for e in encodings])
        mean_vec = all_vecs.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-8:
            averaged_encoding = (mean_vec / norm).astype(np.float32)

    now = datetime.utcnow().isoformat()
    return {
        "player_id": str(player_id),
        "player_name": player_name,
        "created": now,
        "last_updated": now,
        "model": "Facenet512",
        "num_encodings": len(encodings),
        "encodings": encodings,
        "averaged_encoding": averaged_encoding,
        "stats": {
            "total_images_processed": len(image_paths),
            "successful_extractions": len(encodings),
            "failed_extractions": failed,
        },
    }


def write_embeddings_pkl(payload: dict, dest_path: str | Path) -> None:
    """Serialize embedding payload to pkl file."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_embeddings_pkl(pkl_path: str | Path) -> dict | None:
    """Load existing embedding payload from pkl file, or None if missing."""
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def rebuild_embeddings_from_training_dir(
    player_id: int | str,
    training_dir: str | Path,
    player_name: str | None = None,
) -> dict:
    """Scan training_dir for all images, generate embeddings, write pkl, return payload.

    Args:
        player_id: Player identifier.
        training_dir: Directory containing training images.
        player_name: Optional player name for metadata.

    Returns:
        Embedding payload dict (same schema as generate_player_embeddings).
    """
    training_dir = Path(training_dir)
    image_paths = sorted(
        p for p in training_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ) if training_dir.exists() else []

    payload = generate_player_embeddings(player_id, image_paths, player_name)

    # Write pkl alongside training dir
    pkl_path = training_dir.parent / "embeddings.pkl"
    write_embeddings_pkl(payload, pkl_path)

    return payload


def add_embeddings_from_images(
    player_id: int | str,
    image_paths: list[str | Path],
    pkl_path: str | Path,
    player_name: str | None = None,
) -> dict:
    """Generate embeddings for new images, merge with existing pkl, save, return payload.

    This is the incremental version: existing embeddings are preserved and new ones
    are appended. Images can be deleted after this call since only the pkl matters.

    Args:
        player_id: Player identifier.
        image_paths: Paths to NEW training images.
        pkl_path: Path to the embeddings.pkl file (may or may not exist yet).
        player_name: Optional player name for metadata.

    Returns:
        Merged embedding payload dict.
    """
    pkl_path = Path(pkl_path)

    # Generate embeddings for new images
    new_payload = generate_player_embeddings(player_id, image_paths, player_name)

    # Load existing embeddings if present
    existing = load_embeddings_pkl(pkl_path)

    if existing and existing.get("encodings"):
        # Merge: combine existing + new encodings
        all_encodings = existing["encodings"] + new_payload["encodings"]
    else:
        all_encodings = new_payload["encodings"]

    # Recompute averaged encoding
    averaged_encoding = None
    if all_encodings:
        all_vecs = np.stack([e["encoding"] for e in all_encodings])
        mean_vec = all_vecs.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-8:
            averaged_encoding = (mean_vec / norm).astype(np.float32)

    now = datetime.utcnow().isoformat()
    total_processed = new_payload["stats"]["total_images_processed"]
    existing_count = len(existing["encodings"]) if existing and existing.get("encodings") else 0

    merged = {
        "player_id": str(player_id),
        "player_name": player_name,
        "created": existing["created"] if existing else now,
        "last_updated": now,
        "model": "Facenet512",
        "num_encodings": len(all_encodings),
        "encodings": all_encodings,
        "averaged_encoding": averaged_encoding,
        "stats": {
            "total_images_processed": total_processed,
            "successful_extractions": new_payload["stats"]["successful_extractions"],
            "failed_extractions": new_payload["stats"]["failed_extractions"],
            "existing_encodings_kept": existing_count,
        },
    }

    write_embeddings_pkl(merged, pkl_path)
    return merged
