"""Multimodal identity helpers: face cues, jersey OCR cues, and lock/unlock logic."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Callable, Literal

import cv2
import numpy as np

from src.identity.matching import cosine_similarity_batch


@dataclass
class FaceSignature:
    """Face-like descriptor aggregated from profile photos."""

    profile_id: str
    player_id: int
    descriptor: np.ndarray
    images_used: int


@dataclass
class FaceEvidence:
    """Track-level face-like evidence against profile signatures."""

    profile_id: str
    player_id: int
    confidence: float
    support_frames: int
    backend: str = "unknown"


@dataclass
class JerseyOCREvidence:
    """Track-level jersey OCR evidence."""

    jersey_number: int
    player_id: int | None
    confidence: float
    support_frames: int
    candidate_player_ids: list[int]
    ambiguous: bool


FusionMethod = Literal["auto", "suggested", "new_player"]
OCRFn = Callable[[np.ndarray], tuple[str | None, float]]
FaceDescriptor = tuple[np.ndarray | None, str]


def _normalize(vec: np.ndarray) -> np.ndarray | None:
    """L2 normalize a vector and return None for degenerate vectors."""
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return None
    return vec / norm


def _load_face_detector() -> cv2.CascadeClassifier | None:
    """Load OpenCV Haar face detector if available."""
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return None
        return detector
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_facenet512_backend() -> tuple[Any | None, str]:
    """Load Facenet512 backend when available."""
    try:
        import torch
        from facenet_pytorch import InceptionResnetV1

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        return (model, str(device))
    except Exception:
        return (None, "unavailable")


def _extract_face_region(
    image: np.ndarray,
    detector: cv2.CascadeClassifier | None,
) -> np.ndarray | None:
    """Extract a face-ish ROI (detected face or robust fallback top-center patch)."""
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    if h < 12 or w < 12:
        return None

    if detector is not None:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=4,
                minSize=(24, 24),
            )
            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda box: box[2] * box[3])
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(w, int(x + fw))
                y2 = min(h, int(y + fh))
                roi = image[y1:y2, x1:x2]
                if roi.size > 0:
                    return roi
        except Exception:
            pass

    # Fallback: use top-center region to bias toward face/head area.
    x1 = int(w * 0.2)
    x2 = int(w * 0.8)
    y1 = 0
    y2 = int(h * 0.45)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi


def _compute_face_descriptor(region: np.ndarray) -> np.ndarray | None:
    """Compute a normalized fallback histogram descriptor for a face-like region."""
    if region is None or region.size == 0:
        return None

    try:
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 180, 0, 256, 0, 256],
        )
        hist = hist.reshape(-1).astype(np.float32)
        return _normalize(hist)
    except Exception:
        return None


def _compute_face_descriptor_facenet(region: np.ndarray) -> np.ndarray | None:
    """Compute Facenet512 embedding for a face-like region."""
    model, device_name = _load_facenet512_backend()
    if model is None:
        return None

    try:
        import torch

        rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0)
        tensor = (tensor - 127.5) / 128.0
        device = torch.device(device_name)
        tensor = tensor.to(device)

        with torch.no_grad():
            embedding = model(tensor)
        vector = embedding.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return _normalize(vector)
    except Exception:
        return None


def _extract_face_descriptor(
    image: np.ndarray,
    detector: cv2.CascadeClassifier | None,
) -> FaceDescriptor:
    """Extract a normalized face descriptor from an image (Facenet512 preferred)."""
    region = _extract_face_region(image, detector)
    if region is None:
        return None, "none"

    facenet_descriptor = _compute_face_descriptor_facenet(region)
    if facenet_descriptor is not None:
        return facenet_descriptor, "facenet512"

    fallback = _compute_face_descriptor(region)
    if fallback is not None:
        return fallback, "histogram_fallback"

    return None, "none"


def build_profile_face_signatures(
    profile_registry: dict[str, Any] | None,
    profile_links: list[dict[str, Any]] | None,
    max_images_per_profile: int = 5,
    min_face_images: int = 1,
) -> tuple[list[FaceSignature], dict[str, Any]]:
    """
    Build player-linked face signatures from profile images.

    Returns:
        (signatures, summary)
    """
    summary: dict[str, Any] = {
        "profiles_seen": 0,
        "profiles_linked": 0,
        "profiles_with_face_signatures": 0,
        "images_used": 0,
        "images_failed": 0,
        "embedding_backend": "none",
    }
    if not profile_registry or not profile_links:
        return [], summary

    profile_to_player: dict[str, int] = {}
    for link in profile_links:
        profile_id = link.get("profile_id")
        player_id = link.get("player_id")
        if profile_id is None or player_id is None:
            continue
        profile_to_player[str(profile_id)] = int(player_id)

    detector = _load_face_detector()
    signatures: list[FaceSignature] = []

    for profile in profile_registry.get("profiles", []):
        summary["profiles_seen"] += 1
        profile_id = str(profile.get("profile_id", ""))
        if not profile_id or profile_id not in profile_to_player:
            continue
        summary["profiles_linked"] += 1

        descriptors: list[np.ndarray] = []
        image_paths = profile.get("image_paths", [])
        if not isinstance(image_paths, list):
            image_paths = []

        for image_path in image_paths[:max_images_per_profile]:
            img = cv2.imread(str(image_path))
            if img is None:
                summary["images_failed"] += 1
                continue
            descriptor, backend = _extract_face_descriptor(img, detector)
            if descriptor is None:
                summary["images_failed"] += 1
                continue
            if summary["embedding_backend"] in ("none", "histogram_fallback") and backend != "none":
                summary["embedding_backend"] = backend
            descriptors.append(descriptor)
            summary["images_used"] += 1

        if len(descriptors) < min_face_images:
            continue

        centroid = np.mean(np.stack(descriptors, axis=0), axis=0).astype(np.float32)
        centroid = _normalize(centroid)
        if centroid is None:
            continue

        signatures.append(
            FaceSignature(
                profile_id=profile_id,
                player_id=profile_to_player[profile_id],
                descriptor=centroid,
                images_used=len(descriptors),
            )
        )
        summary["profiles_with_face_signatures"] += 1

    return signatures, summary


def match_track_face_evidence(
    crop_images: list[np.ndarray],
    signatures: list[FaceSignature],
    suggest_threshold: float = 0.68,
    min_support_frames: int = 1,
) -> FaceEvidence | None:
    """
    Match track crops against profile face signatures.
    """
    if not crop_images or not signatures:
        return None

    detector = _load_face_detector()
    gallery = np.array([s.descriptor for s in signatures], dtype=np.float32)
    votes: dict[int, list[float]] = {}
    backend_votes: dict[str, int] = {}

    for crop in crop_images:
        descriptor, backend = _extract_face_descriptor(crop, detector)
        if descriptor is None:
            continue
        backend_votes[backend] = backend_votes.get(backend, 0) + 1
        similarities = cosine_similarity_batch(descriptor, gallery)
        if similarities.size == 0:
            continue
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        if best_sim < suggest_threshold:
            continue
        votes.setdefault(best_idx, []).append(best_sim)

    if not votes:
        return None

    best_idx, best_scores = max(
        votes.items(),
        key=lambda item: (len(item[1]), float(np.mean(item[1]))),
    )
    if len(best_scores) < min_support_frames:
        return None

    signature = signatures[best_idx]
    backend = "unknown"
    if backend_votes:
        backend = max(backend_votes.items(), key=lambda item: item[1])[0]
    return FaceEvidence(
        profile_id=signature.profile_id,
        player_id=signature.player_id,
        confidence=float(np.mean(best_scores)),
        support_frames=len(best_scores),
        backend=backend,
    )


def _extract_torso_region(image: np.ndarray) -> np.ndarray | None:
    """Extract torso region where jersey number is most likely visible."""
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    if h < 12 or w < 12:
        return None
    y1 = int(h * 0.22)
    y2 = int(h * 0.78)
    x1 = int(w * 0.2)
    x2 = int(w * 0.8)
    torso = image[y1:y2, x1:x2]
    if torso.size == 0:
        return None
    return torso


def _prepare_ocr_image(torso: np.ndarray) -> np.ndarray:
    """Prepare torso crop for digit OCR."""
    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        3,
    )


def _default_ocr_digits(image: np.ndarray) -> tuple[str | None, float]:
    """Run digit OCR using pytesseract when available."""
    try:
        import pytesseract
    except Exception:
        return None, 0.0

    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789"
    try:
        data = pytesseract.image_to_data(
            image,
            config=cfg,
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return None, 0.0

    best_text: str | None = None
    best_conf = 0.0
    texts = data.get("text", [])
    confs = data.get("conf", [])
    for text, conf_raw in zip(texts, confs):
        digits = "".join(re.findall(r"\d+", str(text)))
        if not digits:
            continue
        if len(digits) > 2:
            digits = digits[:2]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = 0.0
        conf = max(0.0, min(100.0, conf)) / 100.0
        if conf > best_conf:
            best_conf = conf
            best_text = digits

    return best_text, best_conf


def build_jersey_player_index(players: list[Any]) -> dict[int, list[int]]:
    """Build jersey number -> player_ids index."""
    index: dict[int, list[int]] = {}
    for player in players:
        if isinstance(player, dict):
            player_id = player.get("player_id")
            jersey_number = player.get("jersey_number")
        else:
            player_id = getattr(player, "player_id", None)
            jersey_number = getattr(player, "jersey_number", None)

        if player_id is None or jersey_number is None:
            continue
        jersey = int(jersey_number)
        index.setdefault(jersey, []).append(int(player_id))
    return index


def extract_jersey_ocr_evidence(
    crop_images: list[np.ndarray],
    jersey_player_index: dict[int, list[int]],
    min_ocr_confidence: float = 0.4,
    min_support_frames: int = 1,
    ocr_fn: OCRFn | None = None,
) -> JerseyOCREvidence | None:
    """
    Extract jersey-number OCR evidence for a track.
    """
    if not crop_images or not jersey_player_index:
        return None

    ocr = ocr_fn or _default_ocr_digits
    per_number_scores: dict[int, list[float]] = {}

    for crop in crop_images:
        torso = _extract_torso_region(crop)
        if torso is None:
            continue
        prepared = _prepare_ocr_image(torso)
        raw_text, raw_conf = ocr(prepared)
        if raw_text is None:
            continue
        try:
            jersey_number = int(raw_text)
        except Exception:
            continue
        if raw_conf < min_ocr_confidence:
            continue
        per_number_scores.setdefault(jersey_number, []).append(float(raw_conf))

    if not per_number_scores:
        return None

    best_number, confidences = max(
        per_number_scores.items(),
        key=lambda item: (len(item[1]), float(np.mean(item[1]))),
    )
    support_frames = len(confidences)
    if support_frames < min_support_frames:
        return None

    candidate_player_ids = jersey_player_index.get(int(best_number), [])
    ambiguous = len(candidate_player_ids) != 1
    player_id = candidate_player_ids[0] if len(candidate_player_ids) == 1 else None

    return JerseyOCREvidence(
        jersey_number=int(best_number),
        player_id=player_id,
        confidence=float(np.mean(confidences)),
        support_frames=support_frames,
        candidate_player_ids=[int(pid) for pid in candidate_player_ids],
        ambiguous=ambiguous,
    )


def apply_multimodal_evidence(
    base_player_id: int | None,
    base_confidence: float,
    base_method: FusionMethod,
    auto_threshold: float,
    suggest_threshold: float,
    face_evidence: FaceEvidence | None = None,
    jersey_evidence: JerseyOCREvidence | None = None,
    face_override_margin: float = 0.08,
    face_agreement_bonus: float = 0.04,
    jersey_override_margin: float = 0.12,
    jersey_agreement_bonus: float = 0.03,
) -> tuple[int | None, float, FusionMethod, dict[str, Any]]:
    """
    Blend body/profile result with optional face and jersey evidence.
    """
    player_id = base_player_id
    confidence = float(base_confidence)
    method: FusionMethod = base_method

    metadata: dict[str, Any] = {
        "base": {
            "player_id": base_player_id,
            "confidence": float(base_confidence),
            "method": base_method,
        },
        "face": None,
        "jersey_ocr": None,
        "applied": [],
    }

    if face_evidence is not None:
        metadata["face"] = {
            "profile_id": face_evidence.profile_id,
            "player_id": face_evidence.player_id,
            "confidence": face_evidence.confidence,
            "support_frames": face_evidence.support_frames,
            "backend": face_evidence.backend,
        }
        if player_id is None and face_evidence.confidence >= suggest_threshold:
            player_id = face_evidence.player_id
            confidence = face_evidence.confidence
            metadata["applied"].append("face_only")
        elif player_id == face_evidence.player_id:
            confidence = min(
                1.0,
                max(confidence, face_evidence.confidence) + face_agreement_bonus,
            )
            metadata["applied"].append("face_agreement_boost")
        elif face_evidence.confidence > confidence + face_override_margin:
            player_id = face_evidence.player_id
            confidence = face_evidence.confidence
            metadata["applied"].append("face_override")

    if jersey_evidence is not None:
        metadata["jersey_ocr"] = {
            "jersey_number": jersey_evidence.jersey_number,
            "player_id": jersey_evidence.player_id,
            "confidence": jersey_evidence.confidence,
            "support_frames": jersey_evidence.support_frames,
            "candidate_player_ids": jersey_evidence.candidate_player_ids,
            "ambiguous": jersey_evidence.ambiguous,
        }
        jersey_player_id = jersey_evidence.player_id
        if jersey_player_id is not None:
            if player_id is None and jersey_evidence.confidence >= suggest_threshold:
                player_id = jersey_player_id
                confidence = jersey_evidence.confidence
                metadata["applied"].append("jersey_only")
            elif player_id == jersey_player_id:
                confidence = min(
                    1.0,
                    max(confidence, jersey_evidence.confidence) + jersey_agreement_bonus,
                )
                metadata["applied"].append("jersey_agreement_boost")
            elif jersey_evidence.confidence > confidence + jersey_override_margin:
                player_id = jersey_player_id
                confidence = jersey_evidence.confidence
                metadata["applied"].append("jersey_override")

    if player_id is None:
        method = "new_player"
    elif confidence >= auto_threshold:
        method = "auto"
    else:
        method = "suggested"

    return player_id, float(confidence), method, metadata


def apply_substitution_locks(
    assignments: list[dict[str, Any]],
    lock_confidence_threshold: float = 0.82,
    overlap_conflict_frames: int = 45,
    substitution_gap_frames: int = 150,
    demote_conflicting_auto: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Apply per-player lock/unlock state with substitution-aware transitions.
    """
    summary = {
        "locks_applied": 0,
        "overlap_conflicts": 0,
        "substitution_unlocks": 0,
        "demoted_conflicts": 0,
    }
    if not assignments:
        return assignments, summary

    by_player: dict[int, list[dict[str, Any]]] = {}
    for row in assignments:
        row.setdefault("lock_state", "candidate")
        row.setdefault("lock_reason", None)
        row.setdefault("lock_conflict_with_track_id", None)
        player_id = row.get("player_id")
        if player_id is None:
            continue
        by_player.setdefault(int(player_id), []).append(row)

    def _frame_start(row: dict[str, Any]) -> int:
        value = row.get("frame_start")
        return int(value) if isinstance(value, (int, float)) else -1

    def _frame_end(row: dict[str, Any]) -> int:
        value = row.get("frame_end")
        return int(value) if isinstance(value, (int, float)) else -1

    for rows in by_player.values():
        rows.sort(key=_frame_start)
        active_lock: dict[str, Any] | None = None

        for row in rows:
            confidence = float(row.get("confidence") or 0.0)
            if confidence < lock_confidence_threshold:
                continue

            if active_lock is None:
                row["lock_state"] = "locked"
                row["lock_reason"] = "initial_lock"
                summary["locks_applied"] += 1
                active_lock = row
                continue

            cur_start = _frame_start(row)
            cur_end = _frame_end(row)
            prev_start = _frame_start(active_lock)
            prev_end = _frame_end(active_lock)

            overlap = max(0, min(cur_end, prev_end) - max(cur_start, prev_start) + 1)
            gap = cur_start - prev_end

            if overlap > overlap_conflict_frames and gap < substitution_gap_frames:
                summary["overlap_conflicts"] += 1
                cur_conf = float(row.get("confidence") or 0.0)
                prev_conf = float(active_lock.get("confidence") or 0.0)

                winner = row if cur_conf > prev_conf else active_lock
                loser = active_lock if winner is row else row

                winner["lock_state"] = "locked"
                if winner.get("lock_reason") is None:
                    winner["lock_reason"] = "conflict_winner"

                loser["lock_state"] = "unlocked"
                loser["lock_reason"] = "overlap_conflict"
                loser["lock_conflict_with_track_id"] = winner.get("track_id")
                if demote_conflicting_auto and loser.get("match_method") == "auto":
                    loser["match_method"] = "suggested"
                    summary["demoted_conflicts"] += 1

                active_lock = winner
                continue

            # Transition to new lock state (e.g., substitution/phase change).
            row["lock_state"] = "locked"
            if gap >= substitution_gap_frames:
                row["lock_reason"] = "substitution_gap_unlock"
            elif overlap == 0:
                row["lock_reason"] = "non_overlap_transition"
            else:
                row["lock_reason"] = "low_overlap_transition"
            summary["substitution_unlocks"] += 1
            summary["locks_applied"] += 1
            active_lock = row

    return assignments, summary
