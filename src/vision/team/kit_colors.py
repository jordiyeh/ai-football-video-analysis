"""Kit image color extraction — extract dominant jersey colors from uploaded kit images."""

from pathlib import Path

import cv2
import numpy as np


def extract_kit_colors(
    image_path: Path | str,
    n_dominant: int = 2,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load a kit image, filter out near-white/black pixels, and k-means cluster
    the remaining pixels to find primary and secondary dominant HSV colors.

    Args:
        image_path: Path to the kit image (JPEG/PNG).
        n_dominant: Number of dominant colors to extract (minimum 1).

    Returns:
        Tuple of (primary_hsv, secondary_hsv | None).
        Each HSV array uses OpenCV convention: H=[0,180], S=[0,255], V=[0,255].
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels = hsv_img.reshape(-1, 3).astype(np.float32)

    # Filter out near-white (low saturation, high value) and near-black (low value)
    sat = pixels[:, 1]
    val = pixels[:, 2]
    mask = (sat > 30) & (val > 40) & (val < 240)
    filtered = pixels[mask]

    if len(filtered) < 10:
        # Fallback: use all pixels if too few remain after filtering
        filtered = pixels

    n_colors = max(1, min(n_dominant, len(filtered)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        filtered,
        n_colors,
        None,
        criteria,
        attempts=5,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    # Sort clusters by count (most dominant first)
    label_counts = np.bincount(labels.flatten(), minlength=n_colors)
    sorted_indices = np.argsort(-label_counts)

    primary = centers[sorted_indices[0]]
    secondary = centers[sorted_indices[1]] if n_colors >= 2 else None

    return primary, secondary


def hsv_to_hex(hsv: np.ndarray | list[float]) -> str:
    """
    Convert OpenCV HSV [0-180, 0-255, 0-255] to #RRGGBB hex string.

    Args:
        hsv: HSV color array (OpenCV convention).

    Returns:
        Hex color string like '#1565D8'.
    """
    hsv_arr = np.array(hsv, dtype=np.float32)
    pixel = hsv_arr.reshape(1, 1, 3).astype(np.uint8)
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)
    b, g, r = int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])
    return f"#{r:02X}{g:02X}{b:02X}"
