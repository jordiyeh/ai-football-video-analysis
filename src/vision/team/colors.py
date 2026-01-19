"""Jersey color extraction and analysis."""

import numpy as np
import cv2
from typing import Literal


def extract_jersey_color(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    sample_region: Literal["upper", "center", "full"] = "upper",
) -> np.ndarray:
    """
    Extract dominant color from player's jersey region.

    Args:
        frame: Video frame (BGR format)
        bbox: Bounding box (x1, y1, x2, y2)
        sample_region: Which part of bbox to sample
            - "upper": Top 40% (jersey, avoids legs)
            - "center": Middle 40% (torso)
            - "full": Entire bbox

    Returns:
        Mean BGR color as numpy array [B, G, R]
    """
    # Check for NaN or invalid values
    if any(np.isnan(v) or np.isinf(v) for v in bbox):
        return np.array([0, 0, 0], dtype=np.float32)

    x1, y1, x2, y2 = map(int, bbox)
    height = y2 - y1
    width = x2 - x1

    # Ensure valid bbox
    if height <= 0 or width <= 0:
        return np.array([0, 0, 0], dtype=np.float32)

    # Clip to frame bounds
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    # Define sampling region
    if sample_region == "upper":
        # Top 40% of bbox (jersey area, exclude legs)
        y_end = y1 + int(height * 0.4)
        y2 = min(y_end, y2)
    elif sample_region == "center":
        # Middle 40% of bbox (torso)
        y_start = y1 + int(height * 0.3)
        y_end = y1 + int(height * 0.7)
        y1 = y_start
        y2 = min(y_end, y2)

    # Extract region
    region = frame[y1:y2, x1:x2]

    if region.size == 0:
        return np.array([0, 0, 0], dtype=np.float32)

    # Convert to float for better precision
    region_float = region.astype(np.float32)

    # Compute mean color (simple but effective)
    mean_color = np.mean(region_float, axis=(0, 1))

    return mean_color


def extract_dominant_color_kmeans(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    n_colors: int = 3,
    sample_region: Literal["upper", "center", "full"] = "upper",
) -> np.ndarray:
    """
    Extract dominant color using k-means clustering on pixels.

    More robust than mean but slower.

    Args:
        frame: Video frame (BGR format)
        bbox: Bounding box (x1, y1, x2, y2)
        n_colors: Number of dominant colors to extract
        sample_region: Which part of bbox to sample

    Returns:
        Most dominant BGR color as numpy array [B, G, R]
    """
    # Check for NaN or invalid values
    if any(np.isnan(v) or np.isinf(v) for v in bbox):
        return np.array([0, 0, 0], dtype=np.float32)

    x1, y1, x2, y2 = map(int, bbox)
    height = y2 - y1
    width = x2 - x1

    if height <= 0 or width <= 0:
        return np.array([0, 0, 0], dtype=np.float32)

    # Clip to frame bounds
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    # Define sampling region
    if sample_region == "upper":
        y_end = y1 + int(height * 0.4)
        y2 = min(y_end, y2)
    elif sample_region == "center":
        y_start = y1 + int(height * 0.3)
        y_end = y1 + int(height * 0.7)
        y1 = y_start
        y2 = min(y_end, y2)

    # Extract region
    region = frame[y1:y2, x1:x2]

    if region.size == 0:
        return np.array([0, 0, 0], dtype=np.float32)

    # Reshape to list of pixels
    pixels = region.reshape(-1, 3).astype(np.float32)

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels,
        n_colors,
        None,
        criteria,
        attempts=3,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    # Find most common cluster (dominant color)
    label_counts = np.bincount(labels.flatten())
    dominant_idx = np.argmax(label_counts)

    return centers[dominant_idx]


def bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR color to HSV.

    Args:
        bgr: BGR color [B, G, R]

    Returns:
        HSV color [H, S, V]
    """
    # Create 1x1 image with the color
    pixel = bgr.reshape(1, 1, 3).astype(np.uint8)
    hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
    return hsv_pixel[0, 0].astype(np.float32)


def color_distance(color1: np.ndarray, color2: np.ndarray, space: str = "bgr") -> float:
    """
    Calculate distance between two colors.

    Args:
        color1: First color
        color2: Second color
        space: Color space ("bgr" or "hsv")

    Returns:
        Euclidean distance between colors
    """
    if space == "hsv":
        # For HSV, handle hue wraparound (0-180 in OpenCV)
        h1, s1, v1 = color1
        h2, s2, v2 = color2

        # Hue distance (circular)
        h_dist = min(abs(h1 - h2), 180 - abs(h1 - h2))

        # Weight hue more, saturation and value less
        dist = np.sqrt((h_dist * 2) ** 2 + (s1 - s2) ** 2 + (v1 - v2) ** 2)
    else:
        # Simple Euclidean distance in BGR
        dist = np.linalg.norm(color1 - color2)

    return float(dist)


def is_similar_color(
    color1: np.ndarray,
    color2: np.ndarray,
    threshold: float = 50.0,
    space: str = "bgr",
) -> bool:
    """
    Check if two colors are similar.

    Args:
        color1: First color
        color2: Second color
        threshold: Distance threshold for similarity
        space: Color space to use

    Returns:
        True if colors are similar
    """
    return color_distance(color1, color2, space) < threshold
