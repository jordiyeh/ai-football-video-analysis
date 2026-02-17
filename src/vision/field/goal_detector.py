"""Goal region detection using visual detection and heuristic fallback.

This module provides visual detection of goal regions using Hough line transforms
and goalpost detection, with temporal smoothing and graceful fallback to heuristics.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.config.schemas import GoalRegionDetectionConfig


@dataclass
class GoalRegion:
    """Represents a detected goal region in the frame."""

    name: str  # "top" or "bottom"
    bounds: dict[str, float]  # x_min, x_max, y_min, y_max
    confidence: float
    detection_method: str  # "visual", "heuristic", "interpolated", "blended"


class GoalRegionProvider(ABC):
    """Abstract interface for goal region providers."""

    @abstractmethod
    def get_goal_regions(self, frame_idx: int) -> list[GoalRegion]:
        """
        Get goal regions for a given frame.

        Args:
            frame_idx: Frame index

        Returns:
            List of GoalRegion objects
        """
        pass

    @abstractmethod
    def is_in_goal_region(
        self, position: tuple[float, float], frame_idx: int
    ) -> tuple[bool, str | None]:
        """
        Check if a position is within a goal region.

        Args:
            position: (x, y) position in pixels
            frame_idx: Frame index

        Returns:
            Tuple of (is_in_goal, goal_name)
        """
        pass


class HeuristicGoalRegionProvider(GoalRegionProvider):
    """
    Fallback provider using hardcoded heuristic goal regions.

    This replicates the original behavior of using fixed percentages
    from the frame edges to define goal regions.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        edge_margin: float = 0.15,
        goal_width_fraction: float = 0.30,
    ):
        """
        Initialize heuristic goal region provider.

        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            edge_margin: Goal region extends this fraction from top/bottom edges
            goal_width_fraction: Goal width as fraction of frame width
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.edge_margin = edge_margin
        self.goal_width_fraction = goal_width_fraction

        # Pre-compute goal regions (they don't change)
        self._goal_regions = self._compute_goal_regions()

    def _compute_goal_regions(self) -> list[GoalRegion]:
        """Compute static goal regions from frame dimensions."""
        x_center = self.frame_width / 2
        goal_half_width = (self.frame_width * self.goal_width_fraction) / 2

        top_goal = GoalRegion(
            name="top",
            bounds={
                "x_min": x_center - goal_half_width,
                "x_max": x_center + goal_half_width,
                "y_min": 0,
                "y_max": self.frame_height * self.edge_margin,
            },
            confidence=1.0,  # Heuristic is always "confident" (it's a fixed assumption)
            detection_method="heuristic",
        )

        bottom_goal = GoalRegion(
            name="bottom",
            bounds={
                "x_min": x_center - goal_half_width,
                "x_max": x_center + goal_half_width,
                "y_min": self.frame_height * (1 - self.edge_margin),
                "y_max": self.frame_height,
            },
            confidence=1.0,
            detection_method="heuristic",
        )

        return [top_goal, bottom_goal]

    def get_goal_regions(self, frame_idx: int) -> list[GoalRegion]:
        """Get goal regions (same for all frames in heuristic mode)."""
        return self._goal_regions

    def is_in_goal_region(
        self, position: tuple[float, float], frame_idx: int
    ) -> tuple[bool, str | None]:
        """Check if position is in a goal region."""
        x, y = position

        for region in self._goal_regions:
            bounds = region.bounds
            if (
                bounds["x_min"] <= x <= bounds["x_max"]
                and bounds["y_min"] <= y <= bounds["y_max"]
            ):
                return True, region.name

        return False, None


@dataclass
class _DetectionResult:
    """Internal result from visual detection for a single goal."""

    name: str
    bounds: dict[str, float]
    crossbar_detected: bool
    left_post_detected: bool
    right_post_detected: bool
    in_expected_zone: bool


class GoalRegionDetector(GoalRegionProvider):
    """
    Visual detection of goal regions using Hough lines and goalpost detection.

    Uses a pipeline of:
    1. Canny edge detection + HoughLinesP for pitch lines
    2. HSV white threshold for goalpost detection
    3. Confidence scoring based on detected features
    4. Fallback to heuristic when confidence is low
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        config: "GoalRegionDetectionConfig",
    ):
        """
        Initialize goal region detector.

        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            config: Goal region detection configuration
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.config = config

        # Heuristic fallback provider
        self._heuristic = HeuristicGoalRegionProvider(
            frame_width,
            frame_height,
            edge_margin=config.heuristic_edge_margin,
            goal_width_fraction=config.heuristic_goal_width_fraction,
        )

        # Temporal tracker
        self._tracker = GoalRegionTracker(
            frame_width,
            frame_height,
            smoothing_window=config.smoothing_window_frames,
            max_displacement=config.max_frame_displacement,
            interpolation_max_gap=config.interpolation_max_gap,
        )

        # Cache of per-frame detections (before smoothing)
        self._raw_detections: dict[int, list[GoalRegion]] = {}

        # Cache of final smoothed regions by frame
        self._smoothed_regions: dict[int, list[GoalRegion]] = {}

    def detect_goals(
        self, frame: np.ndarray, frame_idx: int
    ) -> list[GoalRegion]:
        """
        Detect goal regions in a single frame.

        Args:
            frame: BGR image frame
            frame_idx: Frame index

        Returns:
            List of detected GoalRegion objects
        """
        # Check cache
        if frame_idx in self._smoothed_regions:
            return self._smoothed_regions[frame_idx]

        # Run visual detection
        raw_regions = self._detect_visual(frame, frame_idx)

        # Apply temporal smoothing
        if self.config.enable_temporal_smoothing:
            smoothed = self._tracker.update(raw_regions, frame_idx)
        else:
            smoothed = raw_regions

        # Apply fallback strategy based on confidence
        final_regions = self._apply_fallback(smoothed, frame_idx)

        # Cache results
        self._raw_detections[frame_idx] = raw_regions
        self._smoothed_regions[frame_idx] = final_regions

        return final_regions

    def get_goal_regions(self, frame_idx: int) -> list[GoalRegion]:
        """
        Get goal regions for a frame.

        If the frame hasn't been processed, returns heuristic fallback.
        Call detect_goals() first to process frames with visual detection.
        """
        if frame_idx in self._smoothed_regions:
            return self._smoothed_regions[frame_idx]

        # Try interpolation from tracker
        if self.config.enable_temporal_smoothing:
            interpolated = self._tracker.interpolate(frame_idx)
            if interpolated:
                return interpolated

        # Fall back to heuristic
        return self._heuristic.get_goal_regions(frame_idx)

    def is_in_goal_region(
        self, position: tuple[float, float], frame_idx: int
    ) -> tuple[bool, str | None]:
        """Check if position is in a goal region."""
        x, y = position
        regions = self.get_goal_regions(frame_idx)

        for region in regions:
            bounds = region.bounds
            if (
                bounds["x_min"] <= x <= bounds["x_max"]
                and bounds["y_min"] <= y <= bounds["y_max"]
            ):
                return True, region.name

        return False, None

    def _detect_visual(
        self, frame: np.ndarray, frame_idx: int
    ) -> list[GoalRegion]:
        """Run visual detection pipeline on a frame."""
        # 1. Preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Detect pitch lines
        lines = self._detect_pitch_lines(blurred)

        # 3. Detect goalposts via HSV white threshold
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        goalposts = self._detect_goalposts(hsv)

        # 4. Analyze detected features for each goal region
        regions = []
        for goal_name in ["top", "bottom"]:
            result = self._analyze_goal_region(goal_name, lines, goalposts)
            if result:
                confidence = self._compute_confidence(result)
                region = GoalRegion(
                    name=result.name,
                    bounds=result.bounds,
                    confidence=confidence,
                    detection_method="visual",
                )
                regions.append(region)

        return regions

    def _detect_pitch_lines(self, gray: np.ndarray) -> list[np.ndarray]:
        """Detect pitch lines using Canny + HoughLinesP."""
        edges = cv2.Canny(
            gray,
            self.config.canny_low_threshold,
            self.config.canny_high_threshold,
        )

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.hough_min_line_length,
            maxLineGap=self.config.hough_max_line_gap,
        )

        if lines is None:
            return []

        return [line[0] for line in lines]

    def _detect_goalposts(self, hsv: np.ndarray) -> list[dict]:
        """
        Detect goalposts using HSV white threshold.

        Returns list of dicts with 'x', 'y_min', 'y_max', 'width'.
        """
        # Create mask for white pixels
        white_low = np.array(self.config.white_hsv_low)
        white_high = np.array(self.config.white_hsv_high)
        mask = cv2.inRange(hsv, white_low, white_high)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        goalposts = []
        min_height = self.frame_height * self.config.min_goalpost_height

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size and aspect ratio
            if h < min_height:
                continue

            if w == 0:
                continue

            aspect_ratio = h / w
            if aspect_ratio < self.config.goalpost_aspect_ratio_min:
                continue

            goalposts.append({
                "x": x + w / 2,
                "y_min": y,
                "y_max": y + h,
                "width": w,
            })

        return goalposts

    def _analyze_goal_region(
        self,
        goal_name: str,
        lines: list[np.ndarray],
        goalposts: list[dict],
    ) -> _DetectionResult | None:
        """Analyze detected features for a specific goal region."""
        # Get expected zone from heuristic
        heuristic_regions = self._heuristic.get_goal_regions(0)
        heuristic_bounds = None
        for hr in heuristic_regions:
            if hr.name == goal_name:
                heuristic_bounds = hr.bounds
                break

        if heuristic_bounds is None:
            return None

        # Define search region (slightly expanded)
        search_y_min = heuristic_bounds["y_min"]
        search_y_max = heuristic_bounds["y_max"]

        # Look for horizontal lines near expected crossbar position
        crossbar_detected = False
        crossbar_y = None
        tolerance = self.config.line_angle_tolerance

        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

            # Check if horizontal (angle near 0)
            if angle <= tolerance:
                mid_y = (y1 + y2) / 2
                if search_y_min <= mid_y <= search_y_max:
                    crossbar_detected = True
                    crossbar_y = mid_y
                    break

        # Look for goalposts in the region
        left_post_detected = False
        right_post_detected = False

        x_center = self.frame_width / 2
        for post in goalposts:
            # Check if post is in the expected vertical region
            post_in_region = (
                post["y_min"] <= search_y_max and post["y_max"] >= search_y_min
            )
            if not post_in_region:
                continue

            # Classify as left or right post
            if post["x"] < x_center:
                left_post_detected = True
            else:
                right_post_detected = True

        # Determine if goal is in expected zone based on detections
        in_expected_zone = crossbar_detected or (
            left_post_detected and right_post_detected
        )

        # Compute bounds based on detected features or fall back to heuristic
        if crossbar_detected and (left_post_detected or right_post_detected):
            # Use detected features to refine bounds
            bounds = self._refine_bounds_from_detections(
                goal_name, crossbar_y, goalposts, heuristic_bounds
            )
        else:
            bounds = heuristic_bounds

        return _DetectionResult(
            name=goal_name,
            bounds=bounds,
            crossbar_detected=crossbar_detected,
            left_post_detected=left_post_detected,
            right_post_detected=right_post_detected,
            in_expected_zone=in_expected_zone,
        )

    def _refine_bounds_from_detections(
        self,
        goal_name: str,
        crossbar_y: float | None,
        goalposts: list[dict],
        fallback_bounds: dict[str, float],
    ) -> dict[str, float]:
        """Refine goal bounds based on detected features."""
        bounds = dict(fallback_bounds)

        # Refine vertical bounds from crossbar
        if crossbar_y is not None:
            if goal_name == "top":
                # Crossbar defines bottom of goal region
                bounds["y_max"] = crossbar_y + 20  # Small margin
            else:
                # Crossbar defines top of goal region
                bounds["y_min"] = crossbar_y - 20

        # Refine horizontal bounds from goalposts
        relevant_posts = [
            p for p in goalposts
            if p["y_min"] <= bounds["y_max"] and p["y_max"] >= bounds["y_min"]
        ]

        if len(relevant_posts) >= 2:
            x_positions = [p["x"] for p in relevant_posts]
            bounds["x_min"] = min(x_positions) - 10
            bounds["x_max"] = max(x_positions) + 10

        return bounds

    def _compute_confidence(self, result: _DetectionResult) -> float:
        """Compute confidence score from detection result."""
        # Weights: crossbar 35%, posts 30%, expected zone 35%
        score = 0.0

        if result.crossbar_detected:
            score += 0.35

        if result.left_post_detected:
            score += 0.15
        if result.right_post_detected:
            score += 0.15

        if result.in_expected_zone:
            score += 0.35

        return min(1.0, score)

    def _apply_fallback(
        self, regions: list[GoalRegion], frame_idx: int
    ) -> list[GoalRegion]:
        """Apply fallback strategy based on confidence."""
        heuristic_regions = self._heuristic.get_goal_regions(frame_idx)

        # Create lookup for heuristic regions
        heuristic_by_name = {r.name: r for r in heuristic_regions}

        final_regions = []

        # Process each expected goal region
        for goal_name in ["top", "bottom"]:
            # Find visual detection for this goal
            visual_region = None
            for r in regions:
                if r.name == goal_name:
                    visual_region = r
                    break

            heuristic_region = heuristic_by_name.get(goal_name)

            if visual_region is None:
                # No visual detection - use heuristic
                if heuristic_region:
                    final_regions.append(heuristic_region)
                continue

            confidence = visual_region.confidence

            if confidence >= self.config.blend_threshold:
                # High confidence - use visual detection
                final_regions.append(visual_region)
            elif confidence >= self.config.fallback_confidence_threshold:
                # Medium confidence - blend 70% visual + 30% heuristic
                if heuristic_region:
                    blended = self._blend_regions(
                        visual_region, heuristic_region, 0.7
                    )
                    final_regions.append(blended)
                else:
                    final_regions.append(visual_region)
            else:
                # Low confidence - use heuristic or interpolate
                if heuristic_region:
                    final_regions.append(heuristic_region)

        return final_regions

    def _blend_regions(
        self,
        visual: GoalRegion,
        heuristic: GoalRegion,
        visual_weight: float,
    ) -> GoalRegion:
        """Blend two goal regions with weighted average of bounds."""
        heuristic_weight = 1.0 - visual_weight

        blended_bounds = {}
        for key in ["x_min", "x_max", "y_min", "y_max"]:
            blended_bounds[key] = (
                visual.bounds[key] * visual_weight
                + heuristic.bounds[key] * heuristic_weight
            )

        return GoalRegion(
            name=visual.name,
            bounds=blended_bounds,
            confidence=visual.confidence,
            detection_method="blended",
        )


@dataclass
class _TrackedGoal:
    """Internal tracking state for a goal region."""

    name: str
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    last_frame: int = -1


class GoalRegionTracker:
    """
    Temporal smoothing for goal region detections.

    Uses weighted averaging with outlier rejection and interpolation.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        smoothing_window: int = 30,
        max_displacement: float = 50.0,
        interpolation_max_gap: int = 60,
    ):
        """
        Initialize goal region tracker.

        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            smoothing_window: Number of frames for smoothing window
            max_displacement: Max pixels jump between frames to accept
            interpolation_max_gap: Max frames to interpolate across
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.smoothing_window = smoothing_window
        self.max_displacement = max_displacement
        self.interpolation_max_gap = interpolation_max_gap

        # Tracked goals
        self._tracked: dict[str, _TrackedGoal] = {
            "top": _TrackedGoal(name="top", history=deque(maxlen=smoothing_window)),
            "bottom": _TrackedGoal(name="bottom", history=deque(maxlen=smoothing_window)),
        }

    def update(
        self, regions: list[GoalRegion], frame_idx: int
    ) -> list[GoalRegion]:
        """
        Update tracker with new detections and return smoothed regions.

        Args:
            regions: Detected regions for current frame
            frame_idx: Frame index

        Returns:
            Smoothed goal regions
        """
        smoothed = []

        for region in regions:
            tracked = self._tracked.get(region.name)
            if tracked is None:
                continue

            # Check for outlier (large jump from previous)
            if tracked.history and tracked.last_frame >= 0:
                is_outlier = self._is_outlier(region, tracked)
                if is_outlier:
                    # Skip this detection, use interpolation instead
                    interpolated = self._interpolate_single(tracked, frame_idx)
                    if interpolated:
                        smoothed.append(interpolated)
                    continue

            # Add to history
            tracked.history.append((frame_idx, region))
            tracked.last_frame = frame_idx

            # Compute smoothed region
            smoothed_region = self._smooth_region(tracked)
            smoothed.append(smoothed_region)

        return smoothed

    def interpolate(self, frame_idx: int) -> list[GoalRegion]:
        """
        Interpolate goal regions for a frame without detection.

        Args:
            frame_idx: Frame index

        Returns:
            Interpolated goal regions
        """
        regions = []

        for name, tracked in self._tracked.items():
            interpolated = self._interpolate_single(tracked, frame_idx)
            if interpolated:
                regions.append(interpolated)

        return regions

    def _is_outlier(self, region: GoalRegion, tracked: _TrackedGoal) -> bool:
        """Check if region is an outlier based on displacement from history."""
        if not tracked.history:
            return False

        # Get most recent region
        _, last_region = tracked.history[-1]

        # Compute displacement of center
        last_center_x = (last_region.bounds["x_min"] + last_region.bounds["x_max"]) / 2
        last_center_y = (last_region.bounds["y_min"] + last_region.bounds["y_max"]) / 2

        new_center_x = (region.bounds["x_min"] + region.bounds["x_max"]) / 2
        new_center_y = (region.bounds["y_min"] + region.bounds["y_max"]) / 2

        displacement = np.sqrt(
            (new_center_x - last_center_x) ** 2 + (new_center_y - last_center_y) ** 2
        )

        return displacement > self.max_displacement

    def _smooth_region(self, tracked: _TrackedGoal) -> GoalRegion:
        """Compute smoothed region from history."""
        if not tracked.history:
            raise ValueError("Cannot smooth empty history")

        # Weight by recency and confidence
        weights = []
        bounds_list = []

        for i, (frame_idx, region) in enumerate(tracked.history):
            # Recency weight: more recent = higher weight
            recency_weight = (i + 1) / len(tracked.history)
            # Confidence weight
            confidence_weight = region.confidence

            weight = recency_weight * confidence_weight
            weights.append(weight)
            bounds_list.append(region.bounds)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1.0

        # Weighted average of bounds
        smoothed_bounds = {}
        for key in ["x_min", "x_max", "y_min", "y_max"]:
            weighted_sum = sum(
                bounds[key] * w for bounds, w in zip(bounds_list, weights)
            )
            smoothed_bounds[key] = weighted_sum / total_weight

        # Use most recent region's metadata
        _, latest = tracked.history[-1]

        return GoalRegion(
            name=tracked.name,
            bounds=smoothed_bounds,
            confidence=latest.confidence,
            detection_method="smoothed",
        )

    def _interpolate_single(
        self, tracked: _TrackedGoal, frame_idx: int
    ) -> GoalRegion | None:
        """Interpolate a single goal region."""
        if not tracked.history:
            return None

        # Check if within interpolation gap
        _, last_region = tracked.history[-1]
        gap = frame_idx - tracked.last_frame

        if gap > self.interpolation_max_gap:
            return None

        # For now, use last known region with lower confidence
        decay = max(0.1, 1.0 - gap * 0.02)  # 2% decay per frame

        return GoalRegion(
            name=tracked.name,
            bounds=dict(last_region.bounds),
            confidence=last_region.confidence * decay,
            detection_method="interpolated",
        )
