"""
core/swipe_detector.py
======================

Phase 8A: Standalone Dynamic Swipe Gesture Detector.

Implements a real-time, dependency-free kinematic swipe detector that analyzes
palm-center trajectories across discrete timestamped frames.

Key Capabilities:
1. Kinematic & Geometric Analysis:
   - Configurable displacement, duration, and velocity thresholds.
   - Horizontal dominance gating (|dx| / |dy| ratio and angular limits).
   - Path efficiency (net displacement vs. integrated arc length) to filter out
     jitter, waviness, circles, and accidental wander.
2. Scale Invariance:
   - Adapts displacement requirements relative to palm size when available.
3. Tracking Continuity & Loss Handling:
   - Resets state on tracking loss or large temporal gaps (> max_tracking_gap).
   - Guards against single-frame teleportation jumps (> max_frame_displacement).
4. Edge Entry Rejection:
   - Rejects inward motion that originates right at the camera boundary edge,
     distinguishing intentional swipes from hands entering the camera view.
5. Dual-Stage Cooldown & Return-Stroke Prevention:
   - Post-swipe global cooldown period.
   - Extended asymmetric return-stroke cooldown suppressing opposite-direction
     bounce-back motion after a confirmed swipe.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Deque, Dict, List, Optional, Tuple


class SwipeDirection(str, Enum):
    """Supported swipe directions."""
    NONE = "NONE"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@dataclass
class TrajectoryPoint:
    """
    Individual sample point in a palm trajectory.

    Attributes:
        x: Normalized X coordinate [0.0, 1.0] (0 = left, 1 = right).
        y: Normalized Y coordinate [0.0, 1.0] (0 = top, 1 = bottom).
        timestamp: Monotonic timestamp in seconds.
        palm_size: Optional reference palm size (normalized distance or bounding box scale).
    """
    x: float
    y: float
    timestamp: float
    palm_size: Optional[float] = None


@dataclass
class SwipeConfig:
    """
    Configuration parameters for SwipeDetector.

    Attributes:
        min_displacement: Minimum normalized horizontal displacement for a swipe [0.0, 1.0].
        min_palm_displacement_ratio: Minimum displacement relative to palm_size when palm_size is given.
        min_duration: Minimum duration (seconds) required for a deliberate swipe (rejects teleports).
        max_duration: Maximum duration (seconds) allowed for a ballistic swipe gesture.
        min_velocity: Minimum horizontal velocity in normalized units per second.
        horizontal_dominance_ratio: Required ratio of |dx| / (|dy| + eps) for directional purity.
        max_vertical_angle_deg: Maximum allowable deviation from horizontal axis in degrees.
        min_path_efficiency: Minimum ratio of net displacement to total trajectory arc length (linearity).
        max_tracking_gap: Maximum allowed elapsed time between consecutive frames before tracking is lost.
        max_frame_displacement: Maximum allowed displacement in a single frame to reject teleportation.
        cooldown_duration: Universal cooldown duration (seconds) after a detected swipe.
        return_stroke_cooldown: Extended cooldown (seconds) suppressing opposite return-stroke motion.
        edge_boundary_margin: Normalized boundary margin where edge-entry rejection applies.
        reject_edge_entry: Whether to reject swipes originating at camera edges.
        min_points: Minimum number of trajectory points required to evaluate a swipe.
    """
    min_displacement: float = 0.15
    min_palm_displacement_ratio: float = 0.80
    min_duration: float = 0.08
    max_duration: float = 0.60
    min_velocity: float = 0.35
    horizontal_dominance_ratio: float = 1.75
    max_vertical_angle_deg: float = 30.0
    min_path_efficiency: float = 0.82
    max_tracking_gap: float = 0.20
    max_frame_displacement: float = 0.40
    cooldown_duration: float = 0.40
    return_stroke_cooldown: float = 0.60
    edge_boundary_margin: float = 0.06
    reject_edge_entry: bool = True
    min_points: int = 3

    def __post_init__(self) -> None:
        if self.min_duration <= 0.0 or self.max_duration <= self.min_duration:
            raise ValueError(
                f"Invalid duration bounds: min_duration={self.min_duration}, max_duration={self.max_duration}"
            )
        if self.min_displacement <= 0.0:
            raise ValueError(f"min_displacement must be > 0, got {self.min_displacement}")
        if self.min_velocity <= 0.0:
            raise ValueError(f"min_velocity must be > 0, got {self.min_velocity}")
        if not (0.0 < self.min_path_efficiency <= 1.0):
            raise ValueError(
                f"min_path_efficiency must be in (0, 1], got {self.min_path_efficiency}"
            )
        if self.return_stroke_cooldown < self.cooldown_duration:
            raise ValueError(
                f"return_stroke_cooldown ({self.return_stroke_cooldown}) must be >= "
                f"cooldown_duration ({self.cooldown_duration})"
            )


@dataclass
class SwipeResult:
    """
    Evaluation output of SwipeDetector for a frame.

    Attributes:
        direction: Detected SwipeDirection (LEFT, RIGHT, or NONE).
        confidence: Estimated confidence score [0.0, 1.0].
        velocity: Horizontal velocity (units/sec).
        displacement: Signed horizontal displacement (dx).
        duration: Elapsed gesture duration (seconds).
        path_efficiency: Linearity ratio of net displacement to integrated arc length.
        start_point: (x, y) coordinates of the gesture start point, if detected.
        end_point: (x, y) coordinates of the gesture end point, if detected.
        timestamp: Timestamp of the frame.
        reason: Diagnostic string explaining detection or rejection cause.
    """
    direction: SwipeDirection = SwipeDirection.NONE
    confidence: float = 0.0
    velocity: float = 0.0
    displacement: float = 0.0
    duration: float = 0.0
    path_efficiency: float = 0.0
    start_point: Optional[Tuple[float, float]] = None
    end_point: Optional[Tuple[float, float]] = None
    timestamp: float = 0.0
    reason: str = ""


class SwipeDetector:
    """
    Standalone dynamic swipe gesture detector based on palm-center trajectories.
    """

    def __init__(self, config: Optional[SwipeConfig] = None) -> None:
        self.config: SwipeConfig = config or SwipeConfig()
        self._history: Deque[TrajectoryPoint] = deque()
        self._cooldown_until: float = 0.0
        self._return_stroke_cooldown_until: float = 0.0
        self._last_swipe_direction: SwipeDirection = SwipeDirection.NONE
        self._entry_edge: Optional[str] = None
        self._entry_timestamp: float = 0.0

    @property
    def is_in_cooldown(self) -> bool:
        """Indicates whether the detector is currently within general cooldown."""
        return self._history and self._history[-1].timestamp < self._cooldown_until

    def reset(self) -> None:
        """Resets all internal trajectory history and cooldown states."""
        self._history.clear()
        self._cooldown_until = 0.0
        self._return_stroke_cooldown_until = 0.0
        self._last_swipe_direction = SwipeDirection.NONE
        self._entry_edge = None
        self._entry_timestamp = 0.0

    def update(
        self,
        x: float,
        y: float,
        timestamp: float,
        palm_size: Optional[float] = None,
        hand_present: bool = True,
    ) -> SwipeResult:
        """
        Updates trajectory buffer with the current frame observation and evaluates swipe gestures.

        Args:
            x: Normalized palm center X coordinate [0.0, 1.0].
            y: Normalized palm center Y coordinate [0.0, 1.0].
            timestamp: Frame monotonic timestamp in seconds.
            palm_size: Optional palm scale metric (e.g. wrist-to-knuckle distance).
            hand_present: Boolean flag indicating if hand tracking is actively valid.

        Returns:
            SwipeResult containing detection status, direction, and diagnostics.
        """
        # 1. Handle tracking loss
        if not hand_present:
            self._history.clear()
            self._entry_edge = None
            return SwipeResult(
                direction=SwipeDirection.NONE,
                timestamp=timestamp,
                reason="TRACKING_LOST",
            )

        # 2. Check general cooldown
        if timestamp < self._cooldown_until:
            # Clear trajectory during cooldown so return stroke motion cannot accumulate
            self._history.clear()
            self._entry_edge = None
            return SwipeResult(
                direction=SwipeDirection.NONE,
                timestamp=timestamp,
                reason="COOLDOWN_ACTIVE",
            )

        # 3. Check tracking continuity against previous frame
        if self._history:
            prev = self._history[-1]
            dt_prev = timestamp - prev.timestamp

            # Negative or non-monotonic timestamps guard
            if dt_prev <= 0.0:
                dt_prev = 1e-4

            # Check time gap (dropped frames / intermittent detection)
            if dt_prev > self.config.max_tracking_gap:
                self._history.clear()
                self._entry_edge = None

            # Check spatial discontinuity (teleportation / tracking swap)
            elif math.hypot(x - prev.x, y - prev.y) > self.config.max_frame_displacement:
                self._history.clear()
                self._entry_edge = None

        # 4. Check if this is the start of a new trajectory segment for edge entry tracking
        if not self._history:
            if x <= self.config.edge_boundary_margin:
                self._entry_edge = "LEFT"
                self._entry_timestamp = timestamp
            elif x >= (1.0 - self.config.edge_boundary_margin):
                self._entry_edge = "RIGHT"
                self._entry_timestamp = timestamp
            else:
                self._entry_edge = None
                self._entry_timestamp = 0.0
        else:
            # If entry edge was recorded, clear it once the hand has remained tracked past max_duration
            if self._entry_edge and (timestamp - self._entry_timestamp > self.config.max_duration):
                self._entry_edge = None

        # 5. Add current point and prune outdated history
        current_pt = TrajectoryPoint(x=x, y=y, timestamp=timestamp, palm_size=palm_size)
        self._history.append(current_pt)
        self._prune_history(timestamp)

        # 6. Check point count requirement
        if len(self._history) < self.config.min_points:
            return SwipeResult(
                direction=SwipeDirection.NONE,
                timestamp=timestamp,
                reason="INSUFFICIENT_POINTS",
            )

        # 7. Evaluate candidate swipes ending at current_pt
        return self._evaluate_candidates(current_pt)

    def _prune_history(self, current_time: float) -> None:
        """Removes trajectory points older than max_duration from the buffer."""
        cutoff = current_time - self.config.max_duration
        while self._history and self._history[0].timestamp < cutoff:
            self._history.popleft()

    def _evaluate_candidates(self, end_pt: TrajectoryPoint) -> SwipeResult:
        """
        Scans possible trajectory start points to determine if a valid swipe occurred.
        """
        best_candidate: Optional[SwipeResult] = None
        best_score: float = -1.0
        rejection_reasons: List[str] = []

        history_list = list(self._history)
        n = len(history_list)

        # Scan candidate start points from earliest to latest valid point
        for i in range(n - self.config.min_points + 1):
            start_pt = history_list[i]
            dt = end_pt.timestamp - start_pt.timestamp

            if dt < self.config.min_duration:
                continue
            if dt > self.config.max_duration:
                continue

            dx = end_pt.x - start_pt.x
            dy = end_pt.y - start_pt.y
            abs_dx = abs(dx)
            abs_dy = abs(dy)

            # Determine direction
            if dx > 0:
                candidate_dir = SwipeDirection.RIGHT
            elif dx < 0:
                candidate_dir = SwipeDirection.LEFT
            else:
                rejection_reasons.append("ZERO_DISPLACEMENT")
                continue

            # Scale-invariant displacement check
            palm_ref = end_pt.palm_size if end_pt.palm_size is not None else start_pt.palm_size
            if palm_ref is not None and palm_ref > 0.0:
                eff_min_disp = max(
                    self.config.min_displacement * 0.5,
                    palm_ref * self.config.min_palm_displacement_ratio,
                )
            else:
                eff_min_disp = self.config.min_displacement

            if abs_dx < eff_min_disp:
                rejection_reasons.append("LOW_DISPLACEMENT")
                continue

            # Velocity check
            vx = abs_dx / dt
            if vx < self.config.min_velocity:
                rejection_reasons.append("LOW_VELOCITY")
                continue

            # Horizontal dominance check
            dominance_ratio = abs_dx / (abs_dy + 1e-6)
            if dominance_ratio < self.config.horizontal_dominance_ratio:
                rejection_reasons.append("LOW_HORIZONTAL_DOMINANCE")
                continue

            angle_deg = math.degrees(math.atan2(abs_dy, abs_dx))
            if angle_deg > self.config.max_vertical_angle_deg:
                rejection_reasons.append("EXCEEDED_MAX_ANGLE")
                continue

            # Path efficiency (linearity) check
            sub_points = history_list[i:]
            arc_length = 0.0
            for j in range(1, len(sub_points)):
                arc_length += math.hypot(
                    sub_points[j].x - sub_points[j - 1].x,
                    sub_points[j].y - sub_points[j - 1].y,
                )

            net_disp = math.hypot(dx, dy)
            path_efficiency = net_disp / max(arc_length, 1e-6)

            if path_efficiency < self.config.min_path_efficiency:
                rejection_reasons.append("LOW_PATH_EFFICIENCY")
                continue

            # Edge entry check
            if self.config.reject_edge_entry:
                # Check if entire trajectory originated at the corresponding entry edge
                if self._entry_edge == "LEFT" and candidate_dir == SwipeDirection.RIGHT:
                    rejection_reasons.append("EDGE_ENTRY_REJECTED")
                    continue
                if self._entry_edge == "RIGHT" and candidate_dir == SwipeDirection.LEFT:
                    rejection_reasons.append("EDGE_ENTRY_REJECTED")
                    continue

                # Also verify the candidate start point is not directly in the boundary margin
                if candidate_dir == SwipeDirection.RIGHT and start_pt.x <= self.config.edge_boundary_margin:
                    rejection_reasons.append("EDGE_ENTRY_REJECTED")
                    continue
                if candidate_dir == SwipeDirection.LEFT and start_pt.x >= (1.0 - self.config.edge_boundary_margin):
                    rejection_reasons.append("EDGE_ENTRY_REJECTED")
                    continue

            # Calculate confidence score
            norm_vel = min(vx / (2.0 * self.config.min_velocity), 1.0)
            norm_dom = min(dominance_ratio / (2.0 * self.config.horizontal_dominance_ratio), 1.0)
            confidence = max(
                0.0,
                min(1.0, 0.4 * path_efficiency + 0.3 * norm_vel + 0.3 * norm_dom),
            )

            # Heuristic score for selecting best candidate window
            score = abs_dx * confidence

            if score > best_score:
                best_score = score
                best_candidate = SwipeResult(
                    direction=candidate_dir,
                    confidence=round(confidence, 3),
                    velocity=round(vx, 3),
                    displacement=round(dx, 3),
                    duration=round(dt, 3),
                    path_efficiency=round(path_efficiency, 3),
                    start_point=(round(start_pt.x, 3), round(start_pt.y, 3)),
                    end_point=(round(end_pt.x, 3), round(end_pt.y, 3)),
                    timestamp=end_pt.timestamp,
                    reason="SWIPE_DETECTED",
                )

        if best_candidate is not None:
            # 8. Check return-stroke cooldown
            if (
                end_pt.timestamp < self._return_stroke_cooldown_until
                and self._last_swipe_direction != SwipeDirection.NONE
                and best_candidate.direction != self._last_swipe_direction
            ):
                self._history.clear()
                self._entry_edge = None
                return SwipeResult(
                    direction=SwipeDirection.NONE,
                    timestamp=end_pt.timestamp,
                    reason="RETURN_STROKE_SUPPRESSED",
                )

            # 9. Register confirmed swipe and trigger cooldowns
            self._cooldown_until = end_pt.timestamp + self.config.cooldown_duration
            self._return_stroke_cooldown_until = end_pt.timestamp + self.config.return_stroke_cooldown
            self._last_swipe_direction = best_candidate.direction
            self._history.clear()
            self._entry_edge = None
            return best_candidate

        # No valid candidate passed all gates
        primary_reason = rejection_reasons[0] if rejection_reasons else "NO_SWIPE"
        return SwipeResult(
            direction=SwipeDirection.NONE,
            timestamp=end_pt.timestamp,
            reason=primary_reason,
        )
