"""
core/adaptive_threshold.py
==========================

Phase 7B: Adaptive Confidence Threshold Mechanism.

Implements a deterministic Dual-Threshold Commitment Hysteresis policy:
- Initiation Phase (consecutive_frames < commit_frames):
  Requires a strict entry threshold (default: 0.80) to prevent transient
  flickers, background motion, or pose transitions from starting a streak.
- Sustained Hold Phase (consecutive_frames >= commit_frames):
  Once consecutive_frames >= 3, the user has demonstrated clear physical commitment.
  Permits a slight sustain margin (default: 0.75) to absorb natural optical tracking
  wobble and hand micro-jitters without resetting the accumulated hold duration.

Hard Safety Constraints:
- Bounded strictly between min_bound (0.72) and max_bound (0.85).
- If the gesture label changes or hand is lost, consecutive_frames resets to 0,
  immediately restoring the strict entry threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AdaptiveThresholdConfig:
    """
    Configuration parameters for AdaptiveThresholdPolicy.

    Attributes:
        default_entry_threshold: Threshold applied when consecutive_frames < commit_frames.
                                  Default: 0.80.
        default_sustain_threshold: Threshold applied when consecutive_frames >= commit_frames.
                                   Default: 0.75.
        commit_frames: Consecutive frame count required to enter sustain phase.
                       Frames 1-2 use entry threshold; consecutive_frames >= 3 uses sustain.
                       Default: 3.
        min_bound: Hard floor below which threshold cannot drop. Default: 0.72.
        max_bound: Hard ceiling above which threshold cannot rise. Default: 0.85.
        class_offsets: Optional per-gesture baseline adjustments.
    """
    default_entry_threshold: float = 0.80
    default_sustain_threshold: float = 0.75
    commit_frames: int = 3
    min_bound: float = 0.72
    max_bound: float = 0.85
    class_offsets: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_bound <= self.max_bound <= 1.0):
            raise ValueError(f"Invalid bounds: min_bound={self.min_bound}, max_bound={self.max_bound}")
        if self.default_sustain_threshold > self.default_entry_threshold:
            raise ValueError(
                f"Sustain threshold ({self.default_sustain_threshold}) cannot exceed "
                f"entry threshold ({self.default_entry_threshold})."
            )
        if self.commit_frames < 1:
            raise ValueError(f"commit_frames must be >= 1, got {self.commit_frames}")


class AdaptiveThresholdPolicy:
    """
    Policy engine determining the effective confidence threshold for a given frame.

    Invariants:
    1. Consistent Frame Thresholding:
       - consecutive_frames < commit_frames (e.g. frames 1-2): uses entry threshold.
       - consecutive_frames >= commit_frames (e.g. >= 3): uses sustain threshold.
    2. Strict Clamping:
       - Output is always clamped to [min_bound, max_bound].
    """

    def __init__(self, config: Optional[AdaptiveThresholdConfig] = None) -> None:
        self.config: AdaptiveThresholdConfig = config or AdaptiveThresholdConfig()

    @property
    def entry_threshold(self) -> float:
        return self.config.default_entry_threshold

    @property
    def sustain_threshold(self) -> float:
        return self.config.default_sustain_threshold

    @property
    def commit_frames(self) -> int:
        return self.config.commit_frames

    def get_threshold(
        self,
        gesture: Optional[str] = None,
        consecutive_frames: int = 0,
    ) -> float:
        """
        Calculates the effective confidence threshold.

        Args:
            gesture: The candidate gesture class name (or None).
            consecutive_frames: Current accumulated stable streak for this gesture.

        Returns:
            float: Clamped confidence threshold.
        """
        base_entry = self.config.default_entry_threshold
        base_sustain = self.config.default_sustain_threshold

        # Apply optional class offset if configured
        if gesture and gesture in self.config.class_offsets:
            offset = self.config.class_offsets[gesture]
            base_entry += offset
            base_sustain += offset

        # Policy rule:
        # frames 1-2 (consecutive_frames < 3) use entry threshold
        # consecutive_frames >= 3 uses sustain threshold
        if consecutive_frames >= self.config.commit_frames:
            target = base_sustain
        else:
            target = base_entry

        # Hard clamp between min_bound and max_bound
        return float(max(self.config.min_bound, min(self.config.max_bound, target)))
