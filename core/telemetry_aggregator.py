"""
core/telemetry_aggregator.py
============================

IntentSense Phase 4: Telemetry Aggregation & Struggle Detection.

Maintains a bounded, time-based ring buffer of recent decision metadata to
detect physical interaction struggles without sending images, frames, or
42D landmark vectors.

Detects measurable physical patterns:
1. HOLD_DEFICIT: User repeatedly holds a gesture for 40-90% of the target
   duration and drops before confirmation (releasing slightly too early).
2. LOW_CONFIDENCE: 3+ consecutive attempts with confidence slightly below
   the effective threshold.
3. LOCKED_SUPPRESSION: Repeated attempts to execute command gestures while
   the system is in LOCKED state.
4. JITTER_INSTABILITY: Rapid alternation of multiple distinct gestures in
   a very short window (< 1.0s) without holding any pose.

Guarantees:
- Zero raw vision leakage: stores only scalar telemetry (strings, floats, ints).
- Suppresses isolated one-off failures (false-positive prevention).
- Enforces a coaching cooldown to avoid spamming the user interface.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple


class StruggleType(str, Enum):
    """Categorization of measurable physical interaction struggles."""
    HOLD_DEFICIT = "HOLD_DEFICIT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    LOCKED_SUPPRESSION = "LOCKED_SUPPRESSION"
    JITTER_INSTABILITY = "JITTER_INSTABILITY"


@dataclass(frozen=True)
class TelemetryRecord:
    """
    Lightweight, privacy-safe record of a single frame or decision event.
    Strictly contains NO image buffers, video frames, or landmark coordinates.
    """
    timestamp: float
    category: str
    gesture: Optional[str]
    confidence: float
    effective_threshold: float
    hold_duration: float
    target_hold: float
    reason: str
    context: str
    is_swipe: bool = False
    state: str = "LOCKED"

    def __post_init__(self) -> None:
        # Runtime safeguard ensuring no raw landmarks or visual arrays are stored
        for field_name in ("landmarks", "frame", "image", "raw_features"):
            if hasattr(self, field_name):
                raise ValueError(f"TelemetryRecord cannot contain visual field: {field_name}")


@dataclass(frozen=True)
class StrugglePattern:
    """Detailed diagnosis of an identified interaction struggle."""
    pattern_type: StruggleType
    gesture: Optional[str]
    evidence: str
    metrics: Dict[str, float]
    timestamp: float


class TelemetryAggregator:
    """
    Bounded, time-based ring buffer analyzing interaction telemetry to
    proactively identify when a user is struggling with touchless interaction.
    """

    def __init__(
        self,
        window_seconds: float = 6.0,
        max_records: int = 150,
        coaching_cooldown: float = 8.0,
    ) -> None:
        self.window_seconds: float = window_seconds
        self.max_records: int = max_records
        self.coaching_cooldown: float = coaching_cooldown

        self._records: Deque[TelemetryRecord] = deque(maxlen=max_records)
        self._last_coached_time: Dict[StruggleType, float] = {}
        self._global_last_coached_time: float = 0.0

    @property
    def record_count(self) -> int:
        """Returns the number of active records in the buffer."""
        return len(self._records)

    def record(
        self,
        timestamp: float,
        category: str,
        gesture: Optional[str],
        confidence: float,
        effective_threshold: float = 0.80,
        hold_duration: float = 0.0,
        target_hold: float = 0.50,
        reason: str = "",
        context: str = "MEDIA",
        is_swipe: bool = False,
        state: str = "LOCKED",
    ) -> None:
        """Appends a new scalar telemetry record and prunes expired history."""
        rec = TelemetryRecord(
            timestamp=timestamp,
            category=category,
            gesture=gesture,
            confidence=confidence,
            effective_threshold=effective_threshold,
            hold_duration=hold_duration,
            target_hold=target_hold,
            reason=reason,
            context=context,
            is_swipe=is_swipe,
            state=state,
        )
        self._records.append(rec)
        self.prune(timestamp)

    def prune(self, current_timestamp: float) -> None:
        """Removes records older than the sliding time window."""
        cutoff = current_timestamp - self.window_seconds
        while self._records and self._records[0].timestamp < cutoff:
            self._records.popleft()

    def clear(self) -> None:
        """Clears all records and reset coaching cooldowns."""
        self._records.clear()
        self._last_coached_time.clear()
        self._global_last_coached_time = 0.0

    def detect_struggle(self, current_timestamp: float) -> Optional[StrugglePattern]:
        """
        Analyzes the buffer to detect measurable struggle patterns.
        Enforces cooldown and anti-spam suppression.
        """
        self.prune(current_timestamp)

        # Global coaching cooldown to avoid overwhelming the user
        if current_timestamp - self._global_last_coached_time < self.coaching_cooldown:
            return None

        # Must have sufficient telemetry to make a grounded assessment
        if len(self._records) < 3:
            return None

        # 1. Check for LOCKED_SUPPRESSION: Attempting non-activation commands while locked
        locked_pattern = self._check_locked_suppression(current_timestamp)
        if locked_pattern:
            return self._validate_and_return(locked_pattern, current_timestamp)

        # 2. Check for HOLD_DEFICIT: Repeatedly releasing just before the hold threshold
        hold_pattern = self._check_hold_deficit(current_timestamp)
        if hold_pattern:
            return self._validate_and_return(hold_pattern, current_timestamp)

        # 3. Check for LOW_CONFIDENCE: Gesture recognized but consistently just below threshold
        conf_pattern = self._check_low_confidence(current_timestamp)
        if conf_pattern:
            return self._validate_and_return(conf_pattern, current_timestamp)

        # 4. Check for JITTER_INSTABILITY: Rapidly flipping gestures without stabilizing
        jitter_pattern = self._check_jitter_instability(current_timestamp)
        if jitter_pattern:
            return self._validate_and_return(jitter_pattern, current_timestamp)

        return None

    def _validate_and_return(
        self,
        pattern: StrugglePattern,
        current_timestamp: float,
    ) -> Optional[StrugglePattern]:
        """Ensures per-pattern cooldown is respected before returning."""
        last_time = self._last_coached_time.get(pattern.pattern_type, 0.0)
        if current_timestamp - last_time < self.coaching_cooldown:
            return None

        self._last_coached_time[pattern.pattern_type] = current_timestamp
        self._global_last_coached_time = current_timestamp
        return pattern

    # -------------------------------------------------------------------------
    # Pattern Detection Logic
    # -------------------------------------------------------------------------
    def _check_locked_suppression(self, current_timestamp: float) -> Optional[StrugglePattern]:
        """
        Detects when a user repeatedly tries to command the system (e.g., ThumbsUp, Pointing)
        while the system is LOCKED, rather than using OpenPalm to unlock.
        """
        locked_command_events: List[TelemetryRecord] = [
            r for r in self._records
            if (
                ("REJECTED" in r.category or "LOCKED" in r.category or "LOCKED" in r.reason)
                and (r.state == "LOCKED" or "LOCKED" in r.reason or "LOCKED" in r.category)
                and r.gesture is not None
                and r.gesture != "OpenPalm"
                and r.confidence >= (r.effective_threshold - 0.05)
            )
        ]

        if len(locked_command_events) >= 3:
            # Group by gesture
            gestures = [r.gesture for r in locked_command_events if r.gesture]
            most_common_gesture = max(set(gestures), key=gestures.count) if gestures else "Command"
            return StrugglePattern(
                pattern_type=StruggleType.LOCKED_SUPPRESSION,
                gesture=most_common_gesture,
                evidence=(
                    f"IntentSense is locked. {most_common_gesture} was detected {len(locked_command_events)} times "
                    f"with high confidence, but actions are blocked while LOCKED."
                ),
                metrics={"attempt_count": float(len(locked_command_events))},
                timestamp=current_timestamp,
            )
        return None

    def _check_hold_deficit(self, current_timestamp: float) -> Optional[StrugglePattern]:
        """
        Detects when a user repeatedly reaches 40%-90% of the target hold duration
        and drops before confirmation (e.g. holds 0.35s when 0.50s is required).
        """
        # Look for peak holds that dropped back to 0
        records = list(self._records)
        peaks: List[TelemetryRecord] = []

        for i in range(len(records) - 1):
            curr_rec = records[i]
            next_rec = records[i + 1]

            if curr_rec.gesture and curr_rec.target_hold > 0:
                ratio = curr_rec.hold_duration / curr_rec.target_hold
                # If current was high (40%-90%) and next dropped significantly (< 20%)
                if 0.40 <= ratio < 0.95 and next_rec.hold_duration < 0.20:
                    peaks.append(curr_rec)

        if len(peaks) >= 2:
            latest_peak = peaks[-1]
            avg_hold = sum(p.hold_duration for p in peaks) / len(peaks)
            target = latest_peak.target_hold
            return StrugglePattern(
                pattern_type=StruggleType.HOLD_DEFICIT,
                gesture=latest_peak.gesture,
                evidence=(
                    f"Repeatedly held '{latest_peak.gesture}' for ~{avg_hold:.2f}s "
                    f"before dropping (target: {target:.2f}s)."
                ),
                metrics={
                    "avg_hold": avg_hold,
                    "target_hold": target,
                    "deficit": target - avg_hold,
                    "occurrences": float(len(peaks)),
                },
                timestamp=current_timestamp,
            )
        return None

    def _check_low_confidence(self, current_timestamp: float) -> Optional[StrugglePattern]:
        """
        Detects when a specific gesture is recognized for 4+ frames or 2+ instances
        with confidence slightly below the threshold (e.g., 55% - 78% vs 80% threshold).
        """
        near_misses: List[TelemetryRecord] = [
            r for r in self._records
            if r.gesture and r.category == "REJECTED" and "CONFIDENCE" in r.reason
            and 0.50 <= r.confidence < r.effective_threshold
        ]

        if len(near_misses) >= 3:
            gestures = [r.gesture for r in near_misses if r.gesture]
            target_gesture = max(set(gestures), key=gestures.count) if gestures else None
            matching_events = [r for r in near_misses if r.gesture == target_gesture]

            if len(matching_events) >= 3:
                avg_conf = sum(r.confidence for r in matching_events) / len(matching_events)
                thresh = matching_events[-1].effective_threshold
                return StrugglePattern(
                    pattern_type=StruggleType.LOW_CONFIDENCE,
                    gesture=target_gesture,
                    evidence=(
                        f"Gesture '{target_gesture}' confidence averaged {int(avg_conf*100)}%, "
                        f"just below required {int(thresh*100)}% threshold."
                    ),
                    metrics={
                        "avg_confidence": avg_conf,
                        "threshold": thresh,
                        "gap": thresh - avg_conf,
                        "count": float(len(matching_events)),
                    },
                    timestamp=current_timestamp,
                )
        return None

    def _check_jitter_instability(self, current_timestamp: float) -> Optional[StrugglePattern]:
        """
        Detects rapid alternation between 3+ different gestures in the last 1.5 seconds,
        indicating the user is rapidly cycling hands or unstable posture.
        """
        recent_records = [
            r for r in self._records
            if current_timestamp - r.timestamp <= 1.5 and r.gesture is not None
        ]

        if len(recent_records) >= 8:
            distinct_gestures = set(r.gesture for r in recent_records if r.gesture)
            # 3 or more distinct gestures in 1.5s indicates jitter / indecision
            if len(distinct_gestures) >= 3:
                # Check that none of them had a substantial hold duration
                max_hold = max((r.hold_duration for r in recent_records), default=0.0)
                if max_hold < 0.25:
                    return StrugglePattern(
                        pattern_type=StruggleType.JITTER_INSTABILITY,
                        gesture=None,
                        evidence=(
                            f"Rapidly switched between {len(distinct_gestures)} poses "
                            f"({', '.join(sorted(distinct_gestures))}) without holding."
                        ),
                        metrics={
                            "distinct_gestures": float(len(distinct_gestures)),
                            "max_hold": max_hold,
                        },
                        timestamp=current_timestamp,
                    )
        return None
