"""
core/temporal_filter.py
=======================

IntentSense Phase 3: Confidence Thresholding & Temporal Stability Filter.

A gesture prediction from a machine learning model (e.g., Random Forest)
must NOT immediately become a command. Real-world camera feeds produce
intermittent misclassifications, transition jitters, and sensor noise.

This module provides a robust, modular temporal filter that:
1. Ingests model predictions frame-by-frame.
2. Extracts prediction and confidence from Random Forest predict_proba().
3. Enforces a configurable confidence threshold.
4. Tracks recent predictions using a configurable sliding window.
5. Requires a configurable minimum number of consecutive/stable frames.
6. Rejects unstable, changing, or low-confidence predictions.
7. Exposes whether a gesture is CONFIRMED or NOT_CONFIRMED.
8. Applies a configurable cooldown to prevent rapid repeated confirmations.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class ConfirmationStatus(str, Enum):
    """Status enumeration for temporal filter output."""
    CONFIRMED = "CONFIRMED"
    NOT_CONFIRMED = "NOT_CONFIRMED"


@dataclass(frozen=True)
class PredictionRecord:
    """Historical record for a single frame prediction in the sliding window."""
    gesture: Optional[str]
    confidence: float
    frame_id: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class TemporalFilterResult:
    """
    Standardized result emitted by TemporalFilter for each processed frame.

    Attributes:
        status: ConfirmationStatus (CONFIRMED or NOT_CONFIRMED).
        gesture: The confirmed gesture label if CONFIRMED, else None.
        confidence: The confidence score of the current frame.
        candidate_gesture: The current candidate gesture being tracked.
        consecutive_frames: Count of consecutive stable frames accumulated.
        cooldown_remaining: Cooldown frames remaining for this candidate gesture.
        reason: Diagnostic reason code for the filter decision.
        hold_duration: Continuous hold duration (in seconds) of the candidate gesture.
    """
    status: ConfirmationStatus
    gesture: Optional[str]
    confidence: float
    candidate_gesture: Optional[str]
    consecutive_frames: int
    cooldown_remaining: int
    reason: str
    hold_duration: float = 0.0

    @property
    def is_confirmed(self) -> bool:
        """Returns True if the gesture is confirmed, False otherwise."""
        return self.status == ConfirmationStatus.CONFIRMED

    @property
    def confirmed_gesture(self) -> Optional[str]:
        """Convenience property returning the confirmed gesture name or None."""
        return self.gesture if self.is_confirmed else None


def extract_prediction_from_proba(
    probabilities: Union[Sequence[float], np.ndarray],
    classes: Sequence[str],
) -> Tuple[str, float]:
    """
    Extracts the highest-probability class label and its confidence score from predict_proba() output.

    Args:
        probabilities: Array-like or sequence of predicted class probabilities.
        classes: Sequence of class names corresponding to probability indices.

    Returns:
        Tuple[str, float]: (best_gesture_label, confidence_score)

    Raises:
        ValueError: If probabilities is empty or lengths do not match classes.
    """
    proba_arr = np.asarray(probabilities, dtype=float).ravel()
    if proba_arr.size == 0 or len(classes) == 0:
        raise ValueError("Probabilities and classes must not be empty.")
    if proba_arr.size != len(classes):
        raise ValueError(
            f"Shape mismatch: {proba_arr.size} probabilities vs {len(classes)} classes."
        )

    best_idx = int(np.argmax(proba_arr))
    best_label = str(classes[best_idx])
    best_confidence = float(proba_arr[best_idx])
    return best_label, best_confidence


def predict_with_confidence(
    model: Any,
    features: Union[Sequence[float], np.ndarray],
) -> Tuple[str, float]:
    """
    Executes predict_proba() using a trained scikit-learn model or model bundle.

    Args:
        model: Trained classifier (with predict_proba and classes_)
               or a dictionary model bundle containing {'model': ..., 'classes': ...}.
        features: 1D sequence or 2D array of feature values (e.g. 42 normalized coords).

    Returns:
        Tuple[str, float]: (predicted_gesture_label, confidence_score)
    """
    # Support model bundle dictionary as saved by train_model.py
    clf = model.get("model", model) if isinstance(model, dict) else model

    feat_arr = np.asarray(features, dtype=float)
    if feat_arr.ndim == 1:
        feat_arr = feat_arr.reshape(1, -1)

    probas = clf.predict_proba(feat_arr)[0]
    classes = getattr(clf, "classes_", None)
    if classes is None and isinstance(model, dict):
        classes = model.get("classes", None)

    if classes is None:
        raise AttributeError("Classifier has no classes_ attribute and none provided.")

    return extract_prediction_from_proba(probas, classes)


class TemporalFilter:
    """
    Frame-by-frame temporal stability and confidence filter for gesture recognition.

    Design & Invariants:
    --------------------
    1. Confidence Gate:
       Predictions with confidence < confidence_threshold are rejected immediately.
    2. Stability Requirement:
       A gesture must maintain confidence >= confidence_threshold for at least
       `stable_frames` consecutive frames before it is CONFIRMED.
    3. Fluctuation Rejection:
       Any transition to a different gesture or to None breaks the consecutive
       sequence, preventing accidental triggers during hand transitions.
    4. Cooldown Mechanism:
       Upon confirmation, a cooldown of `cooldown_frames` is activated for that
       gesture (or globally if configured). During cooldown, repeated confirmations
       are blocked.
    5. Sliding Window Tracking:
       The filter maintains a configurable FIFO sliding window of recent predictions
       for introspection, smoothing, and validation.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.80,
        stable_frames: int = 5,
        cooldown_frames: int = 15,
        window_size: Optional[int] = None,
        cooldown_same_gesture_only: bool = True,
        cooldown_blocks_accumulation: bool = True,
        min_hold_duration: float = 0.5,
        threshold_policy: Optional[Any] = None,
    ) -> None:
        """
        Initializes the TemporalFilter.

        Args:
            confidence_threshold: Minimum probability (0.0 to 1.0) required to consider
                                  a prediction valid. Default: 0.80.
            stable_frames: Number of consecutive high-confidence frames required
                           for gesture confirmation. Default: 5.
            cooldown_frames: Number of frames during which repeated confirmation is
                             blocked after a confirmation. Default: 15.
            window_size: Size of sliding history window (must be >= stable_frames).
                         Defaults to max(stable_frames, 10).
            cooldown_same_gesture_only: If True, cooldown only suppresses the confirmed
                                        gesture. If False, all gestures are blocked
                                        during cooldown. Default: True.
            cooldown_blocks_accumulation: If True, stability accumulation for a gesture
                                         is held at 0 while in cooldown. Default: True.
            min_hold_duration: Minimum continuous hold duration in seconds required
                               for confirmation (measured via time.monotonic() or timestamps).
                               Default: 0.5 seconds.
            threshold_policy: Optional adaptive threshold policy instance (e.g. AdaptiveThresholdPolicy).
                              When None, uses fixed self.confidence_threshold.
        """
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
            )
        if stable_frames < 1:
            raise ValueError(f"stable_frames must be >= 1, got {stable_frames}")
        if cooldown_frames < 0:
            raise ValueError(f"cooldown_frames must be >= 0, got {cooldown_frames}")
        if min_hold_duration < 0.0:
            raise ValueError(f"min_hold_duration must be >= 0.0, got {min_hold_duration}")

        self.confidence_threshold: float = float(confidence_threshold)
        self.stable_frames: int = int(stable_frames)
        self.cooldown_frames: int = int(cooldown_frames)
        self.cooldown_same_gesture_only: bool = bool(cooldown_same_gesture_only)
        self.cooldown_blocks_accumulation: bool = bool(cooldown_blocks_accumulation)
        self.min_hold_duration: float = float(min_hold_duration)
        self.threshold_policy: Optional[Any] = threshold_policy

        if window_size is None:
            self.window_size: int = max(self.stable_frames, 10)
        else:
            if window_size < self.stable_frames:
                raise ValueError(
                    f"window_size ({window_size}) must be >= stable_frames ({self.stable_frames})"
                )
            self.window_size = int(window_size)

        # Internal state
        self.history: Deque[PredictionRecord] = deque(maxlen=self.window_size)
        self.candidate_gesture: Optional[str] = None
        self.consecutive_frames: int = 0
        self.frame_count: int = 0
        self._cooldown_counters: Dict[str, int] = {}
        self._global_cooldown: int = 0
        self._last_confirmed_gesture: Optional[str] = None
        self._last_confirmed_frame: int = -1

        # Phase 3.1: Hold duration tracking
        self._hold_start_time: Optional[float] = None
        self._current_hold_duration: float = 0.0

    def get_effective_threshold(self, gesture: Optional[str]) -> float:
        """
        Determines the effective confidence threshold for the candidate gesture.

        When threshold_policy is configured:
            Calculates 1-based consecutive frame count:
            - frames 1-2 (consecutive_frames < 3) use the entry threshold.
            - consecutive_frames >= 3 uses the sustain threshold.
        When threshold_policy is None:
            Returns self.confidence_threshold (exact fixed baseline behavior).
        """
        if self.threshold_policy is not None:
            if gesture is not None and gesture == self.candidate_gesture:
                current_consecutive = self.consecutive_frames + 1
            else:
                current_consecutive = 1
            return float(self.threshold_policy.get_threshold(gesture, current_consecutive))
        return self.confidence_threshold

    def get_cooldown(self, gesture: Optional[str]) -> int:
        """
        Returns the remaining cooldown frames for the given gesture or global cooldown.

        Args:
            gesture: The gesture label to check.

        Returns:
            int: Remaining cooldown frames (0 if not in cooldown).
        """
        if not self.cooldown_same_gesture_only:
            return self._global_cooldown
        if gesture is None:
            return 0
        return self._cooldown_counters.get(gesture, 0)

    def _tick_cooldowns(self) -> None:
        """Decrements all active cooldown counters by 1 (down to 0)."""
        for g in list(self._cooldown_counters.keys()):
            if self._cooldown_counters[g] > 0:
                self._cooldown_counters[g] -= 1
                if self._cooldown_counters[g] == 0:
                    del self._cooldown_counters[g]

        if self._global_cooldown > 0:
            self._global_cooldown -= 1

    def process(
        self,
        gesture: Optional[str],
        confidence: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> TemporalFilterResult:
        """
        Processes a single frame's prediction and confidence through the temporal filter.

        Args:
            gesture: Predicted gesture label (e.g. 'Fist', 'OpenPalm'), or None if
                     no hand is detected or frame is empty.
            confidence: Model confidence score (0.0 to 1.0).
            timestamp: Optional monotonic timestamp in seconds. If None, time.monotonic()
                       is called automatically.

        Returns:
            TemporalFilterResult with confirmation status, hold duration, and diagnostics.
        """
        now: float = time.monotonic() if timestamp is None else float(timestamp)
        self.frame_count += 1
        confidence = float(max(0.0, min(1.0, confidence)))

        # Sanitize empty string to None
        if gesture is not None and not str(gesture).strip():
            gesture = None

        # Check cooldown state BEFORE ticking for this frame
        cooldown_remaining_before_tick = self.get_cooldown(gesture)

        # Tick cooldowns for the current frame
        self._tick_cooldowns()

        # Append to sliding history window
        record = PredictionRecord(
            gesture=gesture,
            confidence=confidence,
            frame_id=self.frame_count,
            timestamp=now,
        )
        self.history.append(record)

        # -------------------------------------------------------------
        # 1. Handle Empty / Missing Input -> Resets streak & hold time
        # -------------------------------------------------------------
        if gesture is None:
            self.candidate_gesture = None
            self.consecutive_frames = 0
            self._hold_start_time = None
            self._current_hold_duration = 0.0
            return TemporalFilterResult(
                status=ConfirmationStatus.NOT_CONFIRMED,
                gesture=None,
                confidence=0.0,
                candidate_gesture=None,
                consecutive_frames=0,
                cooldown_remaining=0,
                reason="NO_GESTURE",
                hold_duration=0.0,
            )

        # -------------------------------------------------------------
        # 2. Confidence Threshold Gate -> Resets streak & hold time
        # -------------------------------------------------------------
        if confidence < self.get_effective_threshold(gesture):
            self.candidate_gesture = gesture
            self.consecutive_frames = 0
            self._hold_start_time = None
            self._current_hold_duration = 0.0
            return TemporalFilterResult(
                status=ConfirmationStatus.NOT_CONFIRMED,
                gesture=None,
                confidence=confidence,
                candidate_gesture=gesture,
                consecutive_frames=0,
                cooldown_remaining=self.get_cooldown(gesture),
                reason="LOW_CONFIDENCE",
                hold_duration=0.0,
            )

        # -------------------------------------------------------------
        # 3. Check Cooldown
        # -------------------------------------------------------------
        if cooldown_remaining_before_tick > 0:
            current_cd = self.get_cooldown(gesture)
            if self.cooldown_blocks_accumulation:
                self.candidate_gesture = gesture
                self.consecutive_frames = 0
                self._hold_start_time = None
                self._current_hold_duration = 0.0
            else:
                if gesture == self.candidate_gesture:
                    self.consecutive_frames += 1
                    if self._hold_start_time is None:
                        self._hold_start_time = now
                    self._current_hold_duration = max(0.0, now - self._hold_start_time)
                else:
                    self.candidate_gesture = gesture
                    self.consecutive_frames = 1
                    self._hold_start_time = now
                    self._current_hold_duration = 0.0

            return TemporalFilterResult(
                status=ConfirmationStatus.NOT_CONFIRMED,
                gesture=None,
                confidence=confidence,
                candidate_gesture=gesture,
                consecutive_frames=self.consecutive_frames,
                cooldown_remaining=current_cd,
                reason="IN_COOLDOWN",
                hold_duration=self._current_hold_duration,
            )

        # -------------------------------------------------------------
        # 4. Gesture Consistency & Hold Duration Accumulation
        # -------------------------------------------------------------
        if gesture == self.candidate_gesture:
            self.consecutive_frames += 1
            if self._hold_start_time is None:
                self._hold_start_time = now
            self._current_hold_duration = max(0.0, now - self._hold_start_time)
        else:
            # Gesture changed -> reset streak and reset hold timing for new candidate
            self.candidate_gesture = gesture
            self.consecutive_frames = 1
            self._hold_start_time = now
            self._current_hold_duration = 0.0

        # -------------------------------------------------------------
        # 5. Stability Gate: Consecutive Frames Requirement
        # -------------------------------------------------------------
        if self.consecutive_frames < self.stable_frames:
            return TemporalFilterResult(
                status=ConfirmationStatus.NOT_CONFIRMED,
                gesture=None,
                confidence=confidence,
                candidate_gesture=gesture,
                consecutive_frames=self.consecutive_frames,
                cooldown_remaining=0,
                reason="INSUFFICIENT_FRAMES",
                hold_duration=self._current_hold_duration,
            )

        # -------------------------------------------------------------
        # 6. Stability Gate: Minimum Hold Duration Requirement
        # -------------------------------------------------------------
        if self._current_hold_duration < (self.min_hold_duration - 1e-5):
            return TemporalFilterResult(
                status=ConfirmationStatus.NOT_CONFIRMED,
                gesture=None,
                confidence=confidence,
                candidate_gesture=gesture,
                consecutive_frames=self.consecutive_frames,
                cooldown_remaining=0,
                reason="INSUFFICIENT_HOLD_TIME",
                hold_duration=self._current_hold_duration,
            )

        # -------------------------------------------------------------
        # 7. All Conditions Satisfied -> CONFIRMED
        # -------------------------------------------------------------
        confirmed_gesture = gesture
        confirmed_hold_time = self._current_hold_duration

        # Activate cooldown
        if self.cooldown_frames > 0:
            self._cooldown_counters[confirmed_gesture] = self.cooldown_frames
            self._global_cooldown = self.cooldown_frames

        self._last_confirmed_gesture = confirmed_gesture
        self._last_confirmed_frame = self.frame_count

        # Reset consecutive frames and hold timing to prevent repeated immediate confirmation
        frames_matched = self.consecutive_frames
        self.consecutive_frames = 0
        self._hold_start_time = None
        self._current_hold_duration = 0.0

        return TemporalFilterResult(
            status=ConfirmationStatus.CONFIRMED,
            gesture=confirmed_gesture,
            confidence=confidence,
            candidate_gesture=confirmed_gesture,
            consecutive_frames=frames_matched,
            cooldown_remaining=self.cooldown_frames,
            reason="CONFIRMED",
            hold_duration=confirmed_hold_time,
        )

    def process_proba(
        self,
        probabilities: Union[Sequence[float], np.ndarray],
        classes: Sequence[str],
        timestamp: Optional[float] = None,
    ) -> TemporalFilterResult:
        """
        Accepts model predict_proba() output directly, extracts top prediction and
        confidence, and processes through the filter.

        Args:
            probabilities: Array or list of class probabilities from predict_proba().
            classes: Class labels corresponding to the probabilities.
            timestamp: Optional monotonic timestamp in seconds.

        Returns:
            TemporalFilterResult
        """
        gesture, confidence = extract_prediction_from_proba(probabilities, classes)
        return self.process(gesture, confidence, timestamp=timestamp)

    def process_model(
        self,
        model: Any,
        features: Union[Sequence[float], np.ndarray],
        timestamp: Optional[float] = None,
    ) -> TemporalFilterResult:
        """
        Runs model.predict_proba() on the feature vector and processes the result.

        Args:
            model: Trained scikit-learn classifier or model bundle dictionary.
            features: 42 normalized landmark features.
            timestamp: Optional monotonic timestamp in seconds.

        Returns:
            TemporalFilterResult
        """
        gesture, confidence = predict_with_confidence(model, features)
        return self.process(gesture, confidence, timestamp=timestamp)

    def reset(self) -> None:
        """Resets all internal state, history window, consecutive counts, hold timing, and cooldowns."""
        self.history.clear()
        self.candidate_gesture = None
        self.consecutive_frames = 0
        self.frame_count = 0
        self._cooldown_counters.clear()
        self._global_cooldown = 0
        self._last_confirmed_gesture = None
        self._last_confirmed_frame = -1
        self._hold_start_time = None
        self._current_hold_duration = 0.0

    def get_history(self) -> List[PredictionRecord]:
        """Returns a snapshot list of recent predictions in the sliding window."""
        return list(self.history)

    def get_window_stats(self) -> Dict[str, Any]:
        """
        Computes summary statistics over the current sliding window.

        Returns:
            Dict containing:
                - total_records: Number of records in window
                - gesture_counts: Dict mapping gesture name to frequency
                - average_confidence: Mean confidence across window
                - most_frequent_gesture: Gesture with highest count
        """
        records = list(self.history)
        if not records:
            return {
                "total_records": 0,
                "gesture_counts": {},
                "average_confidence": 0.0,
                "most_frequent_gesture": None,
            }

        counts: Dict[str, int] = {}
        total_conf: float = 0.0
        for r in records:
            key = r.gesture if r.gesture is not None else "None"
            counts[key] = counts.get(key, 0) + 1
            total_conf += r.confidence

        most_freq = max(counts.items(), key=lambda kv: kv[1])[0]
        if most_freq == "None":
            most_freq_gesture = None
        else:
            most_freq_gesture = most_freq

        return {
            "total_records": len(records),
            "gesture_counts": counts,
            "average_confidence": total_conf / len(records),
            "most_frequent_gesture": most_freq_gesture,
        }
