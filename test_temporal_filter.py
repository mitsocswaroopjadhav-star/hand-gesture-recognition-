"""
test_temporal_filter.py
=======================

Unit test suite for IntentSense Phase 3: Temporal Stability & Confidence Filter.

Covers:
1. Stable repeated gesture -> CONFIRMED after stable_frames
2. Changing / fluctuating gestures -> NOT_CONFIRMED
3. Low confidence predictions (< threshold) -> NOT_CONFIRMED
4. Cooldown mechanism -> Prevents immediate repeated confirmation
5. Reset and empty input handling (None, empty string) -> NOT_CONFIRMED & resets state
6. Custom configurations (custom stable_frames, confidence_threshold, cooldown_frames)
7. Integration with RandomForestClassifier predict_proba() and trained model bundle
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.temporal_filter import (
    ConfirmationStatus,
    TemporalFilter,
    TemporalFilterResult,
    extract_prediction_from_proba,
    predict_with_confidence,
)


class TestTemporalFilter(unittest.TestCase):
    """Test suite for core/temporal_filter.py."""

    def setUp(self):
        """Set up standard filter instance with Phase 3 frame-based defaults."""
        self.default_conf = 0.80
        self.default_stable = 5
        self.default_cooldown = 15
        self.filter = TemporalFilter(
            confidence_threshold=self.default_conf,
            stable_frames=self.default_stable,
            cooldown_frames=self.default_cooldown,
            min_hold_duration=0.0,  # 0.0s for instantaneous frame-counter unit tests
        )

    # -------------------------------------------------------------------------
    # Test 1: Stable Repeated Gesture -> Confirmed
    # -------------------------------------------------------------------------
    def test_stable_repeated_gesture_confirms(self):
        """
        Verify that repeating the same gesture with confidence >= threshold
        remains NOT_CONFIRMED for frames 1..4, and confirms on frame 5.
        """
        # Frames 1 to 4 should NOT be confirmed
        for i in range(1, self.default_stable):
            res = self.filter.process("Fist", 0.92)
            self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
            self.assertFalse(res.is_confirmed)
            self.assertIsNone(res.gesture)
            self.assertEqual(res.candidate_gesture, "Fist")
            self.assertEqual(res.consecutive_frames, i)
            self.assertEqual(res.reason, "INSUFFICIENT_FRAMES")

        # Frame 5 (stable_frames reached) must be CONFIRMED
        res = self.filter.process("Fist", 0.92)
        self.assertEqual(res.status, ConfirmationStatus.CONFIRMED)
        self.assertTrue(res.is_confirmed)
        self.assertEqual(res.gesture, "Fist")
        self.assertEqual(res.confirmed_gesture, "Fist")
        self.assertEqual(res.candidate_gesture, "Fist")
        self.assertEqual(res.consecutive_frames, 5)
        self.assertEqual(res.cooldown_remaining, self.default_cooldown)
        self.assertEqual(res.reason, "CONFIRMED")

    # -------------------------------------------------------------------------
    # Test 2: Changing / Fluctuating Gestures -> Not Confirmed
    # -------------------------------------------------------------------------
    def test_changing_gestures_not_confirmed(self):
        """
        Verify that rapid gesture switching resets the consecutive count
        and prevents confirmation even with 100% confidence.
        """
        gestures = ["Fist", "OpenPalm", "Peace", "Pointing", "ThumbsUp", "ThumbsDown"]
        for g in gestures * 3:
            res = self.filter.process(g, 0.99)
            self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
            self.assertFalse(res.is_confirmed)
            self.assertIsNone(res.gesture)
            self.assertEqual(res.consecutive_frames, 1)

    def test_gesture_switch_at_boundary_breaks_streak(self):
        """
        Verify that 4 frames of 'Fist' followed by 1 frame of 'Peace'
        does NOT confirm on frame 5.
        """
        for _ in range(4):
            res = self.filter.process("Fist", 0.95)
            self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)

        # 5th frame switches to Peace
        res = self.filter.process("Peace", 0.95)
        self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
        self.assertFalse(res.is_confirmed)
        self.assertEqual(res.candidate_gesture, "Peace")
        self.assertEqual(res.consecutive_frames, 1)

    # -------------------------------------------------------------------------
    # Test 3: Low Confidence -> Not Confirmed
    # -------------------------------------------------------------------------
    def test_low_confidence_rejected(self):
        """
        Verify that predictions with confidence below threshold are rejected
        and do not accumulate consecutive frames.
        """
        # Confidence 0.75 is below default 0.80
        for _ in range(10):
            res = self.filter.process("Fist", 0.75)
            self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
            self.assertFalse(res.is_confirmed)
            self.assertIsNone(res.gesture)
            self.assertEqual(res.consecutive_frames, 0)
            self.assertEqual(res.reason, "LOW_CONFIDENCE")

    def test_dip_in_confidence_resets_streak(self):
        """
        Verify that 4 high-confidence frames followed by 1 low-confidence frame
        resets the sequence, requiring a fresh 5 stable frames.
        """
        for _ in range(4):
            self.filter.process("Fist", 0.90)

        # 5th frame dips to 0.70
        res = self.filter.process("Fist", 0.70)
        self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
        self.assertEqual(res.consecutive_frames, 0)
        self.assertEqual(res.reason, "LOW_CONFIDENCE")

        # Frame 6 with high confidence starts streak at 1, not 5
        res = self.filter.process("Fist", 0.90)
        self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
        self.assertEqual(res.consecutive_frames, 1)

    # -------------------------------------------------------------------------
    # Test 4: Cooldown Prevents Immediate Repeated Confirmation
    # -------------------------------------------------------------------------
    def test_cooldown_prevents_immediate_repeated_confirmation(self):
        """
        Verify that once confirmed, the same gesture cannot be confirmed again
        during the cooldown window (default 15 frames).
        """
        # Confirm Fist in 5 frames
        for _ in range(5):
            res = self.filter.process("Fist", 0.90)
        self.assertEqual(res.status, ConfirmationStatus.CONFIRMED)

        # Immediate next frame (Frame 6) MUST be blocked by cooldown
        res = self.filter.process("Fist", 0.90)
        self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
        self.assertFalse(res.is_confirmed)
        self.assertEqual(res.reason, "IN_COOLDOWN")
        self.assertEqual(res.cooldown_remaining, 14)

        # Process remaining 14 cooldown frames
        for _ in range(14):
            res = self.filter.process("Fist", 0.90)
            self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
            self.assertEqual(res.reason, "IN_COOLDOWN")

        # Cooldown is now completely finished (cooldown_remaining == 0)
        self.assertEqual(self.filter.get_cooldown("Fist"), 0)

        # Now building fresh stable frames (5 frames required)
        for i in range(1, 5):
            res = self.filter.process("Fist", 0.90)
            self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
            self.assertEqual(res.consecutive_frames, i)

        # 5th frame after cooldown confirms again!
        res = self.filter.process("Fist", 0.90)
        self.assertEqual(res.status, ConfirmationStatus.CONFIRMED)
        self.assertTrue(res.is_confirmed)

    def test_cooldown_same_gesture_only_allows_other_gesture(self):
        """
        With cooldown_same_gesture_only=True (default), confirming 'Fist'
        should not block a different gesture 'OpenPalm' once stable.
        """
        # Confirm Fist
        for _ in range(5):
            self.filter.process("Fist", 0.95)

        # Now stably perform OpenPalm for 5 frames
        for _ in range(4):
            res = self.filter.process("OpenPalm", 0.95)
            self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)

        res = self.filter.process("OpenPalm", 0.95)
        self.assertEqual(res.status, ConfirmationStatus.CONFIRMED)
        self.assertEqual(res.gesture, "OpenPalm")

    # -------------------------------------------------------------------------
    # Test 5: Reset and Empty Input Handling
    # -------------------------------------------------------------------------
    def test_none_input_resets_streak(self):
        """
        Verify that None input (no hand detected) returns NOT_CONFIRMED
        and resets consecutive stable frame count to 0.
        """
        for _ in range(4):
            self.filter.process("Fist", 0.90)

        # Hand lost / None input
        res = self.filter.process(None, 0.0)
        self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
        self.assertFalse(res.is_confirmed)
        self.assertIsNone(res.candidate_gesture)
        self.assertEqual(res.consecutive_frames, 0)
        self.assertEqual(res.reason, "NO_GESTURE")

        # Hand reappears: must start from 1
        res = self.filter.process("Fist", 0.90)
        self.assertEqual(res.consecutive_frames, 1)
        self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)

    def test_empty_string_treated_as_no_gesture(self):
        """Verify that empty string or whitespace is treated as no gesture."""
        res = self.filter.process("  ", 0.90)
        self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
        self.assertEqual(res.reason, "NO_GESTURE")

    def test_manual_reset(self):
        """Verify that reset() method fully clears all internal state."""
        for _ in range(5):
            self.filter.process("Fist", 0.90)
        self.assertGreater(self.filter.get_cooldown("Fist"), 0)
        self.assertGreater(len(self.filter.history), 0)

        self.filter.reset()

        self.assertEqual(len(self.filter.history), 0)
        self.assertIsNone(self.filter.candidate_gesture)
        self.assertEqual(self.filter.consecutive_frames, 0)
        self.assertEqual(self.filter.get_cooldown("Fist"), 0)
        self.assertEqual(self.filter.frame_count, 0)

    # -------------------------------------------------------------------------
    # Test 6: Configurable Parameters
    # -------------------------------------------------------------------------
    def test_custom_configuration(self):
        """
        Verify custom parameters (e.g., stable_frames=3, conf=0.60, cooldown=4).
        """
        custom_filter = TemporalFilter(
            confidence_threshold=0.60,
            stable_frames=3,
            cooldown_frames=4,
            window_size=6,
            min_hold_duration=0.0,
        )

        # Confidence 0.65 should be accepted
        res = custom_filter.process("Peace", 0.65)
        self.assertEqual(res.consecutive_frames, 1)

        custom_filter.process("Peace", 0.65)
        res = custom_filter.process("Peace", 0.65)
        self.assertEqual(res.status, ConfirmationStatus.CONFIRMED)
        self.assertEqual(res.gesture, "Peace")

        # Cooldown should be 4
        self.assertEqual(custom_filter.get_cooldown("Peace"), 4)

    def test_invalid_parameters_raise(self):
        """Verify that invalid initialization parameters raise ValueError."""
        with self.assertRaises(ValueError):
            TemporalFilter(confidence_threshold=1.5)
        with self.assertRaises(ValueError):
            TemporalFilter(confidence_threshold=-0.1)
        with self.assertRaises(ValueError):
            TemporalFilter(stable_frames=0)
        with self.assertRaises(ValueError):
            TemporalFilter(cooldown_frames=-1)
        with self.assertRaises(ValueError):
            TemporalFilter(stable_frames=5, window_size=3)  # window < stable

    # -------------------------------------------------------------------------
    # Test 7: Sliding Window Statistics & Introspection
    # -------------------------------------------------------------------------
    def test_sliding_window_tracking(self):
        """Verify that the sliding window maintains history and computes statistics."""
        filter_win = TemporalFilter(window_size=5, stable_frames=3)
        filter_win.process("Fist", 0.90)
        filter_win.process("Fist", 0.85)
        filter_win.process("OpenPalm", 0.95)

        stats = filter_win.get_window_stats()
        self.assertEqual(stats["total_records"], 3)
        self.assertEqual(stats["gesture_counts"]["Fist"], 2)
        self.assertEqual(stats["gesture_counts"]["OpenPalm"], 1)
        self.assertEqual(stats["most_frequent_gesture"], "Fist")
        self.assertAlmostEqual(stats["average_confidence"], (0.90 + 0.85 + 0.95) / 3, places=4)

    # -------------------------------------------------------------------------
    # Test 8: RandomForest predict_proba() Integration
    # -------------------------------------------------------------------------
    def test_extract_prediction_from_proba(self):
        """Verify probability extraction utility function."""
        classes = ["Fist", "OpenPalm", "Peace", "Pointing", "ThumbsDown", "ThumbsUp"]
        probas = [0.05, 0.85, 0.02, 0.03, 0.02, 0.03]

        label, conf = extract_prediction_from_proba(probas, classes)
        self.assertEqual(label, "OpenPalm")
        self.assertAlmostEqual(conf, 0.85, places=4)

    def test_process_proba_integration(self):
        """Verify process_proba() method handles probability vectors directly."""
        classes = ["Fist", "OpenPalm", "Peace"]
        probas = [0.90, 0.05, 0.05]

        for _ in range(4):
            res = self.filter.process_proba(probas, classes)
            self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)

        res = self.filter.process_proba(probas, classes)
        self.assertEqual(res.status, ConfirmationStatus.CONFIRMED)
        self.assertEqual(res.gesture, "Fist")

    def test_integration_with_saved_gesture_model(self):
        """
        Verify end-to-end integration with the real trained model
        models/gesture_model.pkl from Phase 2.
        """
        import joblib

        model_path = os.path.join(PROJECT_ROOT, "models", "gesture_model.pkl")
        self.assertTrue(os.path.isfile(model_path), f"Model file not found at {model_path}")

        bundle = joblib.load(model_path)
        self.assertIn("model", bundle)
        self.assertIn("classes", bundle)

        clf = bundle["model"]
        classes = bundle["classes"]

        # Synthetic feature vector (42 elements)
        dummy_features = np.zeros(42)

        # Model should predict probabilities
        proba = clf.predict_proba(dummy_features.reshape(1, -1))[0]
        self.assertEqual(len(proba), len(classes))

        # Test predict_with_confidence helper
        pred_label, pred_conf = predict_with_confidence(bundle, dummy_features)
        self.assertIn(pred_label, classes)
        self.assertGreaterEqual(pred_conf, 0.0)
        self.assertLessEqual(pred_conf, 1.0)

        # Process through filter
        res = self.filter.process_model(bundle, dummy_features)
        self.assertIsInstance(res, TemporalFilterResult)
        self.assertEqual(res.raw_gesture if hasattr(res, 'raw_gesture') else res.candidate_gesture, pred_label)


# =============================================================================
# Phase 3.1: Minimum Hold Duration Tests
# =============================================================================
class TestHoldDuration(unittest.TestCase):
    """Test suite for Phase 3.1 Minimum Hold Duration requirement."""

    def test_default_min_hold_duration_is_half_second(self):
        """Verify that min_hold_duration defaults to 0.5 seconds."""
        tf = TemporalFilter()
        self.assertEqual(tf.min_hold_duration, 0.5)

    def test_stable_gesture_held_less_than_half_second_not_confirmed(self):
        """
        Verify that 5 consecutive frames occurring within < 0.5s (e.g. 0.25s)
        do NOT confirm, returning reason='INSUFFICIENT_HOLD_TIME'.
        """
        tf = TemporalFilter(confidence_threshold=0.80, stable_frames=5, min_hold_duration=0.5)

        # 5 frames across 0.20s (e.g. 25 FPS)
        timestamps = [0.0, 0.05, 0.10, 0.15, 0.20]
        res = None
        for t in timestamps:
            res = tf.process("Fist", 0.95, timestamp=t)

        # 5 frames accumulated, but hold duration is only 0.20s < 0.50s
        self.assertEqual(res.consecutive_frames, 5)
        self.assertAlmostEqual(res.hold_duration, 0.20, places=2)
        self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
        self.assertFalse(res.is_confirmed)
        self.assertEqual(res.reason, "INSUFFICIENT_HOLD_TIME")

    def test_stable_gesture_held_greater_or_equal_half_second_confirms(self):
        """
        Verify that once BOTH >= 5 frames AND >= 0.5s continuous hold duration
        are satisfied, the gesture is CONFIRMED.
        """
        tf = TemporalFilter(confidence_threshold=0.80, stable_frames=5, min_hold_duration=0.5)

        # Frame 1 to 5 at 0.1s intervals (t = 0.0, 0.1, 0.2, 0.3, 0.4s)
        for i in range(5):
            res = tf.process("Fist", 0.95, timestamp=i * 0.10)
            self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)

        # Frame 6 at t = 0.50s (hold duration = 0.50s >= 0.50s)
        res = tf.process("Fist", 0.95, timestamp=0.50)
        self.assertEqual(res.status, ConfirmationStatus.CONFIRMED)
        self.assertTrue(res.is_confirmed)
        self.assertEqual(res.gesture, "Fist")
        self.assertAlmostEqual(res.hold_duration, 0.50, places=2)
        self.assertEqual(res.reason, "CONFIRMED")

    def test_changing_gesture_resets_hold_duration(self):
        """
        Verify that switching gestures resets the hold duration counter
        for the new candidate.
        """
        tf = TemporalFilter(confidence_threshold=0.80, stable_frames=3, min_hold_duration=0.5)

        # Hold Fist for 0.4s
        tf.process("Fist", 0.95, timestamp=0.0)
        tf.process("Fist", 0.95, timestamp=0.2)
        res = tf.process("Fist", 0.95, timestamp=0.4)
        self.assertAlmostEqual(res.hold_duration, 0.40, places=2)

        # Switch to OpenPalm at t = 0.45s (starts new hold timer)
        res = tf.process("OpenPalm", 0.95, timestamp=0.45)
        self.assertEqual(res.candidate_gesture, "OpenPalm")
        self.assertEqual(res.consecutive_frames, 1)
        self.assertAlmostEqual(res.hold_duration, 0.0, places=2)
        self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)

        # Hold OpenPalm until t = 0.85s (elapsed for OpenPalm = 0.40s < 0.5s)
        tf.process("OpenPalm", 0.95, timestamp=0.65)
        res = tf.process("OpenPalm", 0.95, timestamp=0.85)
        self.assertAlmostEqual(res.hold_duration, 0.40, places=2)
        self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
        self.assertEqual(res.reason, "INSUFFICIENT_HOLD_TIME")

        # OpenPalm reaches 0.50s at t = 0.95s
        res = tf.process("OpenPalm", 0.95, timestamp=0.95)
        self.assertEqual(res.status, ConfirmationStatus.CONFIRMED)
        self.assertEqual(res.gesture, "OpenPalm")

    def test_low_confidence_resets_hold_duration(self):
        """
        Verify that a low-confidence frame resets the hold duration timer to 0.0s.
        """
        tf = TemporalFilter(confidence_threshold=0.80, stable_frames=3, min_hold_duration=0.5)

        # Hold Fist for 0.4s
        tf.process("Fist", 0.95, timestamp=0.0)
        tf.process("Fist", 0.95, timestamp=0.2)
        res = tf.process("Fist", 0.95, timestamp=0.4)
        self.assertAlmostEqual(res.hold_duration, 0.40, places=2)

        # Dip to 0.70 confidence at t = 0.42s
        res = tf.process("Fist", 0.70, timestamp=0.42)
        self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
        self.assertEqual(res.reason, "LOW_CONFIDENCE")
        self.assertEqual(res.hold_duration, 0.0)

        # High confidence resumes at t = 0.45s -> must start fresh
        res = tf.process("Fist", 0.95, timestamp=0.45)
        self.assertEqual(res.consecutive_frames, 1)
        self.assertEqual(res.hold_duration, 0.0)

    def test_none_input_resets_hold_duration(self):
        """
        Verify that None input (hand lost) resets hold duration to 0.0s.
        """
        tf = TemporalFilter(confidence_threshold=0.80, stable_frames=3, min_hold_duration=0.5)

        tf.process("Fist", 0.95, timestamp=0.0)
        tf.process("Fist", 0.95, timestamp=0.3)

        # Hand lost
        res = tf.process(None, 0.0, timestamp=0.35)
        self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
        self.assertEqual(res.reason, "NO_GESTURE")
        self.assertEqual(res.hold_duration, 0.0)

    def test_reset_clears_hold_timing(self):
        """
        Verify that manual reset() clears all hold timing internal variables.
        """
        tf = TemporalFilter(min_hold_duration=0.5)
        tf.process("Fist", 0.95, timestamp=0.0)
        tf.process("Fist", 0.95, timestamp=0.3)

        tf.reset()
        self.assertIsNone(tf._hold_start_time)
        self.assertEqual(tf._current_hold_duration, 0.0)

    def test_cooldown_still_works_with_hold_duration(self):
        """
        Verify that cooldown blocks immediate re-confirmation even if gesture is held.
        """
        tf = TemporalFilter(confidence_threshold=0.80, stable_frames=2, cooldown_frames=5, min_hold_duration=0.2)

        tf.process("Fist", 0.95, timestamp=0.0)
        res = tf.process("Fist", 0.95, timestamp=0.25)
        self.assertEqual(res.status, ConfirmationStatus.CONFIRMED)

        # Next frame is within cooldown
        res = tf.process("Fist", 0.95, timestamp=0.30)
        self.assertEqual(res.status, ConfirmationStatus.NOT_CONFIRMED)
        self.assertEqual(res.reason, "IN_COOLDOWN")
        self.assertEqual(res.hold_duration, 0.0)

    def test_real_time_monotonic_clock(self):
        """
        Verify that omitting timestamp automatically uses time.monotonic().
        """
        import time

        tf = TemporalFilter(confidence_threshold=0.80, stable_frames=2, min_hold_duration=0.05)
        tf.process("Fist", 0.95)
        time.sleep(0.06)  # Sleep slightly more than min_hold_duration (50ms)
        res = tf.process("Fist", 0.95)
        self.assertEqual(res.status, ConfirmationStatus.CONFIRMED)
        self.assertGreaterEqual(res.hold_duration, 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
