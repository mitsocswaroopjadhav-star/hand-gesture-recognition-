"""
test_personalized_gestures.py
==============================

Unit test suite for IntentSense Phase 11: Personalized Gestures Architecture.

Covers:
1. Exact mathematical confidence boundary calculations.
2. Hard boundary (d <= 0.30) vs entry requirement (d <= 0.233 / C >= 0.80).
3. Ambiguity gating between competing custom gestures (delta_d < 0.10).
4. Multi-take enrollment with holdout cross-validation.
5. Built-in conflict and custom overlap rejection.
6. 4-Tier Arbitration Hierarchy (Built-in Priority, Built-in Guard, Custom, No Hand).
7. Profile persistence, renaming, remapping, and deletion.
8. Temporal filter integration (streak accumulation at C >= 0.80 vs streak rejection at C = 0.70).
"""

import os
import shutil
import sys
import tempfile
import unittest
from typing import List

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.context_manager import Context
from core.intent_engine import Intent
from core.personalized_gestures import (
    BUILTIN_GUARD_THRESHOLD,
    CONFIDENCE_BOUNDARY,
    DELTA_AMBIGUITY,
    TAU_ACCEPT,
    TAU_ENTRY_THRESHOLD,
    TAU_IDEAL,
    TEMPORAL_ENTRY_CONFIDENCE,
    GestureProfile,
    PersonalizedGestureManager,
    calculate_custom_confidence,
    validate_enrollment_samples,
)
from core.temporal_filter import TemporalFilter


class TestPersonalizedGestures(unittest.TestCase):
    """Test suite for core/personalized_gestures.py."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.profile_path = os.path.join(self.temp_dir, "custom_gestures.json")
        self.manager = PersonalizedGestureManager(profile_path=self.profile_path)

        # Baseline 42D sample centroid (e.g. horns / RockOn gesture)
        self.rock_on_centroid = [0.0] * 42
        self.rock_on_centroid[16] = 1.2   # Index tip
        self.rock_on_centroid[40] = 1.1   # Pinky tip

        # Secondary 42D sample centroid (e.g. three fingers)
        self.three_centroid = [0.0] * 42
        self.three_centroid[16] = 1.0     # Index tip
        self.three_centroid[20] = 1.1     # Middle tip
        self.three_centroid[24] = 1.0     # Ring tip

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    # -------------------------------------------------------------------------
    # 1. Mathematical Confidence Boundary Tests
    # -------------------------------------------------------------------------
    def test_confidence_at_ideal_distance(self):
        """d <= TAU_IDEAL (0.10) must yield exactly 1.0."""
        self.assertEqual(calculate_custom_confidence(0.00), 1.0)
        self.assertEqual(calculate_custom_confidence(0.05), 1.0)
        self.assertEqual(calculate_custom_confidence(TAU_IDEAL), 1.0)

    def test_confidence_intermediate_linear_decay(self):
        """d = 0.15 is halfway between 0.10 and 0.20 -> 0.925."""
        c_15 = calculate_custom_confidence(0.15)
        # 1.0 - (0.05 / 0.20) * 0.30 = 1.0 - 0.075 = 0.925
        self.assertAlmostEqual(c_15, 0.925, places=3)

    def test_confidence_at_temporal_entry_boundary(self):
        """At TAU_ENTRY_THRESHOLD (~0.2333), confidence must equal TEMPORAL_ENTRY_CONFIDENCE (0.80)."""
        c_entry = calculate_custom_confidence(TAU_ENTRY_THRESHOLD)
        self.assertAlmostEqual(c_entry, 0.80, places=3)

    def test_confidence_at_hard_boundary(self):
        """At d = TAU_ACCEPT (0.30), confidence is exactly 0.70."""
        c_accept = calculate_custom_confidence(TAU_ACCEPT)
        self.assertAlmostEqual(c_accept, 0.70, places=3)

    def test_confidence_beyond_boundary(self):
        """Beyond TAU_ACCEPT (d > 0.30), confidence drops to 0.0."""
        self.assertEqual(calculate_custom_confidence(0.301), 0.0)
        self.assertEqual(calculate_custom_confidence(0.50), 0.0)

    # -------------------------------------------------------------------------
    # 2. Hard Boundary vs Entry Requirement Distinction
    # -------------------------------------------------------------------------
    def test_boundary_match_rejected_by_temporal_filter(self):
        """
        A sample at d = 0.30 is within the hard matching radius (d <= 0.30),
        but its confidence (0.70) falls below entry threshold (0.80).
        It must NOT start or sustain a temporal filter confirmation streak.
        """
        self.manager.register_gesture(
            name="RockOn",
            centroid=self.rock_on_centroid,
            dispersion=0.05,
            sample_count=20,
        )

        # Create a sample vector exactly 0.30 distance away
        offset_sample = list(self.rock_on_centroid)
        offset_sample[0] = 0.30  # norm = 0.30

        # Arbitrate directly
        arb = self.manager.arbitrate(
            builtin_gesture="Pointing",
            builtin_confidence=0.20,   # Low built-in confidence
            builtin_threshold=0.80,
            features=offset_sample,
            hand_present=True,
        )
        self.assertIsNone(arb.candidate_gesture)
        self.assertEqual(arb.reason, "CUSTOM_WEAK_MATCH")
        self.assertAlmostEqual(arb.confidence, 0.70, places=3)

        # Feed 10 consecutive frames into a TemporalFilter (requires 0.80 confidence)
        t_filter = TemporalFilter(confidence_threshold=0.80, stable_frames=5)
        for i in range(10):
            res = t_filter.process("RockOn", arb.confidence, timestamp=1.0 + i * 0.033)
            self.assertFalse(res.is_confirmed)
            self.assertEqual(res.consecutive_frames, 0)
            self.assertEqual(res.reason, "LOW_CONFIDENCE")

    def test_entry_match_accumulates_temporal_streak(self):
        """A sample at d = 0.15 (C = 0.925 >= 0.80) correctly accumulates streak and confirms."""
        self.manager.register_gesture(
            name="RockOn",
            centroid=self.rock_on_centroid,
            dispersion=0.05,
            sample_count=20,
        )
        sample = list(self.rock_on_centroid)
        sample[0] = 0.15

        arb = self.manager.arbitrate(
            builtin_gesture=None,
            builtin_confidence=0.10,
            builtin_threshold=0.80,
            features=sample,
            hand_present=True,
        )
        self.assertEqual(arb.candidate_gesture, "RockOn")
        self.assertEqual(arb.reason, "CUSTOM_CANDIDATE")
        self.assertAlmostEqual(arb.confidence, 0.925, places=3)

        # Feed into TemporalFilter
        t_filter = TemporalFilter(confidence_threshold=0.80, stable_frames=5, min_hold_duration=0.1)
        for i in range(4):
            res = t_filter.process(arb.candidate_gesture, arb.confidence, timestamp=1.0 + i * 0.033)
            self.assertFalse(res.is_confirmed)
            self.assertEqual(res.consecutive_frames, i + 1)

        # Frame 5: Confirmation achieved!
        res_5 = t_filter.process(arb.candidate_gesture, arb.confidence, timestamp=1.0 + 4 * 0.033)
        self.assertTrue(res_5.is_confirmed)
        self.assertEqual(res_5.gesture, "RockOn")

    # -------------------------------------------------------------------------
    # 3. Multi-Custom Ambiguity Rejection
    # -------------------------------------------------------------------------
    def test_ambiguity_rejection_when_centroids_too_close(self):
        """When two custom gestures both match with delta_d < 0.10, candidate is rejected."""
        self.manager.register_gesture("GestureA", self.rock_on_centroid, 0.05, 20)

        # Gesture B centroid slightly offset by 0.06 palm lengths
        b_centroid = list(self.rock_on_centroid)
        b_centroid[1] = 0.06
        self.manager.register_gesture("GestureB", b_centroid, 0.05, 20)

        # Query vector at midpoint
        mid_vector = list(self.rock_on_centroid)
        mid_vector[1] = 0.03

        match_res = self.manager.match(mid_vector)
        self.assertFalse(match_res.is_match)
        self.assertIsNone(match_res.gesture)
        self.assertEqual(match_res.reason, "AMBIGUOUS_CUSTOM_MATCH")

        # In arbitration:
        arb = self.manager.arbitrate(
            builtin_gesture=None,
            builtin_confidence=0.10,
            builtin_threshold=0.80,
            features=mid_vector,
            hand_present=True,
        )
        self.assertIsNone(arb.candidate_gesture)
        self.assertEqual(arb.reason, "AMBIGUOUS_CUSTOM_MATCH")

    # -------------------------------------------------------------------------
    # 4. 4-Tier Arbitration Hierarchy
    # -------------------------------------------------------------------------
    def test_tier_1_decisive_builtin_wins_unconditionally(self):
        """When built-in confidence >= 0.80, built-in gesture wins; custom matcher bypassed."""
        self.manager.register_gesture("RockOn", self.rock_on_centroid, 0.05, 20)

        arb = self.manager.arbitrate(
            builtin_gesture="Pointing",
            builtin_confidence=0.88,
            builtin_threshold=0.80,
            features=self.rock_on_centroid,  # Exact custom match!
            hand_present=True,
        )
        self.assertEqual(arb.tier, 1)
        self.assertEqual(arb.candidate_gesture, "Pointing")
        self.assertEqual(arb.source, "BUILTIN")
        self.assertEqual(arb.reason, "BUILTIN_CONFIRMED")

    def test_tier_2_builtin_guard_inhibits_custom_gestures(self):
        """When built-in confidence is in [0.50, 0.80), custom gestures are strictly inhibited."""
        self.manager.register_gesture("RockOn", self.rock_on_centroid, 0.05, 20)

        arb = self.manager.arbitrate(
            builtin_gesture="Peace",
            builtin_confidence=0.65,
            builtin_threshold=0.80,
            features=self.rock_on_centroid,  # Exact custom match!
            hand_present=True,
        )
        self.assertEqual(arb.tier, 2)
        self.assertIsNone(arb.candidate_gesture)
        self.assertEqual(arb.source, "BUILTIN")
        self.assertEqual(arb.reason, "BUILTIN_UNCERTAIN_GUARD")

    def test_tier_3_low_builtin_conf_evaluates_custom(self):
        """When built-in confidence < 0.50, custom matcher evaluates and succeeds."""
        self.manager.register_gesture("RockOn", self.rock_on_centroid, 0.05, 20)

        arb = self.manager.arbitrate(
            builtin_gesture="Fist",
            builtin_confidence=0.25,
            builtin_threshold=0.80,
            features=self.rock_on_centroid,
            hand_present=True,
        )
        self.assertEqual(arb.tier, 3)
        self.assertEqual(arb.candidate_gesture, "RockOn")
        self.assertEqual(arb.source, "CUSTOM")
        self.assertEqual(arb.reason, "CUSTOM_CANDIDATE")

    def test_tier_4_no_hand_detected(self):
        """When hand is missing, arbitration emits Tier 4 NO_HAND."""
        arb = self.manager.arbitrate(
            builtin_gesture=None,
            builtin_confidence=0.0,
            builtin_threshold=0.80,
            features=None,
            hand_present=False,
        )
        self.assertEqual(arb.tier, 4)
        self.assertIsNone(arb.candidate_gesture)
        self.assertEqual(arb.reason, "NO_HAND")

    # -------------------------------------------------------------------------
    # 5. Multi-Take Enrollment Validation
    # -------------------------------------------------------------------------
    def test_valid_3_take_enrollment(self):
        """Steady 3-take enrollment successfully calculates centroid and passes validation."""
        rng = np.random.RandomState(42)
        base = np.array(self.rock_on_centroid)

        take1 = [base + rng.normal(0, 0.015, 42) for _ in range(10)]
        take2 = [base + rng.normal(0, 0.015, 42) for _ in range(10)]
        take3 = [base + rng.normal(0, 0.015, 42) for _ in range(10)]

        profile = self.manager.enroll_gesture(
            name="RockOn",
            take1=take1,
            take2=take2,
            take3_val=take3,
            mappings={"MEDIA": "PLAY_PAUSE", "PRESENTATION": "START_STOP"},
        )
        self.assertEqual(profile.name, "RockOn")
        self.assertEqual(profile.sample_count, 30)
        self.assertLessEqual(profile.dispersion, 0.12)
        self.assertEqual(profile.mappings["MEDIA"], "PLAY_PAUSE")

    def test_unstable_training_samples_rejected(self):
        """Excessive dispersion (> 0.12) across Take 1 & 2 raises ENROLLMENT_UNSTABLE."""
        rng = np.random.RandomState(42)
        base = np.array(self.rock_on_centroid)

        take1 = [base + rng.normal(0, 0.15, 42) for _ in range(10)]
        take2 = [base + rng.normal(0, 0.15, 42) for _ in range(10)]
        take3 = [base for _ in range(10)]

        with self.assertRaises(RuntimeError) as ctx:
            self.manager.enroll_gesture("Shaky", take1, take2, take3)
        self.assertIn("ENROLLMENT_UNSTABLE", str(ctx.exception))

    def test_validation_take_failure_rejected(self):
        """Inconsistent validation burst (Take 3 differs from Takes 1 & 2) fails validation."""
        base1 = np.array(self.rock_on_centroid)
        base_different = np.array(self.three_centroid)

        take1 = [base1 for _ in range(10)]
        take2 = [base1 for _ in range(10)]
        take3 = [base_different for _ in range(10)]  # User switched pose in Take 3!

        with self.assertRaises(RuntimeError) as ctx:
            self.manager.enroll_gesture("Inconsistent", take1, take2, take3)
        self.assertIn("ENROLLMENT_VALIDATION_FAILED", str(ctx.exception))

    def test_conflict_with_builtin_rejected(self):
        """Mock classifier predicting high confidence triggers CONFLICT_WITH_BUILTIN."""
        class MockClassifier:
            classes_ = ["Fist", "Pointing"]
            def predict_proba(self, X):
                # Always returns 0.85 for Pointing
                return np.array([[0.15, 0.85] for _ in range(len(X))])

        base = np.array(self.rock_on_centroid)
        take1 = [base for _ in range(10)]
        take2 = [base for _ in range(10)]
        take3 = [base for _ in range(10)]

        with self.assertRaises(RuntimeError) as ctx:
            self.manager.enroll_gesture(
                "PointingLookalike", take1, take2, take3, builtin_classifier=MockClassifier()
            )
        self.assertIn("CONFLICT_WITH_BUILTIN", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 6. Profile Persistence & Isolation
    # -------------------------------------------------------------------------
    def test_profile_persistence_and_reload(self):
        """Saving profiles to JSON and reloading preserves centroids and mappings."""
        self.manager.register_gesture(
            name="RockOn",
            centroid=self.rock_on_centroid,
            dispersion=0.045,
            sample_count=30,
            mappings={"MEDIA": "PLAY_PAUSE", "CUSTOM": "VOLUME_UP"},
            description="Rock on hand horns",
        )

        # Create new manager pointing to same JSON file
        new_manager = PersonalizedGestureManager(profile_path=self.profile_path)
        p = new_manager.get_gesture("RockOn")
        self.assertIsNotNone(p)
        self.assertEqual(p.name, "RockOn")
        self.assertAlmostEqual(p.dispersion, 0.045)
        self.assertEqual(p.mappings["MEDIA"], "PLAY_PAUSE")
        self.assertEqual(p.mappings["CUSTOM"], "VOLUME_UP")
        self.assertEqual(p.description, "Rock on hand horns")

    def test_rename_remap_delete(self):
        """Test rename, remap, and delete operations."""
        self.manager.register_gesture("OldName", self.rock_on_centroid, 0.05, 20)
        self.assertTrue(self.manager.rename_gesture("OldName", "NewName"))
        self.assertIsNone(self.manager.get_gesture("OldName"))
        self.assertIsNotNone(self.manager.get_gesture("NewName"))

        # Remap context
        self.assertTrue(self.manager.remap_gesture("NewName", Context.PRESENTATION, Intent.NEXT_SLIDE))
        self.assertEqual(self.manager.get_gesture("NewName").mappings["PRESENTATION"], "NEXT_SLIDE")

        # Delete
        self.assertTrue(self.manager.delete_gesture("NewName"))
        self.assertIsNone(self.manager.get_gesture("NewName"))

    def test_reserved_name_rejected(self):
        """Cannot register or rename gesture using reserved built-in names."""
        for reserved in ("OpenPalm", "Fist", "Peace", "Pointing", "ThumbsUp", "ThumbsDown"):
            with self.assertRaises(ValueError):
                self.manager.register_gesture(reserved, self.rock_on_centroid, 0.05, 20)

    def test_builtin_mappings_remain_unaffected_by_custom_lifecycle(self):
        """Custom registration, rename, remap, and deletion must NEVER modify built-in mappings."""
        from core.intent_engine import DEFAULT_INTENT_MAPPINGS, IntentEngine

        engine = IntentEngine()
        # Snapshot built-in mappings before custom manipulation
        media_before = engine.get_mappings(Context.MEDIA)
        pres_before = engine.get_mappings(Context.PRESENTATION)

        # Built-in mappings must match defaults
        self.assertEqual(media_before, DEFAULT_INTENT_MAPPINGS[Context.MEDIA])
        self.assertEqual(pres_before, DEFAULT_INTENT_MAPPINGS[Context.PRESENTATION])

        # Register custom gesture
        engine.register_custom_gesture("RockOn", {Context.MEDIA: Intent.VOLUME_UP})
        self.assertEqual(engine.get_mappings(Context.MEDIA)["RockOn"], Intent.VOLUME_UP)
        # Verify original built-ins unchanged
        for builtin_g, builtin_intent in DEFAULT_INTENT_MAPPINGS[Context.MEDIA].items():
            self.assertEqual(engine.get_mappings(Context.MEDIA)[builtin_g], builtin_intent)

        # Rename custom gesture
        engine.rename_custom_gesture("RockOn", "Horns")
        self.assertNotIn("RockOn", engine.get_mappings(Context.MEDIA))
        self.assertEqual(engine.get_mappings(Context.MEDIA)["Horns"], Intent.VOLUME_UP)
        for builtin_g, builtin_intent in DEFAULT_INTENT_MAPPINGS[Context.MEDIA].items():
            self.assertEqual(engine.get_mappings(Context.MEDIA)[builtin_g], builtin_intent)

        # Remap custom gesture
        engine.set_mapping(Context.MEDIA, "Horns", Intent.PLAY_PAUSE)
        self.assertEqual(engine.get_mappings(Context.MEDIA)["Horns"], Intent.PLAY_PAUSE)
        for builtin_g, builtin_intent in DEFAULT_INTENT_MAPPINGS[Context.MEDIA].items():
            self.assertEqual(engine.get_mappings(Context.MEDIA)[builtin_g], builtin_intent)

        # Delete custom gesture
        engine.unregister_custom_gesture("Horns")
        media_after = engine.get_mappings(Context.MEDIA)
        pres_after = engine.get_mappings(Context.PRESENTATION)

        # Must exactly match original snapshots
        self.assertEqual(media_after, media_before)
        self.assertEqual(pres_after, pres_before)

    def test_duplicate_gesture_name_rejected_by_default(self):
        """Registering or enrolling with an existing name raises ValueError unless allow_overwrite=True."""
        self.manager.register_gesture("RockOn", self.rock_on_centroid, 0.05, 20)
        with self.assertRaises(ValueError) as ctx:
            self.manager.register_gesture("RockOn", self.rock_on_centroid, 0.05, 20)
        self.assertIn("already exists", str(ctx.exception))

        # With allow_overwrite=True, it succeeds
        p = self.manager.register_gesture("RockOn", self.rock_on_centroid, 0.04, 20, allow_overwrite=True)
        self.assertEqual(p.name, "RockOn")
        self.assertAlmostEqual(p.dispersion, 0.04)

    def test_supported_action_map_resolves_all_intents(self):
        """Every mapped action in SUPPORTED_ACTION_MAP must resolve to a valid Intent enum."""
        from core.personalized_gestures import SUPPORTED_ACTION_MAP
        for action_name, intent in SUPPORTED_ACTION_MAP.items():
            self.assertIsInstance(intent, Intent)
            self.assertTrue(len(action_name) > 0)

    def test_delete_gesture_removes_from_disk_and_memory(self):
        """Deleting a gesture removes it from profiles and persists the change to JSON."""
        self.manager.register_gesture("TempGesture", self.rock_on_centroid, 0.05, 20)
        self.assertIn("TempGesture", self.manager.profiles)

        success = self.manager.delete_gesture("TempGesture")
        self.assertTrue(success)
        self.assertNotIn("TempGesture", self.manager.profiles)

        # Reload from disk
        new_manager = PersonalizedGestureManager(profile_path=self.profile_path)
        self.assertNotIn("TempGesture", new_manager.profiles)

    def test_format_enrollment_error_messages(self):
        """Friendly error explanations must be generated for all failure modes."""
        from core.personalized_gestures import format_enrollment_error

        msg1 = format_enrollment_error("TOO_FEW_SAMPLES")
        self.assertIn("Not enough stable hand frames", msg1)

        msg2 = format_enrollment_error("OUTLIERS_DETECTED")
        self.assertIn("Inconsistent hand position", msg2)

        msg3 = format_enrollment_error("HIGH_DISPERSION")
        self.assertIn("Hand moved too much", msg3)

        msg4 = format_enrollment_error("CONFLICTS_WITH_BUILTIN:Pointing")
        self.assertIn("Pointing", msg4)
        self.assertIn("too similar", msg4)

        msg5 = format_enrollment_error("CONFLICTS_WITH_CUSTOM:Horns")
        self.assertIn("Horns", msg5)
        self.assertIn("too similar", msg5)

        msg6 = format_enrollment_error("UNKNOWN_CODE")
        self.assertIn("Enrollment could not be validated", msg6)

    def test_atomic_persistence_and_action_remapping(self):
        """Verify atomic saving and remapping actions."""
        # 1. Register gesture with initial action mapping
        self.manager.register_gesture(
            "RockOn",
            self.rock_on_centroid,
            0.05,
            20,
            mappings={"MEDIA": "VOLUME_UP"},
            description="Rock on custom gesture"
        )
        self.assertEqual(self.manager.profiles["RockOn"].mappings.get("MEDIA"), "VOLUME_UP")

        # 2. Remap action
        remap_res = self.manager.remap_gesture("RockOn", Context.MEDIA, Intent.NEXT)
        self.assertTrue(remap_res)
        self.assertEqual(self.manager.profiles["RockOn"].mappings.get("MEDIA"), "NEXT")

        # 3. Reload from disk to verify atomic JSON file write
        new_manager = PersonalizedGestureManager(profile_path=self.profile_path)
        self.assertIn("RockOn", new_manager.profiles)
        self.assertEqual(new_manager.profiles["RockOn"].mappings.get("MEDIA"), "NEXT")


if __name__ == "__main__":
    unittest.main()

