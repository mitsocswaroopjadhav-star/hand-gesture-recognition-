"""
test_intent_engine.py
=====================

Unit test suite for IntentSense Phase 5: IntentEngine.

Covers:
 7. MEDIA ThumbsUp -> VOLUME_UP
 8. MEDIA ThumbsDown -> VOLUME_DOWN
 9. MEDIA Pointing -> NEXT
10. PRESENTATION Pointing -> NEXT_SLIDE
11. PRESENTATION ThumbsUp -> ZOOM_IN
12. Locked system -> rejected (SYSTEM_LOCKED)
13. Unconfirmed gesture -> rejected (GESTURE_NOT_CONFIRMED)
14. Unknown gesture -> rejected (UNKNOWN_GESTURE)
15. Unmapped gesture -> rejected (NO_MAPPING)
16. Valid result contains correct gesture/context/intent/confidence
17. Rejection result contains explicit reason
Additional tests:
- Full MEDIA mappings (Fist -> PLAY_PAUSE, Peace -> PREVIOUS)
- Full PRESENTATION mappings (Fist -> START_STOP, Peace -> PREVIOUS_SLIDE, ThumbsDown -> ZOOM_OUT)
- OpenPalm in MEDIA & PRESENTATION returns NO_MAPPING (activation gesture decoupled)
- Integration with Phase 4 ActivationResult
- CUSTOM context mapping and extensibility
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.activation_state import ActivationResult, SystemState
from core.context_manager import Context, ContextManager
from core.intent_engine import (
    DEFAULT_INTENT_MAPPINGS,
    DEFAULT_SWIPE_MAPPINGS,
    KNOWN_GESTURES,
    Intent,
    IntentEngine,
    IntentReason,
    IntentResult,
)
from core.swipe_detector import SwipeDirection, SwipeResult


class TestIntentEngine(unittest.TestCase):
    """Test suite for core/intent_engine.py."""

    def setUp(self):
        """Initializes ContextManager and IntentEngine with defaults."""
        self.context_manager = ContextManager()
        self.engine = IntentEngine(context_manager=self.context_manager)

    # -------------------------------------------------------------------------
    # 7. MEDIA ThumbsUp -> VOLUME_UP
    # -------------------------------------------------------------------------
    def test_media_thumbs_up_maps_to_volume_up(self):
        """Verify ThumbsUp in MEDIA context produces VOLUME_UP."""
        res = self.engine.process(
            gesture="ThumbsUp",
            is_confirmed=True,
            is_active=True,
            confidence=0.96,
            context=Context.MEDIA,
        )
        self.assertTrue(res.accepted)
        self.assertEqual(res.intent, Intent.VOLUME_UP)
        self.assertEqual(res.context, Context.MEDIA)
        self.assertEqual(res.gesture, "ThumbsUp")
        self.assertEqual(res.reason, IntentReason.CONFIRMED)

    # -------------------------------------------------------------------------
    # 8. MEDIA ThumbsDown -> VOLUME_DOWN
    # -------------------------------------------------------------------------
    def test_media_thumbs_down_maps_to_volume_down(self):
        """Verify ThumbsDown in MEDIA context produces VOLUME_DOWN."""
        res = self.engine.process(
            gesture="ThumbsDown",
            is_confirmed=True,
            is_active=True,
            confidence=0.88,
            context=Context.MEDIA,
        )
        self.assertTrue(res.accepted)
        self.assertEqual(res.intent, Intent.VOLUME_DOWN)
        self.assertEqual(res.context, Context.MEDIA)
        self.assertEqual(res.reason, IntentReason.CONFIRMED)

    # -------------------------------------------------------------------------
    # 9. MEDIA Pointing -> NEXT
    # -------------------------------------------------------------------------
    def test_media_pointing_maps_to_next(self):
        """Verify Pointing in MEDIA context produces NEXT."""
        res = self.engine.process(
            gesture="Pointing",
            is_confirmed=True,
            is_active=True,
            confidence=0.92,
            context=Context.MEDIA,
        )
        self.assertTrue(res.accepted)
        self.assertEqual(res.intent, Intent.NEXT)
        self.assertEqual(res.context, Context.MEDIA)
        self.assertEqual(res.reason, IntentReason.CONFIRMED)

    # -------------------------------------------------------------------------
    # Additional MEDIA mappings: Fist -> PLAY_PAUSE, Peace -> PREVIOUS
    # -------------------------------------------------------------------------
    def test_media_remaining_mappings(self):
        """Verify Fist and Peace in MEDIA context."""
        res_fist = self.engine.process("Fist", is_confirmed=True, is_active=True, context=Context.MEDIA)
        self.assertTrue(res_fist.accepted)
        self.assertEqual(res_fist.intent, Intent.PLAY_PAUSE)

        res_peace = self.engine.process("Peace", is_confirmed=True, is_active=True, context=Context.MEDIA)
        self.assertTrue(res_peace.accepted)
        self.assertEqual(res_peace.intent, Intent.PREVIOUS)

    # -------------------------------------------------------------------------
    # 10. PRESENTATION Pointing -> NEXT_SLIDE
    # -------------------------------------------------------------------------
    def test_presentation_pointing_maps_to_next_slide(self):
        """Verify Pointing in PRESENTATION context produces NEXT_SLIDE."""
        self.context_manager.set_context(Context.PRESENTATION)
        res = self.engine.process(
            gesture="Pointing",
            is_confirmed=True,
            is_active=True,
            confidence=0.95,
        )
        self.assertTrue(res.accepted)
        self.assertEqual(res.intent, Intent.NEXT_SLIDE)
        self.assertEqual(res.context, Context.PRESENTATION)
        self.assertEqual(res.reason, IntentReason.CONFIRMED)

    # -------------------------------------------------------------------------
    # 11. PRESENTATION ThumbsUp -> ZOOM_IN
    # -------------------------------------------------------------------------
    def test_presentation_thumbs_up_maps_to_zoom_in(self):
        """Verify ThumbsUp in PRESENTATION context produces ZOOM_IN."""
        self.context_manager.set_context(Context.PRESENTATION)
        res = self.engine.process(
            gesture="ThumbsUp",
            is_confirmed=True,
            is_active=True,
            confidence=0.94,
        )
        self.assertTrue(res.accepted)
        self.assertEqual(res.intent, Intent.ZOOM_IN)
        self.assertEqual(res.context, Context.PRESENTATION)
        self.assertEqual(res.reason, IntentReason.CONFIRMED)

    # -------------------------------------------------------------------------
    # Additional PRESENTATION mappings: Fist, Peace, ThumbsDown
    # -------------------------------------------------------------------------
    def test_presentation_remaining_mappings(self):
        """Verify Fist, Peace, and ThumbsDown in PRESENTATION context."""
        self.context_manager.set_context(Context.PRESENTATION)

        res_fist = self.engine.process("Fist", is_confirmed=True, is_active=True)
        self.assertTrue(res_fist.accepted)
        self.assertEqual(res_fist.intent, Intent.START_STOP)

        res_peace = self.engine.process("Peace", is_confirmed=True, is_active=True)
        self.assertTrue(res_peace.accepted)
        self.assertEqual(res_peace.intent, Intent.PREVIOUS_SLIDE)

        res_td = self.engine.process("ThumbsDown", is_confirmed=True, is_active=True)
        self.assertTrue(res_td.accepted)
        self.assertEqual(res_td.intent, Intent.ZOOM_OUT)

    # -------------------------------------------------------------------------
    # 12. Locked system -> rejected (SYSTEM_LOCKED)
    # -------------------------------------------------------------------------
    def test_locked_system_is_rejected(self):
        """Verify that any gesture is rejected with SYSTEM_LOCKED when system is locked."""
        # Using boolean flag
        res1 = self.engine.process(
            gesture="ThumbsUp",
            is_confirmed=True,
            is_active=False,
            confidence=0.99,
        )
        self.assertFalse(res1.accepted)
        self.assertIsNone(res1.intent)
        self.assertEqual(res1.reason, IntentReason.SYSTEM_LOCKED)

        # Using SystemState enum
        res2 = self.engine.process(
            gesture="ThumbsUp",
            is_confirmed=True,
            state=SystemState.LOCKED,
            confidence=0.99,
        )
        self.assertFalse(res2.accepted)
        self.assertIsNone(res2.intent)
        self.assertEqual(res2.reason, IntentReason.SYSTEM_LOCKED)

    # -------------------------------------------------------------------------
    # 13. Unconfirmed gesture -> rejected (GESTURE_NOT_CONFIRMED)
    # -------------------------------------------------------------------------
    def test_unconfirmed_gesture_is_rejected(self):
        """Verify that an unconfirmed gesture while ACTIVE is rejected with GESTURE_NOT_CONFIRMED."""
        res = self.engine.process(
            gesture="ThumbsUp",
            is_confirmed=False,
            is_active=True,
            confidence=0.85,
        )
        self.assertFalse(res.accepted)
        self.assertIsNone(res.intent)
        self.assertEqual(res.reason, IntentReason.GESTURE_NOT_CONFIRMED)

    # -------------------------------------------------------------------------
    # 14. Unknown gesture -> rejected (UNKNOWN_GESTURE)
    # -------------------------------------------------------------------------
    def test_unknown_gesture_is_rejected(self):
        """Verify that unrecognized or None gestures are rejected with UNKNOWN_GESTURE."""
        invalid_gestures = ["RandomWave", "Spock", "SpiderMan", "", None]
        for g in invalid_gestures:
            res = self.engine.process(
                gesture=g,
                is_confirmed=True,
                is_active=True,
                confidence=0.95,
            )
            self.assertFalse(res.accepted)
            self.assertIsNone(res.intent)
            self.assertEqual(res.reason, IntentReason.UNKNOWN_GESTURE)

    # -------------------------------------------------------------------------
    # 15. Unmapped gesture -> rejected (NO_MAPPING)
    # -------------------------------------------------------------------------
    def test_unmapped_gesture_is_rejected(self):
        """
        Verify that a known gesture with no mapping in the current context
        is rejected with NO_MAPPING.
        """
        # OpenPalm is a known gesture, but intentionally unmapped in MEDIA
        res_openpalm_media = self.engine.process(
            gesture="OpenPalm",
            is_confirmed=True,
            is_active=True,
            context=Context.MEDIA,
        )
        self.assertFalse(res_openpalm_media.accepted)
        self.assertIsNone(res_openpalm_media.intent)
        self.assertEqual(res_openpalm_media.reason, IntentReason.NO_MAPPING)

        # OpenPalm in PRESENTATION is also unmapped
        res_openpalm_pres = self.engine.process(
            gesture="OpenPalm",
            is_confirmed=True,
            is_active=True,
            context=Context.PRESENTATION,
        )
        self.assertFalse(res_openpalm_pres.accepted)
        self.assertIsNone(res_openpalm_pres.intent)
        self.assertEqual(res_openpalm_pres.reason, IntentReason.NO_MAPPING)

        # In CUSTOM context (initially empty), all known gestures produce NO_MAPPING
        for g in KNOWN_GESTURES:
            res_custom = self.engine.process(
                gesture=g,
                is_confirmed=True,
                is_active=True,
                context=Context.CUSTOM,
            )
            self.assertFalse(res_custom.accepted)
            self.assertIsNone(res_custom.intent)
            self.assertEqual(res_custom.reason, IntentReason.NO_MAPPING)

    # -------------------------------------------------------------------------
    # 16. Valid result contains correct gesture/context/intent/confidence
    # -------------------------------------------------------------------------
    def test_valid_result_structure(self):
        """Verify that a successful IntentResult populates all fields correctly."""
        res = self.engine.process(
            gesture="ThumbsUp",
            is_confirmed=True,
            is_active=True,
            confidence=0.975,
            context=Context.MEDIA,
        )
        self.assertIsInstance(res, IntentResult)
        self.assertEqual(res.gesture, "ThumbsUp")
        self.assertEqual(res.intent, Intent.VOLUME_UP)
        self.assertEqual(res.context, Context.MEDIA)
        self.assertAlmostEqual(res.confidence, 0.975, places=3)
        self.assertTrue(res.accepted)
        self.assertEqual(res.reason, IntentReason.CONFIRMED)

    # -------------------------------------------------------------------------
    # 17. Rejection result contains explicit reason
    # -------------------------------------------------------------------------
    def test_rejection_result_structure(self):
        """Verify that rejection results maintain proper metadata and explicit reasons."""
        # Test locked rejection
        r_locked = self.engine.process("Pointing", is_confirmed=True, is_active=False)
        self.assertFalse(r_locked.accepted)
        self.assertIsNone(r_locked.intent)
        self.assertEqual(r_locked.reason, IntentReason.SYSTEM_LOCKED)

        # Test unconfirmed rejection
        r_unconfirmed = self.engine.process("Pointing", is_confirmed=False, is_active=True)
        self.assertFalse(r_unconfirmed.accepted)
        self.assertIsNone(r_unconfirmed.intent)
        self.assertEqual(r_unconfirmed.reason, IntentReason.GESTURE_NOT_CONFIRMED)

        # Test unknown gesture rejection
        r_unknown = self.engine.process("AlienSalute", is_confirmed=True, is_active=True)
        self.assertFalse(r_unknown.accepted)
        self.assertIsNone(r_unknown.intent)
        self.assertEqual(r_unknown.reason, IntentReason.UNKNOWN_GESTURE)

        # Test no mapping rejection
        r_nomap = self.engine.process("OpenPalm", is_confirmed=True, is_active=True)
        self.assertFalse(r_nomap.accepted)
        self.assertIsNone(r_nomap.intent)
        self.assertEqual(r_nomap.reason, IntentReason.NO_MAPPING)

    # -------------------------------------------------------------------------
    # 18. Integration with Phase 4 ActivationResult
    # -------------------------------------------------------------------------
    def test_phase4_activation_result_integration(self):
        """Verify seamless pipeline chaining with ActivationResult from Phase 4."""
        # Locked ActivationResult -> rejected
        locked_act = ActivationResult(
            state=SystemState.LOCKED,
            command=None,
            is_active=False,
            just_activated=False,
            just_locked=False,
            time_since_active=0.0,
            time_remaining=0.0,
            activation_progress=0.0,
            reason="LOCKED_LISTENING",
        )
        res_locked = self.engine.process(activation_result=locked_act)
        self.assertFalse(res_locked.accepted)
        self.assertEqual(res_locked.reason, IntentReason.SYSTEM_LOCKED)

        # Active ActivationResult with command ThumbsDown -> VOLUME_DOWN
        active_act = ActivationResult(
            state=SystemState.ACTIVE,
            command="ThumbsDown",
            is_active=True,
            just_activated=False,
            just_locked=False,
            time_since_active=0.1,
            time_remaining=9.9,
            activation_progress=1.0,
            reason="COMMAND_ALLOWED",
        )
        res_active = self.engine.process(activation_result=active_act, confidence=0.91)
        self.assertTrue(res_active.accepted)
        self.assertEqual(res_active.intent, Intent.VOLUME_DOWN)
        self.assertEqual(res_active.reason, IntentReason.CONFIRMED)
        self.assertEqual(res_active.confidence, 0.91)

    # -------------------------------------------------------------------------
    # 19. Custom Mapping Registration
    # -------------------------------------------------------------------------
    def test_custom_mapping_registration(self):
        """Verify dynamic mapping extension for CUSTOM context."""
        self.context_manager.set_context(Context.CUSTOM)

        # Initially unmapped
        res_before = self.engine.process("ThumbsUp", is_confirmed=True, is_active=True)
        self.assertFalse(res_before.accepted)
        self.assertEqual(res_before.reason, IntentReason.NO_MAPPING)

        # Register mapping
        self.engine.set_mapping(Context.CUSTOM, "ThumbsUp", Intent.ZOOM_IN)

        # Now resolves successfully!
        res_after = self.engine.process("ThumbsUp", is_confirmed=True, is_active=True)
        self.assertTrue(res_after.accepted)
        self.assertEqual(res_after.intent, Intent.ZOOM_IN)

    # -------------------------------------------------------------------------
    # 20. Dynamic Swipe Tests (Phase 8B)
    # -------------------------------------------------------------------------
    def test_swipe_left_media_maps_to_previous(self):
        """Verify Swipe LEFT in MEDIA context produces PREVIOUS."""
        swipe_res = SwipeResult(
            direction=SwipeDirection.LEFT,
            confidence=0.88,
            velocity=1.2,
            displacement=-0.35,
            duration=0.25,
            path_efficiency=0.96,
            reason="SWIPE_DETECTED",
        )
        res = self.engine.process_swipe(swipe_res, is_active=True, context=Context.MEDIA)
        self.assertTrue(res.accepted)
        self.assertEqual(res.intent, Intent.PREVIOUS)
        self.assertEqual(res.context, Context.MEDIA)
        self.assertEqual(res.gesture, "SwipeLeft")
        self.assertEqual(res.reason, IntentReason.CONFIRMED)

    def test_swipe_right_media_maps_to_next(self):
        """Verify Swipe RIGHT in MEDIA context produces NEXT."""
        swipe_res = SwipeResult(
            direction=SwipeDirection.RIGHT,
            confidence=0.91,
            velocity=1.5,
            displacement=0.40,
            duration=0.22,
            path_efficiency=0.98,
            reason="SWIPE_DETECTED",
        )
        res = self.engine.process_swipe(swipe_res, is_active=True, context=Context.MEDIA)
        self.assertTrue(res.accepted)
        self.assertEqual(res.intent, Intent.NEXT)
        self.assertEqual(res.context, Context.MEDIA)
        self.assertEqual(res.gesture, "SwipeRight")
        self.assertEqual(res.reason, IntentReason.CONFIRMED)

    def test_swipe_left_presentation_maps_to_previous_slide(self):
        """Verify Swipe LEFT in PRESENTATION context produces PREVIOUS_SLIDE."""
        swipe_res = SwipeResult(
            direction=SwipeDirection.LEFT,
            confidence=0.85,
            velocity=1.1,
            displacement=-0.30,
            duration=0.25,
            path_efficiency=0.95,
            reason="SWIPE_DETECTED",
        )
        res = self.engine.process_swipe(swipe_res, is_active=True, context=Context.PRESENTATION)
        self.assertTrue(res.accepted)
        self.assertEqual(res.intent, Intent.PREVIOUS_SLIDE)
        self.assertEqual(res.context, Context.PRESENTATION)
        self.assertEqual(res.gesture, "SwipeLeft")
        self.assertEqual(res.reason, IntentReason.CONFIRMED)

    def test_swipe_right_presentation_maps_to_next_slide(self):
        """Verify Swipe RIGHT in PRESENTATION context produces NEXT_SLIDE."""
        swipe_res = SwipeResult(
            direction=SwipeDirection.RIGHT,
            confidence=0.93,
            velocity=1.6,
            displacement=0.38,
            duration=0.20,
            path_efficiency=0.97,
            reason="SWIPE_DETECTED",
        )
        res = self.engine.process_swipe(swipe_res, is_active=True, context=Context.PRESENTATION)
        self.assertTrue(res.accepted)
        self.assertEqual(res.intent, Intent.NEXT_SLIDE)
        self.assertEqual(res.context, Context.PRESENTATION)
        self.assertEqual(res.gesture, "SwipeRight")
        self.assertEqual(res.reason, IntentReason.CONFIRMED)

    def test_swipe_while_locked_rejected_system_locked(self):
        """Verify Swipe while LOCKED is strictly rejected with SYSTEM_LOCKED."""
        swipe_res = SwipeResult(
            direction=SwipeDirection.RIGHT,
            confidence=0.95,
            velocity=1.4,
            displacement=0.35,
            duration=0.20,
            reason="SWIPE_DETECTED",
        )
        # Test via is_active=False
        res1 = self.engine.process_swipe(swipe_res, is_active=False, context=Context.MEDIA)
        self.assertFalse(res1.accepted)
        self.assertEqual(res1.reason, IntentReason.SYSTEM_LOCKED)
        self.assertIsNone(res1.intent)

        # Test via state=SystemState.LOCKED
        res2 = self.engine.process_swipe(swipe_res, state=SystemState.LOCKED, context=Context.MEDIA)
        self.assertFalse(res2.accepted)
        self.assertEqual(res2.reason, IntentReason.SYSTEM_LOCKED)

    def test_swipe_unconfirmed_or_none_rejected(self):
        """Verify Swipe with direction NONE is rejected with GESTURE_NOT_CONFIRMED."""
        swipe_none = SwipeResult(
            direction=SwipeDirection.NONE,
            confidence=0.0,
            reason="NO_SWIPE",
        )
        res = self.engine.process_swipe(swipe_none, is_active=True, context=Context.MEDIA)
        self.assertFalse(res.accepted)
        self.assertEqual(res.reason, IntentReason.GESTURE_NOT_CONFIRMED)

    def test_swipe_custom_context_unmapped_and_custom_registration(self):
        """Verify Swipe in CUSTOM context is unmapped by default and can be registered."""
        swipe_res = SwipeResult(
            direction=SwipeDirection.RIGHT,
            confidence=0.90,
            velocity=1.2,
            displacement=0.30,
            duration=0.22,
        )
        # Initially unmapped in CUSTOM
        res_before = self.engine.process_swipe(swipe_res, is_active=True, context=Context.CUSTOM)
        self.assertFalse(res_before.accepted)
        self.assertEqual(res_before.reason, IntentReason.NO_MAPPING)

        # Register custom mapping
        self.engine.set_swipe_mapping(Context.CUSTOM, SwipeDirection.RIGHT, Intent.ZOOM_IN)
        res_after = self.engine.process_swipe(swipe_res, is_active=True, context=Context.CUSTOM)
        self.assertTrue(res_after.accepted)
        self.assertEqual(res_after.intent, Intent.ZOOM_IN)

    def test_swipe_result_direction_validation(self):
        """Verify SwipeResult direction reflects valid swipe detection."""
        res_right = SwipeResult(direction=SwipeDirection.RIGHT)
        res_left = SwipeResult(direction=SwipeDirection.LEFT)
        res_none = SwipeResult(direction=SwipeDirection.NONE)
        self.assertEqual(res_right.direction, SwipeDirection.RIGHT)
        self.assertEqual(res_left.direction, SwipeDirection.LEFT)
        self.assertEqual(res_none.direction, SwipeDirection.NONE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
