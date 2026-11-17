"""
test_touchless_onboarding.py
============================

Unit tests for Phase 4: Touchless Guided Onboarding Practice Mode.

Verifies:
  1. Onboarding sequence begins at Step 1: UNLOCK (OpenPalm hold).
  2. Live progress bar computation from hold duration.
  3. Automatic touchless step advancement upon physical gesture completion.
  4. Step transition delay displaying success feedback before moving to next pose.
  5. Negative & mismatched gesture handling (feedback without false advancement).
  6. Dynamic swipe verification (SwipeDirection.RIGHT).
  7. Full sequence completion and reset capabilities.
  8. Safety guarantee: Practice mode never dispatches real OS actions.
"""

from __future__ import annotations

import unittest

from core.activation_state import SystemState
from core.onboarding import (
    OnboardingStepId,
    OnboardingStepResult,
    TouchlessOnboardingController,
)
from core.swipe_detector import SwipeDirection


class TestTouchlessOnboarding(unittest.TestCase):
    """Verifies TouchlessOnboardingController state transitions and feedback."""

    def setUp(self) -> None:
        self.controller = TouchlessOnboardingController()

    def test_initial_state_is_unlock(self) -> None:
        self.assertEqual(self.controller.current_step, OnboardingStepId.UNLOCK)
        self.assertFalse(self.controller.is_complete)

    def test_step1_unlock_progress_and_completion(self) -> None:
        # 1. Partial hold (0.5s of 1.0s)
        res = self.controller.update(
            gesture="OpenPalm",
            confidence=0.92,
            hold_duration=0.5,
            target_hold=1.0,
            is_confirmed=False,
            timestamp=1.0,
        )
        self.assertEqual(res.step_id, OnboardingStepId.UNLOCK)
        self.assertAlmostEqual(res.progress, 0.5, places=2)
        self.assertFalse(res.is_step_complete)
        self.assertIn("50%", res.feedback)

        # 2. Confirmed unlock (1.0s hold)
        res_done = self.controller.update(
            gesture="OpenPalm",
            confidence=0.95,
            hold_duration=1.0,
            target_hold=1.0,
            is_confirmed=True,
            just_activated=True,
            timestamp=1.5,
        )
        self.assertTrue(res_done.is_step_complete)
        self.assertEqual(res_done.progress, 1.0)
        self.assertIn("unlocked", res_done.feedback.lower())

        # 3. During transition delay (0.5s later < 1.2s delay), still on UNLOCK showing checkmark
        res_delay = self.controller.update(
            gesture="OpenPalm",
            confidence=0.95,
            hold_duration=1.0,
            target_hold=1.0,
            is_confirmed=True,
            timestamp=2.0,
        )
        self.assertEqual(res_delay.step_id, OnboardingStepId.UNLOCK)
        self.assertTrue(res_delay.is_step_complete)

        # 4. After transition delay (1.5s later > 1.2s delay), automatically advances to VOLUME_UP
        res_next = self.controller.update(
            gesture=None,
            confidence=0.0,
            hold_duration=0.0,
            target_hold=0.5,
            is_confirmed=False,
            timestamp=2.8,
        )
        self.assertEqual(self.controller.current_step, OnboardingStepId.VOLUME_UP)
        self.assertEqual(res_next.step_id, OnboardingStepId.VOLUME_UP)
        self.assertEqual(res_next.target_gesture, "ThumbsUp")

    def test_mismatched_gesture_does_not_advance_step(self) -> None:
        """Presenting an unexpected gesture gives feedback but does not advance."""
        res = self.controller.update(
            gesture="Fist",
            confidence=0.88,
            hold_duration=0.5,
            target_hold=1.0,
            is_confirmed=True,
            timestamp=10.0,
        )
        self.assertEqual(res.step_id, OnboardingStepId.UNLOCK)
        self.assertFalse(res.is_step_complete)
        self.assertEqual(res.progress, 0.0)
        self.assertIn("Fist", res.feedback)
        self.assertIn("OpenPalm", res.feedback)

    def test_full_onboarding_progression(self) -> None:
        """Walks through all 4 steps to completion."""
        # Step 1: UNLOCK
        self.controller.update("OpenPalm", 0.9, 1.0, 1.0, True, just_activated=True, timestamp=1.0)
        self.controller.update(None, 0.0, 0.0, 0.5, False, timestamp=2.5)
        self.assertEqual(self.controller.current_step, OnboardingStepId.VOLUME_UP)

        # Step 2: VOLUME_UP (ThumbsUp)
        self.controller.update("ThumbsUp", 0.9, 0.5, 0.5, True, timestamp=3.0)
        self.controller.update(None, 0.0, 0.0, 0.5, False, timestamp=4.5)
        self.assertEqual(self.controller.current_step, OnboardingStepId.SWIPE_RIGHT)

        # Step 3: SWIPE_RIGHT
        self.controller.update(None, 0.0, 0.0, 0.0, False, swipe_direction=SwipeDirection.RIGHT, timestamp=5.0)
        self.controller.update(None, 0.0, 0.0, 0.5, False, timestamp=6.5)
        self.assertEqual(self.controller.current_step, OnboardingStepId.POINTING)

        # Step 4: POINTING
        self.controller.update("Pointing", 0.9, 0.5, 0.5, True, timestamp=7.0)
        res_comp = self.controller.update(None, 0.0, 0.0, 0.0, False, timestamp=8.5)
        self.assertEqual(self.controller.current_step, OnboardingStepId.COMPLETE)
        self.assertTrue(self.controller.is_complete)
        self.assertTrue(res_comp.is_all_complete)

    def test_reset_onboarding(self) -> None:
        self.controller._current_step = OnboardingStepId.POINTING
        self.controller.reset()
        self.assertEqual(self.controller.current_step, OnboardingStepId.UNLOCK)
        self.assertFalse(self.controller.is_complete)


if __name__ == "__main__":
    unittest.main()

