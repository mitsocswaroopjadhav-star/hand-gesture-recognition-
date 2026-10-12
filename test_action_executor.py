"""
test_action_executor.py
=======================

Unit test suite for IntentSense Phase 6: Action Execution Layer.

Covers:
 1. Accepted VOLUME_UP executes the correct handler.
 2. Accepted VOLUME_DOWN executes the correct handler.
 3. Accepted NEXT executes the correct handler.
 4. Accepted PREVIOUS executes the correct handler.
 5. Accepted PLAY_PAUSE executes the correct handler.
 6. Accepted PRESENTATION intent routes to the correct handler.
 7. Rejected IntentResult executes NOTHING.
 8. SYSTEM_LOCKED result executes NOTHING.
 9. GESTURE_NOT_CONFIRMED result executes NOTHING.
10. UNKNOWN_GESTURE result executes NOTHING.
11. NO_MAPPING result executes NOTHING.
12. None IntentResult is rejected safely.
13. Unknown/unregistered intent returns NO_HANDLER.
14. Handler exception is caught and converted into a structured failure result.
15. Successful execution returns success=True.
16. Failed execution returns success=False.
17. ActionResult contains the correct intent.
18. ActionResult contains the correct action name.
19. Registry supports registering a new action.
20. Registering a new action does not require changing execute().
21. Multiple executors/registries do not accidentally share mutable state.
22. Repeated valid execution behaves deterministically.
23. No real OS side effects occur during unit tests (mock/dry-run verified).
Additional:
24. Action cooldown / debouncing extension point test.
25. Integration with actions/ media and presentation dry-run handlers.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from actions.media_actions import register_media_handlers
from actions.presentation_actions import register_presentation_handlers
from core.action_executor import (
    ActionExecutor,
    ActionReason,
    ActionResult,
)
from core.context_manager import Context
from core.intent_engine import (
    Intent,
    IntentReason,
    IntentResult,
)


class TestActionExecutor(unittest.TestCase):
    """Test suite for core/action_executor.py."""

    def setUp(self):
        """Creates a clean, isolated ActionExecutor for each test."""
        self.executor = ActionExecutor()
        self.mock_vol_up = MagicMock(return_value="vol_up_done")
        self.mock_vol_down = MagicMock(return_value="vol_down_done")
        self.mock_next = MagicMock(return_value="next_done")
        self.mock_prev = MagicMock(return_value="prev_done")
        self.mock_play_pause = MagicMock(return_value="play_pause_done")
        self.mock_next_slide = MagicMock(return_value="next_slide_done")

        self.executor.register(Intent.VOLUME_UP, self.mock_vol_up, action_name="volume_up")
        self.executor.register(Intent.VOLUME_DOWN, self.mock_vol_down, action_name="volume_down")
        self.executor.register(Intent.NEXT, self.mock_next, action_name="next_track")
        self.executor.register(Intent.PREVIOUS, self.mock_prev, action_name="previous_track")
        self.executor.register(Intent.PLAY_PAUSE, self.mock_play_pause, action_name="play_pause")
        self.executor.register(Intent.NEXT_SLIDE, self.mock_next_slide, action_name="next_slide")

    def _create_accepted_intent(self, intent: Intent, context: Context = Context.MEDIA) -> IntentResult:
        """Helper to create an accepted IntentResult."""
        return IntentResult(
            gesture="TestGesture",
            intent=intent,
            context=context,
            confidence=0.95,
            accepted=True,
            reason=IntentReason.CONFIRMED,
        )

    # -------------------------------------------------------------------------
    # 1. Accepted VOLUME_UP executes the correct handler
    # -------------------------------------------------------------------------
    def test_accepted_volume_up_executes_correct_handler(self):
        intent_res = self._create_accepted_intent(Intent.VOLUME_UP)
        result = self.executor.execute(intent_res)

        self.assertTrue(result.success)
        self.mock_vol_up.assert_called_once()
        self.mock_vol_down.assert_not_called()
        self.assertEqual(result.action_name, "volume_up")
        self.assertEqual(result.reason, ActionReason.EXECUTED)

    # -------------------------------------------------------------------------
    # 2. Accepted VOLUME_DOWN executes the correct handler
    # -------------------------------------------------------------------------
    def test_accepted_volume_down_executes_correct_handler(self):
        intent_res = self._create_accepted_intent(Intent.VOLUME_DOWN)
        result = self.executor.execute(intent_res)

        self.assertTrue(result.success)
        self.mock_vol_down.assert_called_once()
        self.mock_vol_up.assert_not_called()
        self.assertEqual(result.action_name, "volume_down")

    # -------------------------------------------------------------------------
    # 3. Accepted NEXT executes the correct handler
    # -------------------------------------------------------------------------
    def test_accepted_next_executes_correct_handler(self):
        intent_res = self._create_accepted_intent(Intent.NEXT)
        result = self.executor.execute(intent_res)

        self.assertTrue(result.success)
        self.mock_next.assert_called_once()
        self.assertEqual(result.action_name, "next_track")

    # -------------------------------------------------------------------------
    # 4. Accepted PREVIOUS executes the correct handler
    # -------------------------------------------------------------------------
    def test_accepted_previous_executes_correct_handler(self):
        intent_res = self._create_accepted_intent(Intent.PREVIOUS)
        result = self.executor.execute(intent_res)

        self.assertTrue(result.success)
        self.mock_prev.assert_called_once()
        self.assertEqual(result.action_name, "previous_track")

    # -------------------------------------------------------------------------
    # 5. Accepted PLAY_PAUSE executes the correct handler
    # -------------------------------------------------------------------------
    def test_accepted_play_pause_executes_correct_handler(self):
        intent_res = self._create_accepted_intent(Intent.PLAY_PAUSE)
        result = self.executor.execute(intent_res)

        self.assertTrue(result.success)
        self.mock_play_pause.assert_called_once()
        self.assertEqual(result.action_name, "play_pause")

    # -------------------------------------------------------------------------
    # 6. Accepted PRESENTATION intent routes to the correct handler
    # -------------------------------------------------------------------------
    def test_accepted_presentation_intent_routes_correctly(self):
        intent_res = self._create_accepted_intent(Intent.NEXT_SLIDE, context=Context.PRESENTATION)
        result = self.executor.execute(intent_res)

        self.assertTrue(result.success)
        self.mock_next_slide.assert_called_once()
        self.mock_next.assert_not_called()
        self.assertEqual(result.action_name, "next_slide")
        self.assertEqual(result.intent, Intent.NEXT_SLIDE)

    # -------------------------------------------------------------------------
    # 7. Rejected IntentResult executes NOTHING
    # -------------------------------------------------------------------------
    def test_rejected_intent_result_executes_nothing(self):
        rejected_res = IntentResult(
            gesture="Fist",
            intent=None,
            context=Context.MEDIA,
            confidence=0.5,
            accepted=False,
            reason=IntentReason.GESTURE_NOT_CONFIRMED,
        )
        result = self.executor.execute(rejected_res)

        self.assertFalse(result.success)
        self.assertEqual(result.reason, ActionReason.INTENT_REJECTED)
        self.mock_vol_up.assert_not_called()
        self.mock_play_pause.assert_not_called()

    # -------------------------------------------------------------------------
    # 8. SYSTEM_LOCKED result executes NOTHING
    # -------------------------------------------------------------------------
    def test_system_locked_result_executes_nothing(self):
        locked_res = IntentResult(
            gesture="ThumbsUp",
            intent=None,
            context=Context.MEDIA,
            confidence=0.98,
            accepted=False,
            reason=IntentReason.SYSTEM_LOCKED,
        )
        result = self.executor.execute(locked_res)

        self.assertFalse(result.success)
        self.assertEqual(result.reason, ActionReason.INTENT_REJECTED)
        self.assertIn("SYSTEM_LOCKED", result.message)
        self.mock_vol_up.assert_not_called()

    # -------------------------------------------------------------------------
    # 9. GESTURE_NOT_CONFIRMED result executes NOTHING
    # -------------------------------------------------------------------------
    def test_gesture_not_confirmed_result_executes_nothing(self):
        unconfirmed_res = IntentResult(
            gesture="ThumbsUp",
            intent=None,
            context=Context.MEDIA,
            confidence=0.75,
            accepted=False,
            reason=IntentReason.GESTURE_NOT_CONFIRMED,
        )
        result = self.executor.execute(unconfirmed_res)

        self.assertFalse(result.success)
        self.assertEqual(result.reason, ActionReason.INTENT_REJECTED)
        self.mock_vol_up.assert_not_called()

    # -------------------------------------------------------------------------
    # 10. UNKNOWN_GESTURE result executes NOTHING
    # -------------------------------------------------------------------------
    def test_unknown_gesture_result_executes_nothing(self):
        unknown_res = IntentResult(
            gesture="AlienSign",
            intent=None,
            context=Context.MEDIA,
            confidence=0.99,
            accepted=False,
            reason=IntentReason.UNKNOWN_GESTURE,
        )
        result = self.executor.execute(unknown_res)

        self.assertFalse(result.success)
        self.assertEqual(result.reason, ActionReason.INTENT_REJECTED)
        self.mock_vol_up.assert_not_called()

    # -------------------------------------------------------------------------
    # 11. NO_MAPPING result executes NOTHING
    # -------------------------------------------------------------------------
    def test_no_mapping_result_executes_nothing(self):
        nomap_res = IntentResult(
            gesture="OpenPalm",
            intent=None,
            context=Context.MEDIA,
            confidence=0.99,
            accepted=False,
            reason=IntentReason.NO_MAPPING,
        )
        result = self.executor.execute(nomap_res)

        self.assertFalse(result.success)
        self.assertEqual(result.reason, ActionReason.INTENT_REJECTED)
        self.mock_vol_up.assert_not_called()

    # -------------------------------------------------------------------------
    # 12. None IntentResult is rejected safely
    # -------------------------------------------------------------------------
    def test_none_intent_result_rejected_safely(self):
        result = self.executor.execute(None)

        self.assertFalse(result.success)
        self.assertIsNone(result.intent)
        self.assertEqual(result.reason, ActionReason.INTENT_REJECTED)
        self.assertIn("None", result.message)

    # -------------------------------------------------------------------------
    # 13. Unknown/unregistered intent returns NO_HANDLER
    # -------------------------------------------------------------------------
    def test_unregistered_intent_returns_no_handler(self):
        unregistered_res = self._create_accepted_intent(Intent.ZOOM_IN)
        result = self.executor.execute(unregistered_res)

        self.assertFalse(result.success)
        self.assertEqual(result.intent, Intent.ZOOM_IN)
        self.assertEqual(result.reason, ActionReason.NO_HANDLER)
        self.assertIsNone(result.action_name)
        self.assertIn("No action handler registered", result.message)

    # -------------------------------------------------------------------------
    # 14. Handler exception is caught and converted into structured failure
    # -------------------------------------------------------------------------
    def test_handler_exception_caught_safely(self):
        failing_handler = MagicMock(side_effect=RuntimeError("System audio service unavailable"))
        self.executor.register(Intent.ZOOM_OUT, failing_handler, action_name="zoom_out")

        res = self._create_accepted_intent(Intent.ZOOM_OUT)
        result = self.executor.execute(res)

        self.assertFalse(result.success)
        self.assertEqual(result.reason, ActionReason.EXECUTION_FAILED)
        self.assertEqual(result.action_name, "zoom_out")
        self.assertIn("System audio service unavailable", result.message)

    # -------------------------------------------------------------------------
    # 15. Successful execution returns success=True
    # -------------------------------------------------------------------------
    def test_successful_execution_returns_success_true(self):
        res = self._create_accepted_intent(Intent.VOLUME_UP)
        result = self.executor.execute(res)
        self.assertIs(result.success, True)

    # -------------------------------------------------------------------------
    # 16. Failed execution returns success=False
    # -------------------------------------------------------------------------
    def test_failed_execution_returns_success_false(self):
        failing_handler = MagicMock(side_effect=ValueError("Invalid parameter"))
        self.executor.register(Intent.START_STOP, failing_handler, action_name="start_stop")

        res = self._create_accepted_intent(Intent.START_STOP)
        result = self.executor.execute(res)
        self.assertIs(result.success, False)

    # -------------------------------------------------------------------------
    # 17. ActionResult contains the correct intent
    # -------------------------------------------------------------------------
    def test_action_result_contains_correct_intent(self):
        res = self._create_accepted_intent(Intent.PREVIOUS)
        result = self.executor.execute(res)
        self.assertEqual(result.intent, Intent.PREVIOUS)

    # -------------------------------------------------------------------------
    # 18. ActionResult contains the correct action name
    # -------------------------------------------------------------------------
    def test_action_result_contains_correct_action_name(self):
        res = self._create_accepted_intent(Intent.NEXT)
        result = self.executor.execute(res)
        self.assertEqual(result.action_name, "next_track")

    # -------------------------------------------------------------------------
    # 19. Registry supports registering a new action
    # -------------------------------------------------------------------------
    def test_registry_supports_registering_new_action(self):
        custom_handler = MagicMock(return_value="custom_done")
        self.assertFalse(self.executor.has_handler(Intent.ZOOM_IN))

        self.executor.register(Intent.ZOOM_IN, custom_handler, action_name="custom_zoom")
        self.assertTrue(self.executor.has_handler(Intent.ZOOM_IN))

        res = self._create_accepted_intent(Intent.ZOOM_IN)
        result = self.executor.execute(res)
        self.assertTrue(result.success)
        self.assertEqual(result.action_name, "custom_zoom")
        custom_handler.assert_called_once()

    # -------------------------------------------------------------------------
    # 20. Registering a new action does not require changing execute()
    # -------------------------------------------------------------------------
    def test_registering_new_action_requires_no_execute_change(self):
        new_intent_handler = MagicMock(return_value="registered_dynamically")
        self.executor.register(Intent.ZOOM_OUT, new_intent_handler, action_name="zoom_out_dynamic")

        res = self._create_accepted_intent(Intent.ZOOM_OUT)
        result = self.executor.execute(res)
        self.assertTrue(result.success)
        self.assertEqual(result.action_name, "zoom_out_dynamic")
        new_intent_handler.assert_called_once()

    # -------------------------------------------------------------------------
    # 21. Multiple executors do not share mutable state
    # -------------------------------------------------------------------------
    def test_multiple_executors_do_not_share_mutable_state(self):
        exec1 = ActionExecutor()
        exec2 = ActionExecutor()

        mock_h = MagicMock()
        exec1.register(Intent.VOLUME_UP, mock_h, action_name="exec1_vol")

        self.assertTrue(exec1.has_handler(Intent.VOLUME_UP))
        self.assertFalse(exec2.has_handler(Intent.VOLUME_UP))

    # -------------------------------------------------------------------------
    # 22. Repeated valid execution behaves deterministically
    # -------------------------------------------------------------------------
    def test_repeated_valid_execution_behaves_deterministically(self):
        res = self._create_accepted_intent(Intent.VOLUME_UP)
        for i in range(5):
            out = self.executor.execute(res)
            self.assertTrue(out.success)
            self.assertEqual(out.reason, ActionReason.EXECUTED)
            self.assertEqual(out.action_name, "volume_up")

        self.assertEqual(self.mock_vol_up.call_count, 5)

    # -------------------------------------------------------------------------
    # 23. No real OS side effects occur during unit tests
    # -------------------------------------------------------------------------
    def test_no_real_os_side_effects_occur(self):
        """Verifies that handlers used are mocks or dry-run, guaranteeing safety."""
        fresh_executor = ActionExecutor()
        register_media_handlers(fresh_executor, dry_run=True)
        register_presentation_handlers(fresh_executor, dry_run=True)

        for intent in [
            Intent.VOLUME_UP,
            Intent.VOLUME_DOWN,
            Intent.PLAY_PAUSE,
            Intent.NEXT,
            Intent.PREVIOUS,
            Intent.START_STOP,
            Intent.NEXT_SLIDE,
            Intent.PREVIOUS_SLIDE,
            Intent.ZOOM_IN,
            Intent.ZOOM_OUT,
        ]:
            res = self._create_accepted_intent(intent)
            out = fresh_executor.execute(res)
            self.assertTrue(out.success)
            self.assertIn("[DRY RUN]", out.message)

    # -------------------------------------------------------------------------
    # 24. Action cooldown / debouncing extension point test
    # -------------------------------------------------------------------------
    def test_action_cooldown_extension_point(self):
        """Verifies that action_cooldown rate-limits repeated identical actions."""
        debounced_exec = ActionExecutor(action_cooldown=1.0)
        debounced_exec.register(Intent.VOLUME_UP, self.mock_vol_up, action_name="volume_up")

        res = self._create_accepted_intent(Intent.VOLUME_UP)

        # First call succeeds
        r1 = debounced_exec.execute(res)
        self.assertTrue(r1.success)
        self.assertEqual(r1.reason, ActionReason.EXECUTED)

        # Immediate second call is suppressed by cooldown
        r2 = debounced_exec.execute(res)
        self.assertFalse(r2.success)
        self.assertEqual(r2.reason, ActionReason.COOLDOWN_ACTIVE)
        self.assertIn("cooldown", r2.message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
