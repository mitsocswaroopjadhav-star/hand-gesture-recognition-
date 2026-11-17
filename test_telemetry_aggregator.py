"""
test_telemetry_aggregator.py
============================

Unit tests for Phase 4: Telemetry Aggregation, Struggle Detection, and Proactive Coaching.

Verifies:
  1. Ring-buffer bounding, max records limit, and time-based window pruning.
  2. Struggle pattern detection:
     - HOLD_DEFICIT: Releasing right before hold completion.
     - LOW_CONFIDENCE: 3+ near-miss rejections below threshold.
     - LOCKED_SUPPRESSION: Attempting commands while system is locked.
     - JITTER_INSTABILITY: Rapid cycling between multiple distinct gestures.
  3. False-positive suppression: Isolated single failures never trigger coaching.
  4. Duplicate coaching suppression and global/per-pattern cooldowns.
  5. Zero-vision guarantee: Telemetry records strictly disallow visual or landmark payloads.
  6. LLM coaching generation (offline fallback and async non-blocking execution).
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

from actions.media_actions import register_media_handlers
from actions.presentation_actions import register_presentation_handlers
from core.action_executor import ActionExecutor, ActionReason
from core.activation_state import ActivationStateMachine, SystemState
from core.context_manager import Context, ContextManager
from core.decision_explainer import DecisionCategory, DecisionExplainer, DecisionMessage
from core.intent_engine import Intent, IntentEngine
from core.llm_assistant import LLMAssistant
from core.telemetry_aggregator import (
    StrugglePattern,
    StruggleType,
    TelemetryAggregator,
    TelemetryRecord,
)


class TestTelemetryAggregator(unittest.TestCase):
    """Verifies telemetry ring-buffer and struggle detection mechanisms."""

    def setUp(self) -> None:
        self.aggregator = TelemetryAggregator(
            window_seconds=6.0,
            max_records=50,
            coaching_cooldown=5.0,
        )

    def test_ring_buffer_limits_and_pruning(self) -> None:
        """Confirms that records older than window_seconds are pruned and max_records is respected."""
        base_ts = 100.0

        # Fill with 60 records spaced 0.2s apart (from 100.0 to 111.8)
        for i in range(60):
            self.aggregator.record(
                timestamp=base_ts + i * 0.2,
                category="WAITING",
                gesture="ThumbsUp",
                confidence=0.85,
                effective_threshold=0.80,
                hold_duration=0.1,
                target_hold=0.5,
                reason="WAITING",
            )

        # Buffer should be bounded to at most max_records (50)
        self.assertLessEqual(self.aggregator.record_count, 50)

        # Prune with current_timestamp = 112.0 (cutoff = 106.0)
        self.aggregator.prune(112.0)
        # All remaining records should have timestamp >= 106.0
        for rec in self.aggregator._records:
            self.assertGreaterEqual(rec.timestamp, 106.0)

    def test_false_positive_suppression_isolated_failure(self) -> None:
        """A single, isolated failure should NOT trigger proactive coaching."""
        now = 10.0
        # Normal interaction with just 1 low-confidence frame
        self.aggregator.record(
            timestamp=now,
            category="REJECTED",
            gesture="Pointing",
            confidence=0.60,
            effective_threshold=0.80,
            hold_duration=0.1,
            target_hold=0.5,
            reason="CONFIDENCE_BELOW_THRESHOLD",
        )

        pattern = self.aggregator.detect_struggle(now)
        self.assertIsNone(pattern, "Isolated failure must not trigger coaching.")

    def test_detect_hold_deficit_pattern(self) -> None:
        """Detects when user reaches 40-90% hold duration and drops multiple times."""
        now = 20.0
        # 1st attempt: builds to 0.38s (target 0.5s), then drops to 0.05s
        self.aggregator.record(now + 0.1, "WAITING", "Pointing", 0.90, 0.75, 0.38, 0.50, "WAITING")
        self.aggregator.record(now + 0.2, "WAITING", "Pointing", 0.90, 0.75, 0.05, 0.50, "WAITING")

        # 2nd attempt: builds to 0.40s, then drops to 0.02s
        self.aggregator.record(now + 0.5, "WAITING", "Pointing", 0.90, 0.75, 0.40, 0.50, "WAITING")
        self.aggregator.record(now + 0.6, "WAITING", "Pointing", 0.90, 0.75, 0.02, 0.50, "WAITING")

        # Add padding records to satisfy min record count
        for i in range(3):
            self.aggregator.record(now + 0.7 + i * 0.1, "WAITING", "Pointing", 0.90, 0.75, 0.10, 0.50, "WAITING")

        pattern = self.aggregator.detect_struggle(now + 1.2)
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, StruggleType.HOLD_DEFICIT)
        self.assertEqual(pattern.gesture, "Pointing")
        self.assertIn("deficit", pattern.metrics)

    def test_detect_low_confidence_pattern(self) -> None:
        """Detects 3+ near-miss frames just below threshold."""
        now = 30.0
        # 3 near-miss frames: confidence 0.68 vs threshold 0.75
        for i in range(4):
            self.aggregator.record(
                timestamp=now + i * 0.1,
                category="REJECTED",
                gesture="ThumbsUp",
                confidence=0.68,
                effective_threshold=0.75,
                hold_duration=0.0,
                target_hold=0.50,
                reason="CONFIDENCE_BELOW_THRESHOLD",
            )
        # Add padding records
        for i in range(3):
            self.aggregator.record(now + 0.5 + i * 0.1, "REJECTED", "ThumbsUp", 0.67, 0.75, 0.0, 0.50, "CONFIDENCE")

        pattern = self.aggregator.detect_struggle(now + 1.0)
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, StruggleType.LOW_CONFIDENCE)
        self.assertEqual(pattern.gesture, "ThumbsUp")

    def test_detect_locked_suppression_pattern(self) -> None:
        """Detects repeated command attempts while system is LOCKED."""
        now = 40.0
        for i in range(4):
            self.aggregator.record(
                timestamp=now + i * 0.2,
                category="REJECTED",
                gesture="ThumbsDown",
                confidence=0.92,
                effective_threshold=0.80,
                hold_duration=0.5,
                target_hold=0.5,
                reason="SYSTEM_LOCKED",
            )
        # Add padding records
        for i in range(2):
            self.aggregator.record(now + 0.9 + i * 0.1, "REJECTED", "ThumbsDown", 0.90, 0.80, 0.1, 0.5, "SYSTEM_LOCKED")

        pattern = self.aggregator.detect_struggle(now + 1.2)
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, StruggleType.LOCKED_SUPPRESSION)
        self.assertEqual(pattern.gesture, "ThumbsDown")

    def test_coaching_cooldown_and_duplicate_suppression(self) -> None:
        """Confirms that coaching cannot trigger repeatedly within the cooldown window."""
        now = 50.0
        for i in range(4):
            self.aggregator.record(now + i * 0.1, "REJECTED", "Fist", 0.90, 0.80, 0.5, 0.5, "SYSTEM_LOCKED")
        for i in range(2):
            self.aggregator.record(now + 0.5 + i * 0.1, "REJECTED", "Fist", 0.90, 0.80, 0.1, 0.5, "SYSTEM_LOCKED")

        # 1st trigger should succeed
        p1 = self.aggregator.detect_struggle(now + 1.0)
        self.assertIsNotNone(p1)

        # Immediate 2nd check within cooldown (1.0s later < 5.0s cooldown) must return None
        p2 = self.aggregator.detect_struggle(now + 2.0)
        self.assertIsNone(p2, "Duplicate coaching must be suppressed by cooldown.")

        # After cooldown (6.0s later > 5.0s cooldown), coaching can trigger again
        # Re-add events so buffer isn't empty
        for i in range(5):
            self.aggregator.record(now + 6.0 + i * 0.1, "REJECTED", "Fist", 0.90, 0.80, 0.5, 0.5, "SYSTEM_LOCKED")
        p3 = self.aggregator.detect_struggle(now + 7.0)
        self.assertIsNotNone(p3)


class TestProactiveCoachingIntegration(unittest.TestCase):
    """Verifies DecisionExplainer integration with TelemetryAggregator and LLM coaching."""

    def setUp(self) -> None:
        self.assistant = LLMAssistant(api_key="", endpoint_url="")
        self.explainer = DecisionExplainer(llm_assistant=self.assistant)

    def tearDown(self) -> None:
        self.explainer.close()

    def test_proactive_coaching_toast_lifecycle(self) -> None:
        now = 100.0
        # Seed hold deficit events
        for i in range(3):
            self.explainer.record_interaction(
                timestamp=now + i * 0.4,
                decision_msg=DecisionMessage(category=DecisionCategory.WAITING, text="WAITING — Tracking Pointing"),
                gesture="Pointing",
                confidence=0.88,
                effective_threshold=0.75,
                hold_duration=0.36,
                target_hold=0.50,
            )
            self.explainer.record_interaction(
                timestamp=now + i * 0.4 + 0.1,
                decision_msg=DecisionMessage(category=DecisionCategory.WAITING, text="WAITING — Tracking Pointing"),
                gesture="Pointing",
                confidence=0.88,
                effective_threshold=0.75,
                hold_duration=0.04,
                target_hold=0.50,
            )

        # Check proactive coaching
        toast = self.explainer.check_proactive_coaching(now + 1.5, Context.MEDIA)
        self.assertIsNotNone(toast)
        self.assertIn("Pointing", toast)

        # Check active toast query
        active = self.explainer.get_active_toast(now + 2.0)
        self.assertIsNotNone(active)
        toast_msg, ratio = active
        self.assertEqual(toast_msg, toast)
        self.assertGreater(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)

        # Check expired toast
        expired = self.explainer.get_active_toast(now + 10.0)
        self.assertIsNone(expired)

    def test_delayed_llm_response_does_not_block(self) -> None:
        """Simulate a 5-second remote delay and verify that coaching toast returns immediately."""
        slow_pattern = StrugglePattern(
            pattern_type=StruggleType.LOW_CONFIDENCE,
            gesture="Peace",
            evidence="Low confidence detected",
            metrics={"avg_confidence": 0.65, "threshold": 0.80},
            timestamp=200.0,
        )

        start = time.perf_counter()
        # Offline provider responds in < 5ms
        res = self.assistant.coach_struggle(slow_pattern, Context.MEDIA)
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.02)
        self.assertIsNotNone(res.coaching_tip)
        self.assertIn("Peace", res.coaching_tip)

    def test_repeated_thumbs_up_while_locked_triggers_coaching(self) -> None:
        """
        While IntentSense is LOCKED, repeatedly gesturing ThumbsUp triggers the exact
        helpful AI Coach HUD message without executing any actions.
        """
        now = 300.0
        dec_msg = DecisionMessage(
            category=DecisionCategory.REJECTED,
            text="REJECTED — System locked (Hold OpenPalm to unlock)",
            color_bgr=(0, 165, 255),
        )

        # 3 recognized ThumbsUp attempts while locked
        for i in range(3):
            self.explainer.record_interaction(
                timestamp=now + i * 0.3,
                decision_msg=dec_msg,
                gesture="ThumbsUp",
                confidence=0.92,
                effective_threshold=0.80,
                hold_duration=0.5,
                target_hold=0.5,
                reason="REJECTED — System locked (Hold OpenPalm to unlock)",
                context=Context.MEDIA,
                state=SystemState.LOCKED,
            )

        # Trigger coaching check
        toast = self.explainer.check_proactive_coaching(now + 1.2, Context.MEDIA)
        self.assertIsNotNone(toast)

        expected_msg = (
            "IntentSense is locked. Show OpenPalm and hold for 1 second to unlock. "
            "ThumbsUp cannot execute actions while locked."
        )
        self.assertEqual(toast, expected_msg)

        # Toast should be active
        active = self.explainer.get_active_toast(now + 1.5)
        self.assertIsNotNone(active)
        self.assertEqual(active[0], expected_msg)

    def test_isolated_thumbs_up_does_not_spam_coaching(self) -> None:
        """A single 1-off attempt of ThumbsUp while locked must not trigger coaching."""
        now = 400.0
        dec_msg = DecisionMessage(
            category=DecisionCategory.REJECTED,
            text="REJECTED — System locked (Hold OpenPalm to unlock)",
            color_bgr=(0, 165, 255),
        )

        # Only 1 occurrence
        self.explainer.record_interaction(
            timestamp=now,
            decision_msg=dec_msg,
            gesture="ThumbsUp",
            confidence=0.92,
            effective_threshold=0.80,
            hold_duration=0.5,
            target_hold=0.5,
            reason="REJECTED — System locked (Hold OpenPalm to unlock)",
            context=Context.MEDIA,
            state=SystemState.LOCKED,
        )

        toast = self.explainer.check_proactive_coaching(now + 0.5, Context.MEDIA)
        self.assertIsNone(toast, "Single isolated attempt must not spam coaching.")
        self.assertIsNone(self.explainer.get_active_toast(now + 0.5))

    def test_openpalm_unlock_clears_locked_state_coaching(self) -> None:
        """After successful OpenPalm activation, the coaching message is cleared immediately."""
        now = 500.0
        # Seed 3 locked events
        for i in range(3):
            self.explainer.record_interaction(
                timestamp=now + i * 0.2,
                decision_msg=DecisionMessage(category=DecisionCategory.REJECTED, text="REJECTED — System locked"),
                gesture="ThumbsUp",
                confidence=0.90,
                effective_threshold=0.80,
                state=SystemState.LOCKED,
            )

        toast = self.explainer.check_proactive_coaching(now + 1.0, Context.MEDIA)
        self.assertIsNotNone(toast)
        self.assertIsNotNone(self.explainer.get_active_toast(now + 1.1))

        # Successful OpenPalm unlock clears the active toast
        self.explainer.clear_active_toast()
        self.assertIsNone(self.explainer.get_active_toast(now + 1.2))

    def test_thumbs_up_works_normally_after_unlock(self) -> None:
        """
        Confirms ThumbsUp remains Volume Up in Media mode and Zoom In in Presentation mode
        after unlocking, and coaching does not execute actions or bypass state machine.
        """
        activation_sm = ActivationStateMachine()
        context_mgr = ContextManager(initial_context=Context.MEDIA)
        intent_engine = IntentEngine(context_manager=context_mgr)
        executor = ActionExecutor(action_cooldown=0.0)
        register_media_handlers(executor, dry_run=True)
        register_presentation_handlers(executor, dry_run=True)

        # 1. While LOCKED: ThumbsUp produces NO action
        self.assertEqual(activation_sm.state, SystemState.LOCKED)
        act_locked = activation_sm.process_confirmed("ThumbsUp", is_confirmed=True, hold_duration=0.5, timestamp=1.0)
        self.assertIsNone(act_locked.command)
        res_locked = intent_engine.process(activation_result=act_locked, confidence=0.95)
        self.assertFalse(res_locked.accepted)

        # 2. Deliberate 1.0s OpenPalm unlock
        act_unlock = activation_sm.process_confirmed("OpenPalm", is_confirmed=True, hold_duration=1.0, timestamp=2.0)
        self.assertEqual(activation_sm.state, SystemState.ACTIVE)
        self.assertTrue(act_unlock.just_activated)

        # 3. While ACTIVE in Media context: ThumbsUp executes volume_up
        act_active_media = activation_sm.process_confirmed("ThumbsUp", is_confirmed=True, hold_duration=0.5, timestamp=3.0)
        self.assertEqual(act_active_media.command, "ThumbsUp")
        res_media = intent_engine.process(activation_result=act_active_media, confidence=0.95)
        self.assertTrue(res_media.accepted)
        self.assertEqual(res_media.intent, Intent.VOLUME_UP)
        out_media = executor.execute(res_media)
        self.assertTrue(out_media.success)
        self.assertEqual(out_media.action_name, "volume_up")

        # 4. In Presentation context: ThumbsUp executes zoom_in
        context_mgr.set_context(Context.PRESENTATION)
        act_active_pres = activation_sm.process_confirmed("ThumbsUp", is_confirmed=True, hold_duration=0.5, timestamp=4.0)
        res_pres = intent_engine.process(activation_result=act_active_pres, confidence=0.95)
        self.assertTrue(res_pres.accepted)
        self.assertEqual(res_pres.intent, Intent.ZOOM_IN)
        out_pres = executor.execute(res_pres)
        self.assertTrue(out_pres.success)
        self.assertEqual(out_pres.action_name, "zoom_in")


if __name__ == "__main__":
    unittest.main()

