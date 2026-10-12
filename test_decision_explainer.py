"""
test_decision_explainer.py
===========================

Comprehensive unit test suite for IntentSense Phase 9: Explainable Decision Engine.

Verifies strict semantic separation, deterministic explanations, dynamic threshold
formatting, and edge-case telemetry synthesis without camera or OpenCV hardware.
"""

from __future__ import annotations

import unittest

from core.action_executor import ActionReason, ActionResult
from core.activation_state import ActivationResult, SystemState
from core.context_manager import Context
from core.decision_explainer import (
    COLOR_CYAN,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_YELLOW,
    DecisionCategory,
    DecisionExplainer,
    DecisionMessage,
    EventSnapshot,
)
from core.intent_engine import Intent, IntentReason, IntentResult
from core.swipe_detector import SwipeDirection, SwipeResult
from core.temporal_filter import ConfirmationStatus, TemporalFilterResult


class TestDecisionExplainer(unittest.TestCase):
    """Unit test suite for DecisionExplainer."""

    def setUp(self) -> None:
        self.explainer = DecisionExplainer()

    # -------------------------------------------------------------------------
    # 1. Idle and No Hand States
    # -------------------------------------------------------------------------
    def test_idle_no_hand_when_locked(self) -> None:
        msg = self.explainer.explain(
            state=SystemState.LOCKED,
            hand_present=False,
        )
        self.assertEqual(msg.category, DecisionCategory.IDLE)
        self.assertIn("Waiting for hand", msg.text)
        self.assertIn("System locked", msg.text)
        self.assertEqual(msg.color_bgr, COLOR_GRAY)

    def test_idle_no_hand_when_active(self) -> None:
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=False,
        )
        self.assertEqual(msg.category, DecisionCategory.IDLE)
        self.assertEqual(msg.text, "IDLE — Waiting for hand")
        self.assertEqual(msg.color_bgr, COLOR_GRAY)

    def test_idle_hand_present_no_gesture_active(self) -> None:
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture=None,
            confidence=0.0,
        )
        self.assertEqual(msg.category, DecisionCategory.IDLE)
        self.assertEqual(msg.text, "IDLE — Waiting for hand")

    # -------------------------------------------------------------------------
    # 2. Locked State Rejections & Safety Gates
    # -------------------------------------------------------------------------
    def test_locked_swipe_rejected(self) -> None:
        swipe_r = SwipeResult(
            direction=SwipeDirection.RIGHT,
            confidence=0.92,
            velocity=1.5,
            displacement=0.35,
            duration=0.20,
            reason="SWIPE_DETECTED",
        )
        msg = self.explainer.explain(
            state=SystemState.LOCKED,
            hand_present=True,
            swipe_result=swipe_r,
        )
        self.assertEqual(msg.category, DecisionCategory.REJECTED)
        self.assertIn("System locked", msg.text)
        self.assertIn("Swipe RIGHT ignored", msg.text)
        self.assertEqual(msg.color_bgr, COLOR_ORANGE)

    def test_locked_non_activation_gesture_rejected(self) -> None:
        msg = self.explainer.explain(
            state=SystemState.LOCKED,
            hand_present=True,
            raw_gesture="ThumbsUp",
            confidence=0.95,
            effective_threshold=0.80,
        )
        self.assertEqual(msg.category, DecisionCategory.REJECTED)
        self.assertIn("System locked", msg.text)
        self.assertIn("Hold OpenPalm to unlock", msg.text)
        self.assertEqual(msg.color_bgr, COLOR_ORANGE)

    def test_locked_non_activation_gesture_low_confidence(self) -> None:
        msg = self.explainer.explain(
            state=SystemState.LOCKED,
            hand_present=True,
            raw_gesture="ThumbsUp",
            confidence=0.65,
            effective_threshold=0.80,
        )
        self.assertEqual(msg.category, DecisionCategory.REJECTED)
        self.assertIn("Confidence below threshold", msg.text)
        self.assertIn("65% < 80%", msg.text)
        self.assertEqual(msg.color_bgr, COLOR_RED)

    # -------------------------------------------------------------------------
    # 3. Activation Progress & Unlocking
    # -------------------------------------------------------------------------
    def test_open_palm_low_confidence_rejected(self) -> None:
        msg = self.explainer.explain(
            state=SystemState.LOCKED,
            hand_present=True,
            raw_gesture="OpenPalm",
            confidence=0.72,
            effective_threshold=0.80,
        )
        self.assertEqual(msg.category, DecisionCategory.REJECTED)
        self.assertIn("Confidence below threshold", msg.text)
        self.assertIn("72% < 80%", msg.text)
        self.assertEqual(msg.color_bgr, COLOR_RED)

    def test_open_palm_hold_progress(self) -> None:
        filter_res = TemporalFilterResult(
            status=ConfirmationStatus.NOT_CONFIRMED,
            gesture=None,
            confidence=0.91,
            candidate_gesture="OpenPalm",
            consecutive_frames=4,
            cooldown_remaining=0,
            reason="INSUFFICIENT_HOLD_TIME",
            hold_duration=0.45,
        )
        msg = self.explainer.explain(
            state=SystemState.LOCKED,
            hand_present=True,
            raw_gesture="OpenPalm",
            confidence=0.91,
            effective_threshold=0.80,
            filter_result=filter_res,
        )
        self.assertEqual(msg.category, DecisionCategory.WAITING)
        self.assertIn("Hold OpenPalm to unlock (0.5s / 1.0s)", msg.text)
        self.assertEqual(msg.color_bgr, COLOR_YELLOW)

    def test_open_palm_just_activated(self) -> None:
        act_res = ActivationResult(
            state=SystemState.ACTIVE,
            command=None,
            is_active=True,
            just_activated=True,
            just_locked=False,
            time_since_active=0.0,
            time_remaining=30.0,
            activation_progress=1.0,
            reason="ACTIVATED",
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="OpenPalm",
            confidence=0.95,
            activation_result=act_res,
        )
        self.assertEqual(msg.category, DecisionCategory.CONFIRMED)
        self.assertEqual(msg.text, "CONFIRMED — System unlocked!")
        self.assertEqual(msg.color_bgr, COLOR_GREEN)

    def test_activation_just_locked_due_to_inactivity(self) -> None:
        act_res = ActivationResult(
            state=SystemState.LOCKED,
            command=None,
            is_active=False,
            just_activated=False,
            just_locked=True,
            time_since_active=30.0,
            time_remaining=0.0,
            activation_progress=0.0,
            reason="LOCKED_INACTIVITY",
        )
        msg = self.explainer.explain(
            state=SystemState.LOCKED,
            hand_present=True,
            activation_result=act_res,
        )
        self.assertEqual(msg.category, DecisionCategory.REJECTED)
        self.assertEqual(msg.text, "REJECTED — System locked due to inactivity")
        self.assertEqual(msg.color_bgr, COLOR_ORANGE)

    # -------------------------------------------------------------------------
    # 4. Low Confidence & Dynamic Adaptive Thresholds (Never Hard-Coded 80%)
    # -------------------------------------------------------------------------
    def test_low_confidence_exact_baseline_threshold(self) -> None:
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="Pointing",
            confidence=0.64,
            effective_threshold=0.80,
        )
        self.assertEqual(msg.category, DecisionCategory.REJECTED)
        self.assertEqual(msg.text, "REJECTED — Confidence below threshold (64% < 80%)")
        self.assertEqual(msg.color_bgr, COLOR_RED)

    def test_low_confidence_exact_adaptive_sustain_threshold(self) -> None:
        # User is in sustain phase with adaptive threshold 0.75
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="Pointing",
            confidence=0.71,
            effective_threshold=0.75,
        )
        self.assertEqual(msg.category, DecisionCategory.REJECTED)
        self.assertEqual(msg.text, "REJECTED — Confidence below threshold (71% < 75%)")
        self.assertEqual(msg.color_bgr, COLOR_RED)

    def test_low_confidence_exact_custom_threshold(self) -> None:
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="Fist",
            confidence=0.83,
            effective_threshold=0.85,
        )
        self.assertEqual(msg.category, DecisionCategory.REJECTED)
        self.assertEqual(msg.text, "REJECTED — Confidence below threshold (83% < 85%)")
        self.assertEqual(msg.color_bgr, COLOR_RED)

    # -------------------------------------------------------------------------
    # 5. Temporal Stability & Hold Progress
    # -------------------------------------------------------------------------
    def test_temporal_waiting_insufficient_frames(self) -> None:
        filter_res = TemporalFilterResult(
            status=ConfirmationStatus.NOT_CONFIRMED,
            gesture=None,
            confidence=0.88,
            candidate_gesture="ThumbsUp",
            consecutive_frames=2,
            cooldown_remaining=0,
            reason="INSUFFICIENT_FRAMES",
            hold_duration=0.06,
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="ThumbsUp",
            confidence=0.88,
            effective_threshold=0.80,
            filter_result=filter_res,
        )
        self.assertEqual(msg.category, DecisionCategory.WAITING)
        self.assertEqual(msg.text, "WAITING — Stabilizing ThumbsUp (2 frames)")
        self.assertEqual(msg.color_bgr, COLOR_YELLOW)

    def test_temporal_waiting_insufficient_hold_time(self) -> None:
        filter_res = TemporalFilterResult(
            status=ConfirmationStatus.NOT_CONFIRMED,
            gesture=None,
            confidence=0.92,
            candidate_gesture="ThumbsUp",
            consecutive_frames=5,
            cooldown_remaining=0,
            reason="INSUFFICIENT_HOLD_TIME",
            hold_duration=0.32,
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="ThumbsUp",
            confidence=0.92,
            effective_threshold=0.80,
            filter_result=filter_res,
            target_hold=0.5,
        )
        self.assertEqual(msg.category, DecisionCategory.WAITING)
        self.assertEqual(msg.text, "WAITING — Hold ThumbsUp for 0.5s (0.32s)")
        self.assertEqual(msg.color_bgr, COLOR_YELLOW)

    def test_temporal_confirmation_alone_does_not_claim_action(self) -> None:
        filter_res = TemporalFilterResult(
            status=ConfirmationStatus.CONFIRMED,
            gesture="ThumbsUp",
            confidence=0.94,
            candidate_gesture="ThumbsUp",
            consecutive_frames=5,
            cooldown_remaining=0,
            reason="CONFIRMED",
            hold_duration=0.52,
        )
        # Without intent or action result, explains confirmation only
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="ThumbsUp",
            confidence=0.94,
            effective_threshold=0.80,
            filter_result=filter_res,
        )
        self.assertEqual(msg.category, DecisionCategory.CONFIRMED)
        self.assertEqual(msg.text, "CONFIRMED — ThumbsUp stable")
        self.assertEqual(msg.color_bgr, COLOR_GREEN)

    # -------------------------------------------------------------------------
    # 6. Intent Resolution & Action Ready
    # -------------------------------------------------------------------------
    def test_intent_accepted_action_ready(self) -> None:
        intent_res = IntentResult(
            gesture="ThumbsUp",
            intent=Intent.VOLUME_UP,
            context=Context.MEDIA,
            confidence=0.95,
            accepted=True,
            reason=IntentReason.CONFIRMED,
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="ThumbsUp",
            confidence=0.95,
            effective_threshold=0.80,
            intent_result=intent_res,
        )
        self.assertEqual(msg.category, DecisionCategory.ACTION_READY)
        self.assertEqual(msg.text, "ACTION READY — [VOLUME_UP]")
        self.assertEqual(msg.color_bgr, COLOR_CYAN)

    def test_unmapped_gesture_in_context(self) -> None:
        intent_res = IntentResult(
            gesture="Fist",
            intent=None,
            context=Context.MEDIA,
            confidence=0.90,
            accepted=False,
            reason=IntentReason.NO_MAPPING,
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="Fist",
            confidence=0.90,
            intent_result=intent_res,
            context=Context.MEDIA,
        )
        self.assertEqual(msg.category, DecisionCategory.REJECTED)
        self.assertEqual(msg.text, "REJECTED — No mapping for Fist in MEDIA")
        self.assertEqual(msg.color_bgr, COLOR_ORANGE)

    def test_unknown_gesture_rejected(self) -> None:
        intent_res = IntentResult(
            gesture="WaveHand",
            intent=None,
            context=Context.MEDIA,
            confidence=0.90,
            accepted=False,
            reason=IntentReason.UNKNOWN_GESTURE,
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="WaveHand",
            confidence=0.90,
            intent_result=intent_res,
        )
        self.assertEqual(msg.category, DecisionCategory.REJECTED)
        self.assertEqual(msg.text, "REJECTED — Unknown gesture (WaveHand)")
        self.assertEqual(msg.color_bgr, COLOR_ORANGE)

    def test_intent_gesture_not_confirmed(self) -> None:
        intent_res = IntentResult(
            gesture="ThumbsUp",
            intent=None,
            context=Context.MEDIA,
            confidence=0.90,
            accepted=False,
            reason=IntentReason.GESTURE_NOT_CONFIRMED,
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="ThumbsUp",
            confidence=0.90,
            intent_result=intent_res,
        )
        self.assertEqual(msg.category, DecisionCategory.REJECTED)
        self.assertEqual(msg.text, "REJECTED — Gesture not confirmed")
        self.assertEqual(msg.color_bgr, COLOR_ORANGE)

    # -------------------------------------------------------------------------
    # 7. Action Success vs Failure vs Cooldown
    # -------------------------------------------------------------------------
    def test_action_executed_success(self) -> None:
        act_res = ActionResult(
            intent=Intent.VOLUME_UP,
            success=True,
            action_name="volume_up",
            reason=ActionReason.EXECUTED,
            message="volume_up executed successfully.",
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="ThumbsUp",
            confidence=0.95,
            action_result=act_res,
        )
        self.assertEqual(msg.category, DecisionCategory.ACTION_EXECUTED)
        self.assertEqual(msg.text, "ACTION EXECUTED — volume_up")
        self.assertEqual(msg.color_bgr, COLOR_GREEN)

    def test_action_execution_failure(self) -> None:
        act_res = ActionResult(
            intent=Intent.VOLUME_UP,
            success=False,
            action_name="volume_up",
            reason=ActionReason.EXECUTION_FAILED,
            message="Audio device disconnected",
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="ThumbsUp",
            confidence=0.95,
            action_result=act_res,
        )
        self.assertEqual(msg.category, DecisionCategory.ERROR)
        self.assertEqual(msg.text, "ERROR — Action failed (Audio device disconnected)")
        self.assertEqual(msg.color_bgr, COLOR_RED)

    def test_action_debounce_cooldown(self) -> None:
        act_res = ActionResult(
            intent=Intent.VOLUME_UP,
            success=False,
            action_name="volume_up",
            reason=ActionReason.COOLDOWN_ACTIVE,
            message="Action 'volume_up' is in cooldown.",
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="ThumbsUp",
            confidence=0.95,
            action_result=act_res,
        )
        self.assertEqual(msg.category, DecisionCategory.COOLDOWN)
        self.assertEqual(msg.text, "COOLDOWN — Action debounce active (Please wait)")
        self.assertEqual(msg.color_bgr, COLOR_ORANGE)

    def test_action_no_handler_registered(self) -> None:
        intent_res = IntentResult(
            gesture="Peace",
            intent=Intent.ZOOM_IN,
            context=Context.PRESENTATION,
            confidence=0.95,
            accepted=True,
            reason=IntentReason.CONFIRMED,
        )
        act_res = ActionResult(
            intent=Intent.ZOOM_IN,
            success=False,
            action_name=None,
            reason=ActionReason.NO_HANDLER,
            message="No action handler registered for ZOOM_IN.",
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="Peace",
            confidence=0.95,
            intent_result=intent_res,
            action_result=act_res,
        )
        self.assertEqual(msg.category, DecisionCategory.REJECTED)
        self.assertEqual(msg.text, "REJECTED — No handler registered for ZOOM_IN")
        self.assertEqual(msg.color_bgr, COLOR_ORANGE)

    # -------------------------------------------------------------------------
    # 8. Dynamic Swipe Telemetry & Outcomes
    # -------------------------------------------------------------------------
    def test_swipe_detected_ready(self) -> None:
        swipe_r = SwipeResult(
            direction=SwipeDirection.RIGHT,
            confidence=0.94,
            velocity=1.6,
            displacement=0.38,
            duration=0.18,
            reason="SWIPE_DETECTED",
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            swipe_result=swipe_r,
        )
        self.assertEqual(msg.category, DecisionCategory.SWIPE_DETECTED)
        self.assertEqual(msg.text, "SWIPE DETECTED — RIGHT")
        self.assertEqual(msg.color_bgr, COLOR_GREEN)

    def test_swipe_action_ready(self) -> None:
        swipe_r = SwipeResult(
            direction=SwipeDirection.RIGHT,
            confidence=0.94,
            velocity=1.6,
            reason="SWIPE_DETECTED",
        )
        intent_sr = IntentResult(
            gesture=None,
            intent=Intent.NEXT,
            context=Context.MEDIA,
            confidence=0.94,
            accepted=True,
            reason=IntentReason.CONFIRMED,
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            swipe_result=swipe_r,
            intent_result=intent_sr,
        )
        self.assertEqual(msg.category, DecisionCategory.ACTION_READY)
        self.assertEqual(msg.text, "ACTION READY — [NEXT] via Swipe RIGHT")
        self.assertEqual(msg.color_bgr, COLOR_CYAN)

    def test_swipe_action_executed(self) -> None:
        swipe_r = SwipeResult(
            direction=SwipeDirection.RIGHT,
            confidence=0.94,
            velocity=1.6,
            reason="SWIPE_DETECTED",
        )
        act_r = ActionResult(
            intent=Intent.NEXT,
            success=True,
            action_name="next_track",
            reason=ActionReason.EXECUTED,
            message="next_track executed.",
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            swipe_result=swipe_r,
            action_result=act_r,
        )
        self.assertEqual(msg.category, DecisionCategory.ACTION_EXECUTED)
        self.assertEqual(msg.text, "ACTION EXECUTED — next_track (Swipe RIGHT)")
        self.assertEqual(msg.color_bgr, COLOR_GREEN)

    def test_swipe_action_debounce_cooldown(self) -> None:
        swipe_l = SwipeResult(
            direction=SwipeDirection.LEFT,
            confidence=0.91,
            velocity=1.4,
            reason="SWIPE_DETECTED",
        )
        act_l = ActionResult(
            intent=Intent.PREVIOUS,
            success=False,
            action_name="previous_track",
            reason=ActionReason.COOLDOWN_ACTIVE,
            message="In cooldown.",
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            swipe_result=swipe_l,
            action_result=act_l,
        )
        self.assertEqual(msg.category, DecisionCategory.COOLDOWN)
        self.assertEqual(msg.text, "COOLDOWN — Action debounce active for Swipe LEFT")
        self.assertEqual(msg.color_bgr, COLOR_ORANGE)

    def test_swipe_detector_in_cooldown(self) -> None:
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            swipe_detector_cooldown=True,
        )
        self.assertEqual(msg.category, DecisionCategory.COOLDOWN)
        self.assertEqual(msg.text, "COOLDOWN — Swipe cooldown active (Please wait)")
        self.assertEqual(msg.color_bgr, COLOR_ORANGE)

    def test_swipe_unmapped_in_custom_context(self) -> None:
        swipe_r = SwipeResult(
            direction=SwipeDirection.RIGHT,
            confidence=0.90,
            velocity=1.5,
            reason="SWIPE_DETECTED",
        )
        intent_sr = IntentResult(
            gesture=None,
            intent=None,
            context=Context.CUSTOM,
            confidence=0.90,
            accepted=False,
            reason=IntentReason.NO_MAPPING,
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            swipe_result=swipe_r,
            intent_result=intent_sr,
            context=Context.CUSTOM,
        )
        self.assertEqual(msg.category, DecisionCategory.REJECTED)
        self.assertEqual(msg.text, "REJECTED — No mapping for Swipe RIGHT in CUSTOM")
        self.assertEqual(msg.color_bgr, COLOR_ORANGE)

    # -------------------------------------------------------------------------
    # 9. Temporal Filter Cooldown
    # -------------------------------------------------------------------------
    def test_temporal_filter_in_cooldown(self) -> None:
        filter_res = TemporalFilterResult(
            status=ConfirmationStatus.NOT_CONFIRMED,
            gesture=None,
            confidence=0.95,
            candidate_gesture="ThumbsUp",
            consecutive_frames=0,
            cooldown_remaining=10,
            reason="IN_COOLDOWN",
            hold_duration=0.0,
        )
        msg = self.explainer.explain(
            state=SystemState.ACTIVE,
            hand_present=True,
            raw_gesture="ThumbsUp",
            confidence=0.95,
            effective_threshold=0.80,
            filter_result=filter_res,
        )
        self.assertEqual(msg.category, DecisionCategory.COOLDOWN)
        self.assertEqual(msg.text, "COOLDOWN — ThumbsUp cooldown active (10 frames remaining)")
        self.assertEqual(msg.color_bgr, COLOR_ORANGE)

    # -------------------------------------------------------------------------
    # 10. Immutability and Pure Python Behavior
    # -------------------------------------------------------------------------
    def test_decision_message_dataclass_is_frozen(self) -> None:
        msg = DecisionMessage(category=DecisionCategory.IDLE, text="Idle test")
        with self.assertRaises((TypeError, AttributeError)):
            msg.text = "Changed"  # type: ignore

    def test_str_representations(self) -> None:
        msg = DecisionMessage(category=DecisionCategory.WAITING, text="WAITING — Hold gesture")
        self.assertEqual(str(msg), "WAITING — Hold gesture")
        self.assertEqual(str(DecisionCategory.ACTION_READY), "ACTION READY")


    # -------------------------------------------------------------------------
    # 11. Display Event Persistence & Classification
    # -------------------------------------------------------------------------
    def test_is_important_event_classification(self) -> None:
        """Verify that all 5 critical events are classified as important."""
        # 1. ACTION EXECUTED
        msg_act = DecisionMessage(category=DecisionCategory.ACTION_EXECUTED, text="ACTION EXECUTED — volume_up")
        self.assertTrue(DecisionExplainer.is_important_event(msg_act, action_executed=True))
        self.assertTrue(DecisionExplainer.is_important_event(msg_act, action_executed=False))

        # 2. SWIPE DETECTED
        msg_swipe = DecisionMessage(category=DecisionCategory.SWIPE_DETECTED, text="SWIPE DETECTED — RIGHT")
        self.assertTrue(DecisionExplainer.is_important_event(msg_swipe, is_swipe=True))
        self.assertTrue(DecisionExplainer.is_important_event(msg_swipe, is_swipe=False))

        # 3. SYSTEM ACTIVATED
        msg_unlock = DecisionMessage(category=DecisionCategory.CONFIRMED, text="CONFIRMED — System unlocked!")
        self.assertTrue(DecisionExplainer.is_important_event(msg_unlock, just_activated=True))
        self.assertTrue(DecisionExplainer.is_important_event(msg_unlock, just_activated=False))

        # 4. SYSTEM LOCKED
        msg_inact = DecisionMessage(category=DecisionCategory.REJECTED, text="REJECTED — System locked due to inactivity")
        self.assertTrue(DecisionExplainer.is_important_event(msg_inact, just_locked=True))
        self.assertTrue(DecisionExplainer.is_important_event(msg_inact, just_locked=False))

        msg_man = DecisionMessage(category=DecisionCategory.REJECTED, text="REJECTED — System manually locked")
        self.assertTrue(DecisionExplainer.is_important_event(msg_man, just_locked=True))
        self.assertTrue(DecisionExplainer.is_important_event(msg_man, just_locked=False))

        # 5. ACTION FAILED
        msg_err = DecisionMessage(category=DecisionCategory.ERROR, text="ERROR — Action failed (Simulated)")
        self.assertTrue(DecisionExplainer.is_important_event(msg_err, action_failed=True))
        self.assertTrue(DecisionExplainer.is_important_event(msg_err, action_failed=False))

    def test_is_important_event_negative_cases(self) -> None:
        """Verify that routine telemetry states do not trigger persistent event display."""
        # IDLE
        msg_idle = DecisionMessage(category=DecisionCategory.IDLE, text="IDLE — Waiting for hand")
        self.assertFalse(DecisionExplainer.is_important_event(msg_idle))

        # WAITING (stabilizing, hold progress, tracking)
        msg_wait1 = DecisionMessage(category=DecisionCategory.WAITING, text="WAITING — Tracking Pointing")
        self.assertFalse(DecisionExplainer.is_important_event(msg_wait1))

        msg_wait2 = DecisionMessage(category=DecisionCategory.WAITING, text="WAITING — Stabilizing Pointing (3 frames)")
        self.assertFalse(DecisionExplainer.is_important_event(msg_wait2))

        msg_wait3 = DecisionMessage(category=DecisionCategory.WAITING, text="WAITING — Hold Pointing for 0.5s (0.25s)")
        self.assertFalse(DecisionExplainer.is_important_event(msg_wait3))

        # COOLDOWN
        msg_cd = DecisionMessage(category=DecisionCategory.COOLDOWN, text="COOLDOWN — Action debounce active (Please wait)")
        self.assertFalse(DecisionExplainer.is_important_event(msg_cd))

        # REJECTED low confidence
        msg_low = DecisionMessage(category=DecisionCategory.REJECTED, text="REJECTED — Confidence below threshold (60% < 80%)")
        self.assertFalse(DecisionExplainer.is_important_event(msg_low))

        # REJECTED locked normal gesture
        msg_lock = DecisionMessage(category=DecisionCategory.REJECTED, text="REJECTED — System locked (Hold OpenPalm to unlock)")
        self.assertFalse(DecisionExplainer.is_important_event(msg_lock))

    def test_resolve_display_decision_persists_within_duration(self) -> None:
        """Verify that an important event persists while elapsed time is strictly less than duration."""
        last_ev = DecisionMessage(category=DecisionCategory.ACTION_EXECUTED, text="ACTION EXECUTED — volume_up")
        live_msg = DecisionMessage(category=DecisionCategory.WAITING, text="WAITING — Tracking ThumbsUp")

        t0 = 100.0
        # Immediately after event
        out0 = DecisionExplainer.resolve_display_decision(
            live_decision=live_msg,
            last_event=last_ev,
            last_event_timestamp=t0,
            current_timestamp=100.0,
            display_duration=2.0,
        )
        self.assertEqual(out0, last_ev)

        # 1.0 second into the 2.0s window
        out1 = DecisionExplainer.resolve_display_decision(
            live_decision=live_msg,
            last_event=last_ev,
            last_event_timestamp=t0,
            current_timestamp=101.0,
            display_duration=2.0,
        )
        self.assertEqual(out1, last_ev)

        # 1.99 seconds into the 2.0s window
        out_edge = DecisionExplainer.resolve_display_decision(
            live_decision=live_msg,
            last_event=last_ev,
            last_event_timestamp=t0,
            current_timestamp=101.99,
            display_duration=2.0,
        )
        self.assertEqual(out_edge, last_ev)

    def test_resolve_display_decision_reverts_after_expiration(self) -> None:
        """Verify that after the display duration expires, display reverts to live decision."""
        last_ev = DecisionMessage(category=DecisionCategory.ACTION_EXECUTED, text="ACTION EXECUTED — volume_up")
        live_msg = DecisionMessage(category=DecisionCategory.WAITING, text="WAITING — Tracking ThumbsUp")

        t0 = 100.0
        # Exactly at expiration (2.0s elapsed)
        out_exp = DecisionExplainer.resolve_display_decision(
            live_decision=live_msg,
            last_event=last_ev,
            last_event_timestamp=t0,
            current_timestamp=102.0,
            display_duration=2.0,
        )
        self.assertEqual(out_exp, live_msg)

        # Past expiration (3.5s elapsed)
        out_past = DecisionExplainer.resolve_display_decision(
            live_decision=live_msg,
            last_event=last_ev,
            last_event_timestamp=t0,
            current_timestamp=103.5,
            display_duration=2.0,
        )
        self.assertEqual(out_past, live_msg)

    def test_resolve_display_decision_no_prior_event(self) -> None:
        """Verify that when last_event is None, live decision is returned immediately."""
        live_msg = DecisionMessage(category=DecisionCategory.IDLE, text="IDLE — Waiting for hand")
        out = DecisionExplainer.resolve_display_decision(
            live_decision=live_msg,
            last_event=None,
            last_event_timestamp=0.0,
            current_timestamp=50.0,
            display_duration=2.0,
        )
        self.assertEqual(out, live_msg)

    def test_resolve_display_decision_sequential_events(self) -> None:
        """Verify that a second important event replaces the first and resets duration."""
        ev1 = DecisionMessage(category=DecisionCategory.CONFIRMED, text="CONFIRMED — System unlocked!")
        ev2 = DecisionMessage(category=DecisionCategory.ACTION_EXECUTED, text="ACTION EXECUTED — next_track (Swipe RIGHT)")
        live_msg = DecisionMessage(category=DecisionCategory.IDLE, text="IDLE — Ready for gesture or swipe")

        # Event 1 occurs at t=10.0
        t1 = 10.0
        out1 = DecisionExplainer.resolve_display_decision(live_msg, ev1, t1, current_timestamp=10.5, display_duration=2.0)
        self.assertEqual(out1, ev1)

        # Event 2 occurs at t=11.2 (superseding event 1)
        t2 = 11.2
        out2 = DecisionExplainer.resolve_display_decision(live_msg, ev2, t2, current_timestamp=11.5, display_duration=2.0)
        self.assertEqual(out2, ev2)

        # At t=12.5 (1.3s after Event 2, but 2.5s after Event 1), Event 2 is still active
        out3 = DecisionExplainer.resolve_display_decision(live_msg, ev2, t2, current_timestamp=12.5, display_duration=2.0)
        self.assertEqual(out3, ev2)

        # At t=13.3 (2.1s after Event 2), display reverts to live_msg
        out4 = DecisionExplainer.resolve_display_decision(live_msg, ev2, t2, current_timestamp=13.3, display_duration=2.0)
        self.assertEqual(out4, live_msg)

    # -------------------------------------------------------------------------
    # 12. Synchronized HUD EventSnapshot Tests
    # -------------------------------------------------------------------------
    def test_event_snapshot_dataclass_is_frozen(self) -> None:
        """Verify that EventSnapshot is immutable."""
        dec = DecisionMessage(category=DecisionCategory.ACTION_EXECUTED, text="ACTION EXECUTED — volume_up")
        snap = EventSnapshot(decision=dec, intent=Intent.VOLUME_UP, action_name="volume_up", action_status="EXECUTED")
        with self.assertRaises((TypeError, AttributeError)):
            snap.action_name = "other"  # type: ignore

    def test_resolve_hud_snapshot_static_action(self) -> None:
        """Verify that a static action event snapshot retains intent, action, status, and decision."""
        dec = DecisionMessage(category=DecisionCategory.ACTION_EXECUTED, text="ACTION EXECUTED — volume_up", color_bgr=COLOR_GREEN)
        snap = EventSnapshot(decision=dec, intent=Intent.VOLUME_UP, action_name="volume_up", action_status="EXECUTED")

        live_dec = DecisionMessage(category=DecisionCategory.WAITING, text="WAITING — Tracking ThumbsUp")
        live_intent = None
        live_action = None
        live_status = "GESTURE_NOT_CONFIRMED"

        t0 = 50.0

        # Within 2.0s persistence window
        d, i, a, s = DecisionExplainer.resolve_hud_snapshot(
            live_decision=live_dec,
            live_intent=live_intent,
            live_action_name=live_action,
            live_action_status=live_status,
            last_event=snap,
            last_event_timestamp=t0,
            current_timestamp=51.5,
            display_duration=2.0,
        )
        self.assertEqual(d, dec)
        self.assertEqual(i, Intent.VOLUME_UP)
        self.assertEqual(a, "volume_up")
        self.assertEqual(s, "EXECUTED")

        # After 2.0s expiration: all four fields revert together to live values
        d_exp, i_exp, a_exp, s_exp = DecisionExplainer.resolve_hud_snapshot(
            live_decision=live_dec,
            live_intent=live_intent,
            live_action_name=live_action,
            live_action_status=live_status,
            last_event=snap,
            last_event_timestamp=t0,
            current_timestamp=52.1,
            display_duration=2.0,
        )
        self.assertEqual(d_exp, live_dec)
        self.assertIsNone(i_exp)
        self.assertIsNone(a_exp)
        self.assertEqual(s_exp, "GESTURE_NOT_CONFIRMED")

    def test_resolve_hud_snapshot_swipe_event(self) -> None:
        """Verify that a swipe event snapshot retains swipe intent, action name, status, and decision."""
        dec_swipe = DecisionMessage(category=DecisionCategory.ACTION_EXECUTED, text="ACTION EXECUTED — next_track (Swipe RIGHT)", color_bgr=COLOR_GREEN)
        snap_swipe = EventSnapshot(decision=dec_swipe, intent=Intent.NEXT, action_name="next_track", action_status="EXECUTED")

        live_dec = DecisionMessage(category=DecisionCategory.IDLE, text="IDLE — Ready for gesture or swipe")
        t0 = 100.0

        d, i, a, s = DecisionExplainer.resolve_hud_snapshot(
            live_decision=live_dec,
            live_intent=None,
            live_action_name=None,
            live_action_status="ACTIVE_IDLE",
            last_event=snap_swipe,
            last_event_timestamp=t0,
            current_timestamp=101.2,
            display_duration=2.0,
        )
        self.assertEqual(d, dec_swipe)
        self.assertEqual(i, Intent.NEXT)
        self.assertEqual(a, "next_track")
        self.assertEqual(s, "EXECUTED")

    def test_resolve_hud_snapshot_newer_event_replacement(self) -> None:
        """Verify that a newer event immediately replaces the previous snapshot."""
        # Event 1: System activated at t=10.0
        dec_unlock = DecisionMessage(category=DecisionCategory.CONFIRMED, text="CONFIRMED — System unlocked!", color_bgr=COLOR_GREEN)
        snap1 = EventSnapshot(decision=dec_unlock, intent=None, action_name=None, action_status="ACTIVATED")

        # Event 2: Swipe right executed at t=11.0
        dec_swipe = DecisionMessage(category=DecisionCategory.ACTION_EXECUTED, text="ACTION EXECUTED — next_track (Swipe RIGHT)", color_bgr=COLOR_GREEN)
        snap2 = EventSnapshot(decision=dec_swipe, intent=Intent.NEXT, action_name="next_track", action_status="EXECUTED")

        live_dec = DecisionMessage(category=DecisionCategory.IDLE, text="IDLE — Ready for gesture or swipe")

        # At t=10.5, Event 1 is displayed
        d1, i1, a1, s1 = DecisionExplainer.resolve_hud_snapshot(
            live_dec, None, None, "ACTIVE_IDLE", snap1, 10.0, current_timestamp=10.5, display_duration=2.0
        )
        self.assertEqual(d1, dec_unlock)
        self.assertIsNone(i1)
        self.assertIsNone(a1)
        self.assertEqual(s1, "ACTIVATED")

        # At t=11.2 (Event 2 has replaced Event 1)
        d2, i2, a2, s2 = DecisionExplainer.resolve_hud_snapshot(
            live_dec, None, None, "ACTIVE_IDLE", snap2, 11.0, current_timestamp=11.2, display_duration=2.0
        )
        self.assertEqual(d2, dec_swipe)
        self.assertEqual(i2, Intent.NEXT)
        self.assertEqual(a2, "next_track")
        self.assertEqual(s2, "EXECUTED")

        # At t=12.5 (2.5s after Event 1, but 1.5s after Event 2): Event 2 persists
        d3, i3, a3, s3 = DecisionExplainer.resolve_hud_snapshot(
            live_dec, None, None, "ACTIVE_IDLE", snap2, 11.0, current_timestamp=12.5, display_duration=2.0
        )
        self.assertEqual(d3, dec_swipe)
        self.assertEqual(i3, Intent.NEXT)

        # At t=13.2 (2.2s after Event 2): reverts to live
        d4, i4, a4, s4 = DecisionExplainer.resolve_hud_snapshot(
            live_dec, None, None, "ACTIVE_IDLE", snap2, 11.0, current_timestamp=13.2, display_duration=2.0
        )
        self.assertEqual(d4, live_dec)
        self.assertIsNone(i4)
        self.assertIsNone(a4)
        self.assertEqual(s4, "ACTIVE_IDLE")

    def test_resolve_hud_snapshot_no_prior_event_or_reset(self) -> None:
        """Verify that when last_event is None, live values are returned."""
        live_dec = DecisionMessage(category=DecisionCategory.IDLE, text="IDLE — Waiting for hand")
        d, i, a, s = DecisionExplainer.resolve_hud_snapshot(
            live_decision=live_dec,
            live_intent=None,
            live_action_name=None,
            live_action_status="LOCKED",
            last_event=None,
            last_event_timestamp=0.0,
            current_timestamp=20.0,
            display_duration=2.0,
        )
        self.assertEqual(d, live_dec)
        self.assertIsNone(i)
        self.assertIsNone(a)
        self.assertEqual(s, "LOCKED")


if __name__ == "__main__":
    unittest.main()


