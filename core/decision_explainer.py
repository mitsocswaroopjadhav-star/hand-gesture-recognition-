"""
core/decision_explainer.py
==========================

Phase 9: Explainable Decision Engine for IntentSense.

Synthesizes multi-stage pipeline telemetry:
  Perception (Raw Inference & Confidence)
      │
  Temporal Stability Filter (Streak, Hold Duration, Filter Cooldown)
      │
  Safety State Machine (LOCKED vs ACTIVE, Activation Hold Progress)
      │
  Context Management (MEDIA, PRESENTATION, CUSTOM)
      │
  Semantic Intent Engine (Intent Mapping, Rejection Reasons)
      │
  Action Executor (Execution Success, Debounce Cooldown, Handlers)
      │
  Dynamic Swipe Detector (Ballistic Directions, Trajectory Cooldown)

Emits a single, prioritized, human-readable DecisionMessage explaining
EXACTLY why a gesture or swipe is:
  - WAITING (confirmation in progress: frames or hold duration)
  - CONFIRMED (temporal confirmation achieved)
  - ACTION READY (intent accepted by IntentEngine, awaiting execution)
  - ACTION EXECUTED (action successfully run by ActionExecutor)
  - COOLDOWN (blocked by temporal, swipe, or debounce cooldown)
  - REJECTED (blocked by safety lock, low confidence, or unmapped context)
  - IDLE (no hand detected or waiting for input)

Strict Semantic Guarantees:
1. Never displays a success message based solely on recognition or confirmation.
2. Uses actual runtime confidence and effective threshold values (no hard-coded 80%).
3. Pure Python with zero external dependencies (no OpenCV or camera hardware required).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from core.action_executor import ActionReason, ActionResult
from core.activation_state import ActivationResult, SystemState
from core.context_manager import Context
from core.intent_engine import Intent, IntentReason, IntentResult
from core.swipe_detector import SwipeDirection, SwipeResult
from core.temporal_filter import ConfirmationStatus, TemporalFilterResult


class DecisionCategory(str, Enum):
    """Semantic category of decision explanation."""
    IDLE = "IDLE"
    WAITING = "WAITING"
    CONFIRMED = "CONFIRMED"
    ACTION_READY = "ACTION READY"
    ACTION_EXECUTED = "ACTION EXECUTED"
    COOLDOWN = "COOLDOWN"
    REJECTED = "REJECTED"
    SWIPE_DETECTED = "SWIPE DETECTED"
    ERROR = "ERROR"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class DecisionMessage:
    """
    Structured outcome of the explainability engine for a single frame.

    Attributes:
        category: Semantic category (DecisionCategory).
        text: Clear human-readable description of the decision.
        color_bgr: Semantic BGR color for UI rendering.
    """
    category: DecisionCategory
    text: str
    color_bgr: Tuple[int, int, int] = (180, 180, 180)

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class EventSnapshot:
    """
    Consistent multi-field snapshot of an important pipeline event for HUD presentation.

    Attributes:
        decision: The DecisionMessage explaining the event.
        intent: The Intent associated with the event (or None).
        action_name: The executed or attempted action name (or None).
        action_status: The status string associated with the action/event.
    """
    decision: DecisionMessage
    intent: Optional[Intent] = None
    action_name: Optional[str] = None
    action_status: str = "IDLE"


# Semantic BGR Color Palette
COLOR_GREEN: Tuple[int, int, int] = (0, 255, 0)      # Confirmed / Executed / Swiped
COLOR_CYAN: Tuple[int, int, int] = (255, 255, 0)     # Action Ready
COLOR_YELLOW: Tuple[int, int, int] = (0, 220, 255)   # Waiting / Progress
COLOR_ORANGE: Tuple[int, int, int] = (0, 165, 255)   # Cooldown / Locked rejection
COLOR_RED: Tuple[int, int, int] = (100, 100, 255)    # Error / Low confidence rejection
COLOR_GRAY: Tuple[int, int, int] = (180, 180, 180)   # Idle / Neutral


class DecisionExplainer:
    """
    Deterministic rule-based decision explanation engine.
    """

    def explain(
        self,
        state: SystemState,
        hand_present: bool = False,
        raw_gesture: Optional[str] = None,
        confidence: float = 0.0,
        effective_threshold: float = 0.80,
        filter_result: Optional[TemporalFilterResult] = None,
        activation_result: Optional[ActivationResult] = None,
        intent_result: Optional[IntentResult] = None,
        action_result: Optional[ActionResult] = None,
        swipe_result: Optional[SwipeResult] = None,
        swipe_detector_cooldown: bool = False,
        context: Optional[Context] = None,
        target_hold: float = 0.5,
        arbitration_reason: Optional[str] = None,
    ) -> DecisionMessage:
        """
        Synthesizes pipeline telemetry into a single, prioritized DecisionMessage.

        Args:
            state: Current SystemState (LOCKED or ACTIVE).
            hand_present: Whether hand landmarks are currently tracked.
            raw_gesture: Predicted gesture label from classifier (or None).
            confidence: Model confidence score for the frame [0.0, 1.0].
            effective_threshold: Effective confidence threshold (adaptive or baseline).
            filter_result: Optional TemporalFilterResult from the active temporal filter.
            activation_result: Optional ActivationResult from the activation state machine.
            intent_result: Optional IntentResult from the semantic intent engine.
            action_result: Optional ActionResult from the action executor.
            swipe_result: Optional SwipeResult from the dynamic swipe detector.
            swipe_detector_cooldown: Whether the swipe detector is currently in cooldown.
            context: Current operating Context (MEDIA, PRESENTATION, CUSTOM).
            target_hold: Configured target hold duration in seconds (0.5s or 1.0s).

        Returns:
            DecisionMessage with category, explanation text, and BGR color.
        """
        conf_pct = int(confidence * 100)
        thresh_pct = int(effective_threshold * 100)
        is_swipe = (
            swipe_result is not None
            and swipe_result.direction in (SwipeDirection.LEFT, SwipeDirection.RIGHT)
        )

        # ---------------------------------------------------------------------
        # Priority 1: Activation State Machine Transitions
        # ---------------------------------------------------------------------
        if activation_result is not None:
            if activation_result.just_activated:
                return DecisionMessage(
                    category=DecisionCategory.CONFIRMED,
                    text="CONFIRMED — System unlocked!",
                    color_bgr=COLOR_GREEN,
                )
            if activation_result.just_locked:
                return DecisionMessage(
                    category=DecisionCategory.REJECTED,
                    text="REJECTED — System locked due to inactivity",
                    color_bgr=COLOR_ORANGE,
                )

        # ---------------------------------------------------------------------
        # Priority 1.5: Hand Not Detected / Empty Input
        # ---------------------------------------------------------------------
        if not hand_present or (
            raw_gesture is None
            and (filter_result is None or filter_result.candidate_gesture is None)
            and not is_swipe
            and not swipe_detector_cooldown
            and intent_result is None
            and action_result is None
        ):
            if state == SystemState.LOCKED:
                return DecisionMessage(
                    category=DecisionCategory.IDLE,
                    text="IDLE — Waiting for hand (System locked)",
                    color_bgr=COLOR_GRAY,
                )
            return DecisionMessage(
                category=DecisionCategory.IDLE,
                text="IDLE — Waiting for hand",
                color_bgr=COLOR_GRAY,
            )

        # ---------------------------------------------------------------------
        # Priority 2: System is LOCKED (Safety Gate Enforced)
        # ---------------------------------------------------------------------
        if state == SystemState.LOCKED:
            # 2a. Swipes are rejected while locked
            if is_swipe and swipe_result is not None:
                return DecisionMessage(
                    category=DecisionCategory.REJECTED,
                    text=f"REJECTED — System locked (Swipe {swipe_result.direction.value} ignored)",
                    color_bgr=COLOR_ORANGE,
                )

            # 2b. OpenPalm activation handling
            if raw_gesture == "OpenPalm":
                if confidence < effective_threshold or (filter_result is not None and filter_result.reason == "LOW_CONFIDENCE"):
                    return DecisionMessage(
                        category=DecisionCategory.REJECTED,
                        text=f"REJECTED — Confidence below threshold ({conf_pct}% < {thresh_pct}%)",
                        color_bgr=COLOR_RED,
                    )

                # Accumulating hold duration towards 1.0s unlock requirement
                hold_dur = filter_result.hold_duration if filter_result else 0.0
                return DecisionMessage(
                    category=DecisionCategory.WAITING,
                    text=f"WAITING — Hold OpenPalm to unlock ({hold_dur:.1f}s / 1.0s)",
                    color_bgr=COLOR_YELLOW,
                )

            # 2c. Non-activation gestures while locked
            if raw_gesture is not None:
                if confidence < effective_threshold or (filter_result is not None and filter_result.reason == "LOW_CONFIDENCE"):
                    return DecisionMessage(
                        category=DecisionCategory.REJECTED,
                        text=f"REJECTED — Confidence below threshold ({conf_pct}% < {thresh_pct}%)",
                        color_bgr=COLOR_RED,
                    )
                return DecisionMessage(
                    category=DecisionCategory.REJECTED,
                    text="REJECTED — System locked (Hold OpenPalm to unlock)",
                    color_bgr=COLOR_ORANGE,
                )

        # ---------------------------------------------------------------------
        # System is ACTIVE: Priority 3 — Dynamic Swipe Event
        # ---------------------------------------------------------------------
        if is_swipe and swipe_result is not None:
            dir_str = swipe_result.direction.value
            if action_result is not None:
                if action_result.reason == ActionReason.EXECUTED:
                    return DecisionMessage(
                        category=DecisionCategory.ACTION_EXECUTED,
                        text=f"ACTION EXECUTED — {action_result.action_name} (Swipe {dir_str})",
                        color_bgr=COLOR_GREEN,
                    )
                if action_result.reason == ActionReason.COOLDOWN_ACTIVE:
                    return DecisionMessage(
                        category=DecisionCategory.COOLDOWN,
                        text=f"COOLDOWN — Action debounce active for Swipe {dir_str}",
                        color_bgr=COLOR_ORANGE,
                    )
                if action_result.reason == ActionReason.EXECUTION_FAILED:
                    return DecisionMessage(
                        category=DecisionCategory.ERROR,
                        text=f"ERROR — Swipe action failed ({action_result.message})",
                        color_bgr=COLOR_RED,
                    )

            if intent_result is not None:
                if intent_result.accepted and intent_result.intent:
                    return DecisionMessage(
                        category=DecisionCategory.ACTION_READY,
                        text=f"ACTION READY — [{intent_result.intent.value}] via Swipe {dir_str}",
                        color_bgr=COLOR_CYAN,
                    )
                if intent_result.reason == IntentReason.NO_MAPPING:
                    ctx_str = context.value if context else "Current"
                    return DecisionMessage(
                        category=DecisionCategory.REJECTED,
                        text=f"REJECTED — No mapping for Swipe {dir_str} in {ctx_str}",
                        color_bgr=COLOR_ORANGE,
                    )

            return DecisionMessage(
                category=DecisionCategory.SWIPE_DETECTED,
                text=f"SWIPE DETECTED — {dir_str}",
                color_bgr=COLOR_GREEN,
            )

        # ---------------------------------------------------------------------
        # Priority 4: Action Execution Outcomes
        # ---------------------------------------------------------------------
        if action_result is not None:
            if action_result.reason == ActionReason.EXECUTED:
                return DecisionMessage(
                    category=DecisionCategory.ACTION_EXECUTED,
                    text=f"ACTION EXECUTED — {action_result.action_name}",
                    color_bgr=COLOR_GREEN,
                )
            if action_result.reason == ActionReason.COOLDOWN_ACTIVE:
                return DecisionMessage(
                    category=DecisionCategory.COOLDOWN,
                    text="COOLDOWN — Action debounce active (Please wait)",
                    color_bgr=COLOR_ORANGE,
                )
            if action_result.reason == ActionReason.EXECUTION_FAILED:
                return DecisionMessage(
                    category=DecisionCategory.ERROR,
                    text=f"ERROR — Action failed ({action_result.message})",
                    color_bgr=COLOR_RED,
                )
            if action_result.reason == ActionReason.NO_HANDLER:
                intent_name = (
                    intent_result.intent.value
                    if intent_result and intent_result.intent
                    else "Intent"
                )
                return DecisionMessage(
                    category=DecisionCategory.REJECTED,
                    text=f"REJECTED — No handler registered for {intent_name}",
                    color_bgr=COLOR_ORANGE,
                )

        # ---------------------------------------------------------------------
        # Priority 5: Intent Confirmation & Action Ready
        # ---------------------------------------------------------------------
        if intent_result is not None:
            if intent_result.accepted and intent_result.intent:
                return DecisionMessage(
                    category=DecisionCategory.ACTION_READY,
                    text=f"ACTION READY — [{intent_result.intent.value}]",
                    color_bgr=COLOR_CYAN,
                )
            if intent_result.reason == IntentReason.SYSTEM_LOCKED:
                return DecisionMessage(
                    category=DecisionCategory.REJECTED,
                    text="REJECTED — System locked (Hold OpenPalm to unlock)",
                    color_bgr=COLOR_ORANGE,
                )
            if intent_result.reason == IntentReason.GESTURE_NOT_CONFIRMED:
                return DecisionMessage(
                    category=DecisionCategory.REJECTED,
                    text="REJECTED — Gesture not confirmed",
                    color_bgr=COLOR_ORANGE,
                )
            if intent_result.reason == IntentReason.NO_MAPPING:
                ctx_str = context.value if context else "Current"
                g_str = intent_result.gesture or raw_gesture or "Gesture"
                return DecisionMessage(
                    category=DecisionCategory.REJECTED,
                    text=f"REJECTED — No mapping for {g_str} in {ctx_str}",
                    color_bgr=COLOR_ORANGE,
                )
            if intent_result.reason == IntentReason.UNKNOWN_GESTURE:
                return DecisionMessage(
                    category=DecisionCategory.REJECTED,
                    text=f"REJECTED — Unknown gesture ({intent_result.gesture})",
                    color_bgr=COLOR_ORANGE,
                )

        # ---------------------------------------------------------------------
        # Priority 6: Temporal Filter Confirmation (Pre-Intent Resolution)
        # ---------------------------------------------------------------------
        if filter_result is not None and filter_result.is_confirmed:
            return DecisionMessage(
                category=DecisionCategory.CONFIRMED,
                text=f"CONFIRMED — {filter_result.gesture} stable",
                color_bgr=COLOR_GREEN,
            )

        # ---------------------------------------------------------------------
        # Priority 7: Cooldown Suppression
        # ---------------------------------------------------------------------
        if filter_result is not None and filter_result.reason == "IN_COOLDOWN":
            cd_rem = filter_result.cooldown_remaining
            cand = filter_result.candidate_gesture or "Gesture"
            return DecisionMessage(
                category=DecisionCategory.COOLDOWN,
                text=f"COOLDOWN — {cand} cooldown active ({cd_rem} frames remaining)",
                color_bgr=COLOR_ORANGE,
            )

        if swipe_detector_cooldown:
            return DecisionMessage(
                category=DecisionCategory.COOLDOWN,
                text="COOLDOWN — Swipe cooldown active (Please wait)",
                color_bgr=COLOR_ORANGE,
            )

        # ---------------------------------------------------------------------
        # Priority 8: Temporal Filter Progress / Waiting
        # ---------------------------------------------------------------------
        if filter_result is not None and filter_result.candidate_gesture is not None:
            cand = filter_result.candidate_gesture

            if confidence < effective_threshold:
                return DecisionMessage(
                    category=DecisionCategory.REJECTED,
                    text=f"REJECTED — Confidence below threshold ({conf_pct}% < {thresh_pct}%)",
                    color_bgr=COLOR_RED,
                )

            if filter_result.reason == "INSUFFICIENT_FRAMES":
                frames = filter_result.consecutive_frames
                return DecisionMessage(
                    category=DecisionCategory.WAITING,
                    text=f"WAITING — Stabilizing {cand} ({frames} frames)",
                    color_bgr=COLOR_YELLOW,
                )

            if filter_result.reason == "INSUFFICIENT_HOLD_TIME":
                hold_dur = filter_result.hold_duration
                return DecisionMessage(
                    category=DecisionCategory.WAITING,
                    text=f"WAITING — Hold {cand} for {target_hold:.1f}s ({hold_dur:.2f}s)",
                    color_bgr=COLOR_YELLOW,
                )

        # ---------------------------------------------------------------------
        # Priority 9: Low Confidence Gate
        # ---------------------------------------------------------------------
        if raw_gesture is not None and confidence < effective_threshold:
            return DecisionMessage(
                category=DecisionCategory.REJECTED,
                text=f"REJECTED — Confidence below threshold ({conf_pct}% < {thresh_pct}%)",
                color_bgr=COLOR_RED,
            )

        # ---------------------------------------------------------------------
        # Priority 9b: Custom Arbitration Diagnostics (when raw_gesture is None)
        # ---------------------------------------------------------------------
        if arbitration_reason == "AMBIGUOUS_CUSTOM_MATCH":
            return DecisionMessage(
                category=DecisionCategory.REJECTED,
                text="REJECTED — Ambiguous custom match (2 poses too similar)",
                color_bgr=COLOR_ORANGE,
            )
        if arbitration_reason == "CUSTOM_WEAK_MATCH":
            return DecisionMessage(
                category=DecisionCategory.WAITING,
                text="WAITING — Weak custom match (hold closer to enrolled pose)",
                color_bgr=COLOR_YELLOW,
            )
        if arbitration_reason == "BUILTIN_UNCERTAIN_GUARD":
            return DecisionMessage(
                category=DecisionCategory.REJECTED,
                text="REJECTED — Built-in transition guard active",
                color_bgr=COLOR_ORANGE,
            )

        # ---------------------------------------------------------------------
        # Priority 10: Baseline Active Waiting
        # ---------------------------------------------------------------------
        if raw_gesture is not None:
            return DecisionMessage(
                category=DecisionCategory.WAITING,
                text=f"WAITING — Tracking {raw_gesture}",
                color_bgr=COLOR_YELLOW,
            )

        return DecisionMessage(
            category=DecisionCategory.IDLE,
            text="IDLE — Ready for gesture or swipe",
            color_bgr=COLOR_GRAY,
        )

    @staticmethod
    def is_important_event(
        msg: DecisionMessage,
        just_activated: bool = False,
        just_locked: bool = False,
        is_swipe: bool = False,
        action_executed: bool = False,
        action_failed: bool = False,
    ) -> bool:
        """
        Determines whether a frame's decision or telemetry represents an important discrete event
        that should persist across subsequent frames on the HUD.

        Important events:
        - ACTION EXECUTED (action successfully executed)
        - SWIPE DETECTED (ballistic swipe gesture detected)
        - SYSTEM ACTIVATED (OpenPalm deliberate unlock)
        - SYSTEM LOCKED (inactivity or manual safety lock)
        - ACTION FAILED (handler exception or execution failure)
        """
        if action_executed or is_swipe or just_activated or just_locked or action_failed:
            return True
        if msg.category in (
            DecisionCategory.ACTION_EXECUTED,
            DecisionCategory.SWIPE_DETECTED,
            DecisionCategory.ERROR,
        ):
            return True
        if msg.category == DecisionCategory.CONFIRMED and "unlocked" in msg.text.lower():
            return True
        if msg.category == DecisionCategory.REJECTED and (
            "system locked due to inactivity" in msg.text.lower()
            or "system manually locked" in msg.text.lower()
        ):
            return True
        return False

    @staticmethod
    def resolve_display_decision(
        live_decision: DecisionMessage,
        last_event: Optional[DecisionMessage],
        last_event_timestamp: float,
        current_timestamp: float,
        display_duration: float = 2.0,
    ) -> DecisionMessage:
        """
        Resolves which DecisionMessage to display on the HUD:
        - If an important event occurred within display_duration of last_event_timestamp,
          returns last_event.
        - Otherwise, returns the current live_decision.
        """
        if last_event is not None and (current_timestamp - last_event_timestamp) < display_duration:
            return last_event
        return live_decision

    @staticmethod
    def resolve_hud_snapshot(
        live_decision: DecisionMessage,
        live_intent: Optional[Intent],
        live_action_name: Optional[str],
        live_action_status: str,
        last_event: Optional[EventSnapshot],
        last_event_timestamp: float,
        current_timestamp: float,
        display_duration: float = 2.0,
    ) -> Tuple[DecisionMessage, Optional[Intent], Optional[str], str]:
        """
        Resolves HUD display fields (decision, intent, action_name, action_status):
        - If an important event occurred within display_duration of last_event_timestamp,
          returns the synchronized fields from last_event.
        - Otherwise, returns the current live frame values together.
        """
        if last_event is not None and (current_timestamp - last_event_timestamp) < display_duration:
            return (
                last_event.decision,
                last_event.intent,
                last_event.action_name,
                last_event.action_status,
            )
        return (live_decision, live_intent, live_action_name, live_action_status)

