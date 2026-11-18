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
from typing import Callable, Dict, Optional, Tuple

from core.action_executor import ActionReason, ActionResult
from core.activation_state import ActivationResult, SystemState
from core.context_manager import Context
from core.intent_engine import Intent, IntentReason, IntentResult
from core.swipe_detector import SwipeDirection, SwipeResult
from core.temporal_filter import ConfirmationStatus, TemporalFilterResult
from core.llm_assistant import ExplanationResponse, LLMAssistant
from core.telemetry_aggregator import StrugglePattern, StruggleType, TelemetryAggregator


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
        coaching: Optional AI coaching response explaining the event.
    """
    decision: DecisionMessage
    intent: Optional[Intent] = None
    action_name: Optional[str] = None
    action_status: str = "IDLE"
    coaching: Optional[ExplanationResponse] = None


# Semantic BGR Color Palette
COLOR_GREEN: Tuple[int, int, int] = (0, 255, 0)      # Confirmed / Executed / Swiped
COLOR_CYAN: Tuple[int, int, int] = (255, 255, 0)     # Action Ready
COLOR_YELLOW: Tuple[int, int, int] = (0, 220, 255)   # Waiting / Progress
COLOR_ORANGE: Tuple[int, int, int] = (0, 165, 255)   # Cooldown / Locked rejection
COLOR_RED: Tuple[int, int, int] = (100, 100, 255)    # Error / Low confidence rejection
COLOR_GRAY: Tuple[int, int, int] = (180, 180, 180)   # Idle / Neutral


class DecisionExplainer:
    """
    Deterministic rule-based decision explanation engine, enhanced with
    intelligent, asynchronous LLM coaching feedback.
    """

    def __init__(
        self,
        llm_assistant: Optional[LLMAssistant] = None,
        telemetry_aggregator: Optional[TelemetryAggregator] = None,
    ) -> None:
        self._llm_assistant = llm_assistant
        self._telemetry_aggregator: TelemetryAggregator = (
            telemetry_aggregator if telemetry_aggregator is not None else TelemetryAggregator()
        )
        self._explanation_cache: Dict[str, ExplanationResponse] = {}
        self._latest_coaching: Optional[ExplanationResponse] = None
        self._pending_cache_key: Optional[str] = None
        self._is_loading_coaching: bool = False
        self._toast_message: Optional[str] = None
        self._toast_expires_at: float = 0.0
        self._toast_duration: float = 4.0

    @property
    def telemetry_aggregator(self) -> TelemetryAggregator:
        """Returns the internal bounded telemetry aggregator."""
        return self._telemetry_aggregator

    @property
    def llm_assistant(self) -> LLMAssistant:
        """Lazily instantiates LLMAssistant if not explicitly injected."""
        if self._llm_assistant is None:
            self._llm_assistant = LLMAssistant()
        return self._llm_assistant

    @property
    def is_loading_coaching(self) -> bool:
        """Indicates whether an asynchronous coaching request is currently in flight."""
        return self._is_loading_coaching

    @property
    def provider_status(self) -> str:
        """Returns the active provider configuration status."""
        if self._llm_assistant and self._llm_assistant.has_active_remote:
            provider_name = getattr(self._llm_assistant.http_provider, "name", "remote").upper()
            return f"ONLINE_{provider_name}"
        return "OFFLINE_HEURISTIC"

    def _compute_cache_key(
        self,
        category: str,
        text: str,
        state: str,
        context: str,
        raw_gesture: Optional[str],
        confidence: float,
        effective_threshold: float,
    ) -> str:
        # 5% confidence buckets to avoid jitter cache misses
        bucketed_conf = int(confidence * 20) * 5
        bucketed_thresh = int(effective_threshold * 100)
        return f"{category}|{text}|{state}|{context}|{raw_gesture or 'None'}|{bucketed_conf}|{bucketed_thresh}"

    def explain_with_llm(
        self,
        decision_msg: DecisionMessage,
        state: SystemState,
        context: Context,
        raw_gesture: Optional[str] = None,
        confidence: float = 0.0,
        effective_threshold: float = 0.80,
        hold_duration: float = 0.0,
        target_hold: float = 0.5,
        is_swipe: bool = False,
    ) -> ExplanationResponse:
        """
        Synchronously resolves a coaching explanation, utilizing semantic caching.
        """
        key = self._compute_cache_key(
            category=decision_msg.category.value,
            text=decision_msg.text,
            state=state.value,
            context=context.value,
            raw_gesture=raw_gesture,
            confidence=confidence,
            effective_threshold=effective_threshold,
        )
        if key in self._explanation_cache:
            self._latest_coaching = self._explanation_cache[key]
            return self._latest_coaching

        explanation = self.llm_assistant.explain_decision(
            decision_category=decision_msg.category.value,
            decision_text=decision_msg.text,
            state=state.value,
            context=context.value,
            raw_gesture=raw_gesture,
            confidence=confidence,
            effective_threshold=effective_threshold,
            hold_duration=hold_duration,
            target_hold=target_hold,
            is_swipe=is_swipe,
        )
        self._explanation_cache[key] = explanation
        self._latest_coaching = explanation
        return explanation

    def explain_snapshot_with_llm(
        self,
        snapshot: EventSnapshot,
        context: Context,
        state: SystemState = SystemState.ACTIVE,
    ) -> ExplanationResponse:
        """Convenience method to resolve coaching explanation for an EventSnapshot."""
        return self.explain_with_llm(
            decision_msg=snapshot.decision,
            state=state,
            context=context,
            raw_gesture=snapshot.action_name,
        )

    def request_coaching_async(
        self,
        decision_msg: DecisionMessage,
        state: SystemState,
        context: Context,
        raw_gesture: Optional[str] = None,
        confidence: float = 0.0,
        effective_threshold: float = 0.80,
        hold_duration: float = 0.0,
        target_hold: float = 0.5,
        is_swipe: bool = False,
        callback: Optional[Callable[[ExplanationResponse], None]] = None,
    ) -> None:
        """
        Dispatches coaching retrieval asynchronously to avoid blocking the webcam processing loop.
        Immediately returns cached response if available.
        """
        key = self._compute_cache_key(
            category=decision_msg.category.value,
            text=decision_msg.text,
            state=state.value,
            context=context.value,
            raw_gesture=raw_gesture,
            confidence=confidence,
            effective_threshold=effective_threshold,
        )
        if key in self._explanation_cache:
            cached = self._explanation_cache[key]
            self._latest_coaching = cached
            if callback:
                callback(cached)
            return

        if self._pending_cache_key == key:
            return  # Query already in flight for this exact event

        self._pending_cache_key = key
        self._is_loading_coaching = True

        def _on_done(res: Optional[ExplanationResponse]) -> None:
            self._is_loading_coaching = False
            self._pending_cache_key = None
            if res is not None:
                self._explanation_cache[key] = res
                self._latest_coaching = res
                if callback:
                    callback(res)
            elif callback:
                fallback = self.llm_assistant.offline_provider.explain_decision(
                    decision_category=decision_msg.category.value,
                    decision_text=decision_msg.text,
                    state=state.value,
                    context=context.value,
                    raw_gesture=raw_gesture,
                    confidence=confidence,
                    effective_threshold=effective_threshold,
                    hold_duration=hold_duration,
                    target_hold=target_hold,
                    is_swipe=is_swipe,
                )
                self._explanation_cache[key] = fallback
                self._latest_coaching = fallback
                callback(fallback)

        self.llm_assistant.query_async(
            self.llm_assistant.explain_decision,
            decision_category=decision_msg.category.value,
            decision_text=decision_msg.text,
            state=state.value,
            context=context.value,
            raw_gesture=raw_gesture,
            confidence=confidence,
            effective_threshold=effective_threshold,
            hold_duration=hold_duration,
            target_hold=target_hold,
            is_swipe=is_swipe,
            callback=_on_done,
        )

    def get_latest_coaching(self) -> Optional[ExplanationResponse]:
        """Returns the most recently resolved coaching response."""
        return self._latest_coaching

    def record_interaction(
        self,
        timestamp: float,
        decision_msg: DecisionMessage,
        gesture: Optional[str] = None,
        confidence: float = 0.0,
        effective_threshold: float = 0.80,
        hold_duration: float = 0.0,
        target_hold: float = 0.50,
        reason: str = "",
        context: Context = Context.MEDIA,
        is_swipe: bool = False,
        state: SystemState = SystemState.LOCKED,
    ) -> None:
        """Records telemetry into the bounded ring buffer."""
        state_str = state.value if isinstance(state, SystemState) else str(state)
        self._telemetry_aggregator.record(
            timestamp=timestamp,
            category=decision_msg.category.value,
            gesture=gesture,
            confidence=confidence,
            effective_threshold=effective_threshold,
            hold_duration=hold_duration,
            target_hold=target_hold,
            reason=reason or decision_msg.text,
            context=context.value,
            is_swipe=is_swipe,
            state=state_str,
        )

    def check_proactive_coaching(
        self,
        current_timestamp: float,
        context: Context = Context.MEDIA,
    ) -> Optional[str]:
        """
        Evaluates recent telemetry for interaction struggle patterns.
        If a pattern is detected, immediately triggers an asynchronous coaching
        request and surfaces a temporary toast notification.
        """
        pattern = self._telemetry_aggregator.detect_struggle(current_timestamp)
        if not pattern:
            return None

        # Immediate grounded offline coaching response
        offline_coach = self.llm_assistant.offline_provider.coach_struggle(pattern, context)
        self._toast_message = offline_coach.coaching_tip
        self._toast_expires_at = current_timestamp + self._toast_duration

        # Background async refinement if remote LLM is available
        def _on_async_coach(res: Optional[ExplanationResponse]) -> None:
            if res and res.coaching_tip and current_timestamp < self._toast_expires_at:
                self._toast_message = res.coaching_tip

        self.llm_assistant.query_async(
            self.llm_assistant.coach_struggle,
            pattern=pattern,
            context=context,
            callback=_on_async_coach,
        )
        return self._toast_message

    def get_active_toast(self, current_timestamp: float) -> Optional[Tuple[str, float]]:
        """
        Returns (toast_message, remaining_ratio) if a proactive coaching toast is active.
        Ratio ranges from 1.0 (fresh) down to 0.0 (expired).
        """
        if self._toast_message and current_timestamp < self._toast_expires_at:
            remaining = self._toast_expires_at - current_timestamp
            ratio = max(0.0, min(1.0, remaining / self._toast_duration))
            return (self._toast_message, ratio)
        self._toast_message = None
        return None

    def clear_active_toast(self) -> None:
        """Clears any currently active proactive coaching toast."""
        self._toast_message = None
        self._toast_expires_at = 0.0

    def clear_cache(self) -> None:
        """Clears the semantic coaching cache and telemetry buffer."""
        self._explanation_cache.clear()
        self._pending_cache_key = None
        self._is_loading_coaching = False
        self._telemetry_aggregator.clear()
        self._toast_message = None
        self._toast_expires_at = 0.0

    def close(self) -> None:
        """Shuts down the internal LLMAssistant if created internally."""
        if self._llm_assistant is not None:
            self._llm_assistant.close()

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

