"""
core/intent_engine.py
=====================

IntentSense Phase 5: Deterministic Intent Engine.

Responsible ONLY for translating a confirmed, active gesture into a context-specific
semantic intent.

Guarantees & Safety Boundary:
- Pure interpreter: does NOT execute actions, make OS calls, or interact with external apps.
- Enforces strict safety gates:
  1. System must be ACTIVE (rejects with SYSTEM_LOCKED if locked).
  2. Gesture must be CONFIRMED (rejects with GESTURE_NOT_CONFIRMED if unconfirmed).
  3. Gesture must be a recognized/known gesture (rejects with UNKNOWN_GESTURE if unknown).
  4. A mapping must exist for the active context (rejects with NO_MAPPING if unmapped).
- Decoupled from Phase 4 activation logic: OpenPalm is consumed by Phase 4 and has no
  command mapping in standard contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from core.activation_state import ActivationResult, SystemState
from core.context_manager import Context, ContextManager
from core.swipe_detector import SwipeDirection, SwipeResult
from core.temporal_filter import TemporalFilterResult


class Intent(str, Enum):
    """Semantic intents emitted by IntentSense."""

    # MEDIA Context Intents
    PLAY_PAUSE = "PLAY_PAUSE"
    PREVIOUS = "PREVIOUS"
    NEXT = "NEXT"
    VOLUME_UP = "VOLUME_UP"
    VOLUME_DOWN = "VOLUME_DOWN"

    # PRESENTATION Context Intents
    START_STOP = "START_STOP"
    PREVIOUS_SLIDE = "PREVIOUS_SLIDE"
    NEXT_SLIDE = "NEXT_SLIDE"
    ZOOM_IN = "ZOOM_IN"
    ZOOM_OUT = "ZOOM_OUT"

    def __str__(self) -> str:
        return self.value


class IntentReason(str, Enum):
    """Explicit diagnostic reasons for intent acceptance or rejection."""

    CONFIRMED = "CONFIRMED"
    SYSTEM_LOCKED = "SYSTEM_LOCKED"
    GESTURE_NOT_CONFIRMED = "GESTURE_NOT_CONFIRMED"
    UNKNOWN_GESTURE = "UNKNOWN_GESTURE"
    NO_MAPPING = "NO_MAPPING"

    def __str__(self) -> str:
        return self.value


# Supported gesture vocabulary recognized by IntentSense
KNOWN_GESTURES: Tuple[str, ...] = (
    "Fist",
    "OpenPalm",
    "Peace",
    "Pointing",
    "ThumbsDown",
    "ThumbsUp",
)


# Default deterministic gesture-to-intent mappings
DEFAULT_INTENT_MAPPINGS: Dict[Context, Dict[str, Intent]] = {
    Context.MEDIA: {
        "Fist": Intent.PLAY_PAUSE,
        "Peace": Intent.PREVIOUS,
        "Pointing": Intent.NEXT,
        "ThumbsUp": Intent.VOLUME_UP,
        "ThumbsDown": Intent.VOLUME_DOWN,
    },
    Context.PRESENTATION: {
        "Fist": Intent.START_STOP,
        "Peace": Intent.PREVIOUS_SLIDE,
        "Pointing": Intent.NEXT_SLIDE,
        "ThumbsUp": Intent.ZOOM_IN,
        "ThumbsDown": Intent.ZOOM_OUT,
    },
    Context.CUSTOM: {},
}

# Default deterministic dynamic swipe-to-intent mappings
DEFAULT_SWIPE_MAPPINGS: Dict[Context, Dict[SwipeDirection, Intent]] = {
    Context.MEDIA: {
        SwipeDirection.LEFT: Intent.PREVIOUS,
        SwipeDirection.RIGHT: Intent.NEXT,
    },
    Context.PRESENTATION: {
        SwipeDirection.LEFT: Intent.PREVIOUS_SLIDE,
        SwipeDirection.RIGHT: Intent.NEXT_SLIDE,
    },
    Context.CUSTOM: {},
}


@dataclass
class IntentResult:
    """
    Structured result emitted by IntentEngine.

    Attributes:
        gesture: The input gesture label evaluated (or None).
        intent: The resulting semantic Intent (None if rejected).
        context: The Context under which evaluation took place.
        confidence: Confidence score of the gesture.
        accepted: True if the gesture mapped to a valid intent, False otherwise.
        reason: Diagnostic IntentReason explaining acceptance or rejection.
    """
    gesture: Optional[str]
    intent: Optional[Intent]
    context: Context
    confidence: float
    accepted: bool
    reason: IntentReason


class IntentEngine:
    """
    Translates confirmed gestures into context-specific semantic intents.

    Safety Rules (evaluated in strict order):
    1. System state: Must be ACTIVE. If LOCKED -> rejected (SYSTEM_LOCKED).
    2. Confirmation: Gesture must be temporally confirmed. If not -> rejected (GESTURE_NOT_CONFIRMED).
    3. Gesture validity: Gesture must be in KNOWN_GESTURES. If not -> rejected (UNKNOWN_GESTURE).
    4. Mapping existence: Current context must have a mapping for gesture. If not -> rejected (NO_MAPPING).
    """

    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        mappings: Optional[Dict[Context, Dict[str, Intent]]] = None,
        swipe_mappings: Optional[Dict[Context, Dict[SwipeDirection, Intent]]] = None,
    ) -> None:
        """
        Initializes the IntentEngine.

        Args:
            context_manager: ContextManager instance. Creates a default instance (MEDIA) if None.
            mappings: Custom static gesture mapping dictionary. Defaults to deep-copy of DEFAULT_INTENT_MAPPINGS.
            swipe_mappings: Custom dynamic swipe mapping dictionary. Defaults to deep-copy of DEFAULT_SWIPE_MAPPINGS.
        """
        self.context_manager: ContextManager = context_manager if context_manager is not None else ContextManager()

        if mappings is not None:
            self._mappings: Dict[Context, Dict[str, Intent]] = {
                ctx: dict(gestures) for ctx, gestures in mappings.items()
            }
        else:
            self._mappings = {
                ctx: dict(gestures) for ctx, gestures in DEFAULT_INTENT_MAPPINGS.items()
            }

        if swipe_mappings is not None:
            self._swipe_mappings: Dict[Context, Dict[SwipeDirection, Intent]] = {
                ctx: dict(swipes) for ctx, swipes in swipe_mappings.items()
            }
        else:
            self._swipe_mappings = {
                ctx: dict(swipes) for ctx, swipes in DEFAULT_SWIPE_MAPPINGS.items()
            }

        self._custom_gestures: set[str] = set()

    def is_known_gesture(self, gesture: str) -> bool:
        """Returns True if gesture is in KNOWN_GESTURES or registered custom gestures."""
        return gesture in KNOWN_GESTURES or gesture in self._custom_gestures

    def register_custom_gesture(
        self,
        name: str,
        mappings: Optional[Dict[Union[Context, str], Union[Intent, str]]] = None,
    ) -> None:
        """Registers an enrolled custom gesture name and its context mappings."""
        clean_name = name.strip()
        if clean_name in KNOWN_GESTURES:
            raise ValueError(f"'{clean_name}' is a reserved built-in gesture.")
        self._custom_gestures.add(clean_name)
        if mappings:
            for ctx_val, intent_val in mappings.items():
                self.set_mapping(ctx_val, clean_name, intent_val)

    def unregister_custom_gesture(self, name: str) -> None:
        """Unregisters a custom gesture and removes its mappings across all contexts."""
        self._custom_gestures.discard(name)
        for ctx in self._mappings:
            self._mappings[ctx].pop(name, None)

    def rename_custom_gesture(self, old_name: str, new_name: str) -> None:
        """Renames a registered custom gesture, migrating all its context mappings."""
        clean_new = new_name.strip()
        if clean_new in KNOWN_GESTURES:
            raise ValueError(f"'{clean_new}' is a reserved built-in gesture.")
        self._custom_gestures.discard(old_name)
        self._custom_gestures.add(clean_new)
        for ctx in self._mappings:
            if old_name in self._mappings[ctx]:
                self._mappings[ctx][clean_new] = self._mappings[ctx].pop(old_name)

    def get_mappings(self, context: Union[Context, str]) -> Dict[str, Intent]:
        """Returns the gesture-to-intent mappings for the specified context."""
        ctx = Context.from_value(context)
        return dict(self._mappings.get(ctx, {}))

    def set_mapping(
        self,
        context: Union[Context, str],
        gesture: str,
        intent: Union[Intent, str],
    ) -> None:
        """
        Registers or updates a gesture-to-intent mapping for a specific context.
        Primarily intended for CUSTOM contexts or extensibility.
        """
        ctx = Context.from_value(context)
        if not self.is_known_gesture(gesture):
            raise ValueError(
                f"Unknown gesture '{gesture}'. Known gestures: {KNOWN_GESTURES}"
            )
        resolved_intent = Intent(intent) if isinstance(intent, str) else intent
        if ctx not in self._mappings:
            self._mappings[ctx] = {}
        self._mappings[ctx][gesture] = resolved_intent

    def get_swipe_mappings(self, context: Union[Context, str]) -> Dict[SwipeDirection, Intent]:
        """Returns the dynamic swipe-to-intent mappings for the specified context."""
        ctx = Context.from_value(context)
        return dict(self._swipe_mappings.get(ctx, {}))

    def set_swipe_mapping(
        self,
        context: Union[Context, str],
        direction: Union[SwipeDirection, str],
        intent: Union[Intent, str],
    ) -> None:
        """
        Registers or updates a swipe-to-intent mapping for a specific context.
        """
        ctx = Context.from_value(context)
        resolved_dir = SwipeDirection(direction) if isinstance(direction, str) else direction
        resolved_intent = Intent(intent) if isinstance(intent, str) else intent
        if ctx not in self._swipe_mappings:
            self._swipe_mappings[ctx] = {}
        self._swipe_mappings[ctx][resolved_dir] = resolved_intent

    def process_swipe(
        self,
        swipe_result: SwipeResult,
        is_active: bool = True,
        context: Optional[Union[Context, str]] = None,
        state: Optional[Union[SystemState, str]] = None,
        activation_result: Optional[ActivationResult] = None,
    ) -> IntentResult:
        """
        Translates a dynamic SwipeResult into a context-specific semantic intent.

        Safety Rules (evaluated in strict order):
        1. System state: Must be ACTIVE. If LOCKED -> rejected (SYSTEM_LOCKED).
        2. Detection validity: swipe_result must be detected with valid direction (LEFT or RIGHT).
           If not -> rejected (GESTURE_NOT_CONFIRMED).
        3. Mapping existence: Current context must have a mapping for the direction.
           If not -> rejected (NO_MAPPING).
        """
        # Resolve active status
        active_status = is_active
        if activation_result is not None:
            active_status = activation_result.is_active
        elif state is not None:
            if isinstance(state, SystemState):
                active_status = (state == SystemState.ACTIVE)
            else:
                active_status = (str(state).strip().upper() == SystemState.ACTIVE.value)

        # Resolve active context
        if context is not None:
            active_context = Context.from_value(context)
        else:
            active_context = self.context_manager.get_current_context()

        direction = swipe_result.direction
        gesture_label = f"Swipe{direction.value.capitalize()}" if direction != SwipeDirection.NONE else None

        # Safety Rule 1: System must be ACTIVE
        if not active_status:
            return IntentResult(
                gesture=gesture_label,
                intent=None,
                context=active_context,
                confidence=swipe_result.confidence,
                accepted=False,
                reason=IntentReason.SYSTEM_LOCKED,
            )

        # Safety Rule 2: Detection validity (must be detected with valid direction LEFT or RIGHT)
        if direction not in (SwipeDirection.LEFT, SwipeDirection.RIGHT):
            return IntentResult(
                gesture=gesture_label,
                intent=None,
                context=active_context,
                confidence=swipe_result.confidence,
                accepted=False,
                reason=IntentReason.GESTURE_NOT_CONFIRMED,
            )

        # Safety Rule 3: Context mapping
        context_swipes = self._swipe_mappings.get(active_context, {})
        if direction not in context_swipes:
            return IntentResult(
                gesture=gesture_label,
                intent=None,
                context=active_context,
                confidence=swipe_result.confidence,
                accepted=False,
                reason=IntentReason.NO_MAPPING,
            )

        # All checks passed: Confirmed Swipe Intent
        return IntentResult(
            gesture=gesture_label,
            intent=context_swipes[direction],
            context=active_context,
            confidence=swipe_result.confidence,
            accepted=True,
            reason=IntentReason.CONFIRMED,
        )

    def process(
        self,
        gesture: Optional[str] = None,
        is_confirmed: bool = True,
        is_active: bool = True,
        confidence: float = 1.0,
        context: Optional[Union[Context, str]] = None,
        state: Optional[Union[SystemState, str]] = None,
        activation_result: Optional[ActivationResult] = None,
        filter_result: Optional[TemporalFilterResult] = None,
    ) -> IntentResult:
        """
        Evaluates input parameters against safety rules and current context mappings.

        Args:
            gesture: Hand gesture string name (e.g. 'ThumbsUp', 'Fist').
            is_confirmed: Whether temporal stability confirmation was achieved.
            is_active: Whether system is in ACTIVE safety state.
            confidence: Model confidence score (0.0 to 1.0).
            context: Context override. If None, uses current_context from ContextManager.
            state: Optional SystemState enum or string ('ACTIVE', 'LOCKED').
            activation_result: Optional Phase 4 ActivationResult for seamless pipeline chaining.
            filter_result: Optional Phase 3 TemporalFilterResult for chaining.

        Returns:
            IntentResult structured output.
        """
        # Resolve active state
        active_status = is_active
        if activation_result is not None:
            active_status = activation_result.is_active
            if gesture is None:
                gesture = activation_result.command
            is_confirmed = activation_result.command is not None
        elif state is not None:
            if isinstance(state, SystemState):
                active_status = (state == SystemState.ACTIVE)
            else:
                active_status = (str(state).strip().upper() == SystemState.ACTIVE.value)

        # Resolve filter result attributes if provided
        if filter_result is not None:
            confidence = filter_result.confidence
            if gesture is None:
                gesture = filter_result.gesture or filter_result.candidate_gesture
            is_confirmed = filter_result.is_confirmed

        # Resolve context
        if context is not None:
            active_context = Context.from_value(context)
        else:
            active_context = self.context_manager.get_current_context()

        # ---------------------------------------------------------------------
        # Safety Rule 1: System must be ACTIVE
        # ---------------------------------------------------------------------
        if not active_status:
            return IntentResult(
                gesture=gesture,
                intent=None,
                context=active_context,
                confidence=confidence,
                accepted=False,
                reason=IntentReason.SYSTEM_LOCKED,
            )

        # ---------------------------------------------------------------------
        # Safety Rule 2: Gesture must be CONFIRMED
        # ---------------------------------------------------------------------
        if not is_confirmed:
            return IntentResult(
                gesture=gesture,
                intent=None,
                context=active_context,
                confidence=confidence,
                accepted=False,
                reason=IntentReason.GESTURE_NOT_CONFIRMED,
            )

        # ---------------------------------------------------------------------
        # Safety Rule 3: Gesture must be recognized (known built-in or registered custom)
        # ---------------------------------------------------------------------
        if gesture is None or not self.is_known_gesture(gesture):
            return IntentResult(
                gesture=gesture,
                intent=None,
                context=active_context,
                confidence=confidence,
                accepted=False,
                reason=IntentReason.UNKNOWN_GESTURE,
            )

        # ---------------------------------------------------------------------
        # Safety Rule 4: Mapping must exist for the active context
        # ---------------------------------------------------------------------
        context_mapping = self._mappings.get(active_context, {})
        if gesture not in context_mapping:
            return IntentResult(
                gesture=gesture,
                intent=None,
                context=active_context,
                confidence=confidence,
                accepted=False,
                reason=IntentReason.NO_MAPPING,
            )

        # ---------------------------------------------------------------------
        # All checks passed: Confirmed Intent
        # ---------------------------------------------------------------------
        return IntentResult(
            gesture=gesture,
            intent=context_mapping[gesture],
            context=active_context,
            confidence=confidence,
            accepted=True,
            reason=IntentReason.CONFIRMED,
        )
