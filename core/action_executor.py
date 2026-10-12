"""
core/action_executor.py
=======================

IntentSense Phase 6: Modular, Fail-Safe Action Execution Layer.

Converts accepted, verified IntentResult instances from Phase 5 into physical
or simulated system actions via an extensible, registry-based architecture.

Architectural Principles:
- Recognition != Intent != Action.
- Independent from ML pipelines, webcam loops, and UI frameworks.
- Hard safety gate: NEVER executes if IntentResult was rejected upstream.
- Fail-safe: Catches handler exceptions and returns structured failure results
  without crashing or silently swallowing errors.
- Extensible: Handlers are registered dynamically without modifying execute().
- Thread-safe / multi-instance safe: No shared mutable state between instances.
- Ready for future WebSocket/REST API integration.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from core.intent_engine import Intent, IntentResult


class ActionReason(str, Enum):
    """Structured diagnostic reasons for action execution outcomes."""
    EXECUTED = "EXECUTED"
    INTENT_REJECTED = "INTENT_REJECTED"
    NO_HANDLER = "NO_HANDLER"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"

    def __str__(self) -> str:
        return self.value


@dataclass
class ActionResult:
    """
    Structured outcome emitted by the ActionExecutor.

    Attributes:
        intent: The Intent that was evaluated (None if rejected before intent resolution).
        success: True if the action handler executed without error, False otherwise.
        action_name: Identifier of the action handler executed (or None).
        reason: Diagnostic ActionReason categorizing the outcome.
        message: Descriptive explanation or error message.
    """
    intent: Optional[Intent]
    success: bool
    action_name: Optional[str]
    reason: ActionReason
    message: str


@dataclass
class ActionRegistration:
    """Internal metadata for a registered action handler."""
    intent: Intent
    handler: Callable[..., Any]
    action_name: str


def _invoke_handler(handler: Callable[..., Any], intent_result: IntentResult) -> Any:
    """
    Invokes a handler callable, passing intent_result if accepted by its signature,
    or calling with no arguments if the handler takes 0 parameters.
    """
    try:
        sig = inspect.signature(handler)
        if len(sig.parameters) == 0:
            return handler()
        return handler(intent_result)
    except (ValueError, TypeError):
        # Fallback for builtins or special callables where inspect might fail
        try:
            return handler(intent_result)
        except TypeError:
            return handler()


class ActionExecutor:
    """
    Modular, fail-safe action execution engine.

    Routes validated IntentResult objects to registered handlers.
    Strictly enforces upstream safety decisions and wraps execution in fail-safe guards.
    """

    def __init__(self, action_cooldown: float = 0.0) -> None:
        """
        Initializes an ActionExecutor instance.

        Args:
            action_cooldown: Minimum seconds between executions of the same intent.
                             Defaults to 0.0 (disabled).
        """
        self.action_cooldown: float = float(action_cooldown)
        self._registry: Dict[Intent, ActionRegistration] = {}
        self._last_execution_times: Dict[Intent, float] = {}

    def register(
        self,
        intent: Union[Intent, str],
        handler: Callable[..., Any],
        action_name: Optional[str] = None,
    ) -> None:
        """
        Registers an action handler for a specific Intent.

        Args:
            intent: The Intent enum or string name.
            handler: Callable to execute when the intent is received.
            action_name: Optional human-readable action name. Defaults to handler name
                         or lower-cased intent string.
        """
        resolved_intent = Intent(intent) if isinstance(intent, str) else intent
        if not callable(handler):
            raise TypeError(f"Handler for {resolved_intent} must be callable, got {type(handler)}")

        if action_name is None:
            raw_name = getattr(handler, "__name__", None)
            if raw_name and raw_name != "<lambda>":
                name = raw_name
            else:
                name = str(resolved_intent).lower()
        else:
            name = str(action_name)

        self._registry[resolved_intent] = ActionRegistration(
            intent=resolved_intent,
            handler=handler,
            action_name=name,
        )

    def unregister(self, intent: Union[Intent, str]) -> bool:
        """
        Unregisters the handler for a given Intent.

        Returns:
            True if a handler was removed, False if none was registered.
        """
        resolved_intent = Intent(intent) if isinstance(intent, str) else intent
        return self._registry.pop(resolved_intent, None) is not None

    def has_handler(self, intent: Union[Intent, str]) -> bool:
        """Returns True if a handler is registered for the specified Intent."""
        resolved_intent = Intent(intent) if isinstance(intent, str) else intent
        return resolved_intent in self._registry

    def get_handler(self, intent: Union[Intent, str]) -> Optional[Callable[..., Any]]:
        """Returns the registered callable for the intent, or None."""
        resolved_intent = Intent(intent) if isinstance(intent, str) else intent
        reg = self._registry.get(resolved_intent)
        return reg.handler if reg else None

    def get_registered_intents(self) -> List[Intent]:
        """Returns a list of all currently registered Intents."""
        return list(self._registry.keys())

    def clear(self) -> None:
        """Clears all registered handlers and execution history."""
        self._registry.clear()
        self._last_execution_times.clear()

    def execute(self, intent_result: Optional[IntentResult]) -> ActionResult:
        """
        Executes an action for the given IntentResult with full safety verification.

        Safety Verification Steps:
        1. intent_result is not None
        2. intent_result.accepted is True
        3. intent_result.intent is not None
        If any check fails, execution is refused with INTENT_REJECTED.

        Execution Steps:
        4. Handler lookup: If no handler is registered, returns NO_HANDLER.
        5. Action cooldown: If cooldown active, returns COOLDOWN_ACTIVE.
        6. Fail-safe call: Handler is called in a protected block. If an exception
           is raised, returns EXECUTION_FAILED with the exception details.
        7. Success: Returns EXECUTED with success=True.

        Args:
            intent_result: Output from Phase 5 IntentEngine.

        Returns:
            ActionResult
        """
        # ---------------------------------------------------------------------
        # Safety Gate 1: None check
        # ---------------------------------------------------------------------
        if intent_result is None:
            return ActionResult(
                intent=None,
                success=False,
                action_name=None,
                reason=ActionReason.INTENT_REJECTED,
                message="Rejected IntentResult was not executed (intent_result is None).",
            )

        # ---------------------------------------------------------------------
        # Safety Gate 2: Upstream Acceptance Check
        # ---------------------------------------------------------------------
        if not getattr(intent_result, "accepted", False):
            reason_str = str(getattr(intent_result, "reason", "REJECTED"))
            return ActionResult(
                intent=intent_result.intent,
                success=False,
                action_name=None,
                reason=ActionReason.INTENT_REJECTED,
                message=f"Rejected IntentResult was not executed (reason: {reason_str}).",
            )

        # ---------------------------------------------------------------------
        # Safety Gate 3: Intent Presence Check
        # ---------------------------------------------------------------------
        if intent_result.intent is None:
            return ActionResult(
                intent=None,
                success=False,
                action_name=None,
                reason=ActionReason.INTENT_REJECTED,
                message="Rejected IntentResult was not executed (intent is None).",
            )

        target_intent = intent_result.intent

        # ---------------------------------------------------------------------
        # Action Registry Lookup
        # ---------------------------------------------------------------------
        registration = self._registry.get(target_intent)
        if registration is None:
            return ActionResult(
                intent=target_intent,
                success=False,
                action_name=None,
                reason=ActionReason.NO_HANDLER,
                message=f"No action handler registered for {target_intent}.",
            )

        action_name = registration.action_name

        # ---------------------------------------------------------------------
        # Cooldown / Debounce Check (Extension Point)
        # ---------------------------------------------------------------------
        now = time.monotonic()
        if self.action_cooldown > 0.0:
            last_time = self._last_execution_times.get(target_intent)
            if last_time is not None and (now - last_time) < (self.action_cooldown - 1e-5):
                remaining = self.action_cooldown - (now - last_time)
                return ActionResult(
                    intent=target_intent,
                    success=False,
                    action_name=action_name,
                    reason=ActionReason.COOLDOWN_ACTIVE,
                    message=f"Action '{action_name}' is in cooldown ({remaining:.2f}s remaining).",
                )

        # ---------------------------------------------------------------------
        # Fail-Safe Execution Guard
        # ---------------------------------------------------------------------
        try:
            handler_output = _invoke_handler(registration.handler, intent_result)

            if self.action_cooldown > 0.0:
                self._last_execution_times[target_intent] = now

            if isinstance(handler_output, ActionResult):
                return handler_output

            msg = (
                str(handler_output)
                if isinstance(handler_output, str)
                else f"{action_name} executed successfully."
            )
            return ActionResult(
                intent=target_intent,
                success=True,
                action_name=action_name,
                reason=ActionReason.EXECUTED,
                message=msg,
            )
        except Exception as exc:
            return ActionResult(
                intent=target_intent,
                success=False,
                action_name=action_name,
                reason=ActionReason.EXECUTION_FAILED,
                message=f"Action handler '{action_name}' raised an exception: {exc}",
            )
