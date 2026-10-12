"""
core/context_manager.py
=======================

IntentSense Phase 5: Deterministic Context Management.

Manages the active operational context of the IntentSense system (e.g. MEDIA,
PRESENTATION, CUSTOM). The ContextManager decouples the meaning of user gestures
from fixed hardcoded actions, allowing the same physical gesture to trigger
different contextual intents.

Responsibilities:
- Maintain current operational context (default: MEDIA).
- Validate and safely switch between supported contexts.
- Reject invalid contexts without corrupting the active context state.
- Purely deterministic; no automatic OS detection or UI dependencies.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Union


class Context(str, Enum):
    """Supported operational contexts for IntentSense."""
    MEDIA = "MEDIA"
    PRESENTATION = "PRESENTATION"
    CUSTOM = "CUSTOM"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def supported_contexts(cls) -> List[str]:
        """Returns list of valid context string names."""
        return [c.value for c in cls]

    @classmethod
    def from_value(cls, value: Any) -> Context:
        """
        Converts a string or Context instance to a valid Context enum.
        Raises ValueError if value is not a recognized context.
        """
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in cls.__members__:
                return cls[normalized]

        supported = ", ".join(cls.supported_contexts())
        raise ValueError(
            f"Invalid context '{value}'. Supported contexts are: {supported}"
        )

    @classmethod
    def is_valid(cls, value: Any) -> bool:
        """Returns True if the value corresponds to a valid Context."""
        try:
            cls.from_value(value)
            return True
        except (ValueError, AttributeError, TypeError):
            return False


class ContextManager:
    """
    Manages and validates the active operating context.

    Guarantees:
    - Initial default context is strictly MEDIA.
    - Context switches only occur when the target context is valid.
    - If an invalid context is provided, an error is raised and the active
      context remains completely uncorrupted.
    """

    def __init__(self, initial_context: Union[Context, str] = Context.MEDIA) -> None:
        """
        Initializes the ContextManager.

        Args:
            initial_context: Context enum or string name. Defaults to Context.MEDIA.
        """
        self._current_context: Context = Context.from_value(initial_context)

    @property
    def current_context(self) -> Context:
        """Returns the currently active Context."""
        return self._current_context

    def get_current_context(self) -> Context:
        """Returns the currently active Context enum."""
        return self._current_context

    def set_context(self, context: Union[Context, str]) -> Context:
        """
        Switches the operational context to the target context.

        Args:
            context: Context enum or string name.

        Returns:
            The newly set Context enum.

        Raises:
            ValueError: If target context is not supported. Current context is preserved.
        """
        valid_context = Context.from_value(context)
        self._current_context = valid_context
        return self._current_context

    def validate_context(self, context: Any) -> bool:
        """
        Checks if the provided context is supported without modifying state.

        Args:
            context: Value to validate.

        Returns:
            True if valid, False otherwise.
        """
        return Context.is_valid(context)

    def reset(self) -> Context:
        """Resets the context back to the default MEDIA context."""
        self._current_context = Context.MEDIA
        return self._current_context

    def __repr__(self) -> str:
        return f"ContextManager(current_context={self._current_context.value})"
