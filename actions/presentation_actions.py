"""
actions/presentation_actions.py
===============================

IntentSense Presentation Action Handlers.

Provides independent, fail-safe handlers for PRESENTATION context intents:
- START_STOP
- PREVIOUS_SLIDE
- NEXT_SLIDE
- ZOOM_IN
- ZOOM_OUT

Design:
- Supports safe dry-run mode (default in testing/unprivileged environments).
- When real execution is enabled, dispatches standard presentation navigation
  keystrokes (e.g. F5, Left/Right arrow, Zoom shortcuts).
- Handlers can be registered directly into ActionExecutor.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Callable, Optional

from core.intent_engine import Intent

if TYPE_CHECKING:
    from core.action_executor import ActionExecutor


def handle_start_stop(dry_run: bool = True) -> str:
    """Starts or exits slideshow presentation."""
    if dry_run:
        return "[DRY RUN] Presentation Start/Stop toggled."

    try:
        import pyautogui
        # Keynote / PowerPoint toggle shortcut (Shift+Cmd+Return on Mac or F5 on PC)
        if sys.platform == "darwin":
            pyautogui.hotkey("command", "option", "p")
        else:
            pyautogui.press("f5")
        return "Presentation slideshow toggled."
    except Exception as e:
        raise RuntimeError(f"Start/Stop presentation failed: {e}") from e


def handle_next_slide(dry_run: bool = True) -> str:
    """Advances to next presentation slide."""
    if dry_run:
        return "[DRY RUN] Next slide executed."

    try:
        import pyautogui
        pyautogui.press("right")
        return "Next slide key (Right Arrow) pressed."
    except Exception as e:
        raise RuntimeError(f"Next slide failed: {e}") from e


def handle_previous_slide(dry_run: bool = True) -> str:
    """Navigates to previous presentation slide."""
    if dry_run:
        return "[DRY RUN] Previous slide executed."

    try:
        import pyautogui
        pyautogui.press("left")
        return "Previous slide key (Left Arrow) pressed."
    except Exception as e:
        raise RuntimeError(f"Previous slide failed: {e}") from e


def handle_zoom_in(dry_run: bool = True) -> str:
    """Zooms in current presentation view."""
    if dry_run:
        return "[DRY RUN] Zoom in executed."

    try:
        import pyautogui
        modifier = "command" if sys.platform == "darwin" else "ctrl"
        pyautogui.hotkey(modifier, "+")
        return f"Zoom in ({modifier}+'+') executed."
    except Exception as e:
        raise RuntimeError(f"Zoom in failed: {e}") from e


def handle_zoom_out(dry_run: bool = True) -> str:
    """Zooms out current presentation view."""
    if dry_run:
        return "[DRY RUN] Zoom out executed."

    try:
        import pyautogui
        modifier = "command" if sys.platform == "darwin" else "ctrl"
        pyautogui.hotkey(modifier, "-")
        return f"Zoom out ({modifier}+'-') executed."
    except Exception as e:
        raise RuntimeError(f"Zoom out failed: {e}") from e


def register_presentation_handlers(executor: ActionExecutor, dry_run: bool = True) -> None:
    """
    Registers standard PRESENTATION action handlers into an ActionExecutor instance.

    Args:
        executor: The ActionExecutor to register handlers into.
        dry_run: When True, handlers simulate execution without modifying OS state.
    """
    executor.register(
        Intent.START_STOP,
        lambda _: handle_start_stop(dry_run=dry_run),
        action_name="start_stop_presentation",
    )
    executor.register(
        Intent.NEXT_SLIDE,
        lambda _: handle_next_slide(dry_run=dry_run),
        action_name="next_slide",
    )
    executor.register(
        Intent.PREVIOUS_SLIDE,
        lambda _: handle_previous_slide(dry_run=dry_run),
        action_name="previous_slide",
    )
    executor.register(
        Intent.ZOOM_IN,
        lambda _: handle_zoom_in(dry_run=dry_run),
        action_name="zoom_in",
    )
    executor.register(
        Intent.ZOOM_OUT,
        lambda _: handle_zoom_out(dry_run=dry_run),
        action_name="zoom_out",
    )
