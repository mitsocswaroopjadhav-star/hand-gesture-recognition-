"""
actions/media_actions.py
========================

IntentSense Media Action Handlers.

Provides independent, fail-safe handlers for MEDIA context intents:
- VOLUME_UP
- VOLUME_DOWN
- PLAY_PAUSE
- NEXT
- PREVIOUS

Design:
- Supports safe dry-run mode (default in testing/unprivileged environments).
- When real execution is enabled on macOS, dispatches via native AppleScript/osascript
  or PyAutoGUI where appropriate.
- Handlers can be registered directly into ActionExecutor.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, Callable, Optional

from core.intent_engine import Intent

if TYPE_CHECKING:
    from core.action_executor import ActionExecutor


def handle_volume_up(dry_run: bool = True) -> str:
    """Increases system audio volume."""
    if dry_run:
        return "[DRY RUN] Volume up executed."

    if sys.platform == "darwin":
        cmd = "set volume output volume (output volume of (get volume settings) + 10)"
        subprocess.run(["osascript", "-e", cmd], check=True, capture_output=True)
        return "macOS volume increased."
    else:
        try:
            import pyautogui
            pyautogui.press("volumeup")
            return "Volume up key pressed."
        except Exception as e:
            raise RuntimeError(f"Volume up failed: {e}") from e


def handle_volume_down(dry_run: bool = True) -> str:
    """Decreases system audio volume."""
    if dry_run:
        return "[DRY RUN] Volume down executed."

    if sys.platform == "darwin":
        cmd = "set volume output volume (output volume of (get volume settings) - 10)"
        subprocess.run(["osascript", "-e", cmd], check=True, capture_output=True)
        return "macOS volume decreased."
    else:
        try:
            import pyautogui
            pyautogui.press("volumedown")
            return "Volume down key pressed."
        except Exception as e:
            raise RuntimeError(f"Volume down failed: {e}") from e


def _send_macos_media_key(key_code: int) -> bool:
    """
    Dispatches a native hardware media key event via macOS AppKit and Quartz.
    Key 16: Play/Pause | Key 17: Next Track | Key 18: Previous Track
    Affects whatever audio/video source is currently active (YouTube, Spotify, Music, etc.).
    """
    try:
        import AppKit
        import Quartz

        for down in (True, False):
            flags = 0xa00 if down else 0xb00
            data1 = (key_code << 16) | flags
            ev = AppKit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                AppKit.NSEventTypeSystemDefined,
                (0, 0),
                flags,
                0,
                0,
                None,
                8,
                data1,
                -1,
            )
            cg_ev = ev.CGEvent()
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, cg_ev)
        return True
    except Exception:
        return False


def handle_play_pause(dry_run: bool = True) -> str:
    """Toggles media play/pause state."""
    if dry_run:
        return "[DRY RUN] Play/Pause toggled."

    if sys.platform == "darwin":
        if _send_macos_media_key(16):
            return "macOS system Play/Pause media key dispatched."
        try:
            subprocess.run(["osascript", "-e", 'tell application "Music" to playpause'], check=True, capture_output=True)
            return "Music play/pause toggled via osascript."
        except Exception:
            try:
                import pyautogui
                pyautogui.press("playpause")
                return "Play/Pause key pressed."
            except Exception as e:
                raise RuntimeError(f"Play/Pause failed: {e}") from e
    else:
        try:
            import pyautogui
            pyautogui.press("playpause")
            return "Play/Pause key pressed."
        except Exception as e:
            raise RuntimeError(f"Play/Pause failed: {e}") from e


def handle_next(dry_run: bool = True) -> str:
    """Advances to next media track."""
    if dry_run:
        return "[DRY RUN] Next track executed."

    if sys.platform == "darwin":
        if _send_macos_media_key(17):
            return "macOS system Next Track media key dispatched."
        try:
            subprocess.run(["osascript", "-e", 'tell application "Music" to next track'], check=True, capture_output=True)
            return "Next track executed via osascript."
        except Exception:
            try:
                import pyautogui
                pyautogui.press("nexttrack")
                return "Next track key pressed."
            except Exception as e:
                raise RuntimeError(f"Next track failed: {e}") from e
    else:
        try:
            import pyautogui
            pyautogui.press("nexttrack")
            return "Next track key pressed."
        except Exception as e:
            raise RuntimeError(f"Next track failed: {e}") from e


def handle_previous(dry_run: bool = True) -> str:
    """Navigates to previous media track."""
    if dry_run:
        return "[DRY RUN] Previous track executed."

    if sys.platform == "darwin":
        if _send_macos_media_key(18):
            return "macOS system Previous Track media key dispatched."
        try:
            subprocess.run(["osascript", "-e", 'tell application "Music" to previous track'], check=True, capture_output=True)
            return "Previous track executed via osascript."
        except Exception:
            try:
                import pyautogui
                pyautogui.press("prevtrack")
                return "Previous track key pressed."
            except Exception as e:
                raise RuntimeError(f"Previous track failed: {e}") from e
    else:
        try:
            import pyautogui
            pyautogui.press("prevtrack")
            return "Previous track key pressed."
        except Exception as e:
            raise RuntimeError(f"Previous track failed: {e}") from e


def register_media_handlers(executor: ActionExecutor, dry_run: bool = True) -> None:
    """
    Registers standard MEDIA action handlers into an ActionExecutor instance.

    Args:
        executor: The ActionExecutor to register handlers into.
        dry_run: When True, handlers simulate execution without modifying OS state.
    """
    executor.register(
        Intent.VOLUME_UP,
        lambda _: handle_volume_up(dry_run=dry_run),
        action_name="volume_up",
    )
    executor.register(
        Intent.VOLUME_DOWN,
        lambda _: handle_volume_down(dry_run=dry_run),
        action_name="volume_down",
    )
    executor.register(
        Intent.PLAY_PAUSE,
        lambda _: handle_play_pause(dry_run=dry_run),
        action_name="play_pause",
    )
    executor.register(
        Intent.NEXT,
        lambda _: handle_next(dry_run=dry_run),
        action_name="next_track",
    )
    executor.register(
        Intent.PREVIOUS,
        lambda _: handle_previous(dry_run=dry_run),
        action_name="previous_track",
    )
