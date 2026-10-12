"""
validate_live_actions.py
========================

IntentSense Phase 6.1: Controlled Live Action Validation Harness.

Validates the execution path:
  Gesture -> Intent -> ActionExecutor -> Registered Handler -> System Action -> ActionResult

Safety Guarantees:
- DEFAULTS TO DRY RUN: Real system actions will NEVER execute unless --live is explicitly passed.
- Isolates the Action Layer from webcam/MediaPipe/RF variability using deterministic IntentResult fixtures.
- Verifies upstream safety rejection handling (SYSTEM_LOCKED, GESTURE_NOT_CONFIRMED, UNKNOWN_GESTURE, NO_MAPPING).
- Verifies Action Layer cooldown/debouncing and error boundary fail-safe handling.
- When --live is enabled, measures observable macOS volume changes before and after execution.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

from actions.media_actions import (
    handle_next,
    handle_play_pause,
    handle_previous,
    handle_volume_down,
    handle_volume_up,
    register_media_handlers,
)
from actions.presentation_actions import (
    handle_next_slide,
    handle_previous_slide,
    handle_start_stop,
    handle_zoom_in,
    handle_zoom_out,
    register_presentation_handlers,
)
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


def get_macos_volume() -> Optional[int]:
    """Reads the current macOS system output volume (0-100) via osascript."""
    if sys.platform != "darwin":
        return None
    try:
        proc = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(proc.stdout.strip())
    except Exception as e:
        print(f"  [Warning] Unable to read macOS volume: {e}")
        return None


def run_safety_gate_validation() -> Dict[str, bool]:
    """
    Validates that rejected or invalid inputs NEVER trigger any action handler.
    Uses mock handlers to verify zero execution calls.
    """
    print("\n" + "=" * 65)
    print(" 1. SAFETY GATE VALIDATION (Verifying 0 executions on bad input)")
    print("=" * 65)

    executor = ActionExecutor()
    mock_handler = MagicMock(return_value="executed")
    for intent in Intent:
        executor.register(intent, mock_handler, action_name=str(intent).lower())

    results: Dict[str, bool] = {}

    # Case A: intent_result is None
    res_a = executor.execute(None)
    mock_handler.assert_not_called()
    pass_a = (not res_a.success) and (res_a.reason == ActionReason.INTENT_REJECTED)
    results["None Input -> No Action"] = pass_a
    print(f"  A. intent_result = None          -> Success={res_a.success}, Reason={res_a.reason.value:<16} [{'PASS' if pass_a else 'FAIL'}]")

    # Case B: accepted = False
    unaccepted = IntentResult(
        gesture="Fist",
        intent=Intent.PLAY_PAUSE,
        context=Context.MEDIA,
        confidence=0.85,
        accepted=False,
        reason=IntentReason.GESTURE_NOT_CONFIRMED,
    )
    res_b = executor.execute(unaccepted)
    mock_handler.assert_not_called()
    pass_b = (not res_b.success) and (res_b.reason == ActionReason.INTENT_REJECTED)
    results["accepted=False -> No Action"] = pass_b
    print(f"  B. accepted = False              -> Success={res_b.success}, Reason={res_b.reason.value:<16} [{'PASS' if pass_b else 'FAIL'}]")

    # Case C: SYSTEM_LOCKED
    locked = IntentResult(
        gesture="ThumbsUp",
        intent=None,
        context=Context.MEDIA,
        confidence=0.99,
        accepted=False,
        reason=IntentReason.SYSTEM_LOCKED,
    )
    res_c = executor.execute(locked)
    mock_handler.assert_not_called()
    pass_c = (not res_c.success) and (res_c.reason == ActionReason.INTENT_REJECTED)
    results["SYSTEM_LOCKED -> No Action"] = pass_c
    print(f"  C. SYSTEM_LOCKED                 -> Success={res_c.success}, Reason={res_c.reason.value:<16} [{'PASS' if pass_c else 'FAIL'}]")

    # Case D: GESTURE_NOT_CONFIRMED
    unconfirmed = IntentResult(
        gesture="ThumbsUp",
        intent=None,
        context=Context.MEDIA,
        confidence=0.72,
        accepted=False,
        reason=IntentReason.GESTURE_NOT_CONFIRMED,
    )
    res_d = executor.execute(unconfirmed)
    mock_handler.assert_not_called()
    pass_d = (not res_d.success) and (res_d.reason == ActionReason.INTENT_REJECTED)
    results["Unconfirmed -> No Action"] = pass_d
    print(f"  D. GESTURE_NOT_CONFIRMED         -> Success={res_d.success}, Reason={res_d.reason.value:<16} [{'PASS' if pass_d else 'FAIL'}]")

    # Case E: UNKNOWN_GESTURE
    unknown_g = IntentResult(
        gesture="AlienWave",
        intent=None,
        context=Context.MEDIA,
        confidence=0.95,
        accepted=False,
        reason=IntentReason.UNKNOWN_GESTURE,
    )
    res_e = executor.execute(unknown_g)
    mock_handler.assert_not_called()
    pass_e = (not res_e.success) and (res_e.reason == ActionReason.INTENT_REJECTED)
    results["UNKNOWN_GESTURE -> No Action"] = pass_e
    print(f"  E. UNKNOWN_GESTURE               -> Success={res_e.success}, Reason={res_e.reason.value:<16} [{'PASS' if pass_e else 'FAIL'}]")

    # Case F: NO_MAPPING
    nomap = IntentResult(
        gesture="OpenPalm",
        intent=None,
        context=Context.MEDIA,
        confidence=0.99,
        accepted=False,
        reason=IntentReason.NO_MAPPING,
    )
    res_f = executor.execute(nomap)
    mock_handler.assert_not_called()
    pass_f = (not res_f.success) and (res_f.reason == ActionReason.INTENT_REJECTED)
    results["NO_MAPPING -> No Action"] = pass_f
    print(f"  F. NO_MAPPING                    -> Success={res_f.success}, Reason={res_f.reason.value:<16} [{'PASS' if pass_f else 'FAIL'}]")

    # Case G: intent is None with accepted=True (edge case)
    null_intent = IntentResult(
        gesture="ThumbsUp",
        intent=None,
        context=Context.MEDIA,
        confidence=0.95,
        accepted=True,
        reason=IntentReason.CONFIRMED,
    )
    res_g = executor.execute(null_intent)
    mock_handler.assert_not_called()
    pass_g = (not res_g.success) and (res_g.reason == ActionReason.INTENT_REJECTED)
    results["intent=None -> No Action"] = pass_g
    print(f"  G. intent = None                 -> Success={res_g.success}, Reason={res_g.reason.value:<16} [{'PASS' if pass_g else 'FAIL'}]")

    return results


def run_cooldown_validation() -> bool:
    """Validates that ActionExecutor action_cooldown rate-limits repeated calls."""
    print("\n" + "=" * 65)
    print(" 2. ACTION COOLDOWN VALIDATION")
    print("=" * 65)

    executor = ActionExecutor(action_cooldown=0.3)
    call_count = 0

    def test_handler():
        nonlocal call_count
        call_count += 1
        return "ok"

    executor.register(Intent.VOLUME_UP, test_handler, action_name="test_vol")
    intent_res = IntentResult(
        gesture="ThumbsUp",
        intent=Intent.VOLUME_UP,
        context=Context.MEDIA,
        confidence=0.95,
        accepted=True,
        reason=IntentReason.CONFIRMED,
    )

    # 1. First execution
    r1 = executor.execute(intent_res)
    pass_r1 = r1.success and (r1.reason == ActionReason.EXECUTED) and (call_count == 1)

    # 2. Immediate repeated execution (must be blocked)
    r2 = executor.execute(intent_res)
    pass_r2 = (not r2.success) and (r2.reason == ActionReason.COOLDOWN_ACTIVE) and (call_count == 1)

    # 3. Wait for cooldown expiration
    time.sleep(0.35)
    r3 = executor.execute(intent_res)
    pass_r3 = r3.success and (r3.reason == ActionReason.EXECUTED) and (call_count == 2)

    all_pass = pass_r1 and pass_r2 and pass_r3
    print(f"  First execution (allowed)        -> Success={r1.success}, Reason={r1.reason.value:<16} [{'PASS' if pass_r1 else 'FAIL'}]")
    print(f"  Immediate repeat (blocked)       -> Success={r2.success}, Reason={r2.reason.value:<16} [{'PASS' if pass_r2 else 'FAIL'}]")
    print(f"  Post-cooldown (allowed)          -> Success={r3.success}, Reason={r3.reason.value:<16} [{'PASS' if pass_r3 else 'FAIL'}]")
    print(f"  Total handler calls: {call_count} (expected 2)")
    return all_pass


def run_error_handling_validation() -> Tuple[bool, bool]:
    """Validates fail-safe error handling on missing handlers and handler exceptions."""
    print("\n" + "=" * 65)
    print(" 3. ERROR BOUNDARY & MISSING HANDLER VALIDATION")
    print("=" * 65)

    executor = ActionExecutor()

    # 1. Missing handler
    unreg_intent = IntentResult(
        gesture="Pointing",
        intent=Intent.NEXT,
        context=Context.MEDIA,
        confidence=0.95,
        accepted=True,
        reason=IntentReason.CONFIRMED,
    )
    r_missing = executor.execute(unreg_intent)
    pass_missing = (not r_missing.success) and (r_missing.reason == ActionReason.NO_HANDLER)
    print(f"  Missing Handler                  -> Success={r_missing.success}, Reason={r_missing.reason.value:<16} [{'PASS' if pass_missing else 'FAIL'}]")

    # 2. Handler exception
    def crashing_handler():
        raise RuntimeError("simulated action failure")

    executor.register(Intent.NEXT, crashing_handler, action_name="crashing_next")
    r_crash = executor.execute(unreg_intent)
    pass_crash = (
        (not r_crash.success)
        and (r_crash.reason == ActionReason.EXECUTION_FAILED)
        and ("simulated action failure" in r_crash.message)
    )
    print(f"  Handler Exception Caught         -> Success={r_crash.success}, Reason={r_crash.reason.value:<16} [{'PASS' if pass_crash else 'FAIL'}]")

    return pass_missing, pass_crash


def run_presentation_actions_dry_run() -> Dict[str, Tuple[str, bool]]:
    """Inspects presentation handlers and validates dry-run mapping without sending keystrokes."""
    print("\n" + "=" * 65)
    print(" 4. PRESENTATION ACTIONS (DRY-RUN INSPECTION)")
    print("=" * 65)

    executor = ActionExecutor()
    register_presentation_handlers(executor, dry_run=True)

    intent_details = [
        (Intent.START_STOP, "Cmd+Opt+P (macOS) / F5 (PC)", "start_stop_presentation"),
        (Intent.NEXT_SLIDE, "Right Arrow", "next_slide"),
        (Intent.PREVIOUS_SLIDE, "Left Arrow", "previous_slide"),
        (Intent.ZOOM_IN, "Cmd+'+' (macOS) / Ctrl+'+'", "zoom_in"),
        (Intent.ZOOM_OUT, "Cmd+'-' (macOS) / Ctrl+'-'", "zoom_out"),
    ]

    results: Dict[str, Tuple[str, bool]] = {}
    for intent, shortcut, name in intent_details:
        res = IntentResult(
            gesture="TestGesture",
            intent=intent,
            context=Context.PRESENTATION,
            confidence=0.95,
            accepted=True,
            reason=IntentReason.CONFIRMED,
        )
        out = executor.execute(res)
        ok = out.success and (out.reason == ActionReason.EXECUTED) and ("[DRY RUN]" in out.message)
        results[intent.value] = (shortcut, ok)
        print(f"  {intent.value:<16} -> Target: {shortcut:<26} [{'PASS' if ok else 'FAIL'}]")
        print(f"    Message: {out.message}")

    return results


def run_media_actions_validation(live_mode: bool = False) -> Dict[str, Tuple[str, bool, str]]:
    """
    Validates MEDIA context actions.
    If live_mode=False: runs in safe dry-run mode.
    If live_mode=True: executes real system actions once, measuring observable volume.
    """
    mode_label = "LIVE EXECUTION" if live_mode else "DRY-RUN EXECUTION"
    print("\n" + "=" * 65)
    print(f" 5. MEDIA ACTIONS VALIDATION ({mode_label})")
    print("=" * 65)

    executor = ActionExecutor()
    register_media_handlers(executor, dry_run=not live_mode)

    results: Dict[str, Tuple[str, bool, str]] = {}
    intents = [
        (Intent.VOLUME_UP, "volume_up", "ThumbsUp"),
        (Intent.VOLUME_DOWN, "volume_down", "ThumbsDown"),
        (Intent.PLAY_PAUSE, "play_pause", "Fist"),
        (Intent.NEXT, "next_track", "Pointing"),
        (Intent.PREVIOUS, "previous_track", "Peace"),
    ]

    for intent, action_name, gesture in intents:
        # For volume tests in live mode, measure before and after
        vol_before = get_macos_volume() if (live_mode and "volume" in action_name) else None

        intent_res = IntentResult(
            gesture=gesture,
            intent=intent,
            context=Context.MEDIA,
            confidence=0.96,
            accepted=True,
            reason=IntentReason.CONFIRMED,
        )

        out = executor.execute(intent_res)
        vol_after = get_macos_volume() if (live_mode and "volume" in action_name) else None

        notes = out.message
        if vol_before is not None and vol_after is not None:
            notes = f"Volume: {vol_before}% -> {vol_after}% (delta: {vol_after - vol_before:+d}%)"

        ok = out.success and (out.reason == ActionReason.EXECUTED)
        results[intent.value] = (action_name, ok, notes)
        status = "PASS" if ok else "FAIL"
        print(f"  {intent.value:<14} -> Handler: {action_name:<16} [{status}]")
        print(f"    Result Message: {notes}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="IntentSense Phase 6.1 Action Validation Harness"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Explicitly enable real live OS action execution (WARNING: executes real system volume/media actions).",
    )
    args = parser.parse_args()

    print("\n" + "#" * 65)
    print("  IntentSense — Phase 6.1 Action Validation Harness")
    print(f"  Mode: {'*** LIVE OS ACTIONS ENABLED ***' if args.live else 'DRY-RUN (Safe Default)'}")
    print("  Platform:", sys.platform)
    print("#" * 65)

    # 1. Safety Gates
    safety_results = run_safety_gate_validation()

    # 2. Cooldown
    cooldown_ok = run_cooldown_validation()

    # 3. Error Handling
    missing_ok, crash_ok = run_error_handling_validation()

    # 4. Presentation Dry-Run
    pres_results = run_presentation_actions_dry_run()

    # 5. Media Actions (Dry-run or Live)
    media_results = run_media_actions_validation(live_mode=args.live)

    # Overall evaluation
    all_safety_pass = all(safety_results.values())
    all_pres_pass = all(ok for _, ok in pres_results.values())
    all_media_pass = all(ok for _, ok, _ in media_results.values())

    overall_pass = all_safety_pass and cooldown_ok and missing_ok and crash_ok and all_pres_pass and all_media_pass

    print("\n" + "=" * 65)
    print(" SUMMARY OF VALIDATION RESULTS")
    print("=" * 65)
    print(f"  Safety Gate Rejections:     {'PASS' if all_safety_pass else 'FAIL'} ({sum(safety_results.values())}/{len(safety_results)})")
    print(f"  Action Cooldown/Debounce:   {'PASS' if cooldown_ok else 'FAIL'}")
    print(f"  Missing Handler Safety:     {'PASS' if missing_ok else 'FAIL'}")
    print(f"  Exception Error Boundary:   {'PASS' if crash_ok else 'FAIL'}")
    print(f"  Presentation Dry-Run Map:   {'PASS' if all_pres_pass else 'FAIL'} ({len(pres_results)}/{len(pres_results)})")
    print(f"  Media Actions ({'LIVE' if args.live else 'DRY RUN'}):   {'PASS' if all_media_pass else 'FAIL'} ({len(media_results)}/{len(media_results)})")
    print("-" * 65)
    print(f"  OVERALL RESULT:             {'ALL CHECKS PASSED ✅' if overall_pass else 'FAILURES DETECTED ❌'}")
    print("=" * 65 + "\n")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
