#!/usr/bin/env python3
"""
app.py
======

IntentSense — Unified End-to-End Live Application Entry Point.

Orchestrates the complete real-time gesture interaction pipeline:
  Webcam (OpenCV)
      │
      ▼
  MediaPipe Hand Keypoint Detection (detector.py)
      │
      ▼
  Normalized Feature Extraction (core/feature_extractor.py)
      │
      ▼
  Random Forest Model Inference (models/gesture_model.pkl)
      │
      ▼
  Confidence Evaluation & Temporal Stability (core/temporal_filter.py)
      │
      ▼
  Minimum Continuous Hold Duration (0.5s / 1.0s)
      │
      ▼
  Safety & Activation State Machine (core/activation_state.py)
      │
      ▼
  Operational Context Manager (core/context_manager.py)
      │
      ▼
  Semantic Intent Engine (core/intent_engine.py)
      │
      ▼
  Modular Action Executor (core/action_executor.py)
      │
      ▼
  System Action (actions/media_actions.py & actions/presentation_actions.py)

Safety & Modes:
- DEFAULTS TO SAFE DRY-RUN MODE:
    ./venv/bin/python app.py
  Real actions will NEVER execute unless --live is explicitly specified.
- LIVE OS EXECUTION MODE:
    ./venv/bin/python app.py --live

Keyboard Controls:
- '1' : Set Context -> MEDIA
- '2' : Set Context -> PRESENTATION
- '3' : Set Context -> CUSTOM
- 'L' : Manually Lock system (ACTIVE -> LOCKED)
- 'R' : Reset temporal & activation state
- 'Q' : Quit application
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from actions.media_actions import register_media_handlers
from actions.presentation_actions import register_presentation_handlers
from core.action_executor import (
    ActionExecutor,
    ActionReason,
    ActionResult,
)
from core.activation_state import (
    ActivationResult,
    ActivationStateMachine,
    SystemState,
)
from core.adaptive_threshold import (
    AdaptiveThresholdConfig,
    AdaptiveThresholdPolicy,
)
from core.context_manager import Context, ContextManager
from core.feature_extractor import HandFeatureExtractor
from core.intent_engine import (
    Intent,
    IntentEngine,
    IntentReason,
    IntentResult,
)
from core.swipe_detector import (
    SwipeConfig,
    SwipeDetector,
    SwipeDirection,
    SwipeResult,
)
from core.temporal_filter import (
    TemporalFilterResult,
    predict_with_confidence,
)
from core.decision_explainer import (
    DecisionCategory,
    DecisionExplainer,
    DecisionMessage,
    EventSnapshot,
)
from core.personalized_gestures import (
    ArbitrationResult,
    CustomMatchResult,
    PersonalizedGestureManager,
    RESERVED_NAMES,
    SUPPORTED_ACTION_MAP,
    format_enrollment_error,
    validate_enrollment_samples,
)
from detector import HandDetector
from enum import Enum


class AppUIMode(str, Enum):
    LIVE = "LIVE"
    ENROLL_NAME = "ENROLL_NAME"
    ENROLL_CAPTURE = "ENROLL_CAPTURE"
    ENROLL_ACTION = "ENROLL_ACTION"
    ENROLL_CONFIRM = "ENROLL_CONFIRM"
    MANAGE_MENU = "MANAGE_MENU"
    TEST_MODE = "TEST_MODE"
    HELP = "HELP"


SUPPORTED_ACTIONS_LIST: List[Tuple[str, str]] = [
    ("play_pause", "Play / Pause Media"),
    ("next_track", "Next Track"),
    ("previous_track", "Previous Track"),
    ("volume_up", "Volume Up"),
    ("volume_down", "Volume Down"),
    ("start_stop_presentation", "Start / Stop Slideshow"),
    ("next_slide", "Next Slide"),
    ("previous_slide", "Previous Slide"),
    ("zoom_in", "Zoom In"),
    ("zoom_out", "Zoom Out"),
]

# ---------------------------------------------------------------------------
# Constants & Configuration Defaults
# ---------------------------------------------------------------------------
MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "gesture_model.pkl")

CONFIDENCE_THRESHOLD: float = 0.80
STABLE_FRAMES: int = 5
COOLDOWN_FRAMES: int = 15
COMMAND_HOLD_DURATION: float = 0.5    # Hold duration for normal commands (s)
ACTIVATION_HOLD_DURATION: float = 1.0  # Deliberate hold for OpenPalm activation (s)
INACTIVITY_TIMEOUT: float = 30.0      # Inactivity window before returning to LOCKED (s)
ACTION_COOLDOWN: float = 0.8          # Rate-limit between identical executed actions (s)
EVENT_DISPLAY_DURATION: float = 2.0   # Display persistence duration for important events (s)


def load_model_bundle(path: str):
    """Loads the Random Forest model bundle dictionary from disk."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Trained gesture model bundle not found at {path}. "
            f"Please run train_model.py first."
        )
    return joblib.load(path)


def draw_hud(
    frame: np.ndarray,
    state: SystemState,
    context: Context,
    gesture: Optional[str],
    confidence: float,
    consecutive_frames: int,
    target_frames: int,
    hold_duration: float,
    target_hold: float,
    intent: Optional[Intent],
    action_name: Optional[str],
    action_status: str,
    time_remaining: float,
    last_action_event: Optional[Tuple[str, str, float]],
    fps: float,
    live_mode: bool,
    swipe_status: str = "READY",
    decision: Optional[DecisionMessage] = None,
) -> None:
    """
    Renders an informative real-time HUD showing state, intent, swipe, action outcome,
    and semantic decision explanation within fixed 420x310 dimensions.
    """
    h, w = frame.shape[:2]

    # Semi-transparent background overlay
    overlay = frame.copy()
    panel_w, panel_h = 420, 310
    cv2.rectangle(overlay, (14, 14), (14 + panel_w, 14 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # State-colored border
    border_color = (0, 220, 0) if state == SystemState.ACTIVE else (0, 140, 255)
    cv2.rectangle(frame, (14, 14), (14 + panel_w, 14 + panel_h), border_color, 2)

    # 1. Header: INTENTSENSE + Mode
    mode_text = "[LIVE]" if live_mode else "[DRY-RUN]"
    mode_color = (0, 0, 255) if live_mode else (200, 200, 200)
    cv2.putText(
        frame,
        "INTENTSENSE",
        (26, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        mode_text,
        (205, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        mode_color,
        2,
        cv2.LINE_AA,
    )

    y = 66
    line_h = 20

    # 2. STATE
    state_color = (0, 255, 0) if state == SystemState.ACTIVE else (0, 165, 255)
    state_str = f"{state.value}"
    if state == SystemState.ACTIVE and time_remaining > 0:
        state_str += f" ({time_remaining:.1f}s remaining)"
    elif state == SystemState.LOCKED:
        state_str += " (Hold OpenPalm to unlock)"
    cv2.putText(frame, f"STATE       : {state_str}", (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, state_color, 1, cv2.LINE_AA)
    y += line_h

    # 3. CONTEXT
    cv2.putText(frame, f"CONTEXT     : {context.value}", (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
    y += line_h

    # 4. GESTURE
    g_display = gesture if gesture else "NONE"
    cv2.putText(frame, f"GESTURE     : {g_display}", (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    y += line_h

    # 5. CONFIDENCE
    conf_pct = int(confidence * 100)
    conf_color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (180, 180, 180)
    cv2.putText(frame, f"CONFIDENCE  : {conf_pct}%", (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, conf_color, 1, cv2.LINE_AA)
    y += line_h

    # 6. STABILITY & HOLD
    stability_str = f"{consecutive_frames}/{target_frames} frames | {hold_duration:.2f}s/{target_hold:.2f}s"
    cv2.putText(frame, f"STABILITY   : {stability_str}", (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    y += line_h

    # 7. INTENT
    intent_str = intent.value if intent else "NONE"
    cv2.putText(frame, f"INTENT      : {intent_str}", (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 0), 1, cv2.LINE_AA)
    y += line_h

    # 8. ACTION
    act_str = action_name if action_name else "NONE"
    cv2.putText(frame, f"ACTION      : {act_str}", (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    y += line_h

    # 9. ACTION STATUS
    status_color = (0, 255, 0) if action_status == "EXECUTED" else (180, 180, 180)
    if "LOCKED" in action_status or "REJECTED" in action_status:
        status_color = (0, 165, 255)
    elif "FAIL" in action_status:
        status_color = (0, 0, 255)
    cv2.putText(frame, f"STATUS      : {action_status}", (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1, cv2.LINE_AA)
    y += line_h

    # 10. SWIPE STATUS
    swipe_color = (0, 255, 0) if ("LEFT" in swipe_status or "RIGHT" in swipe_status) else (180, 180, 180)
    if "COOLDOWN" in swipe_status:
        swipe_color = (0, 165, 255)
    cv2.putText(frame, f"SWIPE       : {swipe_status}", (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, swipe_color, 1, cv2.LINE_AA)
    y += line_h

    # 11. DECISION EXPLANATION
    dec_text = decision.text if decision is not None else "IDLE — Waiting for hand"
    dec_color = decision.color_bgr if decision is not None else (180, 180, 180)
    cv2.putText(frame, f"DECISION    : {dec_text}", (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, dec_color, 1, cv2.LINE_AA)
    y += line_h + 8

    # Controls footer inside panel
    cv2.putText(
        frame,
        "1:Media | 2:Pres | 3:Custom | L:Lock | R:Reset | Q:Quit",
        (26, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    # Active Command Banner (if recently executed)
    if last_action_event is not None:
        act_ev_name, act_ev_status, act_ev_time = last_action_event
        if (time.monotonic() - act_ev_time) < EVENT_DISPLAY_DURATION:
            banner_bg = frame.copy()
            cv2.rectangle(banner_bg, (w // 2 - 220, 20), (w // 2 + 220, 75), (0, 0, 0), -1)
            cv2.addWeighted(banner_bg, 0.8, frame, 0.2, 0, frame)
            banner_text = f">> ACTION: {act_ev_name} [{act_ev_status}] <<"
            cv2.putText(
                frame,
                banner_text,
                (w // 2 - 205, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # FPS counter in top-right
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 120, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_custom_mode_banner(frame: np.ndarray, custom_count: int) -> None:
    """Renders a helpful discovery banner when in CUSTOM context."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    banner_w, banner_h = 420, 64
    bx, by = 14, 334
    cv2.rectangle(overlay, (bx, by), (bx + banner_w, by + banner_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (bx, by), (bx + banner_w, by + banner_h), (0, 215, 255), 1)

    cv2.putText(
        frame,
        f"CUSTOM MODE ({custom_count} Enrolled)",
        (bx + 12, by + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "[E] Enroll  |  [M] Manage  |  [T] Test  |  [H] Help",
        (bx + 12, by + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_modal_backdrop(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    border_color: Tuple[int, int, int] = (0, 215, 255),
    title: str = "",
) -> None:
    """Draws a semi-transparent glassmorphism modal card on top of the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 2)
    if title:
        cv2.putText(
            frame,
            title,
            (x + 20, y + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            border_color,
            2,
            cv2.LINE_AA,
        )
        cv2.line(frame, (x + 20, y + 46), (x + w - 20, y + 46), (80, 80, 80), 1)


def draw_help_overlay(frame: np.ndarray) -> None:
    """Renders a comprehensive, user-friendly help guide for IntentSense and Custom Mode."""
    h, w = frame.shape[:2]
    mw, mh = 700, 460
    mx = max(20, (w - mw) // 2)
    my = max(20, (h - mh) // 2)

    draw_modal_backdrop(frame, mx, my, mw, mh, border_color=(0, 215, 255), title="INTENTSENSE USER GUIDE & CUSTOM MODE HELP")

    y = my + 72
    cv2.putText(frame, "Control your Mac with hand gestures naturally and safely.", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    y += 28

    cv2.putText(frame, "1. ACTIVATION & LOCKING:", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, "   - Hold OpenPalm for 1.0s to unlock the system (ACTIVE state).", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    y += 20
    cv2.putText(frame, "   - Press [L] or stay idle for 30s to return to LOCKED state.", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    y += 28

    cv2.putText(frame, "2. OPERATIONAL CONTEXTS:", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, "   - Press [1]: MEDIA mode (Fist=Play/Pause, Peace=Prev, Point=Next, Thumbs=Vol)", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    y += 20
    cv2.putText(frame, "   - Press [2]: PRESENTATION mode (Fist=Start/Stop, Peace/Point=Slides, Thumbs=Zoom)", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    y += 20
    cv2.putText(frame, "   - Press [3]: CUSTOM mode (Use your own personalized gestures!)", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    y += 28

    cv2.putText(frame, "3. CUSTOM GESTURE WORKFLOW:", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 0), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, "   - [E] Enroll: Name pose -> 3 Guided Takes (Burst 1, 2, Verify) -> Assign Action", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    y += 20
    cv2.putText(frame, "   - [M] Manage: View gestures, remap actions, re-enroll poses, or delete [D]", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    y += 20
    cv2.putText(frame, "   - [T] Safe Test: Live pose distance & confidence check (Zero OS actions fired)", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    y += 36

    cv2.putText(frame, "Press [ESC] or [ENTER] to close this help guide", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1, cv2.LINE_AA)


def draw_enroll_name_overlay(frame: np.ndarray, typing_buffer: str, error_msg: str) -> None:
    """Renders Step 1: Gesture Name Entry modal."""
    h, w = frame.shape[:2]
    mw, mh = 640, 260
    mx = max(20, (w - mw) // 2)
    my = max(40, (h - mh) // 2)

    draw_modal_backdrop(frame, mx, my, mw, mh, border_color=(0, 215, 255), title="ENROLL CUSTOM GESTURE (Step 1/3: Name)")

    cv2.putText(
        frame,
        "Enter a name for your custom gesture:",
        (mx + 24, my + 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Input box
    box_x, box_y, box_w, box_h = mx + 24, my + 104, mw - 48, 44
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (45, 45, 45), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (120, 120, 120), 1)

    display_text = f"> {typing_buffer}_"
    cv2.putText(
        frame,
        display_text,
        (box_x + 14, box_y + 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if error_msg:
        cv2.putText(
            frame,
            f"Note: {error_msg}",
            (mx + 24, my + 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 140, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "Controls: Type (letters/numbers) | [ENTER] Continue | [ESC] Cancel",
        (mx + 24, my + 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


def draw_enroll_capture_overlay(
    frame: np.ndarray,
    gesture_name: str,
    take_num: int,
    take_frames: int,
    target_frames: int,
    substep: str,
    countdown_sec: float,
    dispersion: float,
    error_msg: str,
    hand_present: bool,
) -> None:
    """Renders Step 2: Interactive 3-Take Live Capture modal."""
    h, w = frame.shape[:2]
    mw, mh = 680, 290
    mx = max(20, (w - mw) // 2)
    my = 20  # Position at top so hand in center is clearly visible

    border_color = (0, 215, 255)
    if substep == "VALIDATION_PASSED":
        border_color = (0, 220, 0)
    elif substep == "VALIDATION_FAILED":
        border_color = (0, 80, 255)

    draw_modal_backdrop(frame, mx, my, mw, mh, border_color=border_color, title=f"CAPTURE GESTURE: '{gesture_name}' (Step 2/3)")

    y = my + 76
    if substep == "READY":
        cv2.putText(frame, "Form your hand gesture in front of the camera.", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
        y += 28
        h_str = "Hand Detected - Ready!" if hand_present else "Waiting for hand detection..."
        h_color = (0, 255, 0) if hand_present else (0, 165, 255)
        cv2.putText(frame, h_str, (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, h_color, 1, cv2.LINE_AA)
        y += 40
        cv2.putText(frame, "Press [SPACE] to begin 3-take capture.", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 215, 255), 2, cv2.LINE_AA)
        y += 44
        cv2.putText(frame, "[ESC] Cancel Enrollment", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (160, 160, 160), 1, cv2.LINE_AA)

    elif substep == "RECORDING":
        take_label = "Training Burst 1" if take_num == 1 else ("Training Burst 2" if take_num == 2 else "Holdout Validation")
        cv2.putText(frame, f"Capturing Take {take_num}/3 ({take_label})", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 215, 255), 2, cv2.LINE_AA)
        y += 30
        h_status = "HOLD HAND STEADY..." if hand_present else "PLEASE KEEP HAND IN FRAME!"
        h_color = (0, 255, 0) if hand_present else (0, 0, 255)
        cv2.putText(frame, h_status, (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, h_color, 1, cv2.LINE_AA)
        y += 32

        # Progress bar
        bar_w, bar_h = mw - 48, 24
        cv2.rectangle(frame, (mx + 24, y), (mx + 24 + bar_w, y + bar_h), (50, 50, 50), -1)
        progress = min(1.0, max(0.0, take_frames / max(1, target_frames)))
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (mx + 24, y), (mx + 24 + fill_w, y + bar_h), (0, 215, 255), -1)
        cv2.putText(frame, f"{take_frames}/{target_frames} frames", (mx + 34, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
        y += 44
        cv2.putText(frame, "[ESC] Cancel Enrollment", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

    elif substep in ("PAUSE_1", "PAUSE_2"):
        completed_take = 1 if substep == "PAUSE_1" else 2
        next_take = 2 if substep == "PAUSE_1" else 3
        cv2.putText(frame, f"Take {completed_take}/3 Captured Successfully!", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 0), 2, cv2.LINE_AA)
        y += 30
        cv2.putText(frame, "Relax your hand slightly, then re-form pose for next take.", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        y += 36
        rem_sec = max(0.0, 2.0 - countdown_sec)
        cv2.putText(frame, f"Next take ({next_take}/3) starting in: {rem_sec:.1f}s ...", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 215, 255), 2, cv2.LINE_AA)

    elif substep == "VALIDATING":
        cv2.putText(frame, "Analyzing 4-Gate Stability & Conflict Validation...", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 215, 255), 2, cv2.LINE_AA)

    elif substep == "VALIDATION_PASSED":
        cv2.putText(frame, "SUCCESS: 3-Take Enrollment Validated!", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)
        y += 30
        cv2.putText(frame, f"Pose Stability Dispersion: {dispersion:.3f} (Max: 0.120) [PASSED]", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        y += 24
        cv2.putText(frame, "Holdout Verification     : Cross-validation matched [PASSED]", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        y += 36
        cv2.putText(frame, "Press [ENTER] or [SPACE] to Assign Action", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)
        y += 36
        cv2.putText(frame, "[R] Retake Capture  |  [ESC] Cancel", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1, cv2.LINE_AA)

    elif substep == "VALIDATION_FAILED":
        cv2.putText(frame, "ENROLLMENT VALIDATION FAILED", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 80, 255), 2, cv2.LINE_AA)
        y += 30
        clean_err = error_msg if error_msg else "Sample was unstable or conflicted."
        cv2.putText(frame, f"Feedback: {clean_err[:70]}", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        if len(clean_err) > 70:
            y += 22
            cv2.putText(frame, clean_err[70:140], (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        y += 40
        cv2.putText(frame, "Press [R] to Retry Capture  |  [ESC] Cancel", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 215, 255), 2, cv2.LINE_AA)


def draw_enroll_action_overlay(
    frame: np.ndarray,
    gesture_name: str,
    action_list: Sequence[Tuple[str, str]],
    selected_action: str,
    selected_ctx: str,
) -> None:
    """Renders Step 3: Action & Context Selection modal."""
    h, w = frame.shape[:2]
    mw, mh = 680, 380
    mx = max(20, (w - mw) // 2)
    my = max(20, (h - mh) // 2)

    draw_modal_backdrop(frame, mx, my, mw, mh, border_color=(0, 215, 255), title=f"ASSIGN ACTION FOR '{gesture_name}' (Step 3/3)")

    y = my + 72
    cv2.putText(frame, "Select an action to trigger (Press [1]-[9], [0]):", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    y += 24

    col1_x = mx + 24
    col2_x = mx + 350
    for idx, (act_id, act_label) in enumerate(action_list[:10]):
        key_num = (idx + 1) % 10
        is_sel = (act_id == selected_action)
        col_x = col1_x if idx < 5 else col2_x
        row_y = y + (idx % 5) * 26
        prefix = ">> " if is_sel else "   "
        text_color = (0, 255, 0) if is_sel else (200, 200, 200)
        cv2.putText(
            frame,
            f"{prefix}[{key_num}] {act_label}",
            (col_x, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            text_color,
            1,
            cv2.LINE_AA,
        )

    y += 5 * 26 + 18
    # Context selector
    ctx_str = f"Context Target: [{selected_ctx}]  — Press [C] for CUSTOM only | [A] for ALL contexts"
    cv2.putText(frame, ctx_str, (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1, cv2.LINE_AA)
    y += 28

    sel_label = selected_action
    for act_id, act_lbl in action_list:
        if act_id == selected_action:
            sel_label = act_lbl
            break

    cv2.putText(
        frame,
        f"Selected: {sel_label} -> Scope: {selected_ctx}",
        (mx + 24, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y += 32

    cv2.putText(
        frame,
        "Press [ENTER] Proceed to Confirmation  |  [ESC] Cancel",
        (mx + 24, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def draw_enroll_confirm_overlay(
    frame: np.ndarray,
    gesture_name: str,
    action_name: str,
    ctx_name: str,
    dispersion: float,
    error_msg: str = "",
) -> None:
    """Renders Confirmation modal before saving."""
    h, w = frame.shape[:2]
    mw, mh = 640, 280
    mx = max(20, (w - mw) // 2)
    my = max(30, (h - mh) // 2)

    draw_modal_backdrop(frame, mx, my, mw, mh, border_color=(0, 255, 0), title="CONFIRM & ACTIVATE CUSTOM GESTURE")

    # Resolve display label
    display_act = action_name
    for act_id, act_lbl in SUPPORTED_ACTIONS_LIST:
        if act_id == action_name:
            display_act = act_lbl
            break

    y = my + 78
    cv2.putText(frame, f"Gesture Name : {gesture_name}", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    y += 28
    cv2.putText(frame, f"Target Action: {display_act}", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
    y += 28
    cv2.putText(frame, f"Context Scope: {ctx_name}", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    y += 28
    cv2.putText(frame, f"Stability    : Passed (Dispersion: {dispersion:.3f} <= 0.120)", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 1, cv2.LINE_AA)
    y += 38

    if error_msg:
        cv2.putText(frame, f"Error: {error_msg}", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 80, 255), 1, cv2.LINE_AA)
        y += 26

    cv2.putText(
        frame,
        "Press [Y] Save & Activate Immediately  |  [N/ESC] Cancel",
        (mx + 24, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def draw_manage_overlay(
    frame: np.ndarray,
    profiles: Sequence[Any],
    selected_idx: int,
    delete_confirm_name: Optional[str] = None,
) -> None:
    """Renders Custom Gestures Management modal."""
    h, w = frame.shape[:2]
    mw, mh = 680, 400
    mx = max(20, (w - mw) // 2)
    my = max(20, (h - mh) // 2)

    title_str = f"GESTURE MANAGER ({len(profiles)} Enrolled)"
    draw_modal_backdrop(frame, mx, my, mw, mh, border_color=(0, 215, 255), title=title_str)

    y = my + 72
    if delete_confirm_name:
        # Deletion confirmation card
        cv2.putText(frame, f"CONFIRM DELETION OF '{delete_confirm_name}'", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 80, 255), 2, cv2.LINE_AA)
        y += 36
        cv2.putText(frame, "This will permanently remove the gesture and its mappings.", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
        y += 50
        cv2.putText(frame, "Press [Y] Confirm Delete  |  [N/ESC] Cancel", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 215, 255), 2, cv2.LINE_AA)
        return

    if not profiles:
        cv2.putText(frame, "No custom gestures enrolled yet.", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)
        y += 32
        cv2.putText(frame, "Press [E] to enroll your first custom gesture!", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 0), 2, cv2.LINE_AA)
        y += 60
        cv2.putText(frame, "[ESC] Return to Live", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
        return

    cv2.putText(frame, "Enrolled Custom Gestures:", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    y += 24

    for idx, p in enumerate(profiles[:8]):
        is_sel = (idx == selected_idx)
        map_str = ", ".join(f"{k}:{v}" for k, v in p.mappings.items()) if p.mappings else "No mappings"
        prefix = ">> " if is_sel else "   "
        line_color = (0, 255, 0) if is_sel else (220, 220, 220)
        cv2.putText(
            frame,
            f"{prefix}[{idx + 1}] {p.name:<12} -> {map_str}",
            (mx + 24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            line_color,
            1,
            cv2.LINE_AA,
        )
        y += 24

    y = my + mh - 70
    cv2.line(frame, (mx + 20, y), (mx + mw - 20, y), (70, 70, 70), 1)
    y += 24
    cv2.putText(frame, "Controls: [1-8] Select  |  [T] Test  |  [C] Remap  |  [R] Re-enroll  |  [D] Delete", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 215, 255), 1, cv2.LINE_AA)
    y += 24
    cv2.putText(frame, "[E] Enroll New  |  [ESC] Return to Live", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1, cv2.LINE_AA)


def draw_test_overlay(
    frame: np.ndarray,
    match_result: Optional[CustomMatchResult],
    consec_frames: int,
    target_frames: int,
    hand_present: bool,
) -> None:
    """Renders Isolated Live Test Mode overlay (Zero Actions Dispatched)."""
    h, w = frame.shape[:2]
    mw, mh = 680, 310
    mx = max(20, (w - mw) // 2)
    my = max(20, (h - mh) // 2)

    draw_modal_backdrop(frame, mx, my, mw, mh, border_color=(255, 180, 0), title="ISOLATED GESTURE TEST (ZERO OS ACTIONS FIRED)")

    y = my + 78
    if not hand_present:
        cv2.putText(frame, "SHOW YOUR HAND IN FRAME TO TEST RECOGNITION", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 1, cv2.LINE_AA)
        y += 36
        cv2.putText(frame, "No hand detected. Pose hand in front of camera.", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    else:
        matched_g = match_result.gesture if (match_result and match_result.is_match) else "NONE"
        dist = match_result.distance if match_result else 999.0
        conf = match_result.confidence if match_result else 0.0

        cv2.putText(frame, f"Detected Pose : {matched_g}", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28
        cv2.putText(frame, f"Distance      : {dist:.3f} (Entry requires <= 0.233)", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)
        y += 28
        cv2.putText(frame, f"Confidence    : {conf * 100:.1f}% (Entry requires >= 80%)", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0) if conf >= 0.80 else (200, 200, 200), 1, cv2.LINE_AA)
        y += 32

        # Status badge
        if match_result and match_result.is_match and conf >= 0.80:
            badge_text = "STATUS: PASSING ENTRY GATE (>= 80% CONFIDENCE)"
            badge_color = (0, 255, 0)
        elif match_result and match_result.is_match:
            badge_text = "STATUS: WEAK MATCH (70% <= C < 80%) - HOLD CLOSER"
            badge_color = (0, 215, 255)
        else:
            badge_text = "STATUS: UNKNOWN POSE / OUT OF BOUNDS"
            badge_color = (0, 80, 255)

        cv2.putText(frame, badge_text, (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, badge_color, 2, cv2.LINE_AA)
        y += 28
        cv2.putText(frame, f"Stability Hold: {consec_frames}/{target_frames} frames", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)

    y = my + mh - 30
    cv2.putText(frame, "Zero OS actions will execute | Press [ESC] or [Q] to exit Test Mode", (mx + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1, cv2.LINE_AA)


def run_smoke_test() -> bool:
    """Runs headless end-to-end component orchestration smoke test."""
    print("\n[Smoke Test] Initializing IntentSense end-to-end components...")

    bundle = load_model_bundle(MODEL_PATH)
    extractor = HandFeatureExtractor()
    context_manager = ContextManager()
    intent_engine = IntentEngine(context_manager=context_manager)
    action_executor = ActionExecutor(action_cooldown=0.0)
    register_media_handlers(action_executor, dry_run=True)
    register_presentation_handlers(action_executor, dry_run=True)

    threshold_policy = AdaptiveThresholdPolicy(
        AdaptiveThresholdConfig(
            default_entry_threshold=CONFIDENCE_THRESHOLD,
            default_sustain_threshold=0.75,
            commit_frames=3,
            min_bound=0.72,
            max_bound=0.85,
        )
    )

    activation_sm = ActivationStateMachine(
        activation_gesture="OpenPalm",
        activation_hold_duration=1.0,
        inactivity_timeout=5.0,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        stable_frames=3,
        command_hold_duration=0.5,
        threshold_policy=threshold_policy,
    )
    swipe_detector = SwipeDetector()
    explainer = DecisionExplainer()

    print("  1. Checking initial state is strictly LOCKED...")
    assert activation_sm.state == SystemState.LOCKED
    idle_dec = explainer.explain(state=activation_sm.state, hand_present=False)
    assert idle_dec.category == DecisionCategory.IDLE
    assert "System locked" in idle_dec.text

    # 1. While LOCKED, feeding ThumbsUp should NOT execute any action
    print("  2. Testing command suppression while LOCKED...")
    act_res = activation_sm.process_confirmed("ThumbsUp", is_confirmed=True, hold_duration=1.0, timestamp=0.0)
    intent_res = intent_engine.process(activation_result=act_res, confidence=0.95)
    assert not intent_res.accepted
    assert intent_res.reason == IntentReason.SYSTEM_LOCKED
    act_out = action_executor.execute(intent_res)
    assert not act_out.success
    assert act_out.reason == ActionReason.INTENT_REJECTED
    dec_locked = explainer.explain(
        state=activation_sm.state,
        hand_present=True,
        raw_gesture="ThumbsUp",
        confidence=0.95,
        intent_result=intent_res,
        action_result=act_out,
    )
    assert dec_locked.category == DecisionCategory.REJECTED
    assert "System locked" in dec_locked.text
    print("     ✓ Verified ThumbsUp while LOCKED produced NO action (SYSTEM_LOCKED).")

    # 1b. While LOCKED, dynamic swipe must also be strictly suppressed
    print("  2b. Testing dynamic swipe suppression while LOCKED...")
    locked_swipe = SwipeResult(
        direction=SwipeDirection.RIGHT,
        confidence=0.95,
        velocity=1.5,
        displacement=0.40,
        duration=0.20,
        reason="SWIPE_DETECTED",
    )
    locked_swipe_intent = intent_engine.process_swipe(locked_swipe, is_active=False)
    assert not locked_swipe_intent.accepted
    assert locked_swipe_intent.reason == IntentReason.SYSTEM_LOCKED
    locked_act_out = action_executor.execute(locked_swipe_intent)
    assert not locked_act_out.success
    assert locked_act_out.reason == ActionReason.INTENT_REJECTED
    dec_locked_swipe = explainer.explain(
        state=activation_sm.state,
        hand_present=True,
        swipe_result=locked_swipe,
        intent_result=locked_swipe_intent,
        action_result=locked_act_out,
    )
    assert dec_locked_swipe.category == DecisionCategory.REJECTED
    assert "Swipe RIGHT ignored" in dec_locked_swipe.text
    print("     ✓ Verified Swipe Right while LOCKED produced NO action (SYSTEM_LOCKED).")

    # 2. Feeding OpenPalm for 1.0s activates the system
    print("  3. Testing deliberate OpenPalm activation (LOCKED -> ACTIVE)...")
    act_res = activation_sm.process_confirmed("OpenPalm", is_confirmed=True, hold_duration=1.0, timestamp=1.0)
    assert act_res.state == SystemState.ACTIVE
    assert act_res.just_activated
    # OpenPalm itself is consumed as activation; produces NO action
    intent_res = intent_engine.process(activation_result=act_res, confidence=0.99)
    assert not intent_res.accepted
    dec_unlock = explainer.explain(
        state=activation_sm.state,
        hand_present=True,
        raw_gesture="OpenPalm",
        confidence=0.99,
        activation_result=act_res,
    )
    assert dec_unlock.category == DecisionCategory.CONFIRMED
    assert dec_unlock.text == "CONFIRMED — System unlocked!"
    print("     ✓ Verified OpenPalm transitioned system to ACTIVE with NO command action.")

    # 3. While ACTIVE, feeding ThumbsUp produces VOLUME_UP action
    print("  4. Testing command execution while ACTIVE in MEDIA context...")
    act_res = activation_sm.process_confirmed("ThumbsUp", is_confirmed=True, hold_duration=0.5, timestamp=2.0)
    assert act_res.state == SystemState.ACTIVE
    assert act_res.command == "ThumbsUp"
    intent_res = intent_engine.process(activation_result=act_res, confidence=0.96)
    assert intent_res.accepted
    assert intent_res.intent == Intent.VOLUME_UP
    act_out = action_executor.execute(intent_res)
    assert act_out.success
    assert act_out.reason == ActionReason.EXECUTED
    assert act_out.action_name == "volume_up"
    dec_act = explainer.explain(
        state=activation_sm.state,
        hand_present=True,
        raw_gesture="ThumbsUp",
        confidence=0.96,
        intent_result=intent_res,
        action_result=act_out,
    )
    assert dec_act.category == DecisionCategory.ACTION_EXECUTED
    assert "volume_up" in dec_act.text
    print("     ✓ Verified ThumbsUp executed 'volume_up' action successfully.")

    # 3b. Testing dynamic swipes in MEDIA context
    print("  4b. Testing dynamic swipes (LEFT -> previous_track, RIGHT -> next_track) in MEDIA...")
    swipe_l = SwipeResult(direction=SwipeDirection.LEFT, confidence=0.90, velocity=1.3, reason="SWIPE_DETECTED")
    intent_sl = intent_engine.process_swipe(swipe_l, is_active=True, context=Context.MEDIA)
    assert intent_sl.accepted
    assert intent_sl.intent == Intent.PREVIOUS
    act_sl = action_executor.execute(intent_sl)
    assert act_sl.success
    assert act_sl.action_name == "previous_track"
    dec_sl = explainer.explain(
        state=activation_sm.state,
        hand_present=True,
        swipe_result=swipe_l,
        intent_result=intent_sl,
        action_result=act_sl,
    )
    assert dec_sl.category == DecisionCategory.ACTION_EXECUTED
    assert "previous_track" in dec_sl.text

    swipe_r = SwipeResult(direction=SwipeDirection.RIGHT, confidence=0.92, velocity=1.5, reason="SWIPE_DETECTED")
    intent_sr = intent_engine.process_swipe(swipe_r, is_active=True, context=Context.MEDIA)
    assert intent_sr.accepted
    assert intent_sr.intent == Intent.NEXT
    act_sr = action_executor.execute(intent_sr)
    assert act_sr.success
    assert act_sr.action_name == "next_track"
    dec_sr = explainer.explain(
        state=activation_sm.state,
        hand_present=True,
        swipe_result=swipe_r,
        intent_result=intent_sr,
        action_result=act_sr,
    )
    assert dec_sr.category == DecisionCategory.ACTION_EXECUTED
    assert "next_track" in dec_sr.text
    print("     ✓ Verified Swipes executed 'previous_track' and 'next_track' in MEDIA.")

    # 4. Context switch to PRESENTATION produces NEXT_SLIDE
    print("  5. Testing context switch to PRESENTATION...")
    context_manager.set_context(Context.PRESENTATION)
    act_res = activation_sm.process_confirmed("Pointing", is_confirmed=True, hold_duration=0.5, timestamp=3.0)
    intent_res = intent_engine.process(activation_result=act_res, confidence=0.94)
    assert intent_res.accepted
    assert intent_res.intent == Intent.NEXT_SLIDE
    act_out = action_executor.execute(intent_res)
    assert act_out.success
    assert act_out.action_name == "next_slide"
    dec_pres = explainer.explain(
        state=activation_sm.state,
        hand_present=True,
        raw_gesture="Pointing",
        confidence=0.94,
        intent_result=intent_res,
        action_result=act_out,
    )
    assert dec_pres.category == DecisionCategory.ACTION_EXECUTED
    assert "next_slide" in dec_pres.text
    print("     ✓ Verified Pointing in PRESENTATION executed 'next_slide' action.")

    # 4b. Testing dynamic swipes in PRESENTATION context
    print("  5b. Testing dynamic swipes (LEFT -> previous_slide, RIGHT -> next_slide) in PRESENTATION...")
    intent_pl = intent_engine.process_swipe(swipe_l, is_active=True, context=Context.PRESENTATION)
    assert intent_pl.accepted
    assert intent_pl.intent == Intent.PREVIOUS_SLIDE
    act_pl = action_executor.execute(intent_pl)
    assert act_pl.success
    assert act_pl.action_name == "previous_slide"

    intent_pr = intent_engine.process_swipe(swipe_r, is_active=True, context=Context.PRESENTATION)
    assert intent_pr.accepted
    assert intent_pr.intent == Intent.NEXT_SLIDE
    act_pr = action_executor.execute(intent_pr)
    assert act_pr.success
    assert act_pr.action_name == "next_slide"
    print("     ✓ Verified Swipes executed 'previous_slide' and 'next_slide' in PRESENTATION.")

    # 5c. Testing runtime adaptive/custom threshold explanation formatting
    print("  5c. Testing runtime dynamic threshold explanation formatting...")
    dec_thresh = explainer.explain(
        state=activation_sm.state,
        hand_present=True,
        raw_gesture="Pointing",
        confidence=0.72,
        effective_threshold=0.75,
    )
    assert dec_thresh.category == DecisionCategory.REJECTED
    assert "(72% < 75%)" in dec_thresh.text
    print("     ✓ Verified runtime threshold dynamically formatted without hardcoded values.")

    # 5d. Testing display event persistence (2.0s hold duration)
    print("  5d. Testing display event persistence for important events...")
    live_waiting = DecisionMessage(category=DecisionCategory.WAITING, text="WAITING — Tracking Pointing")
    # Within 2.0s window: returns the important event
    held_dec = explainer.resolve_display_decision(
        live_decision=live_waiting,
        last_event=dec_pres,
        last_event_timestamp=10.0,
        current_timestamp=11.5,
        display_duration=EVENT_DISPLAY_DURATION,
    )
    assert held_dec == dec_pres
    assert held_dec.category == DecisionCategory.ACTION_EXECUTED
    # Beyond 2.0s window: reverts to live telemetry
    expired_dec = explainer.resolve_display_decision(
        live_decision=live_waiting,
        last_event=dec_pres,
        last_event_timestamp=10.0,
        current_timestamp=12.1,
        display_duration=EVENT_DISPLAY_DURATION,
    )
    assert expired_dec == live_waiting
    assert expired_dec.category == DecisionCategory.WAITING

    # Test synchronized EventSnapshot persistence and expiry across all 4 fields
    snap_pres = EventSnapshot(
        decision=dec_pres,
        intent=Intent.NEXT_SLIDE,
        action_name="next_slide",
        action_status="EXECUTED",
    )
    h_dec, h_intent, h_act, h_stat = explainer.resolve_hud_snapshot(
        live_decision=live_waiting,
        live_intent=None,
        live_action_name=None,
        live_action_status="ACTIVE_IDLE",
        last_event=snap_pres,
        last_event_timestamp=10.0,
        current_timestamp=11.5,
        display_duration=EVENT_DISPLAY_DURATION,
    )
    assert h_dec == dec_pres
    assert h_intent == Intent.NEXT_SLIDE
    assert h_act == "next_slide"
    assert h_stat == "EXECUTED"

    # Beyond 2.0s window: all 4 fields revert together to live values
    e_dec, e_intent, e_act, e_stat = explainer.resolve_hud_snapshot(
        live_decision=live_waiting,
        live_intent=None,
        live_action_name=None,
        live_action_status="ACTIVE_IDLE",
        last_event=snap_pres,
        last_event_timestamp=10.0,
        current_timestamp=12.1,
        display_duration=EVENT_DISPLAY_DURATION,
    )
    assert e_dec == live_waiting
    assert e_intent is None
    assert e_act is None
    assert e_stat == "ACTIVE_IDLE"

    # Test is_important_event identification
    assert not explainer.is_important_event(live_waiting)
    print("     ✓ Verified important events persist for 2.0s and smoothly revert to live explanation.")

    # 5e. Testing Personalized Gestures enrollment, arbitration, and execution
    print("  5e. Testing Personalized Gestures enrollment, arbitration, and execution...")
    temp_custom_json = os.path.join(PROJECT_ROOT, "models", "temp_custom_gestures_smoke.json")
    if os.path.exists(temp_custom_json):
        os.remove(temp_custom_json)

    custom_mgr = PersonalizedGestureManager(profile_path=temp_custom_json)
    rock_on_vec = [0.0] * 42
    rock_on_vec[16] = 1.2
    rock_on_vec[40] = 1.1

    custom_mgr.register_gesture(
        name="RockOn",
        centroid=rock_on_vec,
        dispersion=0.05,
        sample_count=30,
        mappings={Context.MEDIA.value: Intent.PLAY_PAUSE.value},
    )
    intent_engine.register_custom_gesture("RockOn", {Context.MEDIA: Intent.PLAY_PAUSE})

    # Test arbitration: built-in model uncertain (< 0.50), custom matches
    arb_custom = custom_mgr.arbitrate(
        builtin_gesture="Fist",
        builtin_confidence=0.25,
        builtin_threshold=0.80,
        features=rock_on_vec,
        hand_present=True,
    )
    assert arb_custom.tier == 3
    assert arb_custom.candidate_gesture == "RockOn"
    assert arb_custom.confidence >= 0.80

    # Advance temporal filter for RockOn until hold duration >= 0.5s is satisfied and confirmed
    activation_sm.command_filter.reset()
    f_res = None
    for i in range(10):
        step_res = activation_sm.command_filter.process("RockOn", arb_custom.confidence, timestamp=4.0 + i * 0.1)
        if step_res.is_confirmed:
            f_res = step_res
            break
    assert f_res is not None
    assert f_res.is_confirmed
    assert f_res.gesture == "RockOn"

    # Route through activation state machine & intent engine
    act_res_cust = activation_sm.process_filter_result(f_res, timestamp=4.6)
    assert act_res_cust.command == "RockOn"

    intent_res_cust = intent_engine.process(
        activation_result=act_res_cust,
        confidence=arb_custom.confidence,
        context=Context.MEDIA,
    )
    assert intent_res_cust.accepted
    assert intent_res_cust.intent == Intent.PLAY_PAUSE

    # Execute action
    act_out_cust = action_executor.execute(intent_res_cust)
    assert act_out_cust.success
    assert act_out_cust.action_name == "play_pause"

    if os.path.exists(temp_custom_json):
        os.remove(temp_custom_json)
    intent_engine.unregister_custom_gesture("RockOn")
    print("     ✓ Verified Personalized Gesture 'RockOn' arbitrated, confirmed, and executed 'play_pause'.")

    # 6. Manual lock
    print("  6. Testing manual lock (ACTIVE -> LOCKED)...")
    activation_sm.lock()
    swipe_detector.reset()
    assert activation_sm.state == SystemState.LOCKED
    dec_locked_end = explainer.explain(state=activation_sm.state, hand_present=False)
    assert dec_locked_end.category == DecisionCategory.IDLE
    assert "System locked" in dec_locked_end.text
    print("     ✓ System successfully locked and swipe detector reset.")

    print("\n✅ End-to-end orchestration smoke test PASSED successfully!\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IntentSense — Unified End-to-End Real-Time Gesture Application"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Enable LIVE system action execution (default: Safe DRY-RUN).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Run headless component orchestration smoke test and exit.",
    )
    args = parser.parse_args()

    if args.smoke:
        success = run_smoke_test()
        sys.exit(0 if success else 1)

    print("=" * 65)
    print("  IntentSense — Unified Live Application")
    print("=" * 65)
    print(f"  Mode         : {'*** LIVE OS ACTIONS ***' if args.live else 'SAFE DRY-RUN (Default)'}")
    print(f"  Platform     : {sys.platform}")
    print(f"  Model Path   : {MODEL_PATH}")
    print(f"  Confidence   : >= {int(CONFIDENCE_THRESHOLD * 100)}%")
    print(f"  Stability    : {STABLE_FRAMES} frames")
    print(f"  Hold Duration: {COMMAND_HOLD_DURATION:.1f}s commands | {ACTIVATION_HOLD_DURATION:.1f}s activation")
    print(f"  Inactivity   : {INACTIVITY_TIMEOUT:.0f}s timeout")
    print(f"  Action CD    : {ACTION_COOLDOWN:.1f}s debounce")
    print("  Controls     : 1=Media | 2=Pres | 3=Custom | L=Lock | R=Reset | Q=Quit")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # 1. Initialize Reusable Components
    # -------------------------------------------------------------------------
    print("Initializing pipeline components...")
    bundle = load_model_bundle(MODEL_PATH)
    detector = HandDetector()
    extractor = HandFeatureExtractor()

    threshold_policy = AdaptiveThresholdPolicy(
        AdaptiveThresholdConfig(
            default_entry_threshold=CONFIDENCE_THRESHOLD,
            default_sustain_threshold=0.75,
            commit_frames=3,
            min_bound=0.72,
            max_bound=0.85,
        )
    )

    activation_sm = ActivationStateMachine(
        activation_gesture="OpenPalm",
        activation_hold_duration=ACTIVATION_HOLD_DURATION,
        inactivity_timeout=INACTIVITY_TIMEOUT,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        stable_frames=STABLE_FRAMES,
        cooldown_frames=COOLDOWN_FRAMES,
        command_hold_duration=COMMAND_HOLD_DURATION,
        threshold_policy=threshold_policy,
    )

    swipe_detector = SwipeDetector()

    context_manager = ContextManager(initial_context=Context.MEDIA)
    intent_engine = IntentEngine(context_manager=context_manager)

    custom_manager = PersonalizedGestureManager()
    for prof in custom_manager.list_gestures():
        intent_engine.register_custom_gesture(prof.name, prof.mappings)

    action_executor = ActionExecutor(action_cooldown=ACTION_COOLDOWN)
    register_media_handlers(action_executor, dry_run=not args.live)
    register_presentation_handlers(action_executor, dry_run=not args.live)
    explainer = DecisionExplainer()

    # -------------------------------------------------------------------------
    # 2. Open Webcam
    # -------------------------------------------------------------------------
    print("Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        print("Please verify camera access in macOS System Settings > Privacy & Security > Camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_action_event: Optional[Tuple[str, str, float]] = None
    last_event_snapshot: Optional[EventSnapshot] = None
    last_event_timestamp: float = 0.0

    # UI Mode & Interactive Enrollment / Management State
    app_ui_mode: AppUIMode = AppUIMode.LIVE
    typing_buffer: str = ""
    typing_error: str = ""
    current_enroll_name: str = ""
    enroll_substep: str = "READY"  # READY, RECORDING, PAUSE_1, PAUSE_2, VALIDATING, VALIDATION_PASSED, VALIDATION_FAILED
    enroll_take_num: int = 1
    enroll_current_take: List[List[float]] = []
    enroll_take1: List[List[float]] = []
    enroll_take2: List[List[float]] = []
    enroll_take3: List[List[float]] = []
    enroll_dispersion: float = 0.0
    enroll_centroid: List[float] = []
    enroll_error: str = ""
    enroll_pause_start: float = 0.0
    selected_action: str = "play_pause"
    selected_ctx: str = "CUSTOM"
    confirm_error: str = ""
    manage_selected_idx: int = 0
    manage_delete_confirm: Optional[str] = None
    manage_remapping: bool = False
    test_consec_frames: int = 0

    print("\n✅ Application running. Focus OpenCV window and use 'Q' to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            now = time.monotonic()

            # -----------------------------------------------------------------
            # Step 1: Hand Detection & Palm Landmark Extraction
            # -----------------------------------------------------------------
            result = detector.detect_hand(frame)
            landmarks = detector.get_landmarks(result, frame.shape)
            hand_present = (landmarks is not None and len(landmarks) == 21)
            norm_features = extractor.extract_features(landmarks) if hand_present else None

            # Always render visual hand skeleton if detected
            if hand_present and result is not None:
                frame = detector.draw_landmarks(frame, result)

            palm_x, palm_y, palm_size = 0.5, 0.5, None
            if hand_present and result is not None and result.hand_landmarks:
                hlm = result.hand_landmarks[0]
                palm_x = (hlm[0].x + hlm[9].x) / 2.0
                palm_y = (hlm[0].y + hlm[9].y) / 2.0
                palm_size = math.hypot(hlm[9].x - hlm[0].x, hlm[9].y - hlm[0].y)

            # -----------------------------------------------------------------
            # Step 2: Branch by App UI Mode
            # -----------------------------------------------------------------
            if app_ui_mode == AppUIMode.HELP:
                draw_help_overlay(frame)
                cv2.imshow("IntentSense — Unified Live Application", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, 13, 32, ord("h"), ord("H")):  # ESC, ENTER, SPACE, H
                    app_ui_mode = AppUIMode.LIVE
                elif key in (ord("q"), ord("Q")):
                    break

            elif app_ui_mode == AppUIMode.ENROLL_NAME:
                draw_enroll_name_overlay(frame, typing_buffer, typing_error)
                cv2.imshow("IntentSense — Unified Live Application", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    app_ui_mode = AppUIMode.LIVE
                elif key in (10, 13):  # ENTER
                    clean_name = typing_buffer.strip()
                    if not clean_name:
                        typing_error = "Please enter a gesture name."
                    elif clean_name in RESERVED_NAMES:
                        typing_error = f"'{clean_name}' is a reserved built-in gesture name."
                    elif clean_name in custom_manager.profiles and not manage_remapping:
                        typing_error = f"Gesture '{clean_name}' already exists. Use Manage (M) or pick another name."
                    else:
                        current_enroll_name = clean_name
                        enroll_substep = "READY"
                        enroll_take_num = 1
                        enroll_current_take = []
                        enroll_take1 = []
                        enroll_take2 = []
                        enroll_take3 = []
                        enroll_error = ""
                        app_ui_mode = AppUIMode.ENROLL_CAPTURE
                elif key in (8, 127):  # Backspace
                    typing_buffer = typing_buffer[:-1]
                    typing_error = ""
                elif 32 <= key <= 126:
                    ch = chr(key)
                    if (ch.isalnum() or ch in "_- ") and len(typing_buffer) < 20:
                        typing_buffer += ch
                        typing_error = ""

            elif app_ui_mode == AppUIMode.ENROLL_CAPTURE:
                if enroll_substep == "RECORDING":
                    if hand_present and norm_features is not None:
                        enroll_current_take.append(norm_features)
                        if len(enroll_current_take) >= 10:
                            if enroll_take_num == 1:
                                enroll_take1 = list(enroll_current_take)
                                enroll_substep = "PAUSE_1"
                                enroll_pause_start = now
                            elif enroll_take_num == 2:
                                enroll_take2 = list(enroll_current_take)
                                enroll_substep = "PAUSE_2"
                                enroll_pause_start = now
                            elif enroll_take_num == 3:
                                enroll_take3 = list(enroll_current_take)
                                enroll_substep = "VALIDATING"

                elif enroll_substep in ("PAUSE_1", "PAUSE_2"):
                    if (now - enroll_pause_start) >= 2.0:
                        enroll_take_num = 2 if enroll_substep == "PAUSE_1" else 3
                        enroll_current_take = []
                        enroll_substep = "RECORDING"

                elif enroll_substep == "VALIDATING":
                    try:
                        existing_centroids = {
                            g_name: g.centroid for g_name, g in custom_manager.profiles.items() if g_name != current_enroll_name
                        }
                        centroid, dispersion = validate_enrollment_samples(
                            take1=enroll_take1,
                            take2=enroll_take2,
                            take3_val=enroll_take3,
                            builtin_classifier=bundle,
                            existing_centroids=existing_centroids,
                        )
                        enroll_centroid = centroid
                        enroll_dispersion = dispersion
                        enroll_substep = "VALIDATION_PASSED"
                    except Exception as e:
                        enroll_error = format_enrollment_error(e)
                        enroll_substep = "VALIDATION_FAILED"

                countdown_sec = (now - enroll_pause_start) if enroll_substep in ("PAUSE_1", "PAUSE_2") else 0.0
                draw_enroll_capture_overlay(
                    frame=frame,
                    gesture_name=current_enroll_name,
                    take_num=enroll_take_num,
                    take_frames=len(enroll_current_take),
                    target_frames=10,
                    substep=enroll_substep,
                    countdown_sec=countdown_sec,
                    dispersion=enroll_dispersion,
                    error_msg=enroll_error,
                    hand_present=hand_present,
                )
                cv2.imshow("IntentSense — Unified Live Application", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    app_ui_mode = AppUIMode.LIVE
                elif enroll_substep == "READY" and key == 32:  # SPACE
                    if hand_present and norm_features is not None:
                        enroll_substep = "RECORDING"
                        enroll_take_num = 1
                        enroll_current_take = []
                elif enroll_substep == "VALIDATION_PASSED" and key in (10, 13, 32):  # ENTER or SPACE
                    selected_action = "play_pause"
                    selected_ctx = "CUSTOM"
                    confirm_error = ""
                    app_ui_mode = AppUIMode.ENROLL_ACTION
                elif enroll_substep in ("VALIDATION_PASSED", "VALIDATION_FAILED") and key in (ord("r"), ord("R")):
                    enroll_substep = "READY"
                    enroll_take_num = 1
                    enroll_current_take = []
                    enroll_take1 = []
                    enroll_take2 = []
                    enroll_take3 = []
                    enroll_error = ""

            elif app_ui_mode == AppUIMode.ENROLL_ACTION:
                draw_enroll_action_overlay(
                    frame=frame,
                    gesture_name=current_enroll_name,
                    action_list=SUPPORTED_ACTIONS_LIST,
                    selected_action=selected_action,
                    selected_ctx=selected_ctx,
                )
                cv2.imshow("IntentSense — Unified Live Application", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    app_ui_mode = AppUIMode.LIVE
                elif ord("1") <= key <= ord("9"):
                    idx = key - ord("1")
                    if idx < len(SUPPORTED_ACTIONS_LIST):
                        selected_action = SUPPORTED_ACTIONS_LIST[idx][0]
                elif key == ord("0"):
                    if len(SUPPORTED_ACTIONS_LIST) >= 10:
                        selected_action = SUPPORTED_ACTIONS_LIST[9][0]
                elif key in (ord("c"), ord("C")):
                    selected_ctx = "CUSTOM"
                elif key in (ord("a"), ord("A")):
                    selected_ctx = "ALL"
                elif key in (10, 13):  # ENTER
                    confirm_error = ""
                    app_ui_mode = AppUIMode.ENROLL_CONFIRM

            elif app_ui_mode == AppUIMode.ENROLL_CONFIRM:
                draw_enroll_confirm_overlay(
                    frame=frame,
                    gesture_name=current_enroll_name,
                    action_name=selected_action,
                    ctx_name=selected_ctx,
                    dispersion=enroll_dispersion,
                    error_msg=confirm_error,
                )
                cv2.imshow("IntentSense — Unified Live Application", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("n"), ord("N")):  # ESC or N
                    app_ui_mode = AppUIMode.LIVE
                elif key in (10, 13, ord("y"), ord("Y")):  # ENTER or Y
                    resolved_intent = SUPPORTED_ACTION_MAP.get(selected_action, Intent.PLAY_PAUSE)
                    if selected_ctx == "ALL":
                        mappings = {
                            Context.MEDIA.value: resolved_intent.value,
                            Context.PRESENTATION.value: resolved_intent.value,
                            Context.CUSTOM.value: resolved_intent.value,
                        }
                    else:
                        mappings = {
                            Context.CUSTOM.value: resolved_intent.value,
                        }

                    try:
                        if manage_remapping and current_enroll_name in custom_manager.profiles and not enroll_centroid:
                            custom_manager._profiles[current_enroll_name].mappings = mappings
                            custom_manager.save_profiles()
                        else:
                            custom_manager.register_gesture(
                                name=current_enroll_name,
                                centroid=enroll_centroid,
                                dispersion=enroll_dispersion,
                                sample_count=30,
                                mappings=mappings,
                                allow_overwrite=True,
                            )
                        intent_engine.register_custom_gesture(current_enroll_name, mappings)
                        last_action_event = (current_enroll_name, "SAVED", now)
                        context_manager.set_context(Context.CUSTOM)
                        print(f"[CustomMode] ✅ Saved & registered custom gesture '{current_enroll_name}' -> {selected_action} ({selected_ctx})")
                        app_ui_mode = AppUIMode.LIVE
                    except Exception as save_err:
                        confirm_error = f"Failed to save: {save_err}"

            elif app_ui_mode == AppUIMode.MANAGE_MENU:
                profiles = custom_manager.list_gestures()
                draw_manage_overlay(frame, profiles, manage_selected_idx, manage_delete_confirm)
                cv2.imshow("IntentSense — Unified Live Application", frame)
                key = cv2.waitKey(1) & 0xFF

                if manage_delete_confirm is not None:
                    if key in (ord("y"), ord("Y"), 10, 13):
                        del_name = manage_delete_confirm
                        custom_manager.delete_gesture(del_name)
                        intent_engine.unregister_custom_gesture(del_name)
                        manage_delete_confirm = None
                        manage_selected_idx = max(0, min(manage_selected_idx, len(custom_manager.list_gestures()) - 1))
                        print(f"[CustomMode] 🗑️ Deleted custom gesture '{del_name}'")
                    elif key in (27, ord("n"), ord("N")):
                        manage_delete_confirm = None
                else:
                    if key == 27:  # ESC
                        app_ui_mode = AppUIMode.LIVE
                    elif ord("1") <= key <= ord("8"):
                        sel_i = key - ord("1")
                        if sel_i < len(profiles):
                            manage_selected_idx = sel_i
                    elif key in (ord("t"), ord("T")):
                        if profiles:
                            app_ui_mode = AppUIMode.TEST_MODE
                            test_consec_frames = 0
                    elif key in (ord("c"), ord("C")):  # Remap
                        if profiles:
                            p = profiles[manage_selected_idx]
                            current_enroll_name = p.name
                            enroll_dispersion = p.dispersion
                            enroll_centroid = []  # Keep existing centroid
                            manage_remapping = True
                            selected_action = "play_pause"
                            for act_id, intent_val in SUPPORTED_ACTION_MAP.items():
                                if p.mappings and intent_val.value in p.mappings.values():
                                    selected_action = act_id
                                    break
                            selected_ctx = "CUSTOM" if (len(p.mappings) == 1 and Context.CUSTOM.value in p.mappings) else "ALL"
                            confirm_error = ""
                            app_ui_mode = AppUIMode.ENROLL_ACTION
                    elif key in (ord("r"), ord("R")):  # Re-enroll
                        if profiles:
                            p = profiles[manage_selected_idx]
                            current_enroll_name = p.name
                            manage_remapping = True
                            enroll_substep = "READY"
                            enroll_take_num = 1
                            enroll_current_take = []
                            enroll_take1 = []
                            enroll_take2 = []
                            enroll_take3 = []
                            enroll_error = ""
                            app_ui_mode = AppUIMode.ENROLL_CAPTURE
                    elif key in (ord("d"), ord("D")):  # Delete
                        if profiles:
                            manage_delete_confirm = profiles[manage_selected_idx].name
                    elif key in (ord("e"), ord("E")):  # Enroll new
                        app_ui_mode = AppUIMode.ENROLL_NAME
                        typing_buffer = ""
                        typing_error = ""
                        manage_remapping = False

            elif app_ui_mode == AppUIMode.TEST_MODE:
                match_res = custom_manager.match(norm_features) if (hand_present and norm_features is not None) else None
                if match_res and match_res.is_match and match_res.confidence >= 0.80:
                    test_consec_frames = min(3, test_consec_frames + 1)
                else:
                    test_consec_frames = 0

                draw_test_overlay(frame, match_res, test_consec_frames, 3, hand_present)
                cv2.imshow("IntentSense — Unified Live Application", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q"), ord("Q"), ord("t"), ord("T")):  # ESC, Q, T
                    app_ui_mode = AppUIMode.LIVE

            else:
                # -------------------------------------------------------------
                # Standard Live Application Execution Pipeline
                # -------------------------------------------------------------
                swipe_res = swipe_detector.update(
                    x=palm_x,
                    y=palm_y,
                    timestamp=now,
                    palm_size=palm_size,
                    hand_present=hand_present,
                )

                raw_gesture: Optional[str] = None
                confidence: float = 0.0

                active_filter = (
                    activation_sm.activation_filter
                    if activation_sm.state == SystemState.LOCKED
                    else activation_sm.command_filter
                )

                filter_res: Optional[TemporalFilterResult] = None
                arb_res: Optional[ArbitrationResult] = None

                if hand_present and norm_features is not None:
                    builtin_raw, builtin_conf = predict_with_confidence(bundle, norm_features)
                    effective_thresh = active_filter.get_effective_threshold(builtin_raw)

                    arb_res = custom_manager.arbitrate(
                        builtin_gesture=builtin_raw,
                        builtin_confidence=builtin_conf,
                        builtin_threshold=effective_thresh,
                        features=norm_features,
                        hand_present=True,
                    )
                    raw_gesture = arb_res.candidate_gesture
                    confidence = arb_res.confidence

                    filter_res = active_filter.process(raw_gesture, confidence, timestamp=now)
                    act_res = activation_sm.process_filter_result(filter_res, timestamp=now)
                else:
                    filter_res = active_filter.process(None, 0.0, timestamp=now)
                    act_res = activation_sm.process_filter_result(filter_res, timestamp=now)

                if act_res.just_activated or act_res.just_locked:
                    activation_sm.activation_filter.reset()
                    activation_sm.command_filter.reset()
                    swipe_detector.reset()

                consec_frames = filter_res.consecutive_frames if filter_res is not None else 0
                hold_dur = filter_res.hold_duration if filter_res is not None else 0.0
                target_hold = (
                    ACTIVATION_HOLD_DURATION
                    if activation_sm.state == SystemState.LOCKED
                    else COMMAND_HOLD_DURATION
                )

                intent_res = intent_engine.process(
                    activation_result=act_res,
                    confidence=confidence,
                    context=context_manager.get_current_context(),
                )

                is_swipe_detected = swipe_res.direction in (
                    SwipeDirection.LEFT,
                    SwipeDirection.RIGHT,
                )

                swipe_intent_res: Optional[IntentResult] = None
                action_name: Optional[str] = None
                action_status: str = "IDLE"
                act_out: Optional[ActionResult] = None

                if act_res.just_activated:
                    action_status = "ACTIVATED"
                    last_action_event = ("OpenPalm", "ACTIVATED", now)
                    swipe_detector.reset()
                    print(f"[Event] 🔓 SYSTEM ACTIVATED at t={now:.1f}s")

                elif act_res.just_locked:
                    action_status = "LOCKED_INACTIVITY"
                    swipe_detector.reset()
                    print(f"[Event] 🔒 SYSTEM LOCKED due to inactivity at t={now:.1f}s")

                elif is_swipe_detected and activation_sm.state == SystemState.ACTIVE:
                    swipe_intent_res = intent_engine.process_swipe(
                        swipe_result=swipe_res,
                        is_active=True,
                        context=context_manager.get_current_context(),
                    )
                    if swipe_intent_res.accepted:
                        act_out = action_executor.execute(swipe_intent_res)
                        action_name = act_out.action_name
                        action_status = act_out.reason.value

                        activation_sm.command_filter.reset()
                        activation_sm._last_activity_time = now

                        if act_out.success:
                            last_action_event = (act_out.action_name or "Action", "EXECUTED", now)
                            print(
                                f"[Swipe] ⚡ {swipe_intent_res.intent.value:<14} -> "
                                f"{act_out.action_name} (Swipe {swipe_res.direction.value})"
                            )
                        else:
                            action_status = act_out.reason.value
                    else:
                        action_status = swipe_intent_res.reason.value

                elif intent_res.accepted:
                    act_out = action_executor.execute(intent_res)
                    action_name = act_out.action_name
                    action_status = act_out.reason.value

                    if act_out.success:
                        last_action_event = (act_out.action_name or "Action", "EXECUTED", now)
                        print(
                            f"[Action] ⚡ {intent_res.intent.value:<14} -> "
                            f"{act_out.action_name} ({act_out.message})"
                        )
                    else:
                        action_status = act_out.reason.value
                else:
                    if act_res.state == SystemState.LOCKED:
                        if raw_gesture == "OpenPalm":
                            prog = int((hold_dur / ACTIVATION_HOLD_DURATION) * 100)
                            action_status = f"ACTIVATING ({prog}%)"
                        elif raw_gesture is not None:
                            action_status = "SYSTEM_LOCKED"
                        else:
                            action_status = "LOCKED"
                    else:
                        action_status = intent_res.reason.value if raw_gesture else "ACTIVE_IDLE"

                fps = detector.calculate_fps()

                if is_swipe_detected:
                    swipe_hud_str = f"{swipe_res.direction.value} ({action_status})"
                elif swipe_detector.is_in_cooldown:
                    swipe_hud_str = "COOLDOWN"
                elif not hand_present:
                    swipe_hud_str = "NO_HAND"
                else:
                    swipe_hud_str = "READY"

                effective_thresh = active_filter.get_effective_threshold(raw_gesture)

                decision_msg = explainer.explain(
                    state=activation_sm.state,
                    hand_present=hand_present,
                    raw_gesture=raw_gesture,
                    confidence=confidence,
                    effective_threshold=effective_thresh,
                    filter_result=filter_res,
                    activation_result=act_res,
                    intent_result=intent_res,
                    action_result=act_out,
                    swipe_result=swipe_res,
                    swipe_detector_cooldown=swipe_detector.is_in_cooldown,
                    context=context_manager.get_current_context(),
                    target_hold=target_hold,
                    arbitration_reason=arb_res.reason if (hand_present and arb_res is not None) else None,
                )

                if explainer.is_important_event(
                    msg=decision_msg,
                    just_activated=act_res.just_activated,
                    just_locked=act_res.just_locked,
                    is_swipe=(is_swipe_detected and activation_sm.state == SystemState.ACTIVE),
                    action_executed=(act_out is not None and act_out.success and act_out.reason == ActionReason.EXECUTED),
                    action_failed=(act_out is not None and act_out.reason == ActionReason.EXECUTION_FAILED),
                ):
                    current_event_intent = (
                        swipe_intent_res.intent
                        if (is_swipe_detected and swipe_intent_res is not None and swipe_intent_res.intent)
                        else (intent_res.intent if (intent_res is not None and intent_res.accepted) else None)
                    )
                    last_event_snapshot = EventSnapshot(
                        decision=decision_msg,
                        intent=current_event_intent,
                        action_name=action_name,
                        action_status=action_status,
                    )
                    last_event_timestamp = now

                (
                    display_decision,
                    display_intent,
                    display_action_name,
                    display_action_status,
                ) = explainer.resolve_hud_snapshot(
                    live_decision=decision_msg,
                    live_intent=intent_res.intent,
                    live_action_name=action_name,
                    live_action_status=action_status,
                    last_event=last_event_snapshot,
                    last_event_timestamp=last_event_timestamp,
                    current_timestamp=now,
                    display_duration=EVENT_DISPLAY_DURATION,
                )

                draw_hud(
                    frame=frame,
                    state=activation_sm.state,
                    context=context_manager.get_current_context(),
                    gesture=raw_gesture,
                    confidence=confidence,
                    consecutive_frames=consec_frames,
                    target_frames=STABLE_FRAMES,
                    hold_duration=hold_dur,
                    target_hold=target_hold,
                    intent=display_intent,
                    action_name=display_action_name,
                    action_status=display_action_status,
                    time_remaining=activation_sm.time_remaining,
                    last_action_event=last_action_event,
                    fps=fps,
                    live_mode=args.live,
                    swipe_status=swipe_hud_str,
                    decision=display_decision,
                )

                if context_manager.get_current_context() == Context.CUSTOM:
                    draw_custom_mode_banner(frame, len(custom_manager.profiles))

                cv2.imshow("IntentSense — Unified Live Application", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    print("\nQuit requested by user.")
                    break
                elif key in (ord("r"), ord("R")):
                    activation_sm.reset()
                    swipe_detector.reset()
                    last_action_event = None
                    last_event_snapshot = None
                    last_event_timestamp = 0.0
                    print("[Control] Temporal and activation state reset.")
                elif key in (ord("l"), ord("L")):
                    activation_sm.lock(reason="MANUAL_LOCK")
                    swipe_detector.reset()
                    last_action_event = None
                    last_event_snapshot = EventSnapshot(
                        decision=DecisionMessage(
                            category=DecisionCategory.REJECTED,
                            text="REJECTED — System manually locked",
                            color_bgr=(0, 165, 255),
                        ),
                        intent=None,
                        action_name=None,
                        action_status="MANUAL_LOCK",
                    )
                    last_event_timestamp = now
                    print("[Control] System manually LOCKED.")
                elif key == ord("1"):
                    context_manager.set_context(Context.MEDIA)
                    print("[Control] Context switched -> MEDIA")
                elif key == ord("2"):
                    context_manager.set_context(Context.PRESENTATION)
                    print("[Control] Context switched -> PRESENTATION")
                elif key == ord("3"):
                    context_manager.set_context(Context.CUSTOM)
                    print("[Control] Context switched -> CUSTOM")
                elif key in (ord("e"), ord("E")):
                    app_ui_mode = AppUIMode.ENROLL_NAME
                    typing_buffer = ""
                    typing_error = ""
                    manage_remapping = False
                elif key in (ord("m"), ord("M")):
                    app_ui_mode = AppUIMode.MANAGE_MENU
                    manage_selected_idx = 0
                    manage_delete_confirm = None
                elif key in (ord("t"), ord("T")):
                    app_ui_mode = AppUIMode.TEST_MODE
                    test_consec_frames = 0
                elif key in (ord("h"), ord("H"), ord("?")):
                    app_ui_mode = AppUIMode.HELP

    finally:
        cap.release()
        detector.hand_landmarker.close()
        cv2.destroyAllWindows()
        print("IntentSense live application terminated safely.")


if __name__ == "__main__":
    main()
