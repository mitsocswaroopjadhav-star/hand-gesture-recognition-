#!/usr/bin/env python3
"""
test_phase3_live.py
===================

Minimal Phase 3 Integration Test for IntentSense.

Connects:
  Webcam (OpenCV)
      │
      ▼
  Landmark Detection (detector.py / MediaPipe)
      │
      ▼
  Feature Normalization (core/feature_extractor.py)
      │
      ▼
  Model Inference (models/gesture_model.pkl via predict_proba)
      │
      ▼
  Confidence + Temporal Filter (core/temporal_filter.py)
      │
      ▼
  HUD Display (Raw Prediction, Confidence, Temporal Status, Cooldown)

Press 'Q' to quit | Press 'R' to reset temporal filter.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple

import cv2
import joblib
import numpy as np

# Ensure project root is in path
PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detector import HandDetector
from core.feature_extractor import HandFeatureExtractor
from core.temporal_filter import (
    ConfirmationStatus,
    TemporalFilter,
    predict_with_confidence,
)

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "gesture_model.pkl")

# Phase 3 Configurable Defaults
CONFIDENCE_THRESHOLD: float = 0.80
STABLE_FRAMES: int = 5
COOLDOWN_FRAMES: int = 15
MIN_HOLD_DURATION: float = 0.5  # Phase 3.1: Minimum continuous hold duration in seconds


def load_model_bundle(path: str):
    """Load model bundle dictionary from disk."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Trained model not found at {path}. Please run train_model.py first."
        )
    return joblib.load(path)


def draw_hud(
    frame: np.ndarray,
    raw_gesture: Optional[str],
    confidence: float,
    status: ConfirmationStatus,
    confirmed_gesture: Optional[str],
    consecutive_frames: int,
    target_frames: int,
    hold_duration: float,
    target_hold_duration: float,
    cooldown_remaining: int,
    reason: str,
    last_confirmed_display: Optional[Tuple[str, float]],
    fps: float,
) -> None:
    """Renders a clean HUD showing the Phase 3.1 pipeline state."""
    h, w = frame.shape[:2]

    # Background panel for status
    overlay = frame.copy()
    panel_w, panel_h = 390, 235
    cv2.rectangle(overlay, (12, 12), (12 + panel_w, 12 + panel_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (12, 12), (12 + panel_w, 12 + panel_h), (70, 70, 70), 1)

    y = 35
    # Title
    cv2.putText(
        frame,
        "IntentSense - Phase 3.1 Live Test",
        (22, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y += 28

    # 1. Raw Prediction & Confidence
    if raw_gesture is not None:
        conf_pct = int(confidence * 100)
        conf_color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        raw_text = f"Raw: {raw_gesture} ({conf_pct}%)"
    else:
        raw_text = "Raw: No Hand Detected"
        conf_color = (150, 150, 150)

    cv2.putText(
        frame,
        raw_text,
        (22, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        conf_color,
        1,
        cv2.LINE_AA,
    )
    y += 24

    # 2. Confidence Threshold Check
    gate_text = f"Gate (>= {int(CONFIDENCE_THRESHOLD * 100)}%): "
    if raw_gesture is not None:
        gate_status = "PASS" if confidence >= CONFIDENCE_THRESHOLD else "FAIL (Low)"
        gate_color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)
    else:
        gate_status = "N/A"
        gate_color = (150, 150, 150)

    cv2.putText(
        frame,
        gate_text + gate_status,
        (22, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        gate_color,
        1,
        cv2.LINE_AA,
    )
    y += 24

    # 3. Temporal Stability Progress (Frames & Hold Duration)
    stability_text = f"Stability: {consecutive_frames}/{target_frames} frames"
    cv2.putText(
        frame,
        stability_text,
        (22, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    y += 22

    # Phase 3.1: Continuous Hold Time
    hold_color = (0, 255, 0) if hold_duration >= target_hold_duration else (200, 200, 200)
    hold_text = f"Hold Time: {hold_duration:.2f}s / {target_hold_duration:.2f}s"
    cv2.putText(
        frame,
        hold_text,
        (22, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        hold_color,
        1,
        cv2.LINE_AA,
    )
    y += 24

    # 4. Cooldown status
    if cooldown_remaining > 0:
        cd_text = f"Cooldown: Active ({cooldown_remaining} frames)"
        cd_color = (0, 165, 255)
    else:
        cd_text = "Cooldown: Ready"
        cd_color = (180, 180, 180)

    cv2.putText(
        frame,
        cd_text,
        (22, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        cd_color,
        1,
        cv2.LINE_AA,
    )
    y += 28

    # 5. Confirmation Status Banner
    if status == ConfirmationStatus.CONFIRMED:
        status_text = f"[CONFIRMED] {confirmed_gesture}"
        status_color = (0, 255, 0)
    else:
        status_text = f"NOT CONFIRMED ({reason})"
        status_color = (160, 160, 160)

    cv2.putText(
        frame,
        status_text,
        (22, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2,
        cv2.LINE_AA,
    )
    y += 24

    # Mini Footer Hint
    cv2.putText(
        frame,
        "Keys: Q = Quit  |  R = Reset Filter",
        (22, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (130, 130, 130),
        1,
        cv2.LINE_AA,
    )

    # Big Banner on screen if confirmed within the last 1.2 seconds
    if last_confirmed_display is not None:
        last_g, last_time = last_confirmed_display
        elapsed = time.time() - last_time
        if elapsed < 1.2:
            banner_text = f">> CONFIRMED: {last_g} <<"
            cv2.putText(
                frame,
                banner_text,
                (w // 2 - 190, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

    # FPS counter in top-right
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 120, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run_smoke_test() -> None:
    """Headless smoke test verifying components connect without opening webcam."""
    print("Running Phase 3 headless component integration test...")
    bundle = load_model_bundle(MODEL_PATH)
    extractor = HandFeatureExtractor()
    temporal_filter = TemporalFilter(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        stable_frames=STABLE_FRAMES,
        cooldown_frames=COOLDOWN_FRAMES,
    )

    # 1. Test with real sample from dataset/gestures.csv (high confidence)
    import pandas as pd
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "dataset", "gestures.csv"))
    real_features = df.iloc[0, :-1].values.astype(float)
    expected_label = df.iloc[0, -1]

    label, conf = predict_with_confidence(bundle, real_features)
    print(f"✓ Real sample inference: predicted '{label}' with confidence {conf:.1%} (expected: '{expected_label}')")
    assert conf >= CONFIDENCE_THRESHOLD, f"Expected high confidence on training sample, got {conf}"

    # Feed frames at 0.12s intervals (frame 5 at t=0.48s is < 0.50s, frame 6 at t=0.60s is >= 0.50s)
    for i in range(1, STABLE_FRAMES + 1):
        t = (i - 1) * 0.12  # i=5 -> t=0.48s (hold=0.48s < 0.50s)
        res = temporal_filter.process(label, conf, timestamp=t)
        assert res.status == ConfirmationStatus.NOT_CONFIRMED
        if i == STABLE_FRAMES:
            assert res.reason == "INSUFFICIENT_HOLD_TIME"
            print(f"✓ Hold duration gate verified on frame {i}: {res.hold_duration:.2f}s < 0.50s (INSUFFICIENT_HOLD_TIME)")

    # 6th frame at t = 0.60s reaches hold duration >= 0.50s -> CONFIRMED
    res = temporal_filter.process(label, conf, timestamp=0.60)
    assert res.status == ConfirmationStatus.CONFIRMED
    print(f"✓ Temporal filter confirmed at t=0.60s (hold: {res.hold_duration:.2f}s >= 0.50s): {res.gesture}")

    # 2. Cooldown test
    res = temporal_filter.process(label, conf, timestamp=0.65)
    assert res.status == ConfirmationStatus.NOT_CONFIRMED
    assert res.reason == "IN_COOLDOWN"
    print("✓ Cooldown verified: immediate re-confirmation prevented")

    # 3. Low confidence rejection test
    res_low = temporal_filter.process(label, 0.50, timestamp=0.70)
    assert res_low.status == ConfirmationStatus.NOT_CONFIRMED
    assert res_low.reason == "LOW_CONFIDENCE"
    print("✓ Low confidence rejection verified (< 80%)")

    print("\n✅ Phase 3.1 smoke test PASSED successfully!")


def main() -> None:
    """Main live webcam loop."""
    if "--smoke" in sys.argv or "--test" in sys.argv:
        run_smoke_test()
        return

    print("=" * 60)
    print("  IntentSense — Phase 3.1 Live Integration Test")
    print("=" * 60)
    print(f"  Model bundle : {MODEL_PATH}")
    print(f"  Confidence   : >= {int(CONFIDENCE_THRESHOLD * 100)}%")
    print(f"  Stability    : {STABLE_FRAMES} consecutive frames")
    print(f"  Hold Time    : >= {MIN_HOLD_DURATION:.2f} seconds")
    print(f"  Cooldown     : {COOLDOWN_FRAMES} frames")
    print("  Keys         : Q = Quit  |  R = Reset Filter")
    print("=" * 60)

    # 1. Load trained model
    bundle = load_model_bundle(MODEL_PATH)

    # 2. Setup modules
    extractor = HandFeatureExtractor()
    detector = HandDetector()
    temporal_filter = TemporalFilter(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        stable_frames=STABLE_FRAMES,
        cooldown_frames=COOLDOWN_FRAMES,
        min_hold_duration=MIN_HOLD_DURATION,
    )

    # 3. Open webcam
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        print("Check camera permissions in macOS System Settings > Privacy > Camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_confirmed_display: Optional[Tuple[str, float]] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            # Hand detection
            result = detector.detect_hand(frame)
            landmarks = detector.get_landmarks(result, frame.shape)

            raw_gesture: Optional[str] = None
            confidence: float = 0.0

            if landmarks is not None and len(landmarks) == 21:
                # 1. Feature normalization
                norm_features = extractor.extract_features(landmarks)

                # 2. Model predict_proba() -> (label, confidence)
                raw_gesture, confidence = predict_with_confidence(bundle, norm_features)

                # 3. Temporal Filter
                filter_res = temporal_filter.process(raw_gesture, confidence)

                # Draw landmark mesh on hand
                frame = detector.draw_landmarks(frame, result)
            else:
                # No hand visible -> pass None to filter to reset sequence
                filter_res = temporal_filter.process(None, 0.0)

            # Diagnostic per-frame logging requested by user
            if raw_gesture is not None:
                status_str = filter_res.status.value if hasattr(filter_res.status, 'value') else str(filter_res.status)
                cand_str = str(filter_res.candidate_gesture) if filter_res.candidate_gesture else "None"
                print(
                    f"[Frame {temporal_filter.frame_count:04d}] "
                    f"Pred: {raw_gesture:<10} | "
                    f"Conf: {confidence*100:5.1f}% | "
                    f"Cand: {cand_str:<10} | "
                    f"Consec: {filter_res.consecutive_frames}/{STABLE_FRAMES} | "
                    f"Hold: {filter_res.hold_duration:4.2f}s | "
                    f"Status: {status_str:<13} | "
                    f"Reason: {filter_res.reason:<22} | "
                    f"CD: {filter_res.cooldown_remaining}"
                )

            # Check if newly confirmed
            if filter_res.is_confirmed and filter_res.gesture:
                last_confirmed_display = (filter_res.gesture, time.time())
                print(
                    f"  >>> ⭐ [CONFIRMED] {filter_res.gesture} "
                    f"(conf: {filter_res.confidence:.1%}, consecutive: {filter_res.consecutive_frames}, hold: {filter_res.hold_duration:.2f}s)"
                )

            # Render HUD
            fps = detector.calculate_fps()
            draw_hud(
                frame=frame,
                raw_gesture=raw_gesture,
                confidence=confidence,
                status=filter_res.status,
                confirmed_gesture=filter_res.confirmed_gesture,
                consecutive_frames=filter_res.consecutive_frames,
                target_frames=STABLE_FRAMES,
                hold_duration=filter_res.hold_duration,
                target_hold_duration=MIN_HOLD_DURATION,
                cooldown_remaining=filter_res.cooldown_remaining,
                reason=filter_res.reason,
                last_confirmed_display=last_confirmed_display,
                fps=fps,
            )

            cv2.imshow("IntentSense - Phase 3 Live Integration Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("\nQuit requested by user.")
                break
            elif key in (ord("r"), ord("R")):
                temporal_filter.reset()
                last_confirmed_display = None
                print("[Reset] Temporal filter reset.")

    finally:
        cap.release()
        detector.hand_landmarker.close()
        cv2.destroyAllWindows()
        print("Live integration test terminated.")


if __name__ == "__main__":
    main()
