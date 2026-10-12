#!/usr/bin/env python3
"""
collect_data.py
===============

Real-world data collection tool for IntentSense hand gesture recognition.

Uses the existing MediaPipe Tasks detector (detector.py) and normalized
feature extraction (core/feature_extractor.py) to record labelled samples
from a live webcam feed.

Gesture Classes & Key Bindings:
-------------------------------
  1 -> OpenPalm
  2 -> Fist
  3 -> Peace
  4 -> ThumbsUp
  5 -> ThumbsDown
  6 -> Pointing
  S -> Save current sample (requires one visible hand and a selected label)
  Q -> Quit

Output:
  dataset/gestures.csv  — 42 normalized features + label column per row.

Data Quality Notes:
  - Only normalized (wrist-relative, scale-invariant) features are stored.
  - Duplicate consecutive frames are rejected automatically.
  - Only single-hand captures are accepted.
  - Physically vary hand position, distance, and rotation for robust data.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from detector import HandDetector
from core.feature_extractor import HandFeatureExtractor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR: str = os.path.join(SCRIPT_DIR, "dataset")
CSV_PATH: str = os.path.join(DATASET_DIR, "gestures.csv")

SAMPLES_PER_GESTURE: int = 200  # Target samples per class

GESTURE_CLASSES: List[str] = [
    "OpenPalm",
    "Fist",
    "Peace",
    "ThumbsUp",
    "ThumbsDown",
    "Pointing",
]

KEY_TO_LABEL: Dict[int, str] = {
    ord("1"): "OpenPalm",
    ord("2"): "Fist",
    ord("3"): "Peace",
    ord("4"): "ThumbsUp",
    ord("5"): "ThumbsDown",
    ord("6"): "Pointing",
}

feature_extractor = HandFeatureExtractor()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def ensure_dataset_dir() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)


def write_header_if_needed(csv_path: str) -> None:
    """Write the normalized-feature header row if the CSV is new or empty."""
    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        return
    header: List[str] = feature_extractor.get_feature_names() + ["label"]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)


def load_per_class_counts(csv_path: str) -> Dict[str, int]:
    """Read existing CSV and return {label: count} for every gesture class."""
    counts: Dict[str, int] = {g: 0 for g in GESTURE_CLASSES}
    if not os.path.isfile(csv_path):
        return counts
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                lbl = row[-1]
                counts[lbl] = counts.get(lbl, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Key parsing
# ---------------------------------------------------------------------------
def parse_keypress(key: int) -> Optional[str]:
    if key == -1 or key == 0:
        return None
    code: int = key & 0xFF
    if code in (ord("q"), ord("Q")):
        return "quit"
    if code in (ord("s"), ord("S")):
        return "save"
    if code in KEY_TO_LABEL:
        return KEY_TO_LABEL[code]
    return None


# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------
def draw_hud(
    frame,
    current_label: str,
    counts: Dict[str, int],
    hand_visible: bool,
    num_hands: int,
) -> None:
    """
    Render a clear status panel on the webcam frame.

    Layout:
      Gesture: <label>
      Samples: <count> / <target>
      Hands detected: <n>
      [status line]
      per-class progress bar
      key hints
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark panel on the left side
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (440, 275), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    y_pos = 32  # vertical cursor for text placement
    line_h = 28  # line height

    # --- Gesture label ---
    if current_label:
        lbl_color = (0, 255, 120)
        lbl_text = f"Gesture: {current_label}"
    else:
        lbl_color = (0, 200, 255)
        lbl_text = "Gesture: [Press 1-6 to select]"
    cv2.putText(frame, lbl_text, (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, lbl_color, 2, cv2.LINE_AA)
    y_pos += line_h

    # --- Samples progress ---
    current_count = counts.get(current_label, 0) if current_label else 0
    target = SAMPLES_PER_GESTURE
    progress_text = f"Samples: {current_count} / {target}"
    prog_color = (0, 255, 0) if current_count >= target else (255, 255, 255)
    cv2.putText(frame, progress_text, (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, prog_color, 2, cv2.LINE_AA)
    y_pos += line_h

    # --- Hands detected ---
    hand_text = f"Hands detected: {num_hands}"
    hand_color = (0, 255, 0) if num_hands == 1 else (0, 0, 255)
    cv2.putText(frame, hand_text, (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, hand_color, 1, cv2.LINE_AA)
    y_pos += line_h

    # --- Status line ---
    if not hand_visible:
        status = "Show your hand to the camera"
        status_color = (0, 0, 255)
    elif not current_label:
        status = "Select a gesture label (1-6)"
        status_color = (0, 200, 255)
    elif current_count >= target:
        status = f"{current_label} COMPLETE! Switch gesture."
        status_color = (0, 255, 255)
    else:
        status = "Press S to capture"
        status_color = (0, 255, 0)
    cv2.putText(frame, status, (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, status_color, 2, cv2.LINE_AA)
    y_pos += line_h + 4

    # --- Per-class mini progress ---
    for i, gesture in enumerate(GESTURE_CLASSES):
        c = counts.get(gesture, 0)
        tag = f"{i+1}:{gesture[:6]}  {c}/{target}"
        color = (0, 255, 0) if c >= target else (180, 180, 180)
        cv2.putText(frame, tag, (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
        y_pos += 16

    # --- Key hints at bottom of panel ---
    y_pos += 4
    cv2.putText(frame, "S=Save  Q=Quit  1-6=Label", (18, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    # --- Overall progress in top-right corner ---
    total = sum(counts.values())
    total_target = SAMPLES_PER_GESTURE * len(GESTURE_CLASSES)
    overall_text = f"Total: {total}/{total_target}"
    cv2.putText(frame, overall_text, (w - 230, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------
def collect() -> None:
    """Interactive webcam data collection session."""
    ensure_dataset_dir()
    write_header_if_needed(CSV_PATH)
    counts = load_per_class_counts(CSV_PATH)

    # Open webcam (macOS AVFoundation backend preferred)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector()

    current_label: str = ""
    last_saved_features: Optional[Tuple[float, ...]] = None

    total_target = SAMPLES_PER_GESTURE * len(GESTURE_CLASSES)
    print("=" * 60)
    print("  IntentSense — Real Data Collector")
    print("=" * 60)
    print(f"  Dataset      : {CSV_PATH}")
    print(f"  Classes      : {len(GESTURE_CLASSES)}")
    print(f"  Target/class : {SAMPLES_PER_GESTURE}")
    print(f"  Total target : {total_target}")
    print(f"  Existing     : {sum(counts.values())}")
    print()
    print("  Keys: 1-6 = Select gesture | S = Save | Q = Quit")
    print("  Tip: Vary hand position, distance, and angle!")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Frame capture failed.")
            break

        # --- Detection ---
        results = detector.detect_hand(frame)
        num_hands = len(results.hand_landmarks) if results and results.hand_landmarks else 0
        hand_visible = num_hands == 1

        landmarks: Optional[List[Tuple[int, int]]] = None
        normalized_features: Optional[List[float]] = None

        if hand_visible:
            frame = detector.draw_landmarks(frame, results)
            landmarks = detector.get_landmarks(results, frame.shape)
            if landmarks:
                normalized_features = feature_extractor.extract_features(landmarks)

        # --- HUD ---
        draw_hud(frame, current_label, counts, hand_visible, num_hands)
        cv2.imshow("IntentSense Data Collector", frame)

        # --- Input ---
        key = cv2.waitKey(1)
        action = parse_keypress(key)
        if action is None:
            continue

        if action == "quit":
            break

        if action == "save":
            if not hand_visible or normalized_features is None:
                print("[Skip] No single hand detected.")
                continue
            if not current_label:
                print("[Skip] No gesture label selected. Press 1-6 first.")
                continue

            # Warn (but still allow) if target is already met
            current_count = counts.get(current_label, 0)
            if current_count >= SAMPLES_PER_GESTURE:
                print(f"[Note] {current_label} already has {current_count}/{SAMPLES_PER_GESTURE} samples.")

            # Duplicate guard
            feat_tuple = tuple(normalized_features)
            if feat_tuple == last_saved_features:
                print("[Skip] Duplicate frame — move your hand slightly.")
                continue

            # Write to CSV
            row = list(normalized_features) + [current_label]
            with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

            last_saved_features = feat_tuple
            counts[current_label] = current_count + 1
            total = sum(counts.values())
            print(
                f"[Saved] #{total}  {current_label}: "
                f"{counts[current_label]}/{SAMPLES_PER_GESTURE}"
            )
            continue

        # Label selection
        if action in GESTURE_CLASSES:
            current_label = action
            print(f"[Label] → {current_label}  ({counts[current_label]}/{SAMPLES_PER_GESTURE})")

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()

    total = sum(counts.values())
    print()
    print("=" * 60)
    print("  Collection Session Summary")
    print("=" * 60)
    for g in GESTURE_CLASSES:
        c = counts.get(g, 0)
        mark = "✓" if c >= SAMPLES_PER_GESTURE else " "
        print(f"  [{mark}] {g:12s}  {c:4d} / {SAMPLES_PER_GESTURE}")
    print(f"  {'─' * 30}")
    print(f"      Total:       {total:4d} / {SAMPLES_PER_GESTURE * len(GESTURE_CLASSES)}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        collect()
    except KeyboardInterrupt:
        print("\nInterrupted — exiting collector.")
        sys.exit(0)
