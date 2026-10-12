#!/usr/bin/env python3
"""
validate_dataset.py
===================

Pre-training dataset validation for IntentSense.

Checks:
  1. File exists and is readable
  2. Exactly 42 feature columns + 1 label column (43 total)
  3. Feature column names match HandFeatureExtractor.get_feature_names()
  4. No missing / NaN values
  5. Exactly 6 expected gesture classes
  6. No unexpected labels
  7. Per-class sample counts and balance assessment

Run this BEFORE training to catch data issues early.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import numpy as np

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.feature_extractor import HandFeatureExtractor

DATASET_DIR: str = os.path.join(SCRIPT_DIR, "dataset")
CSV_PATH: str = os.path.join(DATASET_DIR, "gestures.csv")

EXPECTED_CLASSES = sorted([
    "OpenPalm", "Fist", "Peace", "ThumbsUp", "ThumbsDown", "Pointing"
])

feature_extractor = HandFeatureExtractor()
EXPECTED_FEATURE_NAMES = feature_extractor.get_feature_names()
EXPECTED_FEATURE_COUNT = len(EXPECTED_FEATURE_NAMES)


def validate(csv_path: str = CSV_PATH) -> bool:
    """
    Run all validation checks on the dataset CSV.

    Returns True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("  IntentSense — Dataset Validation")
    print("=" * 60)
    print(f"  File: {csv_path}")
    print()

    errors = []
    warnings = []

    # --- Check 1: File exists ---
    if not os.path.isfile(csv_path):
        print(f"❌ FAIL: File not found: {csv_path}")
        return False

    file_size = os.path.getsize(csv_path)
    print(f"  File size: {file_size:,} bytes")

    # --- Load ---
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ FAIL: Could not parse CSV: {e}")
        return False

    total_rows = len(df)
    total_cols = len(df.columns)
    print(f"  Rows: {total_rows}")
    print(f"  Columns: {total_cols}")
    print()

    # --- Check 2: Column count ---
    expected_cols = EXPECTED_FEATURE_COUNT + 1  # features + label
    if total_cols != expected_cols:
        errors.append(
            f"Column count mismatch: expected {expected_cols}, found {total_cols}"
        )

    # --- Check 3: Feature column names ---
    if "label" not in df.columns:
        errors.append("Missing 'label' column")
    else:
        feature_cols = [c for c in df.columns if c != "label"]
        if feature_cols != EXPECTED_FEATURE_NAMES:
            missing = set(EXPECTED_FEATURE_NAMES) - set(feature_cols)
            extra = set(feature_cols) - set(EXPECTED_FEATURE_NAMES)
            if missing:
                errors.append(f"Missing feature columns: {sorted(missing)}")
            if extra:
                errors.append(f"Unexpected feature columns: {sorted(extra)}")
            if not missing and not extra:
                errors.append(
                    "Feature columns exist but ordering does not match "
                    "HandFeatureExtractor.get_feature_names()"
                )

    # --- Check 4: Missing / NaN values ---
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        null_cols = df.columns[df.isnull().any()].tolist()
        errors.append(f"{null_count} missing/NaN values in columns: {null_cols}")
    else:
        print("  ✓ No missing or NaN values")

    # --- Check 5 & 6: Label classes ---
    if "label" in df.columns:
        actual_classes = sorted(df["label"].unique().tolist())
        if actual_classes != EXPECTED_CLASSES:
            missing_cls = set(EXPECTED_CLASSES) - set(actual_classes)
            extra_cls = set(actual_classes) - set(EXPECTED_CLASSES)
            if missing_cls:
                errors.append(f"Missing gesture classes: {sorted(missing_cls)}")
            if extra_cls:
                errors.append(f"Unexpected gesture classes: {sorted(extra_cls)}")
        else:
            print(f"  ✓ All 6 expected gesture classes present")

    # --- Check 7: Per-class distribution ---
    if "label" in df.columns:
        print()
        print("  Per-Class Sample Counts:")
        print("  " + "-" * 35)
        counts = df["label"].value_counts()
        for cls in EXPECTED_CLASSES:
            c = counts.get(cls, 0)
            print(f"    {cls:12s}  {c:5d}")
        print("  " + "-" * 35)
        print(f"    {'Total':12s}  {total_rows:5d}")
        print()

        # Balance assessment
        if len(counts) >= 2:
            max_c = counts.max()
            min_c = counts.min()
            ratio = min_c / max_c if max_c > 0 else 0
            if ratio < 0.5:
                warnings.append(
                    f"Class imbalance detected: smallest class has {min_c} samples, "
                    f"largest has {max_c} (ratio {ratio:.2f}). "
                    f"Aim for roughly equal counts per class."
                )
            else:
                print(f"  ✓ Class balance ratio: {ratio:.2f} (good)")

    # --- Check: Wrist origin values ---
    if "norm_x0" in df.columns and "norm_y0" in df.columns:
        wrist_x_ok = (df["norm_x0"] == 0.0).all()
        wrist_y_ok = (df["norm_y0"] == 0.0).all()
        if wrist_x_ok and wrist_y_ok:
            print("  ✓ Wrist origin (norm_x0, norm_y0) is (0.0, 0.0) for all rows")
        else:
            errors.append(
                "Wrist landmark (norm_x0, norm_y0) is not consistently (0.0, 0.0). "
                "Data may not be properly normalized."
            )

    # --- Summary ---
    print()
    if warnings:
        print("  ⚠ Warnings:")
        for w in warnings:
            print(f"    - {w}")
        print()

    if errors:
        print("  ❌ VALIDATION FAILED")
        for e in errors:
            print(f"    - {e}")
        return False
    else:
        print("  ✅ VALIDATION PASSED — Dataset is ready for training.")
        return True


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
