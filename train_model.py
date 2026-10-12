#!/usr/bin/env python3
"""
train_model.py
==============

Training script for normalized hand gesture classification using Random Forest.

Features: 42 normalized float features (norm_x0, norm_y0, ..., norm_x20, norm_y20)
Target: label string ('OpenPalm', 'Fist', 'Peace', 'ThumbsUp', 'ThumbsDown', 'Pointing')

Saves trained model to models/gesture_model.pkl.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# Project imports
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.feature_extractor import HandFeatureExtractor

DATASET_DIR: str = os.path.join(SCRIPT_DIR, "dataset")
MODELS_DIR: str = os.path.join(SCRIPT_DIR, "models")
CSV_PATH: str = os.path.join(DATASET_DIR, "gestures.csv")
MODEL_PATH: str = os.path.join(MODELS_DIR, "gesture_model.pkl")

feature_extractor = HandFeatureExtractor()
EXPECTED_FEATURE_NAMES: List[str] = feature_extractor.get_feature_names()
EXPECTED_FEATURE_COUNT: int = len(EXPECTED_FEATURE_NAMES)

N_ESTIMATORS: int = 200
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2


def setup_directories() -> None:
    """Create models directory if needed."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def validate_and_prep_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Validate DataFrame structure and return (X, y)."""
    if "label" not in df.columns:
        raise ValueError("Dataset missing 'label' column")

    feature_cols = [c for c in df.columns if c != "label"]

    if len(feature_cols) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} feature columns, but found {len(feature_cols)}"
        )

    # Re-order features to match exact order from HandFeatureExtractor
    if feature_cols != EXPECTED_FEATURE_NAMES:
        print("Note: Aligning dataset columns to HandFeatureExtractor feature ordering.")
        df = df[EXPECTED_FEATURE_NAMES + ["label"]]

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values")

    X = df[EXPECTED_FEATURE_NAMES]
    y = df["label"]

    print("\nDataset label distribution:")
    print(y.value_counts().to_string())

    print(f"\n✓ Dataset validation passed: {len(df)} samples, {len(EXPECTED_FEATURE_NAMES)} features, {y.nunique()} classes")

    return X, y


def train() -> None:
    """Main model training and evaluation execution."""
    print("=" * 60)
    print("IntentSense — Model Training (Normalized Features)")
    print("=" * 60)

    setup_directories()

    if not os.path.isfile(CSV_PATH):
        print(f"❌ Error: Dataset file not found at {CSV_PATH}")
        print("Please run collect_data.py to gather normalized gesture samples.")
        sys.exit(1)

    print(f"📥 Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    if len(df) < 10:
        print(f"❌ Error: Dataset has only {len(df)} samples. Please collect more samples before training.")
        sys.exit(1)

    X, y = validate_and_prep_dataset(df)

    # Check minimum samples per class for stratified split
    min_samples = y.value_counts().min()
    use_stratify = y if min_samples >= 2 else None

    print(f"\n🔀 Splitting dataset (test_size={TEST_SIZE}, random_state={RANDOM_STATE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=use_stratify
    )

    print(f"✓ Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    print(f"\n🌳 Training RandomForestClassifier (n_estimators={N_ESTIMATORS})...")
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("✓ Model training complete")

    # Evaluation
    y_pred = clf.predict(X_test)
    train_acc = clf.score(X_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 40)
    print("📈 Model Performance Metrics")
    print("=" * 40)
    print(f"Training Accuracy : {train_acc:.4f} ({train_acc * 100:.2f}%)")
    print(f"Testing Accuracy  : {test_acc:.4f} ({test_acc * 100:.2f}%)")

    print("\nClassification Report (Per-Class Precision, Recall, F1):")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model along with feature ordering metadata
    model_bundle = {
        "model": clf,
        "feature_names": EXPECTED_FEATURE_NAMES,
        "classes": sorted(list(y.unique())),
    }

    print(f"\n💾 Saving model bundle to {MODEL_PATH}...")
    joblib.dump(model_bundle, MODEL_PATH)
    print("✓ Model saved successfully!")


if __name__ == "__main__":
    train()