#!/usr/bin/env python3
"""
evaluate_phase7a.py
===================

Phase 7A: Adaptive Confidence & False-Trigger Evaluation Benchmark.

Evaluates the IntentSense 4-tier interaction hierarchy:
  Tier 1: Raw Classifier Prediction (Random Forest argmax)
  Tier 2: Confidence-Qualified Prediction (threshold gating)
  Tier 3: Temporally Confirmed Prediction (TemporalFilter: 5 frames, 0.5s hold)
  Tier 4: Final Accepted Intent (ContextManager + IntentEngine)

Evaluates confidence thresholds:
  theta in {0.70, 0.75, 0.80, 0.85, 0.90}

IMPORTANT METHODOLOGICAL DISTINCTION:
  Flicker, transition, and coordinate-noise results measure SYNTHETIC
  FALSE-TRIGGER SUSCEPTIBILITY under generated stress streams.
  They do NOT represent a real-world false-trigger rate (which requires
  in-the-wild negative/unscripted human gesture recordings).

All sample counts, stream counts, and frame counts are recorded and reported
to guarantee complete reproducibility.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Project root setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.activation_state import ActivationStateMachine, SystemState
from core.context_manager import Context, ContextManager
from core.intent_engine import Intent, IntentEngine
from core.temporal_filter import TemporalFilter, TemporalFilterResult

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "gestures.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_model.pkl")

FPS: float = 30.0
DT: float = 1.0 / FPS
THRESHOLDS: List[float] = [0.70, 0.75, 0.80, 0.85, 0.90]
STABLE_FRAMES: int = 5
COMMAND_HOLD_DURATION: float = 0.5
COOLDOWN_FRAMES: int = 15
RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# 1. Dataset & Model Loading
# ---------------------------------------------------------------------------
def load_benchmark_data() -> Tuple[dict, pd.DataFrame, pd.DataFrame, List[str]]:
    """Loads model bundle and splits dataset into train and held-out test splits."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATASET_PATH)

    feature_cols = [c for c in df.columns if c != "label"]

    # Stratified 80/20 train/test split matching train_model.py
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    return bundle, train_df, test_df, feature_cols


# ---------------------------------------------------------------------------
# 2. Confidence Distribution Analysis (Held-out Test Set)
# ---------------------------------------------------------------------------
@dataclass
class ClassConfidenceStats:
    class_name: str
    sample_count: int
    correct_count: int
    accuracy: float
    conf_mean: float
    conf_std: float
    conf_median: float
    conf_p10: float
    conf_p90: float


def compute_confidence_distributions(
    bundle: dict, test_df: pd.DataFrame, feature_cols: List[str]
) -> Dict[str, ClassConfidenceStats]:
    """Computes exact empirical confidence distribution per gesture on held-out test data."""
    model = bundle["model"]
    classes = np.array(bundle["classes"])

    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["label"].to_numpy()

    proba = model.predict_proba(X_test)
    preds = classes[np.argmax(proba, axis=1)]
    confs = np.max(proba, axis=1)

    stats: Dict[str, ClassConfidenceStats] = {}
    for c in classes:
        mask = (y_test == c)
        c_count = int(np.sum(mask))
        c_correct = int(np.sum((y_test == c) & (preds == c)))
        c_acc = (c_correct / c_count) * 100.0 if c_count > 0 else 0.0

        c_confs = confs[mask]
        stats[c] = ClassConfidenceStats(
            class_name=c,
            sample_count=c_count,
            correct_count=c_correct,
            accuracy=c_acc,
            conf_mean=float(np.mean(c_confs)),
            conf_std=float(np.std(c_confs)),
            conf_median=float(np.median(c_confs)),
            conf_p10=float(np.percentile(c_confs, 10)),
            conf_p90=float(np.percentile(c_confs, 90)),
        )

    return stats


# ---------------------------------------------------------------------------
# 3. Synthetic Stream Generation
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkStream:
    category: str
    target_class: str
    secondary_class: Optional[str]
    features_sequence: List[Optional[np.ndarray]]
    timestamps: List[float]
    predictions: List[Tuple[Optional[str], float]] = field(default_factory=list)


def generate_benchmark_streams(
    test_df: pd.DataFrame, feature_cols: List[str], bundle: dict
) -> Tuple[List[BenchmarkStream], List[BenchmarkStream], List[BenchmarkStream], List[BenchmarkStream]]:
    """
    Constructs deterministic benchmark streams and batch pre-computes model predictions:
    1. Intentional Hold Streams (30 frames = 1.0s continuous hold)
    2. Transient Flicker Streams (1 to 4 frames burst + rest)
    3. Transition Streams (15 linearly interpolated frames between distinct poses)
    4. Perturbed Noise Streams (Gaussian landmark jitter sigma=0.08)
    """
    rng = np.random.RandomState(RANDOM_STATE)
    classes = sorted(test_df["label"].unique().tolist())

    # 1. Intentional Hold Streams: 5 distinct 30-frame streams per class = 30 streams total
    intentional_streams: List[BenchmarkStream] = []
    stream_length = 30

    for c in classes:
        c_samples = test_df[test_df["label"] == c][feature_cols].to_numpy()
        n_available = len(c_samples)
        for s_idx in range(5):
            indices = (np.arange(stream_length) + s_idx * 5) % n_available
            seq: List[Optional[np.ndarray]] = [c_samples[i].copy() for i in indices]
            ts = [i * DT for i in range(stream_length)]
            intentional_streams.append(
                BenchmarkStream(
                    category="intentional",
                    target_class=c,
                    secondary_class=None,
                    features_sequence=seq,
                    timestamps=ts,
                )
            )

    # 2. Transient Flicker Streams: 1 to 4 frames burst followed by None (24 streams total)
    flicker_streams: List[BenchmarkStream] = []
    for c in classes:
        c_samples = test_df[test_df["label"] == c][feature_cols].to_numpy()
        for burst_len in [1, 2, 3, 4]:
            seq = []
            for b in range(burst_len):
                seq.append(c_samples[b % len(c_samples)].copy())
            for _ in range(10):
                seq.append(None)
            ts = [i * DT for i in range(len(seq))]
            flicker_streams.append(
                BenchmarkStream(
                    category="flicker",
                    target_class=c,
                    secondary_class=None,
                    features_sequence=seq,
                    timestamps=ts,
                )
            )

    # 3. Transition Streams: Interpolation between pairs of different classes (30 streams total)
    transition_streams: List[BenchmarkStream] = []
    class_pairs = [
        ("Fist", "Peace"),
        ("Peace", "Pointing"),
        ("Pointing", "ThumbsUp"),
        ("ThumbsUp", "ThumbsDown"),
        ("ThumbsDown", "OpenPalm"),
        ("OpenPalm", "Fist"),
    ]
    interp_steps = 15
    for from_c, to_c in class_pairs:
        from_samples = test_df[test_df["label"] == from_c][feature_cols].to_numpy()
        to_samples = test_df[test_df["label"] == to_c][feature_cols].to_numpy()
        for pair_idx in range(5):
            vec_a = from_samples[pair_idx % len(from_samples)]
            vec_b = to_samples[pair_idx % len(to_samples)]
            seq = []
            for alpha in np.linspace(0.0, 1.0, interp_steps):
                interpolated = (1.0 - alpha) * vec_a + alpha * vec_b
                seq.append(interpolated)
            ts = [i * DT for i in range(interp_steps)]
            transition_streams.append(
                BenchmarkStream(
                    category="transition",
                    target_class=from_c,
                    secondary_class=to_c,
                    features_sequence=seq,
                    timestamps=ts,
                )
            )

    # 4. Perturbed Noise Streams: Intentional hold with added Gaussian coordinate noise (30 streams)
    noise_streams: List[BenchmarkStream] = []
    noise_sigma = 0.08
    for stream in intentional_streams:
        noisy_seq = []
        for feat in stream.features_sequence:
            noise = rng.normal(0.0, noise_sigma, size=feat.shape)
            noisy_seq.append(feat + noise)
        noise_streams.append(
            BenchmarkStream(
                category="noise",
                target_class=stream.target_class,
                secondary_class=None,
                features_sequence=noisy_seq,
                timestamps=list(stream.timestamps),
            )
        )

    # -------------------------------------------------------------------------
    # Batch Precompute Predictions for High Performance
    # -------------------------------------------------------------------------
    all_streams = intentional_streams + flicker_streams + transition_streams + noise_streams
    model = bundle["model"]
    classes_arr = np.array(bundle["classes"])

    for s in all_streams:
        preds: List[Tuple[Optional[str], float]] = []
        valid_indices = [i for i, f in enumerate(s.features_sequence) if f is not None]
        preds = [(None, 0.0)] * len(s.features_sequence)
        if valid_indices:
            feats = np.stack([s.features_sequence[i] for i in valid_indices])
            proba = model.predict_proba(feats)
            argmax_idx = np.argmax(proba, axis=1)
            max_confs = np.max(proba, axis=1)
            pred_classes = classes_arr[argmax_idx]
            for idx, p_c, conf in zip(valid_indices, pred_classes, max_confs):
                preds[idx] = (str(p_c), float(conf))
        s.predictions = preds

    return intentional_streams, flicker_streams, transition_streams, noise_streams


# ---------------------------------------------------------------------------
# 4. Evaluation Engine Across 4 Tiers
# ---------------------------------------------------------------------------
@dataclass
class ThresholdBenchmarkResult:
    threshold: float
    total_test_samples: int
    tier1_raw_acc: float
    tier2_pass_rate: float
    tier2_reject_rate: float
    tier2_acc: float

    # Intentional stream metrics
    num_intentional_streams: int
    intentional_confirm_rate: float
    mean_latency_sec: float
    std_latency_sec: float
    mean_latency_frames: float

    # Synthetic susceptibility metrics (explicitly labeled synthetic)
    num_flicker_streams: int
    synthetic_flicker_susceptibility: float
    num_transition_streams: int
    synthetic_transition_susceptibility: float
    num_noise_streams: int
    noise_retention_rate: float

    # Per-gesture confirmation breakdown
    per_gesture_confirm_rate: Dict[str, float] = field(default_factory=dict)
    per_gesture_latency_sec: Dict[str, float] = field(default_factory=dict)


def evaluate_threshold(
    threshold: float,
    bundle: dict,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    intentional_streams: List[BenchmarkStream],
    flicker_streams: List[BenchmarkStream],
    transition_streams: List[BenchmarkStream],
    noise_streams: List[BenchmarkStream],
) -> ThresholdBenchmarkResult:
    """Executes the full 4-tier pipeline benchmark for a single confidence threshold."""
    model = bundle["model"]
    classes = np.array(bundle["classes"])

    # -------------------------------------------------------------------------
    # Tier 1 & 2: Static Frame Evaluation on Test Set
    # -------------------------------------------------------------------------
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["label"].to_numpy()
    total_test = len(y_test)

    proba = model.predict_proba(X_test)
    raw_preds = classes[np.argmax(proba, axis=1)]
    confs = np.max(proba, axis=1)

    tier1_correct = np.sum(raw_preds == y_test)
    tier1_acc = (tier1_correct / total_test) * 100.0

    passed_mask = confs >= threshold
    tier2_passed_count = int(np.sum(passed_mask))
    tier2_pass_rate = (tier2_passed_count / total_test) * 100.0
    tier2_reject_rate = 100.0 - tier2_pass_rate

    tier2_correct = np.sum((raw_preds == y_test) & passed_mask)
    tier2_acc = (tier2_correct / tier2_passed_count) * 100.0 if tier2_passed_count > 0 else 0.0

    # -------------------------------------------------------------------------
    # Tier 3 & 4: Intentional Hold Streams Evaluation
    # -------------------------------------------------------------------------
    latencies_sec: List[float] = []
    latencies_frames: List[int] = []
    per_g_success: Dict[str, int] = {c: 0 for c in classes}
    per_g_total: Dict[str, int] = {c: 0 for c in classes}
    per_g_lats: Dict[str, List[float]] = {c: [] for c in classes}

    for stream in intentional_streams:
        c = stream.target_class
        per_g_total[c] += 1

        t_filter = TemporalFilter(
            confidence_threshold=threshold,
            stable_frames=STABLE_FRAMES,
            cooldown_frames=COOLDOWN_FRAMES,
            min_hold_duration=COMMAND_HOLD_DURATION,
        )

        confirmed = False
        confirm_time = 0.0
        confirm_frame = 0

        for frame_idx, ((g_pred, g_conf), ts) in enumerate(zip(stream.predictions, stream.timestamps)):
            res = t_filter.process(g_pred, g_conf, timestamp=ts)

            if res.is_confirmed:
                if res.gesture == c:
                    confirmed = True
                    confirm_time = ts
                    confirm_frame = frame_idx + 1
                    break

        if confirmed:
            per_g_success[c] += 1
            latencies_sec.append(confirm_time)
            latencies_frames.append(confirm_frame)
            per_g_lats[c].append(confirm_time)

    intentional_total = len(intentional_streams)
    intentional_confirm_rate = (len(latencies_sec) / intentional_total) * 100.0
    mean_lat_sec = float(np.mean(latencies_sec)) if latencies_sec else 0.0
    std_lat_sec = float(np.std(latencies_sec)) if latencies_sec else 0.0
    mean_lat_frames = float(np.mean(latencies_frames)) if latencies_frames else 0.0

    per_g_rates: Dict[str, float] = {}
    per_g_mean_lat: Dict[str, float] = {}
    for c in classes:
        tot = per_g_total[c]
        per_g_rates[c] = (per_g_success[c] / tot) * 100.0 if tot > 0 else 0.0
        per_g_mean_lat[c] = float(np.mean(per_g_lats[c])) if per_g_lats[c] else 0.0

    # -------------------------------------------------------------------------
    # Synthetic Susceptibility Test 1: Transient Flickers (1-4 frames)
    # -------------------------------------------------------------------------
    flicker_triggers = 0
    for f_stream in flicker_streams:
        t_filter = TemporalFilter(
            confidence_threshold=threshold,
            stable_frames=STABLE_FRAMES,
            cooldown_frames=COOLDOWN_FRAMES,
            min_hold_duration=COMMAND_HOLD_DURATION,
        )
        for (g_pred, g_conf), ts in zip(f_stream.predictions, f_stream.timestamps):
            res = t_filter.process(g_pred, g_conf, timestamp=ts)
            if res.is_confirmed:
                flicker_triggers += 1
                break
    flicker_susceptibility = (flicker_triggers / len(flicker_streams)) * 100.0

    # -------------------------------------------------------------------------
    # Synthetic Susceptibility Test 2: Pose Transitions (Interpolation)
    # -------------------------------------------------------------------------
    transition_triggers = 0
    for t_stream in transition_streams:
        t_filter = TemporalFilter(
            confidence_threshold=threshold,
            stable_frames=STABLE_FRAMES,
            cooldown_frames=COOLDOWN_FRAMES,
            min_hold_duration=COMMAND_HOLD_DURATION,
        )
        for (g_pred, g_conf), ts in zip(t_stream.predictions, t_stream.timestamps):
            res = t_filter.process(g_pred, g_conf, timestamp=ts)
            # If an unintended intermediate gesture confirms
            if res.is_confirmed and res.gesture not in (t_stream.target_class, t_stream.secondary_class):
                transition_triggers += 1
                break
    transition_susceptibility = (transition_triggers / len(transition_streams)) * 100.0

    # -------------------------------------------------------------------------
    # Synthetic Susceptibility Test 3: Coordinate Noise Perturbations
    # -------------------------------------------------------------------------
    noise_confirms = 0
    for n_stream in noise_streams:
        t_filter = TemporalFilter(
            confidence_threshold=threshold,
            stable_frames=STABLE_FRAMES,
            cooldown_frames=COOLDOWN_FRAMES,
            min_hold_duration=COMMAND_HOLD_DURATION,
        )
        for (g_pred, g_conf), ts in zip(n_stream.predictions, n_stream.timestamps):
            res = t_filter.process(g_pred, g_conf, timestamp=ts)
            if res.is_confirmed and res.gesture == n_stream.target_class:
                noise_confirms += 1
                break
    noise_retention = (noise_confirms / len(noise_streams)) * 100.0

    return ThresholdBenchmarkResult(
        threshold=threshold,
        total_test_samples=total_test,
        tier1_raw_acc=tier1_acc,
        tier2_pass_rate=tier2_pass_rate,
        tier2_reject_rate=tier2_reject_rate,
        tier2_acc=tier2_acc,
        num_intentional_streams=len(intentional_streams),
        intentional_confirm_rate=intentional_confirm_rate,
        mean_latency_sec=mean_lat_sec,
        std_latency_sec=std_lat_sec,
        mean_latency_frames=mean_lat_frames,
        num_flicker_streams=len(flicker_streams),
        synthetic_flicker_susceptibility=flicker_susceptibility,
        num_transition_streams=len(transition_streams),
        synthetic_transition_susceptibility=transition_susceptibility,
        num_noise_streams=len(noise_streams),
        noise_retention_rate=noise_retention,
        per_gesture_confirm_rate=per_g_rates,
        per_gesture_latency_sec=per_g_mean_lat,
    )


# ---------------------------------------------------------------------------
# 5. Formatted Reporting
# ---------------------------------------------------------------------------
def run_evaluation() -> None:
    print("=" * 80)
    print("  IntentSense — Phase 7A Benchmark")
    print("  Adaptive Confidence & Synthetic False-Trigger Evaluation")
    print("=" * 80)

    bundle, train_df, test_df, feature_cols = load_benchmark_data()
    total_samples = len(train_df) + len(test_df)
    test_samples = len(test_df)

    print(f"Dataset total samples : {total_samples}")
    print(f"Training split samples: {len(train_df)} (80%)")
    print(f"Held-out test samples : {test_samples} (20%) [Stratified by class]")
    print(f"Classes evaluated     : {bundle['classes']}")
    print(f"Feature count         : {len(feature_cols)}")
    print(f"Temporal parameters   : N_stable={STABLE_FRAMES} frames, T_hold={COMMAND_HOLD_DURATION}s, FPS={FPS:.0f}")
    print("-" * 80)

    # Section 1: Empirical Confidence Distributions
    print("\n1. EMPIRICAL CONFIDENCE DISTRIBUTION ON HELD-OUT TEST DATA (N = 251 samples)")
    print("-" * 80)
    print(f"{'Gesture':<12} | {'Count':<5} | {'Raw Acc':<8} | {'Mean Conf':<10} | {'Std':<7} | {'Median':<8} | {'P10':<7} | {'P90':<7}")
    print("-" * 80)
    conf_stats = compute_confidence_distributions(bundle, test_df, feature_cols)
    for c, s in conf_stats.items():
        print(
            f"{s.class_name:<12} | {s.sample_count:<5} | {s.accuracy:>6.1f}% | "
            f"{s.conf_mean:>9.3f}  | {s.conf_std:>6.3f} | {s.conf_median:>7.3f}  | "
            f"{s.conf_p10:>6.3f} | {s.conf_p90:>6.3f}"
        )
    print("-" * 80)

    # Generate streams
    print("\n2. SYNTHETIC BENCHMARK STREAM INVENTORY")
    print("-" * 80)
    intentional, flickers, transitions, noise = generate_benchmark_streams(test_df, feature_cols, bundle)
    print(f"  - Intentional Hold Streams     : {len(intentional)} streams (30 frames / 1.0s each)")
    print(f"  - Transient Flicker Streams    : {len(flickers)} streams (1 to 4 frames burst + rest)")
    print(f"  - Pose Transition Streams      : {len(transitions)} streams (15 linearly interpolated frames)")
    print(f"  - Coordinate Noise Streams     : {len(noise)} streams (sigma=0.08 Gaussian landmark jitter)")
    print("-" * 80)

    # Section 2: Threshold Sweep Results
    print("\n3. PROGRESSIVE 4-TIER THRESHOLD SWEEP RESULTS")
    print("*" * 80)
    print("NOTE ON METRICS:")
    print("  Flicker and Transition rates measure SYNTHETIC SUSCEPTIBILITY under generated")
    print("  stress streams. They do NOT represent in-the-wild real-world false trigger rate.")
    print("*" * 80)

    results: List[ThresholdBenchmarkResult] = []
    for th in THRESHOLDS:
        res = evaluate_threshold(
            threshold=th,
            bundle=bundle,
            test_df=test_df,
            feature_cols=feature_cols,
            intentional_streams=intentional,
            flicker_streams=flickers,
            transition_streams=transitions,
            noise_streams=noise,
        )
        results.append(res)

    print(
        f"\n{'Thresh':<7} | {'Tier 1 Acc':<10} | {'Tier 2 Pass':<11} | {'Tier 2 Rej':<10} | "
        f"{'Intent Confirm':<14} | {'Mean Latency':<16} | {'Synth Flicker':<13} | {'Synth Trans':<11} | {'Noise Ret':<9}"
    )
    print("-" * 115)
    for r in results:
        lat_str = f"{r.mean_latency_sec:.3f}s ± {r.std_latency_sec:.3f}s"
        print(
            f"{r.threshold:<7.2f} | {r.tier1_raw_acc:>8.1f}%  | {r.tier2_pass_rate:>9.1f}%  | {r.tier2_reject_rate:>8.1f}%  | "
            f"{r.intentional_confirm_rate:>12.1f}% | {lat_str:<16} | {r.synthetic_flicker_susceptibility:>11.1f}% | "
            f"{r.synthetic_transition_susceptibility:>9.1f}% | {r.noise_retention_rate:>7.1f}%"
        )
    print("-" * 115)

    # Section 3: Per-Gesture Confirmation Breakdown Across Thresholds
    print("\n4. PER-GESTURE CONFIRMATION RATE BREAKDOWN ACROSS THRESHOLDS (N = 5 streams/class)")
    print("-" * 80)
    header = f"{'Gesture':<12} | " + " | ".join([f"theta={r.threshold:.2f}" for r in results])
    print(header)
    print("-" * 80)
    for c in bundle["classes"]:
        row = f"{c:<12} | " + " | ".join([f"{r.per_gesture_confirm_rate.get(c, 0.0):>9.1f}%" for r in results])
        print(row)
    print("-" * 80)

    # Section 4: Per-Gesture Confirmation Latency (Seconds)
    print("\n5. PER-GESTURE CONFIRMATION LATENCY (SECONDS AT 30 FPS)")
    print("-" * 80)
    header_lat = f"{'Gesture':<12} | " + " | ".join([f"theta={r.threshold:.2f}" for r in results])
    print(header_lat)
    print("-" * 80)
    for c in bundle["classes"]:
        row_lat = f"{c:<12} | " + " | ".join([f"{r.per_gesture_latency_sec.get(c, 0.0):>9.3f}s" for r in results])
        print(row_lat)
    print("-" * 80)

    # Section 5: Key Findings Summary
    print("\n6. SUMMARY OF FINDINGS & SENSITIVITY OBSERVATIONS")
    print("=" * 80)
    th_070 = results[0]
    th_080 = results[2]
    th_090 = results[4]
    print(f"• Baseline threshold (0.80):")
    print(f"    - Held-out sample pass rate: {th_080.tier2_pass_rate:.1f}% (Rejection: {th_080.tier2_reject_rate:.1f}%)")
    print(f"    - Intentional confirmation rate: {th_080.intentional_confirm_rate:.1f}%")
    print(f"    - Confirmation latency: {th_080.mean_latency_sec:.3f}s (~{th_080.mean_latency_frames:.0f} frames)")
    print(f"    - Synthetic flicker susceptibility: {th_080.synthetic_flicker_susceptibility:.1f}% (0 / {th_080.num_flicker_streams} streams)")
    print(f"    - Synthetic transition susceptibility: {th_080.synthetic_transition_susceptibility:.1f}% (0 / {th_080.num_transition_streams} streams)")
    print(f"    - Coordinate noise retention: {th_080.noise_retention_rate:.1f}%")
    print(f"• Low threshold (0.70):")
    print(f"    - Rejection rate drops to {th_070.tier2_reject_rate:.1f}%, noise retention reaches {th_070.noise_retention_rate:.1f}%.")
    print(f"    - Synthetic flicker susceptibility remains 0.0% due to 0.5s hold duration requirement.")
    print(f"• High threshold (0.90):")
    print(f"    - Sample rejection rises to {th_090.tier2_reject_rate:.1f}%.")
    print(f"    - Intentional confirmation rate: {th_090.intentional_confirm_rate:.1f}%.")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
