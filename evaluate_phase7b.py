#!/usr/bin/env python3
"""
evaluate_phase7b.py
===================

Phase 7B: Adaptive Confidence Threshold Evaluation Benchmark.

Compares:
  Policy A (Baseline): Fixed 0.80 Confidence Threshold
  Policy B (Proposed): Adaptive Dual-Threshold Commitment Policy
                       - Entry threshold: 0.80 (frames 1-2, consecutive_frames < 3)
                       - Sustain threshold: 0.75 (consecutive_frames >= 3)
                       - Commit frames: 3
                       - Min bound: 0.72, Max bound: 0.85

Evaluation Dimensions:
  1. Clean Intentional Streams: Confirmation rate and confirmation latency.
  2. 1-Frame Dip Resilience: Natural optical jitter / confidence dip (0.76-0.78)
     at frame 4 during an intentional hold.
  3. Synthetic Flicker Susceptibility: 1-4 frame bursts followed by hand-off.
  4. Synthetic Transition Susceptibility: Gesture-to-gesture morphing.
  5. Synthetic Noise Retention: Coordinate perturbation under Gaussian noise (sigma=0.08).
  6. End-to-End Safety Gates: LOCKED state command suppression, OpenPalm 1.0s hold,
     and cooldown debouncing.

IMPORTANT METHODOLOGICAL NOTE:
  All numbers are calculated dynamically from actual empirical evaluation of the
  held-out test set and generated streams. No targets (such as >=75% noise retention
  or <=0.540s latency) are hard-coded, guaranteed, or presented as measured outcomes.
  Synthetic stream results measure synthetic susceptibility under stress, NOT
  real-world unscripted human false-trigger rates.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Project root setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.activation_state import ActivationStateMachine, SystemState
from core.adaptive_threshold import AdaptiveThresholdConfig, AdaptiveThresholdPolicy
from core.context_manager import Context, ContextManager
from core.intent_engine import Intent, IntentEngine
from core.temporal_filter import TemporalFilter, TemporalFilterResult

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "gestures.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_model.pkl")

FPS: float = 30.0
DT: float = 1.0 / FPS
STABLE_FRAMES: int = 5
COMMAND_HOLD_DURATION: float = 0.5
COOLDOWN_FRAMES: int = 15
RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# 1. Dataset & Model Loading
# ---------------------------------------------------------------------------
def load_benchmark_data() -> Tuple[dict, pd.DataFrame, pd.DataFrame, List[str]]:
    """Loads model bundle and partitions dataset into train and test splits."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATASET_PATH)

    feature_cols = [c for c in df.columns if c != "label"]

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    return bundle, train_df, test_df, feature_cols


# ---------------------------------------------------------------------------
# 2. Benchmark Stream Definitions
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
    bundle: dict,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[
    List[BenchmarkStream],
    List[BenchmarkStream],
    List[BenchmarkStream],
    List[BenchmarkStream],
    List[BenchmarkStream],
]:
    """Generates clean, dip, flicker, transition, and noise streams."""
    rng = np.random.RandomState(RANDOM_STATE)
    classes = sorted(test_df["label"].unique().tolist())
    stream_length = 20  # ~0.67s at 30 FPS

    # 1. Clean Intentional Streams (30 streams: 5 per class)
    intentional_streams: List[BenchmarkStream] = []
    for c in classes:
        c_samples = test_df[test_df["label"] == c][feature_cols].to_numpy()
        n_available = len(c_samples)
        for s_idx in range(5):
            indices = (np.arange(stream_length) + s_idx * 5) % n_available
            seq = [c_samples[i].copy() for i in indices]
            ts = [i * DT for i in range(stream_length)]
            intentional_streams.append(
                BenchmarkStream(
                    category="intentional_clean",
                    target_class=c,
                    secondary_class=None,
                    features_sequence=seq,
                    timestamps=ts,
                )
            )

    # 2. Intentional Streams with 1-Frame Confidence Dip (30 streams)
    # Natural tracking wobble: at frame 4 (0-indexed index 3, streak=3),
    # inject confidence between 0.76 and 0.78 while preserving gesture class.
    # We will build features for predictions directly or tweak predicted proba later.
    dip_streams: List[BenchmarkStream] = []
    for stream in intentional_streams:
        dip_seq = [f.copy() if f is not None else None for f in stream.features_sequence]
        dip_streams.append(
            BenchmarkStream(
                category="intentional_dip",
                target_class=stream.target_class,
                secondary_class=None,
                features_sequence=dip_seq,
                timestamps=list(stream.timestamps),
            )
        )

    # 3. Synthetic Transient Flicker Streams (24 streams: 1-4 frame bursts)
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

    # 4. Synthetic Transition Streams (30 streams: cross-fading pairs)
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

    # 5. Synthetic Coordinate Noise Streams (30 streams: Gaussian noise sigma=0.08)
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

    # Batch precompute predictions for all streams except dip streams
    model = bundle["model"]
    classes_arr = np.array(bundle["classes"])

    standard_streams = intentional_streams + flicker_streams + transition_streams + noise_streams
    for s in standard_streams:
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

    # Build predictions for dip_streams: identical to intentional_streams, but inject
    # a momentary 0.77 confidence dip at frame 4 (index 3) while keeping target class
    for s_clean, s_dip in zip(intentional_streams, dip_streams):
        dip_preds = list(s_clean.predictions)
        if len(dip_preds) > 3:
            # At 1-based frame 4 (index 3):
            # consecutive_frames accumulated so far was 3.
            # Inject a realistic confidence dip of 0.77 (below 0.80 fixed threshold,
            # but above 0.75 sustain threshold).
            orig_class, _ = dip_preds[3]
            dip_preds[3] = (orig_class or s_dip.target_class, 0.77)
        s_dip.predictions = dip_preds

    return intentional_streams, dip_streams, flicker_streams, transition_streams, noise_streams


# ---------------------------------------------------------------------------
# 3. Policy Evaluation Routine
# ---------------------------------------------------------------------------
@dataclass
class PolicyEvaluationResult:
    policy_name: str
    threshold_desc: str

    # Clean intentional streams
    num_clean_streams: int
    clean_confirm_rate: float
    clean_mean_latency_sec: float
    clean_std_latency_sec: float
    clean_mean_latency_frames: float

    # 1-Frame Dip streams
    num_dip_streams: int
    dip_confirm_rate: float
    dip_mean_latency_sec: float
    dip_streak_resets: int

    # Synthetic Stress Susceptibility
    num_flicker_streams: int
    synthetic_flicker_susceptibility: float
    num_transition_streams: int
    synthetic_transition_susceptibility: float
    num_noise_streams: int
    noise_retention_rate: float

    # Per-gesture breakdown
    per_gesture_confirm_rate: Dict[str, float] = field(default_factory=dict)
    per_gesture_latency_sec: Dict[str, float] = field(default_factory=dict)

    # Safety Gate Verifications
    locked_state_violations: int = 0
    cooldown_violations: int = 0


def run_policy_benchmark(
    policy_name: str,
    threshold_policy: Optional[AdaptiveThresholdPolicy],
    fixed_threshold: float,
    classes: List[str],
    clean_streams: List[BenchmarkStream],
    dip_streams: List[BenchmarkStream],
    flicker_streams: List[BenchmarkStream],
    transition_streams: List[BenchmarkStream],
    noise_streams: List[BenchmarkStream],
) -> PolicyEvaluationResult:
    """Evaluates a single threshold configuration across all benchmark dimensions."""

    def make_filter() -> TemporalFilter:
        return TemporalFilter(
            confidence_threshold=fixed_threshold,
            stable_frames=STABLE_FRAMES,
            cooldown_frames=COOLDOWN_FRAMES,
            min_hold_duration=COMMAND_HOLD_DURATION,
            threshold_policy=threshold_policy,
        )

    # -------------------------------------------------------------------------
    # A. Clean Intentional Streams
    # -------------------------------------------------------------------------
    clean_latencies_sec: List[float] = []
    clean_latencies_frames: List[int] = []
    per_g_success: Dict[str, int] = {c: 0 for c in classes}
    per_g_total: Dict[str, int] = {c: 0 for c in classes}
    per_g_lats: Dict[str, List[float]] = {c: [] for c in classes}

    for stream in clean_streams:
        c = stream.target_class
        per_g_total[c] += 1
        t_filter = make_filter()
        confirmed = False
        confirm_time = 0.0
        confirm_frame = 0

        for frame_idx, ((g_pred, g_conf), ts) in enumerate(zip(stream.predictions, stream.timestamps)):
            res = t_filter.process(g_pred, g_conf, timestamp=ts)
            if res.is_confirmed and res.gesture == c:
                confirmed = True
                confirm_time = ts
                confirm_frame = frame_idx + 1
                break

        if confirmed:
            per_g_success[c] += 1
            clean_latencies_sec.append(confirm_time)
            clean_latencies_frames.append(confirm_frame)
            per_g_lats[c].append(confirm_time)

    clean_confirm_rate = (len(clean_latencies_sec) / len(clean_streams)) * 100.0
    clean_mean_lat = float(np.mean(clean_latencies_sec)) if clean_latencies_sec else 0.0
    clean_std_lat = float(np.std(clean_latencies_sec)) if clean_latencies_sec else 0.0
    clean_mean_frames = float(np.mean(clean_latencies_frames)) if clean_latencies_frames else 0.0

    per_g_rate = {
        c: (per_g_success[c] / per_g_total[c] * 100.0) if per_g_total[c] > 0 else 0.0
        for c in classes
    }
    per_g_lat_mean = {
        c: (float(np.mean(per_g_lats[c])) if per_g_lats[c] else 0.0)
        for c in classes
    }

    # -------------------------------------------------------------------------
    # B. Intentional Streams with Mid-Hold 1-Frame Dip (at Frame 4)
    # -------------------------------------------------------------------------
    dip_latencies_sec: List[float] = []
    streak_resets_count = 0

    for stream in dip_streams:
        c = stream.target_class
        t_filter = make_filter()
        confirmed = False
        confirm_time = 0.0

        for frame_idx, ((g_pred, g_conf), ts) in enumerate(zip(stream.predictions, stream.timestamps)):
            res = t_filter.process(g_pred, g_conf, timestamp=ts)
            # Track if frame 4 caused a reset
            if frame_idx == 3 and res.reason == "LOW_CONFIDENCE":
                streak_resets_count += 1
            if res.is_confirmed and res.gesture == c:
                confirmed = True
                confirm_time = ts
                break

        if confirmed:
            dip_latencies_sec.append(confirm_time)

    dip_confirm_rate = (len(dip_latencies_sec) / len(dip_streams)) * 100.0
    dip_mean_lat = float(np.mean(dip_latencies_sec)) if dip_latencies_sec else 0.0

    # -------------------------------------------------------------------------
    # C. Synthetic Flicker Susceptibility (24 streams)
    # -------------------------------------------------------------------------
    flicker_triggers = 0
    for stream in flicker_streams:
        t_filter = make_filter()
        for (g_pred, g_conf), ts in zip(stream.predictions, stream.timestamps):
            res = t_filter.process(g_pred, g_conf, timestamp=ts)
            if res.is_confirmed:
                flicker_triggers += 1
                break
    synthetic_flicker_suscep = (flicker_triggers / len(flicker_streams)) * 100.0

    # -------------------------------------------------------------------------
    # D. Synthetic Transition Susceptibility (30 streams)
    # -------------------------------------------------------------------------
    transition_triggers = 0
    for stream in transition_streams:
        t_filter = make_filter()
        for (g_pred, g_conf), ts in zip(stream.predictions, stream.timestamps):
            res = t_filter.process(g_pred, g_conf, timestamp=ts)
            if res.is_confirmed:
                transition_triggers += 1
                break
    synthetic_trans_suscep = (transition_triggers / len(transition_streams)) * 100.0

    # -------------------------------------------------------------------------
    # E. Synthetic Coordinate Noise Retention (30 streams)
    # -------------------------------------------------------------------------
    noise_confirms = 0
    for stream in noise_streams:
        c = stream.target_class
        t_filter = make_filter()
        for (g_pred, g_conf), ts in zip(stream.predictions, stream.timestamps):
            res = t_filter.process(g_pred, g_conf, timestamp=ts)
            if res.is_confirmed and res.gesture == c:
                noise_confirms += 1
                break
    noise_retention = (noise_confirms / len(noise_streams)) * 100.0

    # -------------------------------------------------------------------------
    # F. End-to-End Safety Gate Verifications
    # -------------------------------------------------------------------------
    # 1. Verify commands strictly suppressed when SYSTEM_LOCKED
    locked_violations = 0
    asm = ActivationStateMachine()
    ctx = ContextManager()
    engine = IntentEngine()
    tf_test = make_filter()

    # Feed 10 frames of ThumbsUp with 0.95 confidence while in LOCKED state
    for f in range(10):
        t_res = tf_test.process("ThumbsUp", 0.95, timestamp=f * DT)
        i_res = engine.process(
            filter_result=t_res,
            state=asm.state,
            context=ctx.current_context,
        )
        if i_res.accepted:
            locked_violations += 1

    # 2. Verify cooldown prevents immediate re-confirmation
    cooldown_violations = 0
    tf_cd = make_filter()
    # Confirm once at frame 16 (0.5s hold)
    first_confirmed = False
    for f in range(25):
        t_res = tf_cd.process("Fist", 0.95, timestamp=f * DT)
        if t_res.is_confirmed:
            if not first_confirmed:
                first_confirmed = True
            else:
                # Re-confirmed during cooldown!
                cooldown_violations += 1

    desc = (
        f"Fixed theta={fixed_threshold:.2f}"
        if threshold_policy is None
        else f"Entry={threshold_policy.entry_threshold:.2f}, Sustain={threshold_policy.sustain_threshold:.2f}, Commit={threshold_policy.commit_frames}"
    )

    return PolicyEvaluationResult(
        policy_name=policy_name,
        threshold_desc=desc,
        num_clean_streams=len(clean_streams),
        clean_confirm_rate=clean_confirm_rate,
        clean_mean_latency_sec=clean_mean_lat,
        clean_std_latency_sec=clean_std_lat,
        clean_mean_latency_frames=clean_mean_frames,
        num_dip_streams=len(dip_streams),
        dip_confirm_rate=dip_confirm_rate,
        dip_mean_latency_sec=dip_mean_lat,
        dip_streak_resets=streak_resets_count,
        num_flicker_streams=len(flicker_streams),
        synthetic_flicker_susceptibility=synthetic_flicker_suscep,
        num_transition_streams=len(transition_streams),
        synthetic_transition_susceptibility=synthetic_trans_suscep,
        num_noise_streams=len(noise_streams),
        noise_retention_rate=noise_retention,
        per_gesture_confirm_rate=per_g_rate,
        per_gesture_latency_sec=per_g_lat_mean,
        locked_state_violations=locked_violations,
        cooldown_violations=cooldown_violations,
    )


# ---------------------------------------------------------------------------
# 4. Unit Verification: Invariants & Safety Gates
# ---------------------------------------------------------------------------
def run_unit_invariants_verification() -> bool:
    """Verifies architectural invariants of AdaptiveThresholdPolicy and TemporalFilter."""
    print("=" * 80)
    print("0. ARCHITECTURAL INVARIANTS & POLICY VERIFICATION")
    print("=" * 80)

    # 1. Policy Configuration Invariants
    cfg = AdaptiveThresholdConfig(
        default_entry_threshold=0.80,
        default_sustain_threshold=0.75,
        commit_frames=3,
        min_bound=0.72,
        max_bound=0.85,
    )
    policy = AdaptiveThresholdPolicy(cfg)

    # Frame 1 & 2 (1-indexed streak < 3) must return entry threshold 0.80
    assert policy.get_threshold("Pointing", 1) == 0.80, "Streak 1 must return entry threshold 0.80"
    assert policy.get_threshold("Pointing", 2) == 0.80, "Streak 2 must return entry threshold 0.80"

    # Frame 3+ (1-indexed streak >= 3) must return sustain threshold 0.75
    assert policy.get_threshold("Pointing", 3) == 0.75, "Streak 3 must return sustain threshold 0.75"
    assert policy.get_threshold("Pointing", 4) == 0.75, "Streak 4 must return sustain threshold 0.75"
    assert policy.get_threshold("Pointing", 5) == 0.75, "Streak 5 must return sustain threshold 0.75"

    # Clamping invariants
    assert policy.get_threshold("Pointing", 0) == 0.80
    print("  ✓ Policy frame threshold rules verified (frames 1-2: 0.80, frames 3+: 0.75)")

    # 2. TemporalFilter Integration & Streak Behavior
    tf = TemporalFilter(
        confidence_threshold=0.80,
        stable_frames=5,
        cooldown_frames=15,
        min_hold_duration=0.0,
        threshold_policy=policy,
    )

    # Frame 1: confidence 0.81 passes (entry 0.80)
    r1 = tf.process("Fist", 0.81)
    assert r1.consecutive_frames == 1
    assert tf.get_effective_threshold("Fist") == 0.80

    # Frame 2: confidence 0.81 passes (entry 0.80)
    r2 = tf.process("Fist", 0.81)
    assert r2.consecutive_frames == 2
    assert tf.get_effective_threshold("Fist") == 0.75

    # Frame 3: confidence 0.76 passes (sustain 0.75)
    r3 = tf.process("Fist", 0.76)
    assert r3.consecutive_frames == 3
    assert tf.get_effective_threshold("Fist") == 0.75

    # Frame 4: confidence 0.74 fails (below sustain 0.75) -> streak resets to 0
    r4 = tf.process("Fist", 0.74)
    assert r4.consecutive_frames == 0
    assert r4.reason == "LOW_CONFIDENCE"
    assert tf.get_effective_threshold("Fist") == 0.80

    print("  ✓ TemporalFilter streak accumulation and reset behavior verified")

    # 3. Backward Compatibility: When threshold_policy is None, fixed threshold 0.80 is strict
    tf_fixed = TemporalFilter(confidence_threshold=0.80, threshold_policy=None)
    assert tf_fixed.get_effective_threshold("Fist") == 0.80
    tf_fixed.process("Fist", 0.85)
    tf_fixed.process("Fist", 0.85)
    tf_fixed.process("Fist", 0.85)
    # Streak is now 3, but threshold remains strictly 0.80
    assert tf_fixed.get_effective_threshold("Fist") == 0.80
    r_fail = tf_fixed.process("Fist", 0.78)
    assert r_fail.reason == "LOW_CONFIDENCE"
    print("  ✓ Backward compatibility verified (policy=None preserves 100% fixed 0.80 behavior)")
    print()
    return True


# ---------------------------------------------------------------------------
# 5. Main Benchmark Execution
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print(" IntentSense Phase 7B: Adaptive Threshold Benchmark")
    print("=" * 80)
    print()

    # Step 0: Invariants & unit verification
    run_unit_invariants_verification()

    # Step 1: Load data and split
    print("Loading model bundle and held-out test dataset...")
    bundle, train_df, test_df, feature_cols = load_benchmark_data()
    classes = sorted(test_df["label"].unique().tolist())
    total_test = len(test_df)
    print(f"  Test set samples: {total_test} across {len(classes)} classes")
    print()

    # Step 2: Generate benchmark streams
    print("Generating reproducible benchmark streams...")
    clean_streams, dip_streams, flicker_streams, transition_streams, noise_streams = (
        generate_benchmark_streams(bundle, test_df, feature_cols)
    )
    print(f"  Intentional Clean Streams:      {len(clean_streams)} streams (20 frames each)")
    print(f"  Mid-Hold 1-Frame Dip Streams:   {len(dip_streams)} streams (dip at frame 4)")
    print(f"  Synthetic Flicker Streams:      {len(flicker_streams)} streams (1-4 frame bursts)")
    print(f"  Synthetic Transition Streams:   {len(transition_streams)} streams (15 interpolation steps)")
    print(f"  Synthetic Noise Streams:        {len(noise_streams)} streams (Gaussian sigma=0.08)")
    print()

    # Step 3: Run Baseline (Fixed 0.80)
    print("Running Policy A: Fixed 0.80 Baseline...")
    res_baseline = run_policy_benchmark(
        policy_name="Fixed Baseline",
        threshold_policy=None,
        fixed_threshold=0.80,
        classes=classes,
        clean_streams=clean_streams,
        dip_streams=dip_streams,
        flicker_streams=flicker_streams,
        transition_streams=transition_streams,
        noise_streams=noise_streams,
    )

    # Step 4: Run Proposed Adaptive Policy (0.80 Entry / 0.75 Sustain / Commit=3)
    print("Running Policy B: Adaptive Commitment Policy (0.80 / 0.75)...")
    policy_cfg = AdaptiveThresholdConfig(
        default_entry_threshold=0.80,
        default_sustain_threshold=0.75,
        commit_frames=3,
        min_bound=0.72,
        max_bound=0.85,
    )
    adaptive_policy = AdaptiveThresholdPolicy(policy_cfg)
    res_adaptive = run_policy_benchmark(
        policy_name="Adaptive Policy",
        threshold_policy=adaptive_policy,
        fixed_threshold=0.80,
        classes=classes,
        clean_streams=clean_streams,
        dip_streams=dip_streams,
        flicker_streams=flicker_streams,
        transition_streams=transition_streams,
        noise_streams=noise_streams,
    )

    # Step 5: Format and Display Comparison Tables
    print()
    print("=" * 80)
    print(" COMPARATIVE BENCHMARK RESULTS: FIXED 0.80 vs. ADAPTIVE (0.80 -> 0.75)")
    print("=" * 80)

    delta_clean_conf = res_adaptive.clean_confirm_rate - res_baseline.clean_confirm_rate
    delta_clean_lat = res_adaptive.clean_mean_latency_sec - res_baseline.clean_mean_latency_sec
    delta_dip_conf = res_adaptive.dip_confirm_rate - res_baseline.dip_confirm_rate
    delta_dip_lat = res_adaptive.dip_mean_latency_sec - res_baseline.dip_mean_latency_sec
    delta_noise = res_adaptive.noise_retention_rate - res_baseline.noise_retention_rate

    print(
        f"{'Metric':<38} | {'Fixed 0.80 Baseline':<18} | {'Adaptive (0.80/0.75)':<20} | {'Delta':<10}"
    )
    print("-" * 94)
    print(
        f"{'Clean Confirmation Rate (%)':<38} | {res_baseline.clean_confirm_rate:>16.1f}% | {res_adaptive.clean_confirm_rate:>18.1f}% | {delta_clean_conf:>+8.1f}%"
    )
    print(
        f"{'Clean Mean Latency (s)':<38} | {res_baseline.clean_mean_latency_sec:>17.3f}s | {res_adaptive.clean_mean_latency_sec:>19.3f}s | {delta_clean_lat:>+8.3f}s"
    )
    print(
        f"{'Clean Latency StdDev (s)':<38} | {res_baseline.clean_std_latency_sec:>17.3f}s | {res_adaptive.clean_std_latency_sec:>19.3f}s |"
    )
    print(
        f"{'Clean Latency (mean frames)':<38} | {res_baseline.clean_mean_latency_frames:>18.1f} | {res_adaptive.clean_mean_latency_frames:>20.1f} |"
    )
    print("-" * 94)
    print(
        f"{'1-Frame Dip Confirm Rate (%)':<38} | {res_baseline.dip_confirm_rate:>16.1f}% | {res_adaptive.dip_confirm_rate:>18.1f}% | {delta_dip_conf:>+8.1f}%"
    )
    print(
        f"{'1-Frame Dip Mean Latency (s)':<38} | {res_baseline.dip_mean_latency_sec:>17.3f}s | {res_adaptive.dip_mean_latency_sec:>19.3f}s | {delta_dip_lat:>+8.3f}s"
    )
    print(
        f"{'1-Frame Dip Streak Resets (count)':<38} | {res_baseline.dip_streak_resets:>18d} | {res_adaptive.dip_streak_resets:>20d} | {res_adaptive.dip_streak_resets - res_baseline.dip_streak_resets:>+8d}"
    )
    print("-" * 94)
    print(
        f"{'Synthetic Flicker Susceptibility (%)':<38} | {res_baseline.synthetic_flicker_susceptibility:>16.1f}% | {res_adaptive.synthetic_flicker_susceptibility:>18.1f}% | {res_adaptive.synthetic_flicker_susceptibility - res_baseline.synthetic_flicker_susceptibility:>+8.1f}%"
    )
    print(
        f"{'Synthetic Transition Susceptibility (%)':<38} | {res_baseline.synthetic_transition_susceptibility:>16.1f}% | {res_adaptive.synthetic_transition_susceptibility:>18.1f}% | {res_adaptive.synthetic_transition_susceptibility - res_baseline.synthetic_transition_susceptibility:>+8.1f}%"
    )
    print(
        f"{'Synthetic Noise Retention Rate (%)':<38} | {res_baseline.noise_retention_rate:>16.1f}% | {res_adaptive.noise_retention_rate:>18.1f}% | {delta_noise:>+8.1f}%"
    )
    print("-" * 94)
    print(
        f"{'LOCKED State Command Violations':<38} | {res_baseline.locked_state_violations:>18d} | {res_adaptive.locked_state_violations:>20d} |"
    )
    print(
        f"{'Action Cooldown Violations':<38} | {res_baseline.cooldown_violations:>18d} | {res_adaptive.cooldown_violations:>20d} |"
    )
    print("=" * 94)
    print()

    # Step 6: Per-Gesture Breakdown
    print("=" * 80)
    print(" PER-GESTURE CONFIRMATION RATE BREAKDOWN")
    print("=" * 80)
    print(f"{'Gesture Class':<16} | {'Fixed 0.80 Confirm':<20} | {'Adaptive Confirm':<18} | {'Delta':<10}")
    print("-" * 72)
    for c in classes:
        base_rate = res_baseline.per_gesture_confirm_rate.get(c, 0.0)
        adap_rate = res_adaptive.per_gesture_confirm_rate.get(c, 0.0)
        d_rate = adap_rate - base_rate
        print(f"{c:<16} | {base_rate:>18.1f}% | {adap_rate:>16.1f}% | {d_rate:>+8.1f}%")
    print("=" * 72)
    print()

    print("=" * 80)
    print(" SUMMARY OF EMPIRICAL FINDINGS")
    print("=" * 80)
    print(f"1. Clean Intentional Latency:")
    print(f"   - Fixed 0.80:    {res_baseline.clean_mean_latency_sec:.3f}s ({res_baseline.clean_mean_latency_frames:.1f} frames)")
    print(f"   - Adaptive:      {res_adaptive.clean_mean_latency_sec:.3f}s ({res_adaptive.clean_mean_latency_frames:.1f} frames)")
    print()
    print(f"2. Fluctuation (1-Frame Dip) Resilience:")
    print(f"   - Fixed 0.80:    {res_baseline.dip_confirm_rate:.1f}% confirmed (resets: {res_baseline.dip_streak_resets}/{res_baseline.num_dip_streams})")
    print(f"   - Adaptive:      {res_adaptive.dip_confirm_rate:.1f}% confirmed (resets: {res_adaptive.dip_streak_resets}/{res_adaptive.num_dip_streams})")
    print(f"   - Latency under dip: Fixed={res_baseline.dip_mean_latency_sec:.3f}s vs Adaptive={res_adaptive.dip_mean_latency_sec:.3f}s")
    print()
    print(f"3. Synthetic Stress Susceptibility:")
    print(f"   - Synthetic Flicker Susceptibility:    {res_adaptive.synthetic_flicker_susceptibility:.1f}% (target: 0.0%)")
    print(f"   - Synthetic Transition Susceptibility: {res_adaptive.synthetic_transition_susceptibility:.1f}% (target: 0.0%)")
    print(f"   - Coordinate Noise Retention:         {res_adaptive.noise_retention_rate:.1f}% (Fixed was {res_baseline.noise_retention_rate:.1f}%)")
    print()
    print(f"4. Hard Safety Gates:")
    print(f"   - LOCKED state command executions:    {res_adaptive.locked_state_violations} (Zero tolerance)")
    print(f"   - Action cooldown double-triggers:    {res_adaptive.cooldown_violations} (Zero tolerance)")
    print("=" * 80)


if __name__ == "__main__":
    main()
