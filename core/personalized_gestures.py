"""
core/personalized_gestures.py
==============================

IntentSense Phase 11: Personalized Gestures Architecture.

Enables users to enroll custom hand poses, match them using normalized 42-dimensional
wrist-relative feature vectors, and route them to existing supported Intents without
altering or retraining the built-in Random Forest model or dataset.

Key Principles & Guarantees:
1. Strict Built-In Precedence (4-Tier Arbitration):
   - Tier 1: Built-in model confidence >= effective threshold -> Built-in wins.
   - Tier 2: Built-in model confidence in [0.50, effective_threshold) -> Built-in Guard;
             custom gestures are STRICTLY INHIBITED to prevent misfiring during
             built-in transitions or half-formed gestures.
   - Tier 3: Built-in model confidence < 0.50 -> Evaluates custom matcher.
   - Tier 4: No hand detected -> All gestures suppressed.
2. Mathematically Consistent Distance & Confidence:
   - tau_ideal = 0.10, tau_accept = 0.30, C_boundary = 0.70.
   - Exact entry threshold of 0.80 corresponds to d <= 0.233.
   - Distance d in (0.233, 0.30] yields confidence in [0.70, 0.80) (weak match; rejected).
   - Distance d > 0.30 yields confidence 0.00 (out-of-boundary / unknown pose).
3. Ambiguity Rejection:
   - If two custom gestures have centroids within delta_ambiguity (0.10), frame is rejected.
4. Multi-Take Enrollment with Holdout Validation:
   - 3 distinct bursts: Take 1 (10 frames) + Take 2 (10 frames) for training;
     Take 3 (10 frames) for holdout validation.
   - Quality gates: intra-class dispersion sigma <= 0.12, validation pass rate >= 80%.
   - Conflict check: Rejects enrollment if pose shadows a built-in gesture or existing custom gesture.
5. Completely Isolated Persistence:
   - Stored in models/custom_gestures.json; built-in models and dataset remain pristine.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.context_manager import Context
from core.intent_engine import KNOWN_GESTURES, Intent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TAU_IDEAL: float = 0.10              # Distance for ideal steady hold (palm lengths)
TAU_ACCEPT: float = 0.30             # Hard distance cutoff for custom gesture matching
CONFIDENCE_BOUNDARY: float = 0.70    # Confidence score at d = TAU_ACCEPT
DELTA_AMBIGUITY: float = 0.10        # Minimum separation between top two custom candidates
BUILTIN_GUARD_THRESHOLD: float = 0.50 # Built-in confidence floor below which custom poses are evaluated
TEMPORAL_ENTRY_CONFIDENCE: float = 0.80 # Baseline confidence required to accumulate temporal streak

# Distance corresponding to confidence == 0.80:
# C = 1.0 - ((d - 0.10) / (0.30 - 0.10)) * (1.0 - 0.70) = 0.80
# 0.20 = ((d - 0.10) / 0.20) * 0.30 => (d - 0.10) / 0.20 = 2/3 => d = 0.10 + 0.1333... = 0.2333...
TAU_ENTRY_THRESHOLD: float = TAU_IDEAL + ((1.0 - TEMPORAL_ENTRY_CONFIDENCE) / (1.0 - CONFIDENCE_BOUNDARY)) * (TAU_ACCEPT - TAU_IDEAL)

RESERVED_NAMES = set(KNOWN_GESTURES) | {"None", "SwipeLeft", "SwipeRight", "OpenPalm"}

# Mapping of supported action names to Intent enums for user-friendly custom gesture assignment
SUPPORTED_ACTION_MAP: Dict[str, Intent] = {
    "play_pause": Intent.PLAY_PAUSE,
    "previous_track": Intent.PREVIOUS,
    "next_track": Intent.NEXT,
    "volume_up": Intent.VOLUME_UP,
    "volume_down": Intent.VOLUME_DOWN,
    "start_stop_presentation": Intent.START_STOP,
    "start_stop": Intent.START_STOP,
    "next_slide": Intent.NEXT_SLIDE,
    "previous_slide": Intent.PREVIOUS_SLIDE,
    "zoom_in": Intent.ZOOM_IN,
    "zoom_out": Intent.ZOOM_OUT,
}


def calculate_custom_confidence(distance: float) -> float:
    """
    Computes continuous piecewise-linear confidence score from Euclidean distance.

    Args:
        distance: Euclidean distance to gesture centroid in palm-length units.

    Returns:
        float: Confidence score in [0.0, 1.0].
    """
    if distance <= TAU_IDEAL:
        return 1.0
    elif distance <= TAU_ACCEPT:
        ratio = (distance - TAU_IDEAL) / (TAU_ACCEPT - TAU_IDEAL)
        return float(1.0 - ratio * (1.0 - CONFIDENCE_BOUNDARY))
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class GestureProfile:
    """
    Persistent profile representing an enrolled custom gesture.
    """
    name: str
    centroid: List[float]
    dispersion: float
    sample_count: int
    mappings: Dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GestureProfile:
        return cls(
            name=data["name"],
            centroid=[float(v) for v in data["centroid"]],
            dispersion=float(data["dispersion"]),
            sample_count=int(data["sample_count"]),
            mappings=dict(data.get("mappings", {})),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            description=data.get("description", ""),
        )


@dataclass
class CustomMatchResult:
    """Outcome of matching an input feature vector against custom profiles."""
    gesture: Optional[str]
    confidence: float
    distance: float
    is_match: bool
    reason: str


@dataclass
class ArbitrationResult:
    """Outcome of 4-Tier Built-In vs Custom arbitration for a single frame."""
    candidate_gesture: Optional[str]
    confidence: float
    source: str   # "BUILTIN", "CUSTOM", or "NONE"
    tier: int     # 1, 2, 3, or 4
    reason: str


# ---------------------------------------------------------------------------
# Multi-Take Enrollment Validator
# ---------------------------------------------------------------------------
def validate_enrollment_samples(
    take1: Sequence[Sequence[float]],
    take2: Sequence[Sequence[float]],
    take3_val: Sequence[Sequence[float]],
    builtin_classifier: Optional[Any] = None,
    existing_centroids: Optional[Dict[str, Sequence[float]]] = None,
) -> Tuple[List[float], float]:
    """
    Validates samples across a 3-take protocol with holdout cross-validation.

    Args:
        take1: Training burst 1 (min 10 frames of 42D features).
        take2: Training burst 2 (min 10 frames of 42D features).
        take3_val: Holdout validation burst (min 10 frames of 42D features).
        builtin_classifier: Optional model bundle or classifier to check built-in conflicts.
        existing_centroids: Optional mapping of existing custom gesture centroids to check overlap.

    Returns:
        Tuple[List[float], float]: (centroid, dispersion)

    Raises:
        ValueError: For sample size or feature dimension mismatch.
        RuntimeError: For unstable training samples, validation failure, or conflicts.
    """
    if len(take1) < 10 or len(take2) < 10:
        raise ValueError(
            f"Training requires at least 10 samples per take. Got Take 1: {len(take1)}, Take 2: {len(take2)}"
        )
    if len(take3_val) < 10:
        raise ValueError(f"Holdout validation requires at least 10 samples. Got: {len(take3_val)}")

    train_samples = list(take1) + list(take2)
    for i, s in enumerate(train_samples):
        if len(s) != 42:
            raise ValueError(f"Sample {i} must have exactly 42 features; got {len(s)}")
    for j, s in enumerate(take3_val):
        if len(s) != 42:
            raise ValueError(f"Validation sample {j} must have exactly 42 features; got {len(s)}")

    train_arr = np.asarray(train_samples, dtype=float)
    centroid_arr = np.mean(train_arr, axis=0)
    diffs = train_arr - centroid_arr
    distances = np.linalg.norm(diffs, axis=1)
    dispersion = float(np.mean(distances))

    # Gate 1: Intra-class dispersion check (<= 0.12)
    if dispersion > 0.12:
        raise RuntimeError(
            f"ENROLLMENT_UNSTABLE: Training dispersion ({dispersion:.3f}) exceeds stability threshold (0.120). "
            f"Please keep your hand steadier across takes."
        )

    # Gate 2: Conflict check with existing custom centroids
    if existing_centroids:
        for ex_name, ex_cent in existing_centroids.items():
            dist_to_ex = float(np.linalg.norm(centroid_arr - np.asarray(ex_cent, dtype=float)))
            if dist_to_ex < 0.20:
                raise RuntimeError(
                    f"CONFLICT_WITH_CUSTOM: Pose is too close to existing custom gesture '{ex_name}' "
                    f"(distance {dist_to_ex:.3f} < 0.200)."
                )

    # Gate 3: Conflict check with built-in model
    if builtin_classifier is not None:
        clf = builtin_classifier.get("model", builtin_classifier) if isinstance(builtin_classifier, dict) else builtin_classifier
        classes = getattr(clf, "classes_", None)
        if classes is None and isinstance(builtin_classifier, dict):
            classes = builtin_classifier.get("classes", None)

        if hasattr(clf, "predict_proba") and classes is not None:
            val_and_train = list(train_samples) + list(take3_val)
            probas = clf.predict_proba(np.asarray(val_and_train, dtype=float))
            high_conf_count = 0
            conflicting_class = "Unknown"
            for p in probas:
                max_idx = int(np.argmax(p))
                max_conf = float(p[max_idx])
                if max_conf >= 0.65:
                    high_conf_count += 1
                    conflicting_class = str(classes[max_idx])

            conflict_ratio = high_conf_count / len(val_and_train)
            if conflict_ratio > 0.30:
                raise RuntimeError(
                    f"CONFLICT_WITH_BUILTIN: Pose resembles built-in gesture '{conflicting_class}' "
                    f"({conflict_ratio*100:.1f}% of frames had built-in confidence >= 0.65)."
                )

    # Gate 4: Holdout validation
    val_arr = np.asarray(take3_val, dtype=float)
    val_diffs = val_arr - centroid_arr
    val_distances = np.linalg.norm(val_diffs, axis=1)

    qualifying_count = sum(1 for d in val_distances if d <= TAU_ENTRY_THRESHOLD)
    max_val_dist = float(np.max(val_distances))

    if qualifying_count < 8:
        raise RuntimeError(
            f"ENROLLMENT_VALIDATION_FAILED: Only {qualifying_count}/10 holdout validation samples "
            f"satisfied entry requirement (d <= {TAU_ENTRY_THRESHOLD:.3f})."
        )
    if max_val_dist > 0.28:
        raise RuntimeError(
            f"ENROLLMENT_VALIDATION_FAILED: Outlier validation sample detected (max distance {max_val_dist:.3f} > 0.280)."
        )

    return centroid_arr.tolist(), dispersion


def format_enrollment_error(err: Union[Exception, str]) -> str:
    """
    Translates internal mathematical/validation errors into clear, friendly guidance
    for non-programmer users.
    """
    msg = str(err)
    if "TOO_FEW_SAMPLES" in msg or "at least 10 samples" in msg.lower():
        return "Not enough stable hand frames were captured. Please hold your hand in view."
    if "OUTLIER" in msg.upper() or "abruptly" in msg.lower():
        return "Inconsistent hand position or movement detected. Keep your hand steady and retry."
    if "ENROLLMENT_UNSTABLE" in msg or "HIGH_DISPERSION" in msg or "dispersion" in msg.lower():
        return "Hand moved too much during capture. Please hold your hand steady and retry."
    if "CONFLICT_WITH_CUSTOM" in msg or "CONFLICTS_WITH_CUSTOM" in msg:
        # Extract custom gesture name if present
        parts = msg.split(":")
        custom_name = parts[1].strip() if len(parts) > 1 else ""
        if custom_name:
            return f"Pose is too similar to existing custom gesture '{custom_name}'. Try a different hand shape."
        return "Pose is too similar to an existing custom gesture. Try a different hand shape."
    if "CONFLICT_WITH_BUILTIN" in msg or "CONFLICTS_WITH_BUILTIN" in msg:
        parts = msg.split(":")
        builtin_name = parts[1].strip() if len(parts) > 1 else ""
        if builtin_name:
            return f"Pose looks too similar to built-in gesture '{builtin_name}'. Try a different hand shape."
        return "Pose looks too much like a built-in gesture (e.g. Fist/ThumbsUp). Try a different hand shape."
    if "ENROLLMENT_VALIDATION_FAILED" in msg:
        return "Validation failed: pose varied between takes. Please hold the same pose consistently across takes."
    if "reserved" in msg.lower():
        return "That name is reserved for built-in gestures. Please choose a custom name."
    if "already exists" in msg.lower():
        return "A gesture with this name already exists. Use Manage (M) to edit or pick a new name."
    if "empty" in msg.lower():
        return "Gesture name cannot be empty."
    return f"Enrollment could not be validated: {msg}"


# ---------------------------------------------------------------------------
# Personalized Gesture Manager
# ---------------------------------------------------------------------------
class PersonalizedGestureManager:
    """
    Manages custom gesture profiles, Euclidean matching, and 4-tier arbitration.
    """

    def __init__(self, profile_path: Optional[str] = None) -> None:
        if profile_path is not None:
            self.profile_path = profile_path
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.profile_path = os.path.join(base_dir, "models", "custom_gestures.json")

        self._profiles: Dict[str, GestureProfile] = {}
        self.load_profiles()

    @property
    def profiles(self) -> Dict[str, GestureProfile]:
        return dict(self._profiles)

    def load_profiles(self) -> None:
        """Loads custom gesture profiles from disk if file exists."""
        self._profiles.clear()
        if not os.path.isfile(self.profile_path):
            return

        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            gestures_data = data.get("gestures", {})
            for name, g_data in gestures_data.items():
                self._profiles[name] = GestureProfile.from_dict(g_data)
        except Exception as e:
            print(f"[PersonalizedGestures] Warning: Failed to load profiles from {self.profile_path}: {e}")

    def save_profiles(self) -> None:
        """Saves custom gesture profiles persistently to disk with atomic write."""
        dir_path = os.path.dirname(os.path.abspath(self.profile_path))
        os.makedirs(dir_path, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "gestures": {name: p.to_dict() for name, p in self._profiles.items()},
        }
        # Atomic write to temporary file first, then rename
        temp_file = self.profile_path + ".tmp"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_file, self.profile_path)
        except Exception:
            # Fallback direct write
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def register_gesture(
        self,
        name: str,
        centroid: Sequence[float],
        dispersion: float,
        sample_count: int,
        mappings: Optional[Dict[str, str]] = None,
        description: str = "",
        allow_overwrite: bool = False,
    ) -> GestureProfile:
        """Registers a verified custom gesture profile."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Gesture name cannot be empty.")
        if clean_name in RESERVED_NAMES:
            raise ValueError(f"'{clean_name}' is a reserved built-in gesture name.")
        if clean_name in self._profiles and not allow_overwrite:
            raise ValueError(f"Custom gesture '{clean_name}' already exists. Use a different name or delete the existing profile first.")

        validated_mappings: Dict[str, str] = {}
        if mappings:
            for ctx_str, intent_str in mappings.items():
                # Validate context & intent
                ctx = Context.from_value(ctx_str)
                resolved_intent = Intent(intent_str)
                validated_mappings[ctx.value] = resolved_intent.value

        profile = GestureProfile(
            name=clean_name,
            centroid=[float(v) for v in centroid],
            dispersion=float(dispersion),
            sample_count=int(sample_count),
            mappings=validated_mappings,
            description=description,
        )
        self._profiles[clean_name] = profile
        self.save_profiles()
        return profile

    def enroll_gesture(
        self,
        name: str,
        take1: Sequence[Sequence[float]],
        take2: Sequence[Sequence[float]],
        take3_val: Sequence[Sequence[float]],
        mappings: Optional[Dict[str, str]] = None,
        description: str = "",
        builtin_classifier: Optional[Any] = None,
        allow_overwrite: bool = False,
    ) -> GestureProfile:
        """Runs 3-take validation and registers the new custom gesture profile."""
        clean_name = name.strip()
        if clean_name in RESERVED_NAMES:
            raise ValueError(f"Cannot enroll '{clean_name}': reserved built-in gesture name.")
        if clean_name in self._profiles and not allow_overwrite:
            raise ValueError(f"Custom gesture '{clean_name}' already exists. Use a different name or delete the existing profile first.")

        existing_centroids = {
            g_name: g.centroid for g_name, g in self._profiles.items() if g_name != clean_name
        }
        centroid, dispersion = validate_enrollment_samples(
            take1=take1,
            take2=take2,
            take3_val=take3_val,
            builtin_classifier=builtin_classifier,
            existing_centroids=existing_centroids,
        )
        total_samples = len(take1) + len(take2) + len(take3_val)
        return self.register_gesture(
            name=clean_name,
            centroid=centroid,
            dispersion=dispersion,
            sample_count=total_samples,
            mappings=mappings,
            description=description,
            allow_overwrite=allow_overwrite,
        )

    def rename_gesture(self, old_name: str, new_name: str) -> bool:
        """Renames an enrolled custom gesture."""
        if old_name not in self._profiles:
            return False
        clean_new = new_name.strip()
        if not clean_new:
            raise ValueError("New name cannot be empty.")
        if clean_new in RESERVED_NAMES:
            raise ValueError(f"'{clean_new}' is a reserved built-in gesture name.")
        if clean_new in self._profiles and clean_new != old_name:
            raise ValueError(f"Custom gesture '{clean_new}' already exists.")

        profile = self._profiles.pop(old_name)
        profile.name = clean_new
        self._profiles[clean_new] = profile
        self.save_profiles()
        return True

    def remap_gesture(
        self,
        name: str,
        context: Union[Context, str],
        intent: Union[Intent, str],
    ) -> bool:
        """Updates context mapping for an enrolled custom gesture."""
        if name not in self._profiles:
            return False
        ctx = Context.from_value(context)
        resolved_intent = Intent(intent)
        self._profiles[name].mappings[ctx.value] = resolved_intent.value
        self.save_profiles()
        return True

    def delete_gesture(self, name: str) -> bool:
        """Deletes an enrolled custom gesture profile."""
        if name not in self._profiles:
            return False
        del self._profiles[name]
        self.save_profiles()
        return True

    def get_gesture(self, name: str) -> Optional[GestureProfile]:
        return self._profiles.get(name)

    def list_gestures(self) -> List[GestureProfile]:
        return list(self._profiles.values())

    def match(self, features: Sequence[float]) -> CustomMatchResult:
        """
        Matches input 42D feature vector against enrolled custom gestures.

        Enforces:
        - Distance cutoff TAU_ACCEPT (0.30).
        - Ambiguity margin DELTA_AMBIGUITY (0.10) between top two candidates.
        """
        if not self._profiles or len(features) != 42:
            return CustomMatchResult(
                gesture=None, confidence=0.0, distance=999.0, is_match=False, reason="NO_PROFILES"
            )

        feat_arr = np.asarray(features, dtype=float)
        candidates: List[Tuple[float, str]] = []

        for name, profile in self._profiles.items():
            cent_arr = np.asarray(profile.centroid, dtype=float)
            dist = float(np.linalg.norm(feat_arr - cent_arr))
            candidates.append((dist, name))

        candidates.sort(key=lambda item: item[0])
        best_dist, best_name = candidates[0]

        # Check ambiguity if more than one profile exists
        if len(candidates) > 1:
            second_dist, _ = candidates[1]
            if best_dist <= TAU_ACCEPT and second_dist <= TAU_ACCEPT:
                delta = second_dist - best_dist
                if delta < DELTA_AMBIGUITY:
                    return CustomMatchResult(
                        gesture=None,
                        confidence=0.0,
                        distance=best_dist,
                        is_match=False,
                        reason="AMBIGUOUS_CUSTOM_MATCH",
                    )

        if best_dist <= TAU_ACCEPT:
            conf = calculate_custom_confidence(best_dist)
            return CustomMatchResult(
                gesture=best_name,
                confidence=conf,
                distance=best_dist,
                is_match=True,
                reason="CUSTOM_MATCH",
            )
        else:
            return CustomMatchResult(
                gesture=None,
                confidence=0.0,
                distance=best_dist,
                is_match=False,
                reason="UNKNOWN_POSE",
            )

    def arbitrate(
        self,
        builtin_gesture: Optional[str],
        builtin_confidence: float,
        builtin_threshold: float,
        features: Optional[Sequence[float]],
        hand_present: bool = True,
    ) -> ArbitrationResult:
        """
        Executes the 4-Tier Built-In vs Custom arbitration hierarchy.

        Tiers:
        1: Decisive built-in recognition (builtin_confidence >= builtin_threshold) -> BUILTIN.
        2: Built-in transition/uncertainty guard (BUILTIN_GUARD_THRESHOLD <= builtin_confidence < builtin_threshold)
           -> Custom gestures STRICTLY INHIBITED.
        3: Built-in model low confidence (< 0.50) -> Evaluates custom matcher.
        4: Hand missing or feature extraction failed -> NONE.
        """
        # Tier 4: No hand detected
        if not hand_present or features is None:
            return ArbitrationResult(
                candidate_gesture=None,
                confidence=0.0,
                source="NONE",
                tier=4,
                reason="NO_HAND",
            )

        # Tier 1: Decisive built-in match
        if builtin_confidence >= builtin_threshold and builtin_gesture is not None:
            return ArbitrationResult(
                candidate_gesture=builtin_gesture,
                confidence=builtin_confidence,
                source="BUILTIN",
                tier=1,
                reason="BUILTIN_CONFIRMED",
            )

        # Tier 2: Built-in uncertainty zone [0.50, builtin_threshold)
        if builtin_confidence >= BUILTIN_GUARD_THRESHOLD:
            return ArbitrationResult(
                candidate_gesture=None,
                confidence=0.0,
                source="BUILTIN",
                tier=2,
                reason="BUILTIN_UNCERTAIN_GUARD",
            )

        # Tier 3: Low built-in confidence (< 0.50) -> Evaluate custom matcher
        match_res = self.match(features)

        if match_res.is_match:
            # Check if match meets entry threshold (C >= 0.80, approx d <= 0.233)
            if match_res.confidence >= TEMPORAL_ENTRY_CONFIDENCE:
                return ArbitrationResult(
                    candidate_gesture=match_res.gesture,
                    confidence=match_res.confidence,
                    source="CUSTOM",
                    tier=3,
                    reason="CUSTOM_CANDIDATE",
                )
            else:
                # Weak match (d in (0.233, 0.30] -> C in [0.70, 0.80))
                return ArbitrationResult(
                    candidate_gesture=None,
                    confidence=match_res.confidence,
                    source="CUSTOM",
                    tier=3,
                    reason="CUSTOM_WEAK_MATCH",
                )
        elif match_res.reason == "AMBIGUOUS_CUSTOM_MATCH":
            return ArbitrationResult(
                candidate_gesture=None,
                confidence=0.0,
                source="CUSTOM",
                tier=3,
                reason="AMBIGUOUS_CUSTOM_MATCH",
            )
        else:
            return ArbitrationResult(
                candidate_gesture=None,
                confidence=0.0,
                source="NONE",
                tier=3,
                reason="UNKNOWN_POSE",
            )
