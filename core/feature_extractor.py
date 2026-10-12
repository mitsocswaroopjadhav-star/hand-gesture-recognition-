"""
core/feature_extractor.py
=========================

Translation- and Scale-Invariant Hand Landmark Feature Extractor.

This module converts raw 2D landmark coordinates (from MediaPipe Tasks or pixel space)
into a normalized, position- and scale-invariant feature vector suitable for machine learning.

Mathematical Formulation:
-------------------------
1. Translation Invariance (Wrist-relative):
   Given landmarks P_i = (x_i, y_i) for i in 0..20, where P_0 is the Wrist landmark:
   P'_i = (x_i - x_0, y_i - y_0)

2. Scale Invariance (Palm-length normalization):
   We use the Euclidean distance between Wrist (Landmark 0) and Middle MCP joint (Landmark 9)
   as the reference palm span (D_ref):
   D_ref = sqrt((x_9 - x_0)^2 + (y_9 - y_0)^2)

   If D_ref < epsilon (hand collapses or detection artifact), D_ref defaults to 1.0 to prevent division by zero.

   Normalized coordinates:
   x_norm_i = (x_i - x_0) / D_ref
   y_norm_i = (y_i - y_0) / D_ref

3. Output Vector:
   Flattened 42-element float list:
   [x_norm_0, y_norm_0, x_norm_1, y_norm_1, ..., x_norm_20, y_norm_20]
   (Note: x_norm_0 and y_norm_0 are always 0.0, serving as the relative origin).
"""

from __future__ import annotations
import math
from typing import List, Sequence, Tuple, Union, Dict, Any


class HandFeatureExtractor:
    """
    Extracts normalized, translation- and scale-invariant features from 21 hand landmarks.

    Designed with an extensible interface for future feature additions:
    - Joint angles
    - Inter-landmark distances
    - Trajectory/motion dynamics
    """

    def __init__(self, use_angles: bool = False, epsilon: float = 1e-6):
        """
        Args:
            use_angles: Optional flag reserved for future angle feature inclusion.
            epsilon: Safeguard threshold to avoid division by zero.
        """
        self.use_angles = use_angles
        self.epsilon = epsilon

    def normalize_landmarks(
        self, landmarks: Sequence[Tuple[float, float]]
    ) -> List[float]:
        """
        Normalizes 21 (x, y) hand landmarks relative to Wrist (index 0) and Middle MCP (index 9).

        Args:
            landmarks: Sequence of 21 (x, y) numerical tuples.

        Returns:
            List[float]: Flattened 42-element list of normalized coordinates
                         [x0, y0, x1, y1, ..., x20, y20].

        Raises:
            ValueError: If input does not contain exactly 21 landmarks.
        """
        if len(landmarks) != 21:
            raise ValueError(f"Expected 21 landmarks, received {len(landmarks)}")

        # Landmark 0 = Wrist (Origin)
        wrist_x, wrist_y = landmarks[0][0], landmarks[0][1]

        # Landmark 9 = Middle Finger MCP (Palm length reference)
        middle_mcp_x, middle_mcp_y = landmarks[9][0], landmarks[9][1]

        # Calculate reference distance D_ref = distance(Wrist, Middle_MCP)
        dx_ref = middle_mcp_x - wrist_x
        dy_ref = middle_mcp_y - wrist_y
        d_ref = math.sqrt(dx_ref * dx_ref + dy_ref * dy_ref)

        # Guard against zero or near-zero reference distance
        if d_ref < self.epsilon:
            d_ref = 1.0

        normalized_features: List[float] = []

        for x, y in landmarks:
            # 1. Translate relative to wrist
            rel_x = x - wrist_x
            rel_y = y - wrist_y

            # 2. Scale by reference palm length
            norm_x = rel_x / d_ref
            norm_y = rel_y / d_ref

            normalized_features.append(float(norm_x))
            normalized_features.append(float(norm_y))

        return normalized_features

    def get_feature_names(self) -> List[str]:
        """
        Returns feature column names matching the ordering of extract_features().
        Useful for DataFrame header generation and model validation.
        """
        names = []
        for i in range(21):
            names.append(f"norm_x{i}")
            names.append(f"norm_y{i}")
        return names

    def extract_features(
        self, landmarks: Sequence[Tuple[float, float]]
    ) -> List[float]:
        """
        Master feature extraction entry point.

        Args:
            landmarks: 21 (x, y) coordinates of detected hand.

        Returns:
            List[float]: Complete feature vector ready for model input.
        """
        # Base 42 normalized coordinates
        features = self.normalize_landmarks(landmarks)

        # Future extensions (e.g., angles, distances) can be appended here cleanly

        return features
