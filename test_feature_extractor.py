"""
test_feature_extractor.py
=========================
Unit and integration test script for core/feature_extractor.py.
Verifies translation invariance, scale invariance, and output formatting.
"""

import math
from core.feature_extractor import HandFeatureExtractor

def test_translation_invariance():
    extractor = HandFeatureExtractor()
    
    # Base dummy hand landmarks (21 points)
    base_landmarks = [(float(i * 10), float(i * 5)) for i in range(21)]
    
    # Shifted hand landmarks (translated by +100, +200)
    shifted_landmarks = [(x + 100.0, y + 200.0) for x, y in base_landmarks]
    
    feat_base = extractor.extract_features(base_landmarks)
    feat_shifted = extractor.extract_features(shifted_landmarks)
    
    # Features must be identical within floating point precision
    diff = max(abs(a - b) for a, b in zip(feat_base, feat_shifted))
    assert diff < 1e-5, f"Translation invariance failed, max diff: {diff}"
    print(f"✓ Translation invariance test passed (max diff: {diff:.8f})")

def test_scale_invariance():
    extractor = HandFeatureExtractor()
    
    # Base dummy hand landmarks (21 points)
    base_landmarks = [(float(i * 10), float(i * 5)) for i in range(21)]
    
    # Scaled hand landmarks (scaled by factor of 2.5)
    scaled_landmarks = [(x * 2.5, y * 2.5) for x, y in base_landmarks]
    
    feat_base = extractor.extract_features(base_landmarks)
    feat_scaled = extractor.extract_features(scaled_landmarks)
    
    # Features must be identical within floating point precision
    diff = max(abs(a - b) for a, b in zip(feat_base, feat_scaled))
    assert diff < 1e-5, f"Scale invariance failed, max diff: {diff}"
    print(f"✓ Scale invariance test passed (max diff: {diff:.8f})")

def test_output_properties():
    extractor = HandFeatureExtractor()
    dummy_landmarks = [(float(i), float(i*2)) for i in range(21)]
    
    features = extractor.extract_features(dummy_landmarks)
    names = extractor.get_feature_names()
    
    assert len(features) == 42, f"Expected 42 features, got {len(features)}"
    assert len(names) == 42, f"Expected 42 feature names, got {len(names)}"
    assert features[0] == 0.0 and features[1] == 0.0, "Wrist landmark (index 0) must be (0.0, 0.0)"
    
    print("✓ Feature shape and wrist origin tests passed")

if __name__ == "__main__":
    print("Running HandFeatureExtractor tests...")
    test_translation_invariance()
    test_scale_invariance()
    test_output_properties()
    print("All Phase 1 feature extraction unit tests PASSED successfully!")
