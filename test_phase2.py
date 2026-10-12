"""
test_phase2.py
==============
Synthetic verification test for Phase 2 data pipeline and model training.
Simulates 6 gesture classes with position/scale variations, transforms them via
HandFeatureExtractor, populates dataset/gestures.csv, and runs train_model.py.
"""

import os
import math
import random
import pandas as pd
from core.feature_extractor import HandFeatureExtractor
import train_model

def generate_synthetic_gesture(gesture_name: str, offset_x: float, offset_y: float, scale: float):
    """Generate synthetic 21 hand landmarks for a given gesture pose with random spatial offsets and scales."""
    landmarks = []
    
    # Base wrist at (0, 0) relative to hand
    wrist = (0.0, 0.0)
    
    # Define finger extension vectors relative to wrist for 6 gestures
    if gesture_name == "OpenPalm":
        # All 5 fingers extended outward
        angles = [-0.4, -0.2, 0.0, 0.2, 0.4]
        lengths = [0.8, 1.0, 1.1, 1.0, 0.8]
    elif gesture_name == "Fist":
        # All fingers curled near palm
        angles = [-0.3, -0.15, 0.0, 0.15, 0.3]
        lengths = [0.3, 0.3, 0.3, 0.3, 0.3]
    elif gesture_name == "Peace":
        # Index & Middle extended, others curled
        angles = [-0.1, 0.1, 0.0, 0.2, 0.4]
        lengths = [1.0, 1.05, 0.3, 0.3, 0.3]
    elif gesture_name == "ThumbsUp":
        # Thumb extended up (-y), others curled
        angles = [-1.57, -0.2, 0.0, 0.2, 0.4]
        lengths = [1.0, 0.3, 0.3, 0.3, 0.3]
    elif gesture_name == "ThumbsDown":
        # Thumb extended down (+y), others curled
        angles = [1.57, -0.2, 0.0, 0.2, 0.4]
        lengths = [1.0, 0.3, 0.3, 0.3, 0.3]
    elif gesture_name == "Pointing":
        # Index finger extended, others curled
        angles = [-0.2, 0.0, 0.1, 0.2, 0.3]
        lengths = [0.3, 1.0, 0.3, 0.3, 0.3]
    else:
        angles = [0.0] * 5
        lengths = [0.5] * 5

    # Build 21 keypoints (Wrist + 4 points per finger)
    landmarks.append(wrist)
    
    for finger_idx in range(5):
        ang = angles[finger_idx] + random.uniform(-0.02, 0.02)
        l_total = lengths[finger_idx] * (1.0 + random.uniform(-0.03, 0.03))
        
        # 4 segments per finger (MCP, PIP, DIP, TIP)
        for seg in range(1, 5):
            r = (seg / 4.0) * l_total
            x = r * math.sin(ang)
            y = -r * math.cos(ang)
            landmarks.append((x, y))

    # Apply spatial translation and scaling
    transformed_landmarks = []
    for x, y in landmarks:
        tx = (x * scale) + offset_x
        ty = (y * scale) + offset_y
        transformed_landmarks.append((tx, ty))
        
    return transformed_landmarks

def run_test():
    print("Generating normalized synthetic dataset for Phase 2 verification...")
    extractor = HandFeatureExtractor()
    gestures = ["OpenPalm", "Fist", "Peace", "ThumbsUp", "ThumbsDown", "Pointing"]
    
    rows = []
    headers = extractor.get_feature_names() + ["label"]
    
    for gesture in gestures:
        for _ in range(35): # 35 samples per class
            # Random position anywhere in camera frame (e.g. 100..1000)
            off_x = random.uniform(100.0, 1000.0)
            off_y = random.uniform(50.0, 600.0)
            # Random hand scale (distance to camera)
            scale = random.uniform(50.0, 250.0)
            
            raw_lm = generate_synthetic_gesture(gesture, off_x, off_y, scale)
            norm_feats = extractor.extract_features(raw_lm)
            rows.append(norm_feats + [gesture])
            
    os.makedirs("dataset", exist_ok=True)
    df = pd.DataFrame(rows, columns=headers)
    csv_path = os.path.join("dataset", "gestures.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Synthetic dataset with {len(df)} samples created at {csv_path}")
    
    print("\nExecuting train_model.py...")
    train_model.train()

if __name__ == "__main__":
    run_test()
