"""
test_swipe_detector.py
======================

Comprehensive test suite for Phase 8A: Standalone Dynamic Swipe Gesture Detector.

Covers:
1. Clean Swipes (Right and Left direction recognition and single-event emission)
2. Different Gesture Durations (Fast ballistic, Moderate, Slow, Out-of-bounds)
3. Palm Size Scaling Invariance (Small, Normal, Large hands)
4. Small Movement Rejection (Below displacement threshold)
5. Slow Movement Rejection (Below velocity threshold)
6. Instantaneous Teleportation / Spatial Jump Rejection
7. Pure Vertical Motion Rejection (Up, Down)
8. Diagonal Motion Analysis (45-degree rejection vs shallow angle acceptance)
9. Jitter and Non-linear Motion Rejection (Low path efficiency)
10. Tracking Loss Handling (hand_present=False and dropped frame time gaps)
11. Edge Entry Boundary Rejection
12. Return-Stroke Suppression and Dual Cooldown Lifecycle
13. Detector Reset and Configuration Validation
"""

import math
import os
import sys
import unittest
from typing import List, Optional, Tuple

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.swipe_detector import (
    SwipeConfig,
    SwipeDetector,
    SwipeDirection,
    SwipeResult,
    TrajectoryPoint,
)


class TestSwipeDetector(unittest.TestCase):
    """Unit and kinematic test suite for core/swipe_detector.py."""

    def setUp(self) -> None:
        self.config = SwipeConfig()
        self.detector = SwipeDetector(self.config)

    def _feed_trajectory(
        self,
        points: List[Tuple[float, float, float]],
        palm_size: Optional[float] = None,
        hand_present: bool = True,
        detector: Optional[SwipeDetector] = None,
    ) -> List[SwipeResult]:
        """Helper to feed a sequential list of (x, y, timestamp) points into detector."""
        det = detector or self.detector
        results = []
        for pt in points:
            x, y, t = pt
            res = det.update(
                x=x,
                y=y,
                timestamp=t,
                palm_size=palm_size,
                hand_present=hand_present,
            )
            results.append(res)
        return results

    def _generate_linear_path(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        start_t: float,
        duration: float,
        num_frames: int = 10,
    ) -> List[Tuple[float, float, float]]:
        """Generates linearly interpolated trajectory frames."""
        points = []
        for i in range(num_frames):
            alpha = i / (num_frames - 1)
            x = start_x + alpha * (end_x - start_x)
            y = start_y + alpha * (end_y - start_y)
            t = start_t + alpha * duration
            points.append((round(x, 4), round(y, 4), round(t, 4)))
        return points

    # -------------------------------------------------------------------------
    # 1. Clean Swipes (Right and Left)
    # -------------------------------------------------------------------------
    def test_clean_swipe_right(self) -> None:
        """Verify clean left-to-right swipe produces SwipeDirection.RIGHT exactly once."""
        traj = self._generate_linear_path(
            start_x=0.30,
            start_y=0.50,
            end_x=0.70,
            end_y=0.50,
            start_t=1.0,
            duration=0.25,
            num_frames=10,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction == SwipeDirection.RIGHT]

        self.assertEqual(len(detected), 1)
        res = detected[0]
        self.assertEqual(res.direction, SwipeDirection.RIGHT)
        self.assertGreaterEqual(res.confidence, 0.70)
        self.assertGreaterEqual(res.displacement, 0.15)
        self.assertGreater(res.velocity, 1.0)
        self.assertGreaterEqual(res.path_efficiency, 0.95)
        self.assertEqual(res.reason, "SWIPE_DETECTED")

    def test_clean_swipe_left(self) -> None:
        """Verify clean right-to-left swipe produces SwipeDirection.LEFT exactly once."""
        traj = self._generate_linear_path(
            start_x=0.70,
            start_y=0.50,
            end_x=0.30,
            end_y=0.50,
            start_t=1.0,
            duration=0.25,
            num_frames=10,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction == SwipeDirection.LEFT]

        self.assertEqual(len(detected), 1)
        res = detected[0]
        self.assertEqual(res.direction, SwipeDirection.LEFT)
        self.assertGreaterEqual(res.confidence, 0.70)
        self.assertLessEqual(res.displacement, -0.15)
        self.assertGreater(res.velocity, 1.0)
        self.assertGreaterEqual(res.path_efficiency, 0.95)
        self.assertEqual(res.reason, "SWIPE_DETECTED")

    # -------------------------------------------------------------------------
    # 2. Gesture Durations
    # -------------------------------------------------------------------------
    def test_fast_ballistic_swipe(self) -> None:
        """Verify fast ballistic swipe (100ms duration) is successfully detected."""
        traj = self._generate_linear_path(
            start_x=0.35,
            start_y=0.50,
            end_x=0.65,
            end_y=0.50,
            start_t=2.0,
            duration=0.10,
            num_frames=5,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction == SwipeDirection.RIGHT]

        self.assertEqual(len(detected), 1)
        self.assertEqual(detected[0].direction, SwipeDirection.RIGHT)

    def test_moderate_duration_swipe(self) -> None:
        """Verify moderate duration swipe (250ms) is successfully detected."""
        traj = self._generate_linear_path(
            start_x=0.30,
            start_y=0.50,
            end_x=0.60,
            end_y=0.50,
            start_t=3.0,
            duration=0.25,
            num_frames=9,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction == SwipeDirection.RIGHT]
        self.assertEqual(len(detected), 1)

    def test_slow_duration_swipe(self) -> None:
        """Verify slower swipe (450ms) within max_duration is detected."""
        traj = self._generate_linear_path(
            start_x=0.25,
            start_y=0.50,
            end_x=0.65,
            end_y=0.50,
            start_t=4.0,
            duration=0.45,
            num_frames=15,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction == SwipeDirection.RIGHT]
        self.assertEqual(len(detected), 1)

    def test_too_fast_swipe_rejected(self) -> None:
        """Verify motion completing in < min_duration (e.g. 40ms) is rejected."""
        traj = self._generate_linear_path(
            start_x=0.30,
            start_y=0.50,
            end_x=0.60,
            end_y=0.50,
            start_t=5.0,
            duration=0.04,
            num_frames=3,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction != SwipeDirection.NONE]
        self.assertEqual(len(detected), 0)

    def test_too_slow_drift_rejected(self) -> None:
        """Verify slow drift (velocity < min_velocity, e.g. 0.15 units/s) never triggers a swipe."""
        # Moving 0.30 across 2.0 seconds => vx = 0.15 units/s < 0.35
        traj = self._generate_linear_path(
            start_x=0.30,
            start_y=0.50,
            end_x=0.60,
            end_y=0.50,
            start_t=6.0,
            duration=2.00,
            num_frames=30,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction != SwipeDirection.NONE]
        self.assertEqual(len(detected), 0)

    # -------------------------------------------------------------------------
    # 3. Palm Size Scaling Invariance
    # -------------------------------------------------------------------------
    def test_small_palm_adaptive_displacement(self) -> None:
        """Verify smaller palm (further from camera) with proportional movement is detected."""
        # Palm size is 0.08 (small hand/far distance). Displacement of 0.12 exceeds 0.08 * 0.8 = 0.064
        traj = self._generate_linear_path(
            start_x=0.40,
            start_y=0.50,
            end_x=0.52,
            end_y=0.50,
            start_t=7.0,
            duration=0.20,
            num_frames=7,
        )
        results = self._feed_trajectory(traj, palm_size=0.08)
        detected = [r for r in results if r.direction == SwipeDirection.RIGHT]
        self.assertEqual(len(detected), 1)

    def test_large_palm_requires_proportional_movement(self) -> None:
        """Verify large palm (close to camera) rejects minor micro-movement but accepts large swipe."""
        # Palm size is 0.25 (large hand close to lens). Movement of 0.10 is small relative to hand (< 0.25*0.8=0.20)
        traj = self._generate_linear_path(
            start_x=0.40,
            start_y=0.50,
            end_x=0.50,
            end_y=0.50,
            start_t=8.0,
            duration=0.20,
            num_frames=7,
        )
        results = self._feed_trajectory(traj, palm_size=0.25)
        detected = [r for r in results if r.direction != SwipeDirection.NONE]
        self.assertEqual(len(detected), 0)

        # Full proportional swipe with large palm (0.28 displacement >= 0.20) is detected
        self.detector.reset()
        traj_full = self._generate_linear_path(
            start_x=0.30,
            start_y=0.50,
            end_x=0.58,
            end_y=0.50,
            start_t=9.0,
            duration=0.25,
            num_frames=9,
        )
        results_full = self._feed_trajectory(traj_full, palm_size=0.25)
        detected_full = [r for r in results_full if r.direction == SwipeDirection.RIGHT]
        self.assertEqual(len(detected_full), 1)

    # -------------------------------------------------------------------------
    # 4. Small Movements
    # -------------------------------------------------------------------------
    def test_small_movement_rejected(self) -> None:
        """Verify movement below default min_displacement (0.15) is rejected."""
        traj = self._generate_linear_path(
            start_x=0.45,
            start_y=0.50,
            end_x=0.52,  # dx = 0.07 < 0.15
            end_y=0.50,
            start_t=10.0,
            duration=0.15,
            num_frames=6,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction != SwipeDirection.NONE]
        self.assertEqual(len(detected), 0)

    # -------------------------------------------------------------------------
    # 5. Slow Movements (Velocity Gate)
    # -------------------------------------------------------------------------
    def test_low_velocity_rejected(self) -> None:
        """Verify movement with velocity < min_velocity (0.35) is rejected."""
        # dx = 0.16 over 0.55s => vx = 0.29 units/s (< 0.35)
        traj = self._generate_linear_path(
            start_x=0.30,
            start_y=0.50,
            end_x=0.46,
            end_y=0.50,
            start_t=11.0,
            duration=0.55,
            num_frames=18,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction != SwipeDirection.NONE]
        self.assertEqual(len(detected), 0)

    # -------------------------------------------------------------------------
    # 6. Instantaneous Teleportation / Spatial Jump
    # -------------------------------------------------------------------------
    def test_spatial_jump_teleport_resets_tracking(self) -> None:
        """Verify a single-frame teleportation jump (> max_frame_displacement) is rejected."""
        points = [
            (0.20, 0.50, 12.00),
            (0.75, 0.50, 12.03),  # Instant jump of 0.55 > max_frame_displacement 0.40
            (0.76, 0.50, 12.06),
        ]
        results = self._feed_trajectory(points)
        detected = [r for r in results if r.direction != SwipeDirection.NONE]
        self.assertEqual(len(detected), 0)

    # -------------------------------------------------------------------------
    # 7. Pure Vertical Motion
    # -------------------------------------------------------------------------
    def test_pure_vertical_up_rejected(self) -> None:
        """Verify vertical up motion is rejected."""
        traj = self._generate_linear_path(
            start_x=0.50,
            start_y=0.80,
            end_x=0.50,
            end_y=0.20,
            start_t=13.0,
            duration=0.25,
            num_frames=10,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction != SwipeDirection.NONE]
        self.assertEqual(len(detected), 0)

    def test_pure_vertical_down_rejected(self) -> None:
        """Verify vertical down motion is rejected."""
        traj = self._generate_linear_path(
            start_x=0.50,
            start_y=0.20,
            end_x=0.50,
            end_y=0.80,
            start_t=14.0,
            duration=0.25,
            num_frames=10,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction != SwipeDirection.NONE]
        self.assertEqual(len(detected), 0)

    # -------------------------------------------------------------------------
    # 8. Diagonal Motion
    # -------------------------------------------------------------------------
    def test_steep_diagonal_45_deg_rejected(self) -> None:
        """Verify 45-degree diagonal movement is rejected due to lack of horizontal dominance."""
        traj = self._generate_linear_path(
            start_x=0.30,
            start_y=0.30,
            end_x=0.65,
            end_y=0.65,
            start_t=15.0,
            duration=0.25,
            num_frames=10,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction != SwipeDirection.NONE]
        self.assertEqual(len(detected), 0)

    def test_shallow_diagonal_accepted(self) -> None:
        """Verify shallow diagonal movement (e.g. ~8 deg) is accepted as horizontal swipe."""
        traj = self._generate_linear_path(
            start_x=0.30,
            start_y=0.50,
            end_x=0.65,
            end_y=0.55,
            start_t=16.0,
            duration=0.25,
            num_frames=10,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction == SwipeDirection.RIGHT]
        self.assertEqual(len(detected), 1)

    # -------------------------------------------------------------------------
    # 9. Jitter and Non-linear Motion (Path Efficiency)
    # -------------------------------------------------------------------------
    def test_jittery_zigzag_path_rejected(self) -> None:
        """Verify high-frequency jitter / zigzag motion with low path efficiency is rejected."""
        points = [
            (0.30, 0.50, 17.00),
            (0.35, 0.25, 17.04),
            (0.40, 0.75, 17.08),
            (0.45, 0.25, 17.12),
            (0.50, 0.75, 17.16),
            (0.55, 0.30, 17.20),
            (0.60, 0.50, 17.24),
        ]
        results = self._feed_trajectory(points)
        detected = [r for r in results if r.direction != SwipeDirection.NONE]
        self.assertEqual(len(detected), 0)

    def test_circular_waving_motion_rejected(self) -> None:
        """Verify circular looping gesture is rejected due to path efficiency."""
        points = []
        center_x, center_y = 0.5, 0.5
        radius = 0.15
        start_t = 18.0
        for i in range(12):
            angle = i * (2 * math.pi / 11)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            t = start_t + i * 0.025
            points.append((x, y, t))

        results = self._feed_trajectory(points)
        detected = [r for r in results if r.direction != SwipeDirection.NONE]
        self.assertEqual(len(detected), 0)

    # -------------------------------------------------------------------------
    # 10. Tracking Loss Handling
    # -------------------------------------------------------------------------
    def test_tracking_loss_clears_history(self) -> None:
        """Verify hand_present=False mid-gesture resets buffer and prevents false triggers."""
        # Partial swipe
        partial_traj = self._generate_linear_path(
            start_x=0.30,
            start_y=0.50,
            end_x=0.45,
            end_y=0.50,
            start_t=19.0,
            duration=0.10,
            num_frames=4,
        )
        self._feed_trajectory(partial_traj)

        # Tracking lost for 1 frame
        loss_res = self.detector.update(0.45, 0.50, timestamp=19.12, hand_present=False)
        self.assertEqual(loss_res.direction, SwipeDirection.NONE)
        self.assertEqual(loss_res.reason, "TRACKING_LOST")

        # Tracking resumes with a small movement (dx = 0.10 < 0.15 from resume point)
        resume_traj = self._generate_linear_path(
            start_x=0.45,
            start_y=0.50,
            end_x=0.55,
            end_y=0.50,
            start_t=19.15,
            duration=0.08,
            num_frames=3,
        )
        results = self._feed_trajectory(resume_traj)
        detected = [r for r in results if r.direction != SwipeDirection.NONE]
        self.assertEqual(len(detected), 0)

    def test_temporal_tracking_gap_clears_history(self) -> None:
        """Verify frame timestamp gap > max_tracking_gap (e.g. 0.35s) resets buffer."""
        self.detector.update(0.30, 0.50, timestamp=20.00)
        self.detector.update(0.65, 0.50, timestamp=20.35)  # gap 350ms > 200ms
        res = self.detector.update(0.66, 0.50, timestamp=20.38)
        self.assertEqual(res.direction, SwipeDirection.NONE)

    # -------------------------------------------------------------------------
    # 11. Edge Entry Boundary Rejection
    # -------------------------------------------------------------------------
    def test_edge_entry_from_left_border_rejected(self) -> None:
        """Verify hand entering from left camera border (x <= 0.06) is rejected."""
        traj = self._generate_linear_path(
            start_x=0.02,
            start_y=0.50,
            end_x=0.35,
            end_y=0.50,
            start_t=21.0,
            duration=0.20,
            num_frames=8,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction != SwipeDirection.NONE]
        self.assertEqual(len(detected), 0)

    def test_edge_entry_from_right_border_rejected(self) -> None:
        """Verify hand entering from right camera border (x >= 0.94) is rejected."""
        traj = self._generate_linear_path(
            start_x=0.98,
            start_y=0.50,
            end_x=0.65,
            end_y=0.50,
            start_t=22.0,
            duration=0.20,
            num_frames=8,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction != SwipeDirection.NONE]
        self.assertEqual(len(detected), 0)

    def test_edge_entry_disabled_permits_swipe(self) -> None:
        """Verify when reject_edge_entry=False, edge gestures are recognized."""
        custom_detector = SwipeDetector(SwipeConfig(reject_edge_entry=False))
        traj = self._generate_linear_path(
            start_x=0.02,
            start_y=0.50,
            end_x=0.40,
            end_y=0.50,
            start_t=23.0,
            duration=0.20,
            num_frames=8,
        )
        results = self._feed_trajectory(traj, detector=custom_detector)
        detected = [r for r in results if r.direction == SwipeDirection.RIGHT]
        self.assertEqual(len(detected), 1)

    def test_swipe_originating_inside_and_exiting_edge_accepted(self) -> None:
        """Verify swipe starting comfortably inside and exiting to edge is accepted."""
        traj = self._generate_linear_path(
            start_x=0.40,
            start_y=0.50,
            end_x=0.96,
            end_y=0.50,
            start_t=24.0,
            duration=0.25,
            num_frames=9,
        )
        results = self._feed_trajectory(traj)
        detected = [r for r in results if r.direction == SwipeDirection.RIGHT]
        self.assertEqual(len(detected), 1)

    # -------------------------------------------------------------------------
    # 12. Return-Stroke Suppression and Dual Cooldown
    # -------------------------------------------------------------------------
    def test_return_stroke_suppressed_after_swipe_right(self) -> None:
        """
        Verify that after detecting Swipe Right, the biomechanical return stroke
        moving back to the left is cleanly suppressed by return_stroke_cooldown.
        """
        # Step 1: Valid Swipe Right from 0.30 to 0.70 ending at t=25.25
        traj_right = self._generate_linear_path(
            start_x=0.30,
            start_y=0.50,
            end_x=0.70,
            end_y=0.50,
            start_t=25.0,
            duration=0.25,
            num_frames=9,
        )
        res_right = self._feed_trajectory(traj_right)
        detected_right = [r for r in res_right if r.direction == SwipeDirection.RIGHT]
        self.assertEqual(len(detected_right), 1)

        # Step 2: Return stroke moving left from 0.70 to 0.35 during t=25.45 -> t=25.70
        # (Within return_stroke_cooldown of 0.60s, i.e. before t=25.85)
        traj_return = self._generate_linear_path(
            start_x=0.70,
            start_y=0.50,
            end_x=0.35,
            end_y=0.50,
            start_t=25.45,
            duration=0.25,
            num_frames=9,
        )
        res_return = self._feed_trajectory(traj_return)

        # Ensure return stroke NEVER emitted SwipeDirection.LEFT
        for r in res_return:
            self.assertNotEqual(r.direction, SwipeDirection.LEFT)

        # Step 3: Deliberate Swipe Left after full return-stroke cooldown expires (t >= 26.0)
        traj_new_left = self._generate_linear_path(
            start_x=0.70,
            start_y=0.50,
            end_x=0.30,
            end_y=0.50,
            start_t=26.0,
            duration=0.25,
            num_frames=9,
        )
        res_new_left = self._feed_trajectory(traj_new_left)
        detected_left = [r for r in res_new_left if r.direction == SwipeDirection.LEFT]
        self.assertEqual(len(detected_left), 1)

    def test_general_cooldown_blocks_immediate_consecutive_swipe(self) -> None:
        """Verify that general cooldown blocks any swipe within cooldown_duration (0.40s)."""
        # First swipe right at t=27.0 -> 27.25
        traj1 = self._generate_linear_path(0.30, 0.50, 0.65, 0.50, 27.0, 0.25, 9)
        res1 = self._feed_trajectory(traj1)
        detected1 = [r for r in res1 if r.direction == SwipeDirection.RIGHT]
        self.assertEqual(len(detected1), 1)

        # Immediate second swipe right at t=27.30 -> 27.50 (t < 27.25 + 0.40 = 27.65)
        traj2 = self._generate_linear_path(0.30, 0.50, 0.65, 0.50, 27.30, 0.20, 8)
        res2 = self._feed_trajectory(traj2)
        detected2 = [r for r in res2 if r.direction != SwipeDirection.NONE]
        self.assertEqual(len(detected2), 0)

    # -------------------------------------------------------------------------
    # 13. Reset and Configuration Validation
    # -------------------------------------------------------------------------
    def test_detector_reset(self) -> None:
        """Verify reset() restores pristine initial state."""
        traj = self._generate_linear_path(0.30, 0.50, 0.70, 0.50, 28.0, 0.25, 9)
        self._feed_trajectory(traj)

        self.detector.reset()
        self.assertFalse(self.detector.is_in_cooldown)
        self.assertEqual(len(self.detector._history), 0)
        self.assertEqual(self.detector._last_swipe_direction, SwipeDirection.NONE)

    def test_config_validation(self) -> None:
        """Verify invalid configuration values raise ValueError."""
        with self.assertRaises(ValueError):
            SwipeConfig(min_duration=0.5, max_duration=0.2)  # max < min

        with self.assertRaises(ValueError):
            SwipeConfig(min_displacement=-0.1)

        with self.assertRaises(ValueError):
            SwipeConfig(min_velocity=0.0)

        with self.assertRaises(ValueError):
            SwipeConfig(min_path_efficiency=1.5)

        with self.assertRaises(ValueError):
            SwipeConfig(cooldown_duration=0.8, return_stroke_cooldown=0.4)


if __name__ == "__main__":
    unittest.main()
