"""
test_activation_state.py
========================

Unit test suite for IntentSense Phase 4: Activation & Safety State Machine.

Covers:
1. Initial state is LOCKED
2. Confirmed OpenPalm activates the system (LOCKED -> ACTIVE)
3. Unconfirmed OpenPalm does not activate
4. Other confirmed gestures (e.g. Fist, ThumbsUp) do not activate the system
5. Normal confirmed gestures are ignored while LOCKED (command is None)
6. ACTIVE state allows confirmed gestures to pass through as commands
7. Inactivity timeout returns ACTIVE -> LOCKED
8. Manual reset / lock returns to LOCKED
9. Activation requires the configured hold duration (default 1.0s)
10. End-to-end frame stream processing with Phase 3.1 temporal stability filter
"""

import os
import sys
import unittest

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.activation_state import (
    ActivationResult,
    ActivationStateMachine,
    SystemState,
)
from core.adaptive_threshold import (
    AdaptiveThresholdConfig,
    AdaptiveThresholdPolicy,
)
from core.temporal_filter import (
    ConfirmationStatus,
    TemporalFilter,
    TemporalFilterResult,
)


class TestActivationStateMachine(unittest.TestCase):
    """Test suite for core/activation_state.py."""

    def setUp(self):
        """Set up standard ActivationStateMachine instance with defaults."""
        self.state_machine = ActivationStateMachine(
            activation_gesture="OpenPalm",
            activation_hold_duration=1.0,
            inactivity_timeout=10.0,
        )

    # -------------------------------------------------------------------------
    # Test 1: Initial state is LOCKED
    # -------------------------------------------------------------------------
    def test_initial_state_is_locked(self):
        """Verify that the system strictly initializes in LOCKED state."""
        self.assertEqual(self.state_machine.state, SystemState.LOCKED)
        self.assertFalse(self.state_machine.is_active)
        self.assertEqual(self.state_machine.time_remaining, 0.0)

    # -------------------------------------------------------------------------
    # Test 2: Confirmed OpenPalm activates the system
    # -------------------------------------------------------------------------
    def test_confirmed_open_palm_activates(self):
        """
        Verify that a confirmed OpenPalm held for >= activation_hold_duration (1.0s)
        transitions the system from LOCKED -> ACTIVE.
        """
        res = self.state_machine.process_confirmed(
            gesture="OpenPalm",
            is_confirmed=True,
            hold_duration=1.0,
            timestamp=0.0,
        )
        self.assertEqual(self.state_machine.state, SystemState.ACTIVE)
        self.assertTrue(self.state_machine.is_active)
        self.assertEqual(res.state, SystemState.ACTIVE)
        self.assertTrue(res.just_activated)
        self.assertIsNone(res.command)  # Activation gesture itself is not emitted as a command
        self.assertEqual(res.reason, "ACTIVATED")

    # -------------------------------------------------------------------------
    # Test 3: Unconfirmed OpenPalm does not activate
    # -------------------------------------------------------------------------
    def test_unconfirmed_open_palm_does_not_activate(self):
        """
        Verify that an unconfirmed OpenPalm (is_confirmed=False) does NOT activate.
        """
        res = self.state_machine.process_confirmed(
            gesture="OpenPalm",
            is_confirmed=False,
            hold_duration=0.5,
            timestamp=0.0,
        )
        self.assertEqual(self.state_machine.state, SystemState.LOCKED)
        self.assertFalse(self.state_machine.is_active)
        self.assertEqual(res.state, SystemState.LOCKED)
        self.assertFalse(res.just_activated)
        self.assertIsNone(res.command)

    # -------------------------------------------------------------------------
    # Test 4: Other confirmed gestures do not activate the system
    # -------------------------------------------------------------------------
    def test_other_confirmed_gestures_do_not_activate(self):
        """
        Verify that confirmed non-activation gestures (Fist, ThumbsUp, Peace, etc.)
        do NOT activate the system while LOCKED.
        """
        for g in ["Fist", "ThumbsUp", "ThumbsDown", "Peace", "Pointing"]:
            res = self.state_machine.process_confirmed(
                gesture=g,
                is_confirmed=True,
                hold_duration=2.0,
                timestamp=0.0,
            )
            self.assertEqual(self.state_machine.state, SystemState.LOCKED)
            self.assertFalse(self.state_machine.is_active)
            self.assertIsNone(res.command)

    # -------------------------------------------------------------------------
    # Test 5: Normal confirmed gestures are ignored while LOCKED
    # -------------------------------------------------------------------------
    def test_normal_confirmed_gestures_ignored_while_locked(self):
        """
        Verify that while LOCKED, normal confirmed gestures produce command=None.
        """
        res = self.state_machine.process_confirmed(
            gesture="ThumbsUp",
            is_confirmed=True,
            hold_duration=1.0,
            timestamp=0.0,
        )
        self.assertEqual(res.state, SystemState.LOCKED)
        self.assertIsNone(res.command)
        self.assertFalse(res.is_active)

    # -------------------------------------------------------------------------
    # Test 6: ACTIVE state allows confirmed gestures to pass through
    # -------------------------------------------------------------------------
    def test_active_state_allows_confirmed_gestures_as_commands(self):
        """
        Verify that once in ACTIVE state, confirmed gestures pass through
        as valid commands.
        """
        # 1. Activate
        self.state_machine.process_confirmed(
            gesture="OpenPalm",
            is_confirmed=True,
            hold_duration=1.0,
            timestamp=0.0,
        )
        self.assertEqual(self.state_machine.state, SystemState.ACTIVE)

        # 2. Execute command ThumbsUp
        res = self.state_machine.process_confirmed(
            gesture="ThumbsUp",
            is_confirmed=True,
            hold_duration=0.5,
            timestamp=1.0,
        )
        self.assertEqual(res.state, SystemState.ACTIVE)
        self.assertEqual(res.command, "ThumbsUp")
        self.assertEqual(res.reason, "COMMAND_ALLOWED")

        # 3. Unconfirmed gesture while active does not emit a command
        res_unconfirmed = self.state_machine.process_confirmed(
            gesture="Fist",
            is_confirmed=False,
            hold_duration=0.1,
            timestamp=1.5,
        )
        self.assertEqual(res_unconfirmed.state, SystemState.ACTIVE)
        self.assertIsNone(res_unconfirmed.command)

    # -------------------------------------------------------------------------
    # Test 7: Inactivity timeout returns ACTIVE -> LOCKED
    # -------------------------------------------------------------------------
    def test_inactivity_timeout_reverts_to_locked(self):
        """
        Verify that when inactivity_timeout (10.0s) elapses with no activity,
        the system transitions ACTIVE -> LOCKED.
        """
        # 1. Activate at t = 0.0s
        self.state_machine.process_confirmed(
            gesture="OpenPalm",
            is_confirmed=True,
            hold_duration=1.0,
            timestamp=0.0,
        )
        self.assertTrue(self.state_machine.is_active)

        # 2. Frame at t = 5.0s (within 10s window) -> still ACTIVE
        res = self.state_machine.process_confirmed(
            gesture=None,
            is_confirmed=False,
            timestamp=5.0,
        )
        self.assertEqual(res.state, SystemState.ACTIVE)
        self.assertTrue(self.state_machine.is_active)
        self.assertAlmostEqual(res.time_remaining, 5.0, places=1)

        # 3. Frame at t = 10.0s -> Inactivity timeout reached!
        res_timeout = self.state_machine.process_confirmed(
            gesture=None,
            is_confirmed=False,
            timestamp=10.0,
        )
        self.assertEqual(res_timeout.state, SystemState.LOCKED)
        self.assertFalse(self.state_machine.is_active)
        self.assertTrue(res_timeout.just_locked)
        self.assertEqual(res_timeout.reason, "INACTIVITY_TIMEOUT")

    def test_activity_refreshes_inactivity_timer(self):
        """
        Verify that confirmed commands while ACTIVE refresh the inactivity timer.
        """
        # Activate at t = 0.0s
        self.state_machine.process_confirmed("OpenPalm", is_confirmed=True, hold_duration=1.0, timestamp=0.0)

        # Command at t = 8.0s refreshes the timer
        res = self.state_machine.process_confirmed("ThumbsUp", is_confirmed=True, hold_duration=0.5, timestamp=8.0)
        self.assertEqual(res.command, "ThumbsUp")
        self.assertAlmostEqual(res.time_remaining, 10.0, places=1)

        # At t = 15.0s (7 seconds after last command) -> still ACTIVE!
        res_idle = self.state_machine.process_confirmed(None, is_confirmed=False, timestamp=15.0)
        self.assertEqual(res_idle.state, SystemState.ACTIVE)

        # At t = 18.0s (10 seconds after last command) -> TIMEOUT -> LOCKED
        res_lock = self.state_machine.process_confirmed(None, is_confirmed=False, timestamp=18.0)
        self.assertEqual(res_lock.state, SystemState.LOCKED)

    # -------------------------------------------------------------------------
    # Test 8: Manual reset returns to LOCKED
    # -------------------------------------------------------------------------
    def test_manual_lock_and_reset(self):
        """
        Verify that manual lock() and reset() immediately transition system to LOCKED.
        """
        # Activate
        self.state_machine.process_confirmed("OpenPalm", is_confirmed=True, hold_duration=1.0, timestamp=0.0)
        self.assertTrue(self.state_machine.is_active)

        # Manual lock
        self.state_machine.lock()
        self.assertEqual(self.state_machine.state, SystemState.LOCKED)
        self.assertFalse(self.state_machine.is_active)

        # Activate again
        self.state_machine.process_confirmed("OpenPalm", is_confirmed=True, hold_duration=1.0, timestamp=1.0)
        self.assertTrue(self.state_machine.is_active)

        # Manual reset
        self.state_machine.reset()
        self.assertEqual(self.state_machine.state, SystemState.LOCKED)
        self.assertFalse(self.state_machine.is_active)

    # -------------------------------------------------------------------------
    # Test 9: Activation requires the configured hold duration
    # -------------------------------------------------------------------------
    def test_activation_requires_configured_hold_duration(self):
        """
        Verify that OpenPalm held for less than activation_hold_duration (e.g. 0.8s < 1.0s)
        does NOT activate.
        """
        # Held for 0.8s < 1.0s
        res = self.state_machine.process_confirmed(
            gesture="OpenPalm",
            is_confirmed=True,
            hold_duration=0.8,
            timestamp=0.8,
        )
        self.assertEqual(self.state_machine.state, SystemState.LOCKED)
        self.assertFalse(self.state_machine.is_active)
        self.assertEqual(res.reason, "INSUFFICIENT_ACTIVATION_HOLD")
        self.assertAlmostEqual(res.activation_progress, 0.8, places=2)

        # Now held for 1.0s -> Activates!
        res_full = self.state_machine.process_confirmed(
            gesture="OpenPalm",
            is_confirmed=True,
            hold_duration=1.0,
            timestamp=1.0,
        )
        self.assertEqual(self.state_machine.state, SystemState.ACTIVE)
        self.assertTrue(self.state_machine.is_active)
        self.assertEqual(res_full.reason, "ACTIVATED")

    # -------------------------------------------------------------------------
    # Test 10: End-to-end Frame Stream Integration
    # -------------------------------------------------------------------------
    def test_end_to_end_frame_stream_activation_and_commands(self):
        """
        Verify continuous frame-by-frame processing through the activation state machine:
        1. Streaming Fist while LOCKED -> ignored
        2. Streaming OpenPalm for 1.0s -> system activates
        3. Streaming Fist for 0.5s while ACTIVE -> command emitted
        4. Inactivity -> system locks
        """
        sm = ActivationStateMachine(
            activation_gesture="OpenPalm",
            activation_hold_duration=1.0,
            inactivity_timeout=5.0,
            confidence_threshold=0.80,
            stable_frames=3,
        )

        # 1. Stream Fist while LOCKED at t=0.0..0.5s -> no command emitted
        for i in range(5):
            res = sm.process("Fist", 0.95, timestamp=i * 0.1)
            self.assertEqual(res.state, SystemState.LOCKED)
            self.assertIsNone(res.command)

        # 2. Stream OpenPalm at t=1.0..2.1s (1.1s hold >= 1.0s)
        # Frames 0 to 9 (t = 1.0 to 1.9s)
        for i in range(10):
            res = sm.process("OpenPalm", 0.95, timestamp=1.0 + (i * 0.1))
            self.assertEqual(res.state, SystemState.LOCKED)

        # Frame 10 at t = 2.05s reaches 1.05s hold duration -> ACTIVATED!
        res_act = sm.process("OpenPalm", 0.95, timestamp=2.05)
        self.assertEqual(res_act.state, SystemState.ACTIVE)
        self.assertTrue(res_act.just_activated)

        # 3. Stream Fist while ACTIVE at t=3.0..3.6s (hold duration >= 0.5s)
        for i in range(5):
            res_cmd = sm.process("Fist", 0.95, timestamp=3.0 + (i * 0.1))
            self.assertIsNone(res_cmd.command)  # Not confirmed yet

        # Frame at t = 3.6s (consecutive >= 3 and hold >= 0.5s) -> Command emitted!
        res_cmd_done = sm.process("Fist", 0.95, timestamp=3.6)
        self.assertEqual(res_cmd_done.state, SystemState.ACTIVE)
        self.assertEqual(res_cmd_done.command, "Fist")

        # 4. Inactivity until t = 9.0s (5.4s later > 5.0s timeout) -> LOCKED
        res_timeout = sm.process(None, 0.0, timestamp=9.0)
        self.assertEqual(res_timeout.state, SystemState.LOCKED)
        self.assertTrue(res_timeout.just_locked)

    # -------------------------------------------------------------------------
    # Test 11: Adaptive Threshold Policy Wiring & Dual-Threshold Hysteresis
    # -------------------------------------------------------------------------
    def test_adaptive_threshold_policy_wiring_and_hysteresis(self):
        """Verifies that ActivationStateMachine wires policy and respects entry vs sustain thresholds."""
        policy = AdaptiveThresholdPolicy(
            AdaptiveThresholdConfig(
                default_entry_threshold=0.80,
                default_sustain_threshold=0.75,
                commit_frames=3,
            )
        )
        sm = ActivationStateMachine(
            activation_gesture="OpenPalm",
            activation_hold_duration=1.0,
            inactivity_timeout=10.0,
            confidence_threshold=0.80,
            stable_frames=5,
            command_hold_duration=0.5,
            threshold_policy=policy,
        )

        # 1. Filters receive policy
        self.assertIs(sm.activation_filter.threshold_policy, policy)
        self.assertIs(sm.command_filter.threshold_policy, policy)

        # 2. Frame 1 with 0.76 confidence fails entry threshold (requires 0.80)
        r_entry_fail = sm.process("OpenPalm", 0.76, timestamp=0.0)
        self.assertEqual(r_entry_fail.state, SystemState.LOCKED)
        self.assertEqual(sm.activation_filter.consecutive_frames, 0)

        # 3. Frames 1 & 2 with 0.82 confidence succeed (entry passes)
        sm.process("OpenPalm", 0.82, timestamp=0.1)
        self.assertEqual(sm.activation_filter.consecutive_frames, 1)
        sm.process("OpenPalm", 0.82, timestamp=0.2)
        self.assertEqual(sm.activation_filter.consecutive_frames, 2)

        # 4. Frame 3 with 0.76 confidence succeeds (sustain phase permits >= 0.75)
        sm.process("OpenPalm", 0.76, timestamp=0.3)
        self.assertEqual(sm.activation_filter.consecutive_frames, 3)

        # 5. Stricter 1.0s hold duration is still required for activation
        # Hold duration at t=0.3 is only 0.2s < 1.0s -> still LOCKED
        self.assertEqual(sm.state, SystemState.LOCKED)

    def test_backward_compatibility_without_threshold_policy(self):
        """Verifies that when threshold_policy is None, default fixed threshold is used."""
        sm = ActivationStateMachine(
            confidence_threshold=0.80,
        )
        self.assertIsNone(sm.threshold_policy)
        self.assertIsNone(sm.activation_filter.threshold_policy)
        self.assertIsNone(sm.command_filter.threshold_policy)
        self.assertEqual(sm.activation_filter.get_effective_threshold("OpenPalm"), 0.80)
        self.assertEqual(sm.command_filter.get_effective_threshold("Fist"), 0.80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
