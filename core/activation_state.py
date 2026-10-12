"""
core/activation_state.py
========================

IntentSense Phase 4: Activation & Safety State Machine.

A touchless gesture interface must not execute unintended actions from casual
hand movements or background motion. The system operates as a safety state machine:

States:
-------
1. LOCKED (default):
   - All normal gesture commands (e.g., Fist, ThumbsUp, ThumbsDown) are suppressed.
   - The system listens exclusively for a deliberate activation gesture (OpenPalm).
   - Activation requires full Phase 3.1 temporal stability and a minimum hold
     duration (default: 1.0 second).

2. ACTIVE:
   - Entered only after verified activation confirmation.
   - Confirmed gesture commands are permitted to proceed to downstream handlers.
   - An inactivity timeout (default: 10.0 seconds) monitors the channel; if no
     valid gesture activity occurs, the system automatically reverts to LOCKED.
   - Can be manually reset/locked at any time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from core.temporal_filter import ConfirmationStatus, TemporalFilter, TemporalFilterResult


class SystemState(str, Enum):
    """Safety state enumeration."""
    LOCKED = "LOCKED"
    ACTIVE = "ACTIVE"


@dataclass
class ActivationResult:
    """
    Result emitted by ActivationStateMachine on each processed frame or event.

    Attributes:
        state: Current SystemState (LOCKED or ACTIVE).
        command: Confirmed gesture allowed to pass to downstream action layer (None if suppressed).
        is_active: Convenience boolean flag (True if ACTIVE, False if LOCKED).
        just_activated: True on the exact frame the transition LOCKED -> ACTIVE occurs.
        just_locked: True on the exact frame the transition ACTIVE -> LOCKED occurs.
        time_since_active: Elapsed seconds since last valid activity or activation.
        time_remaining: Remaining seconds before inactivity timeout locks the system.
        activation_progress: Normalized progress (0.0 to 1.0) towards activation hold duration.
        reason: Diagnostic reason string describing the decision.
    """
    state: SystemState
    command: Optional[str]
    is_active: bool
    just_activated: bool
    just_locked: bool
    time_since_active: float
    time_remaining: float
    activation_progress: float
    reason: str


class ActivationStateMachine:
    """
    Safety state machine managing the LOCKED <-> ACTIVE lifecycle.

    Guarantees:
    - Default state is LOCKED.
    - Transitions to ACTIVE only via verified activation gesture (OpenPalm) held
      for >= activation_hold_duration (1.0s).
    - While LOCKED, normal confirmed gestures are blocked (command is None).
    - While ACTIVE, confirmed gestures pass through and refresh the inactivity timer.
    - Automatically reverts to LOCKED upon inactivity timeout (default: 10.0s).
    - Supports manual locking/resetting.
    """

    def __init__(
        self,
        activation_gesture: str = "OpenPalm",
        activation_hold_duration: float = 1.0,
        inactivity_timeout: float = 10.0,
        confidence_threshold: float = 0.80,
        stable_frames: int = 5,
        cooldown_frames: int = 15,
        command_hold_duration: float = 0.5,
        threshold_policy: Optional[Any] = None,
    ) -> None:
        """
        Initializes the ActivationStateMachine.

        Args:
            activation_gesture: Gesture required to activate the system. Default: 'OpenPalm'.
            activation_hold_duration: Required hold duration in seconds to activate. Default: 1.0s.
            inactivity_timeout: Inactivity window in seconds before reverting to LOCKED. Default: 10.0s.
            confidence_threshold: Model confidence threshold for underlying filters. Default: 0.80.
            stable_frames: Consecutive frames required for stability. Default: 5.
            cooldown_frames: Cooldown frames after confirmation. Default: 15.
            command_hold_duration: Hold duration required for normal commands when active. Default: 0.5s.
            threshold_policy: Optional adaptive threshold policy instance (e.g. AdaptiveThresholdPolicy).
        """
        if activation_hold_duration < 0.0:
            raise ValueError(f"activation_hold_duration must be >= 0.0, got {activation_hold_duration}")
        if inactivity_timeout <= 0.0:
            raise ValueError(f"inactivity_timeout must be > 0.0, got {inactivity_timeout}")

        self.activation_gesture: str = activation_gesture
        self.activation_hold_duration: float = float(activation_hold_duration)
        self.inactivity_timeout: float = float(inactivity_timeout)
        self.confidence_threshold: float = float(confidence_threshold)
        self.stable_frames: int = int(stable_frames)
        self.cooldown_frames: int = int(cooldown_frames)
        self.command_hold_duration: float = float(command_hold_duration)
        self.threshold_policy: Optional[Any] = threshold_policy

        # Initial state: strictly LOCKED
        self._state: SystemState = SystemState.LOCKED
        self._last_activity_time: Optional[float] = None
        self._activation_start_time: Optional[float] = None

        # Dedicated Phase 3.1 temporal filters for activation and commands
        self.activation_filter: TemporalFilter = TemporalFilter(
            confidence_threshold=self.confidence_threshold,
            stable_frames=self.stable_frames,
            cooldown_frames=self.cooldown_frames,
            min_hold_duration=self.activation_hold_duration,
            threshold_policy=self.threshold_policy,
        )

        self.command_filter: TemporalFilter = TemporalFilter(
            confidence_threshold=self.confidence_threshold,
            stable_frames=self.stable_frames,
            cooldown_frames=self.cooldown_frames,
            min_hold_duration=self.command_hold_duration,
            threshold_policy=self.threshold_policy,
        )

    @property
    def state(self) -> SystemState:
        """Returns current SystemState (LOCKED or ACTIVE)."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Returns True if system is in ACTIVE state."""
        return self._state == SystemState.ACTIVE

    @property
    def time_remaining(self) -> float:
        """Remaining seconds before inactivity timeout locks the system (0.0 if LOCKED)."""
        if self._state == SystemState.LOCKED or self._last_activity_time is None:
            return 0.0
        elapsed = time.monotonic() - self._last_activity_time
        return max(0.0, self.inactivity_timeout - elapsed)

    def lock(self, reason: str = "MANUAL_LOCK") -> None:
        """
        Manually locks the system, reverting state to LOCKED and clearing timers.
        """
        self._state = SystemState.LOCKED
        self._last_activity_time = None
        self._activation_start_time = None
        self.activation_filter.reset()
        self.command_filter.reset()

    def reset(self) -> None:
        """Alias for lock() with full filter and timer reset."""
        self.lock(reason="RESET")

    def _check_inactivity(self, now: float) -> bool:
        """
        Checks if inactivity timeout has elapsed while ACTIVE.
        Returns True if system transitioned from ACTIVE -> LOCKED.
        """
        if self._state == SystemState.ACTIVE and self._last_activity_time is not None:
            if (now - self._last_activity_time) >= (self.inactivity_timeout - 1e-5):
                self.lock(reason="INACTIVITY_TIMEOUT")
                return True
        return False

    def process(
        self,
        gesture: Optional[str],
        confidence: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> ActivationResult:
        """
        End-to-end frame processor: runs input through the appropriate Phase 3.1 filter
        and executes safety state transitions.

        Args:
            gesture: Predicted gesture label (e.g. 'OpenPalm', 'Fist', or None).
            confidence: Model confidence score (0.0 to 1.0).
            timestamp: Optional monotonic timestamp for deterministic execution.

        Returns:
            ActivationResult
        """
        now: float = time.monotonic() if timestamp is None else float(timestamp)

        # 1. Inactivity check for ACTIVE state
        just_locked: bool = self._check_inactivity(now)
        if just_locked:
            return ActivationResult(
                state=SystemState.LOCKED,
                command=None,
                is_active=False,
                just_activated=False,
                just_locked=True,
                time_since_active=0.0,
                time_remaining=0.0,
                activation_progress=0.0,
                reason="INACTIVITY_TIMEOUT",
            )

        # ---------------------------------------------------------------------
        # State: LOCKED
        # ---------------------------------------------------------------------
        if self._state == SystemState.LOCKED:
            act_res = self.activation_filter.process(gesture, confidence, timestamp=now)
            progress = (
                min(1.0, act_res.hold_duration / self.activation_hold_duration)
                if self.activation_hold_duration > 0.0
                else 1.0
            )

            # Check for deliberate activation confirmation
            if act_res.is_confirmed and act_res.gesture == self.activation_gesture:
                self._state = SystemState.ACTIVE
                self._last_activity_time = now
                self.activation_filter.reset()
                self.command_filter.reset()
                return ActivationResult(
                    state=SystemState.ACTIVE,
                    command=None,  # Activation gesture itself is consumed as a state transition
                    is_active=True,
                    just_activated=True,
                    just_locked=False,
                    time_since_active=0.0,
                    time_remaining=self.inactivity_timeout,
                    activation_progress=1.0,
                    reason="ACTIVATED",
                )

            # While LOCKED, all normal gestures are blocked
            reason = "LOCKED_LISTENING"
            if gesture == self.activation_gesture:
                reason = f"LOCKED_ACTIVATING ({progress * 100:.0f}%)"
            elif gesture is not None and confidence >= self.confidence_threshold:
                reason = "LOCKED_GESTURE_IGNORED"

            return ActivationResult(
                state=SystemState.LOCKED,
                command=None,
                is_active=False,
                just_activated=False,
                just_locked=False,
                time_since_active=0.0,
                time_remaining=0.0,
                activation_progress=progress,
                reason=reason,
            )

        # ---------------------------------------------------------------------
        # State: ACTIVE
        # ---------------------------------------------------------------------
        cmd_res = self.command_filter.process(gesture, confidence, timestamp=now)
        time_since = max(0.0, now - self._last_activity_time) if self._last_activity_time is not None else 0.0
        time_rem = max(0.0, self.inactivity_timeout - time_since)

        # Confirmed command allowed to proceed
        if cmd_res.is_confirmed and cmd_res.gesture:
            # Refresh inactivity watchdog
            self._last_activity_time = now
            return ActivationResult(
                state=SystemState.ACTIVE,
                command=cmd_res.gesture,
                is_active=True,
                just_activated=False,
                just_locked=False,
                time_since_active=0.0,
                time_remaining=self.inactivity_timeout,
                activation_progress=1.0,
                reason="COMMAND_ALLOWED",
            )

        # When active, hand presence with high confidence also refreshes keepalive
        if gesture is not None and confidence >= self.confidence_threshold:
            self._last_activity_time = now
            time_since = 0.0
            time_rem = self.inactivity_timeout

        return ActivationResult(
            state=SystemState.ACTIVE,
            command=None,
            is_active=True,
            just_activated=False,
            just_locked=False,
            time_since_active=time_since,
            time_remaining=time_rem,
            activation_progress=1.0,
            reason="ACTIVE_IDLE",
        )

    def process_confirmed(
        self,
        gesture: Optional[str],
        is_confirmed: bool = False,
        hold_duration: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> ActivationResult:
        """
        Direct event processor for pre-evaluated / unit-tested gestures.

        Enforces:
        - In LOCKED state: Only confirmed OpenPalm with hold_duration >= activation_hold_duration
          activates the system. All other gestures are ignored.
        - In ACTIVE state: Confirmed gestures pass through as commands and refresh inactivity.
        - Inactivity timeout enforcement.

        Args:
            gesture: Gesture label.
            is_confirmed: Whether the gesture met temporal stability confirmation.
            hold_duration: Measured continuous hold duration in seconds.
            timestamp: Monotonic timestamp in seconds.

        Returns:
            ActivationResult
        """
        now: float = time.monotonic() if timestamp is None else float(timestamp)

        # Inactivity check
        just_locked: bool = self._check_inactivity(now)
        if just_locked:
            return ActivationResult(
                state=SystemState.LOCKED,
                command=None,
                is_active=False,
                just_activated=False,
                just_locked=True,
                time_since_active=0.0,
                time_remaining=0.0,
                activation_progress=0.0,
                reason="INACTIVITY_TIMEOUT",
            )

        # ---------------------------------------------------------------------
        # State: LOCKED
        # ---------------------------------------------------------------------
        if self._state == SystemState.LOCKED:
            if is_confirmed and gesture == self.activation_gesture:
                if hold_duration >= (self.activation_hold_duration - 1e-5):
                    self._state = SystemState.ACTIVE
                    self._last_activity_time = now
                    return ActivationResult(
                        state=SystemState.ACTIVE,
                        command=None,
                        is_active=True,
                        just_activated=True,
                        just_locked=False,
                        time_since_active=0.0,
                        time_remaining=self.inactivity_timeout,
                        activation_progress=1.0,
                        reason="ACTIVATED",
                    )
                else:
                    prog = min(1.0, hold_duration / self.activation_hold_duration)
                    return ActivationResult(
                        state=SystemState.LOCKED,
                        command=None,
                        is_active=False,
                        just_activated=False,
                        just_locked=False,
                        time_since_active=0.0,
                        time_remaining=0.0,
                        activation_progress=prog,
                        reason="INSUFFICIENT_ACTIVATION_HOLD",
                    )

            # Normal gestures or unconfirmed gestures are ignored while locked
            return ActivationResult(
                state=SystemState.LOCKED,
                command=None,
                is_active=False,
                just_activated=False,
                just_locked=False,
                time_since_active=0.0,
                time_remaining=0.0,
                activation_progress=0.0,
                reason="LOCKED_IGNORE",
            )

        # ---------------------------------------------------------------------
        # State: ACTIVE
        # ---------------------------------------------------------------------
        time_since = max(0.0, now - self._last_activity_time) if self._last_activity_time is not None else 0.0
        time_rem = max(0.0, self.inactivity_timeout - time_since)

        if is_confirmed and gesture:
            self._last_activity_time = now
            return ActivationResult(
                state=SystemState.ACTIVE,
                command=gesture,
                is_active=True,
                just_activated=False,
                just_locked=False,
                time_since_active=0.0,
                time_remaining=self.inactivity_timeout,
                activation_progress=1.0,
                reason="COMMAND_ALLOWED",
            )

        return ActivationResult(
            state=SystemState.ACTIVE,
            command=None,
            is_active=True,
            just_activated=False,
            just_locked=False,
            time_since_active=time_since,
            time_remaining=time_rem,
            activation_progress=1.0,
            reason="ACTIVE_IDLE",
        )

    def process_filter_result(
        self,
        filter_result: TemporalFilterResult,
        timestamp: Optional[float] = None,
    ) -> ActivationResult:
        """
        Processes a TemporalFilterResult directly.

        Args:
            filter_result: Output from a TemporalFilter.
            timestamp: Optional monotonic timestamp.

        Returns:
            ActivationResult
        """
        return self.process_confirmed(
            gesture=filter_result.gesture or filter_result.candidate_gesture,
            is_confirmed=filter_result.is_confirmed,
            hold_duration=filter_result.hold_duration,
            timestamp=timestamp,
        )
