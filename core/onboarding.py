"""
core/onboarding.py
==================

IntentSense Phase 4: Touchless Guided Onboarding Practice Mode.

Guides users through physical gesture interactions step-by-step:
  Step 1: OpenPalm Activation (LOCKED -> ACTIVE safety unlock)
  Step 2: ThumbsUp Command (Media context volume up)
  Step 3: Dynamic Swipe Right (Touchless track/slide navigation)
  Step 4: Pointing Command (Presentation context slide advance)
  Step 5: Practice Complete (Ready for touchless operation)

Guarantees:
- Gesture-first: Advances automatically when physical gesture criteria are met.
  Zero keyboard typing or mouse clicks required to practice.
- Safety: Real OS actions are strictly suppressed during practice.
- Uses only existing built-in gestures and swipe detectors (no invented poses).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

from core.activation_state import SystemState
from core.context_manager import Context
from core.swipe_detector import SwipeDirection


class OnboardingStepId(IntEnum):
    UNLOCK = 0
    VOLUME_UP = 1
    SWIPE_RIGHT = 2
    POINTING = 3
    COMPLETE = 4


@dataclass(frozen=True)
class OnboardingStepResult:
    """State snapshot for the active onboarding practice step."""
    step_id: OnboardingStepId
    step_number: int
    total_steps: int
    title: str
    target_gesture: str
    guidance: str
    progress: float
    feedback: str
    is_step_complete: bool
    is_all_complete: bool


class TouchlessOnboardingController:
    """
    State machine driving touchless onboarding without keyboard interaction.
    Evaluates live telemetry to confirm physical gesture execution.
    """

    TOTAL_STEPS: int = 4  # Steps 1 to 4 (Step 5 is completion summary)

    def __init__(self) -> None:
        self._current_step: OnboardingStepId = OnboardingStepId.UNLOCK
        self._step_completed_at: Optional[float] = None
        self._step_transition_delay: float = 1.2  # Seconds to show success checkmark before next step
        self._feedback: str = "Present OpenPalm towards camera to begin."

    @property
    def current_step(self) -> OnboardingStepId:
        return self._current_step

    @property
    def is_complete(self) -> bool:
        return self._current_step == OnboardingStepId.COMPLETE

    def reset(self) -> None:
        """Resets the onboarding sequence back to Step 1."""
        self._current_step = OnboardingStepId.UNLOCK
        self._step_completed_at = None
        self._feedback = "Present OpenPalm towards camera to begin."

    def update(
        self,
        gesture: Optional[str],
        confidence: float,
        hold_duration: float,
        target_hold: float,
        is_confirmed: bool,
        swipe_direction: SwipeDirection = SwipeDirection.NONE,
        just_activated: bool = False,
        timestamp: float = 0.0,
    ) -> OnboardingStepResult:
        """
        Processes real-time telemetry frame-by-frame and advances steps automatically.
        """
        # If currently pausing to celebrate a completed step
        if self._step_completed_at is not None:
            elapsed = timestamp - self._step_completed_at
            if elapsed >= self._step_transition_delay:
                self._step_completed_at = None
                self._current_step = OnboardingStepId(min(self._current_step + 1, OnboardingStepId.COMPLETE))

        step = self._current_step
        progress = 0.0
        is_step_complete = (self._step_completed_at is not None)

        if step == OnboardingStepId.UNLOCK:
            title = "Step 1: System Activation & Safety Unlock"
            target_gesture = "OpenPalm"
            guidance = "IntentSense starts safely LOCKED. Hold OpenPalm steady for 1.0s to activate."

            if not is_step_complete:
                if gesture == "OpenPalm":
                    progress = min(1.0, hold_duration / max(0.1, target_hold))
                    if just_activated or (is_confirmed and hold_duration >= 0.95):
                        self._step_completed_at = timestamp
                        self._feedback = "✓ Great! System safely unlocked."
                        progress = 1.0
                    else:
                        self._feedback = f"Holding OpenPalm: {int(progress * 100)}% (keep steady)"
                elif gesture:
                    progress = 0.0
                    self._feedback = f"Detected '{gesture}'. Please show OpenPalm with fingers spread."
                else:
                    progress = 0.0
                    self._feedback = "Hold OpenPalm facing camera to unlock."

        elif step == OnboardingStepId.VOLUME_UP:
            title = "Step 2: Media Control (Hold Command)"
            target_gesture = "ThumbsUp"
            guidance = "Hold ThumbsUp steady for 0.5s to trigger Media Volume Up."

            if not is_step_complete:
                if gesture == "ThumbsUp":
                    progress = min(1.0, hold_duration / max(0.1, target_hold))
                    if is_confirmed or hold_duration >= target_hold:
                        self._step_completed_at = timestamp
                        self._feedback = "✓ Verified! Volume Up command confirmed."
                        progress = 1.0
                    else:
                        self._feedback = f"Holding ThumbsUp: {int(progress * 100)}%"
                elif gesture:
                    progress = 0.0
                    self._feedback = f"Detected '{gesture}'. Show ThumbsUp to practice volume control."
                else:
                    progress = 0.0
                    self._feedback = "Show ThumbsUp and hold steady."

        elif step == OnboardingStepId.SWIPE_RIGHT:
            title = "Step 3: Dynamic Gesture (Horizontal Swipe)"
            target_gesture = "Swipe Right"
            guidance = "Perform a quick horizontal hand swipe from Left to Right across the camera."

            if not is_step_complete:
                if swipe_direction == SwipeDirection.RIGHT:
                    self._step_completed_at = timestamp
                    self._feedback = "✓ Excellent! Dynamic Swipe Right detected."
                    progress = 1.0
                elif swipe_direction == SwipeDirection.LEFT:
                    self._feedback = "Detected Swipe Left. Try swiping Left to Right."
                    progress = 0.3
                else:
                    progress = 0.0
                    self._feedback = "Swipe your hand from Left to Right across the camera."

        elif step == OnboardingStepId.POINTING:
            title = "Step 4: Presentation Navigation"
            target_gesture = "Pointing"
            guidance = "Point index finger steady for 0.5s to advance presentation slides."

            if not is_step_complete:
                if gesture == "Pointing":
                    progress = min(1.0, hold_duration / max(0.1, target_hold))
                    if is_confirmed or hold_duration >= target_hold:
                        self._step_completed_at = timestamp
                        self._feedback = "✓ Verified! Pointing gesture confirmed."
                        progress = 1.0
                    else:
                        self._feedback = f"Holding Pointing: {int(progress * 100)}%"
                elif gesture:
                    progress = 0.0
                    self._feedback = f"Detected '{gesture}'. Point with index finger to practice."
                else:
                    progress = 0.0
                    self._feedback = "Point with your index finger towards the screen."

        else:  # COMPLETE
            title = "Touchless Onboarding Complete!"
            target_gesture = "All Gestures Ready"
            guidance = "You have mastered touchless interaction! Press ESC or O to return to live mode."
            progress = 1.0
            self._feedback = "🎉 You're ready to control your Mac touchlessly."

        return OnboardingStepResult(
            step_id=step,
            step_number=int(step) + 1 if step != OnboardingStepId.COMPLETE else self.TOTAL_STEPS,
            total_steps=self.TOTAL_STEPS,
            title=title,
            target_gesture=target_gesture,
            guidance=guidance,
            progress=progress,
            feedback=self._feedback,
            is_step_complete=(self._step_completed_at is not None),
            is_all_complete=(step == OnboardingStepId.COMPLETE),
        )

