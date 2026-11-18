"""
core/llm_assistant.py
=====================

IntentSense Intelligent LLM Assistance Engine (Phase 1).

Provides privacy-preserving, zero-trust, and fail-safe LLM integration for:
  1. Intelligent Gesture & Rejection Explanation (coaching feedback).
  2. Natural Language Gesture Configuration (guided mapping & enrollment).
  3. Interactive AI Onboarding & Troubleshooting (step-by-step guidance).
  4. Natural Language Commands (typed intent parsing through safety gates).
  5. Personalized Profile Assistance (ergonomic & stability recommendations).

Architectural Safety & Privacy Guarantees:
  - Zero Vision / Landmark Data: NEVER receives or accepts video frames or 42D coordinates.
    Enforced by runtime type checks that raise TypeError if arrays or coordinates are passed.
  - Zero Direct OS Execution: Generates strictly typed Intent / Action specifications.
    Contains no subprocess, os.system, or dynamic exec/eval calls.
  - Strict Action Allowlist: Validates all LLM outputs against known Intent and Action sets.
  - Non-blocking Execution: Background worker pool ensures zero frame-drops in the real-time pipeline.
  - Offline Graceful Degradation: Fully functional deterministic rule-based engine if offline,
    unconfigured, or if an API key is missing.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import socket
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from core.context_manager import Context
from core.intent_engine import KNOWN_GESTURES, Intent
from core.personalized_gestures import RESERVED_NAMES, SUPPORTED_ACTION_MAP
from core.telemetry_aggregator import StrugglePattern, StruggleType


# ---------------------------------------------------------------------------
# Strict Action Allowlist & Known Names
# ---------------------------------------------------------------------------
ALLOWED_INTENT_VALUES: Set[str] = {intent.value for intent in Intent}
ALLOWED_ACTION_IDS: Set[str] = set(SUPPORTED_ACTION_MAP.keys())
ALLOWED_CONTEXTS: Set[str] = {c.value for c in Context} | {"ALL"}


def _assert_no_vision_payload(*args: Any) -> None:
    """
    Enforces the zero-vision boundary.
    Raises TypeError if numpy arrays, large numerical sequences, or image payloads are passed.
    """
    for a in args:
        if a is None:
            continue
        if isinstance(a, dict):
            _assert_no_vision_payload(*a.values())
            continue
        # Check for numpy array or image-like structures
        type_str = str(type(a))
        if "ndarray" in type_str or "Image" in type_str:
            raise TypeError("Privacy Violation: Raw vision frames or numpy arrays are strictly forbidden from LLM processing.")
        # Check for list of landmarks / floats
        if isinstance(a, (list, tuple)) and len(a) > 0 and isinstance(a[0], (list, tuple, float, int)):
            if len(a) >= 20:  # e.g., 21 MediaPipe landmarks or 42D normalized features
                raise TypeError("Privacy Violation: Raw landmark or feature coordinate sequences are strictly forbidden from LLM processing.")


def _assert_no_audio_payload(*args: Any) -> None:
    """
    Enforces the zero-audio boundary for external LLM calls.
    Raises TypeError if numpy arrays, audio bytes, or audio buffers are passed.
    """
    for a in args:
        if a is None:
            continue
        if isinstance(a, dict):
            _assert_no_audio_payload(*a.values())
            continue
        type_str = str(type(a))
        if "ndarray" in type_str or "bytes" in type_str or "bytearray" in type_str or "Audio" in type_str:
            raise TypeError("Privacy Violation: Raw audio waveforms, bytes, or arrays are strictly forbidden from LLM processing.")
        if isinstance(a, (list, tuple)) and len(a) > 0:
            first = a[0]
            if isinstance(first, (int, float)) and len(a) > 100:  # Raw PCM audio sample buffer
                raise TypeError("Privacy Violation: Raw audio sample buffers are strictly forbidden from LLM processing.")


def _clean_json_markdown(text: str) -> str:
    """Strips markdown code fences (```json ... ```) from LLM output."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


# ---------------------------------------------------------------------------
# Structured Output Data Models
# ---------------------------------------------------------------------------
@dataclass
class ExplanationResponse:
    """Natural-language coaching explanation of a pipeline decision."""
    summary: str
    coaching_tip: str
    suggested_action: str
    category: str = "EXPLANATION"
    provider: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VoiceAssistantResponse:
    """Natural-language conversational answer to a user's spoken question."""
    question: str
    answer: str
    suggested_action: str = ""
    topic: str = "GENERAL"
    category: str = "HELP"
    provider: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NaturalCommandResponse:
    """Structured translation of a typed command into a verified system intent."""
    intent: Optional[Intent]
    context: Optional[Context]
    confidence: float
    explanation: str
    is_valid: bool
    raw_query: str = ""

    @property
    def action_id(self) -> Optional[str]:
        if not self.intent:
            return None
        mapping = {
            Intent.PLAY_PAUSE: "play_pause",
            Intent.PREVIOUS: "previous_track",
            Intent.NEXT: "next_track",
            Intent.VOLUME_UP: "volume_up",
            Intent.VOLUME_DOWN: "volume_down",
            Intent.START_STOP: "start_stop_slideshow",
            Intent.PREVIOUS_SLIDE: "previous_slide",
            Intent.NEXT_SLIDE: "next_slide",
            Intent.ZOOM_IN: "zoom_in",
            Intent.ZOOM_OUT: "zoom_out",
        }
        return mapping.get(self.intent, self.intent.value.lower())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value if self.intent else None,
            "action_id": self.action_id,
            "context": self.context.value if self.context else None,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "is_valid": self.is_valid,
            "raw_query": self.raw_query,
        }


@dataclass
class ConfigurationResponse:
    """Structured proposal for gesture-to-action mapping with confirmation."""
    gesture_name: str
    is_builtin: bool
    action_id: str
    context: str  # "MEDIA", "PRESENTATION", "CUSTOM", or "ALL"
    needs_enrollment: bool
    confirmation_message: str
    is_valid: bool
    error_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OnboardingStep:
    """Structured guidance for an onboarding or troubleshooting topic."""
    topic_id: str
    title: str
    guidance: str
    key_controls: List[str]
    troubleshooting_tip: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PersonalizedSuggestionResponse:
    """Profile ergonomics, stability checks, and recommended custom mappings."""
    profile_diagnostics: List[Dict[str, str]]
    recommended_mappings: List[Dict[str, str]]
    insights: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Deterministic Rule-Based Offline Fallback Engine
# ---------------------------------------------------------------------------
class OfflineRuleBasedProvider:
    """
    Guarantees 100% functionality offline with zero network or external dependencies.
    Produces strictly structured, helpful responses based on deterministic heuristics.
    """

    def explain_decision(
        self,
        decision_category: str,
        decision_text: str,
        state: str,
        context: str,
        raw_gesture: Optional[str],
        confidence: float,
        effective_threshold: float,
        hold_duration: float,
        target_hold: float,
        is_swipe: bool = False,
    ) -> ExplanationResponse:
        conf_pct = int(confidence * 100)
        thresh_pct = int(effective_threshold * 100)

        if state == "LOCKED":
            if raw_gesture == "OpenPalm":
                prog = f"{hold_duration:.1f}s / {target_hold:.1f}s"
                return ExplanationResponse(
                    summary=f"System is currently locked. Unlocking progress: {prog}.",
                    coaching_tip="Keep your open palm flat and steady in clear view of the camera until the 1.0s unlock completes.",
                    suggested_action="Hold OpenPalm steadily for 1.0s.",
                    category="LOCKED_UNLOCKING",
                )
            elif raw_gesture:
                return ExplanationResponse(
                    summary=f"Detected gesture '{raw_gesture}', but IntentSense is currently LOCKED for safety.",
                    coaching_tip="IntentSense suppresses all commands while locked to prevent accidental triggers. To unlock, hold an OpenPalm steady for 1.0s.",
                    suggested_action="Perform OpenPalm for 1.0s to activate.",
                    category="LOCKED_SUPPRESSED",
                )
            else:
                return ExplanationResponse(
                    summary="System is safely LOCKED and waiting for hand detection.",
                    coaching_tip="Position your hand 1.5 to 3 feet from the webcam with good lighting. Hold an OpenPalm for 1.0s to activate.",
                    suggested_action="Show OpenPalm to unlock.",
                    category="LOCKED_IDLE",
                )

        if decision_category == "COOLDOWN":
            return ExplanationResponse(
                summary="Action triggered recently and is currently in cooldown rate-limiting.",
                coaching_tip="IntentSense enforces an 0.8s rate-limit between repeated actions so you don't accidentally skip multiple tracks or slides.",
                suggested_action="Pause briefly before repeating the gesture.",
                category="COOLDOWN",
            )

        if decision_category in ("ACTION EXECUTED", "ACTION_EXECUTED", "CONFIRMED", "ACTION READY", "SWIPE DETECTED", "SWIPE_DETECTED"):
            return ExplanationResponse(
                summary=f"Successfully recognized and executed '{decision_text}'.",
                coaching_tip="Intent confirmed! Lower your hand or switch poses when ready for your next command.",
                suggested_action="Ready for next gesture.",
                category=decision_category,
            )

        if confidence < effective_threshold and raw_gesture:
            return ExplanationResponse(
                summary=f"Gesture '{raw_gesture}' confidence ({conf_pct}%) was below the required threshold ({thresh_pct}%).",
                coaching_tip="Spread your fingers clearly, keep your palm perpendicular to the camera, and ensure sufficient room lighting.",
                suggested_action="Hold the pose clearly in the center of the frame.",
                category="LOW_CONFIDENCE",
            )

        if decision_category == "WAITING":
            return ExplanationResponse(
                summary=f"Gesture '{raw_gesture}' recognized, accumulating steady hold duration ({hold_duration:.1f}s / {target_hold:.1f}s).",
                coaching_tip="IntentSense requires a continuous 0.5s hold to ensure intentional command input.",
                suggested_action="Hold the pose steady for a fraction of a second longer.",
                category="HOLD_WAITING",
            )

        return ExplanationResponse(
            summary=f"Decision state: {decision_text}." if decision_text else "Insufficient event metadata to determine cause.",
            coaching_tip="Ensure your hand is fully in frame, steady, and facing the camera.",
            suggested_action="Continue regular operation.",
            category="GENERAL",
        )

    def parse_natural_command(self, query: str, active_context: Context) -> NaturalCommandResponse:
        q = query.strip().lower()

        # Media Context Commands
        if any(w in q for w in ["play", "pause", "resume", "toggle music", "toggle playback"]):
            return NaturalCommandResponse(
                intent=Intent.PLAY_PAUSE,
                context=Context.MEDIA,
                confidence=0.95,
                explanation="Mapped to Media Play/Pause toggle.",
                is_valid=True,
                raw_query=query,
            )
        if any(w in q for w in ["volume up", "increase volume", "louder", "turn it up", "sound up"]):
            return NaturalCommandResponse(
                intent=Intent.VOLUME_UP,
                context=Context.MEDIA,
                confidence=0.95,
                explanation="Mapped to Volume Up.",
                is_valid=True,
                raw_query=query,
            )
        if any(w in q for w in ["volume down", "decrease volume", "softer", "quieter", "lower volume", "turn it down", "sound down", "lower sound"]):
            return NaturalCommandResponse(
                intent=Intent.VOLUME_DOWN,
                context=Context.MEDIA,
                confidence=0.95,
                explanation="Mapped to Volume Down.",
                is_valid=True,
                raw_query=query,
            )
        if any(w in q for w in ["next track", "skip track", "skip song", "next song", "skip forward"]):
            return NaturalCommandResponse(
                intent=Intent.NEXT,
                context=Context.MEDIA,
                confidence=0.95,
                explanation="Mapped to Next Track.",
                is_valid=True,
                raw_query=query,
            )
        if any(w in q for w in ["previous track", "prior track", "last song", "previous song", "go back song"]):
            return NaturalCommandResponse(
                intent=Intent.PREVIOUS,
                context=Context.MEDIA,
                confidence=0.95,
                explanation="Mapped to Previous Track.",
                is_valid=True,
                raw_query=query,
            )

        # Presentation Context Commands
        if any(w in q for w in ["next slide", "advance slide", "forward slide", "next page"]):
            return NaturalCommandResponse(
                intent=Intent.NEXT_SLIDE,
                context=Context.PRESENTATION,
                confidence=0.95,
                explanation="Mapped to Next Slide.",
                is_valid=True,
                raw_query=query,
            )
        if any(w in q for w in ["previous slide", "back slide", "prior slide", "previous page", "last slide"]):
            return NaturalCommandResponse(
                intent=Intent.PREVIOUS_SLIDE,
                context=Context.PRESENTATION,
                confidence=0.95,
                explanation="Mapped to Previous Slide.",
                is_valid=True,
                raw_query=query,
            )
        if any(w in q for w in ["start presentation", "start slideshow", "stop presentation", "exit presentation", "toggle slideshow"]):
            return NaturalCommandResponse(
                intent=Intent.START_STOP,
                context=Context.PRESENTATION,
                confidence=0.95,
                explanation="Mapped to Start/Stop Slideshow.",
                is_valid=True,
                raw_query=query,
            )
        if any(w in q for w in ["zoom in", "magnify", "enlarge"]):
            return NaturalCommandResponse(
                intent=Intent.ZOOM_IN,
                context=Context.PRESENTATION,
                confidence=0.90,
                explanation="Mapped to Zoom In.",
                is_valid=True,
                raw_query=query,
            )
        if any(w in q for w in ["zoom out", "shrink", "zoom back"]):
            return NaturalCommandResponse(
                intent=Intent.ZOOM_OUT,
                context=Context.PRESENTATION,
                confidence=0.90,
                explanation="Mapped to Zoom Out.",
                is_valid=True,
                raw_query=query,
            )

        return NaturalCommandResponse(
            intent=None,
            context=None,
            confidence=0.0,
            explanation=f"Command '{query}' could not be safely mapped to any supported IntentSense action.",
            is_valid=False,
            raw_query=query,
        )

    def parse_gesture_configuration(
        self,
        description: str,
        existing_custom_names: Sequence[str] = (),
    ) -> ConfigurationResponse:
        d = description.strip().lower()

        # 1. Identify Action
        matched_action: Optional[str] = None
        if "play" in d or "pause" in d:
            matched_action = "play_pause"
        elif "volume up" in d or "louder" in d:
            matched_action = "volume_up"
        elif "volume down" in d or "quieter" in d or "softer" in d:
            matched_action = "volume_down"
        elif "next track" in d or "skip song" in d or "next song" in d:
            matched_action = "next_track"
        elif "previous track" in d or "prior song" in d:
            matched_action = "previous_track"
        elif "next slide" in d:
            matched_action = "next_slide"
        elif "previous slide" in d:
            matched_action = "previous_slide"
        elif "slideshow" in d or ("presentation" in d and ("start" in d or "stop" in d)):
            matched_action = "start_stop_presentation"
        elif "zoom in" in d:
            matched_action = "zoom_in"
        elif "zoom out" in d:
            matched_action = "zoom_out"

        if not matched_action:
            return ConfigurationResponse(
                gesture_name="",
                is_builtin=False,
                action_id="",
                context="CUSTOM",
                needs_enrollment=False,
                confirmation_message="",
                is_valid=False,
                error_reason="Could not match your description to a supported action. Try: play_pause, volume_up/down, next/prev slide, next/prev track.",
            )

        # 2. Identify Context
        matched_context = "CUSTOM"
        if "presentation" in d:
            matched_context = "PRESENTATION"
        elif "media" in d or "music" in d:
            matched_context = "MEDIA"
        elif "all" in d or "everywhere" in d:
            matched_context = "ALL"

        # 3. Identify Gesture Name
        matched_gesture = ""
        is_builtin = False
        needs_enrollment = False

        for g in KNOWN_GESTURES:
            if g.lower() in d:
                matched_gesture = g
                is_builtin = True
                break

        if not matched_gesture:
            quoted = re.findall(r"['\"]([^'\"]+)['\"]", description)
            if quoted:
                matched_gesture = quoted[0].strip().replace(" ", "")
            else:
                words = re.findall(r"\b[A-Za-z0-9_-]+\b", description)
                candidates = [
                    w for w in words
                    if w.lower() not in {
                        "map", "to", "in", "when", "mode", "use", "for",
                        "the", "a", "i", "want", "set", "gesture", "action"
                    }
                ]
                if candidates:
                    matched_gesture = candidates[0].capitalize()

        if not matched_gesture:
            matched_gesture = "CustomPose"

        if matched_gesture in RESERVED_NAMES and not is_builtin:
            is_builtin = True

        if not is_builtin and matched_gesture not in existing_custom_names:
            needs_enrollment = True

        action_label = matched_action.replace("_", " ").title()
        conf_msg = f"Map gesture '{matched_gesture}' to '{action_label}' in {matched_context} context?"
        if needs_enrollment:
            conf_msg += " (Requires 3-take camera enrollment)."

        return ConfigurationResponse(
            gesture_name=matched_gesture,
            is_builtin=is_builtin,
            action_id=matched_action,
            context=matched_context,
            needs_enrollment=needs_enrollment,
            confirmation_message=conf_msg,
            is_valid=True,
        )

    def get_onboarding_guide(self, topic: str) -> OnboardingStep:
        t = topic.lower().strip()
        if "activation" in t or "unlock" in t:
            return OnboardingStep(
                topic_id="activation",
                title="System Activation & Safety Lock",
                guidance="IntentSense starts safely in LOCKED state to prevent accidental input. To activate, present an OpenPalm towards the camera and hold steadily for 1.0 full second until the HUD border turns green.",
                key_controls=["OpenPalm (hold 1.0s) -> Unlock", "L -> Manually Lock", "R -> Reset State"],
                troubleshooting_tip="If unlock fails, make sure all 5 fingers are spread and clearly visible in good lighting.",
            )
        elif "context" in t or "mode" in t:
            return OnboardingStep(
                topic_id="context_switching",
                title="Context Management (Media vs Presentation vs Custom)",
                guidance="Gestures adapt to your task! In MEDIA context (1), ThumbsUp/Down adjusts volume. In PRESENTATION context (2), Pointing advances slides. In CUSTOM context (3), your personalized poses become active.",
                key_controls=["1 -> Media Context", "2 -> Presentation Context", "3 -> Custom Context"],
                troubleshooting_tip="Swiping left/right works in both Media (prev/next track) and Presentation (prev/next slide).",
            )
        elif "custom" in t or "enroll" in t:
            return OnboardingStep(
                topic_id="enrollment",
                title="Personalized Custom Gesture Enrollment",
                guidance="You can teach IntentSense your own custom poses! Press 'E' to enter Enrollment mode. IntentSense captures 3 bursts (10 frames each) with cross-validation to guarantee zero false positives.",
                key_controls=["E -> Enroll New Gesture", "M -> Manage Gestures", "T -> Test Gesture Matching"],
                troubleshooting_tip="Keep your hand very steady across all 3 takes. If dispersion exceeds 0.12, the safety validator will ask you to retry.",
            )
        else:
            return OnboardingStep(
                topic_id="general_troubleshooting",
                title="General Troubleshooting & Tips",
                guidance="Keep your webcam at eye or chest level, roughly 2-3 feet away. Ensure your hand is not back-lit by a bright window.",
                key_controls=["? or H -> Help Menu", "Q -> Quit Application", ": -> Type Natural Command"],
                troubleshooting_tip="If recognition drops below 80%, move hand slightly closer to the center of the camera frame.",
            )

    def get_personalized_recommendations(
        self,
        profiles_summary: List[Dict[str, Any]],
    ) -> PersonalizedSuggestionResponse:
        diagnostics: List[Dict[str, str]] = []
        recommendations: List[Dict[str, str]] = []

        if not profiles_summary:
            diagnostics.append({
                "status": "NO_CUSTOM_GESTURES",
                "message": "No custom gestures enrolled yet. The system is operating on built-in Random Forest models.",
            })
            recommendations.append({
                "gesture": "V-Sign or Two-Fingers",
                "action": "Play / Pause Media",
                "context": "CUSTOM",
                "rationale": "Easy to hold steadily with low dispersion.",
            })
            recommendations.append({
                "gesture": "Shaka / Call-Me",
                "action": "Start / Stop Slideshow",
                "context": "PRESENTATION",
                "rationale": "High visual separation from standard pointing gestures.",
            })
            insights = "Enrolling 1-2 custom gestures for your most frequent actions can streamline touchless presentations."
        else:
            for p in profiles_summary:
                disp = p.get("dispersion", 0.0)
                name = p.get("name", "Unknown")
                mappings = p.get("mappings", {})
                if disp > 0.09:
                    diagnostics.append({
                        "gesture": name,
                        "status": "MODERATE_DISPERSION",
                        "message": f"Dispersion is {disp:.3f}. Consider re-enrolling (M -> R) while holding hand firmer for better accuracy.",
                    })
                else:
                    diagnostics.append({
                        "gesture": name,
                        "status": "EXCELLENT_STABILITY",
                        "message": f"Dispersion is very tight ({disp:.3f} <= 0.09). Reliable pose stability.",
                    })

                if not mappings:
                    recommendations.append({
                        "gesture": name,
                        "action": "play_pause",
                        "context": "CUSTOM",
                        "rationale": "This gesture has no active mappings. Assign it to Play/Pause.",
                    })

            insights = f"You have {len(profiles_summary)} personalized gesture(s) enrolled. Custom poses are guarded by Tier-2 arbitration so built-in gestures always take precedence."

        return PersonalizedSuggestionResponse(
            profile_diagnostics=diagnostics,
            recommended_mappings=recommendations,
            insights=insights,
        )

    def coach_struggle(
        self,
        pattern: StrugglePattern,
        context: Context = Context.MEDIA,
    ) -> ExplanationResponse:
        """
        Produces deterministic, grounded coaching feedback for detected physical struggle patterns.
        """
        ptype = pattern.pattern_type
        g_name = pattern.gesture or "gesture"

        if ptype == StruggleType.HOLD_DEFICIT:
            avg_hold = pattern.metrics.get("avg_hold", 0.3)
            target = pattern.metrics.get("target_hold", 0.5)
            return ExplanationResponse(
                summary="Hold gesture slightly longer before releasing.",
                coaching_tip=f"You held '{g_name}' for ~{avg_hold:.1f}s. Keep steady until target ({target:.1f}s).",
                suggested_action="Hold hand in frame a fraction of a second longer.",
                category="HOLD_DEFICIT",
                provider="offline_heuristic",
            )
        elif ptype == StruggleType.LOW_CONFIDENCE:
            avg_c = int(pattern.metrics.get("avg_confidence", 0.65) * 100)
            req_t = int(pattern.metrics.get("threshold", 0.8) * 100)
            return ExplanationResponse(
                summary=f"Gesture '{g_name}' needs clearer posture.",
                coaching_tip=f"Confidence for '{g_name}' was {avg_c}% (needed {req_t}%). Move hand to center with fingers visible.",
                suggested_action="Center hand and spread fingers clearly.",
                category="LOW_CONFIDENCE",
                provider="offline_heuristic",
            )
        elif ptype == StruggleType.LOCKED_SUPPRESSION:
            return ExplanationResponse(
                summary="System is LOCKED; commands are suppressed.",
                coaching_tip=f"IntentSense is locked. Show OpenPalm and hold for 1 second to unlock. {g_name} cannot execute actions while locked.",
                suggested_action="Hold OpenPalm for 1.0s to unlock.",
                category="LOCKED_SUPPRESSION",
                provider="offline_heuristic",
            )
        else:  # JITTER_INSTABILITY
            return ExplanationResponse(
                summary="Rapid hand movement detected.",
                coaching_tip="Hold one distinct hand pose steady rather than quickly changing gestures.",
                suggested_action="Pick one gesture and hold still.",
                category="JITTER_INSTABILITY",
                provider="offline_heuristic",
            )

    def answer_voice_question(
        self,
        question: str,
        app_state_summary: Optional[Dict[str, Any]] = None,
    ) -> VoiceAssistantResponse:
        """
        Deterministic, rule-based answering engine for IntentSense voice questions.
        Answers queries about gestures, locks, modes, custom profiles, coaching, and onboarding.
        """
        _assert_no_vision_payload(question, app_state_summary)
        _assert_no_audio_payload(question, app_state_summary)

        q = (question or "").strip().lower()
        summary = app_state_summary or {}
        is_locked = summary.get("is_locked", False) or summary.get("state") == "LOCKED"
        ctx = str(summary.get("context", "MEDIA")).upper()
        custom_gestures = summary.get("custom_gestures", [])

        # Safety: Spoken requests asking to directly execute system actions
        cmd_action_triggers = [
            ("volume up", "Volume Up", "ThumbsUp"),
            ("volume down", "Volume Down", "Fist"),
            ("next slide", "Next Slide", "Pointing or Swipe Right"),
            ("previous slide", "Previous Slide", "Swipe Left"),
            ("next track", "Next Track", "Swipe Right"),
            ("previous track", "Previous Track", "Swipe Left"),
            ("skip", "Skip Track", "Swipe Right"),
            ("play", "Play/Pause", "enrolled custom gesture"),
            ("pause", "Play/Pause", "enrolled custom gesture"),
        ]
        for trigger, act_name, gesture_needed in cmd_action_triggers:
            if re.search(rf"\b{re.escape(trigger)}\b", q) and any(w in q for w in ("can you", "please", "do a", "execute", "trigger", "now", "set")):
                return VoiceAssistantResponse(
                    question=question,
                    answer=f"Voice interaction is for assistance only, not direct system control. To perform {act_name}, unlock with OpenPalm, then use {gesture_needed}.",
                    suggested_action=f"Use gesture '{gesture_needed}'",
                    topic="SAFETY_GATE",
                    category="SAFETY",
                    provider="offline",
                )

        # 1. Locked / Unlocking questions
        if re.search(r"\b(unlock(?:ing)?|why locked|is (?:it )?locked|how to (?:un)?lock|lock (?:the )?(?:app|system|intentsense)|locked)\b", q) or (
            re.search(r"\b(not|isn't)\s+(?:it\s+)?(?:working|responding)\b|\bstuck\b", q)
        ):
            print(f"[LLM Offline] Query \"{question}\" matched IntentSense offline rule: 1_LOCKED_STATE")
            if is_locked:
                ans = "IntentSense is currently LOCKED for safety. Show OpenPalm to the camera and hold it steady for 1.0 second to unlock."
            else:
                ans = "IntentSense is currently ACTIVE. To lock it manually, press 'L' or wait for the 30-second inactivity timeout."
            return VoiceAssistantResponse(
                question=question,
                answer=ans,
                suggested_action="Hold OpenPalm 1.0s to unlock" if is_locked else "Press 'L' to lock",
                topic="LOCKED_STATE",
                category="HELP",
                provider="offline",
            )

        # 2. Supported gestures / What can I use / Mappings / Available Modes
        if re.search(r"\b(supported gestures?|which gestures?|what gestures?|gestures? can|list gestures?|available gestures?|commands? can|what modes?|which modes?|available modes?|list modes?)\b", q) or (
            re.search(r"\bgestures?\b", q) and re.search(r"\b(supported|available|list|mappings?)\b", q)
        ):
            print(f"[LLM Offline] Query \"{question}\" matched IntentSense offline rule: 2_SUPPORTED_GESTURES")
            if ctx == "MEDIA":
                ans = "In Media mode: OpenPalm to unlock. ThumbsUp raises volume, Fist lowers volume, and dynamic horizontal swipes change tracks."
            elif ctx == "PRESENTATION":
                ans = "In Presentation mode: OpenPalm to unlock. Pointing advances slides, and horizontal swipes navigate slides back and forward."
            elif ctx == "CUSTOM":
                if custom_gestures:
                    cg_list = ", ".join(f"'{g}'" for g in custom_gestures)
                    ans = f"In Custom mode, enrolled gestures ({cg_list}) execute their assigned actions. Use Manage (M) to review."
                else:
                    ans = "In Custom mode. No custom gestures enrolled yet; press 'E' to enroll your first custom hand pose."
            else:
                ans = "OpenPalm unlocks the system. Supported gestures vary by mode: Media mode (1), Presentation mode (2), and Custom mode (3)."
            return VoiceAssistantResponse(
                question=question,
                answer=ans,
                suggested_action="Press '1' for Media, '2' for Presentation",
                topic="SUPPORTED_GESTURES",
                category="HELP",
                provider="offline",
            )

        # 3. Media Mode questions
        if re.search(r"\b(media mode|music mode|playback mode)\b", q) or (
            re.search(r"\bmedia\b", q) and re.search(r"\b(mode|controls?|gesture|hand)\b", q)
        ):
            print(f"[LLM Offline] Query \"{question}\" matched IntentSense offline rule: 3_MEDIA_MODE")
            return VoiceAssistantResponse(
                question=question,
                answer="Media mode controls playback: ThumbsUp for Volume Up, Fist for Volume Down, Swipe Left for Previous Track, and Swipe Right for Next Track. Press '1' to select.",
                suggested_action="Press '1' to switch to Media mode",
                topic="MEDIA_MODE",
                category="HELP",
                provider="offline",
            )

        # 4. Presentation Mode questions
        if re.search(r"\b(presentation mode|slides? mode|next slide|previous slide)\b", q) or (
            re.search(r"\bpresentation\b", q) and re.search(r"\b(mode|controls?|gesture|work|slides?)\b", q)
        ):
            print(f"[LLM Offline] Query \"{question}\" matched IntentSense offline rule: 4_PRESENTATION_MODE")
            return VoiceAssistantResponse(
                question=question,
                answer="Presentation mode controls slides: Pointing holds advance slides, and horizontal swipes navigate back and forth. Press '2' to select.",
                suggested_action="Press '2' to switch to Presentation mode",
                topic="PRESENTATION_MODE",
                category="HELP",
                provider="offline",
            )

        # 5a. Safe Test Mode questions
        if re.search(r"\b(safe test(?: mode)?|test (?:gestures? )?safely|test custom gestures?)\b", q):
            print(f"[LLM Offline] Query \"{question}\" matched IntentSense offline rule: 5A_SAFE_TEST_MODE")
            return VoiceAssistantResponse(
                question=question,
                answer="Press 'T' to enter Safe Test mode. It lets you test your custom poses in real-time with live distance and confidence feedback without triggering any system actions.",
                suggested_action="Press 'T' for Safe Test mode",
                topic="SAFE_TEST_MODE",
                category="HELP",
                provider="offline",
            )

        # 5b. Custom Gesture Management questions
        if re.search(r"\b(manage (?:custom |personalized )?gestures?|manage (?:gesture )?profiles?|gesture manager|custom gesture manager|delete (?:a )?custom gesture|remap (?:custom )?actions?)\b", q) or (
            re.search(r"\bmanage\b", q) and re.search(r"\b(custom|gestures?|profiles?)\b", q)
        ):
            print(f"[LLM Offline] Query \"{question}\" matched IntentSense offline rule: 5B_CUSTOM_GESTURE_MANAGEMENT")
            return VoiceAssistantResponse(
                question=question,
                answer="Press 'M' to open Custom Gesture Management. You can review enrolled poses, remap actions, re-enroll takes, or press 'D' to delete profiles.",
                suggested_action="Press 'M' to manage custom gestures",
                topic="CUSTOM_GESTURE_MANAGEMENT",
                category="HELP",
                provider="offline",
            )

        # 5c. Custom Mode questions (e.g. "Is there any custom mode?", "Do you have custom mode?")
        if re.search(r"\b(custom mode|personalized mode)\b", q) or (
            re.search(r"\bcustom\b", q) and re.search(r"\b(mode|have custom|there any custom|is there custom|support custom)\b", q)
        ):
            print(f"[LLM Offline] Query \"{question}\" matched IntentSense offline rule: 5C_CUSTOM_MODE")
            return VoiceAssistantResponse(
                question=question,
                answer="IntentSense includes a dedicated Custom mode! Press '3' to enter Custom mode. Press 'E' to enroll your own hand poses, 'M' to manage profiles, and 'T' for Safe Test mode.",
                suggested_action="Press '3' for Custom mode, 'E' to enroll",
                topic="CUSTOM_MODE",
                category="HELP",
                provider="offline",
            )

        # 5d. Custom Gestures / Enrollment / Creating Own Gestures questions
        if re.search(r"\b(create (?:my |your |a )?own gestures?|make (?:my |your |a )?own gestures?|add (?:a )?(?:new )?custom gesture|enroll(?:ment)?|new gesture|train gesture|personalized gestures?|custom gestures?)\b", q):
            print(f"[LLM Offline] Query \"{question}\" matched IntentSense offline rule: 5D_CUSTOM_GESTURES")
            return VoiceAssistantResponse(
                question=question,
                answer="To enroll a custom gesture, press 'E', enter a unique name, and record 3 steady 10-frame takes. Press 'M' to manage existing profiles.",
                suggested_action="Press 'E' to enroll a gesture",
                topic="CUSTOM_GESTURES",
                category="HELP",
                provider="offline",
            )

        # 6. AI Coaching & Troubleshooting
        if re.search(r"\b(coach(?:ing)?|troubleshoot(?:ing)?|improve (?:gesture )?accuracy|gesture accuracy|rejection reason|why rejected|struggle with gestures?)\b", q):
            print(f"[LLM Offline] Query \"{question}\" matched IntentSense offline rule: 6_AI_COACHING")
            return VoiceAssistantResponse(
                question=question,
                answer="For best recognition, hold gestures steady for at least 0.5s (1.0s for OpenPalm). Keep your hand 1.5 to 3 feet away under good lighting. Press 'X' for deep explanations.",
                suggested_action="Press 'X' to view Decision Explainer",
                topic="AI_COACHING",
                category="HELP",
                provider="offline",
            )

        # 7. Touchless Onboarding & Practice questions
        if re.search(r"\b(onboard(?:ing)?|practice mode|tutorial mode|touchless onboarding|touchless practice|how to practice|practice gestures?)\b", q):
            print(f"[LLM Offline] Query \"{question}\" matched IntentSense offline rule: 7_TOUCHLESS_ONBOARDING")
            return VoiceAssistantResponse(
                question=question,
                answer="Press 'O' to enter Touchless Onboarding Practice Mode. It guides you step-by-step through unlocking, command gestures, and swipes entirely hands-free.",
                suggested_action="Press 'O' for Practice Mode",
                topic="TOUCHLESS_ONBOARDING",
                category="HELP",
                provider="offline",
            )

        # 8. Conversation reset command
        if re.search(r"\b(reset conversation|clear conversation|new conversation|start over|clear history|reset history)\b", q):
            print(f"[LLM Offline] Query \"{question}\" matched IntentSense offline rule: 8_CONVERSATION_RESET")
            return VoiceAssistantResponse(
                question=question,
                answer="Conversation history has been reset. What would you like to discuss next?",
                suggested_action="Ask any question to begin.",
                topic="CONVERSATION_RESET",
                category="CONTROL",
                provider="offline",
            )

        # 9. Empty or whitespace question
        if not q:
            print(f"[LLM Offline] Empty query detected.")
            return VoiceAssistantResponse(
                question=question,
                answer="I did not catch that. Say 'Hello Sense' and ask about gestures, modes, or unlocking.",
                suggested_action="Say 'Hello Sense' to ask a question",
                topic="EMPTY_QUERY",
                category="HELP",
                provider="offline",
            )

        # 10. Supported IntentSense General Help questions
        if re.search(r"\bintentsense\b", q) or re.search(r"\bhelp with (?:gestures?|the app|modes?)\b", q) or re.search(r"\bhow to use (?:intentsense|this app|the app)\b", q):
            print(f"[LLM Offline] Query \"{question}\" matched IntentSense offline rule: 10_GENERAL_HELP")
            return VoiceAssistantResponse(
                question=question,
                answer="IntentSense voice assistant is ready. You can ask about supported gestures, unlocking, Media/Presentation modes, custom gestures, or onboarding.",
                suggested_action="Ask: 'What gestures can I use?'",
                topic="GENERAL_HELP",
                category="HELP",
                provider="offline",
            )

        # 11. Unsupported General Knowledge fallback (requires active Gemini / internet connection)
        print(f"[LLM Offline] Query \"{question}\" did NOT match any IntentSense offline rule; routing to general-knowledge offline-unavailable response.")
        return VoiceAssistantResponse(
            question=question,
            answer="General knowledge answers require an internet connection. The online AI provider is currently unavailable.",
            suggested_action="Configure XAI_API_KEY to enable Grok for general knowledge answers.",
            topic="OFFLINE_UNAVAILABLE",
            category="OFFLINE",
            provider="offline",
        )


# ---------------------------------------------------------------------------
# Remote / Local LLM Provider (Google Gemini / OpenAI / Ollama compatible)
# ---------------------------------------------------------------------------
class LLMHttpProvider:
    """
    Communicates with Grok, legacy Gemini, and OpenAI-compatible endpoints
    using standard Python urllib. No additional SDK is required.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self._explicit_api_key = api_key
        self.endpoint_url = endpoint_url
        if model_name:
            self.model_name = model_name
        elif api_key is not None:
            # Explicit legacy/test keys retain the Gemini default.
            self.model_name = "gemini-1.5-flash"
        else:
            self.model_name = (
                self._read_setting("XAI_MODEL")
                or ("grok-4.7" if self._read_setting("XAI_API_KEY") else "gemini-1.5-flash")
            )
        self.timeout = timeout
        self.last_status: Optional[int] = None
        self.last_error: Optional[str] = None

    @staticmethod
    def _read_setting(name: str) -> str:
        """Reads a value from the environment or the private IntentSense user config."""
        value = os.environ.get(name, "")
        if value and value.strip():
            return value.strip().strip(chr(34) + chr(39))

        config_paths = []
        try:
            from core.paths import get_user_data_dir
            config_paths.append(os.path.join(get_user_data_dir(), ".env"))
        except ImportError:
            pass

        # Also support a gitignored project-root .env during development.
        project_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
        if project_env not in config_paths:
            config_paths.append(project_env)

        for config_path in config_paths:
            try:
                with open(config_path, "r", encoding="utf-8") as config_file:
                    for line in config_file:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, raw_value = line.split("=", 1)
                        if key.strip() == name:
                            return raw_value.strip().strip(chr(34) + chr(39))
            except OSError:
                continue
        return ""

    @property
    def _uses_xai(self) -> bool:
        """Selects Grok for the default provider configuration."""
        return self._explicit_api_key is None and bool(self._read_setting("XAI_API_KEY"))

    @property
    def api_key(self) -> str:
        """Returns an explicit key when supplied; otherwise Grok first, then Gemini."""
        if self._explicit_api_key is not None:
            raw_key = self._explicit_api_key
        else:
            raw_key = self._read_setting("XAI_API_KEY") or self._read_setting("GEMINI_API_KEY")
        return raw_key.strip().strip(chr(34) + chr(39))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key or self.endpoint_url)

    @property
    def name(self) -> str:
        """Identifies provider name for telemetry and HUD display."""
        if self.endpoint_url:
            return "custom_llm"
        if self._uses_xai:
            return "grok"
        return "gemini"

    def _sanitize_error_message(self, msg: str) -> str:
        """Sanitizes error text to strictly avoid exposing API keys or secrets."""
        if not msg:
            return ""
        s = str(msg)
        if self.api_key:
            s = s.replace(self.api_key, "[REDACTED_API_KEY]")
        s = re.sub(r"key=[A-Za-z0-9_\-]+", "key=[REDACTED]", s)
        s = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", s, flags=re.IGNORECASE)
        s = re.sub(r"\?key=[^\s&'\"]+", "?key=[REDACTED]", s)
        return s.strip()

    def generate_json(self, prompt: str, system_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Calls external LLM endpoint and parses response as a JSON dictionary.
        Returns None if request times out or network is unavailable.
        """
        self.last_status = None
        self.last_error = None

        if not self.is_configured:
            self.last_error = "No remote API key or custom endpoint configured"
            print("[LLM Http] Provider not configured (XAI_API_KEY/GEMINI_API_KEY missing and no endpoint URL).")
            return None

        # xAI Responses API (recommended by xAI for new integrations).
        # store=False avoids retaining the prompt and response on xAI servers.
        if not self.endpoint_url and self._uses_xai:
            url = "https://api.x.ai/v1/responses"
            payload = {
                "model": self.model_name,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "text": {"format": {"type": "json_object"}},
                "store": False,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        # Retain the Gemini path so existing installations can still use it.
        elif not self.endpoint_url and self.api_key:
            model_id = self.model_name.replace("models/", "")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": f"{system_prompt}\\n\\nUSER PROMPT: {prompt}"}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.2,
                },
            }
            headers = {"Content-Type": "application/json"}
        elif self.endpoint_url:
            # OpenAI / Ollama compatible endpoint
            url = self.endpoint_url
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            return None

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                http_status = getattr(resp, "status", 200)
                self.last_status = http_status
                self.last_error = None
                print(f"[LLM Http] {self.name.title()} HTTP status: {http_status} (Success)")
                body = resp.read().decode("utf-8")
                data = json.loads(body)

                # Parse xAI Responses API output_text content.
                if "output" in data and isinstance(data["output"], list):
                    raw_text = ""
                    for output_item in data["output"]:
                        for content_item in output_item.get("content", []):
                            if content_item.get("type") == "output_text":
                                raw_text += content_item.get("text", "")
                    if raw_text:
                        cleaned = _clean_json_markdown(raw_text)
                        try:
                            return json.loads(cleaned)
                        except json.JSONDecodeError:
                            if cleaned and not cleaned.startswith("{"):
                                return {"answer": cleaned, "suggested_action": "", "topic": "GENERAL", "category": "HELP"}
                            return None
                    self.last_error = "Grok response returned no output text"
                    return None

                # Parse Gemini response format
                if "candidates" in data and data["candidates"]:
                    candidate = data["candidates"][0]
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    if parts and "text" in parts[0]:
                        raw_text = parts[0]["text"]
                        cleaned = _clean_json_markdown(raw_text)
                        try:
                            return json.loads(cleaned)
                        except json.JSONDecodeError:
                            if cleaned and not cleaned.startswith("{"):
                                return {"answer": cleaned, "suggested_action": "", "topic": "GENERAL", "category": "HELP"}
                            return None
                    else:
                        finish_reason = candidate.get("finishReason", "UNKNOWN")
                        print(f"[LLM Http] {self.name.title()} candidate returned no parts. Finish reason: {finish_reason}")
                        self.last_error = f"{self.name.title()} finished without content: {finish_reason}"
                        return None
                # Parse OpenAI / Ollama response format
                elif "choices" in data and data["choices"]:
                    raw_text = data["choices"][0]["message"]["content"]
                    cleaned = _clean_json_markdown(raw_text)
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        if cleaned and not cleaned.startswith("{"):
                            return {"answer": cleaned, "suggested_action": "", "topic": "GENERAL", "category": "HELP"}
                        return None
                else:
                    print(f"[LLM Http] {self.name.title()} HTTP response returned no candidates or choices.")
                    self.last_error = f"{self.name.title()} response returned no candidates or choices."
                    return None
        except urllib.error.HTTPError as e:
            self.last_status = e.code
            raw_err_msg = ""
            try:
                err_bytes = e.read()
                err_data = json.loads(err_bytes.decode("utf-8", errors="replace"))
                if isinstance(err_data, dict) and "error" in err_data:
                    raw_err_msg = str(err_data["error"].get("message", e.reason))
                else:
                    raw_err_msg = str(e.reason)
            except Exception:
                raw_err_msg = str(e.reason)

            sanitized_msg = self._sanitize_error_message(raw_err_msg)
            self.last_error = f"HTTP {e.code}: {sanitized_msg}"
            print(f"[LLM Http] {self.name.title()} API HTTP error: status={e.code}, message=\"{sanitized_msg}\"")
            return None
        except urllib.error.URLError as e:
            reason_str = str(getattr(e, "reason", e))
            is_timeout = isinstance(getattr(e, "reason", None), (socket.timeout, TimeoutError)) or "timed out" in reason_str.lower()
            if is_timeout:
                self.last_status = 408
                self.last_error = f"Request timed out after {self.timeout:.1f}s"
                print(f"[LLM Http] {self.name.title()} API request timed out after {self.timeout:.1f}s.")
            else:
                self.last_status = None
                reason_type = type(e.reason).__name__ if hasattr(e, "reason") else type(e).__name__
                sanitized_reason = self._sanitize_error_message(str(e.reason))
                self.last_error = f"Network connection failed: {sanitized_reason}"
                print(f"[LLM Http] {self.name.title()} API connection error: {reason_type} ({sanitized_reason})")
            return None
        except (socket.timeout, TimeoutError):
            self.last_status = 408
            self.last_error = f"Request timed out after {self.timeout:.1f}s"
            print(f"[LLM Http] Gemini API request timed out after {self.timeout:.1f}s.")
            return None
        except Exception as e:
            self.last_status = None
            sanitized_err = self._sanitize_error_message(str(e))
            self.last_error = f"{type(e).__name__}: {sanitized_err}"
            print(f"[LLM Http] {self.name.title()} API unexpected error: {type(e).__name__} ({sanitized_err})")
            return None

        return None


# ---------------------------------------------------------------------------
# High-Level LLMAssistant Manager
# ---------------------------------------------------------------------------
class LLMAssistant:
    """
    Unified, fail-safe LLM Assistant for IntentSense.
    Features:
      - Asynchronous thread pool to never block real-time camera processing.
      - Seamless automatic fallback to OfflineRuleBasedProvider when offline or unconfigured.
      - Strict input sanitation (NEVER accepts frames or landmarks).
      - Strict output validation against ALLOWED_INTENT_VALUES and ALLOWED_ACTION_IDS.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        model_name: Optional[str] = None,
        max_workers: int = 2,
        http_provider: Optional[Any] = None,
        max_history_turns: int = 20,
        max_history_chars: int = 12000,
    ) -> None:
        self.offline_provider = OfflineRuleBasedProvider()
        self.http_provider = http_provider or LLMHttpProvider(
            api_key=api_key,
            endpoint_url=endpoint_url,
            model_name=model_name,
        )
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.max_history_turns = max_history_turns
        self.max_history_chars = max_history_chars
        self._conversation_history: List[Dict[str, str]] = []
        self._history_summary: Optional[str] = None
        self._history_lock = threading.RLock()

    @property
    def has_active_remote(self) -> bool:
        return self.http_provider.is_configured

    def reset_conversation_history(self) -> None:
        """Clears all multi-turn dialogue history and active context summaries."""
        with self._history_lock:
            self._conversation_history.clear()
            self._history_summary = None
            print("[LLM Assistant] Conversation history has been reset.")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Returns a snapshot of the current multi-turn conversation history."""
        with self._history_lock:
            return list(self._conversation_history)

    def _add_to_history(self, role: str, content: str) -> None:
        """Adds a dialogue turn and manages context window compaction."""
        with self._history_lock:
            self._conversation_history.append({"role": role, "content": content})
            self._trim_or_summarize_history()

    def _trim_or_summarize_history(self) -> None:
        """
        Maintains relevant conversation history within model context window limits.
        When conversation grows long, summarizes older turns while preserving recent turns.
        """
        with self._history_lock:
            total_chars = sum(len(t["content"]) for t in self._conversation_history)
            if len(self._conversation_history) <= self.max_history_turns and total_chars <= self.max_history_chars:
                return

            turns_to_summarize = self._conversation_history[:-6]
            recent_turns = self._conversation_history[-6:]

            if turns_to_summarize:
                summary_snippets = []
                for turn in turns_to_summarize:
                    prefix = "User" if turn["role"] == "user" else "Assistant"
                    snippet = turn["content"][:100].replace("\n", " ")
                    summary_snippets.append(f"{prefix}: {snippet}")
                new_summary = "Previous context: " + "; ".join(summary_snippets)
                if self._history_summary:
                    self._history_summary = f"{self._history_summary} | {new_summary}"[:1000]
                else:
                    self._history_summary = new_summary[:1000]

            self._conversation_history = recent_turns

    def close(self) -> None:
        """Safely shuts down the asynchronous worker pool."""
        self._executor.shutdown(wait=False)

    # -------------------------------------------------------------------------
    # 1. Intelligent Gesture & Rejection Explanation
    # -------------------------------------------------------------------------
    def explain_decision(
        self,
        decision_category: str,
        decision_text: str,
        state: str,
        context: str,
        raw_gesture: Optional[str] = None,
        confidence: float = 0.0,
        effective_threshold: float = 0.80,
        hold_duration: float = 0.0,
        target_hold: float = 0.5,
        is_swipe: bool = False,
        *args: Any,
    ) -> ExplanationResponse:
        """
        Produces human-readable coaching explanation for a decision event.
        NEVER takes frames or raw landmarks (enforced by _assert_no_vision_payload).
        """
        _assert_no_vision_payload(raw_gesture, *args)

        offline_res = self.offline_provider.explain_decision(
            decision_category=decision_category,
            decision_text=decision_text,
            state=state,
            context=context,
            raw_gesture=raw_gesture,
            confidence=confidence,
            effective_threshold=effective_threshold,
            hold_duration=hold_duration,
            target_hold=target_hold,
            is_swipe=is_swipe,
        )

        if not self.has_active_remote:
            return offline_res

        system_prompt = (
            "You are an expert touchless UI coach for IntentSense. "
            "Never invent a reason that is not directly supported by the event metadata. "
            "If information is insufficient, explicitly state that in the summary. "
            "Output JSON with keys: 'summary' (str), 'coaching_tip' (str), 'suggested_action' (str). "
            "Keep advice concise, encouraging, grounded in facts, and focused on hand posture, camera distance, and hold duration."
        )
        prompt = (
            f"Event Context:\n"
            f"- State: {state}\n"
            f"- Context: {context}\n"
            f"- Decision: {decision_text} ({decision_category})\n"
            f"- Gesture: {raw_gesture or 'None'}\n"
            f"- Confidence: {int(confidence*100)}% (Threshold: {int(effective_threshold*100)}%)\n"
            f"- Hold: {hold_duration:.2f}s (Target: {target_hold:.2f}s)\n"
        )

        try:
            remote_json = self.http_provider.generate_json(prompt, system_prompt)
        except Exception:
            remote_json = None
        if remote_json and "summary" in remote_json and "coaching_tip" in remote_json:
            return ExplanationResponse(
                summary=str(remote_json["summary"]),
                coaching_tip=str(remote_json["coaching_tip"]),
                suggested_action=str(remote_json.get("suggested_action", offline_res.suggested_action)),
                category=decision_category,
            )

        return offline_res

    # -------------------------------------------------------------------------
    # 2. Natural Language Gesture Configuration
    # -------------------------------------------------------------------------
    def parse_gesture_configuration(
        self,
        description: str,
        existing_custom_names: Sequence[str] = (),
    ) -> ConfigurationResponse:
        """
        Parses a natural-language description into a structured configuration.
        Validates against allowable actions, context, and reserved names.
        """
        _assert_no_vision_payload(description)

        offline_res = self.offline_provider.parse_gesture_configuration(
            description=description,
            existing_custom_names=existing_custom_names,
        )

        if not self.has_active_remote or not offline_res.is_valid:
            return offline_res

        system_prompt = (
            "You are a configuration assistant for IntentSense gesture mappings. "
            "Allowed actions: ['play_pause', 'next_track', 'previous_track', 'volume_up', 'volume_down', "
            "'start_stop_presentation', 'next_slide', 'previous_slide', 'zoom_in', 'zoom_out']. "
            "Allowed contexts: ['MEDIA', 'PRESENTATION', 'CUSTOM', 'ALL']. "
            "Known built-in gestures: ['OpenPalm', 'Fist', 'Peace', 'Pointing', 'ThumbsUp', 'ThumbsDown']. "
            "Output JSON with keys: 'gesture_name', 'action_id', 'context', 'confirmation_message'."
        )
        prompt = f"User request: '{description}'"
        remote_json = self.http_provider.generate_json(prompt, system_prompt)

        if remote_json:
            act_id = str(remote_json.get("action_id", "")).strip().lower()
            ctx_id = str(remote_json.get("context", "")).strip().upper()
            g_name = str(remote_json.get("gesture_name", "")).strip().replace(" ", "")

            # Strict validation against allowlists
            if act_id in ALLOWED_ACTION_IDS and ctx_id in ALLOWED_CONTEXTS and g_name:
                is_builtin = g_name in KNOWN_GESTURES
                needs_enrollment = (not is_builtin) and (g_name not in existing_custom_names)
                conf_msg = str(remote_json.get("confirmation_message", "")) or offline_res.confirmation_message

                return ConfigurationResponse(
                    gesture_name=g_name,
                    is_builtin=is_builtin,
                    action_id=act_id,
                    context=ctx_id,
                    needs_enrollment=needs_enrollment,
                    confirmation_message=conf_msg,
                    is_valid=True,
                )

        return offline_res

    # -------------------------------------------------------------------------
    # 3. Interactive AI Onboarding & Troubleshooting
    # -------------------------------------------------------------------------
    def get_onboarding_guide(self, topic: str) -> OnboardingStep:
        """Provides structured onboarding steps and troubleshooting tips."""
        _assert_no_vision_payload(topic)
        return self.offline_provider.get_onboarding_guide(topic)

    # -------------------------------------------------------------------------
    # 4. Natural Language Typed Commands
    # -------------------------------------------------------------------------
    def parse_natural_command(self, query: str, active_context: Context) -> NaturalCommandResponse:
        """
        Translates a natural language command string into a verified Intent.
        Strictly enforces the Intent allowlist and safety validation.
        """
        _assert_no_vision_payload(query)

        offline_res = self.offline_provider.parse_natural_command(query, active_context)
        if not self.has_active_remote:
            return offline_res

        system_prompt = (
            "You are an intent parser for touchless OS control. "
            "Translate the user's natural language command into exactly one of these allowed Intents: "
            "['PLAY_PAUSE', 'PREVIOUS', 'NEXT', 'VOLUME_UP', 'VOLUME_DOWN', 'START_STOP', "
            "'PREVIOUS_SLIDE', 'NEXT_SLIDE', 'ZOOM_IN', 'ZOOM_OUT']. "
            "Output JSON with keys: 'intent' (str or null), 'context' ('MEDIA' or 'PRESENTATION'), "
            "'explanation' (str)."
        )
        prompt = f"Active context: {active_context.value}\nUser command: '{query}'"
        try:
            remote_json = self.http_provider.generate_json(prompt, system_prompt)
        except Exception:
            remote_json = None

        if remote_json:
            intent_val = remote_json.get("intent")
            if intent_val and str(intent_val) in ALLOWED_INTENT_VALUES:
                matched_intent = Intent(str(intent_val))
                ctx_val = remote_json.get("context", active_context.value)
                matched_ctx = Context(ctx_val) if ctx_val in [c.value for c in Context] else active_context
                return NaturalCommandResponse(
                    intent=matched_intent,
                    context=matched_ctx,
                    confidence=0.98,
                    explanation=str(remote_json.get("explanation", "Parsed command via remote LLM.")),
                    is_valid=True,
                    raw_query=query,
                )

        return offline_res

    # -------------------------------------------------------------------------
    # 5. Personalized Profile Assistance & Diagnostics
    # -------------------------------------------------------------------------
    def get_personalized_recommendations(
        self,
        profiles_summary: List[Dict[str, Any]],
    ) -> PersonalizedSuggestionResponse:
        """
        Analyzes custom gesture dispersion and suggests ergonomic improvements.
        """
        return self.offline_provider.get_personalized_recommendations(profiles_summary)

    def coach_struggle(
        self,
        pattern: StrugglePattern,
        context: Context = Context.MEDIA,
    ) -> ExplanationResponse:
        """
        Synthesizes an intelligent coaching response for an identified physical struggle.
        Strictly grounds all advice in the provided telemetry evidence.
        """
        _assert_no_vision_payload(pattern.evidence, context)

        offline_res = self.offline_provider.coach_struggle(pattern, context)
        if not self.has_active_remote:
            return offline_res

        system_prompt = (
            "You are an expert touchless gesture coach for IntentSense. "
            "A user is struggling with a physical gesture. "
            "Never invent reasons not supported by the telemetry evidence. "
            "Output JSON with keys: 'summary' (str, <= 8 words), 'coaching_tip' (str, <= 18 words), 'suggested_action' (str, <= 8 words). "
            "Focus on physical hand posture, distance, lighting, and hold duration."
        )
        prompt = (
            f"Struggle Type: {pattern.pattern_type.value}\n"
            f"Gesture: {pattern.gesture or 'None'}\n"
            f"Evidence: {pattern.evidence}\n"
            f"Context: {context.value}\n"
            f"Metrics: {pattern.metrics}"
        )

        try:
            remote_json = self.http_provider.generate_json(prompt, system_prompt)
        except Exception:
            remote_json = None

        if remote_json and "summary" in remote_json and "coaching_tip" in remote_json:
            return ExplanationResponse(
                summary=str(remote_json["summary"]),
                coaching_tip=str(remote_json["coaching_tip"]),
                suggested_action=str(remote_json.get("suggested_action", offline_res.suggested_action)),
                category=pattern.pattern_type.value,
                provider=self.http_provider.name,
            )

        return offline_res

    # -------------------------------------------------------------------------
    # Non-blocking Asynchronous Helper
    # -------------------------------------------------------------------------
    def query_async(
        self,
        fn: Callable[..., Any],
        *args: Any,
        callback: Optional[Callable[[Any], None]] = None,
        **kwargs: Any,
    ) -> concurrent.futures.Future:
        """
        Dispatches a query to the background thread pool so OpenCV never blocks.
        Invokes callback(result) on completion if provided.
        """
        future = self._executor.submit(fn, *args, **kwargs)
        if callback is not None:
            def _done_callback(fut: concurrent.futures.Future) -> None:
                try:
                    res = fut.result()
                    callback(res)
                except Exception:
                    try:
                        callback(None)
                    except Exception:
                        pass
            future.add_done_callback(_done_callback)
        return future

    # -------------------------------------------------------------------------
    # Phase 6: Voice Question Answering
    # -------------------------------------------------------------------------
    def answer_voice_question(
        self,
        question: str,
        app_state_summary: Optional[Dict[str, Any]] = None,
    ) -> VoiceAssistantResponse:
        """
        Answers a user's spoken question about IntentSense, general knowledge, or technical topics
        using the configured remote provider (Grok preferred) with conversation history and offline fallback.
        Ensures zero-audio and zero-vision data transfer.
        """
        _assert_no_vision_payload(question, app_state_summary)
        _assert_no_audio_payload(question, app_state_summary)

        # Check for explicit conversation reset commands
        q_lower = (question or "").strip().lower()
        if q_lower in ("reset conversation", "clear conversation", "new conversation", "start over", "clear history", "reset history"):
            self.reset_conversation_history()
            return VoiceAssistantResponse(
                question=question,
                answer="Conversation history has been reset. What would you like to discuss next?",
                suggested_action="Ask any question to begin.",
                topic="CONVERSATION_RESET",
                category="CONTROL",
                provider="assistant_control",
            )

        is_key_configured = bool(getattr(self.http_provider, "api_key", False) or getattr(self.http_provider, "is_configured", False))
        selected_provider = self.http_provider.name.title() if self.has_active_remote else "Offline"
        key_label = "XAI_API_KEY" if selected_provider == "Grok" else "GEMINI_API_KEY"
        print(f"[LLM Assistant] {key_label} configured: {is_key_configured}")
        print(f"[LLM Assistant] Selected provider: {selected_provider}")

        if not self.has_active_remote:
            resp = self.offline_provider.answer_voice_question(question, app_state_summary)
            self._add_to_history("user", question)
            self._add_to_history("assistant", resp.answer)
            return resp

        summary = app_state_summary or {}
        state_str = json.dumps(summary, indent=2)

        sys_prompt = (
            "You are IntentSense AI Voice Assistant — a highly capable, knowledgeable, and friendly multimodal AI companion.\n"
            "You can engage in natural, open-ended conversations, answer general-purpose questions, discuss technical topics, "
            "and provide intelligent guidance about the IntentSense hand-gesture system and general knowledge.\n\n"
            "GUIDELINES:\n"
            "1. By default, keep spoken responses natural, direct, and concise (typically 2-4 conversational sentences) so they are comfortable to listen to.\n"
            "2. When the user asks for more detail, in-depth explanations, steps, code, analysis, or technical discussions, provide full, thorough, and detailed answers.\n"
            "3. If the user asks about IntentSense features, gestures, or application state, reference the current state provided below.\n"
            "4. For safety, voice interaction is for assistance and discussion. Do not simulate direct destructive OS shell execution.\n"
            "5. Return strict JSON matching: {\"answer\": string, \"suggested_action\": string, \"topic\": string, \"category\": string}."
        )

        # Build prompt with conversation history
        with self._history_lock:
            history_blocks = []
            if self._history_summary:
                history_blocks.append(f"[CONVERSATION SUMMARY]:\n{self._history_summary}")
            if self._conversation_history:
                history_blocks.append("[RECENT CONVERSATION HISTORY]:")
                for turn in self._conversation_history:
                    role_tag = "User" if turn["role"] == "user" else "Assistant"
                    history_blocks.append(f"{role_tag}: {turn['content']}")
            history_text = "\n".join(history_blocks)

        user_prompt_parts = []
        if history_text:
            user_prompt_parts.append(history_text + "\n")
        user_prompt_parts.append(f"Current User Question: \"{question}\"")
        user_prompt_parts.append(f"\nCurrent Application State:\n{state_str}")
        user_prompt_parts.append("\nRespond to the user with helpful, context-aware assistance.")
        user_prompt = "\n".join(user_prompt_parts)

        try:
            remote_json = self.http_provider.generate_json(user_prompt, sys_prompt)
            if remote_json and isinstance(remote_json, dict) and "answer" in remote_json:
                ans_text = str(remote_json["answer"])
                resp = VoiceAssistantResponse(
                    question=question,
                    answer=ans_text,
                    suggested_action=str(remote_json.get("suggested_action", "")),
                    topic=str(remote_json.get("topic", "GENERAL")),
                    category=str(remote_json.get("category", "HELP")),
                    provider=self.http_provider.name,
                )
                print(f"[LLM Assistant] Parsed {selected_provider} response: answer=\"{ans_text}\", topic=\"{resp.topic}\", category=\"{resp.category}\"")
                self._add_to_history("user", question)
                self._add_to_history("assistant", ans_text)
                return resp
            else:
                print(f"[LLM Assistant] {selected_provider} response missing valid answer; evaluating fallback.")
        except Exception as e:
            print(f"[LLM Assistant] Exception querying {selected_provider}: {type(e).__name__}; evaluating fallback.")

        # Graceful fallback: check if offline provider has a dedicated IntentSense answer
        offline_resp = self.offline_provider.answer_voice_question(question, app_state_summary)
        if offline_resp.topic != "OFFLINE_UNAVAILABLE":
            print(f"[LLM Assistant] Graceful fallback to local IntentSense rule (topic={offline_resp.topic}).")
            self._add_to_history("user", question)
            self._add_to_history("assistant", offline_resp.answer)
            return offline_resp

        # A remote provider was requested but failed: surface sanitized diagnostic reason
        last_status = getattr(self.http_provider, "last_status", None)
        last_error = getattr(self.http_provider, "last_error", None)

        provider_label = "Grok" if self.http_provider.name == "grok" else "Gemini" if self.http_provider.name == "gemini" else "Remote AI"
        api_key_name = "XAI_API_KEY" if self.http_provider.name == "grok" else "GEMINI_API_KEY"
        if last_status in (401, 403):
            err_answer = f"{provider_label} API authentication failed (HTTP {last_status}). Please verify that {api_key_name} is valid."
            err_action = f"Check your {api_key_name} configuration."
        elif last_status == 404:
            err_answer = f"{provider_label} model endpoint not found (HTTP 404). Please verify the model configuration."
            err_action = "Check the configured model name in IntentSense."
        elif last_status == 429:
            err_answer = f"{provider_label} API rate limit or quota exceeded (HTTP 429). Please wait a moment and try again."
            err_action = "Wait for quota refresh or verify your API account credits."
        elif last_status == 408 or (last_error and "timed out" in last_error.lower()):
            err_answer = f"{provider_label} API request timed out. Please check your network connection and try again."
            err_action = "Check network connection or try asking again."
        elif last_status:
            err_answer = f"{provider_label} API returned HTTP {last_status}. {last_error or ''}".strip()
            err_action = f"Check {provider_label} API status and connection."
        else:
            err_answer = f"Unable to reach {provider_label} API. {last_error or 'Please check your internet connection.'}".strip()
            err_action = "Check internet connection."


        diag_resp = VoiceAssistantResponse(
            question=question,
            answer=err_answer,
            suggested_action=err_action,
            topic=f"{self.http_provider.name.upper()}_API_ERROR",
            category="ERROR",
            provider=f"{self.http_provider.name}_error",
        )
        print(f"[LLM Assistant] {provider_label} API failure diagnostic response: \"{err_answer}\" (topic={diag_resp.topic})")
        self._add_to_history("user", question)
        self._add_to_history("assistant", diag_resp.answer)
        return diag_resp

    def answer_voice_question_async(
        self,
        question: str,
        app_state_summary: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[VoiceAssistantResponse], None]] = None,
    ) -> concurrent.futures.Future:
        """
        Dispatches voice question answering to the background thread pool.
        Never blocks the camera loop or HUD rendering.
        """
        return self.query_async(
            self.answer_voice_question,
            question,
            app_state_summary,
            callback=callback,
        )
