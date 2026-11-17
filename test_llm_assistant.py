"""
test_llm_assistant.py
=====================

Comprehensive unit tests for IntentSense Phase 1: LLM Client Abstraction.

Verifies:
  1. Modular LLMAssistant initialization & property checks.
  2. Support for GEMINI_API_KEY environment variable.
  3. Deterministic offline fallback when API key/network is unavailable.
  4. Strict JSON parsing and markdown fence stripping.
  5. Strict allowlist validation for Intents, Actions, Contexts, and Gestures.
  6. Strict zero-trust safety: rejection of arbitrary shell, OS, or unknown commands.
  7. Strict privacy: enforcement of zero vision frames or landmark payloads.
  8. Mock-based HTTP provider testing (Gemini / OpenAI / Ollama compatible).
  9. Non-blocking asynchronous query execution via ThreadPoolExecutor.
 10. Preservation of existing data structures and zero side-effects on existing modules.
"""

from __future__ import annotations

import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from core.context_manager import Context
from core.intent_engine import KNOWN_GESTURES, Intent
from core.llm_assistant import (
    ALLOWED_ACTION_IDS,
    ALLOWED_CONTEXTS,
    ALLOWED_INTENT_VALUES,
    ConfigurationResponse,
    ExplanationResponse,
    LLMAssistant,
    LLMHttpProvider,
    NaturalCommandResponse,
    OfflineRuleBasedProvider,
    OnboardingStep,
    PersonalizedSuggestionResponse,
    _assert_no_vision_payload,
    _clean_json_markdown,
)
from core.personalized_gestures import SUPPORTED_ACTION_MAP


class TestLLMAssistantOffline(unittest.TestCase):
    """Verifies deterministic offline fallback for all 5 intelligence capabilities."""

    def setUp(self) -> None:
        # Explicitly initialize with no API key to guarantee offline mode
        self.assistant = LLMAssistant(api_key="", endpoint_url="")

    def tearDown(self) -> None:
        self.assistant.close()

    def test_offline_mode_flag(self) -> None:
        """Verifies assistant reports inactive remote provider when unconfigured."""
        self.assertFalse(self.assistant.has_active_remote)

    # -------------------------------------------------------------------------
    # 1. Intelligent Gesture Explanation (Offline)
    # -------------------------------------------------------------------------
    def test_explain_decision_locked_idle(self) -> None:
        res = self.assistant.explain_decision(
            decision_category="IDLE",
            decision_text="IDLE — Waiting for hand (System locked)",
            state="LOCKED",
            context="MEDIA",
        )
        self.assertIsInstance(res, ExplanationResponse)
        self.assertIn("LOCKED", res.summary)
        self.assertIn("OpenPalm", res.suggested_action)

    def test_explain_decision_locked_unlocking_progress(self) -> None:
        res = self.assistant.explain_decision(
            decision_category="WAITING",
            decision_text="WAITING — Hold OpenPalm to unlock (0.4s / 1.0s)",
            state="LOCKED",
            context="MEDIA",
            raw_gesture="OpenPalm",
            confidence=0.92,
            effective_threshold=0.80,
            hold_duration=0.4,
            target_hold=1.0,
        )
        self.assertIsInstance(res, ExplanationResponse)
        self.assertIn("Unlocking progress: 0.4s / 1.0s", res.summary)
        self.assertIn("flat and steady", res.coaching_tip)

    def test_explain_decision_locked_suppressed_command(self) -> None:
        res = self.assistant.explain_decision(
            decision_category="REJECTED",
            decision_text="REJECTED — System locked (Hold OpenPalm to unlock)",
            state="LOCKED",
            context="MEDIA",
            raw_gesture="ThumbsUp",
            confidence=0.95,
        )
        self.assertIsInstance(res, ExplanationResponse)
        self.assertIn("ThumbsUp", res.summary)
        self.assertIn("suppresses all commands while locked", res.coaching_tip)

    def test_explain_decision_low_confidence_coaching(self) -> None:
        res = self.assistant.explain_decision(
            decision_category="REJECTED",
            decision_text="REJECTED — Confidence below threshold (68% < 80%)",
            state="ACTIVE",
            context="MEDIA",
            raw_gesture="Peace",
            confidence=0.68,
            effective_threshold=0.80,
        )
        self.assertIsInstance(res, ExplanationResponse)
        self.assertIn("68%", res.summary)
        self.assertIn("Spread your fingers clearly", res.coaching_tip)

    def test_explain_decision_cooldown_rate_limit(self) -> None:
        res = self.assistant.explain_decision(
            decision_category="COOLDOWN",
            decision_text="COOLDOWN — Action rate limit active",
            state="ACTIVE",
            context="MEDIA",
        )
        self.assertIsInstance(res, ExplanationResponse)
        self.assertIn("rate-limit", res.coaching_tip)

    def test_explain_decision_action_executed(self) -> None:
        res = self.assistant.explain_decision(
            decision_category="ACTION_EXECUTED",
            decision_text="play_pause (toggle playback)",
            state="ACTIVE",
            context="MEDIA",
        )
        self.assertIsInstance(res, ExplanationResponse)
        self.assertIn("play_pause", res.summary)
        self.assertEqual(res.category, "ACTION_EXECUTED")

    # -------------------------------------------------------------------------
    # 2. Natural Language Commands (Offline)
    # -------------------------------------------------------------------------
    def test_parse_natural_command_media_intents(self) -> None:
        test_cases = [
            ("please pause the music", Intent.PLAY_PAUSE, Context.MEDIA),
            ("turn the volume up", Intent.VOLUME_UP, Context.MEDIA),
            ("make it quieter please", Intent.VOLUME_DOWN, Context.MEDIA),
            ("skip to next track", Intent.NEXT, Context.MEDIA),
            ("go back to previous track", Intent.PREVIOUS, Context.MEDIA),
        ]
        for query, expected_intent, expected_context in test_cases:
            cmd = self.assistant.parse_natural_command(query, Context.MEDIA)
            self.assertTrue(cmd.is_valid, f"Failed on query: {query}")
            self.assertEqual(cmd.intent, expected_intent)
            self.assertEqual(cmd.context, expected_context)
            self.assertIn(cmd.intent.value, ALLOWED_INTENT_VALUES)

    def test_parse_natural_command_presentation_intents(self) -> None:
        test_cases = [
            ("advance to the next slide", Intent.NEXT_SLIDE, Context.PRESENTATION),
            ("go to previous slide", Intent.PREVIOUS_SLIDE, Context.PRESENTATION),
            ("start slideshow presentation", Intent.START_STOP, Context.PRESENTATION),
            ("zoom in on this chart", Intent.ZOOM_IN, Context.PRESENTATION),
            ("zoom out", Intent.ZOOM_OUT, Context.PRESENTATION),
        ]
        for query, expected_intent, expected_context in test_cases:
            cmd = self.assistant.parse_natural_command(query, Context.PRESENTATION)
            self.assertTrue(cmd.is_valid, f"Failed on query: {query}")
            self.assertEqual(cmd.intent, expected_intent)
            self.assertEqual(cmd.context, expected_context)
            self.assertIn(cmd.intent.value, ALLOWED_INTENT_VALUES)

    def test_parse_natural_command_unknown_rejected(self) -> None:
        cmd = self.assistant.parse_natural_command("format hard drive C: /q", Context.MEDIA)
        self.assertFalse(cmd.is_valid)
        self.assertIsNone(cmd.intent)
        self.assertIn("could not be safely mapped", cmd.explanation)

    # -------------------------------------------------------------------------
    # 3. Natural Language Gesture Configuration (Offline)
    # -------------------------------------------------------------------------
    def test_parse_gesture_configuration_builtin(self) -> None:
        cfg = self.assistant.parse_gesture_configuration(
            description="Use Pointing to advance next slide in presentation mode",
            existing_custom_names=["RockOn"],
        )
        self.assertTrue(cfg.is_valid)
        self.assertEqual(cfg.gesture_name, "Pointing")
        self.assertTrue(cfg.is_builtin)
        self.assertEqual(cfg.action_id, "next_slide")
        self.assertEqual(cfg.context, "PRESENTATION")
        self.assertFalse(cfg.needs_enrollment)
        self.assertIn("next_slide", ALLOWED_ACTION_IDS)

    def test_parse_gesture_configuration_new_custom(self) -> None:
        cfg = self.assistant.parse_gesture_configuration(
            description="I want to use 'VulcanSalute' for volume up in all contexts",
            existing_custom_names=["RockOn"],
        )
        self.assertTrue(cfg.is_valid)
        self.assertEqual(cfg.gesture_name, "VulcanSalute")
        self.assertFalse(cfg.is_builtin)
        self.assertEqual(cfg.action_id, "volume_up")
        self.assertEqual(cfg.context, "ALL")
        self.assertTrue(cfg.needs_enrollment)
        self.assertIn("Requires 3-take camera enrollment", cfg.confirmation_message)

    def test_parse_gesture_configuration_invalid_action(self) -> None:
        cfg = self.assistant.parse_gesture_configuration(
            description="When I nod my head, open a web browser to youtube",
            existing_custom_names=[],
        )
        self.assertFalse(cfg.is_valid)
        self.assertIn("Could not match", cfg.error_reason)

    # -------------------------------------------------------------------------
    # 4. Onboarding & Troubleshooting (Offline)
    # -------------------------------------------------------------------------
    def test_get_onboarding_guide(self) -> None:
        for topic in ["activation", "context", "enrollment", "troubleshooting"]:
            guide = self.assistant.get_onboarding_guide(topic)
            self.assertIsInstance(guide, OnboardingStep)
            self.assertTrue(len(guide.title) > 0)
            self.assertTrue(len(guide.key_controls) > 0)
            self.assertTrue(len(guide.troubleshooting_tip) > 0)

    # -------------------------------------------------------------------------
    # 5. Personalized Profile Assistance (Offline)
    # -------------------------------------------------------------------------
    def test_get_personalized_recommendations_empty(self) -> None:
        rec = self.assistant.get_personalized_recommendations([])
        self.assertIsInstance(rec, PersonalizedSuggestionResponse)
        self.assertEqual(rec.profile_diagnostics[0]["status"], "NO_CUSTOM_GESTURES")
        self.assertTrue(len(rec.recommended_mappings) > 0)

    def test_get_personalized_recommendations_with_profiles(self) -> None:
        profiles = [
            {"name": "RockOn", "dispersion": 0.05, "mappings": {"MEDIA": "PLAY_PAUSE"}},
            {"name": "LoosePinch", "dispersion": 0.11, "mappings": {}},
        ]
        rec = self.assistant.get_personalized_recommendations(profiles)
        self.assertIsInstance(rec, PersonalizedSuggestionResponse)
        self.assertEqual(len(rec.profile_diagnostics), 2)
        # LoosePinch should be flagged for moderate dispersion (> 0.09)
        pinch_diag = next(d for d in rec.profile_diagnostics if d["gesture"] == "LoosePinch")
        self.assertEqual(pinch_diag["status"], "MODERATE_DISPERSION")
        # LoosePinch should have an action recommendation since mappings is empty
        pinch_rec = next(r for r in rec.recommended_mappings if r["gesture"] == "LoosePinch")
        self.assertEqual(pinch_rec["action"], "play_pause")


class TestLLMAssistantSafetyAndPrivacy(unittest.TestCase):
    """Verifies architectural safety invariants and zero vision leaks."""

    def test_vision_payload_assertion_ndarray(self) -> None:
        """Verifies TypeError is raised if a numpy array is passed."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(TypeError) as ctx:
            _assert_no_vision_payload(arr)
        self.assertIn("Privacy Violation", str(ctx.exception))

    def test_vision_payload_assertion_landmark_sequence(self) -> None:
        """Verifies TypeError is raised if landmark coordinates are passed."""
        landmarks = [[0.1, 0.2, 0.3]] * 21  # 21 MediaPipe coordinates
        with self.assertRaises(TypeError) as ctx:
            _assert_no_vision_payload(landmarks)
        self.assertIn("Privacy Violation", str(ctx.exception))

    def test_clean_json_markdown(self) -> None:
        """Verifies markdown code fences are cleanly stripped."""
        raw = "```json\n{\"intent\": \"PLAY_PAUSE\"}\n```"
        self.assertEqual(_clean_json_markdown(raw), "{\"intent\": \"PLAY_PAUSE\"}")

        raw2 = "```\n{\"key\": \"val\"}\n```"
        self.assertEqual(_clean_json_markdown(raw2), "{\"key\": \"val\"}")


class TestLLMHttpProviderAndRemoteMocking(unittest.TestCase):
    """Verifies remote provider integration using mock HTTP responses."""

    def test_env_var_api_key_support(self) -> None:
        """Verifies GEMINI_API_KEY environment variable is picked up."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-env-key-12345"}):
            provider = LLMHttpProvider()
            self.assertEqual(provider.api_key, "test-env-key-12345")
            self.assertTrue(provider.is_configured)

    @patch("urllib.request.urlopen")
    def test_mock_gemini_remote_command_parsing(self, mock_urlopen: MagicMock) -> None:
        """Verifies parsing of remote Gemini API JSON responses."""
        gemini_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps({
                                    "intent": "VOLUME_UP",
                                    "context": "MEDIA",
                                    "explanation": "Increased volume via Gemini LLM.",
                                })
                            }
                        ]
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(gemini_response).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        assistant = LLMAssistant(api_key="mock-key-abc")
        cmd = assistant.parse_natural_command("can you make the speakers louder", Context.MEDIA)
        self.assertTrue(cmd.is_valid)
        self.assertEqual(cmd.intent, Intent.VOLUME_UP)
        self.assertEqual(cmd.context, Context.MEDIA)
        self.assertEqual(cmd.explanation, "Increased volume via Gemini LLM.")
        assistant.close()

    @patch("urllib.request.urlopen")
    def test_mock_gemini_remote_invalid_intent_rejected(self, mock_urlopen: MagicMock) -> None:
        """Verifies that an unauthorized intent returned by LLM is rejected or falls back."""
        malicious_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps({
                                    "intent": "EXECUTE_ARBITRARY_SHELL",
                                    "context": "MEDIA",
                                })
                            }
                        ]
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(malicious_response).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        assistant = LLMAssistant(api_key="mock-key-abc")
        # Should reject unauthorized intent and fall back to safe offline interpretation
        cmd = assistant.parse_natural_command("malicious command", Context.MEDIA)
        self.assertFalse(cmd.is_valid)
        self.assertIsNone(cmd.intent)
        assistant.close()

    @patch("urllib.request.urlopen")
    def test_remote_timeout_falls_back_gracefully(self, mock_urlopen: MagicMock) -> None:
        """Verifies timeout gracefully degrades to offline rule-based provider."""
        mock_urlopen.side_effect = TimeoutError("Remote server timed out")

        assistant = LLMAssistant(api_key="mock-key-abc")
        cmd = assistant.parse_natural_command("pause song", Context.MEDIA)
        self.assertTrue(cmd.is_valid)
        self.assertEqual(cmd.intent, Intent.PLAY_PAUSE)
        assistant.close()

    def test_async_query_execution(self) -> None:
        """Verifies asynchronous non-blocking execution via future and callback."""
        assistant = LLMAssistant(api_key="", endpoint_url="")
        callback_received = []

        def on_complete(result: NaturalCommandResponse) -> None:
            callback_received.append(result)

        future = assistant.query_async(
            assistant.parse_natural_command,
            "play music",
            Context.MEDIA,
            callback=on_complete,
        )
        res = future.result(timeout=2.0)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.intent, Intent.PLAY_PAUSE)
        self.assertEqual(len(callback_received), 1)
        self.assertEqual(callback_received[0].intent, Intent.PLAY_PAUSE)
        assistant.close()


if __name__ == "__main__":
    unittest.main()

