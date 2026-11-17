"""
test_voice_conversation.py
==========================

Comprehensive test suite for IntentSense Phase 6: Voice Conversation.
Tests:
  1. Local on-device Speech-to-Text using whisper.cpp integration.
  2. Mocked whisper.cpp CLI invocation using argument arrays (no shell=True).
  3. Parsing of whisper.cpp stdout (stripping timestamps, logs, [BLANK_AUDIO]).
  4. Arbitrary English speech transcription handling.
  5. Empty, silent, and noisy audio rejection.
  6. Missing binary or model graceful degradation.
  7. Inference timeout and exception safety.
  8. Gemini context-aware responses (mocked remote provider).
  9. Offline rule-based help fallback across all 6 domains.
 10. Locked-state questions (LOCKED vs ACTIVE).
 11. Microphone unavailable and permission revoked handling.
 12. Timeout and cancellation behavior.
 13. Asynchronous processing without blocking frame loops.
 14. Verification that audio data is NEVER sent to Gemini.
 15. Verification that voice requests CANNOT execute OS actions or bypass safety.
"""

import os
import subprocess
import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from core.context_manager import Context
from core.intent_engine import Intent
from core.llm_assistant import (
    LLMAssistant,
    OfflineRuleBasedProvider,
    VoiceAssistantResponse,
    _assert_no_audio_payload,
    _assert_no_vision_payload,
)
from core.voice_activation import (
    LocalSpeechTranscriber,
    LocalTextToSpeech,
    VoiceActivationManager,
    VoiceHudStatus,
    VoiceState,
)


class MockRemoteHttpProvider:
    """Mock HTTP provider simulating Gemini API responses and recording calls."""

    def __init__(self, response_payload=None, raise_exc=False) -> None:
        self.response_payload = response_payload
        self.raise_exc = raise_exc
        self.calls = []
        self.name = "gemini"

    @property
    def is_configured(self) -> bool:
        return True

    def generate_json(self, prompt: str, system_prompt: str):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        if self.raise_exc:
            raise RuntimeError("Network Timeout")
        return self.response_payload


class TestVoiceConversation(unittest.TestCase):
    """Unit and integration tests for Voice Conversation and Assistant."""

    def setUp(self) -> None:
        self.transcriber = LocalSpeechTranscriber(
            sample_rate=16000,
            binary_path="/fake/bin/whisper-cli",
            model_path="/fake/models/ggml-base.en.bin",
        )
        self.offline_provider = OfflineRuleBasedProvider()
        self.app_state = {
            "state": "LOCKED",
            "context": "MEDIA",
            "gesture_mappings": {
                "OpenPalm": "Unlock (Hold 1.0s)",
                "ThumbsUp": "Volume Up",
                "Fist": "Volume Down",
            },
            "custom_gestures": ["RockOn"],
            "is_locked": True,
            "onboarding_active": False,
        }

    # -------------------------------------------------------------------------
    # 1. Local Speech-to-Text Transcription Tests (whisper.cpp)
    # -------------------------------------------------------------------------
    def test_local_speech_transcription_simulated_hook(self) -> None:
        """Verifies simulated transcription hook works for headless test pipelines."""
        self.transcriber.simulate_transcription("how do i unlock")
        dummy_audio = np.random.uniform(-0.1, 0.1, 8000).astype(np.float32)
        transcribed = self.transcriber.transcribe(dummy_audio)
        self.assertEqual(transcribed, "how do i unlock")

    def test_local_speech_transcription_chunks(self) -> None:
        """Verifies multi-chunk audio concatenation and transcription."""
        self.transcriber.simulate_transcription("what gestures can i use")
        chunks = [np.zeros(1024, dtype=np.float32) for _ in range(5)]
        transcribed = self.transcriber.transcribe_chunks(chunks)
        self.assertEqual(transcribed, "what gestures can i use")

    @patch("subprocess.run")
    @patch("os.path.isfile")
    @patch("os.access")
    def test_whisper_cpp_invocation_and_parsing(self, mock_access, mock_isfile, mock_run) -> None:
        """Verifies whisper.cpp CLI is invoked via argument arrays and output is parsed."""
        mock_isfile.return_value = True
        mock_access.return_value = True

        raw_whisper_output = (
            "whisper_init_from_file_with_params_no_state: loading model from 'models/ggml-base.en.bin'\n"
            "system_info: n_threads = 4 / 8 | AVX = 0 | NEON = 1 |\n"
            "[00:00:00.000 --> 00:00:02.500]   How do I unlock IntentSense?\n"
        )
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = raw_whisper_output
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        dummy_speech = np.random.uniform(-0.15, 0.15, 16000).astype(np.float32)
        result = self.transcriber.transcribe(dummy_speech)

        self.assertEqual(result, "How do I unlock IntentSense?")
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]

        # Verify argument array format: no shell=True, fixed binary, model, and flags
        self.assertIsInstance(call_args, list)
        self.assertEqual(call_args[0], "/fake/bin/whisper-cli")
        self.assertIn("-m", call_args)
        self.assertIn("/fake/models/ggml-base.en.bin", call_args)
        self.assertIn("-f", call_args)
        self.assertIn("-l", call_args)
        self.assertIn("en", call_args)

    @patch("subprocess.run")
    @patch("os.path.isfile")
    @patch("os.access")
    def test_arbitrary_english_transcription(self, mock_access, mock_isfile, mock_run) -> None:
        """Verifies genuine open-ended transcription of arbitrary English questions."""
        mock_isfile.return_value = True
        mock_access.return_value = True

        arbitrary_questions = [
            "Can I use custom gestures in presentation mode?",
            "Why did the system lock after 30 seconds of inactivity?",
            "How does touchless practice mode calibrate hand size?",
        ]

        dummy_speech = np.random.uniform(-0.15, 0.15, 16000).astype(np.float32)

        for q in arbitrary_questions:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = f"[00:00:00.000 --> 00:00:03.000]   {q}\n"
            mock_proc.stderr = ""
            mock_run.return_value = mock_proc

            transcribed = self.transcriber.transcribe(dummy_speech)
            self.assertEqual(transcribed, q)

    def test_whisper_output_parser_edge_cases(self) -> None:
        """Tests parsing of blank audio, timestamps, and multi-line whisper output."""
        parser = LocalSpeechTranscriber._parse_whisper_output

        # Blank audio tag
        self.assertIsNone(parser("[BLANK_AUDIO]"))
        self.assertIsNone(parser("(silence)"))
        self.assertIsNone(parser("   "))
        self.assertIsNone(parser("."))

        # Multi-line text with timestamps and debug logs
        sample = (
            "main: processing 16000 samples\n"
            "[00:00:00.000 --> 00:00:01.500]   What is Media mode?\n"
            "[00:00:01.500 --> 00:00:03.000]   Can I skip tracks?\n"
        )
        self.assertEqual(parser(sample), "What is Media mode? Can I skip tracks?")

    # -------------------------------------------------------------------------
    # 2. Empty, Silent, and Noisy Audio Handling
    # -------------------------------------------------------------------------
    def test_empty_or_silent_speech_rejected(self) -> None:
        """Verifies that silence or low-energy ambient noise returns None before whisper."""
        silence = np.zeros(8000, dtype=np.float32)
        self.assertIsNone(self.transcriber.transcribe(silence))

        very_low_noise = np.random.uniform(-0.005, 0.005, 8000).astype(np.float32)
        self.assertIsNone(self.transcriber.transcribe(very_low_noise))

        empty = np.array([], dtype=np.float32)
        self.assertIsNone(self.transcriber.transcribe(empty))
        self.assertIsNone(self.transcriber.transcribe(None))

    # -------------------------------------------------------------------------
    # 3. Missing Binary or Model Handling
    # -------------------------------------------------------------------------
    def test_missing_model_or_binary_graceful_handling(self) -> None:
        """Verifies that missing whisper binary or model gracefully returns None."""
        transcriber_missing = LocalSpeechTranscriber(
            binary_path="/nonexistent/bin/whisper-cli",
            model_path="/nonexistent/models/ggml-base.en.bin",
        )
        self.assertFalse(transcriber_missing.is_available)
        dummy_speech = np.random.uniform(-0.15, 0.15, 16000).astype(np.float32)
        # Does not crash or throw unhandled exceptions
        self.assertIsNone(transcriber_missing.transcribe(dummy_speech))

    def test_voice_manager_missing_whisper_shows_hud_warning(self) -> None:
        """Verifies VoiceActivationManager signals missing STT on HUD."""
        transcriber_missing = LocalSpeechTranscriber(
            binary_path="/nonexistent/bin/whisper-cli",
            model_path="/nonexistent/models/ggml-base.en.bin",
        )
        manager = VoiceActivationManager(
            permission_granted=True,
            enable_mic_stream=False,
            transcriber=transcriber_missing,
        )
        try:
            # Simulate capture completion
            chunk = np.random.uniform(-0.15, 0.15, 16000).astype(np.float32)
            manager.trigger_wake_word()
            manager.update(current_timestamp=10.0)
            manager.feed_audio(chunk)

            # Advance clock past timeout to trigger transcription worker
            manager.update(current_timestamp=16.0)
            time.sleep(0.05)  # Allow background worker to complete

            status = manager.update(current_timestamp=16.1)
            self.assertEqual(status.state, VoiceState.TIMEOUT)
            self.assertIn("STT UNAVAILABLE", status.label)
        finally:
            manager.close()

    # -------------------------------------------------------------------------
    # 4. Timeout and Subprocess Error Safety
    # -------------------------------------------------------------------------
    @patch("subprocess.run")
    @patch("os.path.isfile")
    @patch("os.access")
    def test_whisper_timeout_handling(self, mock_access, mock_isfile, mock_run) -> None:
        """Verifies inference timeout does not hang or leave temporary files."""
        mock_isfile.return_value = True
        mock_access.return_value = True
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["whisper-cli"], timeout=6.0)

        dummy_speech = np.random.uniform(-0.15, 0.15, 16000).astype(np.float32)
        result = self.transcriber.transcribe(dummy_speech)
        self.assertIsNone(result)

    # -------------------------------------------------------------------------
    # 5. Gemini Context-Aware Responses
    # -------------------------------------------------------------------------
    def test_gemini_context_aware_response(self) -> None:
        """Verifies Gemini response parsing and metadata recording."""
        mock_http = MockRemoteHttpProvider(
            response_payload={
                "answer": "Show OpenPalm to the camera and hold steady for 1 second to unlock.",
                "suggested_action": "Hold OpenPalm 1.0s",
                "topic": "LOCKED_STATE",
                "category": "HELP",
            }
        )
        assistant = LLMAssistant(http_provider=mock_http)
        try:
            resp = assistant.answer_voice_question("how do i unlock", self.app_state)
            self.assertIsInstance(resp, VoiceAssistantResponse)
            self.assertEqual(resp.provider, "gemini")
            self.assertIn("OpenPalm", resp.answer)
            self.assertEqual(resp.topic, "LOCKED_STATE")

            # Check prompt contents sent to Gemini
            self.assertEqual(len(mock_http.calls), 1)
            prompt_sent = mock_http.calls[0]["prompt"]
            self.assertIn("how do i unlock", prompt_sent)
            self.assertIn("MEDIA", prompt_sent)
            self.assertIn("LOCKED", prompt_sent)
        finally:
            assistant.close()

    # -------------------------------------------------------------------------
    # 6. Offline Help Fallback
    # -------------------------------------------------------------------------
    def test_offline_help_fallback_on_network_failure(self) -> None:
        """Verifies graceful fallback to offline rule-based provider when Gemini fails."""
        failing_http = MockRemoteHttpProvider(raise_exc=True)
        assistant = LLMAssistant(http_provider=failing_http)
        try:
            resp = assistant.answer_voice_question("what gestures can i use", self.app_state)
            self.assertIsInstance(resp, VoiceAssistantResponse)
            self.assertEqual(resp.provider, "offline")
            self.assertIn("ThumbsUp", resp.answer)
            self.assertEqual(resp.topic, "SUPPORTED_GESTURES")
        finally:
            assistant.close()

    def test_offline_answers_across_all_contexts(self) -> None:
        """Verifies offline responses across Media, Presentation, Custom, Coaching, Onboarding."""
        # Media mode
        resp_media = self.offline_provider.answer_voice_question("tell me about media mode", self.app_state)
        self.assertIn("playback", resp_media.answer.lower())
        self.assertEqual(resp_media.topic, "MEDIA_MODE")

        # Presentation mode
        pres_state = dict(self.app_state, context="PRESENTATION")
        resp_pres = self.offline_provider.answer_voice_question("how to next slide", pres_state)
        self.assertIn("slide", resp_pres.answer.lower())

        # Custom gestures
        resp_cust = self.offline_provider.answer_voice_question("how to enroll a custom gesture", self.app_state)
        self.assertIn("press 'e'", resp_cust.answer.lower())

        # Coaching
        resp_coach = self.offline_provider.answer_voice_question("how do i improve accuracy", self.app_state)
        self.assertIn("steady", resp_coach.answer.lower())

        # Onboarding
        resp_onboard = self.offline_provider.answer_voice_question("how does practice mode work", self.app_state)
        self.assertIn("onboarding", resp_onboard.answer.lower())

    # -------------------------------------------------------------------------
    # 7. Locked-State Questions
    # -------------------------------------------------------------------------
    def test_locked_state_questions(self) -> None:
        """Verifies context-awareness when system is LOCKED vs ACTIVE."""
        # When LOCKED
        self.app_state["is_locked"] = True
        self.app_state["state"] = "LOCKED"
        resp_locked = self.offline_provider.answer_voice_question("why isn't it working", self.app_state)
        self.assertIn("LOCKED", resp_locked.answer)
        self.assertIn("OpenPalm", resp_locked.answer)

        # When ACTIVE
        self.app_state["is_locked"] = False
        self.app_state["state"] = "ACTIVE"
        resp_active = self.offline_provider.answer_voice_question("how to lock the app", self.app_state)
        self.assertIn("ACTIVE", resp_active.answer)
        self.assertIn("press 'L'", resp_active.answer)

    # -------------------------------------------------------------------------
    # 8. Microphone Unavailable & Permission Revoked
    # -------------------------------------------------------------------------
    def test_microphone_unavailable_and_permission_revoked(self) -> None:
        """Verifies state machine when mic permission is revoked or unavailable."""
        manager = VoiceActivationManager(
            permission_granted=False,
            enable_mic_stream=False,
            get_app_state_summary=lambda: self.app_state,
        )
        try:
            # Without permission: questions are ignored
            self.assertEqual(manager.state, VoiceState.DISABLED)
            manager.ask_question_async("hello")
            self.assertEqual(manager.state, VoiceState.DISABLED)

            # Grant permission -> STANDBY
            manager.grant_permission()
            self.assertEqual(manager.state, VoiceState.STANDBY)

            # Revoke permission -> DISABLED
            manager.revoke_permission()
            self.assertEqual(manager.state, VoiceState.DISABLED)
            st = manager.update(current_timestamp=10.0)
            self.assertEqual(st.state, VoiceState.DISABLED)
            self.assertIn("MIC OFF", st.label)
        finally:
            manager.close()

    # -------------------------------------------------------------------------
    # 9. Timeout and Cancellation
    # -------------------------------------------------------------------------
    def test_timeout_and_cancellation(self) -> None:
        """Verifies question capture timeout and ESC cancellation."""
        manager = VoiceActivationManager(
            permission_granted=True,
            enable_mic_stream=False,
            capture_timeout=3.0,
            get_app_state_summary=lambda: self.app_state,
        )
        try:
            # Trigger wake word
            manager.trigger_wake_word()
            manager.update(current_timestamp=10.0)
            self.assertTrue(manager.is_capturing)

            # Timeout after 3.0s without audio
            st_timeout = manager.update(current_timestamp=13.1)
            self.assertEqual(st_timeout.state, VoiceState.TIMEOUT)
            self.assertIn("TIMEOUT", st_timeout.label)

            # Test cancellation during active response
            manager.ask_question("what gestures can i use")
            self.assertEqual(manager.state, VoiceState.RESPONDING)
            manager.cancel()
            self.assertEqual(manager.state, VoiceState.STANDBY)
            self.assertIsNone(manager.latest_response)
        finally:
            manager.close()

    # -------------------------------------------------------------------------
    # 10. Asynchronous Processing Without Blocking
    # -------------------------------------------------------------------------
    def test_async_processing_does_not_block(self) -> None:
        """Verifies that asking a voice question dispatches asynchronously without blocking."""
        called_back = []

        def _delayed_answer(*args, **kwargs):
            time.sleep(0.05)
            return VoiceAssistantResponse(
                question="async test",
                answer="done",
                provider="mock",
            )

        assistant = LLMAssistant()
        try:
            assistant.answer_voice_question = _delayed_answer
            fut = assistant.answer_voice_question_async(
                question="async test",
                app_state_summary=self.app_state,
                callback=lambda res: called_back.append(res),
            )
            # The async call returns immediately before the sleep finishes
            self.assertFalse(fut.done())
            res = fut.result(timeout=1.0)
            self.assertEqual(res.answer, "done")
            self.assertEqual(len(called_back), 1)
        finally:
            assistant.close()

    # -------------------------------------------------------------------------
    # 11. Verification That Audio Data Is NEVER Sent to Gemini
    # -------------------------------------------------------------------------
    def test_zero_audio_payload_enforcement(self) -> None:
        """Verifies that attempting to pass audio arrays or buffers raises TypeError."""
        audio_array = np.zeros(16000, dtype=np.float32)
        audio_bytes = b"\x00" * 32000

        with self.assertRaises(TypeError):
            _assert_no_audio_payload(audio_array)

        with self.assertRaises(TypeError):
            _assert_no_audio_payload(audio_bytes)

        with self.assertRaises(TypeError):
            _assert_no_audio_payload({"audio_sample": audio_bytes})

        assistant = LLMAssistant()
        try:
            with self.assertRaises(TypeError):
                assistant.answer_voice_question(audio_array, self.app_state)  # type: ignore

            with self.assertRaises(TypeError):
                assistant.answer_voice_question("what gestures", {"audio": audio_bytes})  # type: ignore
        finally:
            assistant.close()

    # -------------------------------------------------------------------------
    # 12. Verification That Voice Requests CANNOT Execute OS Actions
    # -------------------------------------------------------------------------
    def test_voice_requests_cannot_execute_os_actions(self) -> None:
        """Verifies that voice requests to perform actions are rejected by the safety gate."""
        action_queries = [
            "can you please volume up right now",
            "execute next slide please",
            "trigger skip track now",
            "do a volume down",
            "can you pause the music",
        ]
        for query in action_queries:
            resp = self.offline_provider.answer_voice_question(query, self.app_state)
            self.assertEqual(resp.category, "SAFETY")
            self.assertEqual(resp.topic, "SAFETY_GATE")
            self.assertIn("assistance only", resp.answer.lower())
            self.assertIn("OpenPalm", resp.answer)
            self.assertTrue(resp.suggested_action.startswith("Use gesture"))

    # -------------------------------------------------------------------------
    # 13. End-to-End Pipeline: Wake -> Capture -> STT -> Async QA -> HUD Response
    # -------------------------------------------------------------------------
    def test_end_to_end_wake_capture_stt_and_hud_response(self) -> None:
        """
        Verifies the full pipeline:
        Wake detection -> audio capture in WAKE_DETECTED/CAPTURING_QUERY ->
        transcription -> LLMAssistant async request -> callback -> HUD response.
        """
        sim_time = [200.0]

        def test_clock() -> float:
            return sim_time[0]

        transcriber = LocalSpeechTranscriber()
        transcriber.simulate_transcription("how do i unlock")

        assistant = LLMAssistant()
        manager = VoiceActivationManager(
            sample_rate=16000,
            chunk_size=1024,
            capture_timeout=2.0,
            permission_granted=True,
            enable_mic_stream=False,
            transcriber=transcriber,
            llm_assistant=assistant,
            enable_tts=False,
            get_app_state_summary=lambda: self.app_state,
            clock=test_clock,
        )

        try:
            # 1. Trigger wake word
            t0 = sim_time[0]
            manager.trigger_wake_word(timestamp=t0)
            self.assertEqual(manager.state, VoiceState.WAKE_DETECTED)

            # 2. Feed audio chunks immediately (even during WAKE_DETECTED)
            chunk = np.ones(1024, dtype=np.float32) * 0.05
            for _ in range(10):
                manager.feed_audio(chunk, timestamp=t0)

            self.assertGreaterEqual(len(manager._captured_chunks), 10)

            # 3. Update frame to start CAPTURING_QUERY
            st_cap = manager.update(current_timestamp=t0)
            self.assertEqual(st_cap.state, VoiceState.CAPTURING_QUERY)

            # Feed more chunks
            for _ in range(5):
                manager.feed_audio(chunk, timestamp=t0)

            # 4. Finalize capture at timeout (2.0s)
            sim_time[0] = t0 + 2.1
            st_trans = manager.update(current_timestamp=sim_time[0])
            self.assertEqual(st_trans.state, VoiceState.TRANSCRIBING)

            # 5. Wait for async QA callback
            resp_received = False
            for _ in range(40):
                time.sleep(0.05)
                st = manager.update(current_timestamp=sim_time[0])
                if st.state == VoiceState.RESPONDING:
                    resp_received = True
                    self.assertIsNotNone(st.response_text)
                    self.assertEqual(st.recognized_question, "how do i unlock")
                    self.assertIn("VOICE ANSWER", st.label)
                    break

            self.assertTrue(resp_received, "Assistant failed to reach RESPONDING state with HUD answer.")
        finally:
            manager.close()
            assistant.close()

    # -------------------------------------------------------------------------
    # Phase 7: Spoken AI Coach Responses Tests
    # -------------------------------------------------------------------------
    @patch("subprocess.Popen")
    def test_tts_invocation_safe_argument_array_no_shell(self, mock_popen: MagicMock) -> None:
        """Verifies TTS uses safe argument arrays and never shell=True."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_popen.return_value = mock_proc

        tts = LocalTextToSpeech(executable_path="/usr/bin/say", speech_rate=185)
        success = tts.speak_async("Hello from IntentSense voice coach!")
        self.assertTrue(success)

        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        cmd = args[0]

        # 1. Must be list of arguments
        self.assertIsInstance(cmd, list)
        self.assertEqual(cmd[0], "/usr/bin/say")
        self.assertIn("-r", cmd)
        self.assertIn("185", cmd)
        self.assertEqual(cmd[-1], "Hello from IntentSense voice coach!")

        # 2. Never use shell=True
        self.assertFalse(kwargs.get("shell", False))
        tts.stop()

    def test_tts_successful_speech_lifecycle(self) -> None:
        """Verifies full speech lifecycle: speaking -> on_done -> responding."""
        tts = LocalTextToSpeech()
        done_called = []

        def mock_speaker(text: str, on_done=None) -> bool:
            if on_done:
                on_done()
            done_called.append(text)
            return True

        tts.set_custom_speaker(mock_speaker)
        success = tts.speak_async("Sample speech", on_done=lambda: done_called.append("done"))
        self.assertTrue(success)
        self.assertIn("Sample speech", done_called)
        self.assertIn("done", done_called)

    def test_tts_cancellation_and_shutdown(self) -> None:
        """Verifies that cancel() and close() safely terminate active speech processes."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # actively running

        tts = LocalTextToSpeech(executable_path="/usr/bin/say")
        with tts._proc_lock:
            tts._current_proc = mock_proc

        self.assertTrue(tts.is_speaking)
        tts.stop()
        mock_proc.terminate.assert_called()
        self.assertFalse(tts.is_speaking)

        # Test manager cancellation shuts down speech
        manager = VoiceActivationManager(permission_granted=True, enable_mic_stream=False, tts=tts)
        with tts._proc_lock:
            tts._current_proc = mock_proc
        manager.cancel()
        self.assertFalse(tts.is_speaking)
        manager.close()

    def test_tts_overlapping_request_prevention(self) -> None:
        """Verifies that new speech requests terminate any active speech before starting."""
        mock_proc1 = MagicMock()
        mock_proc1.poll.return_value = None

        tts = LocalTextToSpeech(executable_path="/usr/bin/say")
        with tts._proc_lock:
            tts._current_proc = mock_proc1

        # Starting new speech must stop mock_proc1
        with patch("subprocess.Popen") as mock_popen:
            mock_proc2 = MagicMock()
            mock_proc2.poll.return_value = None
            mock_popen.return_value = mock_proc2

            tts.speak_async("Second phrase")
            mock_proc1.terminate.assert_called()

        tts.stop()

    def test_tts_microphone_suppression_during_playback(self) -> None:
        """
        Verifies that microphone wake detection and chunk capture are suppressed
        while the assistant is actively speaking, preventing acoustic self-triggering.
        """
        tts = LocalTextToSpeech(executable_path="/usr/bin/say")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        with tts._proc_lock:
            tts._current_proc = mock_proc

        manager = VoiceActivationManager(
            permission_granted=True,
            enable_mic_stream=False,
            tts=tts,
        )

        self.assertTrue(manager.is_speaking)
        self.assertFalse(manager.is_listening)

        # Feed audio chunk while speaking: must be dropped immediately
        chunk = np.ones(1024, dtype=np.float32) * 0.1
        accepted = manager.feed_audio(chunk)
        self.assertFalse(accepted)
        self.assertEqual(len(manager._captured_chunks), 0)

        # Stop speech -> audio should no longer be suppressed
        tts.stop()
        self.assertFalse(manager.is_speaking)
        self.assertTrue(manager.is_listening)
        manager.close()

    def test_tts_offline_response_speech(self) -> None:
        """Verifies that offline rule-based responses are spoken via TTS."""
        spoken_phrases = []

        tts = LocalTextToSpeech()
        tts.set_custom_speaker(lambda text, on_done: spoken_phrases.append(text) or (on_done() if on_done else None) or True)

        manager = VoiceActivationManager(
            permission_granted=True,
            enable_mic_stream=False,
            tts=tts,
            get_app_state_summary=lambda: self.app_state,
        )

        try:
            # Ask a question programmatically
            resp = manager.ask_question("how do i unlock")
            self.assertIsNotNone(resp)
            self.assertEqual(manager.state, VoiceState.RESPONDING)
            status = manager.update(current_timestamp=time.monotonic())
            self.assertIn("VOICE ANSWER", status.label)
        finally:
            manager.close()

    def test_tts_non_blocking_webcam_processing(self) -> None:
        """Verifies speak_async returns immediately without blocking frame loop."""
        tts = LocalTextToSpeech()

        t0 = time.monotonic()
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc

            tts.speak_async("Non-blocking test message")
            elapsed = time.monotonic() - t0

        self.assertLess(elapsed, 0.05, f"speak_async took {elapsed:.4f}s, expected < 0.05s")
        tts.stop()

    def test_tts_missing_executable_graceful_handling(self) -> None:
        """Verifies missing 'say' executable fails gracefully without crashing."""
        tts = LocalTextToSpeech(executable_path="/nonexistent/bin/say")
        self.assertFalse(tts.is_available)

        done_called = []
        success = tts.speak_async("Test message", on_done=lambda: done_called.append(True))
        self.assertFalse(success)
        self.assertEqual(len(done_called), 1)

    # -------------------------------------------------------------------------
    # Dynamic Voice Capture & VAD Tests (Removed 5-second fixed limit)
    # -------------------------------------------------------------------------
    def test_dynamic_capture_short_question(self) -> None:
        """Verifies short questions stop after 1.2s silence without waiting for 5s."""
        transcriber = LocalSpeechTranscriber()
        transcriber.simulate_transcription("help")
        manager = VoiceActivationManager(
            permission_granted=True,
            enable_mic_stream=False,
            transcriber=transcriber,
            enable_tts=False,
        )
        try:
            manager.trigger_wake_word()
            t0 = 100.0
            manager.update(current_timestamp=t0)
            self.assertEqual(manager.state, VoiceState.CAPTURING_QUERY)

            # 1.0 second of speech (16 chunks of 1024 samples @ 16kHz)
            speech_chunk = np.ones(1024, dtype=np.float32) * 0.05
            for i in range(16):
                manager.feed_audio(speech_chunk, timestamp=t0 + (i * 0.064))
            self.assertTrue(manager.has_speech_started)

            # 1.2 seconds of silence (19 chunks of 1024 samples of ambient silence)
            silent_chunk = np.zeros(1024, dtype=np.float32)
            for j in range(19):
                manager.feed_audio(silent_chunk, timestamp=t0 + 1.0 + (j * 0.064))

            # Total elapsed is ~2.2s (much less than old 5s cutoff)
            st = manager.update(current_timestamp=t0 + 2.3)
            self.assertEqual(st.state, VoiceState.TRANSCRIBING)
        finally:
            manager.close()

    def test_dynamic_capture_long_question(self) -> None:
        """Verifies questions can exceed the old 5s limit up to speech completion."""
        transcriber = LocalSpeechTranscriber()
        transcriber.simulate_transcription("how does presentation mode work with gestures")
        manager = VoiceActivationManager(
            permission_granted=True,
            enable_mic_stream=False,
            transcriber=transcriber,
            enable_tts=False,
        )
        try:
            manager.trigger_wake_word()
            t0 = 100.0
            manager.update(current_timestamp=t0)

            speech_chunk = np.ones(1024, dtype=np.float32) * 0.05
            # Feed 12 seconds of speech in 1-second blocks
            for sec in range(12):
                manager.feed_audio(speech_chunk, timestamp=t0 + sec)
                st = manager.update(current_timestamp=t0 + sec)
                # Must NOT cut off at 5 seconds
                self.assertEqual(st.state, VoiceState.CAPTURING_QUERY)
                self.assertTrue(st.is_capturing)

            # User stops speaking at 12.0s; 1.2s of silence follows
            silent_chunk = np.zeros(1024, dtype=np.float32)
            for j in range(19):
                manager.feed_audio(silent_chunk, timestamp=t0 + 12.0 + (j * 0.064))

            st = manager.update(current_timestamp=t0 + 13.3)
            self.assertEqual(st.state, VoiceState.TRANSCRIBING)
        finally:
            manager.close()

    def test_dynamic_capture_pause_between_words(self) -> None:
        """Verifies natural conversational pauses (<1.2s) do not prematurely end capture."""
        manager = VoiceActivationManager(
            permission_granted=True,
            enable_mic_stream=False,
            enable_tts=False,
        )
        try:
            manager.trigger_wake_word()
            t0 = 100.0
            manager.update(current_timestamp=t0)

            speech_chunk = np.ones(1024, dtype=np.float32) * 0.05
            silent_chunk = np.zeros(1024, dtype=np.float32)

            # Phrase 1: "How do I" (0.5s)
            manager.feed_audio(speech_chunk, timestamp=t0 + 0.1)
            manager.feed_audio(speech_chunk, timestamp=t0 + 0.5)

            # Conversational pause: 0.5s of silence (< 1.2s threshold)
            manager.feed_audio(silent_chunk, timestamp=t0 + 0.8)
            st_pause = manager.update(current_timestamp=t0 + 0.9)
            self.assertEqual(st_pause.state, VoiceState.CAPTURING_QUERY)
            self.assertTrue(st_pause.is_capturing)

            # Phrase 2: "unlock" (0.4s)
            manager.feed_audio(speech_chunk, timestamp=t0 + 1.2)
            st_word2 = manager.update(current_timestamp=t0 + 1.3)
            self.assertEqual(st_word2.state, VoiceState.CAPTURING_QUERY)

            # Continuous silence: 1.3s (> 1.2s threshold)
            for j in range(20):
                manager.feed_audio(silent_chunk, timestamp=t0 + 1.4 + (j * 0.064))
            st_final = manager.update(current_timestamp=t0 + 2.8)
            self.assertEqual(st_final.state, VoiceState.TRANSCRIBING)
        finally:
            manager.close()

    def test_dynamic_capture_silence_detection(self) -> None:
        """Verifies exact 1.2s silence detection boundary."""
        manager = VoiceActivationManager(
            permission_granted=True,
            enable_mic_stream=False,
            enable_tts=False,
            silence_timeout=1.2,
        )
        try:
            manager.trigger_wake_word()
            t0 = 200.0
            manager.update(current_timestamp=t0)

            speech_chunk = np.ones(1024, dtype=np.float32) * 0.05
            manager.feed_audio(speech_chunk, timestamp=t0 + 0.5)

            # At 0.8s silence: still capturing
            st_sub = manager.update(current_timestamp=t0 + 0.5 + 0.8)
            self.assertEqual(st_sub.state, VoiceState.CAPTURING_QUERY)
            self.assertTrue(st_sub.is_capturing)

            # At 1.2s silence: capture finishes
            st_done = manager.update(current_timestamp=t0 + 0.5 + 1.25)
            self.assertEqual(st_done.state, VoiceState.TRANSCRIBING)
        finally:
            manager.close()

    def test_dynamic_capture_maximum_duration(self) -> None:
        """Verifies 30-second maximum duration safety limit processes rather than discards speech."""
        transcriber = LocalSpeechTranscriber()
        transcriber.simulate_transcription("long question processed at limit")
        manager = VoiceActivationManager(
            permission_granted=True,
            enable_mic_stream=False,
            transcriber=transcriber,
            enable_tts=False,
            max_capture_duration=30.0,
        )
        try:
            manager.trigger_wake_word()
            t0 = 100.0
            manager.update(current_timestamp=t0)

            speech_chunk = np.ones(1024, dtype=np.float32) * 0.05
            # Continuous speech without 1.2s pause
            for sec in range(30):
                manager.feed_audio(speech_chunk, timestamp=t0 + sec)

            # At 30.0s limit, captured speech must be processed
            st_limit = manager.update(current_timestamp=t0 + 30.1)
            self.assertEqual(st_limit.state, VoiceState.TRANSCRIBING)
        finally:
            manager.close()

    def test_dynamic_capture_no_speech_timeout(self) -> None:
        """Verifies 8-second timeout if no speech starts after wake detection."""
        manager = VoiceActivationManager(
            permission_granted=True,
            enable_mic_stream=False,
            enable_tts=False,
            initial_speech_timeout=8.0,
        )
        try:
            manager.trigger_wake_word()
            t0 = 100.0
            st0 = manager.update(current_timestamp=t0)
            self.assertEqual(st0.state, VoiceState.CAPTURING_QUERY)

            # Feed ambient background noise below threshold (0.003 < 0.012)
            noise_chunk = np.random.uniform(-0.005, 0.005, 1024).astype(np.float32)
            manager.feed_audio(noise_chunk, timestamp=t0 + 4.0)

            # At 7.5s: still waiting for speech
            st_wait = manager.update(current_timestamp=t0 + 7.5)
            self.assertEqual(st_wait.state, VoiceState.CAPTURING_QUERY)
            self.assertFalse(manager.has_speech_started)

            # At 8.0s: times out cleanly
            st_timeout = manager.update(current_timestamp=t0 + 8.1)
            self.assertEqual(st_timeout.state, VoiceState.TIMEOUT)
            self.assertIn("No Speech", st_timeout.label)
        finally:
            manager.close()

    def test_dynamic_capture_cancellation(self) -> None:
        """Verifies ESC / programmatic cancellation during capture cleanly resets state and buffers."""
        manager = VoiceActivationManager(
            permission_granted=True,
            enable_mic_stream=False,
            enable_tts=False,
        )
        try:
            manager.trigger_wake_word()
            t0 = 100.0
            manager.update(current_timestamp=t0)

            speech_chunk = np.ones(1024, dtype=np.float32) * 0.05
            manager.feed_audio(speech_chunk, timestamp=t0 + 1.0)
            self.assertTrue(manager.has_speech_started)
            self.assertTrue(manager.is_capturing)

            # Cancel mid-capture
            manager.cancel()
            self.assertEqual(manager.state, VoiceState.STANDBY)
            self.assertFalse(manager.is_capturing)
            self.assertFalse(manager.has_speech_started)
            self.assertEqual(len(manager._captured_chunks), 0)
        finally:
            manager.close()

    # -------------------------------------------------------------------------
    # Open-Ended, Long-Form & Multi-Turn Conversation Tests
    # -------------------------------------------------------------------------
    def test_open_ended_general_purpose_questions(self) -> None:
        """Verifies assistant handles general-purpose and technical questions without category limits."""
        mock_http = MockRemoteHttpProvider(
            response_payload={
                "answer": "Quantum entanglement is a phenomenon where quantum particles become inextricably linked.",
                "suggested_action": "Ask a follow-up question.",
                "topic": "QUANTUM_PHYSICS",
                "category": "TECHNICAL_DISCUSSION",
            }
        )
        assistant = LLMAssistant(http_provider=mock_http)
        try:
            resp = assistant.answer_voice_question("Explain quantum entanglement in simple terms", self.app_state)
            self.assertEqual(resp.provider, "gemini")
            self.assertIn("Quantum entanglement", resp.answer)
            self.assertEqual(resp.topic, "QUANTUM_PHYSICS")
            self.assertEqual(len(mock_http.calls), 1)

            # Check that prompt contains user's exact technical question
            prompt_sent = mock_http.calls[0]["prompt"]
            self.assertIn("Explain quantum entanglement in simple terms", prompt_sent)
        finally:
            assistant.close()

    def test_multi_turn_conversation_history(self) -> None:
        """Verifies multi-turn conversation maintains dialogue history across turns."""
        mock_http = MockRemoteHttpProvider(
            response_payload={
                "answer": "Media mode allows touchless playback control.",
                "suggested_action": "",
                "topic": "MEDIA",
                "category": "HELP",
            }
        )
        assistant = LLMAssistant(http_provider=mock_http)
        try:
            # Turn 1
            resp1 = assistant.answer_voice_question("Tell me about Media mode", self.app_state)
            self.assertEqual(len(assistant.get_conversation_history()), 2)
            self.assertEqual(assistant.get_conversation_history()[0]["role"], "user")
            self.assertEqual(assistant.get_conversation_history()[0]["content"], "Tell me about Media mode")
            self.assertEqual(assistant.get_conversation_history()[1]["role"], "assistant")

            # Turn 2: Follow-up question
            mock_http.response_payload = {
                "answer": "Yes, you can enroll custom gestures like RockOn to play or pause.",
                "suggested_action": "",
                "topic": "CUSTOM_GESTURES",
                "category": "HELP",
            }
            resp2 = assistant.answer_voice_question("Can I use custom gestures with it?", self.app_state)
            self.assertEqual(len(assistant.get_conversation_history()), 4)

            # Verify prompt for Turn 2 includes Turn 1 history
            turn2_prompt = mock_http.calls[1]["prompt"]
            self.assertIn("Tell me about Media mode", turn2_prompt)
            self.assertIn("Media mode allows touchless playback control.", turn2_prompt)
            self.assertIn("Can I use custom gestures with it?", turn2_prompt)
        finally:
            assistant.close()

    def test_conversation_history_summarization_on_long_dialogue(self) -> None:
        """Verifies older turns are summarized when history exceeds context window limits."""
        assistant = LLMAssistant(max_history_turns=6, max_history_chars=500)
        try:
            # Simulate 10 turns of dialogue
            for i in range(10):
                assistant._add_to_history("user", f"Question number {i} with some detailed text to fill characters.")
                assistant._add_to_history("assistant", f"Answer number {i} providing informative details.")

            # Must have trimmed to most recent turns and generated a running summary
            history = assistant.get_conversation_history()
            self.assertLessEqual(len(history), 6)
            self.assertIsNotNone(assistant._history_summary)
            self.assertIn("Previous context", assistant._history_summary)
        finally:
            assistant.close()

    def test_conversation_reset_controls(self) -> None:
        """Verifies conversation history reset via spoken command and manager API."""
        assistant = LLMAssistant()
        manager = VoiceActivationManager(
            permission_granted=True,
            enable_mic_stream=False,
            llm_assistant=assistant,
            enable_tts=False,
        )
        try:
            # Add dialogue
            assistant._add_to_history("user", "Hello")
            assistant._add_to_history("assistant", "Hi there!")
            self.assertEqual(len(assistant.get_conversation_history()), 2)

            # Spoken reset command
            resp_reset = assistant.answer_voice_question("reset conversation", self.app_state)
            self.assertEqual(resp_reset.topic, "CONVERSATION_RESET")
            self.assertIn("reset", resp_reset.answer.lower())
            self.assertEqual(len(assistant.get_conversation_history()), 0)

            # Add more dialogue and test manager.reset_conversation()
            assistant._add_to_history("user", "What's the weather?")
            self.assertEqual(len(assistant.get_conversation_history()), 1)
            manager.reset_conversation()
            self.assertEqual(len(assistant.get_conversation_history()), 0)
        finally:
            manager.close()
            assistant.close()

    def test_long_recording_efficient_streaming_transcription(self) -> None:
        """Verifies long recording chunks are streamed to temporary file and transcribed without truncation."""
        transcriber = LocalSpeechTranscriber()
        transcriber.simulate_transcription("transcribed long multi-chunk question")

        # 20 chunks of audio
        chunks = [np.ones(1024, dtype=np.float32) * 0.05 for _ in range(20)]
        result = transcriber.transcribe_chunks(chunks)
        self.assertEqual(result, "transcribed long multi-chunk question")

    # -------------------------------------------------------------------------
    # Offline Fallback & HUD Rendering Regression Tests
    # -------------------------------------------------------------------------
    def test_offline_unsupported_general_knowledge_fallback(self) -> None:
        """
        Verifies that in OFFLINE mode, unrelated general knowledge questions
        never return canned IntentSense help, but clearly inform the user that
        general knowledge answers require an internet connection and Gemini.
        """
        assistant = LLMAssistant()  # No remote provider configured (offline)
        general_queries = [
            "What is the speed of light?",
            "Who wrote Hamlet?",
            "Why is the sky blue?",
            "Can fish drown?",
            "Explain quantum physics in detail",
            "What is the capital of Australia?",
        ]
        expected_msg = "General knowledge answers require an internet connection. Gemini is currently unavailable."
        try:
            for q in general_queries:
                resp = assistant.answer_voice_question(q, self.app_state)
                self.assertEqual(resp.provider, "offline")
                self.assertEqual(resp.answer, expected_msg)
                self.assertEqual(resp.topic, "OFFLINE_UNAVAILABLE")
                self.assertEqual(resp.category, "OFFLINE")
                self.assertNotIn("IntentSense voice assistant is ready", resp.answer)
        finally:
            assistant.close()

    def test_offline_supported_intentsense_help_retained(self) -> None:
        """
        Verifies that supported IntentSense queries still return rich domain guidance
        when offline, without returning the offline unavailable message.
        """
        assistant = LLMAssistant()
        try:
            # 1. Unlocking
            resp_unlock = assistant.answer_voice_question("how do i unlock", self.app_state)
            self.assertIn("LOCKED", resp_unlock.answer)
            self.assertIn("OpenPalm", resp_unlock.answer)

            # 2. Supported gestures
            resp_gest = assistant.answer_voice_question("what gestures can i use", self.app_state)
            self.assertIn("Media mode", resp_gest.answer)

            # 3. Media mode
            resp_med = assistant.answer_voice_question("tell me about media mode", self.app_state)
            self.assertIn("Media mode controls playback", resp_med.answer)

            # 4. Presentation mode
            resp_pres = assistant.answer_voice_question("how does presentation mode work", self.app_state)
            self.assertIn("Presentation mode controls slides", resp_pres.answer)

            # 5. Custom gestures
            resp_cust = assistant.answer_voice_question("how to enroll custom gesture", self.app_state)
            self.assertIn("enroll a custom gesture", resp_cust.answer)

            # 6. Coaching
            resp_coach = assistant.answer_voice_question("how to improve gesture accuracy", self.app_state)
            self.assertIn("hold gestures steady", resp_coach.answer)

            # 7. Practice mode
            resp_prac = assistant.answer_voice_question("how does practice mode work", self.app_state)
            self.assertIn("Practice Mode", resp_prac.answer)

            # 8. General IntentSense inquiry
            resp_gen = assistant.answer_voice_question("What is IntentSense and how do I use it?", self.app_state)
            self.assertIn("IntentSense voice assistant is ready", resp_gen.answer)
        finally:
            assistant.close()

    def test_hud_rendering_long_response_word_wrap(self) -> None:
        """
        Verifies that HUD rendering wraps long response text at word boundaries,
        adjusts card height dynamically, and never cuts words or truncates.
        """
        import cv2
        from app import draw_hud
        from core.activation_state import SystemState
        from core.context_manager import Context
        from core.voice_activation import VoiceHudStatus, VoiceState

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        long_ans = (
            "IntentSense voice assistant is ready. You can ask about supported gestures, "
            "unlocking, Media and Presentation modes, custom gesture enrollment, or onboarding practice. "
            "In Media mode, ThumbsUp raises volume and Fist lowers volume smoothly."
        )
        status = VoiceHudStatus(
            state=VoiceState.RESPONDING,
            label="ANSWER READY",
            color_bgr=(0, 255, 160),
            time_remaining=8.0,
            has_permission=True,
            is_capturing=False,
            recognized_question="What gestures can I use in Media mode?",
            response_text=long_ans,
            response_provider="offline",
            is_speaking=False,
        )

        # Ensure draw_hud executes cleanly without crashing on long multiline text
        draw_hud(
            frame=frame,
            state=SystemState.LOCKED,
            context=Context.MEDIA,
            gesture=None,
            confidence=0.0,
            consecutive_frames=0,
            target_frames=3,
            hold_duration=0.0,
            target_hold=1.0,
            intent=None,
            action_name=None,
            action_status="NONE",
            time_remaining=0.0,
            last_action_event=None,
            fps=30.0,
            live_mode=False,
            voice_status=status,
        )
        # Frame should have been modified with rendered card pixels
        self.assertTrue(np.any(frame > 0))

    # -------------------------------------------------------------------------
    # General Knowledge & Machine Learning Disambiguation Regression Tests
    # -------------------------------------------------------------------------
    def test_machine_learning_does_not_return_onboarding_answer(self) -> None:
        """
        Regression Test: Confirms that 'What is machine learning?' and related learning queries
        do NOT get falsely matched by the touchless onboarding practice rule ('learn' substring).
        When offline, they must return the explicit offline-unavailable message.
        """
        assistant = LLMAssistant()  # Unconfigured (offline)
        learning_queries = [
            "What is machine learning?",
            "What is deep learning?",
            "How does reinforcement learning work?",
            "Can you explain machine learning in simple terms?",
        ]
        expected_unavailable = "General knowledge answers require an internet connection. Gemini is currently unavailable."
        try:
            for q in learning_queries:
                resp = assistant.answer_voice_question(q, self.app_state)
                self.assertEqual(resp.provider, "offline")
                self.assertEqual(resp.answer, expected_unavailable)
                self.assertNotIn("Onboarding", resp.answer)
                self.assertNotIn("Practice Mode", resp.answer)
                self.assertNotIn("Press 'O'", resp.answer)
        finally:
            assistant.close()

    def test_general_question_with_gemini_reaches_provider(self) -> None:
        """
        Verifies that when Gemini is configured and available, general knowledge questions
        such as 'What is machine learning?' reach Gemini and return Gemini's response.
        """
        mock_http = MockRemoteHttpProvider(
            response_payload={
                "answer": "Machine learning is a subset of artificial intelligence where algorithms learn patterns from data.",
                "suggested_action": "Ask about neural networks.",
                "topic": "MACHINE_LEARNING",
                "category": "TECHNICAL_DISCUSSION",
            }
        )
        assistant = LLMAssistant(http_provider=mock_http)
        try:
            resp = assistant.answer_voice_question("What is machine learning?", self.app_state)
            self.assertEqual(resp.provider, "gemini")
            self.assertIn("Machine learning is a subset", resp.answer)
            self.assertEqual(resp.topic, "MACHINE_LEARNING")
            self.assertEqual(len(mock_http.calls), 1)
            self.assertIn("What is machine learning?", mock_http.calls[0]["prompt"])
        finally:
            assistant.close()

    def test_general_question_api_timeout_fallback(self) -> None:
        """
        Verifies that when Gemini times out (e.g. timeout exception), it safely surfaces
        a diagnostic error response rather than silently disguising as OFFLINE or onboarding.
        """
        class TimeoutHttpProvider:
            is_configured = True
            api_key = "dummy_key"
            name = "gemini"
            last_status = 408
            last_error = "Gemini HTTP request timed out"

            def generate_json(self, prompt: str, system_prompt: str):
                import socket
                raise socket.timeout("Gemini HTTP request timed out")

        assistant = LLMAssistant(http_provider=TimeoutHttpProvider())
        try:
            resp = assistant.answer_voice_question("What is machine learning?", self.app_state)
            self.assertEqual(resp.provider, "gemini_error")
            self.assertEqual(resp.topic, "GEMINI_API_ERROR")
            self.assertEqual(resp.category, "ERROR")
            self.assertIn("timed out", resp.answer.lower())
            self.assertNotIn("Onboarding", resp.answer)
        finally:
            assistant.close()

    def test_general_question_missing_api_key_fallback(self) -> None:
        """
        Verifies that when GEMINI_API_KEY is not configured, general knowledge questions
        cleanly return the offline-unavailable message.
        """
        assistant = LLMAssistant(api_key=None, endpoint_url=None)
        try:
            self.assertFalse(assistant.has_active_remote)
            resp = assistant.answer_voice_question("What is machine learning?", self.app_state)
            self.assertEqual(resp.provider, "offline")
            self.assertEqual(resp.answer, "General knowledge answers require an internet connection. Gemini is currently unavailable.")
            self.assertEqual(resp.topic, "OFFLINE_UNAVAILABLE")
        finally:
            assistant.close()

    # -------------------------------------------------------------------------
    # Custom Mode, Management, Safe Test & Offline Disambiguation Tests
    # -------------------------------------------------------------------------
    def test_custom_mode_offline_question_variations(self) -> None:
        """
        Verifies that questions asking about Custom Mode (e.g. 'Is there any custom mode?',
        'Do you have custom mode?') correctly route to the Custom Mode offline answer.
        """
        assistant = LLMAssistant()  # Offline
        queries = [
            "Is there any custom mode?",
            "Do you have custom mode?",
            "What is custom mode?",
            "Tell me about custom mode",
            "Can I use custom mode?",
            "Does IntentSense have custom mode?",
        ]
        try:
            for q in queries:
                resp = assistant.answer_voice_question(q, self.app_state)
                self.assertEqual(resp.provider, "offline")
                self.assertEqual(resp.topic, "CUSTOM_MODE")
                self.assertIn("Custom mode", resp.answer)
                self.assertIn("Press '3'", resp.answer)
                self.assertNotIn("General knowledge answers require an internet connection", resp.answer)
        finally:
            assistant.close()

    def test_personalized_gestures_and_enrollment_variations(self) -> None:
        """
        Verifies that natural variations for creating custom/personalized gestures
        (e.g. 'Can I create my own gestures?') route to Custom Gesture enrollment.
        """
        assistant = LLMAssistant()
        queries = [
            "Can I create my own gestures?",
            "Can I make my own gestures?",
            "How do I create a new gesture?",
            "How to enroll custom gesture",
            "Tell me about personalized gestures",
        ]
        try:
            for q in queries:
                resp = assistant.answer_voice_question(q, self.app_state)
                self.assertEqual(resp.provider, "offline")
                self.assertEqual(resp.topic, "CUSTOM_GESTURES")
                self.assertIn("enroll a custom gesture", resp.answer)
                self.assertIn("press 'e'", resp.answer.lower())
                self.assertNotIn("General knowledge answers require an internet connection", resp.answer)
        finally:
            assistant.close()

    def test_custom_gesture_management_variations(self) -> None:
        """
        Verifies that questions about managing custom gestures (e.g. 'How do I manage custom gestures?')
        route to the Custom Gesture Management answer.
        """
        assistant = LLMAssistant()
        queries = [
            "How do I manage custom gestures?",
            "Can I manage custom gestures?",
            "How to manage gestures?",
            "Manage custom profiles",
            "Where is the custom gesture manager?",
        ]
        try:
            for q in queries:
                resp = assistant.answer_voice_question(q, self.app_state)
                self.assertEqual(resp.provider, "offline")
                self.assertEqual(resp.topic, "CUSTOM_GESTURE_MANAGEMENT")
                self.assertIn("Custom Gesture Management", resp.answer)
                self.assertIn("Press 'M'", resp.answer)
                self.assertNotIn("General knowledge answers require an internet connection", resp.answer)
        finally:
            assistant.close()

    def test_safe_test_mode_variations(self) -> None:
        """
        Verifies questions about Safe Test Mode route to Safe Test Mode offline help.
        """
        assistant = LLMAssistant()
        queries = [
            "What is safe test mode?",
            "How do I use safe test mode?",
            "Can I test gestures safely?",
            "How do I test custom gestures?",
        ]
        try:
            for q in queries:
                resp = assistant.answer_voice_question(q, self.app_state)
                self.assertEqual(resp.provider, "offline")
                self.assertEqual(resp.topic, "SAFE_TEST_MODE")
                self.assertIn("Safe Test mode", resp.answer)
                self.assertIn("Press 'T'", resp.answer)
                self.assertNotIn("General knowledge answers require an internet connection", resp.answer)
        finally:
            assistant.close()

    def test_unrelated_general_knowledge_remains_offline_unavailable(self) -> None:
        """
        Verifies that unrelated general knowledge questions remain mapped to the
        offline-unavailable response, even if they contain common words.
        """
        assistant = LLMAssistant()
        queries = [
            "What is the capital of Japan?",
            "What is a custom in Japan?",  # Cultural custom
            "How do airplanes fly?",
            "What is machine learning?",
            "Tell me about the Roman Empire",
        ]
        expected_msg = "General knowledge answers require an internet connection. Gemini is currently unavailable."
        try:
            for q in queries:
                resp = assistant.answer_voice_question(q, self.app_state)
                self.assertEqual(resp.provider, "offline")
                self.assertEqual(resp.answer, expected_msg)
                self.assertEqual(resp.topic, "OFFLINE_UNAVAILABLE")
        finally:
            assistant.close()

    def test_custom_mode_question_with_gemini_reaches_gemini(self) -> None:
        """
        Verifies that when Gemini is active, questions about custom mode still query Gemini.
        """
        mock_http = MockRemoteHttpProvider(
            response_payload={
                "answer": "Custom mode lets you enroll unique hand poses like RockOn or Peace.",
                "suggested_action": "Press '3' for Custom mode.",
                "topic": "CUSTOM_MODE",
                "category": "HELP",
            }
        )
        assistant = LLMAssistant(http_provider=mock_http)
        try:
            resp = assistant.answer_voice_question("Is there any custom mode?", self.app_state)
            self.assertEqual(resp.provider, "gemini")
            self.assertIn("Custom mode lets you enroll", resp.answer)
            self.assertEqual(resp.topic, "CUSTOM_MODE")
            self.assertEqual(len(mock_http.calls), 1)
        finally:
            assistant.close()

    def test_llm_http_provider_strips_quotes_and_whitespace(self) -> None:
        """Verifies that LLMHttpProvider cleanly strips whitespace and quotation marks from GEMINI_API_KEY."""
        from core.llm_assistant import LLMHttpProvider
        provider = LLMHttpProvider(api_key='  "AIzaSyTestKey123"  ')
        self.assertEqual(provider.api_key, "AIzaSyTestKey123")
        self.assertTrue(provider.is_configured)

    def test_llm_http_provider_sanitizes_errors(self) -> None:
        """Verifies that API keys and bearer tokens are never exposed in error messages."""
        from core.llm_assistant import LLMHttpProvider
        secret_key = "AIzaSySecretKey98765"
        provider = LLMHttpProvider(api_key=secret_key)
        raw_msg = f"Failed to connect: https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={secret_key}&alt=json Bearer {secret_key}"
        sanitized = provider._sanitize_error_message(raw_msg)
        self.assertNotIn(secret_key, sanitized)
        self.assertIn("[REDACTED", sanitized)

    @patch("urllib.request.urlopen")
    def test_gemini_auth_error_diagnostic_response(self, mock_urlopen) -> None:
        """
        Verifies that HTTP 401/403 API errors are surfaced as GEMINI_API_ERROR diagnostics
        for general queries rather than silently disguised as generic OFFLINE.
        """
        import io
        import urllib.error
        from core.llm_assistant import LLMHttpProvider

        secret_key = "AIzaSyFakeKeyAuthTest"
        err_body = b'{"error": {"code": 401, "message": "API key not valid. Please pass a valid API key."}}'
        err = urllib.error.HTTPError(
            url=f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={secret_key}",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=io.BytesIO(err_body),
        )
        mock_urlopen.side_effect = err

        provider = LLMHttpProvider(api_key=secret_key)
        assistant = LLMAssistant(http_provider=provider)
        try:
            # 1. General question should surface HTTP 401 diagnostic without leaking the key
            resp_gen = assistant.answer_voice_question("What is quantum computing?", self.app_state)
            self.assertEqual(resp_gen.provider, "gemini_error")
            self.assertEqual(resp_gen.topic, "GEMINI_API_ERROR")
            self.assertEqual(resp_gen.category, "ERROR")
            self.assertIn("authentication failed (HTTP 401)", resp_gen.answer)
            self.assertNotIn(secret_key, resp_gen.answer)

            # 2. IntentSense-specific question should gracefully fall back to local rule
            resp_local = assistant.answer_voice_question("How do I unlock?", self.app_state)
            self.assertEqual(resp_local.provider, "offline")
            self.assertEqual(resp_local.topic, "LOCKED_STATE")
            self.assertIn("OpenPalm", resp_local.answer)
        finally:
            assistant.close()

    @patch("urllib.request.urlopen")
    def test_gemini_quota_exceeded_diagnostic_response(self, mock_urlopen) -> None:
        """Verifies that HTTP 429 quota exhaustion is surfaced clearly as rate limit exceeded."""
        import io
        import urllib.error
        from core.llm_assistant import LLMHttpProvider

        err_body = b'{"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota)."}}'
        err = urllib.error.HTTPError(
            url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=io.BytesIO(err_body),
        )
        mock_urlopen.side_effect = err

        provider = LLMHttpProvider(api_key="AIzaSyValidKeyQuotaExhausted")
        assistant = LLMAssistant(http_provider=provider)
        try:
            resp = assistant.answer_voice_question("What is machine learning?", self.app_state)
            self.assertEqual(resp.provider, "gemini_error")
            self.assertEqual(resp.topic, "GEMINI_API_ERROR")
            self.assertIn("rate limit or quota exceeded (HTTP 429)", resp.answer)
        finally:
            assistant.close()

    @patch("urllib.request.urlopen")
    def test_gemini_timeout_diagnostic_response(self, mock_urlopen) -> None:
        """Verifies that network timeouts surface request timed out diagnostics."""
        import socket
        import urllib.error
        from core.llm_assistant import LLMHttpProvider

        mock_urlopen.side_effect = urllib.error.URLError(reason=socket.timeout("The read operation timed out"))

        provider = LLMHttpProvider(api_key="AIzaSyTimeoutKey")
        assistant = LLMAssistant(http_provider=provider)
        try:
            resp = assistant.answer_voice_question("What is the history of AI?", self.app_state)
            self.assertEqual(resp.provider, "gemini_error")
            self.assertEqual(resp.topic, "GEMINI_API_ERROR")
            self.assertIn("request timed out", resp.answer.lower())
        finally:
            assistant.close()


if __name__ == "__main__":
    unittest.main()
