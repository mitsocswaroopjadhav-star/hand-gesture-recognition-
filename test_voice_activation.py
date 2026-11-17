"""
test_voice_activation.py
========================

Unit test suite for IntentSense Phase 5: Local Voice Activation.
Tests permission gates, acoustic wake-word detection, question capture window,
timeouts, cancellation, and thread lifecycle.
"""

import time
import unittest
import numpy as np

from core.voice_activation import (
    LocalAcousticWakeDetector,
    VoiceActivationManager,
    VoiceHudStatus,
    VoiceState,
)


class TestLocalAcousticWakeDetector(unittest.TestCase):
    """Tests the lightweight on-device acoustic wake-word detector."""

    def setUp(self) -> None:
        self.detector = LocalAcousticWakeDetector(sample_rate=16000, buffer_duration=1.8)

    def test_silence_produces_no_wake(self) -> None:
        silence = np.zeros(1024, dtype=np.float32)
        self.assertFalse(self.detector.process_chunk(silence))

    def test_low_energy_noise_produces_no_wake(self) -> None:
        noise = np.random.uniform(-0.01, 0.01, 1024).astype(np.float32)
        self.assertFalse(self.detector.process_chunk(noise))

    def test_empty_or_none_chunk_safe(self) -> None:
        self.assertFalse(self.detector.process_chunk(None))
        self.assertFalse(self.detector.process_chunk(np.array([], dtype=np.float32)))

    def test_simulated_wake_word(self) -> None:
        self.detector.simulate_wake_word()
        dummy = np.zeros(512, dtype=np.float32)
        self.assertTrue(self.detector.process_chunk(dummy))
        # After triggering, simulation resets
        self.assertFalse(self.detector.process_chunk(dummy))

    def test_reset_clears_buffer_and_simulation(self) -> None:
        self.detector.simulate_wake_word()
        self.detector.reset()
        dummy = np.zeros(512, dtype=np.float32)
        self.assertFalse(self.detector.process_chunk(dummy))


class TestVoiceActivationManager(unittest.TestCase):
    """Tests permission controls, capture state machine, and timeouts."""

    def setUp(self) -> None:
        self.manager = VoiceActivationManager(
            sample_rate=16000,
            chunk_size=1024,
            capture_timeout=5.0,
            permission_granted=False,
            enable_mic_stream=False,
        )

    def tearDown(self) -> None:
        self.manager.close()

    def test_initial_state_disabled_without_permission(self) -> None:
        self.assertEqual(self.manager.state, VoiceState.DISABLED)
        self.assertFalse(self.manager.has_permission)
        self.assertFalse(self.manager.is_listening)
        self.assertFalse(self.manager.is_capturing)

        status = self.manager.update(current_timestamp=10.0)
        self.assertEqual(status.state, VoiceState.DISABLED)
        self.assertIn("MIC OFF", status.label)
        self.assertFalse(status.has_permission)

    def test_feed_and_trigger_ignored_when_disabled(self) -> None:
        dummy_chunk = np.zeros(1024, dtype=np.float32)
        self.assertFalse(self.manager.feed_audio(dummy_chunk))
        self.manager.trigger_wake_word()
        self.assertEqual(self.manager.state, VoiceState.DISABLED)

    def test_permission_grant_and_revoke(self) -> None:
        self.manager.grant_permission()
        self.assertTrue(self.manager.has_permission)
        self.assertEqual(self.manager.state, VoiceState.STANDBY)
        self.assertTrue(self.manager.is_listening)

        status = self.manager.update(current_timestamp=10.0)
        self.assertEqual(status.state, VoiceState.STANDBY)
        self.assertIn("STANDBY", status.label)

        self.manager.revoke_permission()
        self.assertFalse(self.manager.has_permission)
        self.assertEqual(self.manager.state, VoiceState.DISABLED)

    def test_toggle_permission(self) -> None:
        # Initially False -> becomes True
        granted = self.manager.toggle_permission()
        self.assertTrue(granted)
        self.assertTrue(self.manager.has_permission)
        self.assertEqual(self.manager.state, VoiceState.STANDBY)

        # Toggle again -> becomes False
        granted = self.manager.toggle_permission()
        self.assertFalse(granted)
        self.assertFalse(self.manager.has_permission)
        self.assertEqual(self.manager.state, VoiceState.DISABLED)

    def test_wake_word_trigger_and_callback(self) -> None:
        self.manager.grant_permission()

        callback_invoked = []
        self.manager.on_wake_callback = lambda: callback_invoked.append(True)

        self.manager.trigger_wake_word()
        self.assertEqual(len(callback_invoked), 1)
        self.assertEqual(self.manager.state, VoiceState.WAKE_DETECTED)
        self.assertTrue(self.manager.is_capturing)

        # First update transitions WAKE_DETECTED -> CAPTURING_QUERY
        status = self.manager.update(current_timestamp=10.0)
        self.assertEqual(status.state, VoiceState.CAPTURING_QUERY)
        self.assertTrue(status.is_capturing)
        self.assertAlmostEqual(status.time_remaining, 5.0, places=1)
        self.assertIn("CAPTURING", status.label)

    def test_feed_audio_simulated_wake(self) -> None:
        self.manager.grant_permission()
        self.manager._wake_detector.simulate_wake_word()

        dummy_chunk = np.zeros(1024, dtype=np.float32)
        detected = self.manager.feed_audio(dummy_chunk)
        self.assertTrue(detected)
        self.assertEqual(self.manager.state, VoiceState.WAKE_DETECTED)

    def test_capture_timeout_and_reset(self) -> None:
        self.manager.grant_permission()
        self.manager.trigger_wake_word()

        t0 = 100.0
        # Frame 1: start capture
        st1 = self.manager.update(current_timestamp=t0)
        self.assertEqual(st1.state, VoiceState.CAPTURING_QUERY)
        self.assertEqual(st1.time_remaining, 5.0)

        # Frame 2: halfway through capture (2.5s elapsed)
        st2 = self.manager.update(current_timestamp=t0 + 2.5)
        self.assertEqual(st2.state, VoiceState.CAPTURING_QUERY)
        self.assertAlmostEqual(st2.time_remaining, 2.5, places=1)

        # Frame 3: timeout reached (5.0s elapsed)
        st3 = self.manager.update(current_timestamp=t0 + 5.0)
        self.assertEqual(st3.state, VoiceState.TIMEOUT)
        self.assertFalse(st3.is_capturing)
        self.assertIn("TIMEOUT", st3.label)

        # Frame 4: still within 1.2s timeout flash
        st4 = self.manager.update(current_timestamp=t0 + 5.5)
        self.assertEqual(st4.state, VoiceState.TIMEOUT)

        # Frame 5: past 1.2s timeout flash -> resets to STANDBY
        st5 = self.manager.update(current_timestamp=t0 + 6.3)
        self.assertEqual(st5.state, VoiceState.STANDBY)
        self.assertFalse(st5.is_capturing)

    def test_cancellation(self) -> None:
        self.manager.grant_permission()
        self.manager.trigger_wake_word()
        self.manager.update(current_timestamp=10.0)
        self.assertTrue(self.manager.is_capturing)

        self.manager.cancel()
        self.assertEqual(self.manager.state, VoiceState.STANDBY)
        self.assertFalse(self.manager.is_capturing)

        status = self.manager.update(current_timestamp=10.5)
        self.assertEqual(status.state, VoiceState.STANDBY)

    def test_feed_audio_during_capture_appends_chunks(self) -> None:
        self.manager.grant_permission()
        self.manager.trigger_wake_word()
        self.manager.update(current_timestamp=10.0)

        chunk1 = np.ones(512, dtype=np.float32)
        chunk2 = np.ones(512, dtype=np.float32) * 2.0
        self.manager.feed_audio(chunk1)
        self.manager.feed_audio(chunk2)

        self.assertEqual(len(self.manager._captured_chunks), 2)

    def test_clean_shutdown(self) -> None:
        self.manager.grant_permission()
        self.manager.trigger_wake_word()
        self.manager.close()
        self.assertEqual(self.manager.state, VoiceState.DISABLED)
        self.assertFalse(self.manager.has_permission)
        self.assertFalse(self.manager._is_running)

    def test_headless_hardware_mic_stream_handling(self) -> None:
        # With enable_mic_stream=True, if no audio device exists (as in headless/sandbox),
        # the background thread gracefully marks state as ERROR without crashing.
        mgr = VoiceActivationManager(
            permission_granted=True,
            enable_mic_stream=True,
        )
        time.sleep(0.1)
        # In a headless environment with 0 audio devices, mgr handles it safely
        self.assertIn(mgr.state, (VoiceState.STANDBY, VoiceState.ERROR))
        status = mgr.update(current_timestamp=time.monotonic())
        self.assertIsNotNone(status)
        mgr.close()

    def _synthesize_hello_sense(self, offset: float = 0.1, stretch: float = 1.0) -> np.ndarray:
        """Synthesizes realistic 3-syllable 'Hello Sense' with voiced onset and sibilant fricative tail."""
        sr = 16000
        duration = 1.8
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sig = np.zeros_like(t)

        # 1. 'Hel' syllable (voiced vowel)
        s1, e1 = offset, offset + 0.20 * stretch
        m1 = (t >= s1) & (t <= e1)
        sig[m1] += 0.09 * (np.sin(np.pi * (t[m1] - s1) / (e1 - s1)) ** 2) * np.sin(2 * np.pi * 210 * t[m1])

        # 2. 'lo' syllable (voiced liquid vowel)
        s2, e2 = e1 + 0.05, e1 + 0.05 + 0.25 * stretch
        m2 = (t >= s2) & (t <= e2)
        sig[m2] += 0.08 * (np.sin(np.pi * (t[m2] - s2) / (e2 - s2)) ** 2) * np.sin(2 * np.pi * 190 * t[m2])

        # 3. 'Sense' syllable (voiced vowel core + sibilant /s/ fricative)
        s3, e3 = e2 + 0.08, e2 + 0.08 + 0.25 * stretch
        m3 = (t >= s3) & (t <= e3)
        sig[m3] += 0.07 * (np.sin(np.pi * (t[m3] - s3) / (e3 - s3)) ** 2) * np.sin(2 * np.pi * 180 * t[m3])

        # Sibilant /s/ fricative tail in 'Sense'
        ss, es = e3 - 0.05, e3 + 0.15 * stretch
        ms = (t >= ss) & (t <= es)
        sig[ms] += 0.05 * np.random.uniform(-1, 1, np.sum(ms))

        return sig.astype(np.float32)

    def test_circular_buffer_wrap_and_wake_detection_regression(self) -> None:
        """
        Regression test:
        1. Ensures buffer chronological roll works after circular wrap-around.
        2. Ensures realistic speech RMS (0.015-0.030) with 'Hello Sense' cadence
           and sibilant fricative tail reliably triggers detection.
        """
        detector = LocalAcousticWakeDetector(sample_rate=16000, buffer_duration=1.8)

        # 1. Advance write position past wrap-around (feed 40 chunks of silence)
        silent_chunk = np.zeros(1024, dtype=np.float32)
        for _ in range(40):
            detector.process_chunk(silent_chunk)
        self.assertGreater(detector._write_pos, 0)

        # 2. Synthesize realistic 3-syllable "Hello Sense" with sibilant tail
        sig = self._synthesize_hello_sense(offset=0.10, stretch=1.0)

        # 3. Stream in chunks of 1024
        chunks = [sig[i:i + 1024] for i in range(0, len(sig), 1024)]
        detected = False
        for c in chunks:
            if detector.process_chunk(c):
                detected = True
                break

        self.assertTrue(detected, "Wake-word detector failed to detect 'Hello Sense' after buffer wrap-around.")

    def test_hello_sense_cadence_variations(self) -> None:
        """Verifies that normal, brisk, and measured deliveries of 'Hello Sense' are recognized."""
        for name, offset, stretch in [
            ("Normal", 0.10, 1.0),
            ("Fast", 0.15, 0.8),
            ("Slow", 0.05, 1.2),
        ]:
            detector = LocalAcousticWakeDetector(sample_rate=16000, buffer_duration=1.8)
            sig = self._synthesize_hello_sense(offset=offset, stretch=stretch)
            chunks = [sig[i:i + 1024] for i in range(0, len(sig), 1024)]
            detected = any(detector.process_chunk(c) for c in chunks)
            self.assertTrue(detected, f"Failed to detect {name} 'Hello Sense'")

    def test_similar_phrases_rejected(self) -> None:
        """
        Verifies that partial or similar phrases are correctly rejected:
        - 'Hello' alone (lacks sibilant 'Sense')
        - 'Sense' alone (lacks voiced 'Hello' onset)
        - 'Hello Hello' (lacks sibilant 'Sense')
        """
        sr = 16000
        full_sig = self._synthesize_hello_sense(offset=0.10, stretch=1.0)

        # 1. 'Hello' only (clear out 'Sense')
        sig_hello_only = full_sig.copy()
        sig_hello_only[int(0.60 * sr):] = 0.0
        det1 = LocalAcousticWakeDetector(sample_rate=16000, buffer_duration=1.8)
        chunks = [sig_hello_only[i:i + 1024] for i in range(0, len(sig_hello_only), 1024)]
        self.assertFalse(any(det1.process_chunk(c) for c in chunks), "'Hello' alone must be rejected")

        # 2. 'Sense' only (clear out 'Hello')
        sig_sense_only = np.zeros_like(full_sig)
        sig_sense_only[int(0.65 * sr):] = full_sig[int(0.65 * sr):]
        det2 = LocalAcousticWakeDetector(sample_rate=16000, buffer_duration=1.8)
        chunks = [sig_sense_only[i:i + 1024] for i in range(0, len(sig_sense_only), 1024)]
        self.assertFalse(any(det2.process_chunk(c) for c in chunks), "'Sense' alone must be rejected")

        # 3. 'Hello Hello' (two 'Hello' phrases, neither has sibilant fricative)
        sig_hh = np.zeros_like(full_sig)
        h_part = full_sig[:int(0.55 * sr)]
        sig_hh[:int(0.55 * sr)] = h_part
        sig_hh[int(0.60 * sr):int(0.60 * sr) + len(h_part)] = h_part
        det3 = LocalAcousticWakeDetector(sample_rate=16000, buffer_duration=1.8)
        chunks = [sig_hh[i:i + 1024] for i in range(0, len(sig_hh), 1024)]
        self.assertFalse(any(det3.process_chunk(c) for c in chunks), "'Hello Hello' must be rejected")

    def test_steady_background_noise_rejected(self) -> None:
        """Verifies that steady hum/fan noise (uniform subframe energy) is rejected."""
        detector = LocalAcousticWakeDetector(sample_rate=16000, buffer_duration=1.8)
        sr = 16000
        t = np.linspace(0, 1.8, int(sr * 1.8), endpoint=False)
        hum = 0.02 * np.sin(2 * np.pi * 200 * t).astype(np.float32)

        chunks = [hum[i:i + 1024] for i in range(0, len(hum), 1024)]
        detected = any(detector.process_chunk(c) for c in chunks)
        self.assertFalse(detected)

    def test_speech_without_sibilance_rejected(self) -> None:
        """Verifies that vowel hums without sibilant tail /s/ are rejected."""
        detector = LocalAcousticWakeDetector(sample_rate=16000, buffer_duration=1.8)
        sr = 16000
        t = np.linspace(0, 1.8, int(sr * 1.8), endpoint=False)
        vowel_hum = 0.04 * (np.sin(np.pi * t / 1.8) ** 2) * np.sin(2 * np.pi * 220 * t)

        chunks = [vowel_hum[i:i + 1024].astype(np.float32) for i in range(0, len(vowel_hum), 1024)]
        detected = any(detector.process_chunk(c) for c in chunks)
        self.assertFalse(detected)

    def test_full_pipeline_trace_step1_to_step7(self) -> None:
        """
        Traces all 7 pipeline steps with 'Hello Sense':
        1. Permission grant & stream management
        2. Audio chunk ingestion
        3. feed_audio processing
        4. Match decision
        5. on_wake_callback invocation
        6. VoiceState transition to CAPTURING_QUERY
        7. HUD status update
        """
        trace = []

        def wake_cb():
            trace.append("step_5_callback_invoked")

        manager = VoiceActivationManager(
            sample_rate=16000,
            chunk_size=1024,
            capture_timeout=5.0,
            permission_granted=False,
            enable_mic_stream=False,
            on_wake_callback=wake_cb,
        )

        # Step 1: Permission
        manager.grant_permission()
        self.assertEqual(manager.state, VoiceState.STANDBY)
        trace.append("step_1_permission_granted")

        # Step 2 & 3: Audio chunk ingestion and feed_audio processing
        sig = self._synthesize_hello_sense(offset=0.10, stretch=1.0)

        # Wrap buffer first to test ring-buffer safety
        for _ in range(30):
            manager.feed_audio(np.zeros(1024, dtype=np.float32))

        # Feed 'Hello Sense' wake phrase
        chunks = [sig[i:i + 1024] for i in range(0, len(sig), 1024)]
        wake_detected = False
        for c in chunks:
            if manager.feed_audio(c):
                wake_detected = True
                trace.append("step_4_match_decision")
                break

        # Step 4: Match decision
        self.assertTrue(wake_detected)
        self.assertEqual(manager.state, VoiceState.WAKE_DETECTED)

        # Step 5: on_wake_callback
        self.assertIn("step_5_callback_invoked", trace)

        # Step 6: VoiceState transition to CAPTURING_QUERY on update()
        hud_status = manager.update(current_timestamp=50.0)
        self.assertEqual(manager.state, VoiceState.CAPTURING_QUERY)
        trace.append("step_6_transition_capturing")

        # Step 7: HUD status update
        self.assertTrue(hud_status.is_capturing)
        self.assertEqual(hud_status.color_bgr, (0, 255, 0))
        self.assertIn("CAPTURING", hud_status.label)
        trace.append("step_7_hud_status_updated")

        self.assertEqual(
            trace,
            [
                "step_1_permission_granted",
                "step_5_callback_invoked",
                "step_4_match_decision",
                "step_6_transition_capturing",
                "step_7_hud_status_updated",
            ],
        )
        manager.close()


if __name__ == "__main__":
    unittest.main()
