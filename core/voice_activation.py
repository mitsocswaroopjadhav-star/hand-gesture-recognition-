"""
core/voice_activation.py
========================

IntentSense Phase 5: Local Voice Activation & Wake-Word Engine.

Provides privacy-preserving, on-device voice wake-word detection for
"Hello Sense" without streaming audio to external servers.

Architecture & Safety Guarantees:
1. Local Wake-Word Engine:
   - Uses local acoustic syllabic & spectral envelope analysis (sounddevice + numpy + scipy).
   - NEVER streams background audio to Gemini, Cloud APIs, or remote servers.
2. Explicit Permission & Privacy:
   - Requires explicit user permission before opening audio input devices.
   - Defaults to DISABLED until explicitly permitted via CLI flag (--voice) or key shortcut (V).
3. Visible Listening Indicators:
   - Exposes comprehensive VoiceHudStatus for rendering active microphone telemetry on the HUD:
     • DISABLED: Gray / Red ("MIC OFF — Press 'V' to permit")
     • STANDBY: Yellow ("PASSIVE LISTENER — Say 'Hello Sense'")
     • WAKE_DETECTED / CAPTURING: Green ("VOICE ACTIVE — Listening for question (Xs)")
     • TIMEOUT: Orange ("TIMEOUT — Returning to standby")
4. Non-Blocking & Asynchronous:
   - Runs audio capture and keyword processing on a dedicated daemon thread.
   - Never blocks OpenCV webcam frame capture or HUD rendering.
5. Signal-Only Voice Gate:
   - Signals when the wake word is detected and opens a bounded query capture window.
   - Does NOT add conversational processing or TTS.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple
from collections import deque

import numpy as np

from core.paths import get_resource_path

PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if TYPE_CHECKING:
    from core.llm_assistant import LLMAssistant, VoiceAssistantResponse

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    sd = None
    SOUNDDEVICE_AVAILABLE = False


class VoiceState(str, Enum):
    """Lifecycle state of the voice activation pipeline."""
    DISABLED = "DISABLED"                # Microphone permission not granted or turned off
    STANDBY = "STANDBY"                  # Passively listening for local "Hello Sense" wake word
    WAKE_DETECTED = "WAKE_DETECTED"      # "Hello Sense" detected; ready to capture user question
    CAPTURING_QUERY = "CAPTURING_QUERY"  # Bounded capture window for user question
    TRANSCRIBING = "TRANSCRIBING"        # Local on-device Speech-to-Text in progress
    THINKING = "THINKING"                # Async LLM question answering in progress
    SPEAKING = "SPEAKING"                # Local Text-to-Speech active playback of AI answer
    RESPONDING = "RESPONDING"            # Displaying question and AI answer on HUD
    TIMEOUT = "TIMEOUT"                  # Listening timeout reached; cleanly resetting to STANDBY
    ERROR = "ERROR"                      # Hardware/driver error (e.g., no microphone device)


@dataclass(frozen=True)
class VoiceHudStatus:
    """Telemetry structure for real-time HUD listening indicators."""
    state: VoiceState
    label: str
    color_bgr: Tuple[int, int, int]
    time_remaining: float
    has_permission: bool
    is_capturing: bool
    is_processing: bool = False
    is_speaking: bool = False
    recognized_question: Optional[str] = None
    response_text: Optional[str] = None
    response_provider: Optional[str] = None


class LocalAcousticWakeDetector:
    """
    Lightweight, completely on-device acoustic wake-word detector for 'Hello Sense'.
    Detects the characteristic 3-syllable acoustic cadence:
      'Hel' (~0.2s) - 'lo' (~0.25s) - 'Sense' (~0.3s with sibilant /s/ fricative tail).
    Requires zero external network calls or heavy neural dependencies.
    """

    def __init__(self, sample_rate: int = 16000, buffer_duration: float = 1.8) -> None:
        self.sample_rate: int = sample_rate
        self.buffer_size: int = int(sample_rate * buffer_duration)
        self._buffer: np.ndarray = np.zeros(self.buffer_size, dtype=np.float32)
        self._write_pos: int = 0
        self._simulated_wake: bool = False

    def simulate_wake_word(self) -> None:
        """Forces wake word detection on the next frame (used for testing)."""
        self._simulated_wake = True

    def reset(self) -> None:
        """Clears the internal audio ring buffer."""
        self._buffer.fill(0.0)
        self._write_pos = 0
        self._simulated_wake = False

    def process_chunk(self, chunk: np.ndarray) -> bool:
        """
        Ingests an incoming audio chunk and evaluates if 'Hello Sense' was spoken.
        Returns True immediately upon wake-word detection.
        """
        if self._simulated_wake:
            self._simulated_wake = False
            return True

        if chunk is None or len(chunk) == 0:
            return False

        # Flatten to 1D float32
        data = chunk.flatten().astype(np.float32)

        # Append to circular buffer
        n = len(data)
        if n >= self.buffer_size:
            self._buffer[:] = data[-self.buffer_size:]
        else:
            remaining = self.buffer_size - self._write_pos
            if n <= remaining:
                self._buffer[self._write_pos:self._write_pos + n] = data
                self._write_pos = (self._write_pos + n) % self.buffer_size
            else:
                self._buffer[self._write_pos:] = data[:remaining]
                self._buffer[:n - remaining] = data[remaining:]
                self._write_pos = n - remaining

        return self._detect_wake_acoustic()

    def _detect_wake_acoustic(self) -> bool:
        """
        Evaluates short-time energy envelopes and spectral distribution.
        Correctly unrolls the circular buffer chronologically and checks for
        3-syllable cadence ('Hel-lo Sense') and sibilant fricative (/s/) in 'Sense'.
        """
        # Chronologically unroll circular buffer to maintain temporal ordering
        buf = np.roll(self._buffer, -self._write_pos) if self._write_pos != 0 else self._buffer

        # Overall RMS energy of the 1.8s buffer
        rms = float(np.sqrt(np.mean(buf ** 2)))
        if rms < 0.010:
            return False  # Ambient silence / low background noise

        # Divide chronological buffer into 8 temporal sub-frames (~225ms each)
        sub_frames = np.array_split(buf, 8)
        energies = np.array([float(np.sqrt(np.mean(f ** 2))) for f in sub_frames])
        max_e = float(np.max(energies))
        min_e = float(np.min(energies))

        if max_e < 0.018:
            return False

        # Reject steady hum/fan noise where all sub-frames have nearly uniform energy
        if min_e > 0.65 * max_e:
            return False

        # 3-syllable cadence ('Hel-lo Sense'): 2 to 6 prominent subframes
        prominent = energies > (0.35 * max_e)
        num_prominent = int(np.sum(prominent))
        if not (2 <= num_prominent <= 6):
            return False

        # Voiced onset in first 3 subframes ('Hel-lo')
        onset_energy = float(np.max(energies[:3]))
        if onset_energy < 0.35 * max_e:
            return False

        # Spectral analysis: zero-crossing rate approximation across sub-frames
        zcrs = [
            float(np.sum(np.abs(np.diff(np.sign(f)))) / (2 * len(f))) if len(f) > 64 else 0.0
            for f in sub_frames
        ]

        # 'Sense' sibilance in latter portion (subframes 3..7 have /s/ fricative)
        has_tail_sibilance = any(z > 0.12 for z in zcrs[3:])
        if not has_tail_sibilance:
            return False

        # Onset syllables ('Hel-lo') must be voiced vowels (low ZCR < 0.25)
        if any(z > 0.25 for z in zcrs[:2]):
            return False

        if rms >= 0.010:
            max_tail_zcr = max(zcrs[3:]) if len(zcrs) >= 4 else 0.0
            print(
                f"[Voice Debug] Detector eval: buffer_rms={rms:.4f}, max_e={max_e:.4f}, "
                f"min_e={min_e:.4f}, max_tail_zcr={max_tail_zcr:.3f}, match=True"
            )

        return True


class LocalSpeechTranscriber:
    """
    On-device, privacy-preserving English Speech-to-Text recognizer using whisper.cpp.
    Transcribes captured speech completely locally with the ggml-base.en model.
    Zero external cloud or network requests.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        binary_path: Optional[str] = None,
        model_path: Optional[str] = None,
        inference_timeout: float = 12.0,
        num_threads: int = 4,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.inference_timeout: float = inference_timeout
        self.num_threads: int = num_threads
        self._custom_binary_path: Optional[str] = binary_path
        self._custom_model_path: Optional[str] = model_path

        self._simulated_text: Optional[str] = None
        self._custom_transcriber_fn: Optional[Callable[[np.ndarray], Optional[str]]] = None
        self._logged_missing_warning: bool = False

    @property
    def binary_path(self) -> Optional[str]:
        """Resolves the whisper.cpp binary executable path."""
        if self._custom_binary_path:
            p = self._custom_binary_path
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return os.path.abspath(p)
            found = shutil.which(p)
            if found:
                return os.path.abspath(found)
            return p

        candidates = [
            os.environ.get("WHISPER_CPP_PATH", ""),
            os.environ.get("WHISPER_BIN", ""),
            get_resource_path(os.path.join("bin", "whisper-cli")),
            os.path.join(PROJECT_ROOT, "bin", "whisper-cli"),
            os.path.join(PROJECT_ROOT, "bin", "main"),
            os.path.join(PROJECT_ROOT, "whisper.cpp", "build", "bin", "whisper-cli"),
            os.path.join(PROJECT_ROOT, "whisper.cpp", "main"),
        ]
        for c in candidates:
            if c and os.path.isfile(c) and os.access(c, os.X_OK):
                return os.path.abspath(c)

        for name in ("whisper-cli", "whisper-cpp", "main"):
            found = shutil.which(name)
            if found:
                return os.path.abspath(found)

        return None

    @property
    def model_path(self) -> Optional[str]:
        """Resolves the ggml-base.en.bin model file path."""
        if self._custom_model_path:
            p = self._custom_model_path
            if os.path.isfile(p):
                return os.path.abspath(p)
            return p

        candidates = [
            os.environ.get("WHISPER_MODEL_PATH", ""),
            get_resource_path(os.path.join("models", "ggml-base.en.bin")),
            os.path.join(PROJECT_ROOT, "models", "ggml-base.en.bin"),
            os.path.join(PROJECT_ROOT, "models", "whisper-base.en.bin"),
            os.path.join(PROJECT_ROOT, "whisper.cpp", "models", "ggml-base.en.bin"),
            os.path.expanduser("~/.cache/whisper/ggml-base.en.bin"),
        ]
        for c in candidates:
            if c and os.path.isfile(c):
                return os.path.abspath(c)

        return None

    @property
    def is_available(self) -> bool:
        """Returns True if both whisper.cpp binary and base.en model are present and executable."""
        b = self.binary_path
        m = self.model_path
        if not b or not m:
            return False
        return (os.path.isfile(b) and os.access(b, os.X_OK)) and os.path.isfile(m)

    def simulate_transcription(self, text: Optional[str]) -> None:
        """Sets a simulated transcription for testing and synthetic pipelines."""
        self._simulated_text = text

    def set_custom_transcriber(self, fn: Optional[Callable[[np.ndarray], Optional[str]]]) -> None:
        """Allows injecting custom on-device backends or mocks."""
        self._custom_transcriber_fn = fn

    def transcribe_chunks(self, chunks: Sequence[np.ndarray]) -> Optional[str]:
        """
        Concatenates audio chunks or streams them directly into a temporary local audio file,
        running local whisper.cpp on-device transcription without excessive in-memory allocation.
        """
        if not chunks:
            return None

        if self._simulated_text is not None:
            txt = self._simulated_text
            self._simulated_text = None
            return txt

        if self._custom_transcriber_fn is not None:
            valid = [c.flatten().astype(np.float32) for c in chunks if c is not None and len(c) > 0]
            combined = np.concatenate(valid, axis=0) if len(valid) > 1 else (valid[0] if valid else np.array([], dtype=np.float32))
            return self._custom_transcriber_fn(combined)

        valid_chunks = [c for c in chunks if c is not None and len(c) > 0]
        if not valid_chunks:
            return None

        total_samples = sum(len(c) for c in valid_chunks)
        duration_s = total_samples / self.sample_rate
        if duration_s < 0.25:
            return None

        # VAD Energy check across chunks
        sum_sq = 0.0
        max_sub_e = 0.0
        for c in valid_chunks:
            c_rms = float(np.sqrt(np.mean(c ** 2)))
            if c_rms > max_sub_e:
                max_sub_e = c_rms
            sum_sq += float(np.sum(c ** 2))
        overall_rms = float(np.sqrt(sum_sq / total_samples)) if total_samples > 0 else 0.0

        print(
            f"[Voice Debug] STT VAD check: duration={duration_s:.2f}s, "
            f"overall_rms={overall_rms:.5f}, max_subframe_e={max_sub_e:.5f}"
        )

        if overall_rms < 0.003 and max_sub_e < 0.007:
            print("[Voice Debug] STT rejected: Audio below voice activity threshold (ambient silence).")
            return None

        if not self.is_available:
            if not self._logged_missing_warning:
                print(
                    f"[Whisper STT] whisper.cpp binary or ggml-base.en.bin model not found.\n"
                    f"  Binary: {self.binary_path or 'Not found'}\n"
                    f"  Model : {self.model_path or 'Not found'}\n"
                    f"  Run scripts/setup_whisper.sh or set WHISPER_CPP_PATH / WHISPER_MODEL_PATH."
                )
                self._logged_missing_warning = True
            return None

        # Stream chunks directly into temporary local WAV file
        fd, temp_wav = tempfile.mkstemp(prefix="intentsense_stt_", suffix=".wav")
        os.close(fd)

        try:
            with wave.open(temp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                for chunk in valid_chunks:
                    int16_chunk = np.clip(chunk.flatten() * 32767.0, -32768, 32767).astype(np.int16)
                    wf.writeframes(int16_chunk.tobytes())

            return self._run_whisper_file(temp_wav, duration_s=duration_s)
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """
        Performs local speech transcription on a 1D float32 audio waveform.
        Returns None if audio is silent, unintelligible, or whisper is unavailable.
        """
        if audio is None or len(audio) == 0:
            return None
        return self.transcribe_chunks([audio])

    def _run_whisper_file(self, wav_path: str, duration_s: float = 5.0) -> Optional[str]:
        """
        Runs whisper.cpp on a prepared audio WAV file with dynamic timeout scaling.
        """
        b_path = self.binary_path
        m_path = self.model_path
        if not b_path or not m_path:
            return None

        timeout = max(self.inference_timeout, duration_s * 1.5 + 5.0)

        cmd = [
            b_path,
            "-m", m_path,
            "-f", wav_path,
            "-nt",
            "-np",
            "-l", "en",
            "-t", str(self.num_threads),
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )

            print(
                f"[Voice Debug] Whisper subprocess completed: returncode={proc.returncode}, "
                f"stdout_bytes={len(proc.stdout)}, stderr_bytes={len(proc.stderr)}"
            )

            if proc.returncode != 0:
                # If GPU/Metal allocation failed (e.g. sandboxed environment or Metal OOM), fallback to CPU (-ng)
                if "-ng" not in cmd:
                    print(f"[Whisper STT] Primary inference exited with code {proc.returncode}. Retrying with CPU fallback (-ng)...")
                    try:
                        proc = subprocess.run(
                            cmd + ["-ng"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=timeout,
                        )
                        print(
                            f"[Voice Debug] Whisper CPU fallback completed: returncode={proc.returncode}, "
                            f"stdout_bytes={len(proc.stdout)}, stderr_bytes={len(proc.stderr)}"
                        )
                    except Exception as retry_err:
                        print(f"[Whisper STT] CPU fallback failed: {retry_err}")

            if proc.returncode != 0:
                print(f"[Whisper STT] Inference exited with code {proc.returncode}: {proc.stderr[:100]}")
                return None

            res_text = self._parse_whisper_output(proc.stdout)
            print(f"[Voice Debug] Parsed Whisper output: {repr(res_text)}")
            return res_text

        except subprocess.TimeoutExpired:
            print(f"[Whisper STT] Inference timed out after {timeout}s.")
            return None
        except Exception as e:
            print(f"[Whisper STT] Execution failed: {e}")
            return None

    @staticmethod
    def _parse_whisper_output(raw_output: str) -> Optional[str]:
        """
        Parses raw whisper.cpp stdout and extracts clean transcribed text.
        Removes timestamps, debug info, and non-speech markers.
        """
        if not raw_output:
            return None

        cleaned_lines: List[str] = []
        for line in raw_output.splitlines():
            line_str = line.strip()
            if not line_str:
                continue

            # Skip whisper metadata / log lines
            if any(line_str.startswith(prefix) for prefix in (
                "whisper_", "system_info:", "main:", "load_model:", "ggml_", "metal_", "cuda_"
            )):
                continue

            # Remove timestamp patterns: [00:00:00.000 --> 00:00:02.000]
            text = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]", "", line_str)

            # Remove non-speech brackets: [BLANK_AUDIO], [MUSIC], (silence), etc.
            text = re.sub(r"\[(?:BLANK_AUDIO|MUSIC|NOISE|APPLAUSE|LAUGHTER|SILENCE)\]", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\((?:silence|music|applause|laughter|noise)\)", "", text, flags=re.IGNORECASE)

            text = text.strip()
            if text:
                cleaned_lines.append(text)

        if not cleaned_lines:
            return None

        full_text = " ".join(cleaned_lines).strip()
        if len(full_text) < 2 or full_text in (".", ",", "?", "!", "..."):
            return None

        return full_text


class LocalTextToSpeech:
    """
    On-device, native Text-to-Speech synthesizer using macOS native 'say' command.
    Zero external cloud or network requests, zero third-party dependencies.
    """

    def __init__(
        self,
        executable_path: Optional[str] = None,
        voice: Optional[str] = None,
        speech_rate: int = 185,
        enabled: bool = True,
    ) -> None:
        self._custom_executable_path: Optional[str] = executable_path
        self.voice: Optional[str] = voice
        self.speech_rate: int = speech_rate
        self.enabled: bool = enabled
        self._current_proc: Optional[subprocess.Popen] = None
        self._proc_lock: threading.Lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._custom_speaker_fn: Optional[Callable[[str, Optional[Callable[[], None]]], bool]] = None

    @property
    def executable_path(self) -> Optional[str]:
        """Resolves the native 'say' executable path."""
        if self._custom_executable_path:
            p = self._custom_executable_path
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return os.path.abspath(p)
            found = shutil.which(p)
            if found:
                return os.path.abspath(found)
            return p

        # Check standard macOS paths
        candidates = [
            "/usr/bin/say",
            shutil.which("say"),
        ]
        for c in candidates:
            if c and os.path.isfile(c) and os.access(c, os.X_OK):
                return os.path.abspath(c)
        return None

    @property
    def is_available(self) -> bool:
        """Returns True if the 'say' executable exists and is executable."""
        p = self.executable_path
        return bool(p and os.path.isfile(p) and os.access(p, os.X_OK))

    @property
    def is_speaking(self) -> bool:
        """Returns True if speech playback is currently active."""
        with self._proc_lock:
            if self._current_proc is None:
                return False
            poll_res = self._current_proc.poll()
            return poll_res is None

    def set_custom_speaker(
        self, fn: Optional[Callable[[str, Optional[Callable[[], None]]], bool]]
    ) -> None:
        """Allows injecting custom speech backends or mocks for testing."""
        self._custom_speaker_fn = fn

    def speak_async(
        self,
        text: str,
        on_done: Optional[Callable[[], None]] = None,
    ) -> bool:
        """
        Asynchronously synthesizes speech using native macOS 'say'.
        Spawns background subprocess using safe argument arrays (never shell=True).
        Never blocks the caller. Invokes on_done callback upon completion.
        """
        if not self.enabled:
            if on_done:
                on_done()
            return False

        if self._custom_speaker_fn is not None:
            return self._custom_speaker_fn(text, on_done)

        clean_text = self._clean_text(text)
        if not clean_text:
            if on_done:
                on_done()
            return False

        say_bin = self.executable_path
        if not say_bin or not self.is_available:
            print("[TTS Debug] Native 'say' executable unavailable. Skipping speech playback.")
            if on_done:
                on_done()
            return False

        # Stop any active speaking before starting new speech to prevent overlapping
        self.stop()

        cmd = [say_bin, "-r", str(self.speech_rate)]
        if self.voice:
            cmd.extend(["-v", self.voice])
        cmd.append(clean_text)

        try:
            with self._proc_lock:
                self._current_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as e:
            print(f"[TTS Debug] Failed to spawn 'say' process: {e}")
            if on_done:
                on_done()
            return False

        def _monitor() -> None:
            proc = self._current_proc
            if proc is not None:
                try:
                    proc.wait()
                except Exception:
                    pass
            with self._proc_lock:
                if self._current_proc == proc:
                    self._current_proc = None
            if on_done:
                try:
                    on_done()
                except Exception as e:
                    print(f"[TTS Debug] on_done callback exception: {e}")

        self._monitor_thread = threading.Thread(
            target=_monitor,
            daemon=True,
            name="IntentSense-TTSMonitor",
        )
        self._monitor_thread.start()
        return True

    def stop(self) -> None:
        """Safely stops active speech playback and terminates process."""
        with self._proc_lock:
            if self._current_proc is not None:
                try:
                    self._current_proc.terminate()
                    self._current_proc.wait(timeout=0.2)
                except Exception:
                    try:
                        self._current_proc.kill()
                    except Exception:
                        pass
                self._current_proc = None

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strips formatting symbols, markdown, and extra whitespace."""
        if not text:
            return ""
        cleaned = re.sub(r"[\*\_#`]", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned


class VoiceActivationManager:
    """
    Manages the lifecycle, permissions, background listener thread,
    query capture state machine, on-device transcription, and conversational
    LLM assistance for IntentSense voice activation.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        capture_timeout: Optional[float] = None,
        silence_timeout: float = 1.2,
        initial_speech_timeout: float = 8.0,
        max_capture_duration: float = 30.0,
        speech_rms_threshold: float = 0.012,
        permission_granted: bool = False,
        enable_mic_stream: bool = True,
        enable_tts: bool = True,
        tts: Optional[LocalTextToSpeech] = None,
        llm_assistant: Optional[Any] = None,
        transcriber: Optional[LocalSpeechTranscriber] = None,
        get_app_state_summary: Optional[Callable[[], Dict[str, Any]]] = None,
        response_display_duration: float = 6.0,
        on_wake_callback: Optional[Callable[[], None]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.chunk_size: int = chunk_size
        self.silence_timeout: float = silence_timeout
        self.initial_speech_timeout: float = capture_timeout if capture_timeout is not None else initial_speech_timeout
        self.max_capture_duration: float = max_capture_duration
        self.speech_rms_threshold: float = speech_rms_threshold
        self.enable_mic_stream: bool = enable_mic_stream
        self.on_wake_callback: Optional[Callable[[], None]] = on_wake_callback
        self.llm_assistant: Optional[Any] = llm_assistant
        self.transcriber: LocalSpeechTranscriber = transcriber or LocalSpeechTranscriber(sample_rate=sample_rate)
        self.tts: LocalTextToSpeech = tts or LocalTextToSpeech(enabled=enable_tts)
        self.get_app_state_summary: Optional[Callable[[], Dict[str, Any]]] = get_app_state_summary
        self.response_display_duration: float = response_display_duration
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic

        self._has_permission: bool = permission_granted
        self._state: VoiceState = VoiceState.STANDBY if permission_granted else VoiceState.DISABLED
        self._wake_detector: LocalAcousticWakeDetector = LocalAcousticWakeDetector(sample_rate=sample_rate)

        self._capture_start_time: float = 0.0
        self._timeout_start_time: float = 0.0
        self._response_start_time: float = 0.0
        self._is_running: bool = False
        self._audio_thread: Optional[threading.Thread] = None
        self._stream: Optional[Any] = None
        self._lock: threading.RLock = threading.RLock()

        max_chunk_capacity = max(1000, int(math.ceil(self.max_capture_duration * self.sample_rate / self.chunk_size)) + 200)
        self._captured_chunks: Deque[np.ndarray] = deque(maxlen=max_chunk_capacity)

        # Dynamic VAD & speech tracking
        self._current_frame_time: Optional[float] = None
        self._has_speech_started: bool = False
        self._speech_ended: bool = False
        self._speech_start_time: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        self._accumulated_silence_samples: int = 0

        self._current_question: Optional[str] = None
        self._latest_response: Optional[Any] = None
        self._timeout_label: str = "LISTENING TIMEOUT"

        if permission_granted:
            self.start()

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def capture_timeout(self) -> float:
        return self.initial_speech_timeout

    @capture_timeout.setter
    def capture_timeout(self, val: float) -> None:
        self.initial_speech_timeout = val

    @property
    def has_speech_started(self) -> bool:
        with self._lock:
            return self._has_speech_started
    @property
    def state(self) -> VoiceState:
        with self._lock:
            return self._state

    @property
    def has_permission(self) -> bool:
        return self._has_permission

    @property
    def is_listening(self) -> bool:
        with self._lock:
            return (
                self._state in (
                    VoiceState.STANDBY,
                    VoiceState.WAKE_DETECTED,
                    VoiceState.CAPTURING_QUERY,
                )
                and not self.is_speaking
            )

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self.tts.is_speaking or self._state == VoiceState.SPEAKING

    @property
    def is_capturing(self) -> bool:
        with self._lock:
            return self._state in (VoiceState.WAKE_DETECTED, VoiceState.CAPTURING_QUERY)

    @property
    def is_processing(self) -> bool:
        with self._lock:
            return self._state in (VoiceState.TRANSCRIBING, VoiceState.THINKING)

    @property
    def latest_response(self) -> Optional[Any]:
        with self._lock:
            return self._latest_response

    @property
    def current_question(self) -> Optional[str]:
        with self._lock:
            return self._current_question

    # -------------------------------------------------------------------------
    # Permission & State Controls
    # -------------------------------------------------------------------------
    def grant_permission(self) -> bool:
        """Explicitly grants microphone permission and starts background listening."""
        with self._lock:
            self._has_permission = True
            self._state = VoiceState.STANDBY
        self.start()
        return True

    def revoke_permission(self) -> None:
        """Revokes microphone permission and cleanly shuts down audio capture."""
        self.stop()
        self.tts.stop()
        with self._lock:
            self._has_permission = False
            self._state = VoiceState.DISABLED
            self._wake_detector.reset()
            self._captured_chunks.clear()
            self._has_speech_started = False
            self._speech_ended = False
            self._speech_start_time = None
            self._last_speech_time = None
            self._silence_start_time = None
            self._accumulated_silence_samples = 0
            self._current_question = None
            self._latest_response = None

    def toggle_permission(self) -> bool:
        """Toggles microphone permission on or off."""
        if self._has_permission:
            self.revoke_permission()
            return False
        else:
            self.grant_permission()
            return True

    def trigger_wake_word(self, timestamp: Optional[float] = None) -> None:
        """Manually triggers wake-word detection (used for simulation and tests)."""
        with self._lock:
            if not self._has_permission:
                return
            self._state = VoiceState.WAKE_DETECTED
            now = self._clock() if timestamp is None else timestamp
            self._capture_start_time = now
            self._current_frame_time = now
            self._captured_chunks.clear()
            self._has_speech_started = False
            self._speech_ended = False
            self._speech_start_time = None
            self._last_speech_time = None
            self._silence_start_time = None
            self._accumulated_silence_samples = 0

        if self.on_wake_callback is not None:
            try:
                self.on_wake_callback()
            except Exception:
                pass

    def cancel(self) -> None:
        """Cancels any active voice listening, transcription, or response session."""
        self.tts.stop()
        with self._lock:
            if self._has_permission:
                self._state = VoiceState.STANDBY
                self._wake_detector.reset()
                self._captured_chunks.clear()
                self._has_speech_started = False
                self._speech_ended = False
                self._speech_start_time = None
                self._last_speech_time = None
                self._silence_start_time = None
                self._accumulated_silence_samples = 0
                self._current_question = None
                self._latest_response = None

    def reset_conversation(self) -> None:
        """Clears multi-turn conversation history and active QA session."""
        with self._lock:
            if self.llm_assistant is not None and hasattr(self.llm_assistant, "reset_conversation_history"):
                self.llm_assistant.reset_conversation_history()
            self._current_question = None
            self._latest_response = None
            print("[Voice Debug] Voice conversation history reset.")

    # -------------------------------------------------------------------------
    # Frame Loop Update (Timeout & State Progression)
    # -------------------------------------------------------------------------
    def update(self, current_timestamp: Optional[float] = None) -> VoiceHudStatus:
        """
        Called once per frame in the main application loop.
        Manages capture countdown timers, async state advancement, and response display.
        """
        now = self._clock() if current_timestamp is None else current_timestamp

        with self._lock:
            if not self._has_permission:
                self._state = VoiceState.DISABLED
                return VoiceHudStatus(
                    state=VoiceState.DISABLED,
                    label="MIC OFF (Press 'V' to enable)",
                    color_bgr=(120, 120, 120),
                    time_remaining=0.0,
                    has_permission=False,
                    is_capturing=False,
                    is_processing=False,
                )

            self._current_frame_time = now

            # Auto-advance WAKE_DETECTED -> CAPTURING_QUERY
            if self._state == VoiceState.WAKE_DETECTED:
                self._state = VoiceState.CAPTURING_QUERY
                self._capture_start_time = now
                print(f"[Voice Debug] VoiceState transitioned: WAKE_DETECTED -> CAPTURING_QUERY (no-speech timeout: {self.initial_speech_timeout:.1f}s, max: {self.max_capture_duration:.1f}s)")

            # Check capture timeout / dynamic VAD completion
            if self._state == VoiceState.CAPTURING_QUERY:
                elapsed = now - self._capture_start_time

                # 1. Condition A: Speech was detected, followed by continuous silence >= silence_timeout (1.2s)
                if self._has_speech_started:
                    silence_by_samples = (self._accumulated_silence_samples / self.sample_rate) >= self.silence_timeout
                    silence_by_clock = (self._last_speech_time is not None) and ((now - self._last_speech_time) >= self.silence_timeout)
                    if self._speech_ended or silence_by_samples or silence_by_clock:
                        print(f"[Voice Debug] Continuous silence of >= {self.silence_timeout:.1f}s detected after speech. Finalizing capture.")
                        return self._finalize_capture_and_transcribe(now)

                # 2. Condition B: Max recording duration reached (safety fallback at 30.0s)
                if elapsed >= self.max_capture_duration:
                    print(f"[Voice Debug] Max recording duration reached ({elapsed:.1f}s >= {self.max_capture_duration:.1f}s). Processing captured speech.")
                    if self._has_speech_started and len(self._captured_chunks) > 0:
                        return self._finalize_capture_and_transcribe(now)
                    else:
                        return self._handle_capture_timeout(now, "LISTENING TIMEOUT (No Speech Detected)")

                # 3. Condition C: Initial speech timeout (no speech within initial_speech_timeout after wake)
                if not self._has_speech_started:
                    if elapsed >= self.initial_speech_timeout:
                        print(f"[Voice Debug] Initial speech timeout reached ({elapsed:.1f}s >= {self.initial_speech_timeout:.1f}s without speech).")
                        return self._handle_capture_timeout(now, "LISTENING TIMEOUT (No Speech Detected)")

                # Still capturing:
                if not self._has_speech_started:
                    remaining = max(0.0, self.initial_speech_timeout - elapsed)
                    label = f"CAPTURING QUESTION ({remaining:.1f}s)"
                else:
                    remaining = max(0.0, self.max_capture_duration - elapsed)
                    label = f"CAPTURING QUESTION ({elapsed:.1f}s)"

                return VoiceHudStatus(
                    state=VoiceState.CAPTURING_QUERY,
                    label=label,
                    color_bgr=(0, 255, 0),
                    time_remaining=remaining,
                    has_permission=True,
                    is_capturing=True,
                    is_processing=False,
                )

            if self._state == VoiceState.TRANSCRIBING:
                return VoiceHudStatus(
                    state=VoiceState.TRANSCRIBING,
                    label="VOICE: Transcribing speech...",
                    color_bgr=(0, 215, 255),
                    time_remaining=0.0,
                    has_permission=True,
                    is_capturing=False,
                    is_processing=True,
                )

            if self._state == VoiceState.THINKING:
                q_label = f"\"{self._current_question[:25]}\"..." if self._current_question else ""
                return VoiceHudStatus(
                    state=VoiceState.THINKING,
                    label=f"VOICE: Thinking ({q_label})",
                    color_bgr=(0, 215, 255),
                    time_remaining=0.0,
                    has_permission=True,
                    is_capturing=False,
                    is_processing=True,
                    recognized_question=self._current_question,
                )

            if self._state == VoiceState.SPEAKING:
                elapsed_resp = now - self._response_start_time
                resp_ans = self._latest_response.answer if self._latest_response else ""
                resp_prov = self._latest_response.provider if self._latest_response else ""
                resp_q = self._latest_response.question if self._latest_response else self._current_question
                return VoiceHudStatus(
                    state=VoiceState.SPEAKING,
                    label="AI COACH SPEAKING ([ESC] to stop)",
                    color_bgr=(0, 255, 255),
                    time_remaining=max(0.0, self.response_display_duration - elapsed_resp),
                    has_permission=True,
                    is_capturing=False,
                    is_processing=False,
                    is_speaking=True,
                    recognized_question=resp_q,
                    response_text=resp_ans,
                    response_provider=resp_prov,
                )

            if self._state == VoiceState.RESPONDING:
                elapsed_resp = now - self._response_start_time
                if elapsed_resp >= self.response_display_duration:
                    self._state = VoiceState.STANDBY
                    self._current_question = None
                    self._latest_response = None
                else:
                    remaining_resp = max(0.0, self.response_display_duration - elapsed_resp)
                    resp_ans = self._latest_response.answer if self._latest_response else ""
                    resp_prov = self._latest_response.provider if self._latest_response else ""
                    resp_q = self._latest_response.question if self._latest_response else self._current_question
                    return VoiceHudStatus(
                        state=VoiceState.RESPONDING,
                        label=f"VOICE ANSWER ({remaining_resp:.1f}s)",
                        color_bgr=(0, 255, 120),
                        time_remaining=remaining_resp,
                        has_permission=True,
                        is_capturing=False,
                        is_processing=False,
                        recognized_question=resp_q,
                        response_text=resp_ans,
                        response_provider=resp_prov,
                    )

            # Auto-reset TIMEOUT -> STANDBY after brief 1.2s flash
            if self._state == VoiceState.TIMEOUT:
                if (now - self._timeout_start_time) >= 1.2:
                    self._state = VoiceState.STANDBY
                else:
                    return VoiceHudStatus(
                        state=VoiceState.TIMEOUT,
                        label=self._timeout_label,
                        color_bgr=(0, 140, 255),
                        time_remaining=0.0,
                        has_permission=True,
                        is_capturing=False,
                        is_processing=False,
                    )

            if self._state == VoiceState.ERROR:
                return VoiceHudStatus(
                    state=VoiceState.ERROR,
                    label="MIC UNAVAILABLE (No Audio Device)",
                    color_bgr=(100, 100, 255),
                    time_remaining=0.0,
                    has_permission=True,
                    is_capturing=False,
                    is_processing=False,
                )

            # Normal STANDBY passive listener
            return VoiceHudStatus(
                state=VoiceState.STANDBY,
                label="VOICE STANDBY — 'Hello Sense'",
                color_bgr=(0, 215, 255),
                time_remaining=0.0,
                has_permission=True,
                is_capturing=False,
                is_processing=False,
            )

    # -------------------------------------------------------------------------
    # Asynchronous Transcription & Question Answering
    # -------------------------------------------------------------------------
    def _finalize_capture_and_transcribe(self, now: float) -> VoiceHudStatus:
        chunks = list(self._captured_chunks)
        self._captured_chunks.clear()
        self._wake_detector.reset()
        self._has_speech_started = False
        self._speech_ended = False
        self._speech_start_time = None
        self._last_speech_time = None
        self._silence_start_time = None
        self._accumulated_silence_samples = 0

        if not chunks:
            return self._handle_capture_timeout(now, "LISTENING TIMEOUT (No Speech Detected)")

        total_samples = sum(len(c) for c in chunks)
        dur_s = total_samples / self.sample_rate
        print(f"[Voice Debug] Finalized audio capture: {len(chunks)} chunks, {total_samples} samples ({dur_s:.2f}s). State -> TRANSCRIBING.")
        self._state = VoiceState.TRANSCRIBING
        threading.Thread(
            target=self._run_transcription_and_qa,
            args=(chunks,),
            daemon=True,
            name="IntentSense-VoiceTranscribe",
        ).start()

        return VoiceHudStatus(
            state=VoiceState.TRANSCRIBING,
            label="VOICE: Transcribing speech...",
            color_bgr=(0, 215, 255),
            time_remaining=0.0,
            has_permission=True,
            is_capturing=False,
            is_processing=True,
        )

    def _handle_capture_timeout(self, now: float, label: str) -> VoiceHudStatus:
        self._captured_chunks.clear()
        self._wake_detector.reset()
        self._has_speech_started = False
        self._speech_ended = False
        self._speech_start_time = None
        self._last_speech_time = None
        self._silence_start_time = None
        self._accumulated_silence_samples = 0

        self._state = VoiceState.TIMEOUT
        self._timeout_start_time = now
        self._timeout_label = label
        print(f"[Voice Debug] Capture timeout: {label}. State -> TIMEOUT.")
        return VoiceHudStatus(
            state=VoiceState.TIMEOUT,
            label=label,
            color_bgr=(0, 140, 255),
            time_remaining=0.0,
            has_permission=True,
            is_capturing=False,
            is_processing=False,
        )

    def _run_transcription_and_qa(self, chunks: List[np.ndarray]) -> None:
        """Runs local STT and dispatches question answering asynchronously."""
        print(
            f"[Voice Debug] STT check: transcriber.is_available={self.transcriber.is_available}, "
            f"binary={bool(self.transcriber.binary_path)}, model={bool(self.transcriber.model_path)}"
        )
        if (
            not self.transcriber.is_available
            and self.transcriber._simulated_text is None
            and self.transcriber._custom_transcriber_fn is None
        ):
            print("[Voice Debug] STT unavailable: whisper.cpp binary or ggml-base.en.bin model missing.")
            with self._lock:
                self._state = VoiceState.TIMEOUT
                self._timeout_start_time = self._clock()
                self._timeout_label = "STT UNAVAILABLE (whisper.cpp missing)"
            return

        try:
            q_text = self.transcriber.transcribe_chunks(chunks)
        except Exception as e:
            print(f"[Voice Debug] STT transcription exception: {e}")
            q_text = None

        if not q_text or not q_text.strip():
            print("[Voice Debug] STT returned empty or None transcript. State -> TIMEOUT.")
            with self._lock:
                self._state = VoiceState.TIMEOUT
                self._timeout_start_time = self._clock()
                self._timeout_label = "LISTENING TIMEOUT (No Speech Detected)"
            return

        clean_q = q_text.strip()
        print(f"[Voice Debug] Actual Whisper transcript: \"{clean_q}\" (length={len(clean_q)})")
        with self._lock:
            self._state = VoiceState.THINKING
            self._current_question = clean_q

        self._dispatch_qa(self._current_question)

    def _dispatch_qa(self, question: str) -> None:
        """Dispatches question to LLMAssistant or offline provider."""
        summary = self.get_app_state_summary() if self.get_app_state_summary is not None else {}
        has_key = bool(self.llm_assistant.http_provider.api_key) if self.llm_assistant else False
        selected_prov = "Gemini" if (self.llm_assistant and self.llm_assistant.has_active_remote) else "Offline"
        print(f"[Voice Debug] GEMINI_API_KEY configured: {has_key}")
        print(f"[Voice Debug] Selected provider: {selected_prov}")
        print(f"[Voice Debug] Dispatching voice QA: question=\"{question}\", provider_mode={selected_prov}")
        if self.llm_assistant is not None:
            self.llm_assistant.answer_voice_question_async(
                question=question,
                app_state_summary=summary,
                callback=self._on_llm_response,
            )
        else:
            from core.llm_assistant import OfflineRuleBasedProvider
            offline_prov = OfflineRuleBasedProvider()
            resp = offline_prov.answer_voice_question(question, summary)
            self._on_llm_response(resp)

    def _on_llm_response(self, response: Any) -> None:
        """Callback invoked when LLM response is ready."""
        if response is None:
            print("[Voice Debug] Voice QA response was None; generating fallback response.")
            from core.llm_assistant import OfflineRuleBasedProvider
            offline_prov = OfflineRuleBasedProvider()
            summary = self.get_app_state_summary() if self.get_app_state_summary is not None else {}
            response = offline_prov.answer_voice_question(self._current_question or "help", summary)

        ans_len = len(response.answer) if hasattr(response, "answer") else 0
        prov = getattr(response, "provider", "unknown")
        speech_text = getattr(response, "answer", "")
        print(f"[Voice Debug] Final HUD response: provider={prov}, text=\"{speech_text}\" (length={ans_len})")

        with self._lock:
            if self._state not in (VoiceState.THINKING, VoiceState.TRANSCRIBING):
                print(f"[Voice Debug] Callback ignored: state was {self._state.value} (expected THINKING/TRANSCRIBING).")
                return
            self._latest_response = response
            self._response_start_time = self._clock()

            if self.tts.enabled and self.tts.is_available and speech_text:
                self._state = VoiceState.SPEAKING
                print(f"[Voice Debug] Starting spoken AI response via TTS ({len(speech_text)} chars). State -> SPEAKING.")
                self.tts.speak_async(speech_text, on_done=self._on_tts_done)
            else:
                self._state = VoiceState.RESPONDING
                print("[Voice Debug] TTS unavailable, disabled, or empty text. State -> RESPONDING.")

    def _on_tts_done(self) -> None:
        """Invoked when speech synthesis completes."""
        print("[Voice Debug] TTS playback finished. State transitioned: SPEAKING -> RESPONDING.")
        with self._lock:
            if self._state == VoiceState.SPEAKING:
                self._state = VoiceState.RESPONDING
                self._response_start_time = self._clock()
            self._wake_detector.reset()

    def ask_question_async(self, question: str) -> None:
        """Direct programmatic voice question submission (asynchronous)."""
        clean_q = question.strip()
        if not clean_q:
            return
        with self._lock:
            if not self._has_permission:
                return
            self._state = VoiceState.THINKING
            self._current_question = clean_q
        self._dispatch_qa(clean_q)

    def ask_question(self, question: str) -> Optional[Any]:
        """Direct programmatic voice question submission (synchronous, testing)."""
        clean_q = question.strip()
        if not clean_q:
            return None
        with self._lock:
            self._current_question = clean_q
        summary = self.get_app_state_summary() if self.get_app_state_summary is not None else {}
        if self.llm_assistant is not None:
            resp = self.llm_assistant.answer_voice_question(clean_q, summary)
        else:
            from core.llm_assistant import OfflineRuleBasedProvider
            offline_prov = OfflineRuleBasedProvider()
            resp = offline_prov.answer_voice_question(clean_q, summary)
        with self._lock:
            self._state = VoiceState.RESPONDING
            self._latest_response = resp
            self._response_start_time = self._clock()
        return resp

    # -------------------------------------------------------------------------
    # Audio Feed & Processing
    # -------------------------------------------------------------------------
    def feed_audio(self, chunk: np.ndarray, timestamp: Optional[float] = None) -> bool:
        """
        Directly feeds an audio chunk into the wake-word engine.
        Allows deterministic offline testing without requiring a physical microphone.
        """
        now = self._clock() if timestamp is None else timestamp
        with self._lock:
            if not self._has_permission:
                return False

            # Prevent microphone wake detection from hearing the assistant's own voice
            if self.is_speaking or self._state == VoiceState.SPEAKING:
                return False

            if self._state in (VoiceState.WAKE_DETECTED, VoiceState.CAPTURING_QUERY):
                if chunk is not None and len(chunk) > 0:
                    self._captured_chunks.append(chunk.copy())
                    chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))
                    t = now if timestamp is not None else (self._current_frame_time if self._current_frame_time is not None else now)
                    if chunk_rms >= self.speech_rms_threshold:
                        if not self._has_speech_started:
                            self._has_speech_started = True
                            self._speech_start_time = t
                            print(f"[Voice Debug] Speech start detected! RMS={chunk_rms:.4f} >= {self.speech_rms_threshold:.4f}.")
                        self._last_speech_time = t
                        self._accumulated_silence_samples = 0
                        self._silence_start_time = None
                    else:
                        if self._has_speech_started:
                            self._accumulated_silence_samples += len(chunk)
                            if self._silence_start_time is None:
                                self._silence_start_time = t
                            if (self._accumulated_silence_samples / self.sample_rate) >= self.silence_timeout:
                                self._speech_ended = True
                return False

            if self._state in (VoiceState.STANDBY, VoiceState.ERROR):
                detected = self._wake_detector.process_chunk(chunk)
                if detected:
                    self._state = VoiceState.WAKE_DETECTED
                    self._capture_start_time = now if timestamp is not None else (self._current_frame_time if self._current_frame_time is not None else self._clock())
                    self._captured_chunks.clear()
                    self._has_speech_started = False
                    self._speech_ended = False
                    self._speech_start_time = None
                    self._last_speech_time = None
                    self._silence_start_time = None
                    self._accumulated_silence_samples = 0
                    print("[Voice Debug] Wake-word detected! VoiceState transitioned: STANDBY -> WAKE_DETECTED")
                    if self.on_wake_callback is not None:
                        try:
                            print("[Voice Debug] Invoking on_wake_callback...")
                            self.on_wake_callback()
                        except Exception as e:
                            print(f"[Voice Debug] on_wake_callback error: {e}")
                    return True

        return False

    # -------------------------------------------------------------------------
    # Background Thread & Audio Stream Management
    # -------------------------------------------------------------------------
    def start(self) -> None:
        """Starts asynchronous audio streaming on a daemon thread."""
        if not self._has_permission or self._is_running or not self.enable_mic_stream:
            return

        self._is_running = True
        self._audio_thread = threading.Thread(
            target=self._run_audio_listener,
            daemon=True,
            name="IntentSense-VoiceListener",
        )
        self._audio_thread.start()

    def _run_audio_listener(self) -> None:
        """Asynchronous worker loop continuously polling the microphone."""
        if not SOUNDDEVICE_AVAILABLE or sd is None:
            print("[Voice Debug] sounddevice library unavailable or not importable.")
            with self._lock:
                self._state = VoiceState.ERROR
            return

        # Check if an input device exists
        try:
            devices = sd.query_devices()
            has_input = any(d.get("max_input_channels", 0) > 0 for d in devices) if devices else False
            if not has_input:
                print("[Voice Debug] No audio input devices detected.")
                with self._lock:
                    self._state = VoiceState.ERROR
                return
            default_dev_idx = sd.default.device[0] if hasattr(sd.default, "device") else -1
            dev_info = sd.query_devices(default_dev_idx) if default_dev_idx >= 0 else {}
            dev_name = dev_info.get("name", "Default") if isinstance(dev_info, dict) else "Default"
            print(f"[Voice Debug] Audio Device: {dev_name} (Index: {default_dev_idx}, SR: {self.sample_rate}Hz)")
        except Exception as e:
            print(f"[Voice Debug] Audio device query error: {e}")
            with self._lock:
                self._state = VoiceState.ERROR
            return

        def _audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if not self._is_running:
                return
            if status:
                print(f"[Voice Debug] Audio callback status flag: {status}")
            rms = float(np.sqrt(np.mean(indata ** 2)))
            if rms > 0.008:
                print(f"[Voice Debug] Audio chunk received: RMS={rms:.5f}")
            self.feed_audio(indata)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype="float32",
                callback=_audio_callback,
            ):
                print(f"[Voice Debug] sd.InputStream started successfully ({self.sample_rate}Hz, chunk={self.chunk_size}).")
                while self._is_running:
                    time.sleep(0.05)
        except Exception as e:
            print(f"[Voice Debug] sd.InputStream error: {e}")
            with self._lock:
                self._state = VoiceState.ERROR

    def stop(self) -> None:
        """Safely stops the audio listener thread."""
        self._is_running = False
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=0.5)
        self._audio_thread = None

    def close(self) -> None:
        """Full cleanup of voice resources."""
        self.stop()
        self.tts.stop()
        self.revoke_permission()
