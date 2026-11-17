# IntentSense — Vision & Voice Multimodal AI Assistant

[![IntentSense Test Suite](https://github.com/mitsocswaroopjadhav-star/hand-gesture-recognition-/actions/workflows/tests.yml/badge.svg)](https://github.com/mitsocswaroopjadhav-star/hand-gesture-recognition-/actions/workflows/tests.yml)
[![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon%20(ARM64)-blue.svg)](https://www.apple.com/macos/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-brightgreen.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**IntentSense** is a multimodal, real-time hand-gesture and conversational voice assistant engineered natively for macOS (optimized for Apple Silicon M-series). Built on top of MediaPipe Tasks, scikit-learn Random Forests, on-device `whisper.cpp` Speech-to-Text, and optional Google Gemini conversational intelligence, IntentSense enables intuitive, zero-touch control over your media, presentations, and daily workflows.

---

## 🌟 Key Features

### 1. Vision & Gesture Recognition Pipeline (Phases 1 & 2)
- **High-Accuracy Pose Classification**: Recognizes core gestures (`OpenPalm`, `Fist`, `Pointing`, `ThumbsUp`, `ThumbsDown`, `Peace`) with **98.76% accuracy** on a 241-sample held-out test set.
- **Wrist-Relative Geometric Normalization**: Landmark coordinates are translated relative to wrist landmark 0 and scaled by palm span, ensuring invariance to camera distance, hand position, and user orientation.
- **Intentional State Machine**: Prevents accidental triggers via a deliberate 1.0s `OpenPalm` hold-to-unlock mechanism.
- **Dynamic Swipes & Inactivity Locking**: High-velocity horizontal swipes (`Swipe Left`, `Swipe Right`) for rapid media/slide navigation; automatically locks into a safe state after 30 seconds of inactivity.

### 2. Multi-Context Action Execution (Phase 3)
- **Media Mode**: Touchless control over Play/Pause, Volume Up/Down, and Next/Previous track.
- **Presentation Mode**: Next/Previous slide, start/stop slideshow, and touchless zoom.
- **Safe DRY-RUN Default**: Emits visual heads-up display (HUD) feedback without modifying system state. Live OS actions require explicit `--live` opt-in.
- **Typed Natural Language Commands**: Parse and validate user typed text commands against intent safety gates.

### 3. Touchless AI Coaching & Practice Mode (Phase 4)
- **Interactive Onboarding**: Step-by-step guidance designed to calibrate user hand size and introduce unlock holds.
- **Adaptive Threshold Policy**: Automatically adjusts confidence barriers based on lighting and temporal stability.
- **Live Struggle Detection**: Evaluates hesitation or hold deficits, offering real-time ergonomic HUD toasts.

### 4. Acoustic Voice Wake-Word Detection (Phase 5)
- **Local Wake-Word**: Listens continuously for **"Hello Sense"** using lightweight on-device acoustic RMS and Zero Crossing Rate (ZCR) syllable analysis.
- **Zero Cloud Leakage**: Audio is processed entirely in memory inside a circular buffer; raw audio is never logged or stored.

### 5. On-Device Speech Recognition & Multi-Turn AI Assistant (Phases 6 & 7)
- **Offline Whisper STT**: Transcribes questions locally using `whisper.cpp` and `ggml-base.en.bin` with Apple Silicon NEON & Accelerate BLAS acceleration.
- **Dynamic Voice Activity Detection (VAD)**: Listens naturally until the user finishes speaking (1.2s trailing silence cutoff), accommodating variable-length questions up to a 30s safety ceiling.
- **Dual AI Provider Architecture**:
  - **Online**: Google Gemini for rich general-purpose and technical conversational assistance.
  - **Offline Fallback**: Deterministic rule-based expert system answering all questions regarding IntentSense features, custom gestures, and locked states when offline.
- **Spoken AI Audio Responses**: Uses native macOS `/usr/bin/say` for smooth, non-blocking audio answers, with automatic microphone suppression during playback to prevent self-triggering.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Vision Pipeline
        CAM[Webcam Frame] --> MP[MediaPipe Hand Landmarker]
        MP --> NORM[Wrist Normalizer - 42D Vector]
        NORM --> RF[Random Forest Classifier]
        NORM --> SWIPE[Dynamic Swipe Detector]
        RF --> SM[Activation State Machine]
        SWIPE --> SM
    end

    subgraph Audio Pipeline
        MIC[Microphone InputStream] --> WAKE[Acoustic Wake Detector - Hello Sense]
        WAKE --> VAD[RMS Voice Activity Detector]
        VAD --> WHISPER[whisper.cpp Local STT]
    end

    subgraph Decision & Intelligence Engine
        SM --> IE[Intent Engine & Context Manager]
        IE --> EXPLAIN[Decision Explainer & AI Coach]
        WHISPER --> LLM[LLM Assistant]
        LLM -->|Online| GEMINI[Google Gemini API]
        LLM -->|Offline Fallback| RULES[Local Rule-Based Provider]
    end

    subgraph Action & Feedback
        IE -->|Safe Dry-Run / Live| EXEC[Action Executor - macOS Quartz / PyAutoGUI]
        EXPLAIN --> HUD[OpenCV Glassmorphic HUD]
        LLM --> HUD
        LLM --> TTS[macOS Native TTS - say]
    end
```

---

## 📊 Measured Evaluation & Accuracy

During development, the underlying Random Forest gesture classification model achieved:
- **Test Accuracy**: **98.76%** on a held-out test split of 241 landmark samples across all 6 core classes.
- **Feature Dimensionality**: 42 normalized float coordinates (21 hand joints $\times$ 2D).
- **Latency**: Sub-15ms landmark extraction and inference per frame on Apple Silicon M4.
- **Speech-to-Text Speed**: $\sim$1.1s for 11 seconds of audio using `whisper.cpp` CPU NEON/BLAS acceleration.

---

## 💻 Supported Hand Gestures

| Gesture | Media Context | Presentation Context | Custom Context |
| :--- | :--- | :--- | :--- |
| **Open Palm** | Unlock System (Hold 1.0s) | Unlock System (Hold 1.0s) | Unlock System (Hold 1.0s) |
| **Thumbs Up** | Volume Up | — | Configurable |
| **Fist** | Volume Down | — | Configurable |
| **Pointing** | — | Next Slide | Configurable |
| **Swipe Left** | Previous Track | Previous Slide | Configurable |
| **Swipe Right** | Next Track | Next Slide | Configurable |
| **Peace** | Custom Enrollment / Unassigned | Custom Enrollment / Unassigned | Custom Gesture Matching |

---

## 🚀 Quick Start (Development)

### 1. Prerequisites
- macOS 12.0+ (Tested on macOS 14+ / Apple Silicon M-Series).
- Python 3.10 – 3.13.
- Webcam and microphone.

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/mitsocswaroopjadhav-star/hand-gesture-recognition-.git
cd hand-gesture-recognition-

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Local Speech Recognition (Optional for Voice Features)
```bash
# Downloads and compiles whisper.cpp and fetches the ggml-base.en.bin model (~141 MB)
bash scripts/setup_whisper.sh
```

### 4. Running the Application
```bash
# 1. Run in Safe DRY-RUN mode (Default: visual feedback only, no OS actions executed)
python app.py

# 2. Run with Voice Wake-Word and Local STT enabled
python app.py --voice

# 3. Run with Live OS actions enabled (controls system volume, slides, etc.)
python app.py --live --voice

# 4. Run with optional Gemini online AI key
python app.py --voice --api-key "YOUR_GEMINI_API_KEY"

# 5. Run headless component orchestration smoke test
python app.py --smoke
```

---

## 📦 Standalone macOS Application (`IntentSense.app`)

IntentSense can be compiled into a standalone macOS Application bundle (`IntentSense.app`) for Apple Silicon ARM64 using PyInstaller:

### Build the Application
```bash
# Install packaging requirements
pip install -r requirements-build.txt

# Run the automated build script
bash scripts/build_macos_app.sh
```

The compiled application bundle will be created at `dist/IntentSense.app`, along with a distributable archive `dist/IntentSense-macos-arm64.zip`.

### macOS Permissions
When launching `IntentSense.app` on macOS, grant the following permissions when prompted:
1. **Camera**: Required for hand landmark tracking via OpenCV/MediaPipe (`NSCameraUsageDescription`).
2. **Microphone**: Required for "Hello Sense" wake-word detection and query capture (`NSMicrophoneUsageDescription`).
3. **Accessibility & Automation** *(Only required for `--live` mode)*:
   - Navigate to **System Settings > Privacy & Security > Accessibility** and ensure IntentSense or your terminal is toggled on.
   - Navigate to **System Settings > Privacy & Security > Automation** to allow media key simulations.

---

## 🔒 Privacy & User Data

- **Zero Cloud Vision Data**: Video frames and biometric vectors are processed in RAM and discarded immediately.
- **Persistent User Data**: User custom gestures, settings, and logs are saved exclusively in:
  `~/Library/Application Support/IntentSense/custom_gestures.json`
  The application bundle itself remains strictly read-only.
- **Secure Fallback**: If Gemini encounters authentication errors, rate limits, or network timeouts, IntentSense immediately falls back to its offline rule engine without exposing keys or interrupting the camera loop.

---

## 🧪 Testing

IntentSense includes a comprehensive test suite of 335+ unit and integration tests:

```bash
# Run all unit tests headlessly
MPLCONFIGDIR=/tmp/matplotlib python -m unittest discover -s . -p "test_*.py"

# Run voice conversation and assistant tests specifically
python -m unittest test_voice_conversation.py

# Run touchless onboarding tests
python -m unittest test_touchless_onboarding.py
```

---

## 📁 Repository Structure

```text
hand-gesture-recognition-/
├── .github/workflows/      # Automated CI test workflows (macOS ARM64 & Intel)
├── actions/                # OS-level action handlers (Media, Presentation)
├── core/                   # Core pipeline modules
│   ├── action_executor.py       # Safe action execution & rate limiting
│   ├── activation_state.py      # State machine (LOCKED / ACTIVE)
│   ├── adaptive_threshold.py    # Dynamic lighting/confidence thresholds
│   ├── context_manager.py       # Context switching (Media, Pres, Custom)
│   ├── decision_explainer.py    # Decision explanations & AI coaching
│   ├── feature_extractor.py     # 42D wrist-centered landmark normalization
│   ├── intent_engine.py         # Gesture-to-intent routing
│   ├── llm_assistant.py         # Gemini API client & offline rule fallback
│   ├── onboarding.py            # Touchless practice mode state engine
│   ├── paths.py                 # Centralized development & .app path resolver
│   ├── personalized_gestures.py # Custom gesture enrollment & 4-tier arbitration
│   ├── swipe_detector.py        # High-velocity temporal swipe tracker
│   ├── telemetry_aggregator.py  # User struggle pattern detection
│   └── voice_activation.py      # Acoustic wake-word, VAD, whisper.cpp & TTS
├── dataset/                # Anonymous landmark CSV dataset
├── models/                 # Model weights (gesture_model.pkl)
├── packaging/              # PyInstaller .spec configuration
├── scripts/                # Setup & build shell scripts
├── app.py                  # Main entry point & unified OpenCV HUD loop
├── detector.py             # MediaPipe Tasks HandLandmarker wrapper
├── hand_landmarker.task    # MediaPipe HandLandmarker model asset
├── requirements.txt        # Core runtime dependencies
├── requirements-build.txt  # Packaging build dependencies
└── README.md
```

---

## 👨‍💻 Author & Acknowledgements

- **Author**: Swaroop Jadhav
- **Technologies**: Google MediaPipe, OpenCV, scikit-learn, whisper.cpp by Georgi Gerganov, Google Gemini.
