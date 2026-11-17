# Contributing to IntentSense

Thank you for your interest in contributing to **IntentSense**! We welcome bug reports, feature proposals, and pull requests that maintain our high standards of safety, privacy, and touchless ergonomics.

---

## Architecture Principles

1. **Safety First**:
   - The default mode is always **Safe DRY-RUN**. Live operating system execution requires explicit opt-in via `--live`.
   - Voice commands cannot execute destructive OS actions or bypass hand-gesture safety gates.

2. **Privacy by Design**:
   - Zero raw camera frames or 42D biometric landmark vectors are ever transmitted to external networks or LLM APIs.
   - Speech recognition operates completely offline and on-device via `whisper.cpp`.
   - Online AI features (Google Gemini) are optional. A deterministic local rule-based engine provides full offline functionality.

3. **Touchless Ergonomics**:
   - State transitions require intentional unlock poses (`OpenPalm` hold for 1.0s) to prevent false positives.
   - Commands are protected by adaptive confidence thresholds, temporal smoothing, and debounce cooldowns.
   - Inactivity timeouts automatically return the system to `LOCKED`.

---

## Development Setup

### 1. Prerequisites
- macOS 12.0+ (Tested extensively on Apple Silicon ARM64 / macOS 14+).
- Python 3.10 to 3.13.
- CMake (for building local `whisper.cpp` speech recognition).

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/mitsocswaroopjadhav-star/hand-gesture-recognition-.git
cd hand-gesture-recognition-

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core runtime dependencies
pip install -r requirements.txt

# (Optional) For building macOS application bundles
pip install -r requirements-build.txt
```

### 3. Speech Recognition Setup (Optional for voice features)
To enable local, on-device voice wake-word ("Hello Sense") and question answering:
```bash
bash scripts/setup_whisper.sh
```
This builds the `whisper-cli` executable in `bin/` and downloads the 141 MB `ggml-base.en.bin` model to `models/`.

---

## Running and Testing

### Run Headless Component Smoke Test
```bash
python app.py --smoke
```
This runs an end-to-end orchestration test verifying the state machine, classifier, context manager, action executor, onboarding, and offline voice Q&A without opening the physical webcam or microphone.

### Run Full Test Suite
```bash
python -m unittest discover -s . -p "test_*.py"
```

### Run Live Interactive Application
```bash
# Safe Dry-Run (visual HUD feedback only; no actual system actions)
python app.py

# Live Mode with Voice Enabled
python app.py --live --voice

# With optional Gemini API key for online conversational coaching
python app.py --voice --api-key "YOUR_GEMINI_KEY"
```

---

## Code Style & Testing Standards

- Maintain zero-warning clean code adhering to PEP 8.
- Every new feature or bug fix must include comprehensive unit tests.
- Never hardcode user home directory paths. Use `core.paths` helpers for bundle and user support directories.
- Ensure all subprocess calls use argument arrays (`list`), never `shell=True`.

---

## Pull Request Guidelines

1. Create a feature branch from `main` or the latest release branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Run the test suite and verify that all tests pass:
   ```bash
   python -m unittest discover -s . -p "test_*.py"
   python app.py --smoke
   ```
3. Commit with clear, descriptive commit messages.
4. Open a pull request against the repository with a summary of changes and test results.
