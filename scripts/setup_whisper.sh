#!/usr/bin/env bash
# scripts/setup_whisper.sh
# ========================
# Offline Setup for IntentSense Local Speech Recognition (whisper.cpp).
#
# Downloads, builds whisper.cpp and fetches the ggml-base.en.bin model.
# NOTE: Run this script separately; the application will NEVER auto-download
# models or binaries during startup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN_DIR="${REPO_DIR}/bin"
MODELS_DIR="${REPO_DIR}/models"
WHISPER_REPO_DIR="${REPO_DIR}/whisper.cpp"
MODEL_FILE="${MODELS_DIR}/ggml-base.en.bin"

echo "=========================================================="
echo "  IntentSense — Local Speech Recognition Setup (whisper.cpp)"
echo "=========================================================="

mkdir -p "${BIN_DIR}" "${MODELS_DIR}"

# 1. Clone or update whisper.cpp repository
if [ ! -d "${WHISPER_REPO_DIR}" ]; then
    echo "[1/4] Cloning whisper.cpp repository..."
    git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git "${WHISPER_REPO_DIR}"
else
    echo "[1/4] whisper.cpp source already present."
fi

# 2. Build whisper.cpp using cmake or make
echo "[2/4] Building whisper-cli executable..."
cd "${WHISPER_REPO_DIR}"

if command -v cmake >/dev/null 2>&1; then
    cmake -B build -DWHISPER_BUILD_EXAMPLES=ON
    cmake --build build --config Release -j 4
    if [ -f "build/bin/whisper-cli" ]; then
        cp "build/bin/whisper-cli" "${BIN_DIR}/whisper-cli"
        cp build/bin/lib*.dylib "${BIN_DIR}/" 2>/dev/null || true
    elif [ -f "build/bin/main" ]; then
        cp "build/bin/main" "${BIN_DIR}/whisper-cli"
        cp build/bin/lib*.dylib "${BIN_DIR}/" 2>/dev/null || true
    fi
    if [ "$(uname)" = "Darwin" ] && [ -f "${BIN_DIR}/whisper-cli" ]; then
        install_name_tool -add_rpath "@loader_path" "${BIN_DIR}/whisper-cli" 2>/dev/null || true
        install_name_tool -add_rpath "@executable_path" "${BIN_DIR}/whisper-cli" 2>/dev/null || true
    fi
else
    make -j 4
    if [ -f "main" ]; then
        cp "main" "${BIN_DIR}/whisper-cli"
    fi
fi

if [ -f "${BIN_DIR}/whisper-cli" ]; then
    chmod +x "${BIN_DIR}/whisper-cli"
    echo "  ✓ Installed binary at: ${BIN_DIR}/whisper-cli"
else
    echo "  ❌ Warning: Could not locate built binary. Please check compiler output."
fi

# 3. Download ggml-base.en.bin model
echo "[3/4] Downloading ggml-base.en.bin model (~142 MB)..."
if [ -f "${MODEL_FILE}" ]; then
    echo "  ✓ Model already present at: ${MODEL_FILE}"
else
    if [ -f "${WHISPER_REPO_DIR}/models/download-ggml-model.sh" ]; then
        bash "${WHISPER_REPO_DIR}/models/download-ggml-model.sh" base.en
        if [ -f "${WHISPER_REPO_DIR}/models/ggml-base.en.bin" ]; then
            cp "${WHISPER_REPO_DIR}/models/ggml-base.en.bin" "${MODEL_FILE}"
        fi
    else
        curl -L -o "${MODEL_FILE}" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
    fi
    echo "  ✓ Downloaded model to: ${MODEL_FILE}"
fi

# 4. Verify installation
echo "[4/4] Verifying local whisper installation..."
if [ -x "${BIN_DIR}/whisper-cli" ] && [ -f "${MODEL_FILE}" ]; then
    echo "=========================================================="
    echo "✅ Setup successful!"
    echo "  Binary: ${BIN_DIR}/whisper-cli"
    echo "  Model : ${MODEL_FILE}"
    echo "IntentSense local speech recognition is ready for offline use."
    echo "=========================================================="
else
    echo "=========================================================="
    echo "⚠️ Setup incomplete. Please inspect the output above."
    echo "=========================================================="
    exit 1
fi

