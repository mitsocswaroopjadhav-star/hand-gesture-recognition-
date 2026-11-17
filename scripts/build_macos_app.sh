#!/usr/bin/env bash
# scripts/build_macos_app.sh
# ==========================
# Automated build script for compiling IntentSense into a standalone
# macOS Application (.app) on Apple Silicon (ARM64).
#
# Generates:
#   dist/IntentSense.app
#   dist/IntentSense-macos-arm64.zip

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPEC_FILE="${REPO_DIR}/packaging/IntentSense.spec"
DIST_DIR="${REPO_DIR}/dist"
BUILD_DIR="${REPO_DIR}/build"
APP_BUNDLE="${DIST_DIR}/IntentSense.app"
ZIP_ARCHIVE="${DIST_DIR}/IntentSense-macos-arm64.zip"

echo "=========================================================="
echo "  IntentSense — macOS Application Build Pipeline (ARM64)"
echo "=========================================================="

cd "${REPO_DIR}"

# 1. Architecture Check
ARCH="$(uname -m)"
echo "[1/7] Host architecture: ${ARCH}"
if [ "${ARCH}" != "arm64" ]; then
    echo "  ⚠️ Warning: Building on non-arm64 architecture (${ARCH}). Target is Apple Silicon."
fi

# 2. Virtual Environment Check
if [ -f "${REPO_DIR}/venv/bin/pyinstaller" ]; then
    PYINSTALLER="${REPO_DIR}/venv/bin/pyinstaller"
    PYTHON="${REPO_DIR}/venv/bin/python"
elif command -v pyinstaller >/dev/null 2>&1; then
    PYINSTALLER="$(command -v pyinstaller)"
    PYTHON="$(command -v python3)"
else
    echo "❌ Error: PyInstaller not found. Please install requirements-build.txt."
    exit 1
fi
echo "[2/7] Using PyInstaller: ${PYINSTALLER}"

# 3. Model & Resource Validation
echo "[3/7] Validating required model files..."
if [ ! -f "${REPO_DIR}/hand_landmarker.task" ]; then
    echo "  ❌ Error: hand_landmarker.task not found in project root."
    exit 1
fi
if [ ! -f "${REPO_DIR}/models/gesture_model.pkl" ]; then
    echo "  ❌ Error: models/gesture_model.pkl not found."
    exit 1
fi
echo "  ✓ Hand landmarker model present: hand_landmarker.task"
echo "  ✓ Gesture classifier present: models/gesture_model.pkl"

if [ -f "${REPO_DIR}/models/ggml-base.en.bin" ]; then
    echo "  ✓ Whisper GGML model present: models/ggml-base.en.bin"
else
    echo "  ℹ️ Notice: models/ggml-base.en.bin not found. Building without bundled speech model."
    echo "    (User can run scripts/setup_whisper.sh anytime to install it locally)."
fi

# Ensure whisper dylibs have proper rpaths in bin/ if present
if [ -f "${REPO_DIR}/bin/whisper-cli" ]; then
    echo "  ✓ Whisper CLI binary present: bin/whisper-cli"
    install_name_tool -add_rpath "@loader_path" "${REPO_DIR}/bin/whisper-cli" 2>/dev/null || true
    install_name_tool -add_rpath "@executable_path" "${REPO_DIR}/bin/whisper-cli" 2>/dev/null || true
fi

# 4. Clean previous build artifacts & configure local cache
echo "[4/7] Cleaning previous build artifacts..."
rm -rf "${BUILD_DIR}" "${DIST_DIR}"
mkdir -p "${BUILD_DIR}/cache"
export PYINSTALLER_CONFIG_DIR="${BUILD_DIR}/cache"

# 5. Run PyInstaller
echo "[5/7] Running PyInstaller build (ONEDIR + WINDOWED)..."
"${PYINSTALLER}" --noconfirm --clean --workpath "${BUILD_DIR}" --distpath "${DIST_DIR}" "${SPEC_FILE}"

if [ ! -d "${APP_BUNDLE}" ]; then
    echo "❌ Error: Application bundle ${APP_BUNDLE} was not created."
    exit 1
fi
echo "  ✓ Successfully generated application bundle: ${APP_BUNDLE}"

# 6. Apply Ad-Hoc Code Signature
echo "[6/7] Applying macOS ad-hoc code signature..."
# Remove any quarantine flags from bundled dylibs
xattr -cr "${APP_BUNDLE}" 2>/dev/null || true
codesign --force --deep --sign - "${APP_BUNDLE}"
echo "  ✓ Ad-hoc code signature verified:"
codesign --verify --verbose "${APP_BUNDLE}"

# 7. Create Distributable ZIP Archive
echo "[7/7] Packaging distributable archive..."
# Use ditto to preserve Apple bundle resource forks, symlinks, and permissions
ditto -c -k --keepParent "${APP_BUNDLE}" "${ZIP_ARCHIVE}"

APP_SIZE="$(du -sh "${APP_BUNDLE}" | cut -f1)"
ZIP_SIZE="$(du -sh "${ZIP_ARCHIVE}" | cut -f1)"

echo "=========================================================="
echo "✅ Build Complete!"
echo "  Application Bundle : ${APP_BUNDLE} (${APP_SIZE})"
echo "  Distributable ZIP  : ${ZIP_ARCHIVE} (${ZIP_SIZE})"
echo ""
echo "Installation Instructions:"
echo "  1. Unzip ${ZIP_ARCHIVE} (or copy ${APP_BUNDLE} to /Applications)."
echo "  2. On first launch, macOS Gatekeeper may prompt to confirm opening."
echo "     If needed, right-click and select 'Open', or run:"
echo "     xattr -d com.apple.quarantine /Applications/IntentSense.app"
echo "  3. Grant Camera and Microphone permissions in macOS System Settings."
echo "=========================================================="
