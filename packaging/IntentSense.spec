# -*- mode: python ; coding: utf-8 -*-
# packaging/IntentSense.spec
# ==========================
# PyInstaller specification for IntentSense on macOS Apple Silicon (ARM64).
# Generates a standalone, relocatable macOS Application bundle (ONEDIR + WINDOWED).

import glob
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None

SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
PROJECT_ROOT = os.path.dirname(SPEC_DIR)

# -----------------------------------------------------------------------------
# 1. Collect Application Data Files & Assets
# -----------------------------------------------------------------------------
datas = [
    (os.path.join(PROJECT_ROOT, "hand_landmarker.task"), "."),
    (os.path.join(PROJECT_ROOT, "models", "gesture_model.pkl"), "models"),
]

# Include whisper model if present locally
whisper_model = os.path.join(PROJECT_ROOT, "models", "ggml-base.en.bin")
if os.path.isfile(whisper_model):
    datas.append((whisper_model, "models"))

# MediaPipe & PortAudio data files
datas += collect_data_files("mediapipe")
datas += collect_data_files("_sounddevice_data")

# -----------------------------------------------------------------------------
# 2. Collect Binaries & Dynamic Libraries
# -----------------------------------------------------------------------------
binaries = []

# Whisper CLI and dynamic libraries
whisper_cli = os.path.join(PROJECT_ROOT, "bin", "whisper-cli")
if os.path.isfile(whisper_cli):
    binaries.append((whisper_cli, "bin"))

for dylib in glob.glob(os.path.join(PROJECT_ROOT, "bin", "*.dylib")):
    binaries.append((dylib, "bin"))

# MediaPipe & PortAudio dynamic libraries
binaries += collect_dynamic_libs("mediapipe")
binaries += collect_dynamic_libs("_sounddevice_data")
binaries += collect_dynamic_libs("sklearn")

# -----------------------------------------------------------------------------
# 3. Hidden Imports
# -----------------------------------------------------------------------------
hiddenimports = [
    "mediapipe",
    "mediapipe.tasks.python.vision",
    "sklearn",
    "sklearn.ensemble",
    "sklearn.tree",
    "sklearn.neighbors",
    "sounddevice",
    "_sounddevice_data",
    "pyautogui",
    "rubicon.objc",
    "Quartz",
    "AppKit",
    "Foundation",
    "actions",
    "actions.media_actions",
    "actions.presentation_actions",
    "core",
    "core.action_executor",
    "core.activation_state",
    "core.adaptive_threshold",
    "core.context_manager",
    "core.decision_explainer",
    "core.feature_extractor",
    "core.intent_engine",
    "core.llm_assistant",
    "core.onboarding",
    "core.paths",
    "core.personalized_gestures",
    "core.swipe_detector",
    "core.telemetry_aggregator",
    "core.temporal_filter",
    "core.voice_activation",
    "detector",
]
hiddenimports += collect_submodules("sklearn")

# -----------------------------------------------------------------------------
# 4. Analysis
# -----------------------------------------------------------------------------
a = Analysis(
    [os.path.join(PROJECT_ROOT, "app.py")],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# -----------------------------------------------------------------------------
# 5. Executable (ONEDIR)
# -----------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="IntentSense",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # Windowed GUI application
    disable_windowed_traceback=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)

# -----------------------------------------------------------------------------
# 6. Collect Folder
# -----------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="IntentSense",
)

# -----------------------------------------------------------------------------
# 7. macOS Application Bundle (.app)
# -----------------------------------------------------------------------------
app = BUNDLE(
    coll,
    name="IntentSense.app",
    icon=None,
    bundle_identifier="com.swaroopjadhav.intentsense",
    info_plist={
        "CFBundleDisplayName": "IntentSense",
        "CFBundleName": "IntentSense",
        "CFBundleIdentifier": "com.swaroopjadhav.intentsense",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundlePackageType": "APPL",
        "CFBundleSignature": "????",
        "NSCameraUsageDescription": "IntentSense requires camera access for real-time touchless hand gesture recognition.",
        "NSMicrophoneUsageDescription": "IntentSense requires microphone access for offline voice wake-word ('Hello Sense') and conversational question answering.",
        "NSAppleEventsUsageDescription": "IntentSense requires Apple Events permission to control system media playback and presentation slides in live mode.",
        "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.utilities",
        "LSMinimumSystemVersion": "12.0",
    },
)
