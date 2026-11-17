"""
core/paths.py
=============
Centralized path resolution for IntentSense.
Handles both development mode and frozen macOS PyInstaller .app bundle mode.
Ensures runtime resources (models, tasks, binaries) are loaded from bundle,
and persistent user data (custom gestures, logs, settings) are written to
~/Library/Application Support/IntentSense on macOS.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def is_frozen() -> bool:
    """Returns True if running inside a frozen PyInstaller executable/bundle."""
    return getattr(sys, "frozen", False)


def get_bundle_dir() -> str:
    """
    Returns the root directory where read-only application resources reside.
    - Development: directory containing the repository root.
    - Frozen .app: sys._MEIPASS or Contents/MacOS (inside bundle).
    """
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass and os.path.isdir(meipass):
            return meipass
        return os.path.dirname(sys.executable)
    # Repository root (parent of core/)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_resource_path(relative_path: str) -> str:
    """
    Resolves an absolute path to a bundled read-only asset or model.
    Checks bundle directory first, then fallback paths.
    """
    bundle_dir = get_bundle_dir()
    primary = os.path.join(bundle_dir, relative_path)
    if os.path.exists(primary):
        return primary

    # Fallback: check executable directory or cwd
    if is_frozen():
        exec_dir = os.path.dirname(sys.executable)
        secondary = os.path.join(exec_dir, relative_path)
        if os.path.exists(secondary):
            return secondary

    return primary


def get_user_data_dir() -> str:
    """
    Returns the persistent directory for user profiles, logs, and settings.
    macOS: ~/Library/Application Support/IntentSense
    Linux/Other: ~/.intentsense
    """
    if sys.platform == "darwin":
        path = os.path.expanduser("~/Library/Application Support/IntentSense")
    else:
        path = os.path.expanduser("~/.intentsense")
    os.makedirs(path, exist_ok=True)
    return path


def get_custom_gestures_path() -> str:
    """
    Returns the path to the user's custom gestures JSON file.
    Always points to writable user data directory, never the read-only bundle.
    If an existing models/custom_gestures.json exists in dev and user data
    does not have one yet, it will seamlessly seed it.
    """
    user_path = os.path.join(get_user_data_dir(), "custom_gestures.json")
    if not os.path.exists(user_path):
        legacy_path = get_resource_path(os.path.join("models", "custom_gestures.json"))
        if os.path.isfile(legacy_path):
            try:
                import shutil
                shutil.copy2(legacy_path, user_path)
            except Exception:
                pass
    return user_path
