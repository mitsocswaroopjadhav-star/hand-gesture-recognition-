"""
actions/__init__.py
===================

IntentSense Action Handlers Package.
Contains domain-isolated action handlers for MEDIA and PRESENTATION contexts.
"""

from actions.media_actions import register_media_handlers
from actions.presentation_actions import register_presentation_handlers

__all__ = [
    "register_media_handlers",
    "register_presentation_handlers",
]
