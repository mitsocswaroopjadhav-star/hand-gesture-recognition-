"""
test_context_manager.py
=======================

Unit test suite for IntentSense Phase 5: ContextManager.

Covers:
1. Default context is MEDIA
2. Valid context switching
3. PRESENTATION context operations
4. CUSTOM context operations
5. Invalid context is rejected
6. Invalid context does not corrupt current context
7. Context validation helper methods
8. Case-insensitivity and enum interoperability
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.context_manager import Context, ContextManager


class TestContextManager(unittest.TestCase):
    """Test suite for core/context_manager.py."""

    def setUp(self):
        self.manager = ContextManager()

    # 1. Default context is MEDIA
    def test_default_context_is_media(self):
        """Verify that a fresh ContextManager defaults strictly to MEDIA."""
        self.assertEqual(self.manager.get_current_context(), Context.MEDIA)
        self.assertEqual(self.manager.current_context, Context.MEDIA)
        self.assertEqual(self.manager.get_current_context().value, "MEDIA")
        self.assertEqual(str(self.manager.get_current_context()), "MEDIA")

    # 2. Valid context switching
    def test_valid_context_switching(self):
        """Verify seamless switching across all supported contexts."""
        self.assertEqual(self.manager.get_current_context(), Context.MEDIA)

        # Switch to PRESENTATION
        res_pres = self.manager.set_context(Context.PRESENTATION)
        self.assertEqual(res_pres, Context.PRESENTATION)
        self.assertEqual(self.manager.get_current_context(), Context.PRESENTATION)

        # Switch to CUSTOM
        res_custom = self.manager.set_context(Context.CUSTOM)
        self.assertEqual(res_custom, Context.CUSTOM)
        self.assertEqual(self.manager.get_current_context(), Context.CUSTOM)

        # Switch back to MEDIA
        res_media = self.manager.set_context(Context.MEDIA)
        self.assertEqual(res_media, Context.MEDIA)
        self.assertEqual(self.manager.get_current_context(), Context.MEDIA)

    # 3. PRESENTATION context
    def test_presentation_context(self):
        """Verify PRESENTATION context via string and enum."""
        self.manager.set_context("PRESENTATION")
        self.assertEqual(self.manager.get_current_context(), Context.PRESENTATION)
        self.assertEqual(self.manager.current_context, Context.PRESENTATION)

        # Test case-insensitivity
        self.manager.set_context("presentation")
        self.assertEqual(self.manager.get_current_context(), Context.PRESENTATION)

    # 4. CUSTOM context
    def test_custom_context(self):
        """Verify CUSTOM context via string and enum."""
        self.manager.set_context("CUSTOM")
        self.assertEqual(self.manager.get_current_context(), Context.CUSTOM)
        self.assertEqual(self.manager.current_context, Context.CUSTOM)

        # Test case-insensitivity
        self.manager.set_context("custom")
        self.assertEqual(self.manager.get_current_context(), Context.CUSTOM)

    # 5. Invalid context is rejected
    def test_invalid_context_is_rejected(self):
        """Verify that invalid context strings/types raise ValueError."""
        invalid_candidates = [
            "GAMING",
            "BROWSER",
            "INVALID_MODE",
            "",
            123,
            None,
        ]
        for candidate in invalid_candidates:
            with self.assertRaises(ValueError):
                self.manager.set_context(candidate)

    # 6. Invalid context does not corrupt current context
    def test_invalid_context_does_not_corrupt_current_context(self):
        """
        Verify that an invalid context switch attempt leaves the current
        context completely untouched.
        """
        # Set to PRESENTATION
        self.manager.set_context(Context.PRESENTATION)
        self.assertEqual(self.manager.get_current_context(), Context.PRESENTATION)

        # Attempt to set invalid context
        with self.assertRaises(ValueError):
            self.manager.set_context("GAMING")

        # Current context must remain PRESENTATION
        self.assertEqual(self.manager.get_current_context(), Context.PRESENTATION)

        # Reset to default
        self.manager.reset()
        self.assertEqual(self.manager.get_current_context(), Context.MEDIA)

    # 7. Context validation helper methods
    def test_context_validation_helpers(self):
        """Verify validate_context() and Context.is_valid() return booleans correctly."""
        self.assertTrue(self.manager.validate_context(Context.MEDIA))
        self.assertTrue(self.manager.validate_context("MEDIA"))
        self.assertTrue(self.manager.validate_context("presentation"))
        self.assertTrue(self.manager.validate_context("CUSTOM"))

        self.assertFalse(self.manager.validate_context("UNKNOWN"))
        self.assertFalse(self.manager.validate_context(""))
        self.assertFalse(self.manager.validate_context(None))
        self.assertFalse(self.manager.validate_context(42))

    # 8. Custom initial context on construction
    def test_custom_initial_context(self):
        """Verify ContextManager can be initialized with any valid context."""
        cm = ContextManager(initial_context=Context.PRESENTATION)
        self.assertEqual(cm.get_current_context(), Context.PRESENTATION)

        with self.assertRaises(ValueError):
            ContextManager(initial_context="INVALID_CONTEXT")


if __name__ == "__main__":
    unittest.main(verbosity=2)
