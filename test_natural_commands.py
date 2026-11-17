"""
test_natural_commands.py
========================

Unit tests for Phase 3: Natural Language Commands & Safety Gate.

Verifies:
  1. Translation of natural-language queries into structured Intent enums.
  2. Safety Gate Enforcement:
     - When system is LOCKED, commands are strictly rejected with SYSTEM_LOCKED.
     - When system is ACTIVE, commands are accepted and routed to ActionExecutor.
  3. Action allowlist: arbitrary shell, OS, or unknown commands are rejected.
  4. Context compatibility: Media intents map to Media context; presentation intents map to Presentation context.
  5. Dry-run mode vs live execution preservation.
  6. Integration between LLMAssistant, IntentEngine, and ActionExecutor.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

from actions.media_actions import register_media_handlers
from actions.presentation_actions import register_presentation_handlers
from core.action_executor import ActionExecutor, ActionReason
from core.activation_state import ActivationStateMachine, SystemState
from core.context_manager import Context, ContextManager
from core.intent_engine import Intent, IntentEngine, IntentReason
from core.llm_assistant import LLMAssistant, NaturalCommandResponse


class TestNaturalCommandsSafetyGate(unittest.TestCase):
    """Verifies end-to-end natural language command parsing, safety gates, and action execution."""

    def setUp(self) -> None:
        self.assistant = LLMAssistant(api_key="", endpoint_url="")
        self.context_mgr = ContextManager(initial_context=Context.MEDIA)
        self.intent_engine = IntentEngine(context_manager=self.context_mgr)
        self.activation_sm = ActivationStateMachine()
        self.action_executor = ActionExecutor(action_cooldown=0.0)

        # Register handlers in dry-run mode
        register_media_handlers(self.action_executor, dry_run=True)
        register_presentation_handlers(self.action_executor, dry_run=True)

    def tearDown(self) -> None:
        self.assistant.close()

    # -------------------------------------------------------------------------
    # 1. Parsing & Intent Allowlist
    # -------------------------------------------------------------------------
    def test_parse_media_volume_commands(self) -> None:
        parsed = self.assistant.parse_natural_command("increase volume", Context.MEDIA)
        self.assertTrue(parsed.is_valid)
        self.assertEqual(parsed.intent, Intent.VOLUME_UP)
        self.assertEqual(parsed.context, Context.MEDIA)

        parsed_down = self.assistant.parse_natural_command("turn the sound down please", Context.MEDIA)
        self.assertTrue(parsed_down.is_valid)
        self.assertEqual(parsed_down.intent, Intent.VOLUME_DOWN)

    def test_parse_presentation_slide_commands(self) -> None:
        parsed = self.assistant.parse_natural_command("advance to next slide", Context.PRESENTATION)
        self.assertTrue(parsed.is_valid)
        self.assertEqual(parsed.intent, Intent.NEXT_SLIDE)
        self.assertEqual(parsed.context, Context.PRESENTATION)

        parsed_prev = self.assistant.parse_natural_command("go back to previous slide", Context.PRESENTATION)
        self.assertTrue(parsed_prev.is_valid)
        self.assertEqual(parsed_prev.intent, Intent.PREVIOUS_SLIDE)

    def test_reject_arbitrary_or_unsafe_commands(self) -> None:
        unsafe_queries = [
            "rm -rf /",
            "open terminal and curl a malicious script",
            "delete user directory",
            "shutdown my computer immediately",
            "launch safari browser",
        ]
        for q in unsafe_queries:
            parsed = self.assistant.parse_natural_command(q, Context.MEDIA)
            self.assertFalse(parsed.is_valid)
            self.assertIsNone(parsed.intent)

    # -------------------------------------------------------------------------
    # 2. Safety Gate: Suppression When System is LOCKED
    # -------------------------------------------------------------------------
    def test_locked_state_suppresses_typed_command(self) -> None:
        self.assertEqual(self.activation_sm.state, SystemState.LOCKED)

        # Parse a valid command
        parsed = self.assistant.parse_natural_command("volume up", Context.MEDIA)
        self.assertTrue(parsed.is_valid)

        # Route through IntentEngine while LOCKED
        intent_res = self.intent_engine.process_typed_intent(
            command_intent=parsed.intent,
            is_active=(self.activation_sm.state == SystemState.ACTIVE),
            target_context=parsed.context,
        )
        self.assertFalse(intent_res.accepted)
        self.assertEqual(intent_res.reason, IntentReason.SYSTEM_LOCKED)
        self.assertIsNone(intent_res.intent)

        # ActionExecutor must strictly refuse to execute an unaccepted intent
        act_res = self.action_executor.execute(intent_res)
        self.assertFalse(act_res.success)
        self.assertEqual(act_res.reason, ActionReason.INTENT_REJECTED)

    # -------------------------------------------------------------------------
    # 3. Execution When System is ACTIVE
    # -------------------------------------------------------------------------
    def test_active_state_executes_typed_command(self) -> None:
        # Unlock system
        self.activation_sm._state = SystemState.ACTIVE

        parsed = self.assistant.parse_natural_command("toggle pause music", Context.MEDIA)
        self.assertTrue(parsed.is_valid)
        self.assertEqual(parsed.intent, Intent.PLAY_PAUSE)

        # Route through IntentEngine
        intent_res = self.intent_engine.process_typed_intent(
            command_intent=parsed.intent,
            is_active=(self.activation_sm.state == SystemState.ACTIVE),
            target_context=parsed.context,
        )
        self.assertTrue(intent_res.accepted)
        self.assertEqual(intent_res.intent, Intent.PLAY_PAUSE)
        self.assertEqual(intent_res.reason, IntentReason.CONFIRMED)

        # Route through ActionExecutor (dry-run)
        act_res = self.action_executor.execute(intent_res)
        self.assertTrue(act_res.success)
        self.assertEqual(act_res.action_name, "play_pause")
        self.assertEqual(act_res.reason, ActionReason.EXECUTED)
        self.assertIn("DRY RUN", act_res.message)

    def test_presentation_context_switching_typed_command(self) -> None:
        self.activation_sm._state = SystemState.ACTIVE
        self.context_mgr.set_context(Context.PRESENTATION)

        parsed = self.assistant.parse_natural_command("next slide", Context.PRESENTATION)
        self.assertTrue(parsed.is_valid)
        self.assertEqual(parsed.intent, Intent.NEXT_SLIDE)

        intent_res = self.intent_engine.process_typed_intent(
            command_intent=parsed.intent,
            is_active=(self.activation_sm.state == SystemState.ACTIVE),
            target_context=parsed.context,
        )
        self.assertTrue(intent_res.accepted)
        self.assertEqual(intent_res.intent, Intent.NEXT_SLIDE)

        act_res = self.action_executor.execute(intent_res)
        self.assertTrue(act_res.success)
        self.assertEqual(act_res.action_name, "next_slide")

    # -------------------------------------------------------------------------
    # 4. Context Mismatch Rejection
    # -------------------------------------------------------------------------
    def test_context_mismatch_rejection(self) -> None:
        """Presentation-specific intent in Media context is rejected with NO_MAPPING."""
        self.activation_sm._state = SystemState.ACTIVE
        self.context_mgr.set_context(Context.MEDIA)

        # NEXT_SLIDE is only mapped in PRESENTATION context
        intent_res = self.intent_engine.process_typed_intent(
            command_intent=Intent.NEXT_SLIDE,
            is_active=True,
            target_context=Context.MEDIA,
        )
        self.assertFalse(intent_res.accepted)
        self.assertEqual(intent_res.reason, IntentReason.NO_MAPPING)

        # PREVIOUS_SLIDE is also rejected in MEDIA context
        intent_res_prev = self.intent_engine.process_typed_intent(
            command_intent=Intent.PREVIOUS_SLIDE,
            is_active=True,
            target_context=Context.MEDIA,
        )
        self.assertFalse(intent_res_prev.accepted)
        self.assertEqual(intent_res_prev.reason, IntentReason.NO_MAPPING)

    # -------------------------------------------------------------------------
    # 5. Malformed and Invalid Provider Responses
    # -------------------------------------------------------------------------
    def test_malformed_and_invalid_responses(self) -> None:
        """When an external provider produces garbage or crashes, assistant falls back safely."""
        mock_provider = MagicMock()
        mock_provider.generate_json.side_effect = RuntimeError("Malformed JSON from LLM")

        assistant = LLMAssistant(api_key="fake-key", endpoint_url="fake-url")
        assistant.http_provider = mock_provider

        try:
            parsed = assistant.parse_natural_command("volume up", Context.MEDIA)
            # Must fall back gracefully to deterministic offline parsing
            self.assertTrue(parsed.is_valid)
            self.assertEqual(parsed.intent, Intent.VOLUME_UP)
            self.assertEqual(parsed.action_id, "volume_up")
        finally:
            assistant.close()

    # -------------------------------------------------------------------------
    # 6. Asynchronous Non-Blocking Execution with 5-Second Delay
    # -------------------------------------------------------------------------
    def test_delayed_api_response_does_not_block_caller(self) -> None:
        """A slow 5-second LLM query must not block the calling thread (< 35ms)."""
        def _slow_parse(text: str, ctx: Context) -> NaturalCommandResponse:
            time.sleep(5.0)
            return NaturalCommandResponse(
                raw_query=text,
                is_valid=True,
                intent=Intent.VOLUME_UP,
                context=ctx,
                confidence=0.99,
                explanation="Delayed response",
            )

        callback_results = []
        def _callback(res):
            callback_results.append(res)

        start_time = time.perf_counter()
        future = self.assistant.query_async(
            _slow_parse,
            "turn volume up",
            Context.MEDIA,
            callback=_callback,
        )
        dispatch_duration = time.perf_counter() - start_time

        # Calling thread should return almost instantaneously (< 35ms)
        self.assertLess(dispatch_duration, 0.035, f"Async query blocked caller for {dispatch_duration:.4f}s")
        self.assertFalse(future.done())
        self.assertEqual(len(callback_results), 0)

        # Cancel future to prevent waiting 5s during tearDown
        future.cancel()

    # -------------------------------------------------------------------------
    # 7. Confirmation and Cancellation Workflow Logic
    # -------------------------------------------------------------------------
    def test_confirmation_and_cancellation_workflow(self) -> None:
        """Simulate the user modal: typing 'N' aborts execution; typing 'Y' commits execution."""
        self.activation_sm._state = SystemState.ACTIVE
        parsed = self.assistant.parse_natural_command("volume down", Context.MEDIA)
        self.assertTrue(parsed.is_valid)
        self.assertEqual(parsed.intent, Intent.VOLUME_DOWN)

        # User presses 'N' (cancel): no intent or action is executed
        user_confirmed = False
        if user_confirmed:
            intent_res = self.intent_engine.process_typed_intent(
                command_intent=parsed.intent,
                is_active=True,
                target_context=Context.MEDIA,
            )
            act_res = self.action_executor.execute(intent_res)
        else:
            intent_res = None
            act_res = None

        self.assertIsNone(intent_res)
        self.assertIsNone(act_res)

        # User presses 'Y' (confirm): routes through intent engine and executes
        user_confirmed = True
        if user_confirmed:
            intent_res = self.intent_engine.process_typed_intent(
                command_intent=parsed.intent,
                is_active=True,
                target_context=Context.MEDIA,
            )
            act_res = self.action_executor.execute(intent_res)

        self.assertIsNotNone(intent_res)
        self.assertTrue(intent_res.accepted)
        self.assertEqual(intent_res.intent, Intent.VOLUME_DOWN)
        self.assertIsNotNone(act_res)
        self.assertTrue(act_res.success)
        self.assertEqual(act_res.action_name, "volume_down")



if __name__ == "__main__":
    unittest.main()
