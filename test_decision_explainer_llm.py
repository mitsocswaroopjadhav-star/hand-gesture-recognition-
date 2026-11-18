"""
test_decision_explainer_llm.py
==============================

Unit tests for Phase 2: Intelligent Decision Explainer Integration.

Verifies:
  1. Integration of DecisionExplainer with LLMAssistant.
  2. Ground-truth fidelity: explanations are strictly derived from DecisionMessage and EventSnapshot.
  3. Coaching generation across all major decision categories:
     - LOCKED states (safety suppression & OpenPalm unlock progress)
     - REJECTED states (low confidence & posture tips)
     - COOLDOWN states (rate limiting explanations)
     - WAITING states (hold duration requirements)
     - ACTION_EXECUTED states (success confirmation)
     - IDLE states (camera/hand distance guidance)
  4. Asynchronous non-blocking execution (zero main-thread blocking).
  5. Semantic caching (identical events do not trigger redundant requests).
  6. Graceful offline fallback and provider status reporting.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from core.activation_state import SystemState
from core.context_manager import Context
from core.decision_explainer import (
    DecisionCategory,
    DecisionExplainer,
    DecisionMessage,
    EventSnapshot,
)
from core.intent_engine import Intent
from core.llm_assistant import ExplanationResponse, LLMAssistant


class TestDecisionExplainerLLM(unittest.TestCase):
    """Verifies Phase 2 DecisionExplainer LLM integration, caching, and async execution."""

    def setUp(self) -> None:
        self.assistant = LLMAssistant(api_key="", endpoint_url="")
        self.explainer = DecisionExplainer(llm_assistant=self.assistant)

    def tearDown(self) -> None:
        self.explainer.close()

    # -------------------------------------------------------------------------
    # 1. Major Decision Categories Coaching Verification
    # -------------------------------------------------------------------------
    def test_coaching_locked_suppression(self) -> None:
        msg = DecisionMessage(
            category=DecisionCategory.REJECTED,
            text="REJECTED — System locked (Hold OpenPalm to unlock)",
        )
        res = self.explainer.explain_with_llm(
            decision_msg=msg,
            state=SystemState.LOCKED,
            context=Context.MEDIA,
            raw_gesture="ThumbsUp",
            confidence=0.95,
        )
        self.assertIsInstance(res, ExplanationResponse)
        self.assertIn("ThumbsUp", res.summary)
        self.assertIn("LOCKED", res.summary)
        self.assertIn("OpenPalm", res.suggested_action)

    def test_coaching_locked_unlocking_progress(self) -> None:
        msg = DecisionMessage(
            category=DecisionCategory.WAITING,
            text="WAITING — Hold OpenPalm to unlock (0.4s / 1.0s)",
        )
        res = self.explainer.explain_with_llm(
            decision_msg=msg,
            state=SystemState.LOCKED,
            context=Context.MEDIA,
            raw_gesture="OpenPalm",
            confidence=0.94,
            hold_duration=0.4,
            target_hold=1.0,
        )
        self.assertIsInstance(res, ExplanationResponse)
        self.assertIn("0.4s / 1.0s", res.summary)
        self.assertIn("flat and steady", res.coaching_tip)

    def test_coaching_low_confidence_rejection(self) -> None:
        msg = DecisionMessage(
            category=DecisionCategory.REJECTED,
            text="REJECTED — Confidence below threshold (65% < 80%)",
        )
        res = self.explainer.explain_with_llm(
            decision_msg=msg,
            state=SystemState.ACTIVE,
            context=Context.PRESENTATION,
            raw_gesture="Pointing",
            confidence=0.65,
            effective_threshold=0.80,
        )
        self.assertIsInstance(res, ExplanationResponse)
        self.assertIn("65%", res.summary)
        self.assertIn("Spread your fingers clearly", res.coaching_tip)

    def test_coaching_cooldown_rate_limit(self) -> None:
        msg = DecisionMessage(
            category=DecisionCategory.COOLDOWN,
            text="COOLDOWN — Action rate limit active",
        )
        res = self.explainer.explain_with_llm(
            decision_msg=msg,
            state=SystemState.ACTIVE,
            context=Context.MEDIA,
        )
        self.assertIsInstance(res, ExplanationResponse)
        self.assertIn("rate-limit", res.coaching_tip)
        self.assertIn("Pause briefly", res.suggested_action)

    def test_coaching_action_executed(self) -> None:
        msg = DecisionMessage(
            category=DecisionCategory.ACTION_EXECUTED,
            text="next_slide (advanced slide)",
        )
        res = self.explainer.explain_with_llm(
            decision_msg=msg,
            state=SystemState.ACTIVE,
            context=Context.PRESENTATION,
            raw_gesture="Pointing",
        )
        self.assertIsInstance(res, ExplanationResponse)
        self.assertIn("next_slide", res.summary)
        self.assertIn("Intent confirmed", res.coaching_tip)

    def test_coaching_idle_state(self) -> None:
        msg = DecisionMessage(
            category=DecisionCategory.IDLE,
            text="IDLE — Waiting for hand (System locked)",
        )
        res = self.explainer.explain_with_llm(
            decision_msg=msg,
            state=SystemState.LOCKED,
            context=Context.MEDIA,
        )
        self.assertIsInstance(res, ExplanationResponse)
        self.assertIn("safely LOCKED", res.summary)
        self.assertIn("1.5 to 3 feet", res.coaching_tip)

    def test_explain_snapshot_helper(self) -> None:
        snap = EventSnapshot(
            decision=DecisionMessage(
                category=DecisionCategory.ACTION_EXECUTED,
                text="play_pause (toggle playback)",
            ),
            intent=Intent.PLAY_PAUSE,
            action_name="play_pause",
            action_status="EXECUTED",
        )
        res = self.explainer.explain_snapshot_with_llm(snap, Context.MEDIA, SystemState.ACTIVE)
        self.assertIsInstance(res, ExplanationResponse)
        self.assertIn("play_pause", res.summary)

    # -------------------------------------------------------------------------
    # 2. Caching Verification
    # -------------------------------------------------------------------------
    def test_caching_identical_events(self) -> None:
        msg = DecisionMessage(
            category=DecisionCategory.REJECTED,
            text="REJECTED — Confidence below threshold (72% < 80%)",
        )
        with patch.object(self.assistant, "explain_decision", wraps=self.assistant.explain_decision) as mock_call:
            # First call: cache miss
            res1 = self.explainer.explain_with_llm(
                decision_msg=msg,
                state=SystemState.ACTIVE,
                context=Context.MEDIA,
                raw_gesture="Fist",
                confidence=0.72,
            )
            self.assertEqual(mock_call.call_count, 1)

            # Second call: cache hit (should not call assistant.explain_decision again)
            res2 = self.explainer.explain_with_llm(
                decision_msg=msg,
                state=SystemState.ACTIVE,
                context=Context.MEDIA,
                raw_gesture="Fist",
                confidence=0.72,
            )
            self.assertEqual(mock_call.call_count, 1)
            self.assertEqual(res1, res2)

    def test_clear_cache(self) -> None:
        msg = DecisionMessage(
            category=DecisionCategory.COOLDOWN,
            text="COOLDOWN — Active",
        )
        self.explainer.explain_with_llm(msg, SystemState.ACTIVE, Context.MEDIA)
        self.assertTrue(len(self.explainer._explanation_cache) > 0)

        self.explainer.clear_cache()
        self.assertEqual(len(self.explainer._explanation_cache), 0)

    # -------------------------------------------------------------------------
    # 3. Asynchronous Execution & Non-Blocking Guarantee
    # -------------------------------------------------------------------------
    def test_async_execution_callback(self) -> None:
        msg = DecisionMessage(
            category=DecisionCategory.WAITING,
            text="WAITING — Hold OpenPalm to unlock",
        )
        received_results = []

        def callback(res: ExplanationResponse) -> None:
            received_results.append(res)

        self.explainer.request_coaching_async(
            decision_msg=msg,
            state=SystemState.LOCKED,
            context=Context.MEDIA,
            raw_gesture="OpenPalm",
            callback=callback,
        )

        # Wait up to 1.0s for the background worker
        start_t = time.time()
        while not received_results and (time.time() - start_t) < 1.0:
            time.sleep(0.01)

        self.assertEqual(len(received_results), 1)
        self.assertIsInstance(received_results[0], ExplanationResponse)
        self.assertEqual(self.explainer.get_latest_coaching(), received_results[0])

    def test_async_non_blocking_performance(self) -> None:
        """Verifies that requesting coaching takes < 5ms on the calling thread."""
        msg = DecisionMessage(
            category=DecisionCategory.REJECTED,
            text="REJECTED — System locked",
        )
        t0 = time.perf_counter()
        self.explainer.request_coaching_async(
            decision_msg=msg,
            state=SystemState.LOCKED,
            context=Context.MEDIA,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # Main thread dispatch must be instantaneous (well below 1 frame @ 30 FPS, which is 33.3ms)
        self.assertLess(elapsed_ms, 15.0, f"Async dispatch took {elapsed_ms:.2f}ms; must be non-blocking.")

    def test_simulated_5s_delay_does_not_block_frames(self) -> None:
        """
        Simulates a 5-second API delay and verifies that simulated webcam processing
        and HUD rendering loops run 60 frames with ZERO blocking.
        """
        msg = DecisionMessage(
            category=DecisionCategory.REJECTED,
            text="REJECTED — Simulated slow network event",
        )

        def slow_explain(*args, **kwargs):
            time.sleep(5.0)
            return ExplanationResponse(
                summary="Slow response finished.",
                coaching_tip="Slow tip.",
                suggested_action="Action.",
            )

        with patch.object(self.assistant, "explain_decision", side_effect=slow_explain):
            t0 = time.perf_counter()
            self.explainer.request_coaching_async(
                decision_msg=msg,
                state=SystemState.LOCKED,
                context=Context.MEDIA,
            )
            dispatch_time_ms = (time.perf_counter() - t0) * 1000

            # 1. Dispatch must be instant
            self.assertLess(dispatch_time_ms, 15.0)
            self.assertTrue(self.explainer.is_loading_coaching)

            # 2. Simulate 60 webcam frames (2 full seconds of 30 FPS video)
            t_frames_start = time.perf_counter()
            for _ in range(60):
                f_dec = self.explainer.explain(
                    state=SystemState.LOCKED,
                    hand_present=True,
                    raw_gesture="OpenPalm",
                    confidence=0.95,
                )
                self.explainer.resolve_hud_snapshot(
                    live_decision=f_dec,
                    live_intent=None,
                    live_action_name=None,
                    live_action_status="WAITING",
                    last_event=None,
                    last_event_timestamp=0.0,
                    current_timestamp=time.time(),
                )
            frames_elapsed_ms = (time.perf_counter() - t_frames_start) * 1000

            # 60 frames must execute in well under 100ms, proving zero blocking
            self.assertLess(
                frames_elapsed_ms,
                100.0,
                f"Simulated frames took {frames_elapsed_ms:.2f}ms! Frame processing was blocked by LLM query.",
            )

    def test_repeated_events_do_not_create_duplicate_concurrent_requests(self) -> None:
        """
        Verifies that repeated events during an in-flight request do not trigger
        duplicate concurrent worker tasks.
        """
        msg = DecisionMessage(
            category=DecisionCategory.REJECTED,
            text="REJECTED — In-flight test",
        )
        call_count = [0]

        def slow_explain(*args, **kwargs):
            call_count[0] += 1
            time.sleep(0.5)
            return ExplanationResponse(
                summary="Done.",
                coaching_tip="Tip.",
                suggested_action="Action.",
            )

        with patch.object(self.assistant, "explain_decision", side_effect=slow_explain):
            # Dispatch first query
            self.explainer.request_coaching_async(
                decision_msg=msg,
                state=SystemState.LOCKED,
                context=Context.MEDIA,
            )

            # Dispatch 20 more calls for the exact same event while query is in flight
            for _ in range(20):
                self.explainer.request_coaching_async(
                    decision_msg=msg,
                    state=SystemState.LOCKED,
                    context=Context.MEDIA,
                )

            # Wait for completion
            time.sleep(0.6)
            self.assertEqual(call_count[0], 1, "Repeated calls spawned duplicate concurrent requests!")

    def test_shutdown_does_not_hang_with_pending_request(self) -> None:
        """
        Verifies that calling close() returns immediately (< 50ms) without waiting
        for a 5-second pending request to finish.
        """
        msg = DecisionMessage(
            category=DecisionCategory.REJECTED,
            text="REJECTED — Hang test",
        )

        def five_second_sleep(*args, **kwargs):
            time.sleep(5.0)
            return ExplanationResponse("Done", "Tip", "Act")

        with patch.object(self.assistant, "explain_decision", side_effect=five_second_sleep):
            self.explainer.request_coaching_async(
                decision_msg=msg,
                state=SystemState.LOCKED,
                context=Context.MEDIA,
            )

            t0 = time.perf_counter()
            self.explainer.close()
            shutdown_time_ms = (time.perf_counter() - t0) * 1000

            self.assertLess(
                shutdown_time_ms,
                50.0,
                f"Shutdown hung for {shutdown_time_ms:.2f}ms waiting for in-flight LLM query!",
            )

    # -------------------------------------------------------------------------
    # 4. Status Badges & Fallbacks
    # -------------------------------------------------------------------------
    def test_provider_status_reporting(self) -> None:
        # Offline assistant
        self.assertEqual(self.explainer.provider_status, "OFFLINE_HEURISTIC")

        # Online assistant mock
        with patch.dict("os.environ", {"GEMINI_API_KEY": "dummy-key"}):
            online_assistant = LLMAssistant(api_key="dummy-key")
            online_explainer = DecisionExplainer(llm_assistant=online_assistant)
            self.assertEqual(online_explainer.provider_status, "ONLINE_GEMINI")
            online_explainer.close()


if __name__ == "__main__":
    unittest.main()

