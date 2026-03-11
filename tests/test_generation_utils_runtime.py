import asyncio
import unittest

from utils import generation_utils


class _DummyPart:
    def __init__(self, text: str):
        self.text = text


class _DummyContent:
    def __init__(self, text: str):
        self.parts = [_DummyPart(text)]


class _DummyCandidate:
    def __init__(self, text: str):
        self.content = _DummyContent(text)


class _DummyResponse:
    def __init__(self, text: str):
        self.candidates = [_DummyCandidate(text)]


class _DummyModels:
    def __init__(self, text: str, delay: float = 0.0):
        self.text = text
        self.delay = delay

    async def generate_content(self, **kwargs):
        await asyncio.sleep(self.delay)
        return _DummyResponse(self.text)


class _DummyGeminiClient:
    def __init__(self, text: str, delay: float = 0.0):
        self.aio = type("DummyAio", (), {"models": _DummyModels(text, delay)})()


class _DummyConfig:
    def __init__(self, candidate_count=1):
        self.candidate_count = candidate_count
        self.response_modalities = None


class _DummyEvolinkProvider:
    def __init__(self):
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


class GenerationUtilsRuntimeContextTest(unittest.TestCase):
    def test_concurrent_runtime_contexts_use_isolated_clients_and_hooks(self):
        hook_events_a = []
        hook_events_b = []
        context_a = generation_utils.RuntimeContext(
            provider="gemini",
            api_key="key-a",
            event_hook=hook_events_a.append,
            gemini_client=_DummyGeminiClient("result-a", delay=0.02),
        )
        context_b = generation_utils.RuntimeContext(
            provider="gemini",
            api_key="key-b",
            event_hook=hook_events_b.append,
            gemini_client=_DummyGeminiClient("result-b", delay=0.0),
        )

        async def run_with_context(context, model_name):
            with generation_utils.use_runtime_context(context):
                return await generation_utils.call_gemini_with_retry_async(
                    model_name=model_name,
                    contents=[{"type": "text", "text": "hello"}],
                    config=_DummyConfig(),
                    max_attempts=1,
                    retry_delay=0,
                    error_context=f"context={model_name}",
                )

        async def run_both():
            return await asyncio.gather(
                run_with_context(context_a, "model-a"),
                run_with_context(context_b, "model-b"),
            )

        results = asyncio.run(run_both())

        self.assertEqual(results, [["result-a"], ["result-b"]])
        self.assertTrue(any(event.get("model") == "model-a" for event in hook_events_a))
        self.assertTrue(any(event.get("model") == "model-b" for event in hook_events_b))
        self.assertFalse(any(event.get("model") == "model-b" for event in hook_events_a))
        self.assertFalse(any(event.get("model") == "model-a" for event in hook_events_b))

    def test_close_runtime_context_only_closes_owned_provider(self):
        provider = _DummyEvolinkProvider()
        context = generation_utils.RuntimeContext(
            provider="evolink",
            api_key="key",
            evolink_provider=provider,
            owns_evolink_provider=True,
        )

        asyncio.run(generation_utils.close_runtime_context(context))

        self.assertEqual(provider.close_calls, 1)

    def test_gemini_retry_handles_none_candidate_count(self):
        context = generation_utils.RuntimeContext(
            provider="gemini",
            api_key="key-a",
            gemini_client=_DummyGeminiClient("result-a", delay=0.0),
        )

        async def run_once():
            with generation_utils.use_runtime_context(context):
                return await generation_utils.call_gemini_with_retry_async(
                    model_name="model-a",
                    contents=[{"type": "text", "text": "hello"}],
                    config=_DummyConfig(candidate_count=None),
                    max_attempts=1,
                    retry_delay=0,
                    error_context="candidate-count-none",
                )

        results = asyncio.run(run_once())

        self.assertEqual(results, ["result-a"])

    def test_close_runtime_context_skips_unowned_provider(self):
        provider = _DummyEvolinkProvider()
        context = generation_utils.RuntimeContext(
            provider="evolink",
            api_key="key",
            evolink_provider=provider,
            owns_evolink_provider=False,
        )

        asyncio.run(generation_utils.close_runtime_context(context))

        self.assertEqual(provider.close_calls, 0)


if __name__ == "__main__":
    unittest.main()
