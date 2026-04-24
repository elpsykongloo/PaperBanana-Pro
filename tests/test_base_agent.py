import asyncio
import types
import unittest
from pathlib import Path

from agents.base_agent import BaseAgent
from utils import generation_utils
from utils.config import ExpConfig


class _DummyAgent(BaseAgent):
    async def process(self, data, **kwargs):
        return data


class BaseAgentImageApiTest(unittest.TestCase):
    def test_gemini_image_api_uses_prompt_hints_and_preserves_image_input(self):
        captured = {}

        async def fake_call_gemini_with_retry_async(**kwargs):
            captured.update(kwargs)
            return ["fake-image-b64"]

        original = generation_utils.call_gemini_with_retry_async
        generation_utils.call_gemini_with_retry_async = fake_call_gemini_with_retry_async
        try:
            exp_config = ExpConfig(
                dataset_name="PaperBananaBench",
                task_name="diagram",
                provider="gemini",
                work_dir=Path("."),
            )
            agent = _DummyAgent(
                model_name="gemini-3.1-flash-image-preview",
                system_prompt="System prompt",
                exp_config=exp_config,
            )

            result = asyncio.run(
                agent.call_image_api(
                    prompt="Draw a diagram.",
                    contents=[
                        {"type": "text", "text": "old text"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "data": "abc",
                                "media_type": "image/png",
                            },
                        },
                    ],
                    aspect_ratio="16:9",
                    image_resolution="4K",
                )
            )

            self.assertEqual(result, ["fake-image-b64"])
            self.assertEqual(captured["contents"][0]["type"], "text")
            self.assertIn("Aspect ratio: 16:9", captured["contents"][0]["text"])
            self.assertIn("Output resolution preference: 4K", captured["contents"][0]["text"])
            self.assertEqual(captured["contents"][1]["type"], "image")
            self.assertEqual(
                getattr(captured["config"], "response_modalities", None),
                ["IMAGE"],
            )
        finally:
            generation_utils.call_gemini_with_retry_async = original

    def test_image_provider_route_does_not_switch_based_on_custom_model_name(self):
        captured = {"gemini": 0, "openai": 0}

        async def fake_call_gemini_with_retry_async(**kwargs):
            captured["gemini"] += 1
            return ["fake-image-b64"]

        async def fake_call_openai_image_generation_with_retry_async(**kwargs):
            captured["openai"] += 1
            return ["fake-image-b64"]

        original_gemini = generation_utils.call_gemini_with_retry_async
        original_openai = generation_utils.call_openai_image_generation_with_retry_async
        generation_utils.call_gemini_with_retry_async = fake_call_gemini_with_retry_async
        generation_utils.call_openai_image_generation_with_retry_async = (
            fake_call_openai_image_generation_with_retry_async
        )
        try:
            exp_config = ExpConfig(
                dataset_name="PaperBananaBench",
                task_name="diagram",
                provider="gemini",
                work_dir=Path("."),
            )
            agent = _DummyAgent(
                model_name="gpt-image-1",
                system_prompt="System prompt",
                exp_config=exp_config,
            )

            result = asyncio.run(
                agent.call_image_api(
                    prompt="Draw a diagram.",
                    contents=[{"type": "text", "text": "old text"}],
                )
            )

            self.assertEqual(result, ["fake-image-b64"])
            self.assertEqual(captured["gemini"], 1)
            self.assertEqual(captured["openai"], 0)
        finally:
            generation_utils.call_gemini_with_retry_async = original_gemini
            generation_utils.call_openai_image_generation_with_retry_async = original_openai


class BaseAgentProviderValidationTest(unittest.TestCase):
    def test_text_api_routes_openai_compatible_to_openai_chat(self):
        captured = {}

        async def fake_call_openai_with_retry_async(**kwargs):
            captured.update(kwargs)
            return ["ok"]

        original = generation_utils.call_openai_with_retry_async
        generation_utils.call_openai_with_retry_async = fake_call_openai_with_retry_async
        try:
            exp_config = types.SimpleNamespace(
                provider="openai_compatible",
                temperature=0.5,
            )
            agent = _DummyAgent(
                model_name="custom-text-model",
                system_prompt="System prompt",
                exp_config=exp_config,
            )

            result = asyncio.run(agent.call_text_api([{"type": "text", "text": "hello"}]))

            self.assertEqual(result, ["ok"])
            self.assertEqual(captured["model_name"], "custom-text-model")
        finally:
            generation_utils.call_openai_with_retry_async = original

    def test_text_api_routes_official_openai_to_openai_chat(self):
        captured = {}

        async def fake_call_openai_with_retry_async(**kwargs):
            captured.update(kwargs)
            return ["ok"]

        original = generation_utils.call_openai_with_retry_async
        generation_utils.call_openai_with_retry_async = fake_call_openai_with_retry_async
        try:
            exp_config = types.SimpleNamespace(
                provider="openai",
                temperature=0.5,
            )
            agent = _DummyAgent(
                model_name="gpt-5.5",
                system_prompt="System prompt",
                exp_config=exp_config,
            )

            result = asyncio.run(agent.call_text_api([{"type": "text", "text": "hello"}]))

            self.assertEqual(result, ["ok"])
            self.assertEqual(captured["model_name"], "gpt-5.5")
            self.assertEqual(captured["config"]["max_completion_tokens"], 50000)
        finally:
            generation_utils.call_openai_with_retry_async = original

    def test_image_api_routes_openrouter_to_chat_image_generation(self):
        captured = {}

        async def fake_call_openrouter_image_generation_with_retry_async(**kwargs):
            captured.update(kwargs)
            return ["fake-image-b64"]

        original = generation_utils.call_openrouter_image_generation_with_retry_async
        generation_utils.call_openrouter_image_generation_with_retry_async = fake_call_openrouter_image_generation_with_retry_async
        try:
            exp_config = types.SimpleNamespace(
                provider="openrouter",
                temperature=0.5,
            )
            agent = _DummyAgent(
                model_name="sourceful/riverflow-v2-pro",
                system_prompt="System prompt",
                exp_config=exp_config,
            )

            result = asyncio.run(
                agent.call_image_api(
                    prompt="Draw a diagram.",
                    aspect_ratio="16:9",
                    image_resolution="4K",
                )
            )

            self.assertEqual(result, ["fake-image-b64"])
            self.assertEqual(captured["model_name"], "sourceful/riverflow-v2-pro")
            self.assertEqual(captured["config"]["aspect_ratio"], "16:9")
            self.assertEqual(captured["config"]["image_size"], "4K")
            self.assertEqual(captured["contents"][0]["text"], "Draw a diagram.")
        finally:
            generation_utils.call_openrouter_image_generation_with_retry_async = original

    def test_image_api_routes_openai_compatible_to_openai_images(self):
        captured = {}

        async def fake_call_openai_image_generation_with_retry_async(**kwargs):
            captured.update(kwargs)
            return ["fake-image-b64"]

        original = generation_utils.call_openai_image_generation_with_retry_async
        generation_utils.call_openai_image_generation_with_retry_async = fake_call_openai_image_generation_with_retry_async
        try:
            exp_config = types.SimpleNamespace(
                provider="openai_compatible",
                temperature=0.5,
            )
            agent = _DummyAgent(
                model_name="custom-image-model",
                system_prompt="System prompt",
                exp_config=exp_config,
            )

            result = asyncio.run(agent.call_image_api(prompt="Draw a diagram."))

            self.assertEqual(result, ["fake-image-b64"])
            self.assertEqual(captured["model_name"], "custom-image-model")
        finally:
            generation_utils.call_openai_image_generation_with_retry_async = original

    def test_text_api_rejects_unknown_provider(self):
        exp_config = types.SimpleNamespace(
            provider="mystery-provider",
            temperature=0.5,
        )
        agent = _DummyAgent(
            model_name="gemini-3.1-flash-lite-preview",
            system_prompt="System prompt",
            exp_config=exp_config,
        )

        with self.assertRaisesRegex(ValueError, "Unsupported provider for text generation"):
            asyncio.run(agent.call_text_api([{"type": "text", "text": "hello"}]))


if __name__ == "__main__":
    unittest.main()
