import asyncio
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


if __name__ == "__main__":
    unittest.main()
