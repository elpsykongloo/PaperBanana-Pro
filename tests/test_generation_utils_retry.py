import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from utils import generation_utils


class GeminiRetryPolicyTest(unittest.TestCase):
    def test_text_pro_models_fall_back_to_flash_lite_then_flash(self):
        ladder = generation_utils._build_gemini_model_ladder(
            "gemini-3.1-pro-preview",
            is_image_request=False,
        )

        self.assertEqual(
            ladder,
            [
                "gemini-3.1-pro-preview",
                "gemini-3.1-flash-lite-preview",
                "gemini-3-flash-preview",
            ],
        )

    def test_image_pro_models_fall_back_to_flash_image(self):
        ladder = generation_utils._build_gemini_model_ladder(
            "gemini-3-pro-image-preview",
            is_image_request=True,
        )

        self.assertEqual(
            ladder,
            [
                "gemini-3-pro-image-preview",
                generation_utils.DEFAULT_GEMINI_IMAGE_FALLBACK_MODEL,
            ],
        )

    def test_stage_retry_budget_demotes_text_pro_after_first_cycle(self):
        self.assertEqual(
            generation_utils._stage_retry_budget(
                stage_model_name="gemini-3.1-pro-preview",
                primary_model_name="gemini-3.1-pro-preview",
                is_image_request=False,
                cycle_index=0,
                requested_attempts=5,
            ),
            2,
        )
        self.assertEqual(
            generation_utils._stage_retry_budget(
                stage_model_name="gemini-3.1-pro-preview",
                primary_model_name="gemini-3.1-pro-preview",
                is_image_request=False,
                cycle_index=1,
                requested_attempts=5,
            ),
            0,
        )

    def test_permanent_quota_block_uses_long_cooldown(self):
        cooldown = generation_utils._compute_cycle_cooldown_seconds(
            "429 RESOURCE_EXHAUSTED limit: 0 quota exceeded",
            retry_delay=5,
            cycle_index=0,
        )

        self.assertGreaterEqual(cooldown, 300.0)


class OpenAIRetryFailureTest(unittest.IsolatedAsyncioTestCase):
    async def test_text_retry_exhaustion_raises_instead_of_returning_error_string(self):
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(side_effect=RuntimeError("boom")),
                )
            )
        )

        with patch.object(generation_utils, "get_openai_client", return_value=fake_client):
            with patch("utils.generation_utils.asyncio.sleep", new=AsyncMock()):
                with self.assertRaisesRegex(RuntimeError, "planner\\[test\\].*boom"):
                    await generation_utils.call_openai_with_retry_async(
                        model_name="demo-model",
                        contents=[{"type": "text", "text": "hello"}],
                        config={
                            "system_prompt": "只回复 OK",
                            "temperature": 0,
                            "candidate_num": 1,
                            "max_completion_tokens": 16,
                        },
                        max_attempts=2,
                        retry_delay=0,
                        error_context="planner[test]",
                    )

    async def test_image_retry_exhaustion_raises_instead_of_returning_error_string(self):
        fake_client = SimpleNamespace(
            images=SimpleNamespace(
                generate=AsyncMock(side_effect=RuntimeError("image boom")),
            )
        )

        with patch.object(generation_utils, "get_openai_client", return_value=fake_client):
            with patch("utils.generation_utils.asyncio.sleep", new=AsyncMock()):
                with self.assertRaisesRegex(RuntimeError, "visualizer\\[test\\].*image boom"):
                    await generation_utils.call_openai_image_generation_with_retry_async(
                        model_name="demo-image-model",
                        prompt="draw a circle",
                        config={
                            "size": "1024x1024",
                            "quality": "low",
                            "background": "opaque",
                            "output_format": "png",
                        },
                        max_attempts=2,
                        retry_delay=0,
                        error_context="visualizer[test]",
                    )

    async def test_openrouter_image_helper_extracts_images_from_message_model_extra(self):
        fake_message = SimpleNamespace(
            model_extra={
                "images": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,ZmFrZS1pbWFnZS1iNjQ=",
                        },
                    }
                ]
            }
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(
                        return_value=SimpleNamespace(
                            choices=[SimpleNamespace(message=fake_message)],
                        )
                    ),
                )
            )
        )

        with patch.object(generation_utils, "get_openai_client", return_value=fake_client):
            result = await generation_utils.call_openrouter_image_generation_with_retry_async(
                model_name="sourceful/riverflow-v2-pro",
                prompt="draw a circle",
                config={
                    "aspect_ratio": "1:1",
                    "image_size": "1K",
                    "output_format": "png",
                },
                max_attempts=1,
                retry_delay=0,
                error_context="visualizer[test]",
            )

        self.assertEqual(result, ["ZmFrZS1pbWFnZS1iNjQ="])

    async def test_openrouter_image_retry_exhaustion_raises_with_context(self):
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(side_effect=RuntimeError("openrouter boom")),
                )
            )
        )

        with patch.object(generation_utils, "get_openai_client", return_value=fake_client):
            with patch("utils.generation_utils.asyncio.sleep", new=AsyncMock()):
                with self.assertRaisesRegex(RuntimeError, "visualizer\\[test\\].*openrouter boom"):
                    await generation_utils.call_openrouter_image_generation_with_retry_async(
                        model_name="sourceful/riverflow-v2-pro",
                        prompt="draw a circle",
                        config={
                            "aspect_ratio": "1:1",
                            "image_size": "1K",
                            "output_format": "png",
                        },
                        max_attempts=2,
                        retry_delay=0,
                        error_context="visualizer[test]",
                    )


if __name__ == "__main__":
    unittest.main()
