import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from utils import generation_utils
from utils.image_generation_options import (
    get_image_model_capabilities,
    is_valid_custom_image_size,
    normalize_image_generation_options,
)


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
    def test_gpt_image_2_4k_square_maps_to_custom_2880(self):
        options = normalize_image_generation_options(
            provider_type="openai",
            model_name="gpt-image-2",
            aspect_ratio="1:1",
            image_resolution="4K",
        )

        self.assertEqual(options.size, "2880x2880")

    def test_gpt_image_2_accepts_custom_size_for_ui(self):
        capabilities = get_image_model_capabilities("openai", "gpt-image-2")

        self.assertTrue(is_valid_custom_image_size("2304x1024", capabilities))
        self.assertFalse(is_valid_custom_image_size("2305x1024", capabilities))

    async def test_openai_compatible_unknown_model_uses_conservative_params(self):
        fake_response = SimpleNamespace(data=[SimpleNamespace(b64_json="fake-image-b64")])
        fake_client = SimpleNamespace(
            images=SimpleNamespace(
                generate=AsyncMock(return_value=fake_response),
            )
        )

        with patch.object(generation_utils, "get_openai_client", return_value=fake_client):
            result = await generation_utils.call_openai_image_generation_with_retry_async(
                model_name="gateway-image-model",
                prompt="draw a circle",
                config={
                    "size": "2304x1024",
                    "quality": "high",
                    "background": "opaque",
                    "output_format": "jpeg",
                    "moderation": "low",
                    "stream": True,
                    "partial_images": 2,
                },
                provider_type="openai_compatible",
                max_attempts=1,
                retry_delay=0,
            )

        self.assertEqual(result, ["fake-image-b64"])
        sent = fake_client.images.generate.call_args.kwargs
        self.assertEqual(sent["size"], "auto")
        self.assertEqual(sent["quality"], "high")
        self.assertNotIn("background", sent)
        self.assertNotIn("output_format", sent)
        self.assertNotIn("moderation", sent)
        self.assertNotIn("stream", sent)

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

    async def test_gpt_image_2_generate_sanitizes_transparent_and_png_compression(self):
        fake_response = SimpleNamespace(data=[SimpleNamespace(b64_json="fake-image-b64")])
        fake_client = SimpleNamespace(
            images=SimpleNamespace(
                generate=AsyncMock(return_value=fake_response),
            )
        )

        with patch.object(generation_utils, "get_openai_client", return_value=fake_client):
            result = await generation_utils.call_openai_image_generation_with_retry_async(
                model_name="gpt-image-2",
                prompt="draw a circle",
                config={
                    "aspect_ratio": "16:9",
                    "image_resolution": "4K",
                    "background": "transparent",
                    "output_format": "png",
                    "output_compression": 50,
                    "input_fidelity": "high",
                },
                max_attempts=1,
                retry_delay=0,
            )

        self.assertEqual(result, ["fake-image-b64"])
        sent = fake_client.images.generate.call_args.kwargs
        self.assertEqual(sent["model"], "gpt-image-2")
        self.assertEqual(sent["size"], "3840x2160")
        self.assertEqual(sent["background"], "auto")
        self.assertEqual(sent["output_format"], "png")
        self.assertNotIn("output_compression", sent)
        self.assertNotIn("input_fidelity", sent)

    async def test_gpt_image_2_edit_uses_images_edit_for_reference_image(self):
        fake_response = SimpleNamespace(data=[SimpleNamespace(b64_json="edited-image-b64")])
        fake_client = SimpleNamespace(
            images=SimpleNamespace(
                generate=AsyncMock(side_effect=AssertionError("不应走 generate")),
                edit=AsyncMock(return_value=fake_response),
            )
        )

        with patch.object(generation_utils, "get_openai_client", return_value=fake_client):
            result = await generation_utils.call_openai_image_generation_with_retry_async(
                model_name="gpt-image-2",
                prompt="polish this image",
                config={"size": "auto", "output_format": "jpeg", "output_compression": 80},
                contents=[
                    {"type": "text", "text": "polish this image"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "ZmFrZS1pbWFnZQ==",
                        },
                    },
                ],
                max_attempts=1,
                retry_delay=0,
            )

        self.assertEqual(result, ["edited-image-b64"])
        sent = fake_client.images.edit.call_args.kwargs
        self.assertEqual(sent["model"], "gpt-image-2")
        self.assertEqual(sent["output_compression"], 80)
        self.assertEqual(len(sent["image"]), 1)
        self.assertNotIn("input_fidelity", sent)

    async def test_openai_image_generation_does_not_retry_non_recoverable_errors(self):
        class _BadRequest(RuntimeError):
            status_code = 400

        fake_client = SimpleNamespace(
            images=SimpleNamespace(
                generate=AsyncMock(side_effect=_BadRequest("unsupported parameter")),
            )
        )

        with patch.object(generation_utils, "get_openai_client", return_value=fake_client):
            with patch("utils.generation_utils.asyncio.sleep", new=AsyncMock()) as mocked_sleep:
                with self.assertRaisesRegex(RuntimeError, "unsupported parameter"):
                    await generation_utils.call_openai_image_generation_with_retry_async(
                        model_name="gpt-image-2",
                        prompt="draw a circle",
                        config={"size": "1024x1024", "output_format": "png"},
                        max_attempts=5,
                        retry_delay=0,
                    )

        self.assertEqual(fake_client.images.generate.call_count, 1)
        mocked_sleep.assert_not_awaited()

    async def test_openai_stream_response_is_consumed(self):
        class _FakeStream:
            def __aiter__(self):
                self._events = iter(
                    [
                        SimpleNamespace(type="image_generation.partial_image", b64_json="partial-b64"),
                        SimpleNamespace(type="image_generation.completed", b64_json="final-b64"),
                    ]
                )
                return self

            async def __anext__(self):
                try:
                    return next(self._events)
                except StopIteration:
                    raise StopAsyncIteration

        fake_client = SimpleNamespace(
            images=SimpleNamespace(
                generate=AsyncMock(return_value=_FakeStream()),
            )
        )

        with patch.object(generation_utils, "get_openai_client", return_value=fake_client):
            result = await generation_utils.call_openai_image_generation_with_retry_async(
                model_name="gpt-image-2",
                prompt="draw a circle",
                config={"size": "1024x1024", "output_format": "png", "stream": True, "partial_images": 1},
                max_attempts=1,
                retry_delay=0,
            )

        self.assertEqual(result, ["final-b64"])
        sent = fake_client.images.generate.call_args.kwargs
        self.assertTrue(sent["stream"])
        self.assertEqual(sent["partial_images"], 1)

    async def test_openai_stream_emits_candidate_preview_events(self):
        class _FakeStream:
            def __aiter__(self):
                self._events = iter(
                    [
                        SimpleNamespace(type="image_generation.partial_image", b64_json="partial-b64"),
                        SimpleNamespace(type="image_generation.completed", b64_json="final-b64"),
                    ]
                )
                return self

            async def __anext__(self):
                try:
                    return next(self._events)
                except StopIteration:
                    raise StopAsyncIteration

        fake_client = SimpleNamespace(
            images=SimpleNamespace(
                generate=AsyncMock(return_value=_FakeStream()),
            )
        )
        captured_events = []
        original_hook = generation_utils.runtime_event_hook
        generation_utils.runtime_event_hook = captured_events.append

        try:
            with patch.object(generation_utils, "get_openai_client", return_value=fake_client):
                result = await generation_utils.call_openai_image_generation_with_retry_async(
                    model_name="gpt-image-2",
                    prompt="draw a circle",
                    config={"size": "1024x1024", "output_format": "png", "stream": True, "partial_images": 1},
                    max_attempts=1,
                    retry_delay=0,
                    error_context="visualizer-image[candidate=7,key=render]",
                )
        finally:
            generation_utils.runtime_event_hook = original_hook

        self.assertEqual(result, ["final-b64"])
        preview_events = [event for event in captured_events if event.get("kind") == "preview_ready"]
        self.assertEqual(len(preview_events), 2)
        self.assertEqual(preview_events[0]["candidate_id"], "7")
        self.assertEqual(preview_events[0]["preview_image"], "partial-b64")
        self.assertEqual(preview_events[1]["preview_image"], "final-b64")

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
