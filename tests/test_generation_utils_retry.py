import asyncio
import json
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
    def test_text_pro_models_fall_back_to_flash_lite_models(self):
        ladder = generation_utils._build_gemini_model_ladder(
            "gemini-3.1-pro-preview",
            is_image_request=False,
        )

        self.assertEqual(
            ladder,
            ["gemini-3.1-pro-preview", "gemini-3.1-flash-lite"],
        )

    def test_flash_lite_falls_back_within_flash_lite_tier(self):
        ladder = generation_utils._build_gemini_model_ladder("gemini-3.5-flash-lite", is_image_request=False)

        self.assertEqual(ladder, ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"])
        self.assertNotIn("gemini-3-flash-preview", ladder)

    def test_default_flash_model_does_not_silently_downgrade(self):
        ladder = generation_utils._build_gemini_model_ladder("gemini-3.8-flash", is_image_request=False)

        self.assertEqual(ladder, ["gemini-3.8-flash"])

    def test_text_timeout_leaves_room_for_thinking_models(self):
        with patch.dict("os.environ", {"GEMINI_TEXT_TIMEOUT_SEC": ""}):
            self.assertGreaterEqual(generation_utils._get_gemini_request_timeout_seconds(False), 120.0)
        with patch.dict("os.environ", {"GEMINI_TEXT_TIMEOUT_SEC": "60"}):
            self.assertEqual(generation_utils._get_gemini_request_timeout_seconds(False), 60.0)

    def test_nano_banana_is_handled_as_flash_image_model(self):
        config = SimpleNamespace(candidate_count=1, response_modalities=None)

        self.assertTrue(generation_utils._is_gemini_image_request("gemini-nano-banana-2.1", config))
        self.assertEqual(
            generation_utils._build_gemini_model_ladder("gemini-nano-banana-2.1", is_image_request=True),
            ["gemini-nano-banana-2.1"],
        )
        self.assertEqual(
            generation_utils._stage_retry_budget(
                stage_model_name="gemini-nano-banana-2.1",
                primary_model_name="gemini-nano-banana-2.1",
                is_image_request=True,
                cycle_index=0,
                requested_attempts=5,
            ),
            4,
        )

    def test_model_not_found_error_is_detected(self):
        error_text = (
            "404 NOT_FOUND. {'error': {'code': 404, 'message': 'models/gemini-x is not found for API version v1beta, "
            "or is not supported for generateContent.', 'status': 'NOT_FOUND'}}"
        )
        self.assertTrue(generation_utils._is_gemini_model_not_found_error(error_text))
        self.assertFalse(generation_utils._is_gemini_model_not_found_error("503 UNAVAILABLE high demand"))

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


class _NotFoundGeminiModels:
    """按模型名返回结果：dead_models 中的模型抛出 404。"""

    def __init__(self, dead_models: set[str]):
        self.dead_models = dead_models
        self.calls: list[str] = []

    async def generate_content(self, *, model, contents, config):
        self.calls.append(model)
        if model in self.dead_models:
            raise RuntimeError(
                f"404 NOT_FOUND. {{'error': {{'code': 404, 'message': 'models/{model} is not found for API version v1beta', "
                "'status': 'NOT_FOUND'}}"
            )
        part = SimpleNamespace(text=f"ok from {model}")
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])


class GeminiModelNotFoundTest(unittest.IsolatedAsyncioTestCase):
    async def _call(self, models: _NotFoundGeminiModels, model_name: str) -> list[str]:
        client = SimpleNamespace(aio=SimpleNamespace(models=models))
        context = generation_utils.RuntimeContext(provider="gemini", api_key="key", gemini_client=client)
        with generation_utils.use_runtime_context(context):
            return await asyncio.wait_for(
                generation_utils.call_gemini_with_retry_async(
                    model_name=model_name,
                    contents=[{"type": "text", "text": "hello"}],
                    config=SimpleNamespace(candidate_count=1, response_modalities=None),
                    max_attempts=5,
                    retry_delay=0,
                    error_context="model-not-found",
                ),
                timeout=10,
            )

    async def test_retired_model_switches_to_next_fallback_without_retrying(self):
        ladder = generation_utils._build_gemini_model_ladder("gemini-3.1-flash-lite-preview", is_image_request=False)
        models = _NotFoundGeminiModels({ladder[0]})

        result = await self._call(models, ladder[0])

        self.assertEqual(result, [f"ok from {ladder[1]}"])
        self.assertEqual(models.calls, ladder[:2])

    async def test_all_models_missing_returns_error_instead_of_retrying_forever(self):
        ladder = generation_utils._build_gemini_model_ladder("gemini-3.1-flash-lite-preview", is_image_request=False)
        models = _NotFoundGeminiModels(set(ladder))

        result = await self._call(models, ladder[0])

        self.assertEqual(result, ["Error"])
        self.assertEqual(models.calls, ladder)


class _FakeAsyncStream:
    def __init__(self, events):
        self._events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration


def _responses_stream_events(*, with_partial: bool):
    """按 openai SDK 的真实事件结构构造 Responses 图像工具流：最终图不在 completed 事件上。"""
    from openai.types.responses import (
        ResponseImageGenCallCompletedEvent,
        ResponseImageGenCallPartialImageEvent,
        ResponseOutputItemDoneEvent,
    )

    events = []
    if with_partial:
        events.append(
            ResponseImageGenCallPartialImageEvent(
                type="response.image_generation_call.partial_image",
                item_id="ig_1",
                output_index=0,
                sequence_number=1,
                partial_image_index=0,
                partial_image_b64="responses-partial-b64",
            )
        )
    events.append(
        ResponseImageGenCallCompletedEvent(
            type="response.image_generation_call.completed",
            item_id="ig_1",
            output_index=0,
            sequence_number=2,
        )
    )
    events.append(
        ResponseOutputItemDoneEvent.model_validate(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "sequence_number": 3,
                "item": {
                    "type": "image_generation_call",
                    "id": "ig_1",
                    "status": "completed",
                    "result": "responses-final-b64",
                },
            }
        )
    )
    events.append(
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                output=[SimpleNamespace(type="image_generation_call", result="responses-final-b64")]
            ),
        )
    )
    return events


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

    async def test_openai_text_accepts_plain_string_compatible_response(self):
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(return_value="OK"),
                )
            )
        )

        with patch.object(generation_utils, "get_openai_client", return_value=fake_client):
            result = await generation_utils.call_openai_with_retry_async(
                model_name="relay-text-model",
                contents=[{"type": "text", "text": "hello"}],
                config={
                    "system_prompt": "只回复 OK",
                    "temperature": 0,
                    "candidate_num": 1,
                    "max_completion_tokens": 16,
                },
                max_attempts=1,
                retry_delay=0,
            )

        self.assertEqual(result, ["OK"])

    async def test_openai_text_rejects_gateway_homepage_html(self):
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(
                        return_value='<!doctype html><html><head><title>Gateway</title></head></html>'
                    ),
                )
            )
        )

        with patch.object(generation_utils, "get_openai_client", return_value=fake_client):
            with self.assertRaisesRegex(RuntimeError, "HTML 页面"):
                await generation_utils.call_openai_with_retry_async(
                    model_name="relay-text-model",
                    contents=[{"type": "text", "text": "hello"}],
                    config={
                        "system_prompt": "只回复 OK",
                        "temperature": 0,
                        "candidate_num": 1,
                        "max_completion_tokens": 16,
                    },
                    max_attempts=1,
                    retry_delay=0,
                    error_context="planner[test]",
                )

    async def test_openai_text_uses_raw_fallback_when_sdk_shape_fails(self):
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(
                        side_effect=AttributeError("'str' object has no attribute 'choices'")
                    ),
                )
            )
        )
        captured = {}

        async def fake_raw_fallback(**kwargs):
            captured.update(kwargs)
            return "OK"

        context = generation_utils.RuntimeContext(
            provider="openai_compatible",
            api_key="secret",
            base_url="https://relay.example/v1",
            openai_client=fake_client,
        )

        with generation_utils.use_runtime_context(context):
            with patch.object(
                generation_utils,
                "_call_openai_chat_completions_raw_async",
                side_effect=fake_raw_fallback,
            ):
                result = await generation_utils.call_openai_with_retry_async(
                    model_name="relay-text-model",
                    contents=[{"type": "text", "text": "hello"}],
                    config={
                        "system_prompt": "只回复 OK",
                        "temperature": 0,
                        "candidate_num": 1,
                        "max_completion_tokens": 16,
                    },
                    max_attempts=1,
                    retry_delay=0,
                )

        self.assertEqual(result, ["OK"])
        self.assertIs(captured["client"], fake_client)
        self.assertEqual(captured["model_name"], "relay-text-model")

    async def test_openai_text_rejects_html_from_raw_fallback(self):
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(
                        side_effect=AttributeError("'str' object has no attribute 'choices'")
                    ),
                )
            )
        )

        async def fake_raw_fallback(**kwargs):
            return '<!doctype html><html><head><title>Gateway</title></head></html>'

        context = generation_utils.RuntimeContext(
            provider="openai_compatible",
            api_key="secret",
            base_url="https://relay.example",
            openai_client=fake_client,
        )

        with generation_utils.use_runtime_context(context):
            with patch.object(
                generation_utils,
                "_call_openai_chat_completions_raw_async",
                side_effect=fake_raw_fallback,
            ):
                with self.assertRaisesRegex(RuntimeError, "HTML 页面"):
                    await generation_utils.call_openai_with_retry_async(
                        model_name="relay-text-model",
                        contents=[{"type": "text", "text": "hello"}],
                        config={
                            "system_prompt": "只回复 OK",
                            "temperature": 0,
                            "candidate_num": 1,
                            "max_completion_tokens": 16,
                        },
                        max_attempts=1,
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

    async def _call_image_api_with_real_sdk(self, *, model_name="gpt-image-2", config, contents=None):
        # 用真实 SDK + MockTransport：请求参数必须能通过已安装 SDK 的方法签名（AsyncMock 测不出这一点）。
        import httpx
        from openai import AsyncOpenAI

        requests = []

        async def handle(request):
            requests.append(request)
            return httpx.Response(200, json={"created": 0, "data": [{"b64_json": "image-b64"}]})

        async with AsyncOpenAI(
            api_key="test-only", base_url="https://example.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        ) as client:
            with patch.object(generation_utils, "get_openai_client", return_value=client):
                result = await generation_utils.call_openai_image_generation_with_retry_async(
                    model_name=model_name, prompt="polish this image",
                    config={"responses_fallback": "never", **config},
                    contents=contents, max_attempts=1, retry_delay=0,
                )
        return result, requests

    _REFERENCE_IMAGE = [{"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": "ZmFrZS1pbWFnZQ==",
    }}]

    async def test_image_edit_moderation_works_with_installed_sdk(self):
        result, requests = await self._call_image_api_with_real_sdk(
            config={"moderation": "low"}, contents=self._REFERENCE_IMAGE,
        )
        self.assertEqual(result, ["image-b64"])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/v1/images/edits")
        self.assertIn(b'name="moderation"\r\n\r\nlow', requests[0].content)

    async def test_image_edit_omits_default_moderation(self):
        result, requests = await self._call_image_api_with_real_sdk(
            config={"moderation": "auto"}, contents=self._REFERENCE_IMAGE,
        )
        self.assertEqual(result, ["image-b64"])
        self.assertEqual(requests[0].url.path, "/v1/images/edits")
        self.assertNotIn(b'name="moderation"', requests[0].content)

    async def test_image_generate_keeps_moderation_with_installed_sdk(self):
        result, requests = await self._call_image_api_with_real_sdk(config={"moderation": "low"})
        self.assertEqual(result, ["image-b64"])
        self.assertEqual(requests[0].url.path, "/v1/images/generations")
        self.assertEqual(json.loads(requests[0].content)["moderation"], "low")

    async def test_gpt_image_1_mini_edit_never_sends_input_fidelity(self):
        result, requests = await self._call_image_api_with_real_sdk(
            model_name="gpt-image-1-mini",
            config={"input_fidelity": "high"},
            contents=self._REFERENCE_IMAGE,
        )
        self.assertEqual(result, ["image-b64"])
        self.assertNotIn(b'name="input_fidelity"', requests[0].content)
        self.assertFalse(get_image_model_capabilities("openai", "gpt-image-1-mini").supports_input_fidelity)
        self.assertTrue(get_image_model_capabilities("openai", "gpt-image-1.5").supports_input_fidelity)

    def test_transparent_background_forces_png_instead_of_jpeg(self):
        options = normalize_image_generation_options(
            provider_type="openai",
            model_name="gpt-image-1.5",
            raw_options={"background": "transparent", "output_format": "jpeg", "output_compression": 80},
        )
        self.assertEqual(options.background, "transparent")
        self.assertEqual(options.output_format, "png")
        self.assertIsNone(options.output_compression)

    def test_fixed_size_models_keep_aspect_ratio_instead_of_auto(self):
        cases = {
            ("16:9", "2K"): "1536x1024",
            ("9:16", "2K"): "1024x1536",
            ("1:1", "2K"): "1024x1024",
            ("4:3", "1K"): "1536x1024",
            ("21:9", "1K"): "1536x1024",
        }
        for model_name in ("gpt-image-1.5", "gpt-image-1-mini", "chatgpt-image-latest"):
            for (aspect_ratio, resolution), expected in cases.items():
                options = normalize_image_generation_options(
                    provider_type="openai",
                    model_name=model_name,
                    aspect_ratio=aspect_ratio,
                    image_resolution=resolution,
                )
                self.assertEqual(options.size, expected, (model_name, aspect_ratio, resolution))
        # 显式 auto 仍然保留；未知的 openai_compatible 模型不猜固定尺寸
        self.assertEqual(
            normalize_image_generation_options(provider_type="openai", model_name="gpt-image-1.5", image_resolution="auto").size,
            "auto",
        )
        self.assertEqual(
            normalize_image_generation_options(
                provider_type="openai_compatible", model_name="some-proxy-model", aspect_ratio="16:9", image_resolution="2K"
            ).size,
            "auto",
        )

    def test_gpt_image_2_uses_exact_aspect_ratio_sizes(self):
        cases = {
            ("16:9", "1K"): "1536x864",
            ("9:16", "1K"): "864x1536",
            ("21:9", "1K"): "1680x720",
            ("21:9", "2K"): "2352x1008",
            ("21:9", "4K"): "3360x1440",
            ("16:9", "2K"): "2048x1152",
        }
        capabilities = get_image_model_capabilities("openai", "gpt-image-2")
        for (aspect_ratio, resolution), expected in cases.items():
            options = normalize_image_generation_options(
                provider_type="openai", model_name="gpt-image-2", aspect_ratio=aspect_ratio, image_resolution=resolution
            )
            self.assertEqual(options.size, expected, (aspect_ratio, resolution))
            self.assertTrue(is_valid_custom_image_size(expected, capabilities), expected)
        # gpt-image-1 系列仍使用固定尺寸表
        self.assertEqual(
            normalize_image_generation_options(
                provider_type="openai", model_name="gpt-image-1.5", aspect_ratio="16:9", image_resolution="1K"
            ).size,
            "1536x1024",
        )

    def test_user_errors_are_not_retried_or_sent_to_responses_fallback(self):
        from openai import APIError

        request = SimpleNamespace(method="POST", url="https://example.invalid/v1/images/edits")
        # 流式请求中途的 SSE error 事件：SDK 抛出的 APIError 没有 status_code
        stream_error = APIError(
            "Your request was rejected by the safety system.",
            request=request,
            body={"type": "image_generation_user_error", "code": "moderation_blocked"},
        )
        self.assertFalse(generation_utils._is_openai_image_retryable_error(stream_error))

        class _ModerationBlocked(RuntimeError):
            status_code = 400
            code = "moderation_blocked"
            type = "image_generation_user_error"

        self.assertFalse(
            generation_utils._should_try_openai_responses_fallback(
                _ModerationBlocked("blocked"), provider_type="openai", model_name="gpt-image-2", fallback_mode="auto"
            )
        )

        class _UnsupportedParameter(RuntimeError):
            status_code = 400
            code = "unknown_parameter"

        # 非用户错误的 400（如参数不被 Images API 支持）仍保留兜底
        self.assertTrue(
            generation_utils._should_try_openai_responses_fallback(
                _UnsupportedParameter("unsupported parameter"),
                provider_type="openai",
                model_name="gpt-image-2",
                fallback_mode="auto",
            )
        )

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

    async def test_openai_image_response_url_is_downloaded_to_base64(self):
        fake_response = SimpleNamespace(data=[SimpleNamespace(url="https://example.test/image.png")])
        fake_client = SimpleNamespace(
            images=SimpleNamespace(
                generate=AsyncMock(return_value=fake_response),
            )
        )

        with patch.object(generation_utils, "get_openai_client", return_value=fake_client):
            with patch.object(
                generation_utils,
                "_fetch_image_url_as_base64",
                new=AsyncMock(return_value="downloaded-image-b64"),
            ) as mocked_fetch:
                result = await generation_utils.call_openai_image_generation_with_retry_async(
                    model_name="gpt-image-2",
                    prompt="draw a circle",
                    config={"size": "1024x1024", "output_format": "png"},
                    max_attempts=1,
                    retry_delay=0,
                )

        self.assertEqual(result, ["downloaded-image-b64"])
        mocked_fetch.assert_awaited_once_with("https://example.test/image.png")

    def test_openai_image_extractor_rejects_html_as_base64(self):
        self.assertEqual(
            generation_utils._extract_base64_field(
                '<!doctype html><html><head><title>Gateway</title></head></html>'
            ),
            "",
        )

    async def test_openai_image_generation_falls_back_to_responses_tool(self):
        class _BadGateway(RuntimeError):
            status_code = 502

        fake_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="image_generation_call",
                    result="responses-image-b64",
                    status="completed",
                )
            ]
        )
        fake_client = SimpleNamespace(
            images=SimpleNamespace(
                generate=AsyncMock(side_effect=_BadGateway("upstream request failed")),
            ),
            responses=SimpleNamespace(
                create=AsyncMock(return_value=fake_response),
            ),
        )

        with patch.object(generation_utils, "get_openai_client", return_value=fake_client):
            result = await generation_utils.call_openai_image_generation_with_retry_async(
                model_name="gpt-image-2",
                prompt="draw a circle",
                config={
                    "size": "1024x1024",
                    "quality": "low",
                    "output_format": "png",
                    "responses_model": "gpt-5.4-mini",
                },
                max_attempts=1,
                retry_delay=0,
            )

        self.assertEqual(result, ["responses-image-b64"])
        sent = fake_client.responses.create.call_args.kwargs
        self.assertEqual(sent["model"], "gpt-5.4-mini")
        self.assertEqual(sent["tools"][0]["type"], "image_generation")
        self.assertEqual(sent["tools"][0]["model"], "gpt-image-2")
        self.assertEqual(sent["tool_choice"], {"type": "image_generation"})

    async def test_openai_responses_stream_fallback_emits_preview_events(self):
        class _BadGateway(RuntimeError):
            status_code = 502

        fake_client = SimpleNamespace(
            images=SimpleNamespace(
                generate=AsyncMock(side_effect=_BadGateway("upstream request failed")),
            ),
            responses=SimpleNamespace(
                create=AsyncMock(return_value=_FakeAsyncStream(_responses_stream_events(with_partial=True))),
            ),
        )
        captured_events = []
        original_hook = generation_utils.runtime_event_hook
        generation_utils.runtime_event_hook = captured_events.append

        try:
            with patch.object(generation_utils, "get_openai_client", return_value=fake_client):
                result = await generation_utils.call_openai_image_generation_with_retry_async(
                    model_name="gpt-image-2",
                    prompt="draw a circle",
                    config={
                        "size": "1024x1024",
                        "output_format": "png",
                        "stream": True,
                        "partial_images": 1,
                        "responses_model": "gpt-5.4-mini",
                    },
                    max_attempts=1,
                    retry_delay=0,
                    error_context="visualizer-image[candidate=9,key=render]",
                )
        finally:
            generation_utils.runtime_event_hook = original_hook

        self.assertEqual(result, ["responses-final-b64"])
        sent = fake_client.responses.create.call_args.kwargs
        self.assertTrue(sent["stream"])
        self.assertEqual(sent["tools"][0]["partial_images"], 1)
        preview_events = [event for event in captured_events if event.get("kind") == "preview_ready"]
        # output_item.done 与 response.completed 携带同一张最终图，只记一次
        self.assertEqual([event["preview_image"] for event in preview_events], ["responses-partial-b64", "responses-final-b64"])
        self.assertEqual(preview_events[0]["candidate_id"], "9")

    async def test_openai_responses_stream_without_partials_returns_final_image(self):
        result = await generation_utils._extract_openai_stream_images(
            _FakeAsyncStream(_responses_stream_events(with_partial=False)),
            preview_label_prefix="OpenAI Responses",
        )
        self.assertEqual(result, ["responses-final-b64"])

    def test_safe_log_text_redacts_openai_keys_and_bearer_tokens(self):
        text = generation_utils._safe_text_for_log(
            "Authorization: Bearer secret-token-value-12345 api_key='sk-1234567890abcdefSECRET'"
        )

        self.assertNotIn("secret-token-value-12345", text)
        self.assertNotIn("abcdefSECRET", text)

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
