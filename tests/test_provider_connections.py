import sys
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.provider_connections import (
    ProbeResult,
    ProviderConnection,
    classify_probe_error,
    discover_models,
    get_custom_provider_secret_path,
    list_provider_connections,
    load_connection_metadata,
    load_provider_registry,
    probe_image,
    probe_connection,
    probe_text,
    run_async_probe,
    upsert_custom_connection,
    write_connection_probe_result,
)


CONFIG_YAML = """defaults:
  model_name: gemini-default-text
  image_model_name: gemini-default-image
"""


class _StatusError(RuntimeError):
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


class ProviderConnectionsTest(unittest.TestCase):
    def _prepare_root(self) -> Path:
        base_dir = Path(__file__).resolve().parents[1] / ".tmp_tests" / "provider_connections"
        base_dir.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(dir=base_dir))
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        config_dir = root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "model_config.yaml").write_text(CONFIG_YAML, encoding="utf-8")
        return root

    def test_upsert_custom_connection_persists_registry_and_secret(self):
        root = self._prepare_root()

        connection = upsert_custom_connection(
            {
                "connection_id": "My Custom API",
                "display_name": "我的自定义连接",
                "base_url": "https://example.com/v1",
                "text_model": "custom-text",
                "image_model": "custom-image",
                "model_discovery_mode": "hybrid",
                "model_allowlist": ["custom-text", "custom-image"],
                "extra_headers": {"X-Test": "demo"},
                "supports_text": True,
                "supports_image": True,
            },
            api_key="secret-key",
            persist_secret=True,
            base_dir=root,
        )

        registry = load_provider_registry(root)
        connections = list_provider_connections(base_dir=root)
        secret_path = get_custom_provider_secret_path(connection.connection_id, base_dir=root)

        self.assertEqual(connection.connection_id, "my-custom-api")
        self.assertEqual(registry["connections"][0]["connection_id"], "my-custom-api")
        self.assertTrue(secret_path.exists())
        self.assertEqual(secret_path.read_text(encoding="utf-8").strip(), "secret-key")
        self.assertTrue(any(item.connection_id == "my-custom-api" for item in connections))

    def test_discover_models_uses_allowlist_when_models_endpoint_fails(self):
        root = self._prepare_root()
        connection = ProviderConnection(
            connection_id="custom-openai",
            display_name="自定义 OpenAI",
            provider_type="openai_compatible",
            protocol_family="openai",
            base_url="https://example.com/v1",
            text_model="manual-text",
            image_model="manual-image",
            model_allowlist=("manual-text", "manual-image"),
            api_key="secret",
        )

        class _FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.models = types.SimpleNamespace(list=self._list)

            async def _list(self):
                raise _StatusError("service unavailable", 503)

            async def close(self):
                return None

        fake_module = types.SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI)
        with patch.dict(sys.modules, {"openai": fake_module}):
            result = run_async_probe(discover_models(connection))

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_type, "provider_unavailable")
        self.assertEqual(list(result.discovered_models), ["manual-text", "manual-image"])

    def test_probe_text_uses_openai_compatible_runtime_route(self):
        connection = ProviderConnection(
            connection_id="custom-openai",
            display_name="自定义 OpenAI",
            provider_type="openai_compatible",
            protocol_family="openai",
            base_url="https://example.com/v1",
            text_model="custom-text",
            api_key="secret",
        )

        async def fake_call_openai_with_retry_async(**kwargs):
            return ["OK"]

        with patch("utils.generation_utils.call_openai_with_retry_async", side_effect=fake_call_openai_with_retry_async):
            result = run_async_probe(probe_text(connection))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.tested_model, "custom-text")

    def test_probe_text_uses_direct_gemini_client_without_retry_helper(self):
        connection = ProviderConnection(
            connection_id="gemini",
            display_name="Gemini",
            provider_type="gemini",
            protocol_family="gemini",
            text_model="gemini-text",
            api_key="secret",
        )

        class _FakeClient:
            def __init__(self, api_key: str):
                self.api_key = api_key
                self.models = types.SimpleNamespace(generate_content=self._generate_content)

            def _generate_content(self, **kwargs):
                return types.SimpleNamespace(
                    candidates=[
                        types.SimpleNamespace(
                            finish_reason=types.SimpleNamespace(name="STOP"),
                            content=types.SimpleNamespace(parts=[types.SimpleNamespace(text="OK")]),
                        )
                    ]
                )

        fake_genai_module = types.ModuleType("google.genai")
        fake_genai_module.Client = _FakeClient
        fake_genai_module.types = types.SimpleNamespace(
            GenerateContentConfig=lambda **kwargs: dict(kwargs),
        )
        fake_google_module = types.ModuleType("google")
        fake_google_module.genai = fake_genai_module

        with patch.dict(sys.modules, {"google": fake_google_module, "google.genai": fake_genai_module}):
            with patch(
                "utils.generation_utils.call_gemini_with_retry_async",
                side_effect=AssertionError("probe 不应复用 Gemini 重试 helper"),
            ):
                result = run_async_probe(probe_text(connection))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.tested_model, "gemini-text")
        self.assertIn("OK", result.raw_excerpt)

    def test_probe_image_returns_skipped_when_image_capability_disabled(self):
        connection = ProviderConnection(
            connection_id="custom-openai",
            display_name="自定义 OpenAI",
            provider_type="openai_compatible",
            protocol_family="openai",
            image_model="custom-image",
            supports_image=False,
        )

        result = run_async_probe(probe_image(connection))

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.stage, "capability_check")

    def test_probe_image_gemini_fast_fails_when_no_image_returned(self):
        connection = ProviderConnection(
            connection_id="gemini",
            display_name="Gemini",
            provider_type="gemini",
            protocol_family="gemini",
            image_model="gemini-image",
            api_key="secret",
            supports_image=True,
        )

        class _FakeClient:
            def __init__(self, api_key: str):
                self.api_key = api_key
                self.models = types.SimpleNamespace(generate_content=self._generate_content)

            def _generate_content(self, **kwargs):
                return types.SimpleNamespace(
                    candidates=[
                        types.SimpleNamespace(
                            finish_reason=types.SimpleNamespace(name="NO_IMAGE"),
                            content=types.SimpleNamespace(parts=[types.SimpleNamespace(text="no image returned")]),
                        )
                    ]
                )

        fake_genai_module = types.ModuleType("google.genai")
        fake_genai_module.Client = _FakeClient
        fake_genai_module.types = types.SimpleNamespace(
            GenerateContentConfig=lambda **kwargs: dict(kwargs),
        )
        fake_google_module = types.ModuleType("google")
        fake_google_module.genai = fake_genai_module

        with patch.dict(sys.modules, {"google": fake_google_module, "google.genai": fake_genai_module}):
            with patch(
                "utils.generation_utils.call_gemini_with_retry_async",
                side_effect=AssertionError("probe 不应复用 Gemini 重试 helper"),
            ):
                result = run_async_probe(probe_image(connection))

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_type, "response_incompatible")
        self.assertIn("NO_IMAGE", result.raw_excerpt)

    def test_probe_image_gemini_uses_official_compatible_fallback_and_reads_response_parts(self):
        connection = ProviderConnection(
            connection_id="gemini",
            display_name="Gemini",
            provider_type="gemini",
            protocol_family="gemini",
            image_model="gemini-image",
            api_key="secret",
            supports_image=True,
        )
        calls: list[dict] = []

        class _FakeClient:
            def __init__(self, api_key: str):
                self.api_key = api_key
                self.models = types.SimpleNamespace(generate_content=self._generate_content)

            def _generate_content(self, **kwargs):
                calls.append(dict(kwargs))
                if len(calls) == 1:
                    return types.SimpleNamespace(
                        parts=[],
                        candidates=[
                            types.SimpleNamespace(
                                finish_reason=types.SimpleNamespace(name="NO_IMAGE"),
                                content=types.SimpleNamespace(parts=[]),
                            )
                        ],
                    )
                return types.SimpleNamespace(
                    parts=[
                        types.SimpleNamespace(
                            text=None,
                            inline_data=types.SimpleNamespace(mime_type="image/jpeg", data=b"image-bytes"),
                        )
                    ],
                    candidates=[
                        types.SimpleNamespace(
                            finish_reason=types.SimpleNamespace(name="STOP"),
                            content=types.SimpleNamespace(parts=[]),
                        )
                    ],
                )

        fake_genai_module = types.ModuleType("google.genai")
        fake_genai_module.Client = _FakeClient
        fake_genai_module.types = types.SimpleNamespace(
            GenerateContentConfig=lambda **kwargs: dict(kwargs),
            ImageConfig=lambda **kwargs: dict(kwargs),
        )
        fake_google_module = types.ModuleType("google")
        fake_google_module.genai = fake_genai_module

        with patch.dict(sys.modules, {"google": fake_google_module, "google.genai": fake_genai_module}):
            result = run_async_probe(probe_image(connection))

        self.assertEqual(result.status, "success")
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["config"])
        self.assertEqual(
            calls[0]["contents"],
            "Create a simple blue circle icon on a white background. No text.",
        )
        self.assertEqual(calls[1]["config"]["response_modalities"], ["IMAGE"])
        self.assertIn("generate_content_image_only", result.raw_excerpt)

    def test_probe_image_routes_openrouter_to_chat_image_helper(self):
        connection = ProviderConnection(
            connection_id="openrouter",
            display_name="OpenRouter",
            provider_type="openrouter",
            protocol_family="openai",
            base_url="https://openrouter.ai/api/v1",
            image_model="sourceful/riverflow-v2-pro",
            api_key="secret",
            supports_image=True,
        )
        captured = {}

        async def fake_call_openrouter_image_generation_with_retry_async(**kwargs):
            captured.update(kwargs)
            return ["fake-image-b64"]

        with patch(
            "utils.generation_utils.call_openrouter_image_generation_with_retry_async",
            side_effect=fake_call_openrouter_image_generation_with_retry_async,
        ):
            result = run_async_probe(probe_image(connection))

        self.assertEqual(result.status, "success")
        self.assertEqual(captured["model_name"], "sourceful/riverflow-v2-pro")
        self.assertEqual(captured["config"]["aspect_ratio"], "1:1")
        self.assertEqual(captured["config"]["image_size"], "1K")

    def test_probe_connection_emits_stage_callbacks(self):
        connection = ProviderConnection(
            connection_id="custom-openai",
            display_name="自定义 OpenAI",
            provider_type="openai_compatible",
            protocol_family="openai",
            text_model="custom-text",
            image_model="custom-image",
            api_key="secret",
            supports_image=True,
        )
        seen_stages: list[tuple[str, str]] = []

        async def fake_discover(_connection):
            return ProbeResult(target="discovery", stage="models_list", status="success", message="ok")

        async def fake_text(_connection):
            return ProbeResult(target="text", stage="chat_completion", status="success", message="ok")

        async def fake_image(_connection):
            return ProbeResult(target="image", stage="image_generation", status="skipped", message="skip")

        with patch("utils.provider_connections.discover_models", side_effect=fake_discover):
            with patch("utils.provider_connections.probe_text", side_effect=fake_text):
                with patch("utils.provider_connections.probe_image", side_effect=fake_image):
                    run_async_probe(
                        probe_connection(
                            connection,
                            stage_callback=lambda target, status: seen_stages.append((target, status)),
                        )
                    )

        self.assertEqual(
            seen_stages,
            [
                ("discovery", "running"),
                ("discovery", "success"),
                ("text", "running"),
                ("text", "success"),
                ("image", "running"),
                ("image", "skipped"),
            ],
        )

    def test_classify_probe_error_maps_http_like_failures(self):
        error_type, status_code, _ = classify_probe_error(_StatusError("invalid api key", 401))
        self.assertEqual(error_type, "invalid_credentials")
        self.assertEqual(status_code, 401)

        error_type, status_code, _ = classify_probe_error(_StatusError("model not found", 404))
        self.assertEqual(error_type, "model_not_found")
        self.assertEqual(status_code, 404)

        error_type, status_code, _ = classify_probe_error(
            _StatusError("<!DOCTYPE html><html><title>Not Found</title>unauthorized</html>", 404)
        )
        self.assertEqual(error_type, "response_incompatible")
        self.assertEqual(status_code, 404)

    def test_write_connection_probe_result_updates_local_metadata(self):
        root = self._prepare_root()
        result = ProbeResult(
            target="text",
            stage="chat_completion",
            status="success",
            message="文本链路探针成功。",
            tested_model="custom-text",
            timestamp="2026-03-19T00:00:00+00:00",
        )

        write_connection_probe_result("custom-openai", result, base_dir=root)
        metadata = load_connection_metadata(root)

        self.assertIn("custom-openai", metadata)
        self.assertEqual(metadata["custom-openai"]["probe_results"]["text"]["status"], "success")


if __name__ == "__main__":
    unittest.main()
