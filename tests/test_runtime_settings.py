import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.runtime_settings import (
    RuntimeSettings,
    build_runtime_context,
    build_provider_ui_defaults,
    resolve_runtime_settings,
)


CONFIG_YAML = """defaults:
  model_name: gemini-default-text
  image_model_name: gemini-default-image
evolink:
  api_key: yaml-evolink-key
  base_url: https://api.evolink.ai
  model_name: evolink-text
  image_model_name: evolink-image
openai:
  api_key: yaml-openai-key
  base_url: https://api.openai.com/v1
  model_name: openai-text
  image_model_name: gpt-image-2
"""

PROVIDER_REGISTRY_YAML = """version: 1
connections:
  - connection_id: custom-openai
    display_name: 自定义 OpenAI
    provider_type: openai_compatible
    protocol_family: openai
    base_url: https://example.com/v1
    api_key_env_var: CUSTOM_OPENAI_API_KEY
    text_model: custom-text
    image_model: custom-image
    model_discovery_mode: hybrid
    model_allowlist:
      - custom-text
      - custom-image
    extra_headers:
      X-Test: abc
    supports_text: true
    supports_image: true
    enabled: true
"""


class RuntimeSettingsTest(unittest.TestCase):
    def _write_custom_connection_fixture(self, root: Path) -> None:
        config_dir = root / "configs"
        local_dir = config_dir / "local"
        provider_dir = local_dir / "providers"
        provider_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "model_config.yaml").write_text(CONFIG_YAML, encoding="utf-8")
        (config_dir / "provider_registry.yaml").write_text(PROVIDER_REGISTRY_YAML, encoding="utf-8")
        (provider_dir / "custom-openai.txt").write_text("custom-local-key\n", encoding="utf-8")

    def test_resolve_runtime_settings_uses_local_secret_then_yaml_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            local_dir = config_dir / "local"
            local_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "model_config.yaml").write_text(CONFIG_YAML, encoding="utf-8")
            (local_dir / "google_api_key.txt").write_text("local-google-key", encoding="utf-8")

            settings = resolve_runtime_settings(
                "gemini",
                base_dir=root,
            )

            self.assertEqual(settings.provider, "gemini")
            self.assertEqual(settings.api_key, "local-google-key")
            self.assertEqual(settings.model_name, "gemini-default-text")
            self.assertEqual(settings.image_model_name, "gemini-default-image")

    def test_resolve_runtime_settings_prefers_connection_id_for_custom_connection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_custom_connection_fixture(root)

            settings = resolve_runtime_settings(
                "gemini",
                connection_id="custom-openai",
                base_dir=root,
            )

            self.assertEqual(settings.provider, "openai_compatible")
            self.assertEqual(settings.connection_id, "custom-openai")
            self.assertEqual(settings.provider_display_name, "自定义 OpenAI")
            self.assertEqual(settings.api_key, "custom-local-key")
            self.assertEqual(settings.base_url, "https://example.com/v1")
            self.assertEqual(settings.model_name, "custom-text")
            self.assertEqual(settings.image_model_name, "custom-image")
            self.assertEqual(settings.extra_headers, {"X-Test": "abc"})

    def test_build_provider_ui_defaults_reads_custom_connection_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._write_custom_connection_fixture(root)

            defaults = build_provider_ui_defaults("custom-openai", base_dir=root)

            self.assertEqual(defaults["connection_id"], "custom-openai")
            self.assertEqual(defaults["display_name"], "自定义 OpenAI")
            self.assertEqual(defaults["provider_type"], "openai_compatible")
            self.assertEqual(defaults["base_url"], "https://example.com/v1")
            self.assertEqual(defaults["extra_headers"], {"X-Test": "abc"})
            self.assertTrue(defaults["supports_image"])

    def test_build_provider_ui_defaults_exposes_provider_specific_labels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "model_config.yaml").write_text(CONFIG_YAML, encoding="utf-8")

            defaults = build_provider_ui_defaults("evolink", base_dir=root)

            self.assertEqual(defaults["api_key_label"], "API Key")
            self.assertEqual(defaults["model_name"], "evolink-text")
            self.assertEqual(defaults["image_model_name"], "evolink-image")
            self.assertEqual(defaults["api_key_default"], "yaml-evolink-key")
            self.assertEqual(defaults["base_url"], "https://api.evolink.ai")

    def test_resolve_runtime_settings_supports_official_openai(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "model_config.yaml").write_text(CONFIG_YAML, encoding="utf-8")

            settings = resolve_runtime_settings("openai", base_dir=root)

            self.assertEqual(settings.provider, "openai")
            self.assertEqual(settings.connection_id, "openai")
            self.assertEqual(settings.image_model_name, "gpt-image-2")
            self.assertEqual(settings.api_key, "yaml-openai-key")
            self.assertEqual(settings.base_url, "https://api.openai.com/v1")

    def test_build_runtime_context_delegates_to_generation_utils(self):
        settings = RuntimeSettings(
            provider="gemini",
            api_key="runtime-key",
            model_name="text-model",
            image_model_name="image-model",
            base_url="",
        )
        hook = lambda message: message

        with patch("utils.generation_utils.create_runtime_context", return_value={"ok": True}) as mocked_create:
            context = build_runtime_context(settings, status_hook=hook)

        self.assertEqual(context, {"ok": True})
        mocked_create.assert_called_once_with(
            connection_id="",
            provider="gemini",
            api_key="runtime-key",
            base_url="",
            extra_headers={},
            event_hook=None,
            status_hook=hook,
            cancel_check=None,
        )


if __name__ == "__main__":
    unittest.main()
