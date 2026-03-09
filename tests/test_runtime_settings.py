import tempfile
import unittest
from pathlib import Path

from utils.runtime_settings import (
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
"""


class RuntimeSettingsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
