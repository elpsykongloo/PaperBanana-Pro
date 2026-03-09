import tempfile
import unittest
from pathlib import Path

from utils.config import ExpConfig


CONFIG_YAML = """defaults:
  model_name: gemini-default-text
  image_model_name: gemini-default-image
evolink:
  api_key: ""
  base_url: https://api.evolink.ai
  model_name: evolink-text
  image_model_name: evolink-image
"""


class ExpConfigProviderDefaultsTest(unittest.TestCase):
    def test_gemini_provider_uses_default_section(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            config_dir = work_dir / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "model_config.yaml").write_text(CONFIG_YAML, encoding="utf-8")

            exp_config = ExpConfig(
                dataset_name="PaperBananaBench",
                task_name="diagram",
                provider="gemini",
                work_dir=work_dir,
            )

            self.assertEqual(exp_config.model_name, "gemini-default-text")
            self.assertEqual(exp_config.image_model_name, "gemini-default-image")

    def test_evolink_provider_uses_evolink_section(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(tmp_dir)
            config_dir = work_dir / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "model_config.yaml").write_text(CONFIG_YAML, encoding="utf-8")

            exp_config = ExpConfig(
                dataset_name="PaperBananaBench",
                task_name="plot",
                provider="evolink",
                work_dir=work_dir,
            )

            self.assertEqual(exp_config.model_name, "evolink-text")
            self.assertEqual(exp_config.image_model_name, "evolink-image")
            self.assertTrue(exp_config.result_dir.exists())


if __name__ == "__main__":
    unittest.main()
