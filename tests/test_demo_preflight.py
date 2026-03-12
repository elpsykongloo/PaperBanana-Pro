import importlib
import sys
import types
import unittest
from pathlib import Path


if "streamlit" not in sys.modules:
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.set_page_config = lambda *args, **kwargs: None
    fake_streamlit.session_state = {}
    sys.modules["streamlit"] = fake_streamlit

demo = importlib.import_module("demo")


class DemoPreflightTest(unittest.TestCase):
    def test_preflight_report_flags_missing_required_inputs_and_parse_errors(self):
        report = demo.build_generation_preflight_report(
            task_name="plot",
            input_content="",
            visual_intent="",
            content_for_generation="",
            allow_raw_plot_input=False,
            num_candidates=3,
            quality_profile="标准质量",
            effective_settings={
                "retrieval_setting": "auto",
                "max_critic_rounds": 1,
                "exp_mode": "demo_planner_critic",
                "api_key": "",
            },
            retrieval_ref_path=Path("D:/PaperBanana/data/PaperBananaBench/plot/ref.json"),
            resolved_profile_path=None,
            generation_is_running=False,
        )

        self.assertGreaterEqual(len(report["errors"]), 3)
        self.assertTrue(any("缺少主体输入内容" in item for item in report["errors"]))
        self.assertTrue(any("缺少可视化目标" in item for item in report["errors"]))
        self.assertTrue(any("plot 输入尚未通过结构化解析" in item for item in report["errors"]))
        self.assertTrue(any("API Key" in item for item in report["warnings"]))

    def test_build_generation_effective_settings_applies_quality_profile_defaults(self):
        effective = demo._build_generation_effective_settings(
            "高质量",
            {
                "exp_mode": "demo_planner_critic",
                "retrieval_setting": "none",
                "max_critic_rounds": 0,
                "image_resolution": "2K",
            },
            task_name="diagram",
        )

        self.assertEqual(effective["exp_mode"], "demo_full")
        self.assertEqual(effective["retrieval_setting"], "auto-full")
        self.assertEqual(effective["max_critic_rounds"], 3)
        self.assertEqual(effective["image_resolution"], "4K")


if __name__ == "__main__":
    unittest.main()
