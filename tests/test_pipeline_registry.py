import unittest

from utils.pipeline_registry import get_pipeline_spec, get_supported_exp_modes


class PipelineRegistryTest(unittest.TestCase):
    def test_demo_full_pipeline_spec(self):
        spec = get_pipeline_spec("demo_full")

        self.assertEqual(
            spec.stages,
            ("retriever", "planner", "stylist", "visualizer", "critic"),
        )
        self.assertEqual(spec.critic_source, "stylist")
        self.assertTrue(spec.disable_eval)

    def test_supported_modes_include_cli_default(self):
        supported = get_supported_exp_modes()

        self.assertIn("dev_full", supported)
        self.assertIn("demo_planner_critic", supported)
        self.assertIn("vanilla", supported)

    def test_unknown_pipeline_spec_raises(self):
        with self.assertRaises(ValueError):
            get_pipeline_spec("dev")


if __name__ == "__main__":
    unittest.main()
