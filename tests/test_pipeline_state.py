import unittest

from utils.pipeline_state import (
    PipelineState,
    collect_parse_error_round_keys,
    find_final_stage_keys,
    get_render_options,
)


class PipelineStateTest(unittest.TestCase):
    def test_find_final_stage_keys_prefers_eval_image_field(self):
        result = {
            "eval_image_field": "target_diagram_critic_desc1_base64_jpg",
            "target_diagram_critic_desc1_base64_jpg": "abc",
        }

        image_key, desc_key = find_final_stage_keys(result, "diagram", "demo_full")

        self.assertEqual(image_key, "target_diagram_critic_desc1_base64_jpg")
        self.assertEqual(desc_key, "target_diagram_critic_desc1")

    def test_collect_parse_error_round_keys(self):
        result = {
            "target_plot_critic_status0": "ok",
            "target_plot_critic_status1": "parse_error",
            "target_diagram_critic_status3": "parse_error",
        }

        self.assertEqual(
            collect_parse_error_round_keys(result),
            ["target_diagram_critic_status3", "target_plot_critic_status1"],
        )

    def test_get_render_options_uses_additional_info(self):
        options = get_render_options(
            {
                "additional_info": {
                    "rounded_ratio": "16:9",
                    "image_resolution": "4K",
                }
            },
            default_aspect_ratio="1:1",
            default_image_resolution="2K",
        )

        self.assertEqual(options.aspect_ratio, "16:9")
        self.assertEqual(options.image_resolution, "4K")

    def test_pipeline_state_round_accessor(self):
        payload = {}
        state = PipelineState(payload, "plot")
        state.current_critic_round = 2

        self.assertEqual(state.current_critic_round, 2)
        self.assertEqual(payload["current_critic_round"], 2)


if __name__ == "__main__":
    unittest.main()
