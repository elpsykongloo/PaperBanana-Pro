import unittest

from utils.demo_task_utils import (
    build_evolution_stages,
    create_sample_inputs,
    find_final_stage_keys,
    get_task_ui_config,
)


class DemoTaskUtilsTest(unittest.TestCase):
    def test_create_sample_inputs_for_plot_preserves_task_name(self):
        inputs = create_sample_inputs(
            content='[{"x": 1, "y": 2}]',
            visual_intent="Create a scatter plot.",
            task_name="plot",
            num_copies=2,
            max_critic_rounds=4,
        )

        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0]["task_name"], "plot")
        self.assertEqual(inputs[0]["candidate_id"], 0)
        self.assertEqual(inputs[1]["candidate_id"], 1)
        self.assertEqual(inputs[0]["max_critic_rounds"], 4)
        self.assertEqual(inputs[0]["visual_intent"], "Create a scatter plot.")

    def test_find_final_stage_keys_prefers_latest_critic_round(self):
        result = {
            "target_plot_desc0_base64_jpg": "planner",
            "target_plot_critic_desc0_base64_jpg": "round0",
            "target_plot_critic_desc2_base64_jpg": "round2",
        }

        image_key, desc_key = find_final_stage_keys(
            result,
            task_name="plot",
            exp_mode="demo_full",
        )

        self.assertEqual(image_key, "target_plot_critic_desc2_base64_jpg")
        self.assertEqual(desc_key, "target_plot_critic_desc2")

    def test_build_evolution_stages_for_plot_includes_code(self):
        result = {
            "target_plot_desc0": "planner desc",
            "target_plot_desc0_base64_jpg": "planner image",
            "target_plot_desc0_code": "print('planner')",
            "target_plot_critic_desc0": "critic desc",
            "target_plot_critic_desc0_base64_jpg": "critic image",
            "target_plot_critic_desc0_code": "print('critic')",
            "target_plot_critic_suggestions0": "Tighten labels.",
        }

        stages = build_evolution_stages(result, "demo_planner_critic", task_name="plot")

        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0]["code_key"], "target_plot_desc0_code")
        self.assertEqual(stages[1]["code_key"], "target_plot_critic_desc0_code")
        self.assertEqual(stages[1]["suggestions_key"], "target_plot_critic_suggestions0")

    def test_plot_task_ui_config_marks_image_model_unused(self):
        plot_config = get_task_ui_config("plot")
        diagram_config = get_task_ui_config("diagram")

        self.assertFalse(plot_config["uses_image_model"])
        self.assertTrue(diagram_config["uses_image_model"])


if __name__ == "__main__":
    unittest.main()
