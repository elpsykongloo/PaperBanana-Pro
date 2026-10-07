import unittest

from utils.demo_task_utils import (
    build_evolution_stages,
    collect_candidate_references,
    create_sample_inputs,
    find_final_stage_keys,
    get_task_ui_config,
    summarize_retrieval_meta,
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

    def test_build_evolution_stages_uses_registry_for_stylist_pipeline(self):
        result = {
            "exp_mode": "dev_planner_stylist",
            "target_diagram_desc0": "planner desc",
            "target_diagram_desc0_base64_jpg": "planner image",
            "target_diagram_stylist_desc0": "stylist desc",
            "target_diagram_stylist_desc0_base64_jpg": "stylist image",
        }

        stages = build_evolution_stages(result, "dev_planner_stylist", task_name="diagram")

        self.assertEqual([stage["name"] for stage in stages], ["📝 规划草案", "✨ 风格增强"])

    def test_plot_task_ui_config_marks_image_model_unused(self):
        plot_config = get_task_ui_config("plot")
        diagram_config = get_task_ui_config("diagram")

        self.assertFalse(plot_config["uses_image_model"])
        self.assertTrue(diagram_config["uses_image_model"])

    def test_summarize_retrieval_meta_handles_old_and_new_results(self):
        self.assertEqual(summarize_retrieval_meta(None), "")
        self.assertEqual(summarize_retrieval_meta({}), "")
        summary = summarize_retrieval_meta(
            {
                "requested_setting": "auto",
                "setting": "auto",
                "method": "bm25+llm",
                "mode": "lite",
                "pool_size": 298,
                "shortlist_size": 40,
                "selected": 2,
                "topped_up": 8,
                "query_rewritten": True,
                "shared": True,
            }
        )
        self.assertIn("关键词预筛 + 模型挑选", summary)
        self.assertIn("参考池 298 条", summary)
        self.assertIn("按预筛顺序补齐 8 条", summary)
        self.assertIn("中文输入已改写", summary)
        self.assertIn("复用本任务", summary)
        fallback = summarize_retrieval_meta({"requested_setting": "auto", "setting": "none", "method": "none"})
        self.assertIn("已从 auto 回退为 none", fallback)
        visual = summarize_retrieval_meta({"method": "bm25+vlm", "shortlist_size": 40, "thumbnails": 40, "selected": 10})
        self.assertIn("模型看缩略图挑选", visual)
        self.assertIn("预筛 40 条", visual)
        self.assertIn("附缩略图 40 张", visual)
        self.assertIn("补全了 3 个只含数字的 id", summarize_retrieval_meta({"method": "bm25+vlm", "repaired_ids": 3}))
        failed = summarize_retrieval_meta({"method": "bm25+llm", "visual_rerank_failed": True})
        self.assertIn("已改为只看 caption 挑选", failed)

    def test_collect_candidate_references_supports_id_only_results(self):
        result = {
            "top10_references": ["ref_2", "ref_9"],
            "retrieved_examples": [
                {"id": "ref_2", "visual_intent": "Overview pipeline.", "path_to_gt_image": "images/a.jpg"},
            ],
        }
        refs = collect_candidate_references(result)
        self.assertEqual([ref["id"] for ref in refs], ["ref_2", "ref_9"])
        self.assertEqual(refs[0]["caption"], "Overview pipeline.")
        self.assertIsNone(refs[1]["path_to_gt_image"])
        self.assertEqual(collect_candidate_references({}), [])


if __name__ == "__main__":
    unittest.main()
