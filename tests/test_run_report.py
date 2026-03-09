import unittest

from utils.image_utils import build_gemini_image_prompt, normalize_gemini_media_resolution
from utils.run_report import build_failure_manifest, build_result_summary


class RunReportTest(unittest.TestCase):
    def test_summary_counts_failures_parse_errors_and_missing_renders(self):
        results = [
            {
                "candidate_id": 0,
                "status": "ok",
                "eval_image_field": "target_diagram_desc0_base64_jpg",
                "target_diagram_desc0_base64_jpg": "abc",
            },
            {
                "candidate_id": 1,
                "status": "failed",
                "error": "RuntimeError: boom",
                "error_detail": "traceback",
                "eval_image_field": None,
            },
            {
                "candidate_id": 2,
                "status": "ok",
                "eval_image_field": None,
                "target_plot_critic_status0": "parse_error",
            },
        ]

        summary = build_result_summary(results)

        self.assertEqual(summary["total_candidates"], 3)
        self.assertEqual(summary["failed_candidates"], 1)
        self.assertEqual(summary["failed_candidate_ids"], [1])
        self.assertEqual(summary["missing_render_candidates"], [1, 2])
        self.assertEqual(summary["parse_error_candidates"][0]["candidate_id"], 2)

    def test_failure_manifest_includes_all_failure_types(self):
        results = [
            {
                "candidate_id": 1,
                "status": "failed",
                "error": "RuntimeError: boom",
                "error_detail": "traceback",
            },
            {
                "candidate_id": 2,
                "status": "ok",
                "eval_image_field": None,
                "target_plot_critic_status0": "parse_error",
            },
        ]

        manifest = build_failure_manifest(results)
        manifest_types = [item["type"] for item in manifest]

        self.assertIn("pipeline_failure", manifest_types)
        self.assertIn("critic_parse_error", manifest_types)
        self.assertIn("missing_render", manifest_types)

    def test_gemini_media_resolution_mapping(self):
        self.assertEqual(
            normalize_gemini_media_resolution("1K"),
            "MEDIA_RESOLUTION_LOW",
        )
        self.assertEqual(
            normalize_gemini_media_resolution("2K"),
            "MEDIA_RESOLUTION_MEDIUM",
        )
        self.assertEqual(
            normalize_gemini_media_resolution("4K"),
            "MEDIA_RESOLUTION_HIGH",
        )

    def test_gemini_image_prompt_contains_render_hints(self):
        prompt = build_gemini_image_prompt("Draw a diagram.", "16:9", "4K")
        self.assertIn("Aspect ratio: 16:9", prompt)
        self.assertIn("Output resolution preference: 4K", prompt)


if __name__ == "__main__":
    unittest.main()
