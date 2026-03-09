import unittest

from utils.concurrency import compute_effective_concurrency


class ConcurrencyHeuristicsTest(unittest.TestCase):
    def test_manual_mode_respects_requested_limit(self):
        self.assertEqual(
            compute_effective_concurrency(
                "manual",
                7,
                10,
                task_name="diagram",
                retrieval_setting="auto-full",
                exp_mode="demo_full",
                provider="gemini",
            ),
            7,
        )

    def test_auto_mode_limits_heavy_diagram_pipeline(self):
        self.assertEqual(
            compute_effective_concurrency(
                "auto",
                12,
                20,
                task_name="diagram",
                retrieval_setting="auto-full",
                exp_mode="demo_full",
                provider="gemini",
            ),
            2,
        )

    def test_auto_mode_allows_more_parallelism_for_plot_none_retrieval(self):
        self.assertEqual(
            compute_effective_concurrency(
                "auto",
                12,
                20,
                task_name="plot",
                retrieval_setting="none",
                exp_mode="vanilla",
                provider="gemini",
            ),
            6,
        )


if __name__ == "__main__":
    unittest.main()
