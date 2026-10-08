import os
import unittest
from unittest.mock import patch

from scripts.eval import common, retrieval_eval
from scripts.eval.common import BASELINE_DIR, combine_pairwise, load_json, mean_score, paired_bootstrap_ci


class EvalStatsTest(unittest.TestCase):
    def test_mean_score_counts_unscored_refs_as_zero(self):
        scores = {"a": 3, "b": 1, "c": None}
        self.assertAlmostEqual(mean_score(["a", "b", "c", "d"], scores, 4), 1.0)
        self.assertAlmostEqual(mean_score(["a", "b"], scores, 10), 2.0)
        self.assertEqual(mean_score([], scores, 10), 0.0)

    def test_paired_bootstrap_ci_brackets_the_mean(self):
        diff, low, high = paired_bootstrap_ci([2, 3, 4, 5], [1, 1, 1, 1], n=500)
        self.assertAlmostEqual(diff, 2.5)
        self.assertLessEqual(low, diff)
        self.assertGreaterEqual(high, diff)

    def test_pairwise_needs_both_orders_to_win(self):
        self.assertEqual(combine_pairwise({"overall": "A"}, {"overall": "B"}, "overall"), "win")
        self.assertEqual(combine_pairwise({"overall": "B"}, {"overall": "A"}, "overall"), "loss")
        self.assertEqual(combine_pairwise({"overall": "A"}, {"overall": "A"}, "overall"), "tie")
        self.assertEqual(combine_pairwise({"overall": "A"}, {"overall": "Tie"}, "overall"), "win")
        self.assertEqual(combine_pairwise(None, {}, "overall"), "tie")

    def test_call_cost_prices_image_tokens_separately(self):
        call = {"model": "gemini-nano-banana-2.1", "image": True, "in": 1000, "out": 3000, "image_tokens": 1680, "thoughts": 500}
        expected = (1000 * 1.50 + 1680 * 30.0 + (3000 - 1680 + 500) * 7.50) / 1e6
        self.assertAlmostEqual(common.call_cost(call), expected)
        self.assertEqual(common.call_cost({"model": "unknown-model", "in": 10}), 0.0)

    def test_missing_api_key_is_reported_without_value(self):
        with patch.dict(os.environ, {common.API_KEY_ENV: ""}):
            with self.assertRaises(common.MissingApiKeyError):
                common.load_api_key()


class RetrievalBaselineTest(unittest.TestCase):
    def test_every_reference_in_baselines_has_a_score(self):
        for name in ("retrieval_40.json", "retrieval_80.json"):
            data = load_json(BASELINE_DIR / name)
            for item in data["items"]:
                for arm, payload in item["arms"].items():
                    for ref_id in payload["refs"]:
                        self.assertIn(ref_id, item["judge_scores"], f"{name} {item['id']} {arm}")

    def test_report_reproduces_recorded_results(self):
        data = load_json(BASELINE_DIR / "retrieval_40.json")
        self.assertEqual(len(data["items"]), 40)
        at10 = {arm: round(retrieval_eval.summarize_arm(data, arm)["at10"], 2) for arm in (
            "flash-lite-3.1-caption", "flash-lite-3.1-thumbs", "flash-lite-3.5-thumbs", "flash-3.8-thumbs")}
        self.assertEqual(at10, {"flash-lite-3.1-caption": 1.48, "flash-lite-3.1-thumbs": 1.62,
                                "flash-lite-3.5-thumbs": 1.61, "flash-3.8-thumbs": 1.68})
        upper = retrieval_eval.summarize_arm(load_json(BASELINE_DIR / "retrieval_80.json"), retrieval_eval.UPPER_BOUND_ARM)
        self.assertAlmostEqual(upper["at10"], 2.03, places=2)

    def test_best_judged_prefers_higher_scores(self):
        item = {"judge_scores": {"r1": 1, "r2": 3, "r3": None, "r4": 2}}
        self.assertEqual(retrieval_eval.best_judged(item, 2), ["r2", "r4"])

    def test_image_prompt_baseline_keeps_descriptions(self):
        data = load_json(BASELINE_DIR / "image_prompts_16.json")
        self.assertEqual(len(data["items"]), 16)
        for item in data["items"]:
            self.assertTrue(item["descriptions"]["stylist_v2"].strip())
            self.assertIn("V2_vs_V0", item["pairs"])


if __name__ == "__main__":
    unittest.main()
