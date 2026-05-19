import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agents.retriever_agent import RetrieverAgent
from utils.config import ExpConfig


CONFIG_YAML = """defaults:
  model_name: test-text-model
  image_model_name: test-image-model
evolink:
  api_key: dummy-key
  model_name: evolink-text-model
  image_model_name: evolink-image-model
"""


class RetrieverAgentTest(unittest.TestCase):
    def _build_work_dir(self) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        work_dir = Path(temp_dir.name)
        (work_dir / "configs").mkdir(parents=True, exist_ok=True)
        (work_dir / "configs" / "model_config.yaml").write_text(CONFIG_YAML, encoding="utf-8")
        return work_dir

    def _build_agent(
        self,
        work_dir: Path,
        *,
        task_name: str,
        curated_profile: str = "default",
    ) -> RetrieverAgent:
        exp_config = ExpConfig(
            dataset_name="PaperBananaBench",
            task_name=task_name,
            exp_mode="demo_full",
            provider="evolink",
            curated_profile=curated_profile,
            work_dir=work_dir,
        )
        return RetrieverAgent(exp_config=exp_config)

    def test_plot_manual_alias_loads_legacy_curated_examples(self):
        work_dir = self._build_work_dir()
        plot_dir = work_dir / "data" / "PaperBananaBench" / "plot"
        plot_dir.mkdir(parents=True, exist_ok=True)
        manual_examples = [
            {"id": "plot_ref_1", "visual_intent": "line plot", "content": [{"x": 1, "y": 2}]},
            {"id": "plot_ref_2", "visual_intent": "bar plot", "content": [{"x": 1, "y": 3}]},
        ]
        (plot_dir / "agent_selected_12.json").write_text(
            json.dumps(manual_examples, ensure_ascii=False),
            encoding="utf-8",
        )
        agent = self._build_agent(work_dir, task_name="plot")

        result = asyncio.run(
            agent.process(
                {
                    "candidate_id": 0,
                    "content": '[{"step": 1, "score": 62.1}]',
                    "visual_intent": "Create a line plot of score over step.",
                },
                retrieval_setting="manual",
            )
        )

        self.assertEqual(result["top10_references"], ["plot_ref_1", "plot_ref_2"])
        self.assertEqual(len(result["retrieved_examples"]), 2)
        self.assertEqual(result["curated_profile"], "default")
        self.assertEqual(result["curated_profile_source"], "agent_selected_12.json")

    def test_curated_profile_can_join_selected_ids_from_ref_pool(self):
        work_dir = self._build_work_dir()
        diagram_dir = work_dir / "data" / "PaperBananaBench" / "diagram"
        profile_dir = diagram_dir / "manual_profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        reference_examples = [
            {
                "id": "ref_hit",
                "visual_intent": "Overview diagram for an agent pipeline.",
                "content": "We encode papers and refine figures with a critic loop.",
            },
            {
                "id": "ref_extra",
                "visual_intent": "Ablation bar chart",
                "content": "Compare model variants.",
            },
        ]
        (diagram_dir / "ref.json").write_text(
            json.dumps(reference_examples, ensure_ascii=False),
            encoding="utf-8",
        )
        (profile_dir / "paper-profile.json").write_text(
            json.dumps({"selected_ids": ["ref_hit"]}, ensure_ascii=False),
            encoding="utf-8",
        )
        agent = self._build_agent(
            work_dir,
            task_name="diagram",
            curated_profile="paper-profile",
        )

        result = asyncio.run(
            agent.process(
                {
                    "candidate_id": 0,
                    "content": "We encode papers and refine figures with a critic loop.",
                    "visual_intent": "Overview diagram for an agent pipeline.",
                },
                retrieval_setting="curated",
            )
        )

        self.assertEqual(result["top10_references"], ["ref_hit"])
        self.assertEqual([item["id"] for item in result["retrieved_examples"]], ["ref_hit"])
        self.assertEqual(result["curated_profile"], "paper-profile")
        self.assertEqual(result["curated_profile_source"], "paper-profile.json")

    def test_prefilter_candidate_pool_keeps_relevant_examples(self):
        work_dir = self._build_work_dir()
        plot_dir = work_dir / "data" / "PaperBananaBench" / "plot"
        plot_dir.mkdir(parents=True, exist_ok=True)
        candidates = [
            {
                "id": "relevant_plot",
                "visual_intent": "Create a line plot of score over step with one line per method.",
                "content": [{"method": "PaperBanana", "step": 1, "score": 62.1}],
            },
            {
                "id": "also_relevant",
                "visual_intent": "Line chart for accuracy versus training tokens.",
                "content": [{"family": "PaperBanana", "tokens": 10, "accuracy": 58.4}],
            },
        ]
        for idx in range(80):
            candidates.append(
                {
                    "id": f"irrelevant_{idx}",
                    "visual_intent": f"Scatter plot of random noise {idx}",
                    "content": [{"foo": idx, "bar": idx + 1}],
                }
            )
        (plot_dir / "ref.json").write_text(json.dumps(candidates, ensure_ascii=False), encoding="utf-8")
        agent = self._build_agent(work_dir, task_name="plot")

        shortlist = agent._prefilter_candidate_pool(
            {
                "content": '[{"method":"PaperBanana","step":1,"score":62.1}]',
                "visual_intent": "Create a line plot of score over step with one line per method.",
            },
            agent.task_config,
            lite=True,
        )
        shortlisted_ids = [item["id"] for item in shortlist]

        self.assertLessEqual(len(shortlist), agent.task_config["lite_prefilter_limit"])
        self.assertIn("relevant_plot", shortlisted_ids)
        self.assertIn("also_relevant", shortlisted_ids)

    def test_parse_retrieval_result_accepts_common_output_shapes(self):
        work_dir = self._build_work_dir()
        agent = self._build_agent(work_dir, task_name="diagram")

        self.assertEqual(
            agent._parse_retrieval_result('{"top10_references":["ref_a","ref_b"]}', "diagram"),
            ["ref_a", "ref_b"],
        )
        self.assertEqual(
            agent._parse_retrieval_result('```json\n["ref_a","ref_b"]\n```', "diagram"),
            ["ref_a", "ref_b"],
        )
        self.assertEqual(
            agent._parse_retrieval_result(
                '{"selected_ids":[{"id":"ref_a"},{"reference_id":"ref_b"}]}',
                "diagram",
            ),
            ["ref_a", "ref_b"],
        )

    def test_auto_retrieval_populates_retrieved_examples(self):
        work_dir = self._build_work_dir()
        diagram_dir = work_dir / "data" / "PaperBananaBench" / "diagram"
        diagram_dir.mkdir(parents=True, exist_ok=True)
        candidates = [
            {
                "id": "ref_hit",
                "visual_intent": "Overview diagram for an agent pipeline.",
                "content": "We encode papers and refine figures with a critic loop.",
            },
            {
                "id": "ref_miss",
                "visual_intent": "Ablation bar chart",
                "content": "Compare model variants.",
            },
        ]
        (diagram_dir / "ref.json").write_text(json.dumps(candidates, ensure_ascii=False), encoding="utf-8")
        agent = self._build_agent(work_dir, task_name="diagram")

        with patch(
            "agents.retriever_agent.generation_utils.call_evolink_text_with_retry_async",
            new=AsyncMock(return_value=['{"top10_diagrams":["ref_hit"]}']),
        ):
            result = asyncio.run(
                agent.process(
                    {
                        "candidate_id": 0,
                        "content": "We encode papers and refine figures with a critic loop.",
                        "visual_intent": "Overview diagram for an agent pipeline.",
                    },
                    retrieval_setting="auto",
                )
            )

        self.assertEqual(result["top10_references"], ["ref_hit"])
        self.assertEqual([item["id"] for item in result["retrieved_examples"]], ["ref_hit"])

    def test_auto_retrieval_falls_back_when_model_returns_no_valid_ids(self):
        work_dir = self._build_work_dir()
        diagram_dir = work_dir / "data" / "PaperBananaBench" / "diagram"
        diagram_dir.mkdir(parents=True, exist_ok=True)
        candidates = [
            {
                "id": "ref_first",
                "visual_intent": "Overview diagram for an agent pipeline.",
                "content": "We encode papers and refine figures with a critic loop.",
            },
            {
                "id": "ref_second",
                "visual_intent": "Detailed module diagram for retrieval.",
                "content": "A retriever ranks references and forwards selected ids.",
            },
        ]
        (diagram_dir / "ref.json").write_text(json.dumps(candidates, ensure_ascii=False), encoding="utf-8")
        agent = self._build_agent(work_dir, task_name="diagram")

        with patch(
            "agents.retriever_agent.generation_utils.call_evolink_text_with_retry_async",
            new=AsyncMock(return_value=['{"top10_references":["missing_ref"]}']),
        ):
            result = asyncio.run(
                agent.process(
                    {
                        "candidate_id": 0,
                        "content": "We encode papers and refine figures with a critic loop.",
                        "visual_intent": "Overview diagram for an agent pipeline.",
                    },
                    retrieval_setting="auto",
                )
            )

        self.assertEqual(result["top10_references"], ["ref_first", "ref_second"])
        self.assertEqual([item["id"] for item in result["retrieved_examples"]], ["ref_first", "ref_second"])


if __name__ == "__main__":
    unittest.main()
