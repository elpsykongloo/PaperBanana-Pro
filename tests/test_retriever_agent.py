import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

from agents.retriever_agent import VISUAL_RERANK_SYSTEM_PROMPT, RetrieverAgent
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

        # 模型只选了 1 个，按预筛顺序补齐（参考池只有 2 条）
        self.assertEqual(result["top10_references"], ["ref_hit", "ref_miss"])
        self.assertEqual([item["id"] for item in result["retrieved_examples"]], ["ref_hit", "ref_miss"])
        self.assertEqual(result["retrieval_meta"]["selected"], 1)
        self.assertEqual(result["retrieval_meta"]["topped_up"], 1)
        self.assertEqual(result["retrieval_meta"]["method"], "bm25+llm")

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
        self.assertEqual(result["retrieval_meta"]["fallback"], "shortlist_order")
        self.assertEqual(result["retrieval_meta"]["invalid_ids"], ["missing_ref"])

    def _write_diagram_pool(self, work_dir: Path, items: list[dict]) -> None:
        diagram_dir = work_dir / "data" / "PaperBananaBench" / "diagram"
        diagram_dir.mkdir(parents=True, exist_ok=True)
        (diagram_dir / "ref.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _filler_diagrams(count: int) -> list[dict]:
        topics = ["protein folding", "weather forecasting", "speech codec", "graph partition", "robot grasping"]
        return [
            {
                "id": f"ref_{idx}",
                "visual_intent": f"Figure: pipeline for {topics[idx % len(topics)]} variant {idx}.",
                "content": f"We study {topics[idx % len(topics)]} with a standard encoder.",
            }
            for idx in range(count)
        ]

    def test_prefilter_scores_entire_pool_without_cutoff(self):
        work_dir = self._build_work_dir()
        items = self._filler_diagrams(250)
        items[230] = {
            "id": "ref_late",
            "visual_intent": "Figure 1: Overview of the hierarchical retrieval augmented critic loop for diagram synthesis.",
            "content": "A hierarchical retrieval augmented critic loop refines diagram synthesis.",
        }
        self._write_diagram_pool(work_dir, items)
        agent = self._build_agent(work_dir, task_name="diagram")

        shortlist = agent._prefilter_candidate_pool(
            {
                "content": "We propose a hierarchical retrieval augmented critic loop for diagram synthesis.",
                "visual_intent": "Overview of the retrieval augmented critic loop.",
            },
            agent.task_config,
            lite=True,
        )

        self.assertEqual(shortlist[0]["id"], "ref_late")
        self.assertEqual(len(shortlist), agent.task_config["lite_prefilter_limit"])

    def test_auto_retrieval_tops_up_partial_selection_and_labels_candidates_by_id(self):
        work_dir = self._build_work_dir()
        self._write_diagram_pool(work_dir, self._filler_diagrams(50))
        agent = self._build_agent(work_dir, task_name="diagram")
        mock = AsyncMock(return_value=['{"top10_references":["ref_3","ref_7","ref_9999"]}'])

        with patch("agents.retriever_agent.generation_utils.call_evolink_text_with_retry_async", new=mock):
            result = asyncio.run(
                agent.process(
                    {"candidate_id": 0, "content": "speech codec pipeline", "visual_intent": "Pipeline for speech codec."},
                    retrieval_setting="auto",
                )
            )

        ids = result["top10_references"]
        self.assertEqual(ids[:2], ["ref_3", "ref_7"])
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(set(ids)), 10)
        meta = result["retrieval_meta"]
        self.assertEqual(meta["selected"], 2)
        self.assertEqual(meta["topped_up"], 8)
        self.assertEqual(meta["invalid_ids"], ["ref_9999"])
        prompt_text = mock.call_args.kwargs["contents"][0]["text"]
        self.assertIn("Candidate Diagram [ref_3]:", prompt_text)
        self.assertNotRegex(prompt_text, r"Candidate Diagram \d+:")

    def test_concurrent_candidates_share_one_retrieval_call(self):
        work_dir = self._build_work_dir()
        self._write_diagram_pool(work_dir, self._filler_diagrams(30))
        agent = self._build_agent(work_dir, task_name="diagram")

        async def slow_response(**kwargs):
            await asyncio.sleep(0.05)
            return ['{"top10_references":["ref_1","ref_2"]}']

        mock = AsyncMock(side_effect=slow_response)

        async def run_candidates():
            jobs = [
                agent.process(
                    {"candidate_id": idx, "content": "robot grasping pipeline with depth camera", "visual_intent": "Robot grasping."},
                    retrieval_setting="auto",
                )
                for idx in range(3)
            ]
            return await asyncio.gather(*jobs)

        with patch("agents.retriever_agent.generation_utils.call_evolink_text_with_retry_async", new=mock):
            results = asyncio.run(run_candidates())

        self.assertEqual(mock.await_count, 1)
        self.assertEqual(len({tuple(r["top10_references"]) for r in results}), 1)
        self.assertEqual(sorted(r["retrieval_meta"]["shared"] for r in results), [False, True, True])
        # 每个候选拿到独立的示例副本
        results[0]["retrieved_examples"][0]["visual_intent"] = "mutated"
        self.assertNotEqual(results[1]["retrieved_examples"][0]["visual_intent"], "mutated")

    def test_chinese_query_is_rewritten_before_prefilter(self):
        work_dir = self._build_work_dir()
        items = self._filler_diagrams(60)
        items[45] = {
            "id": "ref_target",
            "visual_intent": "Figure 2: Multi-agent retrieval augmented framework for academic illustration.",
            "content": "Agents retrieve references and a critic refines the illustration.",
        }
        self._write_diagram_pool(work_dir, items)
        agent = self._build_agent(work_dir, task_name="diagram")
        prompts: list[str] = []

        async def fake_llm(**kwargs):
            text = kwargs["contents"][0]["text"]
            prompts.append(text)
            if "English keywords" in text:
                return ["multi-agent, retrieval augmented, academic illustration, critic, framework"]
            return ['{"top10_references":["ref_target"]}']

        with patch(
            "agents.retriever_agent.generation_utils.call_evolink_text_with_retry_async",
            new=AsyncMock(side_effect=fake_llm),
        ):
            result = asyncio.run(
                agent.process(
                    {
                        "candidate_id": 0,
                        "content": "我们提出一个多智能体框架，检索参考样例并由评审智能体迭代改进学术插图。",
                        "visual_intent": "图 2：整体框架示意图。",
                    },
                    retrieval_setting="auto",
                )
            )

        self.assertEqual(len(prompts), 2)
        self.assertIn("Candidate Diagram [ref_target]:", prompts[1])
        self.assertEqual(result["top10_references"][0], "ref_target")
        self.assertTrue(result["retrieval_meta"]["query_rewritten"])
        self.assertNotIn("fallback", result["retrieval_meta"])

    def test_chinese_query_without_rewrite_sends_all_captions(self):
        work_dir = self._build_work_dir()
        self._write_diagram_pool(work_dir, self._filler_diagrams(60))
        agent = self._build_agent(work_dir, task_name="diagram")
        prompts: list[str] = []

        async def fake_llm(**kwargs):
            text = kwargs["contents"][0]["text"]
            prompts.append(text)
            if "English keywords" in text:
                return ["Error"]
            return ['{"top10_references":["ref_59"]}']

        with patch(
            "agents.retriever_agent.generation_utils.call_evolink_text_with_retry_async",
            new=AsyncMock(side_effect=fake_llm),
        ):
            result = asyncio.run(
                agent.process(
                    {"candidate_id": 0, "content": "我们提出一种新的方法。", "visual_intent": "图 1：方法示意图。"},
                    retrieval_setting="auto-full",
                )
            )

        meta = result["retrieval_meta"]
        self.assertEqual(meta["fallback"], "no_query_tokens_all_captions")
        self.assertEqual(meta["shortlist_size"], 60)
        self.assertEqual(meta["mode"], "lite")
        self.assertIn("Candidate Diagram [ref_59]:", prompts[-1])
        self.assertNotIn("Methodology section: We study", prompts[-1])
        self.assertEqual(result["top10_references"][0], "ref_59")


    def _write_diagram_pool_with_images(self, work_dir: Path, count: int) -> list[dict]:
        items = self._filler_diagrams(count)
        image_dir = work_dir / "data" / "PaperBananaBench" / "diagram" / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            Image.new("RGB", (1200, 600), (255, 255, 255)).save(image_dir / f"{item['id']}.png")
            item["path_to_gt_image"] = f"images/{item['id']}.png"
        self._write_diagram_pool(work_dir, items)
        return items

    def test_auto_diagram_selection_attaches_candidate_thumbnails(self):
        work_dir = self._build_work_dir()
        self._write_diagram_pool_with_images(work_dir, 50)
        agent = self._build_agent(work_dir, task_name="diagram")
        mock = AsyncMock(return_value=['{"top10_references":["ref_3","ref_8"]}'])

        with patch("agents.retriever_agent.generation_utils.call_evolink_text_with_retry_async", new=mock):
            result = asyncio.run(
                agent.process(
                    {"candidate_id": 0, "content": "speech codec pipeline", "visual_intent": "Pipeline for speech codec."},
                    retrieval_setting="auto",
                )
            )

        self.assertEqual(mock.await_count, 1)
        kwargs = mock.call_args.kwargs
        self.assertEqual(kwargs["config"]["system_prompt"], VISUAL_RERANK_SYSTEM_PROMPT)
        contents = kwargs["contents"]
        image_parts = [part for part in contents if part["type"] == "image"]
        shortlist_size = agent.task_config["lite_prefilter_limit"]
        self.assertEqual(len(image_parts), shortlist_size)
        self.assertEqual(image_parts[0]["source"]["media_type"], "image/jpeg")
        self.assertIn("Candidate ref_", contents[1]["text"])
        meta = result["retrieval_meta"]
        self.assertEqual(meta["method"], "bm25+vlm")
        self.assertEqual(meta["thumbnails"], shortlist_size)
        self.assertEqual(result["top10_references"][:2], ["ref_3", "ref_8"])
        self.assertEqual(len(result["top10_references"]), 10)

    def test_thumbnail_selection_failure_falls_back_to_captions(self):
        work_dir = self._build_work_dir()
        self._write_diagram_pool_with_images(work_dir, 30)
        agent = self._build_agent(work_dir, task_name="diagram")
        mock = AsyncMock(side_effect=[RuntimeError("model does not accept images"), ['{"top10_references":["ref_4"]}']])

        with patch("agents.retriever_agent.generation_utils.call_evolink_text_with_retry_async", new=mock):
            result = asyncio.run(
                agent.process(
                    {"candidate_id": 0, "content": "graph partition pipeline", "visual_intent": "Pipeline for graph partition."},
                    retrieval_setting="auto",
                )
            )

        self.assertEqual(mock.await_count, 2)
        fallback_contents = mock.call_args_list[1].kwargs["contents"]
        self.assertTrue(all(part["type"] == "text" for part in fallback_contents))
        meta = result["retrieval_meta"]
        self.assertEqual(meta["method"], "bm25+llm")
        self.assertTrue(meta["visual_rerank_failed"])
        self.assertEqual(result["top10_references"][0], "ref_4")

    def test_auto_full_keeps_text_only_selection(self):
        work_dir = self._build_work_dir()
        self._write_diagram_pool_with_images(work_dir, 30)
        agent = self._build_agent(work_dir, task_name="diagram")
        mock = AsyncMock(return_value=['{"top10_references":["ref_2"]}'])

        with patch("agents.retriever_agent.generation_utils.call_evolink_text_with_retry_async", new=mock):
            result = asyncio.run(
                agent.process(
                    {"candidate_id": 0, "content": "robot grasping pipeline", "visual_intent": "Pipeline for robot grasping."},
                    retrieval_setting="auto-full",
                )
            )

        self.assertEqual(mock.await_count, 1)
        self.assertTrue(all(part["type"] == "text" for part in mock.call_args.kwargs["contents"]))
        self.assertEqual(result["retrieval_meta"]["method"], "bm25+llm")


    def test_numeric_ids_are_repaired_to_unique_shortlist_ids(self):
        work_dir = self._build_work_dir()
        self._write_diagram_pool_with_images(work_dir, 40)
        agent = self._build_agent(work_dir, task_name="diagram")
        mock = AsyncMock(return_value=['{"top10_references":["3","8","ref_3","9999"]}'])

        with patch("agents.retriever_agent.generation_utils.call_evolink_text_with_retry_async", new=mock):
            result = asyncio.run(
                agent.process(
                    {"candidate_id": 0, "content": "speech codec pipeline", "visual_intent": "Pipeline for speech codec."},
                    retrieval_setting="auto",
                )
            )

        self.assertEqual(result["top10_references"][:2], ["ref_3", "ref_8"])
        meta = result["retrieval_meta"]
        self.assertEqual(meta["repaired_ids"], 2)
        self.assertEqual(meta["selected"], 2)
        self.assertEqual(meta["invalid_ids"], ["9999"])
        final_text = mock.call_args.kwargs["contents"][-1]["text"]
        self.assertIn('Copy each id exactly as written after "Diagram ID:"', final_text)

    def test_numeric_id_repair_skips_ambiguous_numbers(self):
        ids, repaired = RetrieverAgent._repair_numeric_ids(["3", "7"], ["ref_3", "img_3", "ref_7"])
        self.assertEqual(ids, ["3", "ref_7"])
        self.assertEqual(repaired, 1)


if __name__ == "__main__":
    unittest.main()
