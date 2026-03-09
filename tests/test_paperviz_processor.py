import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from utils.config import ExpConfig
from utils.paperviz_processor import PaperVizProcessor
from utils.pipeline_state import PipelineState


class _PassthroughAgent:
    def __init__(self, exp_config):
        self.exp_config = exp_config

    async def process(self, data, **kwargs):
        return data


class _PlannerAgent(_PassthroughAgent):
    async def process(self, data, **kwargs):
        state = PipelineState(data, self.exp_config.task_name)
        data[state.planner_desc_key()] = "planner description"
        return data


class _VisualizerAgent(_PassthroughAgent):
    async def process(self, data, **kwargs):
        state = PipelineState(data, self.exp_config.task_name)
        desc_key = state.stylist_desc_key() if state.stylist_desc_key() in data else state.planner_desc_key()
        if desc_key in data:
            data[state.image_key(desc_key)] = "fake-image"
        current_round = data.get("current_critic_round")
        if current_round is not None and state.critic_desc_key(current_round) in data:
            critic_desc_key = state.critic_desc_key(current_round)
            data[state.image_key(critic_desc_key)] = "critic-image"
        return data


class PaperVizProcessorRegistryTest(unittest.TestCase):
    def _build_processor(self, exp_mode: str) -> PaperVizProcessor:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        work_dir = Path(temp_dir.name)
        (work_dir / "configs").mkdir(parents=True, exist_ok=True)
        (work_dir / "configs" / "model_config.yaml").write_text(
            "defaults:\n  model_name: gemini-text\n  image_model_name: gemini-image\n",
            encoding="utf-8",
        )
        exp_config = ExpConfig(
            dataset_name="PaperBananaBench",
            task_name="diagram",
            exp_mode=exp_mode,
            provider="gemini",
            work_dir=work_dir,
        )
        return PaperVizProcessor(
            exp_config=exp_config,
            vanilla_agent=_PassthroughAgent(exp_config),
            planner_agent=_PlannerAgent(exp_config),
            visualizer_agent=_VisualizerAgent(exp_config),
            stylist_agent=_PassthroughAgent(exp_config),
            critic_agent=_PassthroughAgent(exp_config),
            retriever_agent=_PassthroughAgent(exp_config),
            polish_agent=_PassthroughAgent(exp_config),
        )

    def test_dev_planner_sets_eval_image_field(self):
        processor = self._build_processor("dev_planner")
        result = asyncio.run(
            processor.process_single_query(
                {
                    "candidate_id": 0,
                    "content": "method",
                    "visual_intent": "diagram",
                },
                do_eval=False,
            )
        )

        self.assertEqual(result["eval_image_field"], "target_diagram_desc0_base64_jpg")
        self.assertEqual(result["target_diagram_desc0_base64_jpg"], "fake-image")

    def test_demo_pipeline_skips_eval_even_when_requested(self):
        processor = self._build_processor("demo_planner_critic")

        async def fail_eval(*args, **kwargs):
            raise AssertionError("evaluation_function should not run for demo pipelines")

        processor.evaluation_function = fail_eval
        result = asyncio.run(
            processor.process_single_query(
                {
                    "candidate_id": 0,
                    "content": "method",
                    "visual_intent": "diagram",
                    "max_critic_rounds": 0,
                },
                do_eval=True,
            )
        )

        self.assertEqual(result["eval_image_field"], "target_diagram_desc0_base64_jpg")

    def test_evaluation_function_passes_exp_config_model_name(self):
        processor = self._build_processor("dev_planner")
        payload = {
            "candidate_id": 0,
            "content": "method",
            "visual_intent": "diagram",
        }

        with patch(
            "utils.paperviz_processor.get_score_for_image_referenced",
            new=AsyncMock(return_value={"status": "ok"}),
        ) as mocked_eval:
            result = asyncio.run(
                processor.evaluation_function(payload, exp_config=processor.exp_config)
            )

        self.assertEqual(result, {"status": "ok"})
        mocked_eval.assert_awaited_once_with(
            payload,
            task_name="diagram",
            model_name=processor.exp_config.model_name,
            work_dir=processor.exp_config.work_dir,
        )


if __name__ == "__main__":
    unittest.main()
