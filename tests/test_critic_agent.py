import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agents.critic_agent import CriticAgent, coerce_critic_text
from utils.config import ExpConfig
from utils.paperviz_processor import PaperVizProcessor


CONFIG_YAML = """defaults:
  model_name: test-text-model
  image_model_name: test-image-model
evolink:
  api_key: dummy-key
  model_name: evolink-text-model
  image_model_name: evolink-image-model
"""

PREVIOUS_DESCRIPTION = "previous description"


class CriticAgentCoercionTest(unittest.TestCase):
    def _build_agent(self) -> CriticAgent:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        work_dir = Path(temp_dir.name)
        (work_dir / "configs").mkdir(parents=True, exist_ok=True)
        (work_dir / "configs" / "model_config.yaml").write_text(CONFIG_YAML, encoding="utf-8")
        exp_config = ExpConfig(
            dataset_name="PaperBananaBench",
            task_name="diagram",
            exp_mode="demo_planner_critic",
            provider="evolink",
            work_dir=work_dir,
        )
        return CriticAgent(exp_config=exp_config)

    def _run(self, response: dict) -> dict:
        agent = self._build_agent()
        data = {
            "candidate_id": "c0",
            "content": "method",
            "visual_intent": "caption",
            "current_critic_round": 0,
            "target_diagram_desc0": PREVIOUS_DESCRIPTION,
        }
        with patch.object(agent, "call_text_api", AsyncMock(return_value=[json.dumps(response)])):
            return asyncio.run(agent.process(data, source="planner"))

    def test_coerce_handles_non_string_values(self):
        self.assertEqual(coerce_critic_text(None), "")
        self.assertEqual(coerce_critic_text(float("nan")), "")
        self.assertEqual(coerce_critic_text(["fix A", None, " fix B "]), "fix A\nfix B")
        self.assertEqual(coerce_critic_text({"a": 1}), '{"a": 1}')
        self.assertEqual(coerce_critic_text(3), "3")

    def test_list_suggestions_are_joined_and_revision_is_kept(self):
        result = self._run({"critic_suggestions": ["Fix typo 'lest'.", "Add arrow."], "revised_description": "new description"})

        self.assertEqual(result["target_diagram_critic_status0"], "ok")
        self.assertEqual(result["target_diagram_critic_suggestions0"], "Fix typo 'lest'.\nAdd arrow.")
        self.assertEqual(result["target_diagram_critic_desc0"], "new description")

    def test_null_fields_keep_previous_description_and_stop(self):
        result = self._run({"critic_suggestions": None, "revised_description": None})

        self.assertEqual(result["target_diagram_critic_status0"], "ok")
        self.assertEqual(result["target_diagram_critic_suggestions0"], "No changes needed.")
        self.assertEqual(result["target_diagram_critic_desc0"], PREVIOUS_DESCRIPTION)

    def test_dict_revision_is_serialized_instead_of_crashing(self):
        result = self._run({"critic_suggestions": "Fix layout.", "revised_description": {"layout": "left to right"}})

        self.assertEqual(result["target_diagram_critic_desc0"], '{"layout": "left to right"}')

    def test_processor_tolerates_non_string_suggestions_from_old_results(self):
        processor = PaperVizProcessor.__new__(PaperVizProcessor)
        processor.critic_agent = SimpleCritic()
        processor.visualizer_agent = SimpleNamespace(process=AsyncMock(side_effect=lambda data: data))
        processor._emit_status = lambda *args, **kwargs: None
        processor._emit_event = lambda *args, **kwargs: None
        data = {"target_diagram_desc0": "d", "target_diagram_desc0_base64_jpg": "x" * 200}

        # 意见为 None 时不应抛异常；这一轮没有产出新图，终稿回退到第 0 轮的图
        result = asyncio.run(processor._run_critic_iterations(data, "diagram", max_rounds=1, source="planner"))

        self.assertEqual(result["eval_image_field"], "target_diagram_desc0_base64_jpg")


class SimpleCritic:
    async def process(self, data, source="planner"):
        data["target_diagram_critic_status0"] = "ok"
        data["target_diagram_critic_suggestions0"] = None
        data["target_diagram_critic_desc0"] = data["target_diagram_desc0"]
        return data


if __name__ == "__main__":
    unittest.main()
