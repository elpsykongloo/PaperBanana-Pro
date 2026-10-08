import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agents.stylist_agent import DIAGRAM_STYLIST_AGENT_SYSTEM_PROMPT
from agents.visualizer_agent import VisualizerAgent
from utils.config import ExpConfig


CONFIG_YAML = """defaults:
  model_name: test-text-model
  image_model_name: test-image-model
evolink:
  api_key: dummy-key
  model_name: evolink-text-model
  image_model_name: evolink-image-model
"""


class DiagramPromptRulesTest(unittest.TestCase):
    def test_visualizer_prompt_forbids_rendering_style_words_as_text(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        work_dir = Path(temp_dir.name)
        (work_dir / "configs").mkdir(parents=True, exist_ok=True)
        (work_dir / "configs" / "model_config.yaml").write_text(CONFIG_YAML, encoding="utf-8")
        agent = VisualizerAgent(
            exp_config=ExpConfig(
                dataset_name="PaperBananaBench",
                task_name="diagram",
                exp_mode="demo_planner_critic",
                provider="evolink",
                work_dir=work_dir,
            )
        )
        mock_call = AsyncMock(return_value=["iVBORw0KGgo="])

        with patch.object(agent, "call_image_api", mock_call):
            asyncio.run(
                agent.process(
                    {
                        "candidate_id": 0,
                        "target_diagram_desc0": 'Three boxes "Encoder", "Decoder", "Loss" on a pale blue background.',
                        "max_critic_rounds": 0,
                    }
                )
            )

        prompt = mock_call.await_args.kwargs["prompt"]
        self.assertIn('"Encoder", "Decoder", "Loss"', prompt)
        self.assertIn("never render color names, style names, or layout instructions as text", prompt)

    def test_stylist_prompt_requires_concrete_attributes_and_no_new_text(self):
        self.assertIn("Concrete Attributes Only", DIAGRAM_STYLIST_AGENT_SYSTEM_PROMPT)
        self.assertIn("No New Text", DIAGRAM_STYLIST_AGENT_SYSTEM_PROMPT)
        self.assertLess(
            DIAGRAM_STYLIST_AGENT_SYSTEM_PROMPT.index("No New Text"),
            DIAGRAM_STYLIST_AGENT_SYSTEM_PROMPT.index("## OUTPUT"),
        )


if __name__ == "__main__":
    unittest.main()
