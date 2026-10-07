import asyncio
import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

from agents.planner_agent import DIAGRAM_EXAMPLE_CONTENT_CHAR_LIMIT, REFERENCE_IMAGE_MAX_SIDE, PlannerAgent
from utils.config import ExpConfig


CONFIG_YAML = """defaults:
  model_name: test-text-model
  image_model_name: test-image-model
evolink:
  api_key: dummy-key
  model_name: evolink-text-model
  image_model_name: evolink-image-model
"""


class PlannerAgentTest(unittest.TestCase):
    def _build_agent(self, task_name: str) -> PlannerAgent:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        work_dir = Path(temp_dir.name)
        (work_dir / "configs").mkdir(parents=True, exist_ok=True)
        (work_dir / "configs" / "model_config.yaml").write_text(CONFIG_YAML, encoding="utf-8")
        task_dir = work_dir / "data" / "PaperBananaBench" / task_name / "images"
        task_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2000, 1000), (240, 248, 255)).save(task_dir / "ref.png")
        exp_config = ExpConfig(
            dataset_name="PaperBananaBench",
            task_name=task_name,
            exp_mode="demo_full",
            provider="evolink",
            work_dir=work_dir,
        )
        return PlannerAgent(exp_config=exp_config)

    def _run_and_capture_contents(self, agent: PlannerAgent, data: dict) -> list[dict]:
        mock_call = AsyncMock(return_value=["planned description"])
        with patch.object(agent, "call_text_api", mock_call):
            result = asyncio.run(agent.process(data))
        self.assertEqual(result["target_" + agent.task_config["task_name"] + "_desc0"], "planned description")
        return mock_call.await_args.kwargs["contents"]

    def _run_and_capture_texts(self, agent: PlannerAgent, data: dict) -> list[str]:
        contents = self._run_and_capture_contents(agent, data)
        return [part["text"] for part in contents if part["type"] == "text"]

    def test_reference_image_is_sent_as_downscaled_jpeg_in_source_format(self):
        agent = self._build_agent("diagram")

        contents = self._run_and_capture_contents(
            agent,
            {
                "candidate_id": 0,
                "content": "target method",
                "visual_intent": "Overview of the target method.",
                "retrieved_examples": [
                    {
                        "id": "ref_1",
                        "content": "reference method",
                        "visual_intent": "Overview of a reference method.",
                        "path_to_gt_image": "images/ref.png",
                    }
                ],
            },
        )

        image_parts = [part for part in contents if part["type"] == "image"]
        self.assertEqual(len(image_parts), 1)
        source = image_parts[0]["source"]
        self.assertEqual(source["type"], "base64")
        self.assertEqual(source["media_type"], "image/jpeg")
        with Image.open(io.BytesIO(base64.b64decode(source["data"]))) as img:
            self.assertEqual(img.format, "JPEG")
            self.assertEqual(img.size, (REFERENCE_IMAGE_MAX_SIDE, REFERENCE_IMAGE_MAX_SIDE // 2))

    def test_diagram_example_methodology_is_truncated_but_target_is_not(self):
        agent = self._build_agent("diagram")
        long_example = "e" * (DIAGRAM_EXAMPLE_CONTENT_CHAR_LIMIT + 500)
        long_target = "t" * (DIAGRAM_EXAMPLE_CONTENT_CHAR_LIMIT + 500)

        texts = self._run_and_capture_texts(
            agent,
            {
                "candidate_id": 0,
                "content": long_target,
                "visual_intent": "Overview of the target method.",
                "retrieved_examples": [
                    {
                        "id": "ref_1",
                        "content": long_example,
                        "visual_intent": "Overview of a reference method.",
                        "path_to_gt_image": "images/ref.png",
                    }
                ],
            },
        )

        example_text, target_text = texts
        self.assertIn("e" * DIAGRAM_EXAMPLE_CONTENT_CHAR_LIMIT + " …", example_text)
        self.assertNotIn("e" * (DIAGRAM_EXAMPLE_CONTENT_CHAR_LIMIT + 1), example_text)
        self.assertIn("Diagram Caption: Overview of a reference method.", example_text)
        self.assertIn(long_target, target_text)

    def test_plot_example_raw_data_is_kept_in_full(self):
        agent = self._build_agent("plot")
        raw_rows = [{"step": i, "score": i * 0.5} for i in range(400)]
        self.assertGreater(len(json.dumps(raw_rows)), DIAGRAM_EXAMPLE_CONTENT_CHAR_LIMIT)

        texts = self._run_and_capture_texts(
            agent,
            {
                "candidate_id": 0,
                "content": '[{"step": 1, "score": 62.1}]',
                "visual_intent": "Line plot of score over step.",
                "retrieved_examples": [
                    {
                        "id": "plot_ref_1",
                        "content": raw_rows,
                        "visual_intent": "Line plot of a reference metric.",
                        "path_to_gt_image": "images/ref.png",
                    }
                ],
            },
        )

        self.assertIn(json.dumps(raw_rows), texts[0])


if __name__ == "__main__":
    unittest.main()
