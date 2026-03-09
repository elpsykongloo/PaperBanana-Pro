# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Vanilla Agent - 直接根据方法章节和图注生成图像或代码。
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any
import base64, io, asyncio
from PIL import Image
import json

from utils import image_utils
from utils.pipeline_state import get_render_options
from utils.plot_executor import execute_plot_code_with_details
from .base_agent import BaseAgent


class VanillaAgent(BaseAgent):
    """Vanilla Agent to generate images based on user queries"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "plot" in self.exp_config.task_name:
            self.model_name = self.exp_config.model_name
            self.system_prompt = PLOT_VANILLA_AGENT_SYSTEM_PROMPT
            self.process_executor = ProcessPoolExecutor(
                max_workers=min(os.cpu_count() or 4, 8)
            )
            self.task_config = {
                "task_name": "plot",
                "use_image_generation": False,
                "content_label": "Plot Raw Data",
                "visual_intent_label": "Visual Intent of the Desired Plot",
            }
        else:
            self.model_name = self.exp_config.image_model_name
            self.system_prompt = DIAGRAM_VANILLA_AGENT_SYSTEM_PROMPT
            self.process_executor = None
            self.task_config = {
                "task_name": "diagram",
                "use_image_generation": True,
                "content_label": "Method Section",
                "visual_intent_label": "Diagram Caption",
            }

    def shutdown(self):
        """Explicitly shut down the process pool executor."""
        if self.process_executor:
            self.process_executor.shutdown(wait=False)
            self.process_executor = None

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.task_config
        render_options = get_render_options(
            data,
            default_aspect_ratio="1:1",
            default_image_resolution="1K",
        )

        raw_content = data["content"]
        content = json.dumps(raw_content) if isinstance(raw_content, (dict, list)) else raw_content
        visual_intent = data["visual_intent"]

        prompt_text = f"**{cfg['content_label']}**: {content}\n**{cfg['visual_intent_label']}**: {visual_intent}\n"
        if cfg['task_name'] == 'diagram':
            prompt_text += "Note that do not include figure titles in the image."

        if cfg["use_image_generation"]:
            prompt_text += "**Generated Diagram**: "
        else:
            prompt_text += "\nUse python matplotlib to generate a statistical plot based on the above information. Only provide the code without any explanations. Code:"

        content_list = [{"type": "text", "text": prompt_text}]

        if cfg["use_image_generation"]:
            response_list = await self.call_image_api(
                prompt=prompt_text,
                contents=content_list,
                aspect_ratio=render_options.aspect_ratio,
                image_resolution=render_options.image_resolution,
                max_attempts=5,
                retry_delay=30,
            )
        else:
            response_list = await self.call_text_api(
                contents=content_list,
                max_output_tokens=50000,
                max_attempts=5,
                retry_delay=30,
            )

        output_key = f"vanilla_{cfg['task_name']}_base64_jpg"
        mime_key = f"vanilla_{cfg['task_name']}_mime_type"
        if cfg["use_image_generation"]:
            raw_image_b64 = response_list[0] if response_list else None
            if raw_image_b64 and raw_image_b64 != "Error":
                data[output_key] = raw_image_b64
                data[mime_key] = image_utils.detect_image_mime_from_b64(raw_image_b64)
        else:
            if response_list and response_list[0]:
                raw_code = response_list[0]
                loop = asyncio.get_running_loop()
                exec_result = await loop.run_in_executor(
                    self.process_executor,
                    execute_plot_code_with_details,
                    raw_code,
                    100,
                )
                base64_jpg = exec_result.get("base64_jpg")
                data["vanilla_plot_code"] = raw_code
                data["vanilla_plot_exec"] = exec_result
                if base64_jpg:
                    data[output_key] = base64_jpg
                    data[mime_key] = "image/jpeg"

        return data


DIAGRAM_VANILLA_AGENT_SYSTEM_PROMPT = """
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You will be provided with a "Method Section" and a "Diagram Caption". Your task is to generate a high-quality scientific diagram that effectively illustrates the method described in the text, as the caption requires, and adhering strictly to modern academic visualization standards.

**CRITICAL INSTRUCTION ON CAPTION:**
The "Diagram Caption" is provided solely to describe the visual content and logic you need to draw. **DO NOT render, write, or include the caption text itself (e.g., "Figure 1: ...") inside the generated image.**

## INPUT DATA
-   **Method Section**: [Content of method section]
-   **Diagram Caption**: [Diagram caption]
## OUTPUT
Generate a single, high-resolution image that visually explains the method and aligns well with the caption.
"""

PLOT_VANILLA_AGENT_SYSTEM_PROMPT = """
## ROLE
You are an expert statistical plot illustrator for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You will be provided with "Plot Raw Data" and a "Visual Intent of the Desired Plot". Your task is to write matplotlib code to generate a high-quality statistical plot that effectively visualizes the data according to the visual intent, adhering strictly to modern academic visualization standards.

## INPUT DATA
-   **Plot Raw Data**: [Raw data to be visualized]
-   **Visual Intent of the Desired Plot**: [Description of what the plot should convey]

## OUTPUT
Write Python matplotlib code to generate the plot. Only provide the code without any explanations.
"""
