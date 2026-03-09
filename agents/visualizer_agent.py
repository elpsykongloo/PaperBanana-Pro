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
Visualizer Agent - 将详细描述转换为图像或代码。
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any
import base64, io, asyncio, re
from PIL import Image

from utils import image_utils
from utils.pipeline_state import PipelineState, get_render_options
from utils.plot_executor import execute_plot_code_with_details
from .base_agent import BaseAgent

from utils.log_config import get_logger

logger = get_logger("VisualizerAgent")


def _safe_preview_for_log(value, max_len: int = 20) -> str:
    """
    Build an ASCII-safe preview string for logging.
    This avoids Windows stdout failures caused by invalid characters.
    """
    try:
        if isinstance(value, (bytes, bytearray)):
            return ascii(bytes(value[:max_len]))
        if isinstance(value, str):
            return ascii(value[:max_len])
        return ascii(str(value)[:max_len])
    except Exception:
        return "<unprintable>"


class VisualizerAgent(BaseAgent):
    """Visualizer Agent to generate images based on user queries"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Task-specific configurations
        if "plot" in self.exp_config.task_name:
            self.model_name = self.exp_config.model_name
            self.system_prompt = PLOT_VISUALIZER_AGENT_SYSTEM_PROMPT
            self.process_executor = ProcessPoolExecutor(
                max_workers=min(os.cpu_count() or 4, 8)
            )
            self.task_config = {
                "task_name": "plot",
                "use_image_generation": False,
                "prompt_template": "Use python matplotlib to generate a statistical plot based on the following detailed description: {desc}\n Only provide the code without any explanations. Code:",
                "max_output_tokens": 50000,
            }
        else:
            self.model_name = self.exp_config.image_model_name
            self.system_prompt = DIAGRAM_VISUALIZER_AGENT_SYSTEM_PROMPT
            self.process_executor = None
            self.task_config = {
                "task_name": "diagram",
                "use_image_generation": True,
                "prompt_template": "Render an image based on the following detailed description: {desc}\n Note that do not include figure titles in the image. Diagram: ",
                "max_output_tokens": 50000,
            }

    def shutdown(self):
        """Explicitly shut down the process pool executor."""
        if self.process_executor:
            self.process_executor.shutdown(wait=False)
            self.process_executor = None

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.task_config
        state = PipelineState(data, cfg["task_name"])
        task_name = state.task_name
        candidate_id = data.get("candidate_id", "N/A")
        logger.debug(f"🖼️ 开始处理, task={task_name}, provider={self.exp_config.provider}, model={self.model_name}, 图像生成={cfg['use_image_generation']}")

        desc_keys_to_process = []
        max_critic_rounds = int(data.get("max_critic_rounds", self.exp_config.max_critic_rounds))
        render_options = get_render_options(data)
        for key in [
            state.planner_desc_key(),
            state.stylist_desc_key(),
        ]:
            if key in data and state.image_key(key) not in data:
                desc_keys_to_process.append(key)

        for round_idx in range(max_critic_rounds):
            key = state.critic_desc_key(round_idx)
            if key in data and state.image_key(key) not in data:
                critic_suggestions_key = state.critic_suggestions_key(round_idx)
                critic_suggestions = data.get(critic_suggestions_key, "")

                if critic_suggestions.strip() == "No changes needed." and round_idx > 0:
                    prev_desc_key = state.critic_desc_key(round_idx - 1)
                    prev_base64_key = state.image_key(prev_desc_key)
                    prev_mime_key = state.mime_key(prev_desc_key)
                    if prev_base64_key in data:
                        data[state.image_key(key)] = data[prev_base64_key]
                        if prev_mime_key in data:
                            data[state.mime_key(key)] = data[prev_mime_key]
                        logger.debug(f"🔄 复用第 {round_idx - 1} 轮的 base64 数据到 {key}")
                        continue

                desc_keys_to_process.append(key)

        if not cfg["use_image_generation"]:
            loop = asyncio.get_running_loop()

        logger.debug(f"📋 待处理 desc_keys: {desc_keys_to_process}")

        for desc_key in desc_keys_to_process:
            prompt_text = cfg["prompt_template"].format(desc=data[desc_key])
            content_list = [{"type": "text", "text": prompt_text}]
            logger.debug(f"🔧 处理 {desc_key}, prompt 长度={len(prompt_text)}")

            if cfg["use_image_generation"]:
                response_list = await self.call_image_api(
                    prompt=prompt_text,
                    contents=content_list,
                    aspect_ratio=render_options.aspect_ratio,
                    image_resolution=render_options.image_resolution,
                    max_output_tokens=cfg["max_output_tokens"],
                    max_attempts=5,
                    retry_delay=30,
                    error_context=f"visualizer-image[candidate={candidate_id},key={desc_key}]",
                )
            else:
                response_list = await self.call_text_api(
                    contents=content_list,
                    model_name=self.exp_config.model_name,
                    max_output_tokens=cfg["max_output_tokens"],
                    max_attempts=5,
                    retry_delay=30,
                    error_context=f"visualizer-code[candidate={candidate_id},key={desc_key}]",
                )

            if not response_list or not response_list[0]:
                logger.warning(f"⚠️  {desc_key}: API 返回空响应")
                continue

            resp0 = response_list[0]
            preview = _safe_preview_for_log(resp0, max_len=20)
            logger.debug(
                f"🔧 {desc_key}: API 响应长度={len(resp0)}, 值前20字={preview}..."
            )

            # Post-process based on task type
            if cfg["use_image_generation"]:
                raw_image_b64 = response_list[0]
                if raw_image_b64 and raw_image_b64 != "Error":
                    mime_type = image_utils.detect_image_mime_from_b64(raw_image_b64)
                    data[state.image_key(desc_key)] = raw_image_b64
                    data[state.mime_key(desc_key)] = mime_type
                    logger.info(
                        f"✅ {desc_key}_base64_jpg 已生成, "
                        f"mime={mime_type}, 大小={len(raw_image_b64)}"
                    )
                else:
                    logger.error(f"❌ {desc_key}: 图像输出为空")
            else:
                raw_code = response_list[0]

                if self.process_executor is None:
                    self.process_executor = ProcessPoolExecutor(
                        max_workers=min(os.cpu_count() or 4, 8)
                    )

                exec_result = await loop.run_in_executor(
                    self.process_executor, execute_plot_code_with_details, raw_code
                )
                base64_jpg = exec_result.get("base64_jpg")
                data[state.code_key(desc_key)] = raw_code
                data[state.plot_exec_key(desc_key)] = exec_result

                if base64_jpg:
                    data[state.image_key(desc_key)] = base64_jpg
                    data[state.mime_key(desc_key)] = "image/jpeg"

        return data


DIAGRAM_VISUALIZER_AGENT_SYSTEM_PROMPT = """You are an expert scientific diagram illustrator. Generate high-quality scientific diagrams based on user requests."""

PLOT_VISUALIZER_AGENT_SYSTEM_PROMPT = """You are an expert statistical plot illustrator. Write code to generate high-quality statistical plots based on user requests."""
