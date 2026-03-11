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
Polish Agent - 根据风格指南优化基准图像。
"""

import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any
from PIL import Image

from utils.dataset_paths import resolve_data_asset_path
from utils import generation_utils, image_utils
from utils.pipeline_state import get_render_options
from .base_agent import BaseAgent

from utils.log_config import get_logger

logger = get_logger("PolishAgent")


@lru_cache(maxsize=8)
def _load_style_guide_text(style_guide_path: str) -> str:
    with open(style_guide_path, "r", encoding="utf-8") as f:
        return f.read()


def _load_image_as_base64(image_path: str) -> tuple[str | None, str | None]:
    """Load an image from path and return (base64, mime_type)."""
    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
            image_b64 = base64.b64encode(img_data).decode("utf-8")
            mime_type = image_utils.detect_image_mime_from_bytes(img_data)
            return image_b64, mime_type
    except Exception as e:
        logger.error(f"❌ 加载图像失败 {image_path}: {e}")
        return None, None


class PolishAgent(BaseAgent):
    """Polish Agent to apply style guidelines to ground truth images"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_model_name = self.exp_config.image_model_name
        self.text_model_name = self.exp_config.model_name

        if self.exp_config.task_name == "plot":
            self.style_guide_filename = "neurips2025_plot_style_guide.md"
            self.suggestion_system_prompt = PLOT_SUGGESTION_SYSTEM_PROMPT
            self.system_prompt = PLOT_POLISH_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "plot",
            }
        else:
            self.style_guide_filename = "neurips2025_diagram_style_guide.md"
            self.suggestion_system_prompt = DIAGRAM_SUGGESTION_SYSTEM_PROMPT
            self.system_prompt = DIAGRAM_POLISH_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "diagram",
            }

    async def _generate_suggestions(self, gt_image_b64: str, gt_image_mime: str, style_guide: str) -> str:
        """Step 1: Generate improvement suggestions based on style guide"""
        user_prompt = f"Here is the style guide:\n{style_guide}\n\nPlease analyze the provided image against this style guide and list up to 10 specific improvement suggestions to make the image visually more appealing. If the image is already perfect, just say 'No changes needed'."

        content_list = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": gt_image_mime,
                    "data": gt_image_b64
                }
            }
        ]

        try:
            response_list = await self.call_text_api(
                contents=content_list,
                model_name=self.text_model_name,
                system_prompt=self.suggestion_system_prompt,
                temperature=1,
                max_output_tokens=50000,
                max_attempts=3,
                retry_delay=10,
            )
            return response_list[0] if response_list else ""
        except Exception as e:
            logger.error(f"❌ 生成建议时出错: {e}")
            return ""

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.task_config
        task_name = cfg["task_name"]

        gt_image_path_rel = data.get("path_to_gt_image")
        if not gt_image_path_rel:
            logger.warning("⚠️  data 中没有 GT 图像路径")
            return data

        gt_image_path = resolve_data_asset_path(
            gt_image_path_rel,
            task_name,
            dataset_name=self.exp_config.dataset_name,
            work_dir=self.exp_config.work_dir,
        )
        if gt_image_path is None:
            logger.warning(
                "⚠️  无法解析 GT 图像路径: dataset=%s task=%s path=%s",
                self.exp_config.dataset_name,
                task_name,
                gt_image_path_rel,
            )
            return data

        gt_image_b64, gt_image_mime = _load_image_as_base64(str(gt_image_path))
        if not gt_image_b64:
            logger.warning(f"⚠️  无法加载 GT 图像: {gt_image_path}")
            return data
        if not gt_image_mime:
            gt_image_mime = "image/jpeg"

        style_guide_path = self.exp_config.work_dir / "style_guides" / self.style_guide_filename
        try:
            style_guide = _load_style_guide_text(str(style_guide_path))
        except Exception as e:
            logger.error(f"❌ 加载风格指南失败 {style_guide_path}: {e}")
            return data

        logger.info(f"🎨 [第1步] 为 {task_name} 生成建议...")
        suggestions = await self._generate_suggestions(gt_image_b64, gt_image_mime, style_guide)
        output_key = f"polished_{task_name}_base64_jpg"
        output_mime_key = f"polished_{task_name}_mime_type"

        if not suggestions or "No changes needed" in suggestions:
            logger.info("✨ 该图像无需修改")
            data[f"suggestions_{task_name}"] = suggestions or "No changes needed"
            data[output_key] = gt_image_b64
            data[output_mime_key] = gt_image_mime
            return data

        if suggestions:
            data[f"suggestions_{task_name}"] = suggestions

        logger.debug("polish suggestions preview: %s...", suggestions[:200])

        # Step 2: Polish Image using suggestions
        logger.info(f"🎨 [第2步] 使用建议精修图像...")
        user_prompt = f"Please polish this image based on the following suggestions:\n\n{suggestions}\n\nPolished Image:"

        content_list = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": gt_image_mime,
                    "data": gt_image_b64
                }
            }
        ]

        try:
            # Evolink 模式需要先上传参考图
            image_urls = None
            if self.exp_config.provider == "evolink":
                logger.info("📤 上传参考图到 Evolink 文件服务...")
                ref_image_url = await generation_utils.upload_image_to_evolink(
                    gt_image_b64, media_type=gt_image_mime
                )
                image_urls = [ref_image_url]

            render_options = get_render_options(
                data,
                default_aspect_ratio="16:9",
                default_image_resolution="2K",
            )
            response_list = await self.call_image_api(
                prompt=user_prompt,
                model_name=self.image_model_name,
                contents=content_list,
                aspect_ratio=render_options.aspect_ratio,
                image_resolution=render_options.image_resolution,
                image_urls=image_urls,
                max_attempts=5,
                retry_delay=30,
            )

            if response_list and response_list[0]:
                raw_image_b64 = response_list[0]
                data[output_key] = raw_image_b64
                data[output_mime_key] = image_utils.detect_image_mime_from_b64(raw_image_b64)
            else:
                logger.warning("⚠️  模型未返回图像响应")

        except Exception as e:
            logger.error(f"❌ 图像生成过程中出错: {e}")

        return data


DIAGRAM_SUGGESTION_SYSTEM_PROMPT = """
You are a senior art director for NeurIPS 2025. Your task is to critique a diagram against a provided style guide.
Provide up to 10 concise, actionable improvement suggestions. Focus on aesthetics (color, layout, fonts, icons).
Directly list the suggestions. Do not use filler phrases like "Based on the style guide...".
If the diagram is substantially compliant, output "No changes needed".
"""

PLOT_SUGGESTION_SYSTEM_PROMPT = """
You are a senior data visualization expert for NeurIPS 2025. Your task is to critique a plot against a provided style guide.
Provide up to 10 concise, actionable improvement suggestions. Focus on aesthetics (color, layout, fonts).
Directly list the suggestions. Do not use filler phrases like "Based on the style guide...".
If the plot is substantially compliant, output "No changes needed".
"""

DIAGRAM_POLISH_AGENT_SYSTEM_PROMPT = """
## ROLE
You are a professional diagram polishing expert for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given an existing diagram image and a list of specific improvement suggestions. Your task is to generate a polished version of this diagram by applying these suggestions while preserving the semantic logic and structure of the original diagram.

## OUTPUT
Generate a polished diagram image that maintains the original content while applying the improvement suggestions.
"""

PLOT_POLISH_AGENT_SYSTEM_PROMPT = """
## ROLE
You are a professional plot polishing expert for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given an existing statistical plot image and a list of specific improvement suggestions. Your task is to generate a polished version of this plot by applying these suggestions while preserving all the data and quantitative information.

**Important Instructions:**
1. **Preserve Data:** Do NOT alter any data points, values, or quantitative information in the plot.
2. **Apply Suggestions:** Enhance the visual aesthetics according to the provided suggestions (colors, fonts, layout, etc.).
3. **Maintain Accuracy:** Ensure all numerical values and relationships remain accurate.
4. **Professional Quality:** Ensure the output meets publication standards for top-tier conferences.

## OUTPUT
Generate a polished plot image that maintains the original data while applying the improvement suggestions.
"""
