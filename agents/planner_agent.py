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
Planner Agent - 根据方法章节生成图表的详细描述。
"""

from functools import lru_cache
import json
from pathlib import Path
from typing import Dict, Any
import base64, io, asyncio
from PIL import Image

from utils.dataset_paths import get_reference_file_path, resolve_data_asset_path
from utils import generation_utils, image_utils
from utils.pipeline_state import PipelineState
from .base_agent import BaseAgent

from utils.log_config import get_logger

logger = get_logger("PlannerAgent")

# diagram 参考样例的方法正文只保留前 2500 字符。小样本实测（16 条）：Planner 输入 token 减少约 60%，
# 成对盲评质量持平；参考样例的价值主要来自参考图与图注，正文全文只增加成本。
DIAGRAM_EXAMPLE_CONTENT_CHAR_LIMIT = 2500
# 参考图缩到长边 1024 像素的 JPEG 再发送（与实验口径一致；原图平均 0.4 MB，10 张会让每次请求多传数 MB）
REFERENCE_IMAGE_MAX_SIDE = 1024


def _truncate_example_content(text: str, limit: int | None) -> str:
    if not limit or len(text) <= limit:
        return text
    return text[:limit] + " …"


@lru_cache(maxsize=8)
def _load_reference_items(
    work_dir: str,
    dataset_name: str,
    task_name: str,
) -> dict[str, dict[str, Any]]:
    ref_file = get_reference_file_path(
        dataset_name,
        task_name,
        work_dir=Path(work_dir),
    )
    if not ref_file.exists():
        return {}
    with open(ref_file, "r", encoding="utf-8") as f:
        candidate_pool = json.load(f)
    return {item["id"]: item for item in candidate_pool}


class PlannerAgent(BaseAgent):
    """Planner Agent to generate images based on user queries"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.exp_config.model_name

        # Task-specific configurations
        if "plot" in self.exp_config.task_name:
            self.system_prompt = PLOT_PLANNER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "plot",
                "content_label": "Plot Raw Data",
                "visual_intent_label": "Visual Intent of the Desired Plot",
                # plot 样例正文是原始数据，截断会破坏数据结构，保持全文
                "example_content_char_limit": None,
            }
        else:
            self.system_prompt = DIAGRAM_PLANNER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "diagram",
                "content_label": "Methodology Section",
                "visual_intent_label": "Diagram Caption",
                "example_content_char_limit": DIAGRAM_EXAMPLE_CONTENT_CHAR_LIMIT,
            }

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.task_config
        state = PipelineState(data, cfg["task_name"])
        logger.debug(f"📝 开始处理, task={cfg['task_name']}, provider={self.exp_config.provider}, model={self.model_name}")
        candidate_id = data.get("candidate_id", "N/A")

        raw_content = data["content"]
        content = json.dumps(raw_content) if isinstance(raw_content, (dict, list)) else raw_content
        description = data["visual_intent"]

        content_list = []

        examples = data.get("retrieved_examples", [])
        if not examples:
            retrieved_ids = data.get("top10_references", [])
            id_to_item = _load_reference_items(
                str(self.exp_config.work_dir),
                self.exp_config.dataset_name,
                cfg["task_name"],
            )
            examples = [id_to_item[ref_id] for ref_id in retrieved_ids if ref_id in id_to_item]

        for idx, item in enumerate(examples):
            item_content = item["content"]
            if isinstance(item_content, (dict, list)):
                item_content = json.dumps(item_content)
            item_content = _truncate_example_content(item_content, cfg["example_content_char_limit"])

            image_path = resolve_data_asset_path(
                item.get("path_to_gt_image"),
                cfg["task_name"],
                dataset_name=self.exp_config.dataset_name,
                work_dir=self.exp_config.work_dir,
            )
            if image_path is None:
                logger.warning(
                    "⚠️  跳过缺失参考图像: dataset=%s task=%s path=%s",
                    self.exp_config.dataset_name,
                    cfg["task_name"],
                    item.get("path_to_gt_image"),
                )
                continue

            user_prompt = f"Example {idx+1}:\n"
            user_prompt += f"{cfg['content_label']}: {item_content}\n"
            user_prompt += f"{cfg['visual_intent_label']}: {item['visual_intent']}\nReference {cfg['task_name'].capitalize()}: "
            content_list.append({"type": "text", "text": user_prompt})
            ref_image_base64 = image_utils.load_image_as_jpeg_base64(str(image_path), REFERENCE_IMAGE_MAX_SIDE)
            # 使用 source 嵌套格式：各 provider 的转换函数都支持（Gemini 分支此前会丢弃 image_base64 直接格式）
            content_list.append({
                "type": "image",
                "source": {"type": "base64", "data": ref_image_base64, "media_type": "image/jpeg"},
            })

        user_prompt = ""
        user_prompt += f"Now, based on the following {cfg['content_label'].lower()} and {cfg['visual_intent_label'].lower()}, provide a detailed description for the figure to be generated.\n"
        user_prompt += f"{cfg['content_label']}: {content}\n{cfg['visual_intent_label']}: {description}\n"
        user_prompt += "Detailed description of the target figure to be generated"
        if cfg["task_name"] == "diagram":
            user_prompt += " (do not include figure titles)"
        user_prompt += ":"

        content_list.append({"type": "text", "text": user_prompt})
        logger.debug(f"📋 content_list 长度={len(content_list)}, 示例数={len(examples)}")

        # 调用文本生成 API（自动路由到对应 provider）
        response_list = await self.call_text_api(
            contents=content_list,
            max_output_tokens=50000,
            error_context=f"planner[candidate={candidate_id}]",
        )

        for idx, response in enumerate(response_list):
            if idx == 0:
                data[state.planner_desc_key()] = response.strip()
            else:
                data[f"target_{cfg['task_name']}_desc{idx}"] = response.strip()

        logger.info(f"✅ 完成, 生成 {len(response_list)} 个描述, desc0 长度={len(response_list[0]) if response_list else 0}")
        return data
DIAGRAM_PLANNER_AGENT_SYSTEM_PROMPT = """
I am working on a task: given the 'Methodology' section of a paper, and the caption of the desired figure, automatically generate a corresponding illustrative diagram. I will input the text of the 'Methodology' section, the figure caption, and your output should be a detailed description of an illustrative figure that effectively represents the methods described in the text.

To help you understand the task better, and grasp the principles for generating such figures, I will also provide you with several examples. You should learn from these examples to provide your figure description.

** IMPORTANT: **
Your description should be as detailed as possible. Semantically, clearly describe each element and their connections. Formally, include various details such as background style (typically pure white or very light pastel), colors, line thickness, icon styles, etc. Remember: vague or unclear specifications will only make the generated figure worse, not better.
"""

PLOT_PLANNER_AGENT_SYSTEM_PROMPT = """
I am working on a task: given the raw data (typically in tabular or json format) and a visual intent of the desired plot, automatically generate a corresponding statistical plot that are both accurate and aesthetically pleasing. I will input the raw data and the plot visual intent, and your output should be a detailed description of an illustrative plot that effectively represents the data.  Note that your description should include all the raw data points to be plotted.

To help you understand the task better, and grasp the principles for generating such plots, I will also provide you with several examples. You should learn from these examples to provide your plot description.

** IMPORTANT: **
Your description should be as detailed as possible. For content, explain the precise mapping of variables to visual channels (x, y, hue) and explicitly enumerate every raw data point's coordinate to be drawn to ensure accuracy. For presentation, specify the exact aesthetic parameters, including specific HEX color codes, font sizes for all labels, line widths, marker dimensions, legend placement, and grid styles. You should learn from the examples' content presentation and aesthetic design (e.g., color schemes).
"""
