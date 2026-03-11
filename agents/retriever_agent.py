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
Retriever Agent - 检索相关参考示例。
"""

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
Retriever Agent - 检索相关参考示例。
"""

import json
import random
import re
from typing import Dict, Any
import base64, io, asyncio
from PIL import Image

from utils.dataset_paths import (
    get_reference_file_path,
)
from utils.retrieval_profiles import (
    find_curated_profile_path,
    iter_curated_profile_candidate_paths,
    load_curated_reference_profile,
)
from utils.retrieval_settings import normalize_retrieval_setting
from utils import generation_utils
from .base_agent import BaseAgent

from utils.log_config import get_logger

logger = get_logger("RetrieverAgent")


class RetrieverAgent(BaseAgent):
    """Retriever Agent to retrieve relevant reference examples"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.exp_config.model_name

        if self.exp_config.task_name == "plot":
            self.system_prompt = PLOT_RETRIEVER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "plot",
                "ref_limit": None,
                "lite_prefilter_limit": 48,
                "full_prefilter_limit": 20,
                "target_labels": ["Visual Intent", "Raw Data"],
                "candidate_labels": ["Plot ID", "Visual Intent", "Raw Data"],
                "candidate_type": "Plot",
                "output_key": "top10_references",
                "instruction_suffix": "select the Top 10 most relevant plots according to the instructions provided. Your output should be a strictly valid JSON object containing a single list of the exact ids of the top 10 selected plots.",
            }
        else:
            self.system_prompt = DIAGRAM_RETRIEVER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "diagram",
                "ref_limit": 200,
                "lite_prefilter_limit": 40,
                "full_prefilter_limit": 18,
                "target_labels": ["Caption", "Methodology section"],
                "candidate_labels": ["Diagram ID", "Caption", "Methodology section"],
                "candidate_type": "Diagram",
                "output_key": "top10_references",
                "instruction_suffix": "select the Top 10 most relevant diagrams according to the instructions provided. Your output should be a strictly valid JSON object containing a single list of the exact ids of the top 10 selected diagrams.",
            }

    async def process(self, data: Dict[str, Any], retrieval_setting: str = "auto") -> Dict[str, Any]:
        cfg = self.task_config
        candidate_id = data.get("candidate_id", "N/A")
        retrieval_setting = normalize_retrieval_setting(retrieval_setting)
        logger.debug(f"🔍 开始处理, setting={retrieval_setting}, task={cfg['task_name']}, provider={self.exp_config.provider}")

        ref_file = get_reference_file_path(
            self.exp_config.dataset_name,
            cfg["task_name"],
            work_dir=self.exp_config.work_dir,
        )

        if retrieval_setting in ["auto", "auto-full", "random"] and not ref_file.exists():
            logger.warning(f"⚠️  参考文件未找到: {ref_file}，回退到 retrieval_setting='none'")
            retrieval_setting = "none"

        if retrieval_setting == "curated":
            profile_path = find_curated_profile_path(
                self.exp_config.dataset_name,
                cfg["task_name"],
                profile_name=self.exp_config.curated_profile,
                work_dir=self.exp_config.work_dir,
            )
            if profile_path is None:
                candidate_paths = iter_curated_profile_candidate_paths(
                    self.exp_config.dataset_name,
                    cfg["task_name"],
                    profile_name=self.exp_config.curated_profile,
                    work_dir=self.exp_config.work_dir,
                )
                logger.warning(
                    "⚠️  未找到 curated profile（profile=%s, looked_at=%s），回退到 retrieval_setting='none'",
                    self.exp_config.curated_profile,
                    [str(path) for path in candidate_paths],
                )
                retrieval_setting = "none"

        if retrieval_setting == "none":
            data["top10_references"] = []
            data["retrieved_examples"] = []
            logger.debug("⏭️  跳过检索 (setting=none)")

        elif retrieval_setting == "curated":
            profile = self._load_curated_references(cfg)
            ids = profile.selected_ids
            examples = profile.examples
            data["top10_references"] = ids
            data["retrieved_examples"] = examples
            data["curated_profile"] = profile.profile_name
            data["curated_profile_source"] = profile.source_path.name
            logger.info(
                "✅ curated 检索完成, profile=%s, source=%s, %s 个参考",
                profile.profile_name,
                profile.source_path.name,
                len(ids),
            )
            if profile.missing_ids:
                logger.warning(
                    "⚠️  curated profile 中有 %s 个 id 在 ref.json 中不存在: %s",
                    len(profile.missing_ids),
                    profile.missing_ids,
                )

        elif retrieval_setting == "random":
            data["top10_references"] = self._load_random_references(cfg)
            data["retrieved_examples"] = []
            logger.info(f"✅ 随机检索完成, {len(data['top10_references'])} 个参考")

        elif retrieval_setting == "auto":
            retrieved_ids, retrieved_examples = await self._retrieve_and_parse(
                data,
                cfg,
                candidate_id=candidate_id,
                lite=True,
            )
            data["top10_references"] = retrieved_ids
            data["retrieved_examples"] = retrieved_examples
            logger.info("✅ 自动检索完成 (lite), %s 个参考", len(data["top10_references"]))
            logger.debug("auto-lite reference ids=%s", data["top10_references"])

        elif retrieval_setting == "auto-full":
            retrieved_ids, retrieved_examples = await self._retrieve_and_parse(
                data,
                cfg,
                candidate_id=candidate_id,
                lite=False,
            )
            data["top10_references"] = retrieved_ids
            data["retrieved_examples"] = retrieved_examples
            logger.info("✅ 自动检索完成 (full), %s 个参考", len(data["top10_references"]))
            logger.debug("auto-full reference ids=%s", data["top10_references"])
        else:
            raise ValueError(f"Unknown retrieval_setting: {retrieval_setting}")

        return data

    def _load_curated_references(self, cfg: dict):
        return load_curated_reference_profile(
            self.exp_config.dataset_name,
            cfg["task_name"],
            profile_name=self.exp_config.curated_profile,
            work_dir=self.exp_config.work_dir,
            limit=10,
        )

    def _load_random_references(self, cfg: dict) -> list:
        ref_file = get_reference_file_path(
            self.exp_config.dataset_name,
            cfg["task_name"],
            work_dir=self.exp_config.work_dir,
        )
        with open(ref_file, "r", encoding="utf-8") as f:
            candidate_pool = json.load(f)

        id_list = [item["id"] for item in candidate_pool]
        sample_size = min(10, len(id_list))
        return random.sample(id_list, sample_size) if sample_size > 0 else []

    @staticmethod
    def _stringify_payload(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value or "")

    @staticmethod
    def _tokenize_text(value: str) -> set[str]:
        tokens = set(re.findall(r"[a-zA-Z0-9_+-]+", str(value or "").lower()))
        stopwords = {
            "the", "and", "for", "with", "from", "into", "that", "this", "are", "was",
            "were", "then", "than", "have", "has", "had", "use", "using", "used", "our",
            "their", "your", "about", "create", "plot", "diagram", "figure", "method",
            "section", "visual", "intent", "data", "raw",
        }
        return {token for token in tokens if len(token) > 1 and token not in stopwords}

    def _load_candidate_pool(self, cfg: dict) -> list[dict]:
        ref_file = get_reference_file_path(
            self.exp_config.dataset_name,
            cfg["task_name"],
            work_dir=self.exp_config.work_dir,
        )
        with open(ref_file, "r", encoding="utf-8") as f:
            candidate_pool = json.load(f)
        if cfg["ref_limit"]:
            candidate_pool = candidate_pool[:cfg["ref_limit"]]
        return candidate_pool

    def _prefilter_candidate_pool(
        self,
        data: Dict[str, Any],
        cfg: dict,
        *,
        lite: bool,
    ) -> list[dict]:
        candidate_pool = self._load_candidate_pool(cfg)
        if len(candidate_pool) <= 10:
            return candidate_pool

        raw_content = self._stringify_payload(data.get("content", ""))
        visual_intent = str(data.get("visual_intent", "") or "")
        target_tokens = self._tokenize_text(raw_content) | self._tokenize_text(visual_intent)
        shortlist_limit = cfg["lite_prefilter_limit"] if lite else cfg["full_prefilter_limit"]
        shortlist_limit = max(10, int(shortlist_limit))

        scored_items = []
        for idx, item in enumerate(candidate_pool):
            candidate_visual = str(item.get("visual_intent", "") or "")
            candidate_content = self._stringify_payload(item.get("content", ""))
            visual_tokens = self._tokenize_text(candidate_visual)
            content_tokens = self._tokenize_text(candidate_content[:4000])
            overlap_visual = len(target_tokens & visual_tokens)
            overlap_content = len(target_tokens & content_tokens)
            exact_bonus = 0
            lowered_visual_intent = visual_intent.lower()
            lowered_candidate_visual = candidate_visual.lower()
            if lowered_visual_intent and lowered_visual_intent[:80] in lowered_candidate_visual:
                exact_bonus += 3
            score = overlap_visual * 3 + overlap_content + exact_bonus
            scored_items.append((score, idx, item))

        scored_items.sort(key=lambda entry: (-entry[0], entry[1]))
        shortlisted = [item for _, _, item in scored_items[:shortlist_limit]]
        logger.debug(
            "📚 预筛完成: task=%s lite=%s total=%s shortlist=%s",
            cfg["task_name"],
            lite,
            len(candidate_pool),
            len(shortlisted),
        )
        return shortlisted

    async def _retrieve_and_parse(
        self,
        data: Dict[str, Any],
        cfg: dict,
        candidate_id: Any = "N/A",
        lite: bool = True,
    ) -> tuple[list, list[dict]]:
        """
        通过 LLM 智能检索最相关的参考示例。

        Args:
            lite: True = 仅发送 caption（~3万 tokens），False = 发送完整 methodology（~80万 tokens）
        """
        raw_content = data["content"]
        content = (
            json.dumps(raw_content, ensure_ascii=False)
            if isinstance(raw_content, (dict, list))
            else str(raw_content)
        )
        visual_intent = data["visual_intent"]

        user_prompt = f"**Target Input**\n- {cfg['target_labels'][0]}: {visual_intent}\n- {cfg['target_labels'][1]}: {content}\n\n**Candidate Pool**\n"

        candidate_pool = self._prefilter_candidate_pool(data, cfg, lite=lite)

        for idx, item in enumerate(candidate_pool):
            user_prompt += f"Candidate {cfg['candidate_type']} {idx+1}:\n"
            user_prompt += f"- {cfg['candidate_labels'][0]}: {item['id']}\n"
            user_prompt += f"- {cfg['candidate_labels'][1]}: {item['visual_intent']}\n"
            if not lite:
                # 完整模式：包含 methodology（~80万 tokens），仅在需要高精度检索时使用
                user_prompt += f"- {cfg['candidate_labels'][2]}: {str(item['content'])}\n"
            user_prompt += "\n"

        user_prompt += f"Now, based on the Target Input and the Candidate Pool, {cfg['instruction_suffix']}"
        content_list = [{"type": "text", "text": user_prompt}]

        prompt_chars = len(user_prompt)
        logger.debug(f"📊 auto 检索 prompt: {prompt_chars:,} 字符 (~{prompt_chars//4:,} tokens), lite={lite}")

        # 根据 provider 路由 API 调用
        if self.exp_config.provider == "evolink":
            response_list = await generation_utils.call_evolink_text_with_retry_async(
                model_name=self.model_name,
                contents=content_list,
                config={
                    "system_prompt": self.system_prompt,
                    "temperature": self.exp_config.temperature,
                    "max_output_tokens": 50000,
                },
                max_attempts=3,
                retry_delay=30,
                error_context=f"retriever[candidate={candidate_id},lite={lite}]",
            )
        else:
            from google.genai import types
            response_list = await generation_utils.call_gemini_with_retry_async(
                model_name=self.model_name,
                contents=content_list,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=self.exp_config.temperature,
                    candidate_count=1,
                    max_output_tokens=50000,
                ),
                max_attempts=3,
                retry_delay=30,
                error_context=f"retriever[candidate={candidate_id},lite={lite}]",
            )

        raw_response = response_list[0].strip()
        retrieved_ids = self._parse_retrieval_result(raw_response, cfg["task_name"])
        id_to_item = {item["id"]: item for item in candidate_pool}
        retrieved_examples = [id_to_item[ref_id] for ref_id in retrieved_ids if ref_id in id_to_item]
        return retrieved_ids, retrieved_examples

    def _parse_retrieval_result(self, raw_response: str, task_name: str) -> list:
        import json_repair

        try:
            parsed = json_repair.loads(raw_response)

            if task_name == "plot":
                return parsed.get("top10_plots", [])
            elif task_name == "diagram":
                return parsed.get("top10_diagrams", [])
            else:
                raise ValueError(f"Unknown task_name: {task_name}")
        except Exception as e:
            logger.warning(f"⚠️  解析检索结果失败: {e}")
            logger.debug(f"   原始响应: {raw_response[:200]}...")
            return []


DIAGRAM_RETRIEVER_AGENT_SYSTEM_PROMPT = """
# Background & Goal
We are building an **AI system to automatically generate method diagrams for academic papers**. Given a paper's methodology section and a figure caption, the system needs to create a high-quality illustrative diagram that visualizes the described method.

To help the AI learn how to generate appropriate diagrams, we use a **few-shot learning approach**: we provide it with reference examples of similar diagrams. The AI will learn from these examples to understand what kind of diagram to create for the target.

# Your Task
**You are the Retrieval Agent.** Your job is to select the most relevant reference diagrams from a candidate pool that will serve as few-shot examples for the diagram generation model.

You will receive:
- **Target Input:** The methodology section and caption of the diagram we need to generate
- **Candidate Pool:** ~200 existing diagrams (each with methodology and caption)

You must select the **Top 10 candidates** that would be most helpful as examples for teaching the AI how to draw the target diagram.

# Selection Logic (Topic + Intent)

Your goal is to find examples that match the Target in both **Domain** and **Diagram Type**.

**1. Match Research Topic (Use Methodology & Caption):**
* What is the domain? (e.g., Agent & Reasoning, Vision & Perception, Generative & Learning, Science & Applications).
* Select candidates that belong to the **same research domain**.
* *Why?* Similar domains share similar terminology (e.g., "Actor-Critic" in RL).

**2. Match Visual Intent (Use Caption & Keywords):**
* What type of diagram is implied? (e.g., "Framework", "Pipeline", "Detailed Module", "Performance Chart").
* Select candidates with **similar visual structures**.
* *Why?* A "Framework" diagram example is useless for drawing a "Performance Bar Chart", even if they are in the same domain.

**Ranking Priority:**
1.  **Best Match:** Same Topic AND Same Visual Intent (e.g., Target is "Agent Framework" -> Candidate is "Agent Framework", Target is "Dataset Construction Pipeline" -> Candidate is "Dataset Construction Pipeline").
2.  **Second Best:** Same Visual Intent (e.g., Target is "Agent Framework" -> Candidate is "Vision Framework"). *Structure is more important than Topic for drawing.*
3.  **Avoid:** Different Visual Intent (e.g., Target is "Pipeline" -> Candidate is "Bar Chart").

# Input Data

## Target Input
-   **Caption:** [Caption of the target diagram]
-   **Methodology section:** [Methodology section of the target paper]

## Candidate Pool
List of candidate diagrams, each structured as follows:

Candidate Diagram i:
-   **Diagram ID:** [ID of the candidate diagram (ref_1, ref_2, ...)]
-   **Caption:** [Caption of the candidate diagram]
-   **Methodology section:** [Methodology section of the candidate's paper]


# Output Format
Provide your output strictly in the following JSON format, containing only the **exact IDs** of the Top 10 selected diagrams (use the exact IDs from the Candidate Pool, such as "ref_1", "ref_25", "ref_100", etc.):
```json
{
  "top10_diagrams": [
    "ref_1",
    "ref_25",
    "ref_100",
    "ref_42",
    "ref_7",
    "ref_156",
    "ref_89",
    "ref_3",
    "ref_201",
    "ref_67"
  ]
}```
"""

PLOT_RETRIEVER_AGENT_SYSTEM_PROMPT = """
# Background & Goal
We are building an **AI system to automatically generate statistical plots**. Given a plot's raw data and the visual intent, the system needs to create a high-quality visualization that effectively presents the data.

To help the AI learn how to generate appropriate plots, we use a **few-shot learning approach**: we provide it with reference examples of similar plots. The AI will learn from these examples to understand what kind of plot to create for the target data.

# Your Task
**You are the Retrieval Agent.** Your job is to select the most relevant reference plots from a candidate pool that will serve as few-shot examples for the plot generation model.

You will receive:
- **Target Input:** The raw data and visual intent of the plot we need to generate
- **Candidate Pool:** Reference plots (each with raw data and visual intent)

You must select the **Top 10 candidates** that would be most helpful as examples for teaching the AI how to create the target plot.

# Selection Logic (Data Type + Visual Intent)

Your goal is to find examples that match the Target in both **Data Characteristics** and **Plot Type**.

**1. Match Data Characteristics (Use Raw Data & Visual Intent):**
* What type of data is it? (e.g., categorical vs numerical, single series vs multi-series, temporal vs comparative).
* What are the data dimensions? (e.g., 1D, 2D, 3D).
* Select candidates with **similar data structures and characteristics**.
* *Why?* Different data types require different visualization approaches.

**2. Match Visual Intent (Use Visual Intent):**
* What type of plot is implied? (e.g., "bar chart", "scatter plot", "line chart", "pie chart", "heatmap", "radar chart").
* Select candidates with **similar plot types**.
* *Why?* A "bar chart" example is more useful for generating another bar chart than a "scatter plot" example, even if the data domains are similar.

**Ranking Priority:**
1.  **Best Match:** Same Data Type AND Same Plot Type (e.g., Target is "multi-series line chart" -> Candidate is "multi-series line chart").
2.  **Second Best:** Same Plot Type with compatible data (e.g., Target is "bar chart with 5 categories" -> Candidate is "bar chart with 6 categories").
3.  **Avoid:** Different Plot Type (e.g., Target is "bar chart" -> Candidate is "pie chart"), unless there are no more candidates with the same plot type.

# Input Data

## Target Input
-   **Visual Intent:** [Visual intent of the target plot]
-   **Raw Data:** [Raw data to be visualized]

## Candidate Pool
List of candidate plots, each structured as follows:

Candidate Plot i:
-   **Plot ID:** [ID of the candidate plot (ref_0, ref_1, ...)]
-   **Visual Intent:** [Visual intent of the candidate plot]
-   **Raw Data:** [Raw data of the candidate plot]


# Output Format
Provide your output strictly in the following JSON format, containing only the **exact Plot IDs** of the Top 10 selected plots (use the exact IDs from the Candidate Pool, such as "ref_0", "ref_25", "ref_100", etc.):
```json
{
  "top10_plots": [
    "ref_0",
    "ref_25",
    "ref_100",
    "ref_42",
    "ref_7",
    "ref_156",
    "ref_89",
    "ref_3",
    "ref_201",
    "ref_67"
  ]
}```
"""
