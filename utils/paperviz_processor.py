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
Processing pipeline of PaperBanana
"""

import asyncio
import traceback
from typing import List, Dict, Any, AsyncGenerator, Callable, Optional

from tqdm.asyncio import tqdm

from agents.vanilla_agent import VanillaAgent
from agents.planner_agent import PlannerAgent
from agents.visualizer_agent import VisualizerAgent
from agents.stylist_agent import StylistAgent
from agents.critic_agent import CriticAgent
from agents.retriever_agent import RetrieverAgent
from agents.polish_agent import PolishAgent

from .config import ExpConfig
from .eval_toolkits import get_score_for_image_referenced
from .pipeline_registry import PipelineSpec, get_pipeline_spec
from .pipeline_state import PipelineState

from utils.log_config import get_logger

logger = get_logger("PaperVizProcessor")


class PaperVizProcessor:
    """Main class for multimodal document processor"""

    def __init__(
        self,
        exp_config: ExpConfig,
        vanilla_agent: VanillaAgent,
        planner_agent: PlannerAgent,
        visualizer_agent: VisualizerAgent,
        stylist_agent: StylistAgent,
        critic_agent: CriticAgent,
        retriever_agent: RetrieverAgent,
        polish_agent: PolishAgent,
    ):
        self.exp_config = exp_config
        self.vanilla_agent = vanilla_agent
        self.planner_agent = planner_agent
        self.visualizer_agent = visualizer_agent
        self.stylist_agent = stylist_agent
        self.critic_agent = critic_agent
        self.retriever_agent = retriever_agent
        self.polish_agent = polish_agent

    @staticmethod
    def _format_stage_sequence(spec: PipelineSpec, max_rounds: int) -> str:
        labels = {
            "vanilla": "vanilla",
            "retriever": "retriever",
            "planner": "planner",
            "stylist": "stylist",
            "visualizer": "visualizer",
            "critic": f"critic×{max_rounds}",
            "polish": "polish",
        }
        return " → ".join(labels.get(stage, stage) for stage in spec.stages)

    @staticmethod
    def _resolve_render_desc_key(state: PipelineState) -> str | None:
        stylist_key = state.stylist_desc_key()
        if stylist_key in state.data:
            return stylist_key
        planner_key = state.planner_desc_key()
        if planner_key in state.data:
            return planner_key
        return None

    def _set_eval_image_field(
        self,
        data: Dict[str, Any],
        task_name: str,
        source: str | None,
    ) -> Dict[str, Any]:
        state = PipelineState(data, task_name)
        if source == "planner":
            state.eval_image_field = state.image_key(state.planner_desc_key())
        elif source == "stylist":
            state.eval_image_field = state.image_key(state.stylist_desc_key())
        elif source == "vanilla":
            state.eval_image_field = f"vanilla_{state.task_name}_base64_jpg"
        elif source == "polish":
            state.eval_image_field = f"polished_{state.task_name}_base64_jpg"
        return data

    async def _execute_pipeline_stage(
        self,
        stage_name: str,
        data: Dict[str, Any],
        *,
        task_name: str,
        retrieval_setting: str,
        max_rounds: int,
        critic_source: str | None,
        candidate_id: Any,
        status_callback: Optional[Callable[[str], None]],
    ) -> Dict[str, Any]:
        state = PipelineState(data, task_name)

        if stage_name == "vanilla":
            logger.debug(f"[{candidate_id}] 流水线阶段: vanilla_agent")
            self._emit_status(status_callback, candidate_id, "vanilla 生成中")
            return await self.vanilla_agent.process(data)

        if stage_name == "retriever":
            self._emit_status(status_callback, candidate_id, "retriever 检索中")
            data = await self.retriever_agent.process(data, retrieval_setting=retrieval_setting)
            logger.debug(f"[{candidate_id}] ✅ retriever 完成")
            return data

        if stage_name == "planner":
            self._emit_status(status_callback, candidate_id, "planner 规划中")
            data = await self.planner_agent.process(data)
            state = PipelineState(data, task_name)
            logger.debug(f"[{candidate_id}] ✅ planner 完成, desc0 长度={len(data.get(state.planner_desc_key(), ''))}")
            return data

        if stage_name == "stylist":
            self._emit_status(status_callback, candidate_id, "stylist 风格优化中")
            data = await self.stylist_agent.process(data)
            logger.debug(f"[{candidate_id}] ✅ stylist 完成")
            return data

        if stage_name == "visualizer":
            self._emit_status(status_callback, candidate_id, "visualizer 生图中")
            data = await self.visualizer_agent.process(data)
            state = PipelineState(data, task_name)
            render_desc_key = self._resolve_render_desc_key(state)
            has_img = False
            if render_desc_key:
                has_img = bool(data.get(state.image_key(render_desc_key)))
            logger.debug(f"[{candidate_id}] ✅ visualizer 完成, 图像生成={'成功' if has_img else '失败'}")
            return data

        if stage_name == "critic":
            data = await self._run_critic_iterations(
                data,
                task_name,
                max_rounds=max_rounds,
                source=critic_source or "planner",
                status_callback=status_callback,
                candidate_id=candidate_id,
            )
            logger.debug(f"[{candidate_id}] ✅ critic 迭代完成, eval_image_field={data.get('eval_image_field')}")
            return data

        if stage_name == "polish":
            self._emit_status(status_callback, candidate_id, "polish 精修中")
            return await self.polish_agent.process(data)

        raise ValueError(f"Unsupported pipeline stage: {stage_name}")

    @staticmethod
    def _emit_status(
        status_callback: Optional[Callable[[str], None]],
        candidate_id: Any,
        stage: str,
    ) -> None:
        if status_callback is None:
            return
        try:
            status_callback(f"候选 {candidate_id}: {stage}")
        except Exception as err:
            logger.warning(f"⚠️  status_callback 失败: {err}")

    async def _run_critic_iterations(
        self,
        data: Dict[str, Any],
        task_name: str,
        max_rounds: int = 3,
        source: str = "stylist",
        status_callback: Optional[Callable[[str], None]] = None,
        candidate_id: Any = "N/A",
    ) -> Dict[str, Any]:
        """
        Run multi-round critic iteration (up to max_rounds).
        Returns the data with critic suggestions and updated eval_image_field.
        
        Args:
            data: Input data dictionary
            task_name: Name of the task (e.g., "diagram", "plot")
            max_rounds: Maximum number of critic iterations
            source: Source of the input for round 0 critique ("stylist" or "planner")
        """
        state = PipelineState(data, task_name)
        if source == "planner":
            current_best_image_key = state.image_key(state.planner_desc_key())
        else:
            current_best_image_key = state.image_key(state.stylist_desc_key())
            
        for round_idx in range(max_rounds):
            self._emit_status(
                status_callback,
                candidate_id,
                f"critic 第 {round_idx + 1}/{max_rounds} 轮",
            )
            state.current_critic_round = round_idx
            data = await self.critic_agent.process(data, source=source)
            state = PipelineState(data, task_name)
             
            critic_suggestions_key = state.critic_suggestions_key(round_idx)
            critic_status_key = state.critic_status_key(round_idx)
            critic_status = data.get(critic_status_key, "ok")
            critic_suggestions = data.get(critic_suggestions_key, "")

            if critic_status == "parse_error":
                logger.warning(f"⚠️  Critic 第 {round_idx} 轮解析失败，停止继续迭代并保留上一版本")
                self._emit_status(
                    status_callback,
                    candidate_id,
                    f"critic 第 {round_idx + 1}/{max_rounds} 轮解析失败，保留上一版本",
                )
                break
             
            if critic_suggestions.strip() == "No changes needed.":
                logger.info(f"✅ Critic 第 {round_idx} 轮无需修改，停止迭代")
                self._emit_status(
                    status_callback,
                    candidate_id,
                    f"critic 第 {round_idx + 1}/{max_rounds} 轮无需修改，提前结束",
                )
                break
            
            data = await self.visualizer_agent.process(data)
            state = PipelineState(data, task_name)
            
            # Check if visualization validation succeeded
            new_image_key = state.image_key(state.critic_desc_key(round_idx))
            if new_image_key in data and data[new_image_key]:
                current_best_image_key = new_image_key
                logger.info(f"✅ Critic 第 {round_idx} 轮完成，可视化成功")
                self._emit_status(
                    status_callback,
                    candidate_id,
                    f"critic 第 {round_idx + 1}/{max_rounds} 轮可视化成功",
                )
            else:
                logger.warning(f"⚠️  Critic 第 {round_idx} 轮可视化失败（无有效图像），回滚到: {current_best_image_key}")
                self._emit_status(
                    status_callback,
                    candidate_id,
                    f"critic 第 {round_idx + 1}/{max_rounds} 轮可视化失败，回退上一版本",
                )
                break
         
        if current_best_image_key in data and data.get(current_best_image_key):
            state.eval_image_field = current_best_image_key
        else:
            logger.warning(f"⚠️  最终评测图像缺失: {current_best_image_key}")
            state.eval_image_field = None
        return data

    async def process_single_query(
        self,
        data: Dict[str, Any],
        do_eval=True,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Complete processing pipeline for a single query
        """
        candidate_id = data.get('candidate_id', 'N/A')
        exp_mode = self.exp_config.exp_mode
        task_name = self.exp_config.task_name.lower()
        retrieval_setting = self.exp_config.retrieval_setting
        max_rounds = int(data.get("max_critic_rounds", self.exp_config.max_critic_rounds))
        spec = get_pipeline_spec(exp_mode)
        logger.debug(f"\n── process_single_query 开始 ── candidate={candidate_id}")
        logger.debug(f"   exp_mode={exp_mode}, task={task_name}, retrieval={retrieval_setting}, provider={self.exp_config.provider}")
        logger.debug(f"[{candidate_id}] 流水线: {self._format_stage_sequence(spec, max_rounds)}")
        self._emit_status(status_callback, candidate_id, "开始处理")
        effective_do_eval = do_eval and not spec.disable_eval

        for stage_name in spec.stages:
            data = await self._execute_pipeline_stage(
                stage_name,
                data,
                task_name=task_name,
                retrieval_setting=retrieval_setting,
                max_rounds=max_rounds,
                critic_source=spec.critic_source,
                candidate_id=candidate_id,
                status_callback=status_callback,
            )

        if spec.eval_image_source:
            data = self._set_eval_image_field(data, task_name, spec.eval_image_source)

        logger.debug(f"[{candidate_id}] ── process_single_query 完成 ──")
        self._emit_status(status_callback, candidate_id, "候选流程完成")

        if effective_do_eval:
            self._emit_status(status_callback, candidate_id, "评测中")
            data_with_eval = await self.evaluation_function(data, exp_config=self.exp_config)
            self._emit_status(status_callback, candidate_id, "评测完成")
            return data_with_eval
        else:
            return data

    async def process_queries_batch(
        self,
        data_list: List[Dict[str, Any]],
        max_concurrent: int = 50,
        do_eval: bool = True,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Batch process queries with concurrency support
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(doc):
            candidate_id = doc.get("candidate_id", "N/A")
            self._emit_status(status_callback, candidate_id, "等待并发槽位")
            async with semaphore:
                self._emit_status(status_callback, candidate_id, "进入并发执行")
                try:
                    result = await self.process_single_query(
                        doc,
                        do_eval=do_eval,
                        status_callback=status_callback,
                    )
                    if isinstance(result, dict):
                        result.setdefault("candidate_id", candidate_id)
                        result.setdefault("status", "ok")
                    return result
                except Exception as task_err:
                    err_summary = f"{type(task_err).__name__}: {task_err}"
                    err_detail = traceback.format_exc()
                    self._emit_status(
                        status_callback,
                        candidate_id,
                        f"候选失败: {err_summary}",
                    )
                    try:
                        safe_detail = err_detail.encode("utf-8", errors="backslashreplace").decode("utf-8", errors="ignore")
                        logger.error(f"❌ candidate={candidate_id} 失败: {err_summary}\n{safe_detail}")
                    except Exception:
                        pass
                    return {
                        "candidate_id": candidate_id,
                        "status": "failed",
                        "error": err_summary,
                        "error_detail": err_detail,
                        "eval_image_field": None,
                    }

        # Create all tasks
        tasks = []
        for data in data_list:
            task = asyncio.create_task(process_with_semaphore(data))
            tasks.append(task)
        
        all_result_list = []
        eval_dims = ["faithfulness", "conciseness", "readability", "aesthetics", "overall"]

        with tqdm(total=len(tasks), desc="Processing concurrently") as pbar:
            # Iterate through completed tasks returned by as_completed
            for future in asyncio.as_completed(tasks):
                result_data = await future
                all_result_list.append(result_data)
                postfix_dict = {}
                for dim in eval_dims:
                    winner_key = f"{dim}_outcome"

                    if winner_key in result_data:
                        winners = [d.get(winner_key) for d in all_result_list]
                        total = len(winners)

                        if total > 0:
                            h_cnt = winners.count("Human")
                            m_cnt = winners.count("Model")
                            t_cnt = winners.count("Tie") + winners.count("Both are good") + winners.count("Both are bad")

                            h_rate = (h_cnt / total) * 100
                            m_rate = (m_cnt / total) * 100
                            t_rate = (t_cnt / total) * 100

                            display_key = dim[:5].capitalize()
                            postfix_dict[display_key] = f"{m_rate:.0f}/{t_rate:.0f}/{h_rate:.0f}"

                pbar.set_postfix(postfix_dict)
                pbar.update(1)
                yield result_data

    async def evaluation_function(
        self, data: Dict[str, Any], exp_config: ExpConfig
    ) -> Dict[str, Any]:
        """
        Evaluation function - uses referenced setting (GT shown first)
        """
        data = await get_score_for_image_referenced(
            data,
            task_name=exp_config.task_name,
            model_name=exp_config.model_name,
            work_dir=exp_config.work_dir,
        )
        return data
