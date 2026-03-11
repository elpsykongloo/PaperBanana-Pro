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
Main script to launch PaperBanana
"""

import asyncio
import json
import argparse
from pathlib import Path
import aiofiles
import numpy as np

from utils.dataset_paths import get_dataset_split_path
from utils.log_config import setup_logging, get_logger
from utils.pipeline_registry import get_supported_exp_modes
from utils.retrieval_settings import CLI_RETRIEVAL_SETTING_CHOICES
from utils.result_bundle import (
    build_run_manifest,
    companion_bundle_path,
    write_json_payload_async,
    write_result_bundle_async,
)
from utils.result_order import sort_results_stably
from utils.run_report import build_failure_manifest, build_result_summary
from utils.runtime_settings import build_runtime_context
setup_logging("INFO", mode="cli")

from agents import (
    VanillaAgent, PlannerAgent, VisualizerAgent,
    StylistAgent, CriticAgent, RetrieverAgent, PolishAgent
)

from utils import config, generation_utils, paperviz_processor

logger = get_logger("Main")


async def main():
    """Main function"""
    # add command line args
    parser = argparse.ArgumentParser(description="PaperBanana processing script")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PaperBananaBench",
        help="name of the dataset to use (default: PaperBananaBench)",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="diagram",
        choices=["diagram", "plot"],
        help="task type: diagram or plot (default: diagram)",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="test",
        help="split of the dataset to use (default: test)",
    )
    parser.add_argument(
        "--exp_mode",
        type=str,
        default="dev_full",
        choices=get_supported_exp_modes(),
        help="name of the experiment to use (default: dev_full)",
    )
    parser.add_argument(
        "--retrieval_setting",
        type=str,
        default="auto",
        choices=list(CLI_RETRIEVAL_SETTING_CHOICES),
        help="retrieval setting for planner agent (default: auto; 'manual' is kept as a legacy alias for 'curated')",
    )
    parser.add_argument(
        "--curated_profile",
        type=str,
        default="default",
        help="curated retrieval profile name when retrieval_setting is 'curated' (default: default)",
    )
    parser.add_argument(
        "--max_critic_rounds",
        type=int,
        default=3,
        help="maximum number of critic rounds (default: 3)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="model name to use (default: "")",
    )
    parser.add_argument(
        "--image_model_name",
        type=str,
        default="",
        help='image model name to use (default: "")',
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="evolink",
        choices=["evolink", "gemini"],
        help="provider to use (default: evolink)",
    )
    parser.add_argument(
        "--concurrency_mode",
        type=str,
        default="auto",
        choices=["auto", "manual"],
        help="concurrency strategy label for result metadata (default: auto)",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=20,
        help="maximum concurrent samples to process (default: 20)",
    )
    args = parser.parse_args()

    exp_config = config.ExpConfig(
        dataset_name=args.dataset_name,
        task_name=args.task_name,
        split_name=args.split_name,
        exp_mode=args.exp_mode,
        retrieval_setting=args.retrieval_setting,
        curated_profile=args.curated_profile,
        max_critic_rounds=args.max_critic_rounds,
        concurrency_mode=args.concurrency_mode,
        max_concurrent=args.max_concurrent,
        model_name=args.model_name,
        image_model_name=args.image_model_name,
        provider=args.provider,
        work_dir=Path(__file__).parent,
    )
    
    input_filename = get_dataset_split_path(
        exp_config.dataset_name,
        exp_config.task_name,
        exp_config.split_name,
        work_dir=Path(__file__).parent,
    )
    output_filename = exp_config.result_dir / f"{exp_config.exp_name}.json"
    bundle_filename = companion_bundle_path(output_filename)
    
    logger.info(f"📁 输入文件: {input_filename}  输出文件: {output_filename}")
    logger.info(
        "⚙️ 运行配置 | provider=%s | text_model=%s | image_model=%s | retrieval=%s | critic_rounds=%s | concurrency=%s/%s",
        exp_config.provider,
        exp_config.model_name,
        exp_config.image_model_name,
        exp_config.retrieval_setting,
        exp_config.max_critic_rounds,
        exp_config.concurrency_mode,
        exp_config.max_concurrent,
    )
    with open(input_filename, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # Create processor
    processor = paperviz_processor.PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )

    # Batch process documents
    concurrent_num = exp_config.max_concurrent
    logger.info(f"🚀 最大并发数: {concurrent_num}")
    all_result_list = []
    failed_count = 0

    async def save_results_and_scores(current_results):
        ordered_results = sort_results_stably(current_results)
        logger.info(f"💾 增量保存结果（共 {len(ordered_results)} 条）到 {output_filename}")
        summary = build_result_summary(ordered_results)
        failures = build_failure_manifest(ordered_results)
        manifest = build_run_manifest(
            exp_config=exp_config,
            producer="cli",
            result_count=len(ordered_results),
        )
        await write_json_payload_async(output_filename, ordered_results)
        await write_result_bundle_async(
            bundle_filename,
            ordered_results,
            manifest=manifest,
            summary=summary,
            failures=failures,
        )

    async def save_run_reports(current_results):
        ordered_results = sort_results_stably(current_results)
        summary_path = output_filename.with_suffix(".summary.json")
        failures_path = output_filename.with_suffix(".failures.json")
        summary = build_result_summary(ordered_results)
        manifest = build_run_manifest(
            exp_config=exp_config,
            producer="cli",
            result_count=len(ordered_results),
        )
        summary_payload = {
            "dataset_name": exp_config.dataset_name,
            "task_name": exp_config.task_name,
            "provider": exp_config.provider,
            "model_name": exp_config.model_name,
            "image_model_name": exp_config.image_model_name,
            "exp_mode": exp_config.exp_mode,
            "retrieval_setting": exp_config.retrieval_setting,
            "curated_profile": exp_config.curated_profile,
            "manifest": manifest,
            "summary": summary,
        }
        failures_payload = build_failure_manifest(ordered_results)

        async with aiofiles.open(
            summary_path, "w", encoding="utf-8", errors="surrogateescape"
        ) as f:
            await f.write(json.dumps(summary_payload, ensure_ascii=False, indent=4))
        async with aiofiles.open(
            failures_path, "w", encoding="utf-8", errors="surrogateescape"
        ) as f:
            await f.write(json.dumps(failures_payload, ensure_ascii=False, indent=4))

    # Process samples incrementally
    idx = 0
    runtime_context = build_runtime_context(exp_config.runtime_settings)
    try:
        with generation_utils.use_runtime_context(runtime_context):
            async for result_data in processor.process_queries_batch(
                data_list, max_concurrent=concurrent_num
            ):
                all_result_list.append(result_data)
                idx += 1
                if isinstance(result_data, dict) and result_data.get("status") == "failed":
                    failed_count += 1
                if idx % 5 == 0 or idx == len(data_list):
                    logger.info(
                        f"📈 进度: {idx}/{len(data_list)} | 失败 {failed_count} | 成功 {idx - failed_count}"
                    )
                if idx % 10 == 0:
                    await save_results_and_scores(all_result_list)
    finally:
        await generation_utils.close_runtime_context(runtime_context)
        processor.shutdown()

    # Final save
    await save_results_and_scores(all_result_list)
    await save_run_reports(all_result_list)
    summary = build_result_summary(all_result_list)
    logger.info(
        f"✅ 处理完成 | 总数 {len(all_result_list)} | 失败 {failed_count} | 成功 {len(all_result_list) - failed_count}"
    )
    logger.info(
        "🧾 结果摘要 | rendered=%s | missing_render=%s | parse_error_candidates=%s",
        summary["rendered_candidates"],
        len(summary["missing_render_candidates"]),
        len(summary["parse_error_candidates"]),
    )


if __name__ == "__main__":
    asyncio.run(main())
