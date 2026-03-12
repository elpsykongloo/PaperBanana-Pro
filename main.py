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

from utils.cli_checkpoint import (
    append_cli_checkpoint_event,
    build_cli_checkpoint_payload,
    checkpoint_event_log_path,
    checkpoint_path_for_output,
    collect_completed_input_indices,
    dedupe_results_by_input_index,
    prepare_pending_inputs,
    read_cli_checkpoint,
    write_cli_checkpoint,
)
from utils.dataset_paths import get_dataset_split_path
from utils.log_config import setup_logging, get_logger
from utils.pipeline_registry import get_supported_exp_modes
from utils.retrieval_settings import CLI_RETRIEVAL_SETTING_CHOICES
from utils.result_bundle import (
    build_run_manifest,
    companion_bundle_path,
    load_result_bundle,
    write_json_payload_async,
    write_result_bundle_async,
)
from utils.run_report import build_failure_manifest, build_result_summary
from utils.runtime_settings import DEFAULT_PROVIDER, build_runtime_context
setup_logging("INFO", mode="cli")

from agents import (
    VanillaAgent, PlannerAgent, VisualizerAgent,
    StylistAgent, CriticAgent, RetrieverAgent, PolishAgent
)

from utils import config, generation_utils, paperviz_processor

logger = get_logger("Main")


def resolve_resume_source_path(
    *,
    resume_flag: bool,
    resume_from: str,
    checkpoint_path: Path,
    bundle_path: Path,
    output_path: Path,
) -> tuple[Path | None, dict | None]:
    if resume_from:
        explicit_path = Path(resume_from).expanduser()
        if explicit_path.name.endswith(".checkpoint.json"):
            checkpoint_payload = read_cli_checkpoint(explicit_path)
            if checkpoint_payload is None:
                raise FileNotFoundError(explicit_path)
            for key in ("bundle_file", "output_file"):
                referenced_path = str(checkpoint_payload.get(key, "") or "").strip()
                if referenced_path and Path(referenced_path).exists():
                    return Path(referenced_path), checkpoint_payload
            raise FileNotFoundError(
                f"{explicit_path} 中没有可恢复的结果文件（bundle/output）"
            )
        if not explicit_path.exists():
            raise FileNotFoundError(explicit_path)
        return explicit_path, None

    if not resume_flag:
        return None, None

    if checkpoint_path.exists():
        checkpoint_payload = read_cli_checkpoint(checkpoint_path)
        if checkpoint_payload:
            for key in ("bundle_file", "output_file"):
                referenced_path = str(checkpoint_payload.get(key, "") or "").strip()
                if referenced_path and Path(referenced_path).exists():
                    return Path(referenced_path), checkpoint_payload
    if bundle_path.exists():
        return bundle_path, None
    if output_path.exists():
        return output_path, None
    raise FileNotFoundError(
        "未找到可恢复的 checkpoint / bundle / 结果文件，请先运行一次，或用 --resume_from 指定已有文件。"
    )


def load_resumed_results(resume_source_path: Path | None) -> list[dict]:
    if resume_source_path is None:
        return []
    bundle = load_result_bundle(resume_source_path)
    return dedupe_results_by_input_index(bundle.get("results", []))


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
        default=DEFAULT_PROVIDER,
        choices=["gemini", "evolink"],
        help=f"provider to use (default: {DEFAULT_PROVIDER})",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the latest checkpoint/bundle for this run configuration",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="explicit checkpoint, bundle, or legacy result file to resume from",
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
    summary_path = output_filename.with_suffix(".summary.json")
    failures_path = output_filename.with_suffix(".failures.json")
    checkpoint_filename = checkpoint_path_for_output(output_filename)
    checkpoint_events_filename = checkpoint_event_log_path(checkpoint_filename)
    
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

    resume_source_path, resume_checkpoint = resolve_resume_source_path(
        resume_flag=bool(args.resume),
        resume_from=args.resume_from,
        checkpoint_path=checkpoint_filename,
        bundle_path=bundle_filename,
        output_path=output_filename,
    )
    resumed_results = load_resumed_results(resume_source_path)
    completed_input_indices = collect_completed_input_indices(resumed_results)
    pending_data_list = prepare_pending_inputs(data_list, completed_input_indices)

    if resume_source_path is not None:
        logger.info(
            "♻️ 恢复模式 | source=%s | 已完成 %s/%s | 待继续 %s",
            resume_source_path,
            len(resumed_results),
            len(data_list),
            len(pending_data_list),
        )
    elif args.resume or args.resume_from:
        logger.info("♻️ 恢复模式未发现历史结果，将从空状态开始。")

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
    all_result_list = list(resumed_results)
    failed_count = sum(
        1
        for item in all_result_list
        if isinstance(item, dict) and item.get("status") == "failed"
    )

    append_cli_checkpoint_event(
        checkpoint_events_filename,
        event_type="run_started",
        status="running",
        message="CLI 批处理启动",
        details={
            "input_file": str(input_filename),
            "resume_source": str(resume_source_path or ""),
            "resume_status": str((resume_checkpoint or {}).get("status", "") or ""),
            "total_inputs": len(data_list),
            "pending_inputs": len(pending_data_list),
        },
    )

    async def save_results_and_scores(
        current_results,
        *,
        checkpoint_status: str = "running",
        checkpoint_error: str = "",
    ):
        ordered_results = dedupe_results_by_input_index(current_results)
        logger.info(f"💾 增量保存结果（共 {len(ordered_results)} 条）到 {output_filename}")
        summary = build_result_summary(ordered_results)
        failures = build_failure_manifest(ordered_results)
        current_manifest = build_run_manifest(
            exp_config=exp_config,
            producer="cli",
            result_count=len(ordered_results),
        )
        await write_json_payload_async(output_filename, ordered_results)
        await write_result_bundle_async(
            bundle_filename,
            ordered_results,
            manifest=current_manifest,
            summary=summary,
            failures=failures,
        )
        checkpoint_payload = build_cli_checkpoint_payload(
            manifest=current_manifest,
            input_file=input_filename,
            output_file=output_filename,
            bundle_file=bundle_filename,
            summary_file=summary_path,
            failures_file=failures_path,
            total_inputs=len(data_list),
            results=ordered_results,
            status=checkpoint_status,
            error=checkpoint_error,
            resume_source=str(resume_source_path or ""),
        )
        write_cli_checkpoint(checkpoint_filename, checkpoint_payload)
        append_cli_checkpoint_event(
            checkpoint_events_filename,
            event_type="checkpoint_saved",
            status=checkpoint_status,
            message="已写入 CLI checkpoint",
            details={
                "result_count": len(ordered_results),
                "failed_count": len(failures),
                "output_file": str(output_filename),
                "bundle_file": str(bundle_filename),
            },
        )

    async def save_run_reports(current_results):
        ordered_results = dedupe_results_by_input_index(current_results)
        summary = build_result_summary(ordered_results)
        current_manifest = build_run_manifest(
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
            "manifest": current_manifest,
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
    idx = len(all_result_list)
    processed_since_last_save = 0
    run_status = "completed"
    run_error = ""
    runtime_context = None
    try:
        if pending_data_list:
            runtime_context = build_runtime_context(exp_config.runtime_settings)
            with generation_utils.use_runtime_context(runtime_context):
                async for result_data in processor.process_queries_batch(
                    pending_data_list, max_concurrent=concurrent_num
                ):
                    all_result_list.append(result_data)
                    idx += 1
                    processed_since_last_save += 1
                    if isinstance(result_data, dict) and result_data.get("status") == "failed":
                        failed_count += 1
                    if idx % 5 == 0 or idx == len(data_list):
                        logger.info(
                            f"📈 进度: {idx}/{len(data_list)} | 失败 {failed_count} | 成功 {idx - failed_count}"
                        )
                    if processed_since_last_save >= 10:
                        await save_results_and_scores(all_result_list)
                        processed_since_last_save = 0
        else:
            logger.info("✅ 当前配置下的样本已全部完成，无需重复执行。")
    except KeyboardInterrupt as err:
        run_status = "interrupted"
        run_error = f"{type(err).__name__}: {err}"
        raise
    except BaseException as err:
        run_status = "failed"
        run_error = f"{type(err).__name__}: {err}"
        raise
    finally:
        if runtime_context is not None:
            await generation_utils.close_runtime_context(runtime_context)
        processor.shutdown()
        await save_results_and_scores(
            all_result_list,
            checkpoint_status=run_status,
            checkpoint_error=run_error,
        )
        await save_run_reports(all_result_list)
        append_cli_checkpoint_event(
            checkpoint_events_filename,
            event_type="run_finished",
            status=run_status,
            message="CLI 批处理结束",
            details={
                "result_count": len(dedupe_results_by_input_index(all_result_list)),
                "failed_count": failed_count,
                "error": run_error,
            },
        )

    summary = build_result_summary(dedupe_results_by_input_index(all_result_list))
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
