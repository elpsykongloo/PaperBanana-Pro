"""Run low-cost live smoke tests against configured providers."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import (
    CriticAgent,
    PlannerAgent,
    PolishAgent,
    RetrieverAgent,
    StylistAgent,
    VanillaAgent,
    VisualizerAgent,
)
from utils import generation_utils, paperviz_processor
from utils.config import ExpConfig
from utils.config_loader import (
    load_model_config,
)
from utils.demo_task_utils import create_sample_inputs
from utils.result_bundle import build_run_manifest, write_result_bundle
from utils.run_report import build_failure_manifest, build_result_summary
from utils.runtime_settings import build_runtime_context, resolve_runtime_settings


SMOKE_INPUTS = {
    "diagram": {
        "content": (
            "We first encode the paper text, then route it through a planner, "
            "and finally refine the figure with a critic loop."
        ),
        "visual_intent": (
            "Create a clean pipeline diagram with three stages: Encode, Plan, Refine."
        ),
    },
    "plot": {
        "content": json.dumps(
            [
                {"method": "PaperBanana", "step": 1, "score": 62.1},
                {"method": "PaperBanana", "step": 2, "score": 67.4},
                {"method": "Baseline", "step": 1, "score": 58.2},
                {"method": "Baseline", "step": 2, "score": 61.0},
            ],
            ensure_ascii=False,
        ),
        "visual_intent": (
            "Create a simple line plot of score versus step with one line per method, "
            "clear legend, labeled axes, and a publication-style layout."
        ),
    },
}


async def run_smoke_once(args) -> tuple[list[dict], dict, list[dict], Path]:
    root = ROOT
    model_config = load_model_config(root)
    runtime_settings = resolve_runtime_settings(
        args.provider,
        model_name=args.model_name,
        image_model_name=args.image_model_name,
        concurrency_mode="manual",
        max_concurrent=1,
        max_critic_rounds=args.max_critic_rounds,
        base_dir=root,
        model_config_data=model_config,
    )

    if not runtime_settings.api_key:
        raise RuntimeError(
            f"Provider '{args.provider}' 缺少本地 API key，无法执行 live smoke test。"
        )

    model_name = runtime_settings.model_name
    image_model_name = runtime_settings.image_model_name
    sample = SMOKE_INPUTS[args.task_name]
    data_list = create_sample_inputs(
        content=sample["content"],
        visual_intent=sample["visual_intent"],
        task_name=args.task_name,
        num_copies=1,
        max_critic_rounds=args.max_critic_rounds,
        image_resolution="2K",
    )

    exp_config = ExpConfig(
        dataset_name=args.dataset_name,
        task_name=args.task_name,
        split_name="smoke",
        exp_mode=args.exp_mode,
        retrieval_setting=args.retrieval_setting,
        max_critic_rounds=runtime_settings.max_critic_rounds,
        concurrency_mode=runtime_settings.concurrency_mode,
        max_concurrent=runtime_settings.max_concurrent,
        model_name=model_name,
        image_model_name=image_model_name,
        provider=runtime_settings.provider,
        work_dir=root,
    )

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

    runtime_context = build_runtime_context(runtime_settings)
    results = []
    try:
        with generation_utils.use_runtime_context(runtime_context):
            async for result in processor.process_queries_batch(
                data_list,
                max_concurrent=1,
                do_eval=False,
            ):
                results.append(result)
    finally:
        await generation_utils.close_runtime_context(runtime_context)
        processor.shutdown()

    summary = build_result_summary(results)
    failures = build_failure_manifest(results)

    smoke_dir = root / "results" / "smoke" / args.task_name
    smoke_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = smoke_dir / f"{stamp}_{args.provider}_{args.task_name}.json"
    manifest = build_run_manifest(
        exp_config=exp_config,
        producer="smoke",
        result_count=len(results),
        model_name=model_name,
        image_model_name=image_model_name,
    )
    write_result_bundle(
        output_path,
        results,
        manifest=manifest,
        summary=summary,
        failures=failures,
    )

    return results, summary, failures, output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run a low-cost live smoke test.")
    parser.add_argument(
        "--dataset_name",
        default="PaperBananaBench",
        help="Dataset assets to use for retrieval/reference-path resolution.",
    )
    parser.add_argument(
        "--task_name",
        choices=["diagram", "plot"],
        required=True,
        help="Task to validate.",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "evolink"],
        default="gemini",
        help="Provider to validate.",
    )
    parser.add_argument(
        "--exp_mode",
        default="demo_planner_critic",
        choices=["demo_planner_critic", "demo_full", "dev_planner", "dev_planner_critic"],
        help="Pipeline mode to validate.",
    )
    parser.add_argument(
        "--retrieval_setting",
        default="none",
        choices=["none", "random", "auto", "auto-full", "manual"],
        help="Retrieval mode for smoke test.",
    )
    parser.add_argument(
        "--max_critic_rounds",
        type=int,
        default=0,
        help="Critic rounds for smoke test; 0 keeps cost minimal.",
    )
    parser.add_argument(
        "--model_name",
        default="",
        help="Override text model name.",
    )
    parser.add_argument(
        "--image_model_name",
        default="",
        help="Override image model name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        _, summary, failures, output_path = asyncio.run(run_smoke_once(args))
    except Exception as exc:
        print(f"SMOKE_FAILED {type(exc).__name__}: {exc}")
        sys.exit(2)

    print(
        "SMOKE_SUMMARY",
        json.dumps(
            {
                "provider": args.provider,
                "dataset_name": args.dataset_name,
                "task_name": args.task_name,
                "summary": summary,
                "output_path": str(output_path),
            },
            ensure_ascii=False,
        ),
    )

    if failures or summary["failed_candidates"] > 0 or summary["missing_render_candidates"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
