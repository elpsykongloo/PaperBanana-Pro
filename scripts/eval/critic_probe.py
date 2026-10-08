"""Critic 筛查：换文本模型前，检查它当 Critic 时会不会几乎总回 "No changes needed."。

背景：gemini-3.5-flash-lite 当 Critic 时 36 次中 33 次回复无需修改，连明显错字都没指出，评审轮次形同关闭；
而它在检索评估上与 3.1 Flash-Lite 持平，只看检索指标发现不了这个问题。

输入为 pipeline_measure 的输出目录：对每条结果的第 0 轮图片与描述（Planner 或 Stylist 阶段），
用仓库 CriticAgent 以各模型评若干次。
  python -m scripts.eval.critic_probe --run results/eval/pipeline_xxx --models gemini-3.8-flash,gemini-3.1-flash-lite
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from pathlib import Path

from scripts.eval.common import REPO, load_api_key, load_json

NO_CHANGE = "no changes needed"


def round0_field(record: dict) -> str | None:
    sizes = record.get("image_sizes") or {}
    for field in ("target_diagram_stylist_desc0", "target_diagram_desc0"):
        if field in sizes:
            return field
    return None


async def main_async(args: argparse.Namespace) -> None:
    from agents.critic_agent import CriticAgent
    from utils import generation_utils
    from utils.config import ExpConfig

    run_dir = Path(args.run)
    records = load_json(run_dir / "results.json")["items"]
    generation_utils.init_gemini_client(load_api_key())
    models = [m for m in args.models.split(",") if m]
    agents = {m: CriticAgent(exp_config=ExpConfig(dataset_name="PaperBananaBench", task_name="diagram",
                                                  exp_mode="demo_planner_critic", provider="gemini", model_name=m,
                                                  work_dir=REPO)) for m in models}
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(record: dict, model: str, rep: int):
        field = round0_field(record)
        if not field or field not in record.get("descriptions", {}):
            return None
        data = {
            "candidate_id": f"probe-{record['id']}-{model}-{rep}", "content": record["content"],
            "visual_intent": record["visual_intent"], "current_critic_round": 0,
            field: record["descriptions"][field],
            f"{field}_base64_jpg": base64.b64encode((run_dir / field / f"{record['id']}.png").read_bytes()).decode(),
            f"{field}_mime_type": "image/png",
        }
        async with semaphore:
            out = await agents[model].process(data, source="stylist" if "stylist" in field else "planner")
        return record["id"], model, str(out.get("target_diagram_critic_suggestions0", ""))

    rows = [r for r in await asyncio.gather(*(one(rec, m, k) for rec in records for m in models for k in range(args.reps))) if r]
    print("模型 | 回复“无需修改”的次数 / 总次数")
    for model in models:
        mine = [s for _, m, s in rows if m == model]
        no_change = sum(1 for s in mine if s.strip().lower().startswith(NO_CHANGE))
        print(f"  {model}: {no_change}/{len(mine)}")
    if args.verbose:
        for item_id, model, sugg in rows:
            print(f"- {item_id} {model}: {sugg[:160].replace(chr(10), ' ')}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="pipeline_measure 的输出目录")
    parser.add_argument("--models", required=True, help="逗号分隔的文本模型")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
