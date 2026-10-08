"""整链测量：用仓库 PaperVizProcessor 跑若干条 PaperBananaBench 输入，记录各阶段的模型、耗时、token 与估算费用，
并把每一轮的成图导出为 <输出目录>/<字段名>/<id>.png，供 pairwise_eval 比较不同轮次或不同设置。

示例（新默认模型、GUI 默认的标准流程、2K、3 轮 Critic）：
  python -m scripts.eval.pipeline_measure --n 3 --include-zh --leak-check
  python -m scripts.eval.pipeline_measure --n 20 --seed 11 --critic-rounds 3 --out results/eval/critic_rounds
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import random
import time
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.eval.common import (
    BASELINE_DIR,
    CALL_TAG,
    OUTPUT_DIR,
    REPO,
    UsageLog,
    call_cost,
    load_api_key,
    load_bench,
    load_json,
    make_client,
    print_usage,
    save_json,
    timestamp,
)

ZH_SAMPLE = {
    "id": "zh_sample",
    "content": "本文提出一种检索增强的多模态问答框架。系统首先用视觉编码器提取图像区域特征，同时用文本编码器编码问题；"
    "随后检索模块从外部知识库中召回相关段落，并通过交叉注意力融合模块将区域特征、问题与检索段落对齐；"
    "最后由大语言模型生成答案，并使用一致性校验器对答案与证据做匹配，不一致时触发重新检索。",
    "visual_intent": "图 2：检索增强多模态问答框架的整体流程。",
    "additional_info": {"rounded_ratio": "16:9"},
}


def baseline_ids() -> set[str]:
    """基准文件里已用过的查询；默认抽样时排除，避免与历史实验重叠。"""
    used: set[str] = set()
    for path in BASELINE_DIR.glob("*.json"):
        used |= {it["id"] for it in load_json(path).get("items", [])}
    return used


def pick_items(test: dict[str, dict], args: argparse.Namespace) -> list[dict]:
    if args.ids:
        return [test[i] for i in args.ids.split(",") if i]
    used = baseline_ids()
    pool = sorted((x for i, x in test.items() if i not in used), key=lambda x: x["id"])
    rng = random.Random(args.seed)
    by_cat: dict[str, list[dict]] = {}
    for x in pool:
        by_cat.setdefault(x.get("category", ""), []).append(x)
    picked: list[dict] = []
    cats = sorted(by_cat)
    for cat in cats:
        rng.shuffle(by_cat[cat])
    while len(picked) < args.n and any(by_cat.values()):
        for cat in cats:
            if by_cat[cat] and len(picked) < args.n:
                picked.append(by_cat[cat].pop())
    return picked


def build_processor(args: argparse.Namespace):
    from agents.critic_agent import CriticAgent
    from agents.planner_agent import PlannerAgent
    from agents.polish_agent import PolishAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.stylist_agent import StylistAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.visualizer_agent import VisualizerAgent
    from utils.config import ExpConfig
    from utils.paperviz_processor import PaperVizProcessor

    exp_config = ExpConfig(
        dataset_name="PaperBananaBench", task_name="diagram", exp_mode=args.mode, provider="gemini",
        model_name=args.text_model, image_model_name=args.image_model, max_critic_rounds=args.critic_rounds,
        retrieval_setting=args.retrieval, work_dir=REPO,
    )
    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )
    return exp_config, processor


def export_images(result: dict[str, Any], item_id: str, out_dir: Path) -> dict[str, list[int]]:
    sizes: dict[str, list[int]] = {}
    for key, value in result.items():
        if not key.endswith("_base64_jpg") or not value:
            continue
        raw = base64.b64decode(value)
        field = key[: -len("_base64_jpg")]
        target = out_dir / field / f"{item_id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(io.BytesIO(raw))
        image.save(target)
        sizes[field] = list(image.size)
    return sizes


async def main_async(args: argparse.Namespace) -> Path:
    from scripts.eval.judges import leak_check
    from utils import generation_utils

    test, _ = load_bench()
    items = pick_items(test, args)
    if args.include_zh:
        items.append(dict(ZH_SAMPLE))
    out_dir = Path(args.out) if args.out else OUTPUT_DIR / f"pipeline_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    generation_utils.init_gemini_client(load_api_key())
    usage = UsageLog()
    usage.attach(generation_utils.get_gemini_client())
    warnings: list[dict[str, Any]] = []
    context = generation_utils.get_default_runtime_context()
    context.event_hook = lambda e: warnings.append(e) if str(e.get("level", "")).upper() in {"WARNING", "ERROR"} else None

    _original = generation_utils.call_gemini_with_retry_async

    async def tagged(**kwargs):
        CALL_TAG.set(str(kwargs.get("error_context", "")).split("[", 1)[0] or "other")
        return await _original(**kwargs)

    generation_utils.call_gemini_with_retry_async = tagged
    exp_config, processor = build_processor(args)
    print(f"文本模型 {exp_config.model_name}，生图模型 {exp_config.image_model_name}，流程 {args.mode}，"
          f"{args.resolution}，Critic 最多 {args.critic_rounds} 轮，{len(items)} 条", flush=True)

    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(idx: int, item: dict) -> dict[str, Any]:
        doc = {
            "candidate_id": idx, "input_index": idx, "content": item["content"], "visual_intent": item["visual_intent"],
            "additional_info": {"rounded_ratio": (item.get("additional_info") or {}).get("rounded_ratio", "16:9"),
                                "image_resolution": args.resolution},
        }
        async with semaphore:
            started = time.time()
            result = await processor.process_single_query(doc, do_eval=False)
            return {"result": result, "secs": round(time.time() - started, 1)}

    started = time.time()
    outs = await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
    pipeline_calls = list(usage.calls)
    print(f"流水线耗时 {time.time() - started:.0f}s；各条耗时 {[o['secs'] for o in outs]}", flush=True)

    records = []
    for item, out in zip(items, outs):
        result = out["result"]
        sizes = export_images(result, item["id"], out_dir)
        rounds = sorted(int(k[len("target_diagram_critic_suggestions"):]) for k in result
                        if k.startswith("target_diagram_critic_suggestions") and k[len("target_diagram_critic_suggestions"):].isdigit())
        records.append({
            "id": item["id"], "secs": out["secs"], "eval_image_field": result.get("eval_image_field"),
            "image_sizes": sizes, "retrieval_meta": result.get("retrieval_meta"),
            "top10_references": result.get("top10_references"),
            "critic_rounds": [{"round": r, "status": result.get(f"target_diagram_critic_status{r}"),
                               "suggestions": result.get(f"target_diagram_critic_suggestions{r}")} for r in rounds],
            "descriptions": {k: v for k, v in result.items()
                             if k.startswith("target_diagram_") and k.endswith(tuple("0123456789")) and "desc" in k
                             and isinstance(v, str)},
            "content": item["content"], "visual_intent": item["visual_intent"],
        })

    if args.leak_check:
        client = make_client()
        usage.attach(client)
        leak_sem = asyncio.Semaphore(8)
        CALL_TAG.set("leak_check")
        for item, rec in zip(items, records):
            fields = sorted(rec["image_sizes"])
            results = await asyncio.gather(*(leak_check(client, item, out_dir / f / f"{item['id']}.png", leak_sem) for f in fields))
            rec["leak"] = {f: r.get("leaked_text", []) for f, r in zip(fields, results)}

    per_item_cost = sum(call_cost(c) for c in pipeline_calls) / max(1, len(items))
    summary = {
        "settings": {"text_model": exp_config.model_name, "image_model": exp_config.image_model_name, "mode": args.mode,
                     "resolution": args.resolution, "critic_rounds": args.critic_rounds, "retrieval": args.retrieval},
        "per_item_cost_estimate": round(per_item_cost, 4),
        "warnings": [{"level": w.get("level"), "message": w.get("message")} for w in warnings],
        "calls": usage.calls,
        "items": records,
    }
    save_json(out_dir / "results.json", summary)

    print("\n各阶段用量（流水线，不含非内容文字检查）：")
    print_usage(pipeline_calls)
    print(f"  平均每条约 ${per_item_cost:.3f}；告警/错误事件 {len(warnings)} 条")
    for rec in records:
        meta = rec.get("retrieval_meta") or {}
        changed = sum(1 for r in rec["critic_rounds"] if str(r.get("suggestions") or "").strip() != "No changes needed.")
        print(f"\n[{rec['id']}] {rec['secs']}s 检索={meta.get('method')} 最终={rec['eval_image_field']} "
              f"Critic 轮次 {len(rec['critic_rounds'])}（提出修改 {changed}）")
        for f, size in sorted(rec["image_sizes"].items()):
            leak = (rec.get("leak") or {}).get(f)
            print(f"   {f}: {size[0]}x{size[1]}" + (f"，非内容文字={leak}" if leak is not None else ""))
    print(f"\n结果与图片：{out_dir}")
    processor.shutdown()
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=3, help="从测试集按类别轮流抽取的条数（排除基准里用过的）")
    parser.add_argument("--ids", default="", help="逗号分隔的测试集 id；给出时忽略 --n")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--include-zh", action="store_true", help="额外加入一条中文输入")
    parser.add_argument("--mode", default="demo_planner_critic", choices=["demo_planner_critic", "demo_full"])
    parser.add_argument("--critic-rounds", type=int, default=3)
    parser.add_argument("--resolution", default="2K", choices=["1K", "2K", "4K"])
    parser.add_argument("--retrieval", default="auto")
    parser.add_argument("--text-model", default="", help="默认用配置里的模型")
    parser.add_argument("--image-model", default="", help="默认用配置里的模型")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--leak-check", action="store_true", help="用 Flash 检查每张图里的非内容文字")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
