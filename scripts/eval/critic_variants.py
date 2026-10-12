"""从同一张第 0 轮图出发，用改过提示的 Critic 重跑评审-重画循环，比较修改方式。

输入为 pipeline_measure 的输出目录（其中第 0 轮的图与描述作为起点，不重新规划）。每个方案的各轮成图导出到
<run>/variants/<方案>/r<k>/<id>.png（提前结束时沿用上一张），可直接交给 pairwise_eval / coverage_eval。
  python -m scripts.eval.critic_variants --run results/eval/critic_rounds --variants base,minimal,keep --rounds 1

方案：
  base     仓库现有提示（重跑一次，用来估计同一提示两次运行之间的随机差异）
  minimal  修订描述只改意见涉及的句子，其余逐字保留
  keep     先列出当前图中正确、必须保留的元素，修订描述必须全部保留
  guide    评审时同时参考风格指南（与 Stylist 同一份）
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import shutil
from pathlib import Path

from PIL import Image

from scripts.eval.common import CALL_TAG, REPO, UsageLog, load_api_key, load_bench, load_json, print_usage, save_json

ROUND0_FIELD = "target_diagram_desc0"

IMPORTANT_OLD = "Your Description should primarily be modifications based on the original description, rather than rewriting from scratch."

MINIMAL_RULE = (
    "Your revised description MUST keep every sentence of the original description that your critique does not address, "
    "word for word. Only rewrite the sentences your critique addresses, and add sentences for missing elements. Do not "
    "rephrase, reorder or shorten unaffected parts."
)

KEEP_RULE = (
    "Before revising, identify every element of the current diagram that is already correct (components, labels, "
    "connections, layout, colors). Your revised description MUST explicitly preserve all of them; change only what your "
    "critique says is wrong. Start \"critic_suggestions\" with a line \"Keep: ...\" listing these elements."
)


def build_prompt(variant: str, base_prompt: str) -> str:
    if variant == "minimal":
        return base_prompt.replace(IMPORTANT_OLD, MINIMAL_RULE + " " + IMPORTANT_OLD)
    if variant == "keep":
        return base_prompt.replace(IMPORTANT_OLD, KEEP_RULE + " " + IMPORTANT_OLD)
    return base_prompt


def style_guide_text() -> str:
    """与 StylistAgent 读取的是同一份风格指南。"""
    path = REPO / "style_guides" / "neurips2025_diagram_style_guide.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def export_chain(result: dict, item_id: str, out_dir: Path, rounds: int) -> list[int]:
    fields = [ROUND0_FIELD] + [f"target_diagram_critic_desc{r}" for r in range(rounds)]
    made: list[int] = []
    last: Path | None = None
    for k, field in enumerate(fields):
        target = out_dir / f"r{k}" / f"{item_id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        value = result.get(f"{field}_base64_jpg")
        if value:
            Image.open(io.BytesIO(base64.b64decode(value))).save(target)
            made.append(k)
            last = target
        elif last is not None:
            shutil.copyfile(last, target)
    return made


async def main_async(args: argparse.Namespace) -> None:
    from agents.critic_agent import DIAGRAM_CRITIC_AGENT_SYSTEM_PROMPT
    from scripts.eval.pipeline_measure import build_processor
    from utils import generation_utils

    test, _ = load_bench()
    run_dir = Path(args.run)
    summary = load_json(run_dir / "results.json")
    records = [r for r in summary["items"] if r["id"] in test and (run_dir / ROUND0_FIELD / f"{r['id']}.png").exists()]
    if args.ids:
        wanted = set(args.ids.split(","))
        records = [r for r in records if r["id"] in wanted]
    settings = summary["settings"]
    generation_utils.init_gemini_client(load_api_key())
    usage = UsageLog()
    usage.attach(generation_utils.get_gemini_client())
    _original = generation_utils.call_gemini_with_retry_async

    async def tagged(**kwargs):
        CALL_TAG.set(f"{CALL_TAG.get().split(':')[0]}:{str(kwargs.get('error_context', '')).split('[', 1)[0]}")
        return await _original(**kwargs)

    generation_utils.call_gemini_with_retry_async = tagged
    proc_args = argparse.Namespace(mode=settings["mode"], text_model=settings["text_model"], image_model=settings["image_model"],
                                   critic_rounds=args.rounds, retrieval=settings["retrieval"])
    guide = style_guide_text() if "guide" in args.variants else ""
    processors = {}
    for variant in [v for v in args.variants.split(",") if v]:
        _, processor = build_processor(proc_args)
        prompt = build_prompt(variant, DIAGRAM_CRITIC_AGENT_SYSTEM_PROMPT)
        if variant == "guide":
            if not guide:
                raise SystemExit("找不到风格指南文件。")
            prompt = prompt.replace("## INPUT DATA", "## STYLE GUIDE (use it when judging presentation)\n" + guide + "\n\n## INPUT DATA")
        processor.critic_agent.system_prompt = prompt
        processors[variant] = processor

    sem = asyncio.Semaphore(args.concurrency)

    async def one(variant: str, rec: dict) -> dict:
        item = test[rec["id"]]
        data = {
            "candidate_id": f"{variant}-{rec['id']}", "content": rec["content"], "visual_intent": rec["visual_intent"],
            "additional_info": {"rounded_ratio": (item.get("additional_info") or {}).get("rounded_ratio", "16:9"),
                                "image_resolution": settings["resolution"]},
            ROUND0_FIELD: rec["descriptions"][ROUND0_FIELD],
            f"{ROUND0_FIELD}_base64_jpg": base64.b64encode((run_dir / ROUND0_FIELD / f"{rec['id']}.png").read_bytes()).decode(),
            f"{ROUND0_FIELD}_mime_type": "image/png",
        }
        CALL_TAG.set(variant)
        async with sem:
            result = await processors[variant]._run_critic_iterations(data, "diagram", max_rounds=args.rounds, source="planner")
        made = export_chain(result, rec["id"], run_dir / "variants" / variant, args.rounds)
        return {"variant": variant, "id": rec["id"], "rounds_with_image": made,
                "critic": [{"round": r, "status": result.get(f"target_diagram_critic_status{r}"),
                            "suggestions": result.get(f"target_diagram_critic_suggestions{r}"),
                            "desc": result.get(f"target_diagram_critic_desc{r}")} for r in range(args.rounds)
                           if f"target_diagram_critic_suggestions{r}" in result]}

    rows = await asyncio.gather(*(one(v, r) for v in processors for r in records))
    save_json(run_dir / "variants" / f"variants_{'_'.join(processors)}.json", {"rounds": args.rounds, "rows": rows, "calls": usage.calls})
    for v in processors:
        mine = [r for r in rows if r["variant"] == v]
        print(f"  {v}: {len(mine)} 条，产出新图的轮次合计 {sum(len(r['rounds_with_image']) - 1 for r in mine)}")
    print_usage(usage.calls)
    for p in processors.values():
        p.shutdown()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--variants", default="base,minimal,keep")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--ids", default="")
    parser.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
