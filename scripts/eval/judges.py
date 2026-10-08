"""三种评审：参考有用度（0–3 分）、成图成对盲评、成图中的非内容文字。

评审口径与 baselines/ 中的历史数据一致；改动提示或评审模型后，结果不能再与基准直接比较。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from scripts.eval.common import combine_pairwise, gen_text, image_part, parse_json

USEFULNESS_JUDGE_MODEL = "gemini-3-flash-preview"
PAIRWISE_JUDGE_MODEL = "gemini-3.1-pro-preview"
LEAK_JUDGE_MODEL = "gemini-3-flash-preview"
PAIRWISE_DIMS = ("overall", "faithfulness", "readability", "conciseness", "aesthetics")

USEFULNESS_SYSTEM = "You are an expert designer of figures for top-tier AI papers."
USEFULNESS_RUBRIC = (
    "For EACH candidate, rate how useful it would be as a visual template (diagram type, layout/composition, "
    "visual style) for drawing the TARGET figure. Ignore research topic and wording; focus on structure and style.\n"
    "3 = same diagram type and very similar layout/composition; a designer could closely follow it.\n"
    "2 = same diagram type, but layout or complexity notably different; still useful.\n"
    "1 = weakly useful (different type, only some shared visual conventions).\n"
    "0 = not useful (e.g., target is a pipeline but candidate is a chart, table, or photo grid).\n"
    'Return JSON mapping each label to an integer score, e.g. {"C1": 2, "C2": 0}.'
)

PAIRWISE_SYSTEM = "You are a strict reviewer of figures for top-tier AI conferences."
PAIRWISE_PROMPT = (
    "Two candidate figures (Figure A and Figure B) were generated for the same paper. The original human-drawn figure "
    "is also shown for context only. Compare A and B on: faithfulness (correctly and completely represents the method; "
    "no wrong, missing, or hallucinated key components or text), readability (clear layout, legible text, no clutter), "
    "conciseness (focuses on the key ideas), aesthetics (professional, publication-quality look). "
    'Return JSON: {"faithfulness": "A"|"B"|"Tie", "readability": ..., "conciseness": ..., "aesthetics": ..., '
    '"overall": "A"|"B"|"Tie", "reason": "<=40 words"}'
)

LEAK_PROMPT = (
    "List every piece of visible text in the generated figure that is NOT diagram content, for example color names, "
    "style or design terms (like 'Mint Sage', 'Elbow', 'Glassmorphism', 'Macro-Micro'), layout or rendering instructions, "
    'or placeholder text. Return JSON {"leaked_text": [strings], "count": integer}.'
)


async def judge_usefulness(client, target: dict, target_image: Path, candidates: list[tuple[str, Path]],
                           semaphore: asyncio.Semaphore) -> dict[str, int | None]:
    """对一批（≤10 张）候选参考打 0–3 分。candidates 为 (参考 id, 图片路径)。"""
    parts: list[Any] = [f"TARGET figure caption: {str(target.get('visual_intent', ''))[:600]}\nTARGET figure:",
                        image_part(target_image, 768), "\nCANDIDATE reference figures from other papers:"]
    for j, (_, path) in enumerate(candidates):
        parts += [f"\nC{j + 1}:", image_part(path, 512)]
    parts.append("\n" + USEFULNESS_RUBRIC)
    async with semaphore:
        text = await gen_text(client, USEFULNESS_JUDGE_MODEL, parts, system=USEFULNESS_SYSTEM, temperature=0.0)
    obj = parse_json(text)
    scores: dict[str, int | None] = {}
    for j, (ref_id, _) in enumerate(candidates):
        value = obj.get(f"C{j + 1}") if isinstance(obj, dict) else None
        try:
            scores[ref_id] = int(value)
        except (TypeError, ValueError):
            scores[ref_id] = None
    return scores


async def pairwise_once(client, item: dict, gt_image: Path, image_a: Path, image_b: Path,
                        semaphore: asyncio.Semaphore) -> dict:
    parts = [f"Caption: {item['visual_intent']}\nMethodology (truncated): {str(item['content'])[:12000]}\n\n"
             "Original human-drawn figure (context only):",
             image_part(gt_image, 1024), "\nFigure A:", image_part(image_a, 1536), "\nFigure B:", image_part(image_b, 1536),
             "\n" + PAIRWISE_PROMPT]
    async with semaphore:
        text = await gen_text(client, PAIRWISE_JUDGE_MODEL, parts, system=PAIRWISE_SYSTEM, temperature=0.0)
    obj = parse_json(text)
    return obj if isinstance(obj, dict) else {}


async def pairwise_both_orders(client, item: dict, gt_image: Path, image_x: Path, image_y: Path,
                               semaphore: asyncio.Semaphore) -> dict[str, Any]:
    """x 对 y 的成对盲评：两种顺序各评一次，两次都判 x 胜才记 win。"""
    first, second = await asyncio.gather(
        pairwise_once(client, item, gt_image, image_x, image_y, semaphore),
        pairwise_once(client, item, gt_image, image_y, image_x, semaphore),
    )
    verdicts = {dim: combine_pairwise(first, second, dim) for dim in PAIRWISE_DIMS}
    verdicts["raw"] = [first, second]
    return verdicts


async def leak_check(client, item: dict, image: Path, semaphore: asyncio.Semaphore) -> dict[str, Any]:
    parts = [f"Intended figure caption: {str(item['visual_intent'])[:600]}\nMethod excerpt: {str(item['content'])[:3000]}\n\n"
             "Generated figure:", image_part(image, 1536), LEAK_PROMPT]
    async with semaphore:
        text = await gen_text(client, LEAK_JUDGE_MODEL, parts, temperature=0.0, json_mode=True)
    obj = parse_json(text)
    if not isinstance(obj, dict):
        return {"leaked_text": [], "count": 0}
    try:
        obj["count"] = int(obj.get("count") or len(obj.get("leaked_text") or []))
    except (TypeError, ValueError):
        obj["count"] = len(obj.get("leaked_text") or [])
    return obj
