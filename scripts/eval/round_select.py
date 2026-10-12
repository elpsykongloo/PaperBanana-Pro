"""Critic 轮次整理与"挑最好一轮"选择器。

输入为 pipeline_measure 的输出目录。生成：
  rounds/r0 … rounds/rN：每一轮之后"当前最新"的图（某轮提前结束时沿用上一张），可直接交给 pairwise_eval / coverage_eval；
  rounds/selected：选择器（不看原图，只看方法与图注）从各轮中挑出的一张。
  python -m scripts.eval.round_select --run results/eval/critic_rounds
选择器使用流水线默认文本模型，各轮图片按每条固定的随机顺序给出，避免位置偏好。
"""

from __future__ import annotations

import argparse
import asyncio
import random
import shutil
from pathlib import Path

from scripts.eval.common import CALL_TAG, UsageLog, gen_text, image_part, load_json, make_client, parse_json, print_usage, save_json

SELECT_MODEL = "gemini-3.8-flash"
SELECT_PROMPT = (
    "Several candidate versions of the same method diagram are shown below. Pick the ONE best suited for publication in "
    "a top-tier AI paper. Priorities, in order: (1) faithful to the methodology and caption — no wrong, missing or "
    "hallucinated components, no misspellings; (2) readable — clear flow, legible text, no clutter; (3) professional look. "
    'Return JSON {{"best": "<label>", "reason": "<=30 words"}} where label is one of {labels}.'
)


def round_fields(n_rounds: int) -> list[str]:
    return ["target_diagram_desc0"] + [f"target_diagram_critic_desc{r}" for r in range(n_rounds)]


def build_round_dirs(run_dir: Path, records: list[dict], n_rounds: int) -> dict[str, list[Path]]:
    out = run_dir / "rounds"
    chains: dict[str, list[Path]] = {}
    for rec in records:
        chain: list[Path] = []
        for k, field in enumerate(round_fields(n_rounds)):
            src = run_dir / field / f"{rec['id']}.png"
            if not src.exists() and not chain:
                break
            current = src if src.exists() else chain[-1]
            chain.append(current)
            target = out / f"r{k}" / f"{rec['id']}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(current, target)
        if chain:
            chains[rec["id"]] = chain
    return chains


async def main_async(args: argparse.Namespace) -> None:
    run_dir = Path(args.run)
    summary = load_json(run_dir / "results.json")
    records = summary["items"]
    n_rounds = int(summary["settings"]["critic_rounds"])
    chains = build_round_dirs(run_dir, records, n_rounds)
    by_id = {r["id"]: r for r in records}
    client = make_client()
    usage = UsageLog()
    usage.attach(client)
    sem = asyncio.Semaphore(args.concurrency)
    sel_dir = run_dir / "rounds" / "selected"
    sel_dir.mkdir(parents=True, exist_ok=True)

    async def one(item_id: str, chain: list[Path]) -> dict:
        # 去重：提前结束时后几轮是同一张图
        unique: list[tuple[int, Path]] = []
        for k, p in enumerate(chain):
            if not unique or p != unique[-1][1]:
                unique.append((k, p))
        order = list(range(len(unique)))
        random.Random(f"{args.seed}-{item_id}").shuffle(order)
        labels = [chr(ord("A") + j) for j in range(len(order))]
        rec = by_id[item_id]
        parts: list = [f"Methodology (truncated): {str(rec['content'])[:10000]}\nCaption: {rec['visual_intent']}\n"]
        for label, j in zip(labels, order):
            parts += [f"\nCandidate {label}:", image_part(unique[j][1], 1536)]
        parts.append("\n" + SELECT_PROMPT.format(labels=", ".join(labels)))
        CALL_TAG.set("select")
        if len(unique) == 1:
            pick = 0
            obj = {"reason": "only one distinct image"}
        else:
            async with sem:
                obj = parse_json(await gen_text(client, args.model, parts, json_mode=True))
            obj = obj if isinstance(obj, dict) else {}
            label = str(obj.get("best", "")).strip().upper()[:1]
            pick = order[labels.index(label)] if label in labels else len(unique) - 1
        round_idx, path = unique[pick]
        shutil.copyfile(path, sel_dir / f"{item_id}.png")
        return {"id": item_id, "distinct_rounds": [k for k, _ in unique], "picked_round": round_idx, "reason": obj.get("reason", "")}

    rows = await asyncio.gather(*(one(i, c) for i, c in chains.items()))
    save_json(run_dir / "rounds" / "selected.json", {"model": args.model, "rows": rows, "calls": usage.calls})
    from collections import Counter

    print(f"轮次目录：{run_dir / 'rounds'}（r0…r{n_rounds}，selected）")
    print("选择器挑中的轮次：", dict(sorted(Counter(r["picked_round"] for r in rows).items())))
    print("不同图片的轮数分布：", dict(sorted(Counter(len(r["distinct_rounds"]) for r in rows).items())))
    print_usage(usage.calls)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", default=SELECT_MODEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
