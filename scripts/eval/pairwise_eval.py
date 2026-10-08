"""成图成对盲评：比较两个目录中同名图片（<id>.png），评审能看到原图；两种顺序各评一次，两次都胜才算胜。

常与 pipeline_measure 配合，例如比较第 0 轮（Planner 图）与最后一轮 Critic 图：
  python -m scripts.eval.pairwise_eval --x results/eval/critic_rounds/target_diagram_critic_desc2 \
      --y results/eval/critic_rounds/target_diagram_desc0 --leak-check

只有 PaperBananaBench 测试集里的 id 会被评审（需要原图作参照）。
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter
from pathlib import Path

from scripts.eval.common import (
    CALL_TAG,
    OUTPUT_DIR,
    UsageLog,
    load_bench,
    make_client,
    print_usage,
    resolve_image,
    save_json,
    timestamp,
)
from scripts.eval.judges import PAIRWISE_DIMS, leak_check, pairwise_both_orders


def summarize(rows: list[dict], dims=PAIRWISE_DIMS) -> dict[str, dict[str, int]]:
    return {dim: dict(Counter(r["verdict"][dim] for r in rows)) for dim in dims}


def format_summary(summary: dict[str, dict[str, int]]) -> str:
    return "  ".join(f"{dim}={c.get('win', 0)}/{c.get('tie', 0)}/{c.get('loss', 0)}" for dim, c in summary.items())


async def main_async(args: argparse.Namespace) -> Path:
    test, _ = load_bench()
    dir_x, dir_y = Path(args.x), Path(args.y)
    ids = sorted({p.stem for p in dir_x.glob("*.png")} & {p.stem for p in dir_y.glob("*.png")} & set(test))
    if args.ids:
        ids = [i for i in args.ids.split(",") if i in ids]
    if not ids:
        raise SystemExit("两个目录没有共同的、属于测试集的图片。")
    client = make_client()
    usage = UsageLog()
    usage.attach(client)
    semaphore = asyncio.Semaphore(args.concurrency)
    print(f"比较 {len(ids)} 条：X={dir_x} 对 Y={dir_y}", flush=True)

    async def one(item_id: str) -> dict:
        item = test[item_id]
        gt = resolve_image(item["path_to_gt_image"])
        CALL_TAG.set("pairwise")
        verdict = await pairwise_both_orders(client, item, gt, dir_x / f"{item_id}.png", dir_y / f"{item_id}.png", semaphore)
        row = {"id": item_id, "verdict": verdict}
        if args.leak_check:
            CALL_TAG.set("leak_check")
            lx, ly = await asyncio.gather(leak_check(client, item, dir_x / f"{item_id}.png", semaphore),
                                          leak_check(client, item, dir_y / f"{item_id}.png", semaphore))
            row["leak"] = {"x": lx.get("leaked_text", []), "y": ly.get("leaked_text", [])}
        return row

    started = time.time()
    rows = await asyncio.gather(*(one(i) for i in ids))
    summary = summarize(rows)
    out_path = Path(args.out) if args.out else OUTPUT_DIR / f"pairwise_{timestamp()}.json"
    save_json(out_path, {"x": str(dir_x), "y": str(dir_y), "summary": summary, "rows": rows, "calls": usage.calls})

    print(f"\nX 对 Y（胜/平/负，{time.time() - started:.0f}s）：{format_summary(summary)}")
    if args.leak_check:
        for side in ("x", "y"):
            counts = [len(r["leak"][side]) for r in rows]
            print(f"  {side.upper()} 有非内容文字的图 {sum(1 for c in counts if c)}/{len(counts)}，平均每图 {sum(counts) / len(counts):.2f} 处")
    print("用量：")
    print_usage(usage.calls)
    print(f"结果：{out_path}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--x", required=True, help="被测方图片目录")
    parser.add_argument("--y", required=True, help="对照方图片目录")
    parser.add_argument("--ids", default="", help="只评这些 id（逗号分隔）")
    parser.add_argument("--leak-check", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", default="")
    asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
