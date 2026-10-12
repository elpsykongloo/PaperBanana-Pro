"""内容清单覆盖率：先从方法正文与图注抽出图里必须出现的组件与连接（每条一份，缓存），再逐张图核对每一项。

与成对盲评互补：盲评只说"哪张更好"，覆盖率给出每张图的绝对分（正确 / 缺失 / 有误、拼写错误数）。
  python -m scripts.eval.coverage_eval --dirs results/eval/critic_rounds/rounds/r0,results/eval/critic_rounds/rounds/r3
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np

from scripts.eval.common import (
    CALL_TAG,
    OUTPUT_DIR,
    UsageLog,
    gen_text,
    image_part,
    load_bench,
    load_json,
    make_client,
    paired_bootstrap_ci,
    parse_json,
    print_usage,
    save_json,
    timestamp,
)

CHECKLIST_MODEL = "gemini-3.1-pro-preview"
CHECK_MODEL = "gemini-3.1-pro-preview"
CHECKLIST_CACHE = OUTPUT_DIR / "cache" / "checklists.json"

CHECKLIST_PROMPT = (
    "Methodology: {content}\nFigure caption: {caption}\n\n"
    "List the 6-15 items that a faithful method diagram for this caption MUST show: key components/modules (with the "
    "name a reader expects) and key connections or data flows between them. Only include items clearly required by the "
    "text and caption; skip minor details. Return JSON {{\"items\": [{{\"id\": 1, \"kind\": \"component|connection\", "
    "\"text\": \"...\"}}]}}"
)

CHECK_PROMPT = (
    "Checklist of items a faithful diagram must show:\n{items}\n\n"
    "For EACH item, look at the figure and answer: \"ok\" (present and correct), \"missing\", or \"wrong\" (present but "
    "mislabeled, misconnected or misrepresented). Then list every misspelled word visible in the figure. Return JSON "
    "{{\"checks\": {{\"1\": \"ok\"|\"missing\"|\"wrong\", ...}}, \"misspellings\": [\"...\"]}}"
)


async def get_checklists(client, test: dict, ids: list[str], sem: asyncio.Semaphore) -> dict[str, list[dict]]:
    cache = load_json(CHECKLIST_CACHE) if CHECKLIST_CACHE.exists() else {}

    async def one(item_id: str):
        if item_id in cache:
            return
        item = test[item_id]
        CALL_TAG.set("checklist")
        async with sem:
            obj = parse_json(await gen_text(client, CHECKLIST_MODEL, [CHECKLIST_PROMPT.format(
                content=str(item["content"])[:15000], caption=item["visual_intent"])], json_mode=True))
        items = (obj or {}).get("items", []) if isinstance(obj, dict) else []
        cache[item_id] = [{"id": str(i + 1), "kind": x.get("kind", ""), "text": x.get("text", "")}
                          for i, x in enumerate(items) if isinstance(x, dict)]

    await asyncio.gather(*(one(i) for i in ids))
    save_json(CHECKLIST_CACHE, cache)
    return cache


async def check_image(client, items: list[dict], image: Path, sem: asyncio.Semaphore) -> dict:
    listing = "\n".join(f"{x['id']}. [{x['kind']}] {x['text']}" for x in items)
    CALL_TAG.set("coverage")
    async with sem:
        obj = parse_json(await gen_text(client, CHECK_MODEL, ["Figure:", image_part(image, 1536),
                                                              CHECK_PROMPT.format(items=listing)], json_mode=True))
    obj = obj if isinstance(obj, dict) else {}
    checks = {str(k): str(v).lower() for k, v in (obj.get("checks") or {}).items()}
    marks = [checks.get(x["id"], "missing") for x in items]
    return {
        "ok": marks.count("ok"), "missing": marks.count("missing"), "wrong": marks.count("wrong"), "total": len(items),
        "coverage": marks.count("ok") / max(1, len(items)), "misspellings": list(obj.get("misspellings") or []),
        "checks": checks,
    }


async def main_async(args: argparse.Namespace) -> None:
    test, _ = load_bench()
    dirs = [Path(d) for d in args.dirs.split(",") if d]
    ids = sorted(set.intersection(*({p.stem for p in d.glob("*.png")} for d in dirs)) & set(test))
    if not ids:
        raise SystemExit("这些目录没有共同的测试集图片。")
    client = make_client()
    usage = UsageLog()
    usage.attach(client)
    sem = asyncio.Semaphore(args.concurrency)
    checklists = await get_checklists(client, test, ids, sem)
    results: dict[str, dict[str, dict]] = {}
    for d in dirs:
        rows = await asyncio.gather(*(check_image(client, checklists[i], d / f"{i}.png", sem) for i in ids))
        results[str(d)] = dict(zip(ids, rows))

    print(f"{len(ids)} 条，清单平均 {np.mean([len(checklists[i]) for i in ids]):.1f} 项")
    print("目录 | 覆盖率均值 | 缺失/有误（每图） | 有拼写错误的图")
    for d, rows in results.items():
        cov = [rows[i]["coverage"] for i in ids]
        print(f"  {Path(d).name:16s} {np.mean(cov):.3f}   {np.mean([rows[i]['missing'] for i in ids]):.2f}/"
              f"{np.mean([rows[i]['wrong'] for i in ids]):.2f}   {sum(1 for i in ids if rows[i]['misspellings'])}/{len(ids)}")
    base = str(dirs[0])
    for d in dirs[1:]:
        m, lo, hi = paired_bootstrap_ci([results[str(d)][i]["coverage"] for i in ids], [results[base][i]["coverage"] for i in ids])
        print(f"  {Path(d).name} − {Path(base).name}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    out = Path(args.out) if args.out else OUTPUT_DIR / f"coverage_{timestamp()}.json"
    save_json(out, {"dirs": [str(d) for d in dirs], "results": results, "calls": usage.calls})
    print_usage(usage.calls)
    print(f"结果：{out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dirs", required=True, help="逗号分隔的图片目录；第一个作为对照")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", default="")
    asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
