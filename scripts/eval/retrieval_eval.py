"""参考检索评估：用固定查询集比较检索方案返回的 10 条参考有多大用处（0–3 分模板有用度）。

  report  离线汇总已有结果（不调用 API）
  run     用仓库 RetrieverAgent 跑一个新方案，只给新出现的参考补评分，结果另存为新文件

示例：
  python -m scripts.eval.retrieval_eval report
  python -m scripts.eval.retrieval_eval run --arm flash-3.8-thumbs-rerun --model gemini-3.8-flash
"""

from __future__ import annotations

import argparse
import asyncio
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval.common import (
    BASELINE_DIR,
    CALL_TAG,
    OUTPUT_DIR,
    REPO,
    UsageLog,
    load_json,
    make_client,
    mean_score,
    paired_bootstrap_ci,
    print_usage,
    resolve_image,
    save_json,
    timestamp,
)

DEFAULT_BASELINE = BASELINE_DIR / "retrieval_40.json"
UPPER_BOUND_ARM = "best-judged-10"


def best_judged(item: dict[str, Any], k: int = 10) -> list[str]:
    """已评分参考里分数最高的 k 条（参考池在该评审下的上限，不是可部署的方案）。"""
    scored = [(v, rid) for rid, v in item["judge_scores"].items() if v is not None]
    scored.sort(key=lambda e: (-e[0], e[1]))
    return [rid for _, rid in scored[:k]]


def arm_refs(item: dict[str, Any], arm: str) -> list[str]:
    if arm == UPPER_BOUND_ARM:
        return best_judged(item)
    return list((item["arms"].get(arm) or {}).get("refs") or [])


def per_query(data: dict[str, Any], arm: str, k: int) -> np.ndarray:
    return np.array([mean_score(arm_refs(it, arm), it["judge_scores"], k) for it in data["items"]])


def summarize_arm(data: dict[str, Any], arm: str) -> dict[str, Any]:
    items = data["items"]
    scores10 = [it["judge_scores"].get(x) for it in items for x in arm_refs(it, arm)[:10]]
    scores10 = np.array([s for s in scores10 if s is not None], dtype=float)
    metas = [(it["arms"].get(arm) or {}).get("meta") or {} for it in items]
    secs = [(it["arms"].get(arm) or {}).get("secs") for it in items]
    secs = [s for s in secs if s is not None]
    return {
        "arm": arm,
        "at3": float(per_query(data, arm, 3).mean()),
        "at10": float(per_query(data, arm, 10).mean()),
        "share_ge2": float((scores10 >= 2).mean()) if len(scores10) else 0.0,
        "secs": float(np.mean(secs)) if secs else None,
        "invalid_id_queries": sum(1 for m in metas if m.get("invalid_ids")),
        "topped_up_queries": sum(1 for m in metas if m.get("topped_up")),
        "has3_queries": sum(1 for it in items if any((it["judge_scores"].get(x) or 0) >= 3 for x in arm_refs(it, arm)[:10])),
    }


def available_arms(data: dict[str, Any]) -> list[str]:
    arms: list[str] = list((data.get("meta") or {}).get("arms") or {})
    for it in data["items"]:
        for name in it["arms"]:
            if name not in arms:
                arms.append(name)
    return arms


def print_report(data: dict[str, Any], arms: list[str] | None = None, reference_arm: str | None = None) -> None:
    arms = arms or available_arms(data)
    n = len(data["items"])
    print(f"查询 {n} 条；评审 {data['meta'].get('judge_model')}；口径 {data['meta'].get('rubric')}")
    print("方案 | 平均分@3 | 平均分@10 | ≥2 分占比@10 | 含 3 分参考的查询 | 平均耗时 | 有无效 ID | 需补齐")
    for arm in [*arms, UPPER_BOUND_ARM]:
        s = summarize_arm(data, arm)
        secs = f"{s['secs']:.1f}s" if s["secs"] is not None else "-"
        label = f"{arm}（上限）" if arm == UPPER_BOUND_ARM else arm
        print(f"{label} | {s['at3']:.2f} | {s['at10']:.2f} | {s['share_ge2']:.1%} | {s['has3_queries']}/{n} | {secs} | "
              f"{s['invalid_id_queries']} | {s['topped_up_queries']}")
    reference_arm = reference_arm or (data.get("meta") or {}).get("default_arm")
    if reference_arm and reference_arm in arms:
        print(f"\n逐条配对差值（对照 {reference_arm}，95% 自助法置信区间）")
        for arm in arms:
            if arm == reference_arm:
                continue
            row = []
            for k in (3, 10):
                d, lo, hi = paired_bootstrap_ci(per_query(data, arm, k), per_query(data, reference_arm, k))
                row.append(f"@{k} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            print(f"  {arm}: " + "；".join(row))


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


async def run_arm(args: argparse.Namespace) -> Path:
    from agents.retriever_agent import RetrieverAgent
    from scripts.eval.common import load_api_key, load_bench
    from scripts.eval.judges import USEFULNESS_JUDGE_MODEL, judge_usefulness
    from utils import generation_utils
    from utils.config import ExpConfig

    data = load_json(Path(args.baseline))
    if args.arm in available_arms(data):
        raise SystemExit(f"方案名 {args.arm} 已存在，请换一个名字。")
    if data["meta"].get("judge_model") != USEFULNESS_JUDGE_MODEL:
        print(f"注意：基准的评审模型是 {data['meta'].get('judge_model')}，新参考将用 {USEFULNESS_JUDGE_MODEL} 评分。")
    test, ref = load_bench()
    items = data["items"][: args.limit] if args.limit else data["items"]

    generation_utils.init_gemini_client(load_api_key())
    usage = UsageLog()
    usage.attach(generation_utils.get_gemini_client())
    judge_client = make_client()
    usage.attach(judge_client)

    agent = RetrieverAgent(exp_config=ExpConfig(
        dataset_name="PaperBananaBench", task_name="diagram", exp_mode="demo_planner_critic",
        provider="gemini", model_name=args.model, work_dir=REPO))
    agent.task_config["visual_rerank"] = not args.no_thumbnails
    retrieval_sem = asyncio.Semaphore(args.concurrency)
    judge_sem = asyncio.Semaphore(args.concurrency)
    rng = random.Random(args.seed)

    async def one(item: dict[str, Any]) -> None:
        query = test[item["id"]]
        async with retrieval_sem:
            CALL_TAG.set("retrieval")
            started = time.time()
            out = await agent.process({"candidate_id": query["id"], "content": query["content"],
                                       "visual_intent": query["visual_intent"]}, retrieval_setting="auto")
        item["arms"][args.arm] = {"refs": out.get("top10_references") or [], "meta": out.get("retrieval_meta") or {},
                                  "secs": round(time.time() - started, 2)}
        missing = [x for x in item["arms"][args.arm]["refs"] if x not in item["judge_scores"]]
        rng.shuffle(missing)
        target_image = resolve_image(query["path_to_gt_image"])
        CALL_TAG.set("judge")
        for start in range(0, len(missing), 10):
            batch = [(rid, resolve_image(ref[rid]["path_to_gt_image"])) for rid in missing[start:start + 10] if rid in ref]
            batch = [(rid, p) for rid, p in batch if p is not None]
            if batch and target_image is not None:
                item["judge_scores"].update(await judge_usefulness(judge_client, query, target_image, batch, judge_sem))

    started = time.time()
    done = 0
    for start in range(0, len(items), 8):
        await asyncio.gather(*(one(it) for it in items[start:start + 8]))
        done += len(items[start:start + 8])
        print(f"完成 {done}/{len(items)}（{time.time() - started:.0f}s）", flush=True)

    data["items"] = items
    data["meta"].setdefault("arms", {})[args.arm] = {
        "model": args.model, "thumbnails": not args.no_thumbnails, "created": time.strftime("%Y-%m-%d"),
        "commit": git_commit(), "note": args.note or "",
    }
    out_path = Path(args.out) if args.out else OUTPUT_DIR / f"retrieval_{args.arm}_{timestamp()}.json"
    save_json(out_path, data)
    print(f"\n结果已保存：{out_path}")
    print_report(data, reference_arm=args.arm)
    print("\n用量：")
    print_usage(usage.calls)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    rep = sub.add_parser("report", help="离线汇总已有结果")
    rep.add_argument("--file", default=str(DEFAULT_BASELINE))
    rep.add_argument("--arms", default="", help="逗号分隔；默认全部")
    rep.add_argument("--reference-arm", default="", help="配对差值的对照方案；默认用文件里的 default_arm")

    run = sub.add_parser("run", help="跑一个新方案并补评分（需要 GOOGLE_API_KEY）")
    run.add_argument("--arm", required=True, help="新方案名，不能与已有方案重名")
    run.add_argument("--model", required=True, help="检索用的文本模型")
    run.add_argument("--no-thumbnails", action="store_true", help="只看 caption 挑选")
    run.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    run.add_argument("--limit", type=int, default=0, help="只跑前 N 条，用于试跑")
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--seed", type=int, default=500)
    run.add_argument("--note", default="")
    run.add_argument("--out", default="")

    args = parser.parse_args(argv)
    if args.command == "report":
        data = load_json(Path(args.file))
        arms = [a for a in args.arms.split(",") if a] or None
        print_report(data, arms, args.reference_arm or None)
    else:
        asyncio.run(run_arm(args))


if __name__ == "__main__":
    main()
