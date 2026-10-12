"""Critic 检错实验：在第 0 轮成图上人为制造已知错误，看各版 Critic 能否指出；并在无错图上核查 Critic 提出的内容问题是否属实。

输入为 pipeline_measure 的输出目录（用其中 target_diagram_desc0 的图与描述）。分三步，每步结果都存进 --work 目录，可以分开重跑：
  python -m scripts.eval.critic_detect inject   --run results/eval/critic_rounds --work results/eval/critic_detect
  python -m scripts.eval.critic_detect critique --work results/eval/critic_detect --variants base-3.8,check-3.8,base-pro,base-lite
  python -m scripts.eval.critic_detect judge    --work results/eval/critic_detect

四类错误：typo（标签拼错）、wrong_label（模块名换成与方法矛盾的词）、missing（删掉一个关键组件）、wrong_arrow（一条关键箭头反向）。
每条输入制造两类，四类各约 n/2 张。制造方法：Pro 看图选定要改的元素并写编辑指令 → 生图模型按指令改图 → Pro 对比前后确认只改了这一处。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.eval.common import (
    CALL_TAG,
    REPO,
    UsageLog,
    gen_text,
    image_part,
    load_api_key,
    load_bench,
    load_json,
    make_client,
    parse_json,
    print_usage,
    save_json,
)

ERROR_TYPES = ("typo", "wrong_label", "missing", "wrong_arrow")
ROUND0_FIELD = "target_diagram_desc0"
PLAN_MODEL = "gemini-3.1-pro-preview"
EDIT_MODEL = "gemini-nano-banana-2.1"
JUDGE_MODEL = "gemini-3-flash-preview"
CLAIM_JUDGE_MODEL = "gemini-3.1-pro-preview"

ERROR_RULES = {
    "typo": "Misspell ONE important visible label (a module or component name) by swapping, dropping, or doubling one or "
            "two letters, so it is still readable but clearly misspelled (e.g. 'Retriever' -> 'Retreiver').",
    "wrong_label": "Replace the text of ONE module label with a plausible but WRONG technical term that contradicts the "
                   "methodology (e.g. 'Encoder' -> 'Decoder', 'Cross-Attention' -> 'Max Pooling'). Spell it correctly.",
    "missing": "Remove ONE key component that the methodology requires (its box, icon and label, plus arrows that only "
               "connect to it), filling the area with background so the removal looks natural.",
    "wrong_arrow": "Reverse the direction of ONE key arrow (swap which end has the arrowhead) so the data flow contradicts "
                   "the methodology. Keep everything else, including the arrow's position.",
}

PLAN_PROMPT = (
    "You are preparing a controlled test for an automatic figure reviewer. Below is a generated method diagram and the "
    "methodology it should depict. Choose ONE concrete element in the image to alter so that the figure contains exactly "
    "one error of this type:\n{rule}\n\nPick an element that is clearly visible and important to the method. "
    "Return JSON: {{\"target\": \"the exact visible text or a precise location of the element\", "
    "\"edit_instruction\": \"a precise instruction for an image editor, naming the element and its location\", "
    "\"error_description\": \"one sentence describing what is now wrong compared with the methodology\"}}"
)

EDIT_PROMPT = (
    "Edit this figure. {instruction}\nChange nothing else: keep the same layout, colors, fonts, all other text, and "
    "overall style exactly as they are."
)

VERIFY_PROMPT = (
    "Image 1 is the original figure, Image 2 is an edited version. The intended edit was: {instruction}\n"
    "Answer strictly: was the intended edit clearly applied, and were there any OTHER noticeable changes (text changed, "
    "components added/removed, layout changed)? Return JSON: {{\"applied\": true|false, \"other_changes\": \"none\" or a "
    "short description, \"other_changes_major\": true|false}}"
)

DETECT_PROMPT = (
    "An automatic reviewer critiqued a figure. The figure is known to contain this specific error:\n{error}\n\n"
    "Reviewer critique:\n{critique}\n\n"
    "Did the critique identify THIS error (the same element and the same kind of problem)? A generic remark such as "
    "'check all labels' does not count. Return JSON {{\"detected\": true|false, \"evidence\": \"quote or empty\"}}"
)

CLAIM_PROMPT = (
    "Methodology (truncated): {content}\nCaption: {caption}\n\nBelow is a generated figure and an automatic reviewer's "
    "critique of it.\nCritique:\n{critique}\n\n"
    "List every CONTENT claim in the critique: claims that something in the figure is misspelled, missing, wrong, "
    "hallucinated, mislabeled, or connected/ordered incorrectly. Skip pure style or layout suggestions. For each claim, "
    "check the figure and the methodology and decide whether the claimed problem really exists. Return JSON "
    "{{\"claims\": [{{\"claim\": \"short\", \"kind\": \"typo|missing|wrong|hallucinated|connection|other\", "
    "\"real\": true|false|null}}]}} (null = cannot tell)."
)

CHECKLIST_PROCEDURE = """
## PROCEDURE (follow before writing your critique)
1. From the Methodology Section and Figure Caption, list every key component, module label and connection the diagram must show.
2. For each item, check the diagram: is it present, labelled correctly, and are its arrows pointing in the right direction?
3. Read every visible text label in the diagram letter by letter and note any misspelling.
4. In "critic_suggestions", report every failure from steps 2-3 first (name the exact element), then presentation issues.
"""

VARIANTS = {
    "base-3.8": {"model": "gemini-3.8-flash", "checklist": False},
    "check-3.8": {"model": "gemini-3.8-flash", "checklist": True},
    "base-pro": {"model": "gemini-3.1-pro-preview", "checklist": False},
    "base-lite": {"model": "gemini-3.1-flash-lite", "checklist": False},
}


def image_from_response(response) -> bytes | None:
    for cand in getattr(response, "candidates", None) or []:
        for part in getattr(getattr(cand, "content", None), "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and inline.data:
                return inline.data
    return None


def png_bytes(raw: bytes) -> bytes:
    buf = io.BytesIO()
    Image.open(io.BytesIO(raw)).save(buf, format="PNG")
    return buf.getvalue()


# ---------- inject ----------


async def inject(args: argparse.Namespace) -> None:
    from google.genai import types

    test, _ = load_bench()
    run_dir, work = Path(args.run), Path(args.work)
    records = [r for r in load_json(run_dir / "results.json")["items"]
               if r["id"] in test and (run_dir / ROUND0_FIELD / f"{r['id']}.png").exists()]
    client = make_client()
    usage = UsageLog()
    usage.attach(client)
    sem = asyncio.Semaphore(args.concurrency)
    img_dir = work / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    async def one(idx: int, rec: dict, etype: str) -> dict:
        item = test[rec["id"]]
        src = run_dir / ROUND0_FIELD / f"{rec['id']}.png"
        ratio = (item.get("additional_info") or {}).get("rounded_ratio", "16:9")
        row: dict[str, Any] = {"id": rec["id"], "type": etype, "status": "failed"}
        for attempt in range(2):
            CALL_TAG.set("inject_plan")
            parts = [f"Methodology (truncated): {str(item['content'])[:8000]}\nCaption: {item['visual_intent']}\n\nDiagram:",
                     image_part(src, 1536), PLAN_PROMPT.format(rule=ERROR_RULES[etype])]
            async with sem:
                plan = parse_json(await gen_text(client, PLAN_MODEL, parts, temperature=0.4 * attempt, json_mode=True))
            if not isinstance(plan, dict) or not plan.get("edit_instruction"):
                continue
            CALL_TAG.set("inject_edit")
            config = types.GenerateContentConfig(response_modalities=["IMAGE"],
                                                 image_config=types.ImageConfig(aspect_ratio=ratio, image_size="2K"))
            raw = None
            async with sem:
                for _ in range(3):
                    try:
                        resp = await client.aio.models.generate_content(
                            model=EDIT_MODEL, config=config,
                            contents=[types.Part.from_bytes(data=src.read_bytes(), mime_type="image/png"),
                                      EDIT_PROMPT.format(instruction=plan["edit_instruction"])])
                        raw = image_from_response(resp)
                        if raw:
                            break
                    except Exception as exc:  # noqa: BLE001
                        row["error"] = str(exc)[:200]
                        await asyncio.sleep(5)
            if not raw:
                continue
            out = img_dir / f"{rec['id']}__{etype}.png"
            out.write_bytes(png_bytes(raw))
            CALL_TAG.set("inject_verify")
            parts = ["Image 1:", image_part(src, 1536), "Image 2:", image_part(out, 1536),
                     VERIFY_PROMPT.format(instruction=plan["edit_instruction"])]
            async with sem:
                check = parse_json(await gen_text(client, PLAN_MODEL, parts, json_mode=True))
            check = check if isinstance(check, dict) else {}
            row.update(plan=plan, verify=check, attempt=attempt, image=str(out.relative_to(work)))
            if check.get("applied") is True and check.get("other_changes_major") is not True:
                row["status"] = "ok"
                break
            out.unlink(missing_ok=True)
        print(f"  {rec['id']} {etype}: {row['status']}", flush=True)
        return row

    jobs = [one(i, rec, ERROR_TYPES[(i + k) % 4]) for i, rec in enumerate(records) for k in (0, 2)]
    rows = await asyncio.gather(*jobs)
    save_json(work / "injected.json", {"run": str(run_dir), "rows": rows, "calls": usage.calls})
    by_type = defaultdict(lambda: [0, 0])
    for r in rows:
        by_type[r["type"]][0] += r["status"] == "ok"
        by_type[r["type"]][1] += 1
    print("制造成功 / 尝试：", {k: f"{a}/{b}" for k, (a, b) in sorted(by_type.items())})
    print_usage(usage.calls)


# ---------- critique ----------


async def critique(args: argparse.Namespace) -> None:
    from agents.critic_agent import DIAGRAM_CRITIC_AGENT_SYSTEM_PROMPT, CriticAgent
    from utils import generation_utils
    from utils.config import ExpConfig

    work = Path(args.work)
    injected = load_json(work / "injected.json")
    run_dir = Path(injected["run"])
    records = {r["id"]: r for r in load_json(run_dir / "results.json")["items"]}
    cases = [{"case": f"{r['id']}__{r['type']}", "id": r["id"], "type": r["type"], "image": work / r["image"]}
             for r in injected["rows"] if r["status"] == "ok"]
    clean_ids = sorted({r["id"] for r in injected["rows"]})
    cases += [{"case": f"{i}__clean", "id": i, "type": "clean", "image": run_dir / ROUND0_FIELD / f"{i}.png"} for i in clean_ids]

    generation_utils.init_gemini_client(load_api_key())
    usage = UsageLog()
    usage.attach(generation_utils.get_gemini_client())
    variants = [v for v in args.variants.split(",") if v]
    agents = {}
    for name in variants:
        spec = VARIANTS[name]
        agent = CriticAgent(exp_config=ExpConfig(dataset_name="PaperBananaBench", task_name="diagram",
                                                 exp_mode="demo_planner_critic", provider="gemini",
                                                 model_name=spec["model"], work_dir=REPO))
        if spec["checklist"]:
            agent.system_prompt = DIAGRAM_CRITIC_AGENT_SYSTEM_PROMPT.replace("## INPUT DATA", CHECKLIST_PROCEDURE + "\n## INPUT DATA")
        agents[name] = agent

    out_path = work / "critiques.json"
    done = {(r["case"], r["variant"], r["rep"]): r for r in (load_json(out_path)["rows"] if out_path.exists() else [])}
    sem = asyncio.Semaphore(args.concurrency)

    async def one(case: dict, variant: str, rep: int) -> dict:
        key = (case["case"], variant, rep)
        if key in done:
            return done[key]
        rec = records[case["id"]]
        data = {
            "candidate_id": f"detect-{case['case']}-{variant}-{rep}", "content": rec["content"],
            "visual_intent": rec["visual_intent"], "current_critic_round": 0,
            ROUND0_FIELD: rec["descriptions"][ROUND0_FIELD],
            f"{ROUND0_FIELD}_base64_jpg": base64.b64encode(Path(case["image"]).read_bytes()).decode(),
            f"{ROUND0_FIELD}_mime_type": "image/png",
        }
        CALL_TAG.set(f"critic:{variant}")
        async with sem:
            out = await agents[variant].process(data, source="planner")
        return {"case": case["case"], "id": case["id"], "type": case["type"], "variant": variant, "rep": rep,
                "status": out.get("target_diagram_critic_status0"),
                "suggestions": str(out.get("target_diagram_critic_suggestions0", ""))}

    rows = await asyncio.gather(*(one(c, v, k) for c in cases for v in variants for k in range(args.reps)))
    prev_calls = load_json(out_path).get("calls", []) if out_path.exists() else []
    save_json(out_path, {"rows": list({(r["case"], r["variant"], r["rep"]): r for r in [*done.values(), *rows]}.values()),
                         "calls": prev_calls + usage.calls})
    print(f"评审 {len(rows)} 次（{len(cases)} 张图 × {len(variants)} 种 × {args.reps} 次）")
    print_usage(usage.calls)


# ---------- judge ----------


async def judge(args: argparse.Namespace) -> None:
    test, _ = load_bench()
    work = Path(args.work)
    injected = load_json(work / "injected.json")
    run_dir = Path(injected["run"])
    errors = {f"{r['id']}__{r['type']}": r["plan"]["error_description"] for r in injected["rows"] if r["status"] == "ok"}
    rows = load_json(work / "critiques.json")["rows"]
    out_path = work / "judged.json"
    done = {(r["case"], r["variant"], r["rep"]): r for r in (load_json(out_path)["rows"] if out_path.exists() else [])}
    client = make_client()
    usage = UsageLog()
    usage.attach(client)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(row: dict) -> dict:
        key = (row["case"], row["variant"], row["rep"])
        if key in done and done[key].get("suggestions") == row["suggestions"]:
            return done[key]
        row = dict(row)
        sugg = row["suggestions"].strip()
        if row["type"] != "clean":
            CALL_TAG.set("judge_detect")
            async with sem:
                obj = parse_json(await gen_text(client, JUDGE_MODEL, [DETECT_PROMPT.format(error=errors[row["case"]], critique=sugg[:6000])],
                                                json_mode=True))
            row["detected"] = bool(isinstance(obj, dict) and obj.get("detected") is True)
            row["evidence"] = (obj or {}).get("evidence", "") if isinstance(obj, dict) else ""
        elif not sugg.lower().startswith("no changes needed"):
            item = test[row["id"]]
            CALL_TAG.set("judge_claims")
            parts = [CLAIM_PROMPT.format(content=str(item["content"])[:8000], caption=item["visual_intent"], critique=sugg[:6000]),
                     "Figure:", image_part(run_dir / ROUND0_FIELD / f"{row['id']}.png", 1536)]
            async with sem:
                obj = parse_json(await gen_text(client, CLAIM_JUDGE_MODEL, parts, json_mode=True))
            row["claims"] = (obj or {}).get("claims", []) if isinstance(obj, dict) else []
        else:
            row["claims"] = []
        return row

    judged = await asyncio.gather(*(one(r) for r in rows))
    prev_calls = load_json(out_path).get("calls", []) if out_path.exists() else []
    save_json(out_path, {"rows": judged, "calls": prev_calls + usage.calls})
    print_usage(usage.calls)
    report_rows(judged)


def report_rows(rows: list[dict]) -> None:
    variants = sorted({r["variant"] for r in rows})
    print("\n检出率（检出 / 张数）")
    print("  " + "".join(f"{t:>14s}" for t in (*ERROR_TYPES, "合计", "无需修改")))
    for v in variants:
        mine = [r for r in rows if r["variant"] == v and r["type"] != "clean"]
        cells = []
        for t in ERROR_TYPES:
            sub = [r for r in mine if r["type"] == t]
            cells.append(f"{sum(r['detected'] for r in sub)}/{len(sub)}")
        cells.append(f"{sum(r['detected'] for r in mine)}/{len(mine)}")
        nc = sum(1 for r in rows if r["variant"] == v and r["suggestions"].strip().lower().startswith("no changes needed"))
        cells.append(f"{nc}/{sum(1 for r in rows if r['variant'] == v)}")
        print(f"  {v:10s}" + "".join(f"{c:>14s}" for c in cells))
    print("\n无错图上的内容类意见（属实 / 不属实 / 无法判断；平均每张）")
    for v in variants:
        clean = [r for r in rows if r["variant"] == v and r["type"] == "clean"]
        claims = [c for r in clean for c in r.get("claims", [])]
        real = sum(1 for c in claims if c.get("real") is True)
        false = sum(1 for c in claims if c.get("real") is False)
        unk = len(claims) - real - false
        print(f"  {v:10s} {real}/{false}/{unk}  每张 {len(claims) / max(1, len(clean)):.1f} 条，"
              f"不属实占 {false / max(1, real + false):.0%}")


def report(args: argparse.Namespace) -> None:
    report_rows(load_json(Path(args.work) / "judged.json")["rows"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inject")
    p.add_argument("--run", required=True, help="pipeline_measure 的输出目录")
    p.add_argument("--work", required=True)
    p.add_argument("--concurrency", type=int, default=6)
    p = sub.add_parser("critique")
    p.add_argument("--work", required=True)
    p.add_argument("--variants", default="base-3.8,check-3.8,base-pro,base-lite")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=8)
    p = sub.add_parser("judge")
    p.add_argument("--work", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p = sub.add_parser("report")
    p.add_argument("--work", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "report":
        report(args)
    else:
        asyncio.run({"inject": inject, "critique": critique, "judge": judge}[args.cmd](args))


if __name__ == "__main__":
    main()
