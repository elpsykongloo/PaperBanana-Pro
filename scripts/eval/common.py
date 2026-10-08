"""评估工具的公共部分：路径、API Key、数据集、图片、Gemini 调用、统计与用量计费。

API Key 只从环境变量 GOOGLE_API_KEY 读取，不写入任何文件，也不打印。
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BASELINE_DIR = Path(__file__).resolve().parent / "baselines"
OUTPUT_DIR = REPO / "results" / "eval"
CACHE_DIR = OUTPUT_DIR / "cache"
DIAGRAM_DIR = REPO / "data" / "PaperBananaBench" / "diagram"
API_KEY_ENV = "GOOGLE_API_KEY"

# 每百万 token 的美元标价（Gemini API 标准付费档，2026-10）。生图模型的图片输出单独计价。
PRICES: dict[str, dict[str, float]] = {
    "gemini-3.8-flash": {"in": 0.75, "out": 3.75},
    "gemini-3.5-flash-lite": {"in": 0.30, "out": 2.50},
    "gemini-3.1-flash-lite": {"in": 0.25, "out": 1.50},
    "gemini-3.1-flash-lite-preview": {"in": 0.25, "out": 1.50},
    "gemini-3-flash-preview": {"in": 0.50, "out": 3.00},
    "gemini-3.1-pro-preview": {"in": 2.00, "out": 12.00},
    "gemini-nano-banana-2.1": {"in": 1.50, "out": 7.50, "image": 30.0},
    "gemini-3.1-flash-image": {"in": 0.50, "out": 3.00, "image": 60.0},
    "gemini-3.1-flash-image-preview": {"in": 0.50, "out": 3.00, "image": 60.0},
    "gemini-3-pro-image": {"in": 2.00, "out": 12.00, "image": 120.0},
}


class MissingApiKeyError(RuntimeError):
    pass


def load_api_key() -> str:
    key = os.getenv(API_KEY_ENV, "").strip()
    if not key:
        raise MissingApiKeyError(f"未设置环境变量 {API_KEY_ENV}。真实调用前请先设置它；离线命令（report）不需要。")
    return key


def make_client():
    from google import genai

    return genai.Client(api_key=load_api_key())


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=str)


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ---------- 数据集 ----------


def require_dataset() -> None:
    if not (DIAGRAM_DIR / "test.json").exists():
        raise FileNotFoundError(f"缺少数据集：{DIAGRAM_DIR}。请先按 README 下载 PaperBananaBench。")


def resolve_image(rel_path: str) -> Path | None:
    """解析数据集图片路径，兼容解压造成的文件名乱码。"""
    from utils.dataset_paths import resolve_data_asset_path

    return resolve_data_asset_path(rel_path, "diagram", dataset_name="PaperBananaBench", work_dir=REPO)


def load_bench() -> tuple[dict[str, dict], dict[str, dict]]:
    """返回 (测试集 id → 条目, 参考池 id → 条目)。"""
    require_dataset()
    test = {x["id"]: x for x in load_json(DIAGRAM_DIR / "test.json")}
    ref = {x["id"]: x for x in load_json(DIAGRAM_DIR / "ref.json")}
    return test, ref


# ---------- 图片 ----------


def jpeg_bytes(path: Path, max_side: int) -> bytes:
    ensure_dirs()
    key = hashlib.md5(f"{Path(path).resolve()}|{max_side}".encode("utf-8")).hexdigest()[:20]
    cached = CACHE_DIR / f"img_{key}.jpg"
    if cached.exists():
        return cached.read_bytes()
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    cached.write_bytes(buf.getvalue())
    return buf.getvalue()


def image_part(path: Path, max_side: int = 1024):
    from google.genai import types

    return types.Part.from_bytes(data=jpeg_bytes(path, max_side), mime_type="image/jpeg")


# ---------- Gemini 调用 ----------


async def gen_text(client, model: str, parts: list, *, system: str | None = None, temperature: float = 0.0,
                   json_mode: bool = False, attempts: int = 4) -> str:
    """简单重试、不换模型，保证同一实验内各组条件一致。"""
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json" if json_mode else None,
    )
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = await client.aio.models.generate_content(model=model, contents=parts, config=config)
            return response.text or ""
        except Exception as exc:  # noqa: BLE001
            last = exc
            await asyncio.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{model} 调用失败：{str(last)[:300]}")


def parse_json(text: str) -> Any:
    import json_repair

    raw = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S)
    if fenced:
        raw = fenced.group(1)
    return json_repair.loads(raw)


# ---------- 统计 ----------


def mean_score(refs: list[str], scores: dict[str, Any], k: int) -> float:
    """前 k 条参考的平均分；没有评分的按 0 分计。"""
    top = list(refs or [])[:k]
    if not top:
        return 0.0
    return float(np.mean([scores.get(x) or 0 for x in top]))


def paired_bootstrap_ci(a, b, *, n: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """逐条配对差值的均值与 95% 自助法置信区间。"""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def combine_pairwise(first: dict | None, second: dict | None, key: str) -> str:
    """两种顺序的评审合并：first 里 A 是被测方，second 里 B 是被测方。两次都胜才算胜。"""
    a = str((first or {}).get(key, "Tie")).strip().upper()
    b = str((second or {}).get(key, "Tie")).strip().upper()
    score = (1 if a == "A" else -1 if a == "B" else 0) + (1 if b == "B" else -1 if b == "A" else 0)
    return "win" if score > 0 else "loss" if score < 0 else "tie"


# ---------- 用量与计费 ----------

CALL_TAG: contextvars.ContextVar[str] = contextvars.ContextVar("eval_call_tag", default="")


@dataclass
class UsageLog:
    """包装 genai 客户端的 generate_content，记录每次调用的模型、实际应答版本、token 与耗时。"""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def attach(self, client) -> None:
        models = client.aio.models
        original = models.generate_content

        async def logged(*, model, contents, config=None):
            started = time.time()
            response = await original(model=model, contents=contents, config=config)
            usage = response.usage_metadata
            modalities = getattr(config, "response_modalities", None) or []
            image_tokens = sum(
                d.token_count or 0
                for d in (getattr(usage, "candidates_tokens_details", None) or [])
                if str(d.modality).endswith("IMAGE")
            )
            self.calls.append({
                "tag": CALL_TAG.get(),
                "model": model,
                "version": getattr(response, "model_version", ""),
                "image": "IMAGE" in modalities,
                "secs": round(time.time() - started, 2),
                "in": getattr(usage, "prompt_token_count", 0) or 0,
                "out": getattr(usage, "candidates_token_count", 0) or 0,
                "thoughts": getattr(usage, "thoughts_token_count", 0) or 0,
                "image_tokens": image_tokens,
            })
            return response

        models.generate_content = logged


def call_cost(call: dict[str, Any]) -> float:
    """按标价估算单次调用费用：图片 token 按图片单价，其余输出（含思考）按文本单价。"""
    price = PRICES.get(call.get("model", ""))
    if not price:
        return 0.0
    image_tokens = call.get("image_tokens", 0) if call.get("image") else 0
    text_out = (call.get("out", 0) - image_tokens) + call.get("thoughts", 0)
    return (call.get("in", 0) * price["in"] + image_tokens * price.get("image", 0.0) + text_out * price["out"]) / 1e6


def summarize_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, Any]] = {}
    for c in calls:
        key = (c.get("tag", ""), c.get("model", ""), c.get("version", ""))
        g = groups.setdefault(key, {"tag": key[0], "model": key[1], "version": key[2], "n": 0, "secs": 0.0,
                                    "max_secs": 0.0, "in": 0, "out": 0, "thoughts": 0, "cost": 0.0})
        g["n"] += 1
        g["secs"] += c.get("secs", 0.0)
        g["max_secs"] = max(g["max_secs"], c.get("secs", 0.0))
        for f in ("in", "out", "thoughts"):
            g[f] += c.get(f, 0)
        g["cost"] += call_cost(c)
    return sorted(groups.values(), key=lambda g: (g["tag"], g["model"]))


def print_usage(calls: list[dict[str, Any]]) -> float:
    total = 0.0
    for g in summarize_calls(calls):
        total += g["cost"]
        print(f"  {g['tag'] or '-':28s} {g['model']:26s} 应答={g['version'] or '-':26s} 次数={g['n']:3d} "
              f"平均 {g['secs'] / g['n']:.1f}s 最长 {g['max_secs']:.1f}s 输入={g['in']} 输出={g['out']} 思考={g['thoughts']} "
              f"约 ${g['cost']:.3f}")
    print(f"  合计约 ${total:.3f}（按标价估算）")
    return total
