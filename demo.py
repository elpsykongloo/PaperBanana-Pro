# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PaperBanana 并行 Streamlit 演示
接受用户文本输入，复制 10 份，并行处理以生成多个图表候选方案供比较。
"""

import streamlit as st
import asyncio
import math
import base64
import json
import time
import re
import html
import threading
import uuid
from io import BytesIO
from PIL import Image
from pathlib import Path
import sys
import os
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional

# 将项目根目录添加到路径
sys.path.insert(0, str(Path(__file__).parent))

print("调试：正在导入代理模块...")
try:
    from agents.planner_agent import PlannerAgent
    print("调试：已导入 PlannerAgent")
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    print("调试：已导入所有代理模块")
    from utils import config
    from utils.config_loader import load_model_config
    from utils.dataset_paths import DEFAULT_DATASET_NAME, get_reference_file_path
    from utils.demo_task_utils import (
        build_evolution_stages,
        create_sample_inputs,
        find_final_stage_keys,
        get_task_ui_config,
        normalize_task_name,
    )
    from utils import image_utils
    from utils.concurrency import compute_effective_concurrency
    from utils.paperviz_processor import PaperVizProcessor
    from utils.result_order import get_candidate_id, sort_results_stably
    from utils.result_bundle import (
        build_run_manifest,
        companion_bundle_path,
        write_json_payload,
        write_result_bundle,
    )
    from utils.run_report import build_failure_manifest, build_result_summary
    from utils.runtime_settings import (
        build_all_provider_ui_defaults,
        build_runtime_context,
        resolve_runtime_settings,
    )
    print("调试：已导入工具模块")

    REPO_ROOT = Path(__file__).parent
    model_config_data = load_model_config(REPO_ROOT)

except ImportError as e:
    print(f"调试：导入错误：{e}")
    import traceback
    traceback.print_exc()
    raise e
except Exception as e:
    print(f"调试：导入过程中发生异常：{e}")
    import traceback
    traceback.print_exc()
    raise e

st.set_page_config(
    layout="wide",
    page_title="PaperBanana 并行演示",
    page_icon="🍌"
)

def clean_text(text):
    """清理文本，移除无效的 UTF-8 代理字符。"""
    if not text:
        return text
    if isinstance(text, str):
        # 移除导致 UnicodeEncodeError 的代理字符
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    return text

def base64_to_image(b64_str):
    """将 base64 字符串转换为 PIL 图像。"""
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data))
    except Exception:
        return None


def safe_log_text(value, max_len=2000):
    """将任意日志文本转换为可安全打印的字符串。"""
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        text = repr(value)
    text = text.replace("\x00", "\\x00")
    try:
        safe = text.encode("utf-8", errors="backslashreplace").decode("utf-8", errors="ignore")
    except Exception:
        safe = repr(text)
    if len(safe) > max_len:
        return safe[:max_len] + f"...(truncated {len(safe)-max_len} chars)"
    return safe


def get_refine_request_timeout_seconds(provider: str) -> float:
    """获取精修单次请求超时（秒），避免某次 API 请求无限挂起。"""
    env_val = os.getenv("REFINE_REQUEST_TIMEOUT_SEC", "").strip()
    if env_val:
        try:
            return max(float(env_val), 30.0)
        except ValueError:
            pass
    if provider == "gemini":
        return 240.0
    return 180.0


def get_refine_max_attempts(provider: str) -> int:
    """获取精修最大尝试次数，避免无限重试占满会话。"""
    env_val = os.getenv("REFINE_MAX_ATTEMPTS", "").strip()
    if env_val:
        try:
            return max(int(env_val), 1)
        except ValueError:
            pass
    return 12 if provider == "gemini" else 10


def get_refine_total_timeout_seconds(provider: str) -> float:
    """获取精修总时长上限（秒）。"""
    env_val = os.getenv("REFINE_TOTAL_TIMEOUT_SEC", "").strip()
    if env_val:
        try:
            return max(float(env_val), 60.0)
        except ValueError:
            pass
    return 1800.0 if provider == "gemini" else 1200.0


def extract_retry_delay_seconds(error_text: str) -> Optional[float]:
    """从错误文本中提取建议重试等待时间（秒）。"""
    if not error_text:
        return None
    lowered = error_text.lower()

    patterns = [
        r"retry in\s*([0-9]+(?:\.[0-9]+)?)s",
        r"retrydelay['\"]?\s*[:=]\s*['\"]?([0-9]+(?:\.[0-9]+)?)s",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            try:
                return max(float(match.group(1)), 1.0)
            except ValueError:
                continue
    return None


def emit_refine_status(status_callback: Optional[Callable[[str], None]], message: str):
    """向 UI 发送精修实时状态。"""
    if not status_callback or not message:
        return
    try:
        status_callback(message)
    except Exception as cb_error:
        try:
            print(f"[DEBUG] [WARN] 精修状态回调失败: {safe_log_text(cb_error)}")
        except Exception:
            pass


def normalize_image_mime_type(mime_type: Optional[str]) -> str:
    """将上传 MIME 归一化为 Gemini/Evolink 可接受值。"""
    if not mime_type:
        return "image/png"
    lowered = mime_type.strip().lower()
    if lowered in ("image/jpg", "image/pjpeg"):
        return "image/jpeg"
    if lowered in ("image/jpeg", "image/png"):
        return lowered
    return "image/png"


COMMON_ASPECT_RATIOS = [
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
]


GEMINI_TEXT_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
]

GEMINI_IMAGE_MODELS = [
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
]

def build_provider_defaults():
    return build_all_provider_ui_defaults(
        base_dir=REPO_ROOT,
        model_config_data=model_config_data,
    )


PROVIDER_DEFAULTS = build_provider_defaults()
REFINE_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=2)
REFINE_JOBS_LOCK = threading.Lock()
REFINE_JOBS: dict[str, "RefineJobState"] = {}


@dataclass
class RefineJobState:
    job_id: str
    provider: str
    image_model_name: str
    resolution: str
    aspect_ratio: str
    num_images: int
    input_mime_type: str
    original_image_bytes: bytes = field(repr=False)
    status: str = "running"
    progress_done: int = 0
    progress_total: int = 0
    status_history: list[str] = field(default_factory=list)
    refined_images: list[dict] = field(default_factory=list)
    failed_results: list[dict] = field(default_factory=list)
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    elapsed_seconds: float = 0.0
    cancel_requested: bool = False
    future: Future | None = field(default=None, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "job_id": self.job_id,
                "provider": self.provider,
                "image_model_name": self.image_model_name,
                "resolution": self.resolution,
                "aspect_ratio": self.aspect_ratio,
                "num_images": self.num_images,
                "input_mime_type": self.input_mime_type,
                "status": self.status,
                "progress_done": self.progress_done,
                "progress_total": self.progress_total,
                "status_history": list(self.status_history),
                "refined_images": list(self.refined_images),
                "failed_results": list(self.failed_results),
                "error": self.error,
                "created_at": self.created_at,
                "elapsed_seconds": self.elapsed_seconds,
                "cancel_requested": self.cancel_requested,
                "original_image_bytes": self.original_image_bytes,
            }


def _store_refine_job(job: RefineJobState) -> None:
    with REFINE_JOBS_LOCK:
        REFINE_JOBS[job.job_id] = job


def get_refine_job(job_id: str) -> RefineJobState | None:
    with REFINE_JOBS_LOCK:
        return REFINE_JOBS.get(job_id)


def get_refine_job_snapshot(job_id: str) -> dict | None:
    job = get_refine_job(job_id)
    if job is None:
        return None
    return job.snapshot()


def clear_refine_job(job_id: str) -> None:
    with REFINE_JOBS_LOCK:
        REFINE_JOBS.pop(job_id, None)


def append_refine_job_status(job_id: str, message: str) -> None:
    job = get_refine_job(job_id)
    if job is None or not message:
        return
    with job.lock:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {message}"
        if job.status_history and job.status_history[-1] == line:
            return
        job.status_history.append(line)
        if len(job.status_history) > 20:
            job.status_history = job.status_history[-20:]


def update_refine_job_progress(job_id: str, done_count: int, total_count: int) -> None:
    job = get_refine_job(job_id)
    if job is None:
        return
    with job.lock:
        job.progress_done = done_count
        job.progress_total = total_count


def request_refine_job_cancel(job_id: str) -> None:
    job = get_refine_job(job_id)
    if job is None:
        return
    with job.lock:
        job.cancel_requested = True
        job.cancel_event.set()
    append_refine_job_status(job_id, "[精修] 已收到停止请求，将在当前请求结束后停止后续重试。")


def persist_refine_job_results(snapshot: dict) -> None:
    st.session_state["refined_images"] = snapshot.get("refined_images", [])
    st.session_state["refine_failed_results"] = snapshot.get("failed_results", [])
    st.session_state["refine_timestamp"] = snapshot.get("created_at", "N/A")
    st.session_state["refine_result_resolution"] = snapshot.get("resolution", "N/A")
    st.session_state["refine_count"] = snapshot.get("num_images", 0)
    st.session_state["refine_provider_used"] = snapshot.get("provider", "N/A")
    st.session_state["refine_image_model_used"] = snapshot.get("image_model_name", "N/A")
    st.session_state["refine_original_image_bytes"] = snapshot.get("original_image_bytes", b"")
    st.session_state["refine_original_input_mime_type"] = snapshot.get("input_mime_type", "image/png")
    st.session_state["refine_failed_results"] = snapshot.get("failed_results", [])


async def process_parallel_candidates(
    data_list,
    dataset_name=DEFAULT_DATASET_NAME,
    task_name="diagram",
    exp_mode="dev_planner_critic",
    retrieval_setting="auto",
    model_name="",
    image_model_name="",
    provider="evolink",
    api_key="",
    concurrency_mode="auto",
    max_concurrent=20,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
):
    """使用 PaperVizProcessor 并行处理多个候选方案。"""
    task_name = normalize_task_name(task_name)
    total_candidates = len(data_list)
    effective_concurrent = compute_effective_concurrency(
        concurrency_mode=concurrency_mode,
        max_concurrent=max_concurrent,
        total_candidates=total_candidates,
        task_name=task_name,
        retrieval_setting=retrieval_setting,
        exp_mode=exp_mode,
        provider=provider,
    )

    print(f"\n{'='*60}")
    print(f"[DEBUG] process_parallel_candidates 开始")
    print(f"[DEBUG]   task={task_name}, provider={provider}, model={model_name}, image_model={image_model_name}")
    print(f"[DEBUG]   exp_mode={exp_mode}, retrieval={retrieval_setting}, candidates={total_candidates}")
    print(f"[DEBUG]   concurrency_mode={concurrency_mode}, max_concurrent={max_concurrent}, effective={effective_concurrent}")
    print(f"[DEBUG]   api_key={'已设置 (' + api_key[:8] + '...)' if api_key else '未设置'}")
    print(f"{'='*60}")

    if progress_callback:
        try:
            progress_callback(0, total_candidates, effective_concurrent)
        except Exception as cb_error:
            print(f"[DEBUG] [WARN] 进度回调失败(初始化): {cb_error}")
    if status_callback:
        try:
            status_callback(
                f"任务启动：候选数={total_candidates}, 并发={effective_concurrent}, 流水线={exp_mode}"
            )
        except Exception as cb_error:
            print(f"[DEBUG] [WARN] 状态回调失败(初始化): {cb_error}")

    from utils import generation_utils
    runtime_settings = resolve_runtime_settings(
        provider,
        api_key=api_key,
        model_name=model_name,
        image_model_name=image_model_name,
        concurrency_mode=concurrency_mode,
        max_concurrent=max_concurrent,
        base_dir=REPO_ROOT,
        model_config_data=model_config_data,
    )
    runtime_context = build_runtime_context(
        runtime_settings,
        status_hook=status_callback,
    )
    if not runtime_settings.api_key:
        print(f"[DEBUG] [WARN] 未提供 API Key，Provider 可能无法正常工作")

    # 创建实验配置
    exp_config = config.ExpConfig(
        dataset_name=dataset_name,
        task_name=task_name,
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        concurrency_mode=runtime_settings.concurrency_mode,
        max_concurrent=runtime_settings.max_concurrent,
        model_name=runtime_settings.model_name,
        image_model_name=runtime_settings.image_model_name,
        provider=runtime_settings.provider,
        work_dir=Path(__file__).parent,
    )
    print(
        f"[DEBUG] ExpConfig 已创建: task={exp_config.task_name}, provider={exp_config.provider}, "
        f"model={exp_config.model_name}, image_model={exp_config.image_model_name}"
    )

    # 初始化处理器及所有代理
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

    # 并行处理所有候选方案（并发量由处理器控制）
    results = []
    concurrent_num = effective_concurrent

    try:
        with generation_utils.use_runtime_context(runtime_context):
            async for result_data in processor.process_queries_batch(
                data_list,
                max_concurrent=concurrent_num,
                do_eval=False,
                status_callback=status_callback,
            ):
                results.append(result_data)
                if progress_callback:
                    try:
                        progress_callback(len(results), total_candidates, effective_concurrent)
                    except Exception as cb_error:
                        print(f"[DEBUG] [WARN] 进度回调失败(更新): {cb_error}")
    finally:
        await generation_utils.close_runtime_context(runtime_context)
        processor.shutdown()

    return results, effective_concurrent

async def refine_image_with_nanoviz(
    image_bytes,
    edit_prompt,
    aspect_ratio="21:9",
    image_size="2K",
    api_key="",
    provider="evolink",
    image_model_name="",
    task_id: int = 1,
    status_callback: Optional[Callable[[str], None]] = None,
    input_mime_type: str = "image/png",
    max_attempts: Optional[int] = None,
    max_total_seconds: Optional[float] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    runtime_context=None,
):
    """
    使用图像编辑 API 精修图像，支持 Evolink 和 Gemini 两种 Provider。

    参数：
        image_bytes: 图像字节数据
        edit_prompt: 描述所需修改的文本
        aspect_ratio: 输出宽高比 (21:9, 16:9, 3:2)
        image_size: 输出分辨率 (2K 或 4K)
        api_key: API 密钥
        provider: "evolink" 或 "gemini"

    返回：
        元组 (编辑后的图像字节数据, 成功消息)
    """
    from utils import generation_utils

    attempt = 0
    sleep_seconds = 2.0
    timeout_seconds = get_refine_request_timeout_seconds(provider)
    attempt_limit = max_attempts or get_refine_max_attempts(provider)
    total_time_limit = max_total_seconds or get_refine_total_timeout_seconds(provider)
    started_at = time.perf_counter()
    task_prefix = f"task#{task_id}"
    normalized_mime_type = normalize_image_mime_type(input_mime_type)
    runtime_settings = resolve_runtime_settings(
        provider,
        api_key=api_key,
        image_model_name=image_model_name,
        base_dir=REPO_ROOT,
        model_config_data=model_config_data,
    )
    active_runtime_context = runtime_context or build_runtime_context(
        runtime_settings,
        status_hook=status_callback,
    )
    owns_runtime_context = runtime_context is None

    try:
        with generation_utils.use_runtime_context(active_runtime_context):
            while attempt < attempt_limit:
                if cancel_check and cancel_check():
                    emit_refine_status(
                        status_callback,
                        f"[精修][{task_prefix}] cancelled before attempt={attempt + 1}",
                    )
                    return None, "⛔ 已取消精修任务"
                elapsed = time.perf_counter() - started_at
                if elapsed >= total_time_limit:
                    break
                attempt += 1
                try:
                    if provider == "gemini":
                        # ====== Gemini 路径：多模态 API，直接传图片字节 ======
                        gemini_client = generation_utils.get_gemini_client()
                        if gemini_client is None:
                            await asyncio.sleep(min(sleep_seconds, 10.0))
                            sleep_seconds = min(sleep_seconds * 1.2, 15.0)
                            continue

                        from google.genai import types

                        contents = [
                            types.Part.from_text(
                                text=image_utils.build_gemini_image_prompt(
                                    edit_prompt,
                                    aspect_ratio=aspect_ratio,
                                    image_size=image_size,
                                )
                            ),
                            types.Part.from_bytes(mime_type=normalized_mime_type, data=image_bytes),
                        ]
                        config_kwargs = {
                            "temperature": 1.0,
                            "max_output_tokens": 8192,
                            "response_modalities": ["IMAGE"],
                        }
                        config = types.GenerateContentConfig(
                            **config_kwargs,
                        )

                        selected_model = runtime_settings.image_model_name
                        gemini_model_sequence = [selected_model]
                        if selected_model != "gemini-3.1-flash-image-preview":
                            gemini_model_sequence.append("gemini-3.1-flash-image-preview")

                        # 每 5 次失败切换一次模型（循环）
                        model_index = ((attempt - 1) // 5) % len(gemini_model_sequence)
                        image_model = gemini_model_sequence[model_index]

                        emit_refine_status(
                            status_callback,
                            f"[精修][{task_prefix}] attempt={attempt} model={image_model} timeout={int(timeout_seconds)}s",
                        )
                        response = await asyncio.wait_for(
                            gemini_client.aio.models.generate_content(
                                model=image_model,
                                contents=contents,
                                config=config,
                            ),
                            timeout=timeout_seconds,
                        )

                        if response and response.candidates and response.candidates[0].content.parts:
                            for part in response.candidates[0].content.parts:
                                if hasattr(part, "inline_data") and part.inline_data:
                                    edited_image_data = part.inline_data.data
                                    if isinstance(edited_image_data, bytes) and edited_image_data:
                                        emit_refine_status(
                                            status_callback,
                                            f"[精修][{task_prefix}] success on attempt={attempt} model={image_model}",
                                        )
                                        return edited_image_data, f"✅ 图像精修成功！（第 {attempt} 次尝试）"
                                    if isinstance(edited_image_data, str) and edited_image_data:
                                        emit_refine_status(
                                            status_callback,
                                            f"[精修][{task_prefix}] success on attempt={attempt} model={image_model}",
                                        )
                                        return base64.b64decode(edited_image_data), f"✅ 图像精修成功！（第 {attempt} 次尝试）"

                        raise RuntimeError("Gemini 未返回有效图像数据")

                    else:
                        # ====== Evolink 路径：上传图片获取 URL → image_urls ======
                        evolink_provider = generation_utils.get_evolink_provider()
                        if evolink_provider is None:
                            await asyncio.sleep(min(sleep_seconds, 10.0))
                            sleep_seconds = min(sleep_seconds * 1.2, 15.0)
                            continue

                        image_model = runtime_settings.image_model_name
                        emit_refine_status(
                            status_callback,
                            f"[精修][{task_prefix}] attempt={attempt} model={image_model} timeout={int(timeout_seconds)}s",
                        )

                        # 步骤 1：上传原始图片到 Evolink 文件服务
                        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                        ref_image_url = await generation_utils.upload_image_to_evolink(
                            image_b64,
                            media_type=normalized_mime_type,
                        )
                        try:
                            print(f"[精修] attempt={attempt}, uploaded_ref={safe_log_text(ref_image_url[:80])}...")
                        except Exception:
                            pass

                        # 步骤 2：图像生成 API（传入参考图 URL）
                        result = await asyncio.wait_for(
                            evolink_provider.generate_image(
                                model_name=image_model,
                                prompt=edit_prompt,
                                aspect_ratio=aspect_ratio,
                                quality=image_size,
                                image_urls=[ref_image_url],
                                max_attempts=1,
                                retry_delay=3,
                            ),
                            timeout=timeout_seconds,
                        )

                        if result and result[0] and result[0] != "Error":
                            edited_image_data = base64.b64decode(result[0])
                            emit_refine_status(
                                status_callback,
                                f"[精修][{task_prefix}] success on attempt={attempt} model={image_model}",
                            )
                            return edited_image_data, f"✅ 图像精修成功！（第 {attempt} 次尝试）"

                        raise RuntimeError("Evolink 未返回有效图像数据")

                except asyncio.TimeoutError:
                    err_text = (
                        f"{provider} request timed out after {int(timeout_seconds)}s "
                        f"(attempt={attempt}, {task_prefix})"
                    )
                    delay = min(max(sleep_seconds, 3.0), 20.0)
                    emit_refine_status(
                        status_callback,
                        f"[精修][{task_prefix}] timeout, wait {delay:.1f}s then retry",
                    )
                    try:
                        print(f"[精修][重试] {safe_log_text(err_text, max_len=1200)}")
                    except Exception:
                        pass
                    await asyncio.sleep(delay)
                    sleep_seconds = min(max(delay * 1.2, sleep_seconds * 1.25), 30.0)
                except Exception as e:
                    # 不向前端抛错，持续重试直到成功
                    error_text = safe_log_text(e, max_len=2000)
                    lower_error = error_text.lower()

                    # Windows 套接字异常自愈：重建 client/provider 后继续。
                    if "winerror 10038" in lower_error:
                        try:
                            generation_utils.reinitialize_runtime_context(active_runtime_context)
                        except Exception as reinit_error:
                            try:
                                print(f"[精修][WARN] socket 自愈重建失败: {safe_log_text(reinit_error)}")
                            except Exception:
                                pass

                    suggested_delay = extract_retry_delay_seconds(error_text)
                    delay = min(sleep_seconds, 20.0)
                    if suggested_delay is not None:
                        delay = min(max(delay, suggested_delay), 60.0)
                    if "limit: 0" in lower_error:
                        delay = max(delay, 30.0)
                    emit_refine_status(
                        status_callback,
                        f"[精修][{task_prefix}] failed attempt={attempt}, wait {delay:.1f}s | {error_text[:160]}",
                    )
                    try:
                        print(f"[精修][重试] attempt={attempt}, err={error_text}")
                    except Exception:
                        pass
                    await asyncio.sleep(delay)
                    sleep_seconds = min(max(delay * 1.1, sleep_seconds * 1.25), 30.0)

        elapsed = time.perf_counter() - started_at
        if cancel_check and cancel_check():
            emit_refine_status(
                status_callback,
                f"[精修][{task_prefix}] cancelled after {elapsed:.1f}s",
            )
            return None, "⛔ 已取消精修任务"
        failure_message = (
            f"❌ 图像精修失败：已达到重试上限（attempts={attempt_limit}, elapsed={elapsed:.1f}s）"
        )
        emit_refine_status(
            status_callback,
            f"[精修][{task_prefix}] exhausted attempts={attempt_limit} elapsed={elapsed:.1f}s",
        )
        return None, failure_message
    finally:
        if owns_runtime_context:
            await generation_utils.close_runtime_context(active_runtime_context)


async def refine_images_with_count(
    image_bytes,
    edit_prompt,
    num_images=3,
    aspect_ratio="21:9",
    image_size="2K",
    api_key="",
    provider="evolink",
    image_model_name="",
    input_mime_type="image/png",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
):
    """并发精修多张图像，按完成先后收集，避免被单一慢任务阻塞观感。"""
    safe_count = max(1, int(num_images))
    results = [None] * safe_count
    tasks = []
    runtime_settings = resolve_runtime_settings(
        provider,
        api_key=api_key,
        image_model_name=image_model_name,
        base_dir=REPO_ROOT,
        model_config_data=model_config_data,
    )
    runtime_context = build_runtime_context(
        runtime_settings,
        status_hook=status_callback,
    )
    semaphore = asyncio.Semaphore(min(safe_count, 3 if provider == "gemini" else 2))

    try:
        with generation_utils.use_runtime_context(runtime_context):
            for idx in range(safe_count):
                variant_prompt = (
                    f"{edit_prompt}\n\n"
                    f"[Variant Request #{idx + 1}] Keep the semantics unchanged, "
                    f"but provide a distinct visual variant."
                )
                async def run_one(task_idx=idx, prompt_text=variant_prompt):
                    async with semaphore:
                        if cancel_check and cancel_check():
                            return task_idx, (None, "⛔ 已取消精修任务")
                        try:
                            value = await refine_image_with_nanoviz(
                                image_bytes=image_bytes,
                                edit_prompt=prompt_text,
                                aspect_ratio=aspect_ratio,
                                image_size=image_size,
                                api_key=runtime_settings.api_key,
                                provider=runtime_settings.provider,
                                image_model_name=runtime_settings.image_model_name,
                                task_id=task_idx + 1,
                                status_callback=status_callback,
                                input_mime_type=input_mime_type,
                                cancel_check=cancel_check,
                                runtime_context=runtime_context,
                            )
                            return task_idx, value
                        except asyncio.CancelledError:
                            return task_idx, (None, "⛔ 已取消精修任务")

                task = asyncio.create_task(run_one())
                tasks.append(task)

            done_count = 0
            for future in asyncio.as_completed(tasks):
                if cancel_check and cancel_check():
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    break
                task_idx, value = await future
                results[task_idx] = value
                done_count += 1
                if progress_callback:
                    try:
                        progress_callback(done_count, safe_count)
                    except Exception as cb_error:
                        try:
                            print(f"[DEBUG] [WARN] 精修进度回调失败: {safe_log_text(cb_error)}")
                        except Exception:
                            pass
                emit_refine_status(
                    status_callback,
                    f"[精修] completed {done_count}/{safe_count} (task#{task_idx + 1})",
                )

            return results
    finally:
        await generation_utils.close_runtime_context(runtime_context)


def start_refine_background_job(
    *,
    image_bytes: bytes,
    edit_prompt: str,
    num_images: int,
    aspect_ratio: str,
    image_size: str,
    api_key: str,
    provider: str,
    image_model_name: str,
    input_mime_type: str,
) -> str:
    job_id = f"refine_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    runtime_settings = resolve_runtime_settings(
        provider,
        api_key=api_key,
        image_model_name=image_model_name,
        base_dir=REPO_ROOT,
        model_config_data=model_config_data,
    )
    job = RefineJobState(
        job_id=job_id,
        provider=runtime_settings.provider,
        image_model_name=runtime_settings.image_model_name,
        resolution=image_size,
        aspect_ratio=aspect_ratio,
        num_images=int(num_images),
        input_mime_type=input_mime_type,
        original_image_bytes=image_bytes,
    )
    _store_refine_job(job)
    append_refine_job_status(job_id, f"[精修] input mime={input_mime_type}, bytes={len(image_bytes)}")

    def worker():
        started_at = time.perf_counter()
        try:
            def on_progress(done_count: int, total_count: int):
                update_refine_job_progress(job_id, done_count, total_count)

            def on_status(message: str):
                append_refine_job_status(job_id, message)

            refined_results = asyncio.run(
                refine_images_with_count(
                    image_bytes=image_bytes,
                    edit_prompt=edit_prompt,
                    num_images=int(num_images),
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    api_key=runtime_settings.api_key,
                    provider=runtime_settings.provider,
                    image_model_name=runtime_settings.image_model_name,
                    input_mime_type=input_mime_type,
                    progress_callback=on_progress,
                    status_callback=on_status,
                    cancel_check=job.cancel_event.is_set,
                )
            )

            refined_images = []
            failed_refine_results = []
            for idx, result_item in enumerate(refined_results):
                if not result_item:
                    failed_refine_results.append(
                        {"index": idx + 1, "message": "❌ 未返回任何结果"}
                    )
                    continue
                refined_bytes, message = result_item
                if refined_bytes:
                    refined_images.append({
                        "index": idx + 1,
                        "bytes": refined_bytes,
                        "message": message,
                    })
                else:
                    failed_refine_results.append({
                        "index": idx + 1,
                        "message": message,
                    })

            with job.lock:
                job.refined_images = refined_images
                job.failed_results = failed_refine_results
                job.progress_done = max(job.progress_done, len(refined_images) + len(failed_refine_results))
                job.progress_total = max(job.progress_total, int(num_images))
                job.elapsed_seconds = time.perf_counter() - started_at
                if job.cancel_requested:
                    job.status = "cancelled"
                else:
                    job.status = "completed"
            append_refine_job_status(
                job_id,
                f"[精修] finished status={job.status} success={len(refined_images)} failed={len(failed_refine_results)}",
            )
        except Exception as exc:
            with job.lock:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.elapsed_seconds = time.perf_counter() - started_at
            append_refine_job_status(job_id, f"[精修] failed: {job.error}")

    job.future = REFINE_JOB_EXECUTOR.submit(worker)
    return job_id


def display_candidate_result(result, candidate_id, exp_mode, task_name="diagram"):
    """展示单个候选方案的结果。"""
    task_name = normalize_task_name(task_name)
    task_config = get_task_ui_config(task_name)

    if isinstance(result, dict) and result.get("status") == "failed":
        st.error(f"候选方案 {candidate_id} 失败：{result.get('error', 'Unknown error')}")
        detail = result.get("error_detail")
        if detail:
            with st.expander("查看失败详情", expanded=False):
                st.code(clean_text(detail))
        return

    final_image_key, final_desc_key = find_final_stage_keys(
        result,
        task_name=task_name,
        exp_mode=exp_mode,
    )

    # 展示最终图像
    if final_image_key and final_image_key in result:
        img = base64_to_image(result[final_image_key])
        if img:
            st.image(
                img,
                use_container_width=True,
                caption=f"候选方案 {candidate_id}（{task_config['final_caption']}）",
            )

            # 添加下载按钮
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            st.download_button(
                label="[DOWN] 下载",
                data=buffered.getvalue(),
                file_name=f"candidate_{candidate_id}.png",
                mime="image/png",
                key=f"download_candidate_{candidate_id}",
                use_container_width=True
            )
        else:
            st.error(f"候选方案 {candidate_id} 的图像解码失败")
    else:
        st.warning(f"候选方案 {candidate_id} 未生成图像")

    final_code_key = ""
    if task_name == "plot" and final_desc_key:
        final_code_key = (
            final_desc_key
            if final_desc_key == "vanilla_plot_code"
            else f"{final_desc_key}_code"
        )
    if task_name == "plot" and final_code_key and result.get(final_code_key):
        code_text = clean_text(result[final_code_key])
        with st.expander("🧪 查看最终 Matplotlib 代码", expanded=False):
            st.code(code_text, language="python")
            st.download_button(
                label="[DOWN] 下载代码",
                data=code_text.encode("utf-8"),
                file_name=f"candidate_{candidate_id}.py",
                mime="text/x-python",
                key=f"download_candidate_code_{candidate_id}",
                use_container_width=True,
            )

    # 在折叠面板中展示演化时间线
    stages = build_evolution_stages(result, exp_mode, task_name=task_name)
    if len(stages) > 1:
        with st.expander(f"🔄 查看演化时间线（{len(stages)} 个阶段）", expanded=False):
            st.caption("查看图表在不同流水线阶段的演化过程")

            for idx, stage in enumerate(stages):
                st.markdown(f"### {stage['name']}")
                st.caption(stage['description'])

                # 展示该阶段的图像
                stage_img = base64_to_image(result.get(stage['image_key']))
                if stage_img:
                    st.image(stage_img, use_container_width=True)

                # 展示描述
                if stage['desc_key'] in result:
                    with st.expander(f"📝 描述", expanded=False):
                        cleaned_desc = clean_text(result[stage['desc_key']])
                        st.write(cleaned_desc)

                if task_name == "plot" and stage.get("code_key") and stage["code_key"] in result:
                    with st.expander("🧪 本阶段代码", expanded=False):
                        cleaned_code = clean_text(result[stage["code_key"]])
                        st.code(cleaned_code, language="python")

                # 展示评审建议（如有）
                if 'suggestions_key' in stage and stage['suggestions_key'] in result:
                    suggestions = result[stage['suggestions_key']]
                    with st.expander(f"💡 评审建议", expanded=False):
                        cleaned_sugg = clean_text(suggestions)
                        if cleaned_sugg.strip() == "No changes needed.":
                            st.success("✅ 无需修改——迭代已停止。")
                        else:
                            st.write(cleaned_sugg)

                # 在阶段之间添加分隔线（最后一个除外）
                if idx < len(stages) - 1:
                    st.divider()
    else:
        # 如果只有一个阶段，使用更简洁的折叠面板展示描述
        with st.expander(f"📝 查看描述", expanded=False):
            if final_desc_key and final_desc_key in result:
                # 清理文本，移除无效的 UTF-8 字符
                cleaned_desc = clean_text(result[final_desc_key])
                st.write(cleaned_desc)
            else:
                st.info("暂无描述")

def main():
    st.title("🍌 PaperBanana 演示")
    st.markdown("AI 驱动的科学图表生成与精修")

    # 创建选项卡
    tab1, tab2 = st.tabs(["📊 生成候选方案", "✨ 精修图像"])

    # ==================== 选项卡 1：生成候选方案 ====================
    with tab1:
        task_name = normalize_task_name(st.session_state.get("tab1_task_name", "diagram"))
        task_config = get_task_ui_config(task_name)
        st.markdown(f"### {task_config['intro']}")

        # 侧边栏配置（选项卡 1）
        with st.sidebar:
            st.title("[SET] 生成设置")

            task_name = st.selectbox(
                "任务类型",
                ["diagram", "plot"],
                index=0 if task_name == "diagram" else 1,
                key="tab1_task_name",
                format_func=lambda x: {
                    "diagram": "diagram - 学术图解",
                    "plot": "plot - 统计图",
                }[x],
                help="选择生成方法图解还是统计图",
            )
            task_config = get_task_ui_config(task_name)
            st.info(f"**当前任务：** {task_config['display_name_cn']}")
            if "tab1_dataset_name" not in st.session_state:
                st.session_state["tab1_dataset_name"] = DEFAULT_DATASET_NAME
            dataset_name = st.text_input(
                "参考数据集",
                key="tab1_dataset_name",
                help="用于检索参考样例以及解析数据集内的相对资源路径。",
            ).strip() or DEFAULT_DATASET_NAME
            st.caption(f"当前数据集资源：`{dataset_name}`")

            exp_mode = st.selectbox(
                "流水线模式",
                ["demo_planner_critic", "demo_full"],
                index=0,
                key="tab1_exp_mode",
                help="选择使用哪种代理流水线",
            )

            mode_info = {
                "demo_planner_critic": "规划器 → 可视化器 → 评审器 → 可视化器",
                "demo_full": "检索器 → 规划器 → 风格化器 → 可视化器 → 评审器 → 可视化器。（风格化器能让图表更具美感，但可能过度简化。建议两种模式都尝试并选择最佳结果）",
            }
            st.info(f"**流水线：** {mode_info[exp_mode]}")

            retrieval_setting = st.selectbox(
                "检索设置",
                ["auto", "auto-full", "random", "none"],
                index=0,
                key="tab1_retrieval_setting",
                help="如何检索参考图表",
                format_func=lambda x: {
                    "auto": "auto — LLM 智能选参考，轻量模式",
                    "auto-full": "auto-full — LLM 智能选参考，完整上下文",
                    "random": "random — 随机选 10 个参考（免费）",
                    "none": "none — 不检索参考（免费）",
                }[x],
            )

            retrieval_target_label = "可视化意图" if task_name == "plot" else "图注"
            retrieval_ref_path = get_reference_file_path(
                dataset_name,
                task_name,
                work_dir=REPO_ROOT,
            )
            _retrieval_cost_info = {
                "auto": f"💡 轻量 auto：仅发送{retrieval_target_label}给 LLM 做匹配，适合大多数试跑。",
                "auto-full": "[WARN] 完整 auto：会把候选参考的完整内容发给 LLM 做匹配，成本显著更高，仅在需要高精度检索时使用。",
                "random": "✅ 随机从参考集中抽样，不调用额外检索推理。",
                "none": "✅ 跳过检索，不使用参考图表。",
            }
            st.info(_retrieval_cost_info[retrieval_setting])
            if retrieval_setting != "none" and not retrieval_ref_path.exists():
                st.warning(
                    f"当前仓库未发现数据集 `{dataset_name}` 的 `{task_name}/ref.json`，运行时会自动回退到 `none`。"
                )

            num_candidates = st.number_input(
                "候选方案数量",
                min_value=1,
                max_value=20,
                value=5,
                key="tab1_num_candidates",
                help="要并行生成多少个候选方案",
            )

            concurrency_mode = st.selectbox(
                "并发策略",
                ["auto", "manual"],
                index=0,
                key="tab1_concurrency_mode",
                help="auto：自动并发（默认）| manual：使用固定并发上限",
            )

            max_concurrent = st.number_input(
                "并发上限",
                min_value=1,
                max_value=100,
                value=20,
                step=1,
                key="tab1_max_concurrent",
                help="候选任务并发上限，默认 20",
            )

            effective_concurrency_preview = compute_effective_concurrency(
                concurrency_mode=concurrency_mode,
                max_concurrent=int(max_concurrent),
                total_candidates=int(num_candidates),
            )
            estimated_batches_preview = math.ceil(
                int(num_candidates) / max(1, effective_concurrency_preview)
            )

            with st.expander("📈 并发可视化调节", expanded=True):
                c1, c2 = st.columns(2)
                c1.metric("有效并发", effective_concurrency_preview)
                c2.metric("预计批次数", estimated_batches_preview)
                st.caption(
                    f"策略：{concurrency_mode} | 并发上限：{int(max_concurrent)} | 候选数：{int(num_candidates)}"
                )

            if task_config["uses_render_controls"]:
                aspect_ratio = st.selectbox(
                    "宽高比",
                    COMMON_ASPECT_RATIOS,
                    key="tab1_aspect_ratio",
                    help="生成图表的宽高比",
                )

                provider_for_resolution = st.session_state.get("tab1_provider", "gemini")
                resolution_options = ["1K", "2K", "4K"] if provider_for_resolution == "gemini" else ["2K", "4K"]
                default_resolution = st.session_state.get("tab1_image_resolution", "2K")
                if default_resolution not in resolution_options:
                    default_resolution = "2K" if "2K" in resolution_options else resolution_options[0]

                image_resolution = st.selectbox(
                    "图像分辨率",
                    resolution_options,
                    index=resolution_options.index(default_resolution),
                    key="tab1_image_resolution",
                    help=f"生成图像的分辨率（当前 Provider 支持：{', '.join(resolution_options)}）",
                )
            else:
                aspect_ratio = "16:9"
                image_resolution = "2K"
                st.info("当前任务使用 Matplotlib 代码渲染，宽高比和图像分辨率控件暂不生效。")

            max_critic_rounds = st.number_input(
                "最大评审轮次",
                min_value=1,
                max_value=5,
                value=3,
                key="tab1_max_critic_rounds",
                help="评审优化迭代的最大轮次",
            )

            provider = st.selectbox(
                "API Provider",
                ["gemini", "evolink"],
                index=0,
                key="tab1_provider",
                help="gemini：Google 官方 API（需翻墙）| evolink：国内代理",
            )

            provider_defaults = PROVIDER_DEFAULTS[provider]
            if "tab1_api_key" not in st.session_state:
                st.session_state["tab1_api_key"] = provider_defaults["api_key_default"]
            if "tab1_model_name" not in st.session_state:
                st.session_state["tab1_model_name"] = provider_defaults["model_name"]
            if "tab1_image_model_name" not in st.session_state:
                st.session_state["tab1_image_model_name"] = provider_defaults["image_model_name"]

            if "prev_provider" not in st.session_state:
                st.session_state["prev_provider"] = provider
            if st.session_state["prev_provider"] != provider:
                st.session_state["prev_provider"] = provider
                st.session_state["tab1_model_name"] = provider_defaults["model_name"]
                st.session_state["tab1_image_model_name"] = provider_defaults["image_model_name"]
                st.session_state["tab1_api_key"] = provider_defaults["api_key_default"]
                new_resolution_options = ["1K", "2K", "4K"] if provider == "gemini" else ["2K", "4K"]
                if st.session_state.get("tab1_image_resolution") not in new_resolution_options:
                    st.session_state["tab1_image_resolution"] = (
                        "2K" if "2K" in new_resolution_options else new_resolution_options[0]
                    )
                st.rerun()

            api_key = st.text_input(
                provider_defaults["api_key_label"],
                type="password",
                key="tab1_api_key",
                help=provider_defaults["api_key_help"],
            )

            if provider == "gemini":
                current_gemini_text_model = st.session_state.get("tab1_model_name", GEMINI_TEXT_MODELS[0])
                if current_gemini_text_model not in GEMINI_TEXT_MODELS:
                    current_gemini_text_model = GEMINI_TEXT_MODELS[0]

                model_name = st.selectbox(
                    "文本模型",
                    GEMINI_TEXT_MODELS,
                    index=GEMINI_TEXT_MODELS.index(current_gemini_text_model),
                    key="tab1_model_name",
                    help="用于推理/规划/评审的模型名称（可展开选择）",
                )
            else:
                model_name = st.text_input(
                    "文本模型",
                    key="tab1_model_name",
                    help="用于推理/规划/评审的模型名称",
                )

            if task_config["uses_image_model"]:
                if provider == "gemini":
                    current_gemini_image_model = st.session_state.get(
                        "tab1_image_model_name",
                        "gemini-3-pro-image-preview",
                    )
                    if current_gemini_image_model not in GEMINI_IMAGE_MODELS:
                        current_gemini_image_model = "gemini-3-pro-image-preview"

                    image_model_name = st.selectbox(
                        "图像模型",
                        GEMINI_IMAGE_MODELS,
                        index=GEMINI_IMAGE_MODELS.index(current_gemini_image_model),
                        key="tab1_image_model_name",
                        help="用于图像生成的模型名称（可展开选择）",
                    )
                else:
                    image_model_name = st.text_input(
                        "图像模型",
                        key="tab1_image_model_name",
                        help="用于图像生成的模型名称",
                    )
            else:
                image_model_name = ""
                st.caption("当前任务不会调用图像生成模型，最终图像由文本模型生成的 Matplotlib 代码渲染。")

        st.divider()

        # 输入区域
        st.markdown("## 📝 输入")
        st.caption(task_config["intro"])

        content_state_key = f"tab1_{task_name}_content"
        visual_state_key = f"tab1_{task_name}_visual_intent"
        content_example_key = f"tab1_{task_name}_content_example_selector"
        visual_example_key = f"tab1_{task_name}_visual_example_selector"
        example_options = ["无", task_config["example_name"]]

        col_input1, col_input2 = st.columns([3, 2])

        with col_input1:
            content_example = st.selectbox(
                task_config["content_selector_label"],
                example_options,
                key=content_example_key,
            )
            if content_example == task_config["example_name"]:
                content_value = task_config["example_content"]
            else:
                content_value = st.session_state.get(content_state_key, "")

            input_content = st.text_area(
                task_config["content_label"],
                value=content_value,
                height=250,
                placeholder=task_config["content_placeholder"],
                help=task_config["content_help"],
                key=f"{content_state_key}_editor",
            )

        with col_input2:
            visual_example = st.selectbox(
                task_config["visual_selector_label"],
                example_options,
                key=visual_example_key,
            )
            if visual_example == task_config["example_name"]:
                visual_value = task_config["example_visual_intent"]
            else:
                visual_value = st.session_state.get(visual_state_key, "")

            visual_intent = st.text_area(
                task_config["visual_label"],
                value=visual_value,
                height=250,
                placeholder=task_config["visual_placeholder"],
                help=task_config["visual_help"],
                key=f"{visual_state_key}_editor",
            )

        if st.button("🚀 生成候选方案", type="primary", use_container_width=True):
            if not input_content or not visual_intent:
                st.error(
                    f"请同时提供{task_config['content_input_name']}和{task_config['visual_input_name']}！"
                )
            else:
                st.session_state[content_state_key] = input_content
                st.session_state[visual_state_key] = visual_intent

                with st.spinner(
                    f"正在并行生成 {num_candidates} 个候选方案（策略={concurrency_mode}, 上限={int(max_concurrent)}）..."
                ):
                    input_data_list = create_sample_inputs(
                        content=input_content,
                        visual_intent=visual_intent,
                        task_name=task_name,
                        aspect_ratio=aspect_ratio,
                        num_copies=num_candidates,
                        max_critic_rounds=max_critic_rounds,
                        image_resolution=image_resolution,
                    )

                    effective_concurrency_runtime = compute_effective_concurrency(
                        concurrency_mode=concurrency_mode,
                        max_concurrent=int(max_concurrent),
                        total_candidates=len(input_data_list),
                    )
                    estimated_batches_runtime = math.ceil(
                        len(input_data_list) / max(1, effective_concurrency_runtime)
                    )

                    st.caption(
                        f"并发配置：{concurrency_mode} | 上限 {int(max_concurrent)} | "
                        f"有效并发 {effective_concurrency_runtime} | 预计批次 {estimated_batches_runtime}"
                    )
                    progress_bar = st.progress(0.0, text="等待任务开始...")
                    progress_text = st.empty()
                    status_text = st.empty()
                    status_history = []
                    run_started_at = time.perf_counter()

                    def on_progress(done_count: int, total_count: int, effective_count: int):
                        ratio = 0.0 if total_count <= 0 else min(done_count / total_count, 1.0)
                        elapsed = time.perf_counter() - run_started_at
                        progress_bar.progress(
                            ratio,
                            text=f"并发 {effective_count} | 已完成 {done_count}/{total_count}",
                        )
                        progress_text.caption(
                            f"已耗时 {elapsed:.1f}s | 剩余 {max(total_count - done_count, 0)} 个候选"
                        )

                    def on_status(message: str):
                        if not message:
                            return
                        elapsed = time.perf_counter() - run_started_at
                        ts = datetime.now().strftime("%H:%M:%S")
                        line = f"[{ts}] {message}"
                        if status_history and status_history[-1] == line:
                            return
                        status_history.append(line)
                        if len(status_history) > 50:
                            status_history.pop(0)
                        status_text.markdown(
                            "**实时状态**\n"
                            + "\n".join([f"- {x}" for x in status_history])
                            + f"\n\n`已耗时 {elapsed:.1f}s`"
                        )

                    try:
                        results, used_concurrency = asyncio.run(process_parallel_candidates(
                            input_data_list,
                            dataset_name=dataset_name,
                            task_name=task_name,
                            exp_mode=exp_mode,
                            retrieval_setting=retrieval_setting,
                            model_name=model_name,
                            image_model_name=image_model_name,
                            provider=provider,
                            api_key=api_key,
                            concurrency_mode=concurrency_mode,
                            max_concurrent=int(max_concurrent),
                            progress_callback=on_progress,
                            status_callback=on_status,
                        ))
                        results = sort_results_stably(results)
                        total_elapsed = time.perf_counter() - run_started_at
                        progress_bar.progress(
                            1.0,
                            text=f"并发 {used_concurrency} | 已完成 {len(results)}/{len(input_data_list)}",
                        )
                        progress_text.caption(f"总耗时 {total_elapsed:.1f}s")

                        st.session_state["results"] = results
                        st.session_state["task_name"] = task_name
                        st.session_state["dataset_name"] = dataset_name
                        st.session_state["exp_mode"] = exp_mode
                        st.session_state["concurrency_mode"] = concurrency_mode
                        st.session_state["max_concurrent"] = int(max_concurrent)
                        st.session_state["effective_concurrent"] = int(used_concurrency)
                        st.session_state["estimated_batches"] = int(estimated_batches_runtime)
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state["timestamp"] = timestamp_str

                        try:
                            results_dir = Path(__file__).parent / "results" / "demo" / task_name
                            results_dir.mkdir(parents=True, exist_ok=True)

                            run_stem = config.build_run_name(
                                timestamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                                provider=provider,
                                model_name=model_name,
                                image_model_name=image_model_name,
                                retrieval_setting=retrieval_setting,
                                exp_mode=exp_mode,
                                split_name="demo",
                            )
                            json_filename = results_dir / f"demo_{task_name}_{run_stem}.json"
                            bundle_filename = companion_bundle_path(json_filename)
                            summary = build_result_summary(results)
                            failures = build_failure_manifest(results)
                            manifest = build_run_manifest(
                                producer="demo",
                                result_count=len(results),
                                dataset_name=dataset_name,
                                task_name=task_name,
                                split_name="demo",
                                exp_mode=exp_mode,
                                retrieval_setting=retrieval_setting,
                                provider=provider,
                                model_name=model_name,
                                image_model_name=image_model_name,
                                concurrency_mode=concurrency_mode,
                                max_concurrent=int(max_concurrent),
                                max_critic_rounds=int(max_critic_rounds),
                                timestamp=timestamp_str,
                            )

                            write_json_payload(json_filename, results)
                            write_result_bundle(
                                bundle_filename,
                                results,
                                manifest=manifest,
                                summary=summary,
                                failures=failures,
                            )

                            st.session_state["json_file"] = str(json_filename)
                            st.session_state["bundle_file"] = str(bundle_filename)
                            success_count = sum(
                                1 for item in results
                                if not (isinstance(item, dict) and item.get("status") == "failed")
                            )
                            failed_count = len(results) - success_count
                            st.success(
                                f"✅ {task_config['display_name_cn']}任务完成：成功 {success_count} 个，失败 {failed_count} 个候选方案。"
                            )
                            st.info(f"💾 结果已保存至：`{json_filename.name}`")
                        except Exception as e:
                            st.warning(f"[WARN] 已生成 {len(results)} 个候选方案，但 JSON 保存失败：{e}")
                    except Exception as e:
                        progress_bar.empty()
                        progress_text.empty()
                        status_text.empty()
                        st.error(f"处理过程中出错：{e}")
                        import traceback
                        st.code(traceback.format_exc())

        # 展示结果
        if "results" in st.session_state and st.session_state["results"]:
            results = st.session_state["results"]
            current_task_name = normalize_task_name(
                st.session_state.get(
                    "task_name",
                    results[0].get("task_name", task_name) if results else task_name,
                )
            )
            current_dataset_name = str(
                st.session_state.get(
                    "dataset_name",
                    results[0].get("dataset_name", DEFAULT_DATASET_NAME)
                    if results
                    else DEFAULT_DATASET_NAME,
                )
            ).strip() or DEFAULT_DATASET_NAME
            current_task_config = get_task_ui_config(current_task_name)
            current_mode = st.session_state.get("exp_mode", exp_mode)
            timestamp = st.session_state.get("timestamp", "N/A")
            mode_used = st.session_state.get("concurrency_mode", concurrency_mode)
            max_used = st.session_state.get("max_concurrent", int(max_concurrent))
            effective_used = st.session_state.get(
                "effective_concurrent",
                compute_effective_concurrency(mode_used, int(max_used), len(results)),
            )

            st.divider()
            st.markdown(f"## 🎨 已生成的{current_task_config['display_name_cn']}候选方案")
            success_count = sum(
                1 for item in results
                if not (isinstance(item, dict) and item.get("status") == "failed")
            )
            failed_count = len(results) - success_count
            st.caption(
                f"生成时间：{timestamp} | 任务：{current_task_config['display_name_cn']} | "
                f"数据集：{current_dataset_name} | "
                f"流水线：{mode_info.get(current_mode, current_mode)} | "
                f"并发：{mode_used} (max={max_used}, effective={effective_used}) | "
                f"成功/失败：{success_count}/{failed_count}"
            )

            # 如果有结果文件则显示下载按钮
            json_file_path = Path(st.session_state["json_file"]) if "json_file" in st.session_state else None
            bundle_file_path = Path(st.session_state["bundle_file"]) if "bundle_file" in st.session_state else None
            if json_file_path and json_file_path.exists():
                button_cols = [3, 1]
                if bundle_file_path and bundle_file_path.exists():
                    button_cols.append(1)
                columns = st.columns(button_cols)
                with columns[0]:
                    file_label = f"📄 结果已保存至：`{json_file_path.relative_to(Path.cwd())}`"
                    if bundle_file_path and bundle_file_path.exists():
                        file_label += f"\n\n🧾 Bundle：`{bundle_file_path.relative_to(Path.cwd())}`"
                    st.info(file_label)
                with columns[1]:
                    with open(json_file_path, "r", encoding="utf-8") as f:
                        json_data = f.read()
                    st.download_button(
                        label="[DOWN] 下载 JSON",
                        data=json_data,
                        file_name=json_file_path.name,
                        mime="application/json",
                        use_container_width=True
                    )
                if bundle_file_path and bundle_file_path.exists():
                    with columns[2]:
                        with open(bundle_file_path, "r", encoding="utf-8") as f:
                            bundle_data = f.read()
                        st.download_button(
                            label="[DOWN] 下载 Bundle",
                            data=bundle_data,
                            file_name=bundle_file_path.name,
                            mime="application/json",
                            use_container_width=True
                        )

            # 以网格形式展示结果（3 列）
            num_cols = 3
            num_results = len(results)

            for row_start in range(0, num_results, num_cols):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    result_idx = row_start + col_idx
                    if result_idx < num_results:
                        result_item = results[result_idx]
                        candidate_id = get_candidate_id(result_item, result_idx)
                        with cols[col_idx]:
                            display_candidate_result(
                                result_item,
                                candidate_id,
                                current_mode,
                                task_name=current_task_name,
                            )

            # 添加 ZIP 下载按钮
            st.divider()
            st.markdown("### 💾 批量下载")

            try:
                import zipfile

                zip_buffer = BytesIO()
                zip_export_failures = []
                exported_count = 0
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for fallback_idx, result in enumerate(results):
                        candidate_id = get_candidate_id(result, fallback_idx)
                        if isinstance(result, dict) and result.get("status") == "failed":
                            zip_export_failures.append(
                                f"候选 {candidate_id}: 流水线执行失败，无法导出"
                            )
                            continue

                        final_image_key, final_desc_key = find_final_stage_keys(
                            result,
                            task_name=current_task_name,
                            exp_mode=current_mode,
                        )

                        exported_any = False
                        if final_image_key and final_image_key in result:
                            try:
                                raw_bytes = base64.b64decode(result[final_image_key])
                                img = Image.open(BytesIO(raw_bytes))
                                image_format = (img.format or "").upper()
                                ext_map = {
                                    "JPEG": "jpg",
                                    "JPG": "jpg",
                                    "PNG": "png",
                                    "WEBP": "webp",
                                    "GIF": "gif",
                                }
                                image_ext = ext_map.get(image_format, "bin")
                                zip_file.writestr(
                                    f"candidate_{candidate_id}.{image_ext}",
                                    raw_bytes
                                )
                                exported_any = True
                            except Exception as export_err:
                                zip_export_failures.append(
                                    f"候选 {candidate_id}: 图像导出失败 ({export_err})"
                                )
                        else:
                            zip_export_failures.append(
                                f"候选 {candidate_id}: 未找到最终图像"
                            )

                        if current_task_name == "plot" and final_desc_key:
                            final_code_key = (
                                final_desc_key
                                if final_desc_key == "vanilla_plot_code"
                                else f"{final_desc_key}_code"
                            )
                            if result.get(final_code_key):
                                zip_file.writestr(
                                    f"candidate_{candidate_id}.py",
                                    clean_text(result[final_code_key]).encode("utf-8"),
                                )
                                exported_any = True

                        if exported_any:
                            exported_count += 1

                zip_buffer.seek(0)
                if exported_count > 0:
                    st.download_button(
                        label="[DOWN] 下载 ZIP 压缩包",
                        data=zip_buffer.getvalue(),
                        file_name=(
                            f"paperbanana_{current_task_name}_candidates_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                        ),
                        mime="application/zip",
                        use_container_width=True
                    )
                    if zip_export_failures:
                        st.warning(
                            f"ZIP 已准备好：成功导出 {exported_count} 个，失败 {len(zip_export_failures)} 个。"
                        )
                        with st.expander("查看 ZIP 导出失败详情", expanded=False):
                            for item in zip_export_failures:
                                st.write(item)
                    else:
                        st.success("ZIP 压缩包已准备好，可以下载！")
                else:
                    st.error("ZIP 压缩包创建失败：没有可导出的候选结果。")
                    if zip_export_failures:
                        with st.expander("查看 ZIP 导出失败详情", expanded=True):
                            for item in zip_export_failures:
                                st.write(item)
            except Exception as e:
                st.error(f"创建 ZIP 压缩包失败：{e}")

    # ==================== 选项卡 2：精修图像 ====================
    with tab2:
        st.markdown("### 精修并放大您的图表至高分辨率（2K/4K）")
        st.caption("上传候选方案中的图像或任意图表，描述修改需求，生成高分辨率版本")

        # 精修设置侧边栏
        with st.sidebar:
            st.title("✨ 精修设置")

            refine_resolution = st.selectbox(
                "目标分辨率",
                ["2K", "4K"],
                index=0,
                key="refine_resolution",
                help="更高的分辨率需要更长时间但能产生更好的质量"
            )

            refine_aspect_ratio = st.selectbox(
                "宽高比",
                COMMON_ASPECT_RATIOS,
                index=0,
                key="refine_aspect_ratio",
                help="精修图像的宽高比"
            )

            refine_num_images = st.number_input(
                "精修张数",
                min_value=1,
                max_value=12,
                value=3,
                step=1,
                key="refine_num_images",
                help="并发生成多少张不同的精修结果"
            )

            refine_provider = st.selectbox(
                "精修 Provider",
                ["gemini", "evolink"],
                index=0,
                key="refine_provider",
                help="为精修单独选择 Provider，不依赖生成页设置"
            )
            refine_provider_defaults = PROVIDER_DEFAULTS[refine_provider]

            if "refine_api_key" not in st.session_state:
                st.session_state["refine_api_key"] = refine_provider_defaults["api_key_default"]
            if "refine_image_model_name" not in st.session_state:
                st.session_state["refine_image_model_name"] = refine_provider_defaults["image_model_name"]
            if "refine_prev_provider" not in st.session_state:
                st.session_state["refine_prev_provider"] = refine_provider
            if st.session_state["refine_prev_provider"] != refine_provider:
                st.session_state["refine_prev_provider"] = refine_provider
                st.session_state["refine_api_key"] = refine_provider_defaults["api_key_default"]
                st.session_state["refine_image_model_name"] = refine_provider_defaults["image_model_name"]
                st.rerun()

            refine_api_key = st.text_input(
                refine_provider_defaults["api_key_label"],
                type="password",
                key="refine_api_key",
                help=refine_provider_defaults["api_key_help"]
            )

            if refine_provider == "gemini":
                current_refine_gemini_image_model = st.session_state.get(
                    "refine_image_model_name",
                    PROVIDER_DEFAULTS["gemini"]["image_model_name"],
                )
                if current_refine_gemini_image_model not in GEMINI_IMAGE_MODELS:
                    current_refine_gemini_image_model = PROVIDER_DEFAULTS["gemini"]["image_model_name"]

                refine_image_model_name = st.selectbox(
                    "精修图像模型",
                    GEMINI_IMAGE_MODELS,
                    index=GEMINI_IMAGE_MODELS.index(current_refine_gemini_image_model),
                    key="refine_image_model_name",
                    help="精修流程使用的图像模型"
                )
            else:
                refine_image_model_name = st.text_input(
                    "精修图像模型",
                    key="refine_image_model_name",
                    help="精修流程使用的图像模型"
                )

        st.divider()

        # 上传区域
        st.markdown("## 📤 上传图像")
        uploaded_file = st.file_uploader(
            "选择一个图像文件",
            type=["png", "jpg", "jpeg"],
            help="上传您想要精修的图表"
        )

        if uploaded_file is not None:
            # 展示上传的图像
            uploaded_image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 原始图像")
                st.image(uploaded_image, use_container_width=True)

            with col2:
                st.markdown("### 编辑指令")
                edit_prompt = st.text_area(
                    "描述您想要的修改",
                    height=200,
                    placeholder="例如：'将配色方案改为学术论文风格' 或 '将文字放大加粗' 或 '保持内容不变但输出更高分辨率'",
                    help="描述您想要的修改，或使用'保持内容不变'仅进行放大",
                    key="edit_prompt"
                )

                active_refine_job_id = st.session_state.get("active_refine_job_id")
                active_refine_snapshot = (
                    get_refine_job_snapshot(active_refine_job_id) if active_refine_job_id else None
                )

                if st.button("✨ 精修图像", type="primary", use_container_width=True):
                    if not edit_prompt:
                        st.error("请提供编辑指令！")
                    elif active_refine_snapshot and active_refine_snapshot.get("status") == "running":
                        st.warning("当前已有精修任务在后台运行，请先等待完成或停止当前任务。")
                    else:
                        image_bytes = uploaded_file.getvalue()
                        input_mime_type = normalize_image_mime_type(getattr(uploaded_file, "type", None))
                        job_id = start_refine_background_job(
                            image_bytes=image_bytes,
                            edit_prompt=edit_prompt,
                            num_images=int(refine_num_images),
                            aspect_ratio=refine_aspect_ratio,
                            image_size=refine_resolution,
                            api_key=refine_api_key,
                            provider=refine_provider,
                            image_model_name=refine_image_model_name,
                            input_mime_type=input_mime_type,
                        )
                        st.session_state["refined_images"] = []
                        st.session_state["refine_failed_results"] = []
                        st.session_state["active_refine_job_id"] = job_id
                        st.session_state["last_refine_completed_job_id"] = None
                        st.info("已启动后台精修任务，页面会自动刷新显示进度。")
                        st.rerun()

                finalized_refine_snapshot = None
                active_refine_job_id = st.session_state.get("active_refine_job_id")
                active_refine_snapshot = (
                    get_refine_job_snapshot(active_refine_job_id) if active_refine_job_id else None
                )
                if active_refine_snapshot and active_refine_snapshot.get("status") in {"completed", "cancelled", "failed"}:
                    finalized_refine_snapshot = active_refine_snapshot
                    if st.session_state.get("last_refine_completed_job_id") != active_refine_job_id:
                        persist_refine_job_results(active_refine_snapshot)
                        st.session_state["last_refine_completed_job_id"] = active_refine_job_id
                    st.session_state.pop("active_refine_job_id", None)
                    clear_refine_job(active_refine_job_id)
                    active_refine_snapshot = None

                if active_refine_snapshot and active_refine_snapshot.get("status") == "running":
                    progress_done = active_refine_snapshot.get("progress_done", 0)
                    progress_total = max(active_refine_snapshot.get("progress_total", int(refine_num_images)), 1)
                    ratio = min(progress_done / progress_total, 1.0)
                    st.progress(
                        ratio,
                        text=f"后台精修进度：已完成 {progress_done}/{progress_total}",
                    )
                    st.caption(
                        f"Provider: {active_refine_snapshot.get('provider')} | "
                        f"模型: {active_refine_snapshot.get('image_model_name')} | "
                        f"分辨率: {active_refine_snapshot.get('resolution')} | "
                        f"停止请求: {'已发送' if active_refine_snapshot.get('cancel_requested') else '未发送'}"
                    )
                    status_lines = active_refine_snapshot.get("status_history", [])
                    if status_lines:
                        html_lines = "<br>".join(html.escape(x) for x in status_lines[-10:])
                        st.markdown(
                            (
                                "**精修实时状态（最近10条）**\n"
                                f"<div style='max-height:220px; overflow-y:auto; "
                                f"border:1px solid rgba(255,255,255,0.12); border-radius:8px; "
                                f"padding:8px 10px; line-height:1.6;'>"
                                f"{html_lines}"
                                "</div>"
                            ),
                            unsafe_allow_html=True,
                        )

                    action_col1, action_col2 = st.columns(2)
                    with action_col1:
                        if st.button("🛑 停止精修", use_container_width=True):
                            request_refine_job_cancel(active_refine_snapshot["job_id"])
                            st.warning("已发送停止请求，系统会在当前请求结束后停止后续重试。")
                            st.rerun()
                    with action_col2:
                        if st.button("🔄 刷新状态", use_container_width=True):
                            st.rerun()

                    time.sleep(1.0)
                    st.rerun()

                if finalized_refine_snapshot:
                    completed = len(finalized_refine_snapshot.get("refined_images", []))
                    failed = len(finalized_refine_snapshot.get("failed_results", []))
                    if finalized_refine_snapshot.get("status") == "completed":
                        st.success(f"✅ 后台精修完成：成功 {completed} 张，失败 {failed} 张。")
                    elif finalized_refine_snapshot.get("status") == "cancelled":
                        st.warning(f"⛔ 精修已停止：成功 {completed} 张，失败/取消 {failed} 张。")
                    else:
                        st.error(f"❌ 后台精修失败：{finalized_refine_snapshot.get('error', 'Unknown error')}")

            # 展示精修结果（如有）
            if "refined_images" in st.session_state and st.session_state["refined_images"]:
                st.divider()
                st.markdown("## 🎨 精修结果")
                final_resolution = st.session_state.get("refine_result_resolution", refine_resolution)
                final_count = st.session_state.get("refine_count", len(st.session_state["refined_images"]))
                refine_provider_used = st.session_state.get("refine_provider_used", refine_provider)
                refine_image_model_used = st.session_state.get("refine_image_model_used", refine_image_model_name)
                failed_refine_results = st.session_state.get("refine_failed_results", [])
                st.caption(
                    f"生成时间：{st.session_state.get('refine_timestamp', 'N/A')} | "
                    f"分辨率：{final_resolution} | 张数：{final_count} | "
                    f"Provider：{refine_provider_used} | 模型：{refine_image_model_used}"
                )
                if failed_refine_results:
                    st.warning(f"有 {len(failed_refine_results)} 张精修失败。")
                    with st.expander("查看失败详情", expanded=False):
                        for item in failed_refine_results:
                            st.write(f"结果 {item['index']}: {item['message']}")

                st.markdown("### 精修前")
                original_preview_bytes = st.session_state.get(
                    "refine_original_image_bytes",
                    uploaded_file.getvalue() if uploaded_file is not None else b"",
                )
                if original_preview_bytes:
                    st.image(Image.open(BytesIO(original_preview_bytes)), use_container_width=True)

                st.markdown(f"### 精修后（{final_resolution}）")
                refined_images = st.session_state["refined_images"]

                import zipfile
                zip_buffer = BytesIO()
                zip_name = f"refined_{final_resolution}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for item in refined_images:
                        idx = item.get("index", 0)
                        img_bytes = item.get("bytes", b"")
                        if not img_bytes:
                            continue
                        file_name = f"refined_{final_resolution}_{idx}.png"
                        zip_file.writestr(file_name, img_bytes)

                # 两列网格预览，缩小占位（仅影响预览，不影响下载原图）
                preview_cols = 2
                preview_width_px = 420
                for row_start in range(0, len(refined_images), preview_cols):
                    cols = st.columns(preview_cols, gap="large")
                    for col_offset in range(preview_cols):
                        item_pos = row_start + col_offset
                        if item_pos >= len(refined_images):
                            continue

                        item = refined_images[item_pos]
                        idx = item.get("index", item_pos + 1)
                        img_bytes = item.get("bytes", b"")
                        if not img_bytes:
                            continue

                        with cols[col_offset]:
                            st.markdown(f"#### 结果 {idx}")
                            refined_image = Image.open(BytesIO(img_bytes))
                            st.image(refined_image, width=preview_width_px)

                            file_name = f"refined_{final_resolution}_{idx}.png"
                            st.download_button(
                                label=f"[DOWN] 下载结果 {idx}",
                                data=img_bytes,
                                file_name=file_name,
                                mime="image/png",
                                key=f"download_refined_{idx}_{final_resolution}_{item_pos}",
                                use_container_width=True
                            )

                zip_buffer.seek(0)
                st.download_button(
                    label="[DOWN] 一键下载全部结果（ZIP）",
                    data=zip_buffer.getvalue(),
                    file_name=zip_name,
                    mime="application/zip",
                    use_container_width=True,
                    key="download_refined_zip"
                )

if __name__ == "__main__":
    main()
