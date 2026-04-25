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
import hashlib
import json
import time
import re
import html
import logging
import threading
import uuid
from collections import Counter
from contextlib import contextmanager
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import sys
import os
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional
from urllib.parse import urlparse
from urllib.request import url2pathname

# 将项目根目录添加到路径
sys.path.insert(0, str(Path(__file__).parent))
from utils.log_config import get_logger, setup_logging
from utils.runtime_events import (
    coerce_runtime_event,
    create_runtime_event,
    event_summary_text,
    runtime_event_from_log_record,
)

setup_logging("INFO", mode="streamlit")
logger = get_logger("PaperBananaDemo")


def _looks_like_paperbanana_workspace(path_value: str | Path | None) -> bool:
    if not path_value:
        return False
    candidate = Path(path_value)
    return (
        candidate.exists()
        and (candidate / "demo.py").exists()
        and (candidate / "agents").exists()
        and (candidate / "utils").exists()
        and (
            (candidate / "data").exists()
            or (candidate / "configs").exists()
            or (candidate / "pyproject.toml").exists()
        )
    )


def _load_direct_url_workspace_root(install_root: Path) -> Path | None:
    direct_url_candidates = sorted(install_root.glob("paperbanana_pro-*.dist-info/direct_url.json"))
    for direct_url_path in direct_url_candidates:
        try:
            payload = json.loads(direct_url_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        parsed_url = urlparse(str(payload.get("url", "") or ""))
        if parsed_url.scheme != "file":
            continue
        candidate_text = url2pathname(parsed_url.path or "")
        if parsed_url.netloc:
            candidate_text = f"//{parsed_url.netloc}{candidate_text}"
        candidate = Path(candidate_text)
        if _looks_like_paperbanana_workspace(candidate):
            return candidate.resolve(strict=False)
    return None


def resolve_demo_base_dir(
    install_root: Path | None = None,
    *,
    cwd: Path | None = None,
) -> Path:
    install_root = Path(install_root or Path(__file__).resolve().parent).resolve(strict=False)

    env_candidate = os.getenv("PAPERBANANA_HOME", "").strip()
    if _looks_like_paperbanana_workspace(env_candidate):
        return Path(env_candidate).resolve(strict=False)

    current_dir = Path(cwd or Path.cwd()).resolve(strict=False)
    for candidate in (current_dir, *current_dir.parents):
        if _looks_like_paperbanana_workspace(candidate):
            return candidate.resolve(strict=False)

    direct_url_root = _load_direct_url_workspace_root(install_root)
    if direct_url_root is not None:
        return direct_url_root

    return install_root

try:
    from agents.planner_agent import PlannerAgent
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    from utils import config, generation_utils
    from utils.config_loader import (
        delete_provider_api_key,
        load_model_config,
        write_provider_api_key,
    )
    from utils.dataset_paths import DEFAULT_DATASET_NAME, get_reference_file_path
    from utils.demo_job_store import (
        append_job_event,
        read_job_events,
        read_job_snapshot,
        read_ui_state,
        write_job_snapshot,
        write_ui_state,
    )
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
    from utils.result_order import (
        format_candidate_display_label,
        get_candidate_display_index,
        get_candidate_id,
        sort_results_stably,
    )
    from utils.result_bundle import (
        build_run_manifest,
        companion_bundle_path,
        load_result_bundle,
        write_json_payload,
        write_result_bundle,
    )
    from utils.retrieval_profiles import (
        find_curated_profile_path,
        get_curated_profile_path,
        get_legacy_manual_reference_path,
    )
    from utils.retrieval_settings import (
        DEFAULT_CURATED_PROFILE,
        get_retrieval_setting_label,
        normalize_curated_profile_name,
        normalize_retrieval_setting,
    )
    from utils.run_report import build_failure_manifest, build_result_summary
    from utils.image_generation_options import (
        get_image_model_capabilities,
        is_valid_custom_image_size,
        normalize_image_generation_options,
    )
    from utils.runtime_settings import (
        DEFAULT_PROVIDER,
        build_all_provider_ui_defaults,
        build_runtime_context,
        list_runtime_connections,
        resolve_runtime_settings,
    )
    from utils.provider_connections import (
        CUSTOM_PROVIDER_TYPE,
        BUILTIN_CONNECTION_IDS,
        ProviderConnection,
        delete_custom_connection,
        delete_custom_provider_api_key,
        discover_models,
        format_extra_headers_json,
        normalize_connection_id,
        parse_extra_headers_json,
        probe_connection,
        run_async_probe,
        upsert_custom_connection,
        write_custom_provider_api_key,
        write_connection_probe_result,
    )
    from utils.plot_input_utils import parse_plot_input_text
    from utils.plot_executor import execute_plot_code_with_details

    REPO_ROOT = resolve_demo_base_dir(Path(__file__).parent)
    model_config_data = load_model_config(REPO_ROOT)
except Exception:
    logger.exception("demo.py 初始化导入失败")
    raise

st.set_page_config(
    layout="wide",
    page_title="PaperBanana-Pro 科研插图工作台",
    page_icon="🍌"
)

GENERATION_STATUS_LINE_LIMIT = 200
GENERATION_LOG_RENDER_LIMIT = 14
GENERATION_EVENT_HISTORY_LIMIT = 400
GENERATION_EVENT_RENDER_LIMIT = 10
REFINE_STATUS_LINE_LIMIT = 120
REFINE_LOG_RENDER_LIMIT = 12
REFINE_EVENT_HISTORY_LIMIT = 240
REFINE_EVENT_RENDER_LIMIT = 10
SAFE_DISK_UI_STATE_EXCLUDE_KEYS = {
    "tab1_api_key",
    "tab1_extra_headers_json",
    "refine_api_key",
    "refine_extra_headers_json",
    "refine_uploaded_image_bytes",
}
WORKSPACE_MODE_OPTIONS = [
    "📊 生成候选方案",
    "✨ 精修图像",
]
EXAMPLE_SELECTOR_NONE_OPTION = "无"
GENERATION_TEXTAREA_MAX_CHARS = 200_000
GENERATION_MODE_INFO = {
    "demo_planner_critic": "标准流程：规划 → 首轮出图 → 评审 → 修正出图。速度更快、语义更稳，建议默认使用。",
    "demo_full": "增强风格流程：在标准流程基础上加入风格化阶段；当开启参考检索时，也会利用参考样例辅助生成。视觉表现更强，但耗时和不确定性也更高。",
}
GENERATION_DECISION_FILTER_OPTIONS = [
    "全部候选",
    "仅未淘汰",
    "仅收藏",
    "仅最终候选",
]
GENERATION_EXPORT_SCOPE_OPTIONS = [
    "全部候选",
    "仅未淘汰",
    "仅收藏",
    "仅最终候选",
]
GENERATION_CANDIDATE_DECISION_LABELS = {
    "default": "待决策",
    "favorite": "已收藏",
    "discarded": "已淘汰",
    "final": "最终候选",
}
GENERATION_CANDIDATE_STATUS_RE = re.compile(r"^候选\s+(.+?):\s*(.+)$")
GENERATION_LOGGER_NAMES = {
    "PaperVizProcessor",
    "PlannerAgent",
    "RetrieverAgent",
    "VisualizerAgent",
    "CriticAgent",
    "PolishAgent",
    "GenerationUtils",
    "ImageUtils",
}
REFINE_LOGGER_NAMES = {
    "PaperBananaDemo",
    "GenerationUtils",
    "ImageUtils",
}
BACKGROUND_FRAGMENT_AUTO_REFRESH_INTERVAL = "2s"
BACKGROUND_TERMINAL_STATUSES = {"completed", "cancelled", "failed", "interrupted"}


def streamlit_fragment(**fragment_kwargs):
    fragment_factory = getattr(st, "fragment", None)
    if callable(fragment_factory):
        return fragment_factory(**fragment_kwargs)

    def decorator(func):
        return func

    return decorator


def supports_streamlit_fragment() -> bool:
    return callable(getattr(st, "fragment", None))


def request_streamlit_rerun(*, scope: str = "app") -> None:
    rerun = getattr(st, "rerun", None)
    if not callable(rerun):
        return
    if scope == "fragment":
        try:
            rerun(scope="fragment")
            return
        except Exception:
            pass
    rerun()

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


THUMBNAIL_MAX_SIZE = (480, 480)
THUMBNAIL_JPEG_QUALITY = 70
SINGLE_CANDIDATE_LIVE_PREVIEW_WIDTH = 720
DOUBLE_CANDIDATE_LIVE_PREVIEW_WIDTH = 560
REFINE_WORKSPACE_PREVIEW_WIDTH = 520
REFINE_ORIGINAL_PREVIEW_WIDTH = 620
SINGLE_REFINE_RESULT_PREVIEW_WIDTH = 500
DOUBLE_REFINE_RESULT_PREVIEW_WIDTH = 400
REFINE_HISTORY_PREVIEW_WIDTH = 220
REFINE_HISTORY_PREVIEW_COLS = 4
REFINE_STATUS_LINE_PREVIEW_COUNT = 6
REFINE_WORKSPACE_VIEW_OPTIONS = ["工作台", "结果与状态", "版本历史"]
REFINE_INTERNAL_VERSION_KEY_RE = re.compile(r"^v\d+$")
REFINE_DISPLAY_VERSION_LABEL_RE = re.compile(r"^第\s*\d+\s*版$")
REFINE_TASK_STATUS_RE = re.compile(r"^\[精修\]\[任务\s*(\d+)\]\s*(.+)$")


def image_to_jpeg_thumbnail(img, max_size=THUMBNAIL_MAX_SIZE, quality=THUMBNAIL_JPEG_QUALITY):
    """Convert a PIL image to a JPEG thumbnail bytes for fast grid preview."""
    thumb = img.copy()
    thumb.thumbnail(max_size, Image.LANCZOS)
    if thumb.mode in ("RGBA", "P", "LA"):
        thumb = thumb.convert("RGB")
    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def get_generation_live_preview_width(candidate_count: int):
    """根据候选数量返回流式预览图宽度策略。"""
    normalized_count = max(1, int(candidate_count))
    if normalized_count == 1:
        return SINGLE_CANDIDATE_LIVE_PREVIEW_WIDTH
    if normalized_count == 2:
        return DOUBLE_CANDIDATE_LIVE_PREVIEW_WIDTH
    return "stretch"


def get_generation_live_row_layout(candidate_count: int):
    """根据候选数量返回流式卡片布局。"""
    normalized_count = max(1, int(candidate_count))
    if normalized_count == 1:
        return [1, 2, 1]
    if normalized_count == 2:
        return [0.25, 1, 1, 0.25]
    return min(3, normalized_count)


def get_refine_result_grid_columns(result_count: int) -> int:
    """根据精修结果数量返回预览网格列数。"""
    normalized_count = max(1, int(result_count))
    if normalized_count == 1:
        return 1
    if normalized_count == 2:
        return 2
    return 3


def get_refine_result_preview_width(result_count: int):
    """根据精修结果数量返回预览宽度策略。"""
    return SINGLE_REFINE_RESULT_PREVIEW_WIDTH if int(result_count) == 1 else "stretch"


def get_refine_history_grid_columns(history_count: int) -> int:
    """根据版本历史数量返回预览网格列数。"""
    normalized_count = max(1, int(history_count))
    return min(REFINE_HISTORY_PREVIEW_COLS, normalized_count)


def get_refine_history_preview_width(history_count: int):
    """根据版本历史数量返回预览宽度策略。"""
    return REFINE_HISTORY_PREVIEW_WIDTH if int(history_count) == 1 else "stretch"


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


def _emit_legacy_status(
    status_callback: Optional[Callable[[str], None]],
    event_callback: Optional[Callable[[dict], None]],
    event_payload: dict,
) -> None:
    """Emit structured event first, then fallback to legacy text callbacks."""
    if event_callback is not None:
        try:
            event_callback(dict(event_payload))
            return
        except Exception as cb_error:
            logger.warning("事件回调失败: %s", safe_log_text(cb_error))
    if status_callback is None:
        return
    try:
        status_callback(event_payload.get("message", ""))
    except Exception as cb_error:
        logger.warning("文本状态回调失败: %s", safe_log_text(cb_error))


def _log_structured_event(payload: dict) -> None:
    try:
        level_name = str(payload.get("level", "INFO") or "INFO").upper()
        level_no = getattr(logging, level_name, logging.INFO)
        source_name = str(payload.get("source", "") or "PaperBananaDemo")
        get_logger(source_name).log(
            level_no,
            payload.get("message", ""),
            extra={"paperbanana_event": dict(payload)},
        )
    except Exception as log_error:
        logger.debug("结构化事件日志写入失败: %s", safe_log_text(log_error))


def emit_generation_event(
    *,
    message: str,
    event_callback: Optional[Callable[[dict], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    kind: str = "job",
    level: str = "INFO",
    source: str = "PaperBananaDemo",
    candidate_id: str = "",
    stage: str = "",
    status: str = "",
    provider: str = "",
    model: str = "",
    attempt: int | None = None,
    error_code: int | None = None,
    details: str = "",
) -> dict:
    payload = create_runtime_event(
        level=level,
        kind=kind,
        source=source,
        message=message,
        job_type="generation",
        candidate_id=candidate_id,
        stage=stage,
        status=status,
        provider=provider,
        model=model,
        attempt=attempt,
        error_code=error_code,
        details=details,
    ).to_dict()
    _log_structured_event(payload)
    _emit_legacy_status(status_callback, event_callback, payload)
    return payload


def extract_generation_candidate_stage(message: str) -> tuple[str, str] | None:
    """从状态文本中提取候选当前阶段。"""
    if not message:
        return None
    match = GENERATION_CANDIDATE_STATUS_RE.match(str(message).strip())
    if not match:
        return None
    candidate_id = match.group(1).strip()
    stage = match.group(2).strip()
    if not candidate_id or not stage:
        return None
    return candidate_id, stage


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
    if provider in {"openai", "openai_compatible"}:
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
    if provider == "gemini":
        return 12
    if provider in {"openai", "openai_compatible"}:
        return 8
    return 10


def get_refine_total_timeout_seconds(provider: str) -> float:
    """获取精修总时长上限（秒）。"""
    env_val = os.getenv("REFINE_TOTAL_TIMEOUT_SEC", "").strip()
    if env_val:
        try:
            return max(float(env_val), 60.0)
        except ValueError:
            pass
    if provider == "gemini":
        return 1800.0
    if provider in {"openai", "openai_compatible"}:
        return 1500.0
    return 1200.0


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


def emit_refine_event(
    *,
    message: str,
    event_callback: Optional[Callable[[dict], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    kind: str = "job",
    level: str = "INFO",
    source: str = "PaperBananaDemo",
    status: str = "",
    provider: str = "",
    model: str = "",
    attempt: int | None = None,
    error_code: int | None = None,
    details: str = "",
) -> dict:
    payload = create_runtime_event(
        level=level,
        kind=kind,
        source=source,
        message=message,
        job_type="refine",
        status=status,
        provider=provider,
        model=model,
        attempt=attempt,
        error_code=error_code,
        details=details,
    ).to_dict()
    _log_structured_event(payload)
    _emit_legacy_status(status_callback, event_callback, payload)
    return payload


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


def infer_image_mime_type_from_format(image_format: str | None, *, fallback: str = "image/png") -> str:
    """根据 Pillow 识别到的格式推断 MIME。"""
    format_text = str(image_format or "").strip().upper()
    if format_text in {"JPEG", "JPG"}:
        return "image/jpeg"
    if format_text == "PNG":
        return "image/png"
    return normalize_image_mime_type(fallback)


def validate_refine_image_bytes(
    image_bytes: bytes,
    *,
    input_mime_type: str = "image/png",
    file_name: str = "",
) -> tuple[bytes, str, str | None]:
    """验证精修输入是否为可解码图片，并返回规范化后的 MIME。"""
    normalized_mime_type = normalize_image_mime_type(input_mime_type)
    raw_bytes = bytes(image_bytes or b"")
    if not raw_bytes:
        return b"", normalized_mime_type, "上传文件为空，请重新选择 PNG 或 JPG 图片。"

    try:
        with Image.open(BytesIO(raw_bytes)) as verified_image:
            verified_image.verify()
        with Image.open(BytesIO(raw_bytes)) as loaded_image:
            loaded_image.load()
            resolved_mime_type = infer_image_mime_type_from_format(
                loaded_image.format,
                fallback=normalized_mime_type,
            )
    except UnidentifiedImageError:
        display_name = file_name or "当前上传文件"
        return (
            b"",
            normalized_mime_type,
            f"{display_name} 不是可识别的 PNG/JPG 图片，请重新上传。",
        )
    except Exception as exc:
        display_name = file_name or "当前上传文件"
        return (
            b"",
            normalized_mime_type,
            f"{display_name} 读取失败：{safe_log_text(exc, max_len=200)}",
        )

    return raw_bytes, resolved_mime_type, None


def load_refine_preview_image(
    image_bytes: bytes,
    *,
    source_label: str = "当前图像",
) -> tuple[Image.Image | None, str | None]:
    """安全加载精修预览图，避免坏图像直接炸掉页面。"""
    raw_bytes = bytes(image_bytes or b"")
    if not raw_bytes:
        return None, f"{source_label} 为空，请重新选择图片。"

    try:
        with Image.open(BytesIO(raw_bytes)) as preview_image:
            preview_image.load()
            return preview_image.copy(), None
    except UnidentifiedImageError:
        return None, f"{source_label} 已损坏或不是有效图片，请重新上传或重新载入来源。"
    except Exception as exc:
        return None, f"{source_label} 预览失败：{safe_log_text(exc, max_len=200)}"


def clear_cached_refine_upload(*, clear_error: bool = True, reset_widget: bool = False) -> None:
    """清理已缓存的上传图片状态。"""
    for key in (
        "refine_uploaded_image_bytes",
        "refine_uploaded_input_mime_type",
        "refine_uploaded_filename",
        "refine_uploaded_file_size",
        "refine_uploaded_widget_present",
    ):
        st.session_state.pop(key, None)
    if clear_error:
        st.session_state.pop("refine_upload_error", None)
    if reset_widget:
        current_nonce = int(st.session_state.get("refine_upload_widget_nonce", 0) or 0)
        st.session_state["refine_upload_widget_nonce"] = current_nonce + 1


def cache_refine_uploaded_file(uploaded_file) -> str | None:
    """将上传控件返回的临时文件校验后写入稳定 session state。"""
    if uploaded_file is None:
        return None

    file_name = str(getattr(uploaded_file, "name", "") or "上传图像")
    file_size = int(getattr(uploaded_file, "size", 0) or 0)
    input_mime_type = normalize_image_mime_type(getattr(uploaded_file, "type", None))
    cached_name = str(st.session_state.get("refine_uploaded_filename", "") or "")
    cached_size = int(st.session_state.get("refine_uploaded_file_size", 0) or 0)
    cached_bytes = bytes(st.session_state.get("refine_uploaded_image_bytes", b"") or b"")
    if cached_bytes and cached_name == file_name and cached_size == file_size:
        st.session_state["refine_uploaded_widget_present"] = True
        st.session_state.pop("refine_upload_error", None)
        return None

    validated_bytes, validated_mime_type, validation_error = validate_refine_image_bytes(
        uploaded_file.getvalue(),
        input_mime_type=input_mime_type,
        file_name=file_name,
    )
    st.session_state["refine_uploaded_widget_present"] = True
    if validation_error:
        clear_cached_refine_upload(clear_error=False, reset_widget=True)
        st.session_state["refine_upload_error"] = validation_error
        return validation_error

    st.session_state["refine_uploaded_image_bytes"] = validated_bytes
    st.session_state["refine_uploaded_input_mime_type"] = validated_mime_type
    st.session_state["refine_uploaded_filename"] = file_name
    st.session_state["refine_uploaded_file_size"] = file_size
    st.session_state.pop("refine_upload_error", None)
    return None


def sanitize_refine_version_history() -> list[str]:
    """移除损坏的精修历史版本，避免旧坏数据反复破坏页面渲染。"""
    history = get_refine_version_history()
    if not history:
        return []

    valid_history: list[dict] = []
    removed_labels: list[str] = []
    removed_keys: set[str] = set()
    for entry in history:
        version_key = str(entry.get("version_key", "") or "")
        label = str(entry.get("label", version_key or "未知版本") or "未知版本")
        image_bytes = entry.get("image_bytes", b"") or b""
        _, preview_error = load_refine_preview_image(image_bytes, source_label=label)
        if preview_error:
            removed_labels.append(label)
            if version_key:
                removed_keys.add(version_key)
            continue
        valid_history.append(entry)

    if not removed_labels:
        return []

    st.session_state["refine_version_history"] = valid_history
    active_version_key = str(st.session_state.get("refine_active_version_key", "") or "")
    valid_keys = {
        str(entry.get("version_key", "") or "")
        for entry in valid_history
        if str(entry.get("version_key", "") or "")
    }
    if active_version_key and active_version_key not in valid_keys:
        st.session_state["refine_active_version_key"] = (
            str(valid_history[-1].get("version_key", "") or "") if valid_history else ""
        )
    latest_keys = list(st.session_state.get("refine_latest_version_keys", []) or [])
    st.session_state["refine_latest_version_keys"] = [
        key for key in latest_keys if str(key or "") not in removed_keys
    ]
    return removed_labels


def is_refine_auto_label(label: str, version_key: str) -> bool:
    """判断一个版本标签是否只是系统自动生成的占位名。"""
    normalized_label = clean_text(label).strip()
    normalized_key = str(version_key or "").strip()
    if not normalized_label:
        return True
    if normalized_key and normalized_label == normalized_key:
        return True
    return bool(
        REFINE_INTERNAL_VERSION_KEY_RE.fullmatch(normalized_label)
        or REFINE_DISPLAY_VERSION_LABEL_RE.fullmatch(normalized_label)
    )


def build_refine_version_display_map(history: list[dict] | None = None) -> dict[str, str]:
    """为精修历史构建更易读的显示名称映射。"""
    history = history if history is not None else get_refine_version_history()
    display_map: dict[str, str] = {}
    generated_version_index = 0
    extra_source_index = 0

    for history_index, entry in enumerate(history):
        version_key = str(entry.get("version_key", "") or "")
        if not version_key:
            continue

        stored_label = clean_text(str(entry.get("label", "") or "")).strip()
        source_label = clean_text(str(entry.get("source_label", "") or "")).strip()
        parent_version_key = str(entry.get("parent_version_key", "") or "")
        if parent_version_key:
            generated_version_index += 1
            auto_label = f"第{generated_version_index}版"
        elif history_index == 0:
            auto_label = "原图"
        elif source_label:
            auto_label = source_label
        else:
            extra_source_index += 1
            auto_label = f"输入图{extra_source_index}"

        display_map[version_key] = (
            stored_label if stored_label and not is_refine_auto_label(stored_label, version_key) else auto_label
        )
    return display_map


def get_refine_version_display_label(
    *,
    version_key: str = "",
    entry: dict | None = None,
    display_map: dict | None = None,
    history: list[dict] | None = None,
) -> str:
    """返回精修版本在界面中展示的友好名称。"""
    resolved_entry = entry
    resolved_history = history
    if resolved_entry is None and version_key:
        resolved_entry = find_refine_version_entry(version_key)
    if resolved_entry is not None and not version_key:
        version_key = str(resolved_entry.get("version_key", "") or "")
    if display_map is None:
        display_map = build_refine_version_display_map(resolved_history)
    if version_key and version_key in display_map:
        return display_map[version_key]
    if resolved_entry is not None:
        fallback_label = clean_text(str(resolved_entry.get("label", "") or "")).strip()
        if fallback_label:
            return fallback_label
    return clean_text(str(version_key or "版本") or "版本")


def get_next_refine_generated_label(history: list[dict] | None = None) -> str:
    """为新的精修结果生成下一个用户可读标签。"""
    history = history if history is not None else get_refine_version_history()
    generated_count = sum(
        1
        for entry in history
        if str(entry.get("parent_version_key", "") or "").strip()
    )
    return f"第{generated_count + 1}版"


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

CUSTOM_MODEL_OPTION = "自定义"
CUSTOM_CONNECTION_CREATE_OPTION = "__create_custom_connection__"
CONNECTION_DISCOVERY_STATUS_LABELS = {
    "success": "已刷新",
    "warning": "接口为空",
    "failed": "刷新失败",
    "skipped": "已跳过",
}
CONNECTION_PROBE_TARGET_LABELS = {
    "discovery": "模型发现",
    "text": "文本链路",
    "image": "图像链路",
}
CONNECTION_PROBE_TARGET_ICONS = {
    "discovery": "🧭",
    "text": "💬",
    "image": "🖼️",
}
CONNECTION_PROBE_STATUS_LABELS = {
    "success": "成功",
    "warning": "警告",
    "failed": "失败",
    "skipped": "已跳过",
    "running": "进行中",
}
TASK_OPTION_LABELS = {
    "diagram": "学术图解",
    "plot": "统计图表",
}
PIPELINE_OPTION_LABELS = {
    "demo_planner_critic": "标准流程（推荐）",
    "demo_full": "增强风格流程",
}
RETRIEVAL_NOTICE_LEVELS = {
    "auto": "info",
    "auto-full": "warning",
    "curated": "info",
    "random": "info",
    "none": "success",
}
RETRIEVAL_NOTICE_TEXT = {
    "auto": "默认推荐。只依据你的图注或可视化目标来匹配参考，成本低、速度快，适合大多数试跑。",
    "auto-full": "高精度模式。会把候选参考的完整内容交给模型判断，命中率更稳，但耗时和成本都会明显增加。",
    "curated": "固定参考集模式。使用你指定的 few-shot 配置，适合做复现实验、A/B 对照和开发调试。",
    "random": "随机样本模式。直接从参考池抽取示例，不额外调用检索推理，适合快速试跑。",
    "none": "纯生成模式。不加载任何参考样例，成本最低，适合先看基础出图效果。",
}


def build_provider_defaults():
    return build_all_provider_ui_defaults(
        base_dir=REPO_ROOT,
        model_config_data=model_config_data,
    )


def list_available_connections() -> list:
    return list_runtime_connections(
        base_dir=REPO_ROOT,
        model_config_data=model_config_data,
        include_disabled=False,
    )


def get_connection_options() -> list[tuple[str, str]]:
    options = []
    for connection in list_available_connections():
        suffix = "（内置）" if connection.builtin else "（自定义）"
        options.append((connection.connection_id, f"{connection.display_name} {suffix}"))
    options.append((CUSTOM_CONNECTION_CREATE_OPTION, "新建自定义连接"))
    return options


def find_connection_by_id(connection_id: str):
    normalized_id = normalize_connection_id(connection_id, default=str(connection_id or ""))
    for connection in list_available_connections():
        if connection.connection_id == normalized_id:
            return connection
    return None


_BACKGROUND_JOB_RUNTIME_FALLBACK = None


def render_preset_or_custom_model_input(
    label: str,
    preset_options: list[str],
    *,
    value_key: str,
    selector_key: str,
    custom_value_key: str,
    default_value: str,
    select_help: str,
    custom_help: str | None = None,
) -> str:
    normalized_options: list[str] = []
    seen_options: set[str] = set()
    for option in preset_options:
        normalized_option = str(option or "").strip()
        if (
            not normalized_option
            or normalized_option == CUSTOM_MODEL_OPTION
            or normalized_option in seen_options
        ):
            continue
        seen_options.add(normalized_option)
        normalized_options.append(normalized_option)

    default_model_name = str(default_value or "").strip()
    current_model_name = str(st.session_state.get(value_key, default_model_name) or "").strip()
    if value_key not in st.session_state:
        st.session_state[value_key] = current_model_name

    if custom_value_key not in st.session_state:
        st.session_state[custom_value_key] = (
            current_model_name if current_model_name and current_model_name not in normalized_options else ""
        )

    selector_options = [*normalized_options, CUSTOM_MODEL_OPTION]
    current_selector_value = st.session_state.get(selector_key)
    if current_selector_value not in selector_options:
        if current_model_name and current_model_name not in normalized_options:
            st.session_state[selector_key] = CUSTOM_MODEL_OPTION
        elif current_model_name in normalized_options:
            st.session_state[selector_key] = current_model_name
        elif default_model_name in normalized_options:
            st.session_state[selector_key] = default_model_name
        elif normalized_options:
            st.session_state[selector_key] = normalized_options[0]
        else:
            st.session_state[selector_key] = CUSTOM_MODEL_OPTION

    selected_option = st.selectbox(
        label,
        selector_options,
        key=selector_key,
        help=select_help,
    )

    if selected_option == CUSTOM_MODEL_OPTION:
        current_custom_value = str(st.session_state.get(custom_value_key, "") or "").strip()
        if (
            not current_custom_value
            and current_model_name
            and current_model_name not in normalized_options
        ):
            st.session_state[custom_value_key] = current_model_name
        custom_model_name = st.text_input(
            f"{label}（自定义）",
            key=custom_value_key,
            help=custom_help or select_help,
        )
        resolved_model_name = str(custom_model_name or "").strip()
    else:
        resolved_model_name = selected_option

    st.session_state[value_key] = resolved_model_name
    return resolved_model_name


def get_provider_ui_defaults(provider: str) -> dict[str, str]:
    provider_defaults = build_provider_defaults()
    normalized_provider = str(provider or DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER
    return provider_defaults.get(normalized_provider, provider_defaults[DEFAULT_PROVIDER])


def get_connection_ui_defaults(connection_id: str) -> dict[str, Any]:
    provider_defaults = build_provider_defaults()
    normalized_id = normalize_connection_id(connection_id, default=str(connection_id or DEFAULT_PROVIDER))
    return provider_defaults.get(normalized_id, provider_defaults[DEFAULT_PROVIDER])


def _build_connection_state_keys(prefix: str) -> dict[str, str]:
    return {
        "select_key": f"{prefix}_provider",
        "api_key": f"{prefix}_api_key",
        "base_url": f"{prefix}_base_url",
        "display_name": f"{prefix}_connection_display_name",
        "connection_id": f"{prefix}_connection_id",
        "api_key_env_var": f"{prefix}_api_key_env_var",
        "extra_headers_json": f"{prefix}_extra_headers_json",
        "supports_image": f"{prefix}_supports_image",
        "supports_text": f"{prefix}_supports_text",
        "enabled": f"{prefix}_connection_enabled",
        "persist_secret": f"{prefix}_persist_secret",
        "model_discovery_mode": f"{prefix}_model_discovery_mode",
        "model_allowlist": f"{prefix}_model_allowlist",
        "probe_results": f"{prefix}_probe_results",
        "discovered_models": f"{prefix}_discovered_models",
        "discovery_notice": f"{prefix}_discovery_notice",
    }


def _apply_connection_defaults_to_session(
    prefix: str,
    connection_id: str,
    *,
    defer_widget_updates: bool = False,
) -> dict[str, Any]:
    state_keys = _build_connection_state_keys(prefix)
    defaults = get_connection_ui_defaults(connection_id)
    updates = {
        state_keys["connection_id"]: defaults["connection_id"],
        state_keys["display_name"]: defaults["display_name"],
        state_keys["base_url"]: defaults["base_url"],
        state_keys["api_key_env_var"]: defaults.get("api_key_env_var", ""),
        state_keys["extra_headers_json"]: format_extra_headers_json(defaults.get("extra_headers")),
        state_keys["supports_text"]: bool(defaults.get("supports_text", True)),
        state_keys["supports_image"]: bool(defaults.get("supports_image", True)),
        state_keys["enabled"]: bool(defaults.get("enabled", True)),
        state_keys["persist_secret"]: True,
        state_keys["model_discovery_mode"]: defaults.get("model_discovery_mode", "manual"),
        state_keys["model_allowlist"]: "\n".join(defaults.get("model_allowlist", [])),
        state_keys["probe_results"]: dict(defaults.get("probe_results", {})),
        state_keys["discovered_models"]: list(defaults.get("model_allowlist", [])),
        state_keys["discovery_notice"]: "",
    }
    if defer_widget_updates:
        _queue_generation_widget_state_updates(updates)
    else:
        for key, value in updates.items():
            st.session_state[key] = value
    return defaults


def ensure_connection_editor_state(prefix: str, connection_id: str) -> dict[str, Any]:
    state_keys = _build_connection_state_keys(prefix)
    current_selected = str(st.session_state.get(state_keys["select_key"], connection_id) or connection_id)
    prev_key = f"{prefix}_prev_connection_id"
    if prev_key not in st.session_state:
        st.session_state[prev_key] = current_selected
    if st.session_state.get(prev_key) != current_selected:
        st.session_state[prev_key] = current_selected
        if current_selected == CUSTOM_CONNECTION_CREATE_OPTION:
            st.session_state[state_keys["connection_id"]] = "custom-openai"
            st.session_state[state_keys["display_name"]] = "自定义 OpenAI 兼容连接"
            st.session_state[state_keys["base_url"]] = ""
            st.session_state[state_keys["api_key_env_var"]] = ""
            st.session_state[state_keys["extra_headers_json"]] = ""
            st.session_state[state_keys["supports_text"]] = True
            st.session_state[state_keys["supports_image"]] = False
            st.session_state[state_keys["enabled"]] = True
            st.session_state[state_keys["persist_secret"]] = True
            st.session_state[state_keys["model_discovery_mode"]] = "hybrid"
            st.session_state[state_keys["model_allowlist"]] = ""
            st.session_state[state_keys["probe_results"]] = {}
            st.session_state[state_keys["discovered_models"]] = []
            st.session_state[state_keys["discovery_notice"]] = ""
        else:
            return _apply_connection_defaults_to_session(prefix, current_selected)
    if current_selected == CUSTOM_CONNECTION_CREATE_OPTION:
        return {
            "connection_id": str(st.session_state.get(state_keys["connection_id"], "custom-openai") or "custom-openai"),
            "display_name": str(st.session_state.get(state_keys["display_name"], "自定义 OpenAI 兼容连接") or "自定义 OpenAI 兼容连接"),
            "provider_type": CUSTOM_PROVIDER_TYPE,
            "api_key_default": str(st.session_state.get(state_keys["api_key"], "") or ""),
            "model_name": str(st.session_state.get(f"{prefix}_model_name", "") or ""),
            "image_model_name": str(st.session_state.get(f"{prefix}_image_model_name", "") or ""),
            "base_url": str(st.session_state.get(state_keys["base_url"], "") or ""),
            "builtin": False,
            "supports_text": bool(st.session_state.get(state_keys["supports_text"], True)),
            "supports_image": bool(st.session_state.get(state_keys["supports_image"], False)),
            "probe_results": dict(st.session_state.get(state_keys["probe_results"], {}) or {}),
        }
    return get_connection_ui_defaults(current_selected)


def hydrate_api_key_session_state(
    *,
    session_key: str,
    provider_defaults: dict[str, str],
) -> str:
    current_value = str(st.session_state.get(session_key, "") or "").strip()
    fallback_value = str(provider_defaults.get("api_key_default", "") or "").strip()
    if not current_value and fallback_value:
        st.session_state[session_key] = fallback_value
        return fallback_value
    return current_value


def get_api_key_widget_key(session_key: str) -> str:
    """为 API Key 输入框生成独立 widget key。"""
    return f"{session_key}_input"


def prepare_api_key_widget_state(
    *,
    session_key: str,
    clear_request_key: str,
    provider_defaults: dict[str, str],
) -> str:
    widget_key = get_api_key_widget_key(session_key)
    if st.session_state.pop(clear_request_key, False):
        st.session_state[session_key] = ""
        st.session_state[widget_key] = ""

    # 只在首次初始化时把业务值同步到 widget，避免每次 rerun 都用旧值覆盖
    # 用户刚输入但尚未提交（界面会显示 “Press Enter to apply”）的内容。
    if widget_key not in st.session_state:
        resolved_value = hydrate_api_key_session_state(
            session_key=session_key,
            provider_defaults=provider_defaults,
        )
        st.session_state[widget_key] = resolved_value
        return resolved_value

    current_widget_value = str(st.session_state.get(widget_key, "") or "").strip()
    if session_key not in st.session_state:
        st.session_state[session_key] = current_widget_value
    return current_widget_value


def sync_connection_runtime_input_state(
    *,
    prefix: str,
    selected_connection_id: str,
    provider_defaults: dict[str, Any],
) -> None:
    """连接切换时同步 API Key 和模型输入，避免沿用上一条连接的残留状态。"""
    tracked_connection_key = f"{prefix}_runtime_input_connection_id"
    normalized_selected = str(selected_connection_id or "").strip()
    if str(st.session_state.get(tracked_connection_key, "") or "").strip() == normalized_selected:
        return

    api_key_session_key = f"{prefix}_api_key"
    api_key_value = str(provider_defaults.get("api_key_default", "") or "").strip()
    st.session_state[api_key_session_key] = api_key_value
    st.session_state[get_api_key_widget_key(api_key_session_key)] = api_key_value

    st.session_state[f"{prefix}_model_name"] = str(provider_defaults.get("model_name", "") or "").strip()
    st.session_state[f"{prefix}_image_model_name"] = str(provider_defaults.get("image_model_name", "") or "").strip()
    st.session_state.pop(f"{prefix}_model_name_selector", None)
    st.session_state.pop(f"{prefix}_model_name_custom", None)
    st.session_state.pop(f"{prefix}_image_model_name_selector", None)
    st.session_state.pop(f"{prefix}_image_model_name_custom", None)
    st.session_state[tracked_connection_key] = normalized_selected


def persist_provider_api_key_input(provider: str, api_key: str) -> None:
    normalized_value = str(api_key or "").strip()
    if not normalized_value:
        return
    normalized_provider = normalize_connection_id(provider, default=str(provider or ""))
    if normalized_provider in BUILTIN_CONNECTION_IDS:
        write_provider_api_key(normalized_provider, normalized_value, base_dir=REPO_ROOT)
    else:
        write_custom_provider_api_key(normalized_provider, normalized_value, base_dir=REPO_ROOT)


def request_clear_provider_api_key(
    *,
    provider: str,
    session_key: str,
    clear_request_key: str,
) -> None:
    normalized_provider = normalize_connection_id(provider, default=str(provider or ""))
    if normalized_provider in BUILTIN_CONNECTION_IDS:
        delete_provider_api_key(normalized_provider, base_dir=REPO_ROOT)
    else:
        delete_custom_provider_api_key(normalized_provider, base_dir=REPO_ROOT)
    st.session_state[clear_request_key] = True
    st.rerun()


def build_api_key_storage_notice(
    provider_defaults: dict[str, str],
    *,
    persist_secret: bool = True,
    allow_local_persist: bool = True,
) -> str:
    if not persist_secret:
        return "当前输入仅在本次会话生效，不会写入本地 txt。"
    if not allow_local_persist:
        return "当前是未保存的自定义连接草稿；API Key 会先保留在本次会话，保存连接后才会写入本地 txt。"
    if str(provider_defaults.get("api_key_default", "") or "").strip():
        return "已在本机保存当前 Provider 的密钥，刷新页面后仍会保留。"
    return "密钥只保存在当前电脑；输入后会自动写入本地 txt。"


def render_image_generation_option_controls(
    *,
    provider_type: str,
    image_model_name: str,
    aspect_ratio: str,
    image_resolution: str,
    widget_prefix: str = "tab1",
) -> dict[str, Any]:
    capabilities = get_image_model_capabilities(provider_type, image_model_name)
    st.caption(f"当前图像能力：{capabilities.model_family}")

    default_options = normalize_image_generation_options(
        provider_type=provider_type,
        model_name=image_model_name,
        aspect_ratio=aspect_ratio,
        image_resolution=image_resolution,
    )
    size_choices = list(capabilities.size_options)
    default_size = default_options.size
    custom_size_key = f"{widget_prefix}_image_custom_size"
    use_custom_size = capabilities.supports_custom_size and default_size not in size_choices
    if capabilities.supports_custom_size:
        size_choices.append("自定义像素尺寸")
        if custom_size_key not in st.session_state:
            st.session_state[custom_size_key] = default_size if default_size != "auto" else "2048x2048"

    selected_size_label = st.selectbox(
        "OpenAI 输出尺寸",
        size_choices,
        index=size_choices.index("自定义像素尺寸") if use_custom_size else size_choices.index(default_size),
        key=f"{widget_prefix}_image_size",
        help="GPT Image 2 可使用 auto、常用尺寸或满足约束的自定义像素尺寸。",
    )
    size = selected_size_label
    if capabilities.supports_custom_size and selected_size_label == "自定义像素尺寸":
        requested_custom_size = st.text_input(
            "自定义像素尺寸",
            key=custom_size_key,
            help="格式为 宽x高，例如 2304x1024；宽高需为 16 的倍数，单边不超过 3840。",
        )
        if is_valid_custom_image_size(requested_custom_size, capabilities):
            size = str(requested_custom_size).strip()
        else:
            st.warning("当前自定义尺寸不满足 GPT Image 2 约束，已回退为 auto。")
            size = "auto"
    quality = st.selectbox(
        "OpenAI 渲染质量",
        list(capabilities.quality_options),
        index=list(capabilities.quality_options).index(default_options.quality),
        key=f"{widget_prefix}_image_quality",
    )
    background = st.selectbox(
        "OpenAI 背景",
        list(capabilities.background_options),
        index=list(capabilities.background_options).index(default_options.background),
        key=f"{widget_prefix}_image_background",
        help="GPT Image 2 当前不支持 transparent，界面会自动隐藏该选项。",
    )
    output_format = st.selectbox(
        "OpenAI 输出格式",
        list(capabilities.output_format_options),
        index=list(capabilities.output_format_options).index(default_options.output_format),
        key=f"{widget_prefix}_image_output_format",
    )
    output_compression = None
    if capabilities.supports_output_compression and output_format in {"jpeg", "webp"}:
        output_compression = int(
            st.slider(
                "OpenAI 输出压缩",
                min_value=0,
                max_value=100,
                value=100,
                key=f"{widget_prefix}_image_output_compression",
                help="仅 jpeg/webp 生效；png 不发送该参数。",
            )
        )
    moderation = "auto"
    if capabilities.supports_moderation:
        moderation = st.selectbox(
            "OpenAI 审核强度",
            ["auto", "low"],
            key=f"{widget_prefix}_image_moderation",
        )
    input_fidelity = "auto"
    if capabilities.supports_input_fidelity:
        input_fidelity = st.selectbox(
            "OpenAI 参考图保真度",
            ["auto", "low", "high"],
            key=f"{widget_prefix}_image_input_fidelity",
            help="仅带参考图的编辑链路生效。",
        )
    stream = False
    partial_images = 0
    if capabilities.supports_stream:
        stream = st.checkbox(
            "启用流式图像事件",
            value=False,
            key=f"{widget_prefix}_image_stream",
            help="当前仅在后台消费最终图；partial images 会额外消耗图像输出 token。",
        )
        if stream and capabilities.supports_partial_images:
            partial_images = int(
                st.slider(
                    "流式预览张数",
                    min_value=0,
                    max_value=3,
                    value=0,
                    key=f"{widget_prefix}_image_partial_images",
                    help="仅用于流式请求；当前不会在前端实时展示中间图。",
                )
            )

    options = normalize_image_generation_options(
        provider_type=provider_type,
        model_name=image_model_name,
        aspect_ratio=aspect_ratio,
        image_resolution=image_resolution,
        raw_options={
            "size": size,
            "quality": quality,
            "background": background,
            "output_format": output_format,
            "output_compression": output_compression,
            "moderation": moderation,
            "input_fidelity": input_fidelity,
            "stream": stream,
            "partial_images": partial_images,
        },
    )
    return options.to_dict()


def get_selected_connection_runtime_id(prefix: str, selected_connection_id: str) -> str:
    if selected_connection_id != CUSTOM_CONNECTION_CREATE_OPTION:
        return normalize_connection_id(selected_connection_id, default=DEFAULT_PROVIDER)
    state_keys = _build_connection_state_keys(prefix)
    draft_connection_id = str(st.session_state.get(state_keys["connection_id"], "") or "").strip()
    return normalize_connection_id(draft_connection_id, default="custom-openai")


def parse_extra_headers_json_safe(raw_text: str) -> tuple[dict[str, str], str]:
    normalized_text = str(raw_text or "").strip()
    if not normalized_text:
        return {}, ""
    try:
        return parse_extra_headers_json(normalized_text), ""
    except Exception as exc:
        return {}, f"额外请求头格式错误：{exc}"


def build_connection_draft(
    *,
    prefix: str,
    selected_connection_id: str,
    api_key: str,
    model_name: str,
    image_model_name: str,
) -> ProviderConnection:
    state_keys = _build_connection_state_keys(prefix)
    runtime_connection_id = get_selected_connection_runtime_id(prefix, selected_connection_id)
    extra_headers, error_message = parse_extra_headers_json_safe(
        str(st.session_state.get(state_keys["extra_headers_json"], "") or "")
    )
    if error_message:
        raise ValueError(error_message)

    if selected_connection_id != CUSTOM_CONNECTION_CREATE_OPTION:
        runtime_settings = resolve_runtime_settings(
            runtime_connection_id,
            connection_id=runtime_connection_id if runtime_connection_id not in BUILTIN_CONNECTION_IDS else "",
            api_key=api_key,
            model_name=model_name,
            image_model_name=image_model_name,
            base_url=str(st.session_state.get(state_keys["base_url"], "") or ""),
            extra_headers=extra_headers,
            base_dir=REPO_ROOT,
            model_config_data=model_config_data,
        )
        if runtime_settings.provider_connection is None:
            raise ValueError("未能解析当前连接。")
        return runtime_settings.provider_connection

    model_allowlist = [
        line.strip()
        for line in str(st.session_state.get(state_keys["model_allowlist"], "") or "").splitlines()
        if line.strip()
    ]
    return ProviderConnection(
        connection_id=runtime_connection_id,
        display_name=str(st.session_state.get(state_keys["display_name"], "") or runtime_connection_id).strip() or runtime_connection_id,
        provider_type=CUSTOM_PROVIDER_TYPE,
        protocol_family="openai",
        base_url=str(st.session_state.get(state_keys["base_url"], "") or "").strip(),
        api_key_env_var=str(st.session_state.get(state_keys["api_key_env_var"], "") or "").strip(),
        text_model=str(model_name or "").strip(),
        image_model=str(image_model_name or "").strip(),
        model_discovery_mode=str(st.session_state.get(state_keys["model_discovery_mode"], "hybrid") or "hybrid").strip() or "hybrid",
        model_allowlist=tuple(model_allowlist),
        extra_headers=extra_headers,
        supports_text=bool(st.session_state.get(state_keys["supports_text"], True)),
        supports_image=bool(st.session_state.get(state_keys["supports_image"], False)),
        enabled=bool(st.session_state.get(state_keys["enabled"], True)),
        builtin=False,
        api_key=str(api_key or "").strip(),
        probe_results=dict(st.session_state.get(state_keys["probe_results"], {}) or {}),
    )


def format_repo_relative_path(path_value: str | Path | None, *, base_dir: Path | None = None) -> str:
    if not path_value:
        return ""
    path_obj = Path(path_value)
    root_candidates = [base_dir or REPO_ROOT, Path.cwd()]
    for root_dir in root_candidates:
        try:
            relative_path = path_obj.resolve(strict=False).relative_to(
                Path(root_dir).resolve(strict=False)
            )
            return relative_path.as_posix()
        except Exception:
            continue
    return path_obj.as_posix()


def ensure_session_choice_state(
    key: str,
    options: list[str],
    default_value: str,
) -> str:
    normalized_options = [option for option in options if option is not None]
    if not normalized_options:
        raise ValueError(f"{key} 缺少可用选项。")

    current_value = st.session_state.get(key, default_value)
    if current_value not in normalized_options:
        current_value = default_value if default_value in normalized_options else normalized_options[0]

    if st.session_state.get(key) != current_value:
        st.session_state[key] = current_value
    return str(current_value)


def ensure_session_int_state(
    key: str,
    default_value: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    raw_value = st.session_state.get(key, default_value)
    try:
        normalized_value = int(raw_value)
    except (TypeError, ValueError):
        normalized_value = int(default_value)

    if min_value is not None:
        normalized_value = max(int(min_value), normalized_value)
    if max_value is not None:
        normalized_value = min(int(max_value), normalized_value)

    if st.session_state.get(key) != normalized_value:
        st.session_state[key] = normalized_value
    return normalized_value


def render_provider_api_key_controls(
    *,
    provider: str,
    provider_defaults: dict[str, str],
    session_key: str,
    clear_request_key: str,
    clear_button_key: str,
    persist_secret: bool = True,
    allow_local_persist: bool = True,
) -> str:
    widget_key = get_api_key_widget_key(session_key)
    prepare_api_key_widget_state(
        session_key=session_key,
        clear_request_key=clear_request_key,
        provider_defaults=provider_defaults,
    )
    api_key = st.text_input(
        provider_defaults["api_key_label"],
        type="password",
        key=widget_key,
        help=provider_defaults["api_key_help"],
    )
    normalized_api_key = str(api_key or "").strip()
    if str(st.session_state.get(session_key, "") or "").strip() != normalized_api_key:
        st.session_state[session_key] = normalized_api_key
    notice_col, clear_col = st.columns([4, 1], vertical_alignment="center")
    with notice_col:
        st.caption(
            build_api_key_storage_notice(
                provider_defaults,
                persist_secret=persist_secret,
                allow_local_persist=allow_local_persist,
            )
        )
    with clear_col:
        if st.button(
            "清除",
            key=clear_button_key,
            width="stretch",
            help="删除当前 Provider 已保存的本地 txt 密钥，并清空输入框。",
        ):
            request_clear_provider_api_key(
                provider=provider,
                session_key=session_key,
                clear_request_key=clear_request_key,
            )
    if persist_secret and allow_local_persist:
        persist_provider_api_key_input(provider, normalized_api_key)
    return normalized_api_key


def render_connection_probe_results(prefix: str) -> None:
    state_keys = _build_connection_state_keys(prefix)
    probe_results = dict(st.session_state.get(state_keys["probe_results"], {}) or {})
    notice_text = str(st.session_state.get(state_keys["discovery_notice"], "") or "").strip()
    if notice_text:
        st.markdown(
            f'<div class="pb-inline-status is-success"><strong>最近诊断</strong><span>{html.escape(notice_text)}</span></div>',
            unsafe_allow_html=True,
        )
    if not probe_results:
        return

    with st.container(border=True):
        st.markdown('<div class="pb-probe-title">连接诊断</div>', unsafe_allow_html=True)
        for target in ("discovery", "text", "image"):
            payload = probe_results.get(target)
            if not isinstance(payload, dict):
                continue
            icon = CONNECTION_PROBE_TARGET_ICONS.get(target, "•")
            target_label = CONNECTION_PROBE_TARGET_LABELS.get(target, target)
            status = str(payload.get("status", "") or "") or "unknown"
            status_label = CONNECTION_PROBE_STATUS_LABELS.get(status, status)
            message = str(payload.get("message", "") or "").strip()
            error_type = str(payload.get("error_type", "") or "").strip()
            tested_model = str(payload.get("tested_model", "") or "").strip()
            raw_excerpt = str(payload.get("raw_excerpt", "") or "").strip()
            latency_ms = int(payload.get("latency_ms", 0) or 0)
            discovered_models = list(payload.get("discovered_models", []) or [])
            http_status = int(payload.get("http_status", 0) or 0)
            timestamp = str(payload.get("timestamp", "") or "").strip()
            meta_parts = []
            if tested_model:
                meta_parts.append(f"模型 {tested_model}")
            if discovered_models:
                preview = ", ".join(discovered_models[:2])
                suffix = f" 等 {len(discovered_models)} 个" if len(discovered_models) > 2 else ""
                meta_parts.append(f"发现 {preview}{suffix}")
            if latency_ms > 0:
                meta_parts.append(f"{latency_ms} ms")
            if error_type:
                meta_parts.append(f"分类 {error_type}")
            badge_class = f"pb-probe-badge-{status if status in {'success', 'warning', 'failed', 'skipped'} else 'skipped'}"
            summary_text = html.escape(message or status_label)
            meta_text = html.escape(" · ".join(meta_parts))
            st.markdown(
                f"""
                <div class="pb-probe-row">
                  <div class="pb-probe-head">
                    <div class="pb-probe-label">{icon} {html.escape(target_label)}</div>
                    <span class="pb-probe-badge {badge_class}">{html.escape(status_label)}</span>
                  </div>
                  <div class="pb-probe-summary">{summary_text}</div>
                  {f'<div class="pb-probe-meta">{meta_text}</div>' if meta_text else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )
            has_detail_section = bool(raw_excerpt or timestamp or http_status or discovered_models)
            if has_detail_section:
                with st.expander(f"{target_label}详情", expanded=False):
                    if raw_excerpt:
                        st.code(raw_excerpt, language="text")
                    if discovered_models:
                        st.caption(f"完整模型列表：{', '.join(discovered_models)}")
                    detail_parts = []
                    if http_status:
                        detail_parts.append(f"HTTP {http_status}")
                    if timestamp:
                        detail_parts.append(timestamp)
                    if detail_parts:
                        st.caption(" · ".join(detail_parts))


def _set_connection_probe_notice(prefix: str, message: str) -> None:
    state_keys = _build_connection_state_keys(prefix)
    st.session_state[state_keys["discovery_notice"]] = str(message or "").strip()


def _build_connection_stage_message(target: str, status: str) -> str:
    target_label = CONNECTION_PROBE_TARGET_LABELS.get(target, target)
    if status == "running":
        return f"正在检查{target_label}"
    status_label = CONNECTION_PROBE_STATUS_LABELS.get(status, status)
    return f"{target_label}：{status_label}"


def _build_probe_summary_message(serialized: dict[str, dict[str, Any]]) -> str:
    parts = []
    for target in ("discovery", "text", "image"):
        payload = serialized.get(target, {})
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status", "") or "unknown")
        parts.append(
            f"{CONNECTION_PROBE_TARGET_LABELS.get(target, target)}={CONNECTION_PROBE_STATUS_LABELS.get(status, status)}"
        )
    return " | ".join(parts)


def emit_connection_action_feedback(message: str) -> None:
    toast = getattr(st, "toast", None)
    if callable(toast):
        toast(message)
    else:
        st.info(message)


def _render_connection_progress_message(placeholder, *, label: str, tone: str = "running") -> None:
    css_class = {
        "running": "pb-inline-status is-running",
        "success": "pb-inline-status is-success",
        "error": "pb-inline-status is-error",
    }.get(tone, "pb-inline-status")
    placeholder.markdown(
        f'<div class="{css_class}"><strong>连接操作</strong><span>{html.escape(label)}</span></div>',
        unsafe_allow_html=True,
    )


def save_connection_draft(
    *,
    prefix: str,
    selected_connection_id: str,
    api_key: str,
    model_name: str,
    image_model_name: str,
) -> tuple[bool, str, dict[str, Any]]:
    state_keys = _build_connection_state_keys(prefix)
    try:
        extra_headers = parse_extra_headers_json(
            str(st.session_state.get(state_keys["extra_headers_json"], "") or "")
        )
    except Exception as exc:
        return False, f"额外请求头格式错误：{exc}", {}

    if selected_connection_id in BUILTIN_CONNECTION_IDS:
        normalized_api_key = str(api_key or "").strip()
        if bool(st.session_state.get(state_keys["persist_secret"], True)):
            write_provider_api_key(selected_connection_id, normalized_api_key, base_dir=REPO_ROOT)
        else:
            delete_provider_api_key(selected_connection_id, base_dir=REPO_ROOT)
        defaults = _apply_connection_defaults_to_session(
            prefix,
            selected_connection_id,
            defer_widget_updates=True,
        )
        return True, "已保存内置连接配置。", defaults

    payload = {
        "connection_id": str(st.session_state.get(state_keys["connection_id"], "") or ""),
        "display_name": str(st.session_state.get(state_keys["display_name"], "") or ""),
        "base_url": str(st.session_state.get(state_keys["base_url"], "") or ""),
        "api_key_env_var": str(st.session_state.get(state_keys["api_key_env_var"], "") or ""),
        "text_model": model_name,
        "image_model": image_model_name,
        "model_discovery_mode": str(st.session_state.get(state_keys["model_discovery_mode"], "hybrid") or "hybrid"),
        "model_allowlist": [
            line.strip()
            for line in str(st.session_state.get(state_keys["model_allowlist"], "") or "").splitlines()
            if line.strip()
        ],
        "extra_headers": extra_headers,
        "supports_text": bool(st.session_state.get(state_keys["supports_text"], True)),
        "supports_image": bool(st.session_state.get(state_keys["supports_image"], False)),
        "enabled": bool(st.session_state.get(state_keys["enabled"], True)),
    }
    try:
        connection = upsert_custom_connection(
            payload,
            api_key=api_key,
            persist_secret=bool(st.session_state.get(state_keys["persist_secret"], True)),
            base_dir=REPO_ROOT,
        )
    except Exception as exc:
        return False, f"保存连接失败：{exc}", {}
    _queue_generation_widget_state_updates(
        {
            state_keys["select_key"]: connection.connection_id,
        }
    )
    defaults = _apply_connection_defaults_to_session(
        prefix,
        connection.connection_id,
        defer_widget_updates=True,
    )
    return True, f"已保存自定义连接：{connection.display_name}", defaults


def refresh_connection_models(
    *,
    prefix: str,
    connection_id: str,
    api_key: str,
    model_name: str,
    image_model_name: str,
    stage_callback: Callable[[str, str], None] | None = None,
) -> tuple[dict[str, Any], str]:
    state_keys = _build_connection_state_keys(prefix)
    try:
        connection = build_connection_draft(
            prefix=prefix,
            selected_connection_id=connection_id,
            api_key=api_key,
            model_name=model_name,
            image_model_name=image_model_name,
        )
    except Exception as exc:
        message = str(exc)
        _set_connection_probe_notice(prefix, message)
        return {}, message
    if stage_callback is not None:
        stage_callback("discovery", "running")
    result = run_async_probe(discover_models(connection))
    if stage_callback is not None:
        stage_callback("discovery", result.status)
    st.session_state[state_keys["probe_results"]] = {
        **dict(st.session_state.get(state_keys["probe_results"], {}) or {}),
        "discovery": result.to_dict(),
    }
    st.session_state[state_keys["discovered_models"]] = list(result.discovered_models)
    write_connection_probe_result(connection.connection_id, result, base_dir=REPO_ROOT)
    label = CONNECTION_DISCOVERY_STATUS_LABELS.get(result.status, result.status)
    message = f"{label}：{result.message}"
    _set_connection_probe_notice(prefix, message)
    return st.session_state[state_keys["probe_results"]], message


def test_connection_draft(
    *,
    prefix: str,
    connection_id: str,
    api_key: str,
    model_name: str,
    image_model_name: str,
    stage_callback: Callable[[str, str], None] | None = None,
) -> tuple[dict[str, Any], str]:
    state_keys = _build_connection_state_keys(prefix)
    try:
        connection = build_connection_draft(
            prefix=prefix,
            selected_connection_id=connection_id,
            api_key=api_key,
            model_name=model_name,
            image_model_name=image_model_name,
        )
    except Exception as exc:
        message = str(exc)
        _set_connection_probe_notice(prefix, message)
        return {}, message
    probe_results = run_async_probe(probe_connection(connection, stage_callback=stage_callback))
    serialized = {key: value.to_dict() for key, value in probe_results.items()}
    st.session_state[state_keys["probe_results"]] = serialized
    discovery_payload = serialized.get("discovery", {})
    if isinstance(discovery_payload, dict):
        st.session_state[state_keys["discovered_models"]] = list(discovery_payload.get("discovered_models", []) or [])
    for result in probe_results.values():
        write_connection_probe_result(connection.connection_id, result, base_dir=REPO_ROOT)
    summary = _build_probe_summary_message(serialized)
    _set_connection_probe_notice(prefix, summary)
    return serialized, summary


def run_refresh_models_with_status(
    *,
    prefix: str,
    connection_id: str,
    api_key: str,
    model_name: str,
    image_model_name: str,
) -> tuple[dict[str, Any], str]:
    status_placeholder = st.empty()

    def stage_callback(target: str, status: str) -> None:
        if target != "discovery":
            return
        _render_connection_progress_message(
            status_placeholder,
            label=_build_connection_stage_message(target, status),
            tone="running" if status == "running" else ("error" if status == "failed" else "success"),
        )

    with st.spinner("正在刷新..."):
        probe_results, message = refresh_connection_models(
            prefix=prefix,
            connection_id=connection_id,
            api_key=api_key,
            model_name=model_name,
            image_model_name=image_model_name,
            stage_callback=stage_callback,
        )
        result_payload = probe_results.get("discovery", {})
        result_status = str(result_payload.get("status", "") or "")
        _render_connection_progress_message(
            status_placeholder,
            label=message,
            tone="error" if (not probe_results or result_status == "failed") else "success",
        )
        return probe_results, message


def run_connection_probe_with_status(
    *,
    prefix: str,
    connection_id: str,
    api_key: str,
    model_name: str,
    image_model_name: str,
) -> tuple[dict[str, Any], str]:
    status_placeholder = st.empty()

    def stage_callback(target: str, status: str) -> None:
        _render_connection_progress_message(
            status_placeholder,
            label=_build_connection_stage_message(target, status),
            tone="running" if status == "running" else ("error" if status == "failed" else "success"),
        )

    with st.spinner("正在测试..."):
        probe_results, message = test_connection_draft(
            prefix=prefix,
            connection_id=connection_id,
            api_key=api_key,
            model_name=model_name,
            image_model_name=image_model_name,
            stage_callback=stage_callback,
        )
        final_state = "complete"
        if not probe_results:
            final_state = "error"
        elif any(
            str((probe_results.get(target) or {}).get("status", "") or "") == "failed"
            for target in ("text",)
        ):
            final_state = "error"
        _render_connection_progress_message(
            status_placeholder,
            label=message or "连接测试完成",
            tone=final_state,
        )
        return probe_results, message


def inject_refine_tab_sidebar_autocollapse_hook() -> None:
    if not hasattr(st, "html"):
        return
    st.html(
        """
<script>
(() => {
  const cleanupKey = "__paperbananaRefineSidebarAutoCollapse";
  const stateKey = "__paperbananaRefineSidebarAutoCollapseState";
  try {
    window[cleanupKey]?.cleanup?.();
  } catch (error) {
    console.warn("cleanup previous sidebar observer failed", error);
  }

  const normalizeText = (value) => (value || "").replace(/\\s+/g, " ").trim();
  const findSidebar = () => document.querySelector("section.stSidebar");
  if (!findSidebar()) {
    return;
  }
  const syncState = window[stateKey] || { autoCollapsed: false };
  window[stateKey] = syncState;

  const findActiveTab = () =>
    Array.from(document.querySelectorAll('button[role="tab"]')).find(
      (tab) => tab.getAttribute("aria-selected") === "true"
    );

  const isRefineTabActive = () => {
    const activeTab = findActiveTab();
    return Boolean(activeTab) && normalizeText(activeTab.innerText).includes("精修图像");
  };

  const findCollapseButton = () =>
    findSidebar()?.querySelector('button[kind="headerNoPadding"]') ||
    findSidebar()?.querySelector('button[kind="header"]') ||
    document.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
    document.querySelector('[data-testid="stSidebarCollapseButton"]');

  const findExpandButton = () =>
    document.querySelector('[data-testid="stExpandSidebarButton"] button') ||
    document.querySelector('[data-testid="stExpandSidebarButton"]');

  let frameHandle = null;
  const syncSidebar = () => {
    if (frameHandle !== null) {
      cancelAnimationFrame(frameHandle);
    }
    frameHandle = requestAnimationFrame(() => {
      frameHandle = null;
      const sidebar = findSidebar();
      if (!sidebar || !sidebar.isConnected) {
        return;
      }
      const isExpanded = sidebar.getAttribute("aria-expanded") === "true";
      if (isRefineTabActive()) {
        if (!isExpanded) {
          return;
        }
        const collapseButton = findCollapseButton();
        if (!collapseButton) {
          return;
        }
        syncState.autoCollapsed = true;
        collapseButton.click();
        return;
      }
      if (!isExpanded && syncState.autoCollapsed) {
        const expandButton = findExpandButton();
        if (!expandButton) {
          return;
        }
        syncState.autoCollapsed = false;
        expandButton.click();
        return;
      }
      if (isExpanded) {
        syncState.autoCollapsed = false;
      }
    });
  };

  const observer = new MutationObserver(() => {
    syncSidebar();
  });

  observer.observe(document.body, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ["aria-selected", "aria-expanded"],
  });
  syncSidebar();

  window[cleanupKey] = {
    cleanup() {
      observer.disconnect();
      if (frameHandle !== null) {
        cancelAnimationFrame(frameHandle);
      }
    },
  };
})();
</script>
        """,
        unsafe_allow_javascript=True,
        width="content",
    )


def format_candidate_slot_label(candidate_id: int | str, *, fallback_index: int = 0) -> str:
    raw_candidate_id = str(candidate_id or "").strip()
    try:
        display_index = int(raw_candidate_id) + 1
    except Exception:
        display_index = max(1, int(fallback_index) + 1)
    label = f"候选 {display_index:02d}"
    if raw_candidate_id and raw_candidate_id != str(display_index - 1):
        return f"{label} | 标识 `{raw_candidate_id}`"
    return label


def initialize_curated_profile_state(
    *,
    profile_key: str,
    input_key: str,
    default_value: str = DEFAULT_CURATED_PROFILE,
) -> str:
    normalized_profile = normalize_curated_profile_name(
        st.session_state.get(profile_key, default_value)
    )
    if st.session_state.get(profile_key) != normalized_profile:
        st.session_state[profile_key] = normalized_profile
    if input_key not in st.session_state:
        st.session_state[input_key] = normalized_profile
    return normalized_profile


def resolve_curated_profile_input(raw_value: str, *, profile_key: str) -> str:
    normalized_profile = normalize_curated_profile_name(raw_value)
    if st.session_state.get(profile_key) != normalized_profile:
        st.session_state[profile_key] = normalized_profile
    return normalized_profile


def _build_background_job_runtime() -> dict:
    return {
        "generation_executor": ThreadPoolExecutor(max_workers=1),
        "generation_jobs_lock": threading.Lock(),
        "generation_jobs": {},
        "refine_executor": ThreadPoolExecutor(max_workers=2),
        "refine_jobs_lock": threading.Lock(),
        "refine_jobs": {},
        "demo_ui_state_lock": threading.Lock(),
        "demo_ui_state": {},
    }


def _normalize_background_job_runtime(runtime: dict | None) -> dict:
    safe_runtime = runtime if isinstance(runtime, dict) else {}
    safe_runtime.setdefault("generation_executor", ThreadPoolExecutor(max_workers=1))
    safe_runtime.setdefault("generation_jobs_lock", threading.Lock())
    safe_runtime.setdefault("generation_jobs", {})
    safe_runtime.setdefault("refine_executor", ThreadPoolExecutor(max_workers=2))
    safe_runtime.setdefault("refine_jobs_lock", threading.Lock())
    safe_runtime.setdefault("refine_jobs", {})
    safe_runtime.setdefault("demo_ui_state_lock", threading.Lock())
    safe_runtime.setdefault("demo_ui_state", {})
    return safe_runtime


if hasattr(st, "cache_resource"):
    @st.cache_resource(show_spinner=False)
    def get_background_job_runtime() -> dict:
        return _normalize_background_job_runtime(_build_background_job_runtime())
else:
    def get_background_job_runtime() -> dict:
        global _BACKGROUND_JOB_RUNTIME_FALLBACK
        if _BACKGROUND_JOB_RUNTIME_FALLBACK is None:
            _BACKGROUND_JOB_RUNTIME_FALLBACK = _build_background_job_runtime()
        return _normalize_background_job_runtime(_BACKGROUND_JOB_RUNTIME_FALLBACK)


BACKGROUND_JOB_RUNTIME = _normalize_background_job_runtime(get_background_job_runtime())
GENERATION_JOB_EXECUTOR = BACKGROUND_JOB_RUNTIME["generation_executor"]
GENERATION_JOBS_LOCK = BACKGROUND_JOB_RUNTIME["generation_jobs_lock"]
GENERATION_JOBS = BACKGROUND_JOB_RUNTIME["generation_jobs"]
REFINE_JOB_EXECUTOR = BACKGROUND_JOB_RUNTIME["refine_executor"]
REFINE_JOBS_LOCK = BACKGROUND_JOB_RUNTIME["refine_jobs_lock"]
REFINE_JOBS = BACKGROUND_JOB_RUNTIME["refine_jobs"]
DEMO_UI_STATE_LOCK = BACKGROUND_JOB_RUNTIME["demo_ui_state_lock"]
DEMO_UI_STATE = BACKGROUND_JOB_RUNTIME["demo_ui_state"]


@dataclass
class GenerationJobState:
    job_id: str
    dataset_name: str
    task_name: str
    exp_mode: str
    retrieval_setting: str
    curated_profile: str
    provider: str
    model_name: str
    image_model_name: str
    concurrency_mode: str
    max_concurrent: int
    requested_candidates: int
    max_critic_rounds: int
    aspect_ratio: str
    image_resolution: str
    content: str
    visual_intent: str
    connection_id: str = ""
    provider_display_name: str = ""
    status: str = "running"
    progress_done: int = 0
    progress_total: int = 0
    effective_concurrent: int = 0
    estimated_batches: int = 0
    status_history: list[str] = field(default_factory=list)
    event_history: list[dict] = field(default_factory=list)
    candidate_stage_map: dict[str, str] = field(default_factory=dict)
    candidate_snapshots: dict[str, dict] = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    elapsed_seconds: float = 0.0
    cancel_requested: bool = False
    json_file: str = ""
    bundle_file: str = ""
    future: Future | None = field(default=None, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict:
        with self.lock:
            _sync_generation_job_progress_locked(self)
            worker_done = self.future.done() if self.future is not None else self.status in {"completed", "cancelled", "failed", "interrupted"}
            return {
                "job_id": self.job_id,
                "dataset_name": self.dataset_name,
                "task_name": self.task_name,
                "exp_mode": self.exp_mode,
                "retrieval_setting": self.retrieval_setting,
                "curated_profile": self.curated_profile,
                "provider": self.provider,
                "connection_id": self.connection_id,
                "provider_display_name": self.provider_display_name,
                "model_name": self.model_name,
                "image_model_name": self.image_model_name,
                "concurrency_mode": self.concurrency_mode,
                "max_concurrent": self.max_concurrent,
                "requested_candidates": self.requested_candidates,
                "max_critic_rounds": self.max_critic_rounds,
                "aspect_ratio": self.aspect_ratio,
                "image_resolution": self.image_resolution,
                "content": self.content,
                "visual_intent": self.visual_intent,
                "status": self.status,
                "progress_done": self.progress_done,
                "progress_total": self.progress_total,
                "effective_concurrent": self.effective_concurrent,
                "estimated_batches": self.estimated_batches,
                "status_history": list(self.status_history),
                "event_history": [dict(item) for item in self.event_history],
                "candidate_stage_map": dict(self.candidate_stage_map),
                "candidate_snapshots": {
                    str(candidate_id): {
                        **dict(snapshot),
                        "result": dict(snapshot["result"]) if isinstance(snapshot.get("result"), dict) else snapshot.get("result"),
                    }
                    for candidate_id, snapshot in self.candidate_snapshots.items()
                },
                "event_timeline": [dict(item) for item in self.event_history],
                "results": list(self.results),
                "summary": dict(self.summary),
                "failures": list(self.failures),
                "error": self.error,
                "created_at": self.created_at,
                "elapsed_seconds": self.elapsed_seconds,
                "cancel_requested": self.cancel_requested,
                "json_file": self.json_file,
                "bundle_file": self.bundle_file,
                "worker_done": worker_done,
            }


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
    connection_id: str = ""
    provider_display_name: str = ""
    status: str = "running"
    progress_done: int = 0
    progress_total: int = 0
    status_history: list[str] = field(default_factory=list)
    event_history: list[dict] = field(default_factory=list)
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
            worker_done = self.future.done() if self.future is not None else self.status in {"completed", "cancelled", "failed", "interrupted"}
            return {
                "job_id": self.job_id,
                "provider": self.provider,
                "connection_id": self.connection_id,
                "provider_display_name": self.provider_display_name,
                "image_model_name": self.image_model_name,
                "resolution": self.resolution,
                "aspect_ratio": self.aspect_ratio,
                "num_images": self.num_images,
                "input_mime_type": self.input_mime_type,
                "status": self.status,
                "progress_done": self.progress_done,
                "progress_total": self.progress_total,
                "status_history": list(self.status_history),
                "event_history": [dict(item) for item in self.event_history],
                "refined_images": list(self.refined_images),
                "failed_results": list(self.failed_results),
                "error": self.error,
                "created_at": self.created_at,
                "elapsed_seconds": self.elapsed_seconds,
                "cancel_requested": self.cancel_requested,
                "original_image_bytes": self.original_image_bytes,
                "worker_done": worker_done,
            }


def get_demo_results_root() -> Path:
    return REPO_ROOT / "results" / "demo"


def get_demo_results_dir(task_name: str) -> Path:
    results_dir = get_demo_results_root() / normalize_task_name(task_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _load_persisted_demo_ui_state_payload() -> dict:
    with DEMO_UI_STATE_LOCK:
        memory_payload = dict(DEMO_UI_STATE)
    if memory_payload:
        return memory_payload
    try:
        return read_ui_state(base_dir=REPO_ROOT)
    except Exception:
        logger.warning("读取磁盘 UI 状态失败", exc_info=True)
        return {}


def _persist_demo_ui_state_payload(state_payload: dict) -> None:
    with DEMO_UI_STATE_LOCK:
        DEMO_UI_STATE.clear()
        DEMO_UI_STATE.update(state_payload)

    safe_disk_payload = {
        key: value
        for key, value in state_payload.items()
        if key not in SAFE_DISK_UI_STATE_EXCLUDE_KEYS
    }
    try:
        write_ui_state(safe_disk_payload, base_dir=REPO_ROOT)
    except Exception:
        logger.warning("写入磁盘 UI 状态失败", exc_info=True)


PERSISTED_UI_STATE_KEYS = {
    "workspace_mode",
    "tab1_task_name",
    "tab1_dataset_name",
    "tab1_exp_mode",
    "tab1_retrieval_setting",
    "tab1_curated_profile",
    "tab1_curated_profile_input",
    "tab1_num_candidates",
    "tab1_concurrency_mode",
    "tab1_max_concurrent",
    "tab1_aspect_ratio",
    "tab1_image_resolution",
    "tab1_max_critic_rounds",
    "tab1_provider",
    "tab1_api_key",
    "tab1_base_url",
    "tab1_connection_display_name",
    "tab1_connection_id",
    "tab1_api_key_env_var",
    "tab1_supports_image",
    "tab1_supports_text",
    "tab1_connection_enabled",
    "tab1_model_discovery_mode",
    "tab1_model_allowlist",
    "tab1_model_name",
    "tab1_model_name_selector",
    "tab1_model_name_custom",
    "tab1_image_model_name",
    "tab1_image_model_name_selector",
    "tab1_image_model_name_custom",
    "tab1_diagram_content",
    "tab1_diagram_visual_intent",
    "tab1_diagram_content_editor",
    "tab1_diagram_visual_intent_editor",
    "tab1_plot_content",
    "tab1_plot_visual_intent",
    "tab1_plot_content_editor",
    "tab1_plot_visual_intent_editor",
    "refine_resolution",
    "refine_aspect_ratio",
    "refine_num_images",
    "refine_provider",
    "refine_api_key",
    "refine_base_url",
    "refine_connection_display_name",
    "refine_connection_id",
    "refine_api_key_env_var",
    "refine_supports_image",
    "refine_supports_text",
    "refine_connection_enabled",
    "refine_model_discovery_mode",
    "refine_model_allowlist",
    "refine_image_model_name",
    "refine_image_model_name_selector",
    "refine_image_model_name_custom",
    "refine_workspace_view",
    "edit_prompt",
    "refine_input_source",
    "refine_staged_input_mime_type",
    "refine_staged_source_label",
    "refine_uploaded_input_mime_type",
    "refine_uploaded_filename",
    "refine_uploaded_file_size",
    "refine_upload_error",
    "generation_display_scope_diagram",
    "generation_display_scope_plot",
    "generation_export_scope_diagram",
    "generation_export_scope_plot",
    "generation_final_candidate_id",
    "refine_version_counter",
    "refine_active_version_key",
    "active_generation_job_id",
    "last_generation_completed_job_id",
    "active_refine_job_id",
    "last_refine_completed_job_id",
    "json_file",
    "bundle_file",
    "result_source_label",
}
PERSISTED_UI_STATE_BYTES_KEYS = {
    "refine_staged_image_bytes",
    "refine_uploaded_image_bytes",
}
PERSISTED_UI_STATE_COMPLEX_KEYS = {
    "generation_candidate_decisions",
    "refine_latest_version_keys",
    "refine_version_history",
}


def _serialize_ui_state_value(key: str, value):
    if isinstance(value, bytes) or key in PERSISTED_UI_STATE_BYTES_KEYS:
        raw_bytes = bytes(value or b"")
        return {
            "__type__": "bytes",
            "data": base64.b64encode(raw_bytes).decode("utf-8"),
        }
    if isinstance(value, list):
        return {
            "__type__": "list",
            "data": [_serialize_ui_state_value(key, item) for item in value],
        }
    if isinstance(value, dict):
        return {
            "__type__": "dict",
            "data": {
                str(item_key): _serialize_ui_state_value(key, item_value)
                for item_key, item_value in value.items()
            },
        }
    return value


def _deserialize_ui_state_value(key: str, value):
    if isinstance(value, dict):
        value_type = value.get("__type__")
        if value_type == "bytes":
            encoded = str(value.get("data", "") or "")
            if encoded:
                try:
                    return base64.b64decode(encoded)
                except Exception:
                    return b""
            return b""
        if value_type == "list":
            return [_deserialize_ui_state_value(key, item) for item in value.get("data", [])]
        if value_type == "dict":
            return {
                str(item_key): _deserialize_ui_state_value(key, item_value)
                for item_key, item_value in dict(value.get("data", {})).items()
            }
    return value


def _restore_active_background_job(
    *,
    session_key: str,
    job_kind: str,
    get_snapshot_fn: Callable[[str], dict | None],
) -> dict | None:
    restored_job_id = str(st.session_state.get(session_key, "") or "").strip()
    if not restored_job_id:
        return None

    snapshot = hydrate_persisted_job_snapshot(
        get_snapshot_fn(restored_job_id),
        job_kind=job_kind,
    )
    if snapshot is None:
        st.session_state.pop(session_key, None)
        return None
    return snapshot


def restore_persisted_demo_ui_state() -> None:
    # NOTE: Do not auto-restore completed generation results (contains large
    # base64 images) to avoid significant slowdown on page refresh.
    # Users can manually reload history via the history replay panel.
    st.session_state.pop("bundle_file", None)
    st.session_state.pop("json_file", None)

    payload = _load_persisted_demo_ui_state_payload()
    if not payload:
        return

    _skip_restore_keys = {"bundle_file", "json_file", "results"}
    for key, raw_value in payload.items():
        if key in st.session_state or key in _skip_restore_keys:
            continue
        st.session_state[key] = _deserialize_ui_state_value(key, raw_value)

    generation_snapshot = _restore_active_background_job(
        session_key="active_generation_job_id",
        job_kind="generation",
        get_snapshot_fn=get_generation_job_snapshot,
    )
    if generation_snapshot is not None and generation_snapshot.get("status") in {"completed", "cancelled", "failed"}:
        # 终态任务交给后续 fragment 按现有 finalize 路径收口；这里只保留 job 指针。
        pass

    if "refined_images" not in st.session_state:
        refine_snapshot = _restore_active_background_job(
            session_key="active_refine_job_id",
            job_kind="refine",
            get_snapshot_fn=get_refine_job_snapshot,
        )
        if refine_snapshot and refine_snapshot.get("status") in {"completed", "cancelled", "failed"}:
            persist_refine_job_results(refine_snapshot)


def persist_demo_ui_state() -> None:
    state_payload = {}
    for key in sorted(PERSISTED_UI_STATE_KEYS | PERSISTED_UI_STATE_BYTES_KEYS):
        if key not in st.session_state:
            continue
        value = st.session_state.get(key)
        if (
            isinstance(value, (str, int, float, bool))
            or value is None
            or key in PERSISTED_UI_STATE_BYTES_KEYS
            or key in PERSISTED_UI_STATE_COMPLEX_KEYS
        ):
            state_payload[key] = _serialize_ui_state_value(key, value)
    _persist_demo_ui_state_payload(state_payload)


def hydrate_persisted_job_snapshot(snapshot: dict | None, *, job_kind: str) -> dict | None:
    if not snapshot:
        return None
    if snapshot.get("snapshot_source") != "disk":
        return snapshot
    if snapshot.get("status") != "running":
        return snapshot

    interrupted_snapshot = dict(snapshot)
    interrupted_snapshot["status"] = "interrupted"
    interrupted_snapshot["error"] = interrupted_snapshot.get("error") or "应用重启前任务仍在运行，已恢复最后一次持久化快照。"
    status_history = list(interrupted_snapshot.get("status_history", []) or [])
    interruption_line = (
        "[恢复] 检测到磁盘快照来自一个未正常收尾的后台"
        + ("生成" if job_kind == "generation" else "精修")
        + "任务，当前仅恢复最后快照。"
    )
    if not status_history or status_history[-1] != interruption_line:
        status_history.append(interruption_line)
    interrupted_snapshot["status_history"] = status_history
    return interrupted_snapshot


def save_demo_generation_artifacts(
    *,
    results: list[dict],
    dataset_name: str,
    task_name: str,
    exp_mode: str,
    retrieval_setting: str,
    curated_profile: str,
    provider: str,
    connection_id: str,
    provider_display_name: str,
    model_name: str,
    image_model_name: str,
    concurrency_mode: str,
    max_concurrent: int,
    max_critic_rounds: int,
    requested_candidates: int,
    effective_concurrent: int,
    timestamp_str: str | None = None,
    run_status: str = "completed",
) -> dict:
    normalized_task_name = normalize_task_name(task_name)
    timestamp_str = timestamp_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results_dir = get_demo_results_dir(normalized_task_name)
    run_stem = config.build_run_name(
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        provider=connection_id or provider,
        model_name=model_name,
        image_model_name=image_model_name,
        retrieval_setting=retrieval_setting,
        curated_profile=curated_profile,
        exp_mode=exp_mode,
        split_name="demo",
    )
    json_filename = results_dir / f"demo_{normalized_task_name}_{run_stem}.json"
    bundle_filename = companion_bundle_path(json_filename)
    summary = build_result_summary(results)
    failures = build_failure_manifest(results)
    manifest = build_run_manifest(
        producer="demo",
        result_count=len(results),
        dataset_name=dataset_name,
        task_name=normalized_task_name,
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        curated_profile=curated_profile,
        provider=provider,
        connection_id=connection_id,
        provider_display_name=provider_display_name,
        model_name=model_name,
        image_model_name=image_model_name,
        concurrency_mode=concurrency_mode,
        max_concurrent=int(max_concurrent),
        max_critic_rounds=int(max_critic_rounds),
        timestamp=timestamp_str,
        extra={
            "requested_candidates": int(requested_candidates),
            "effective_concurrent": int(effective_concurrent),
            "run_status": run_status,
            "curated_profile": curated_profile,
        },
    )

    write_json_payload(json_filename, results)
    write_result_bundle(
        bundle_filename,
        results,
        manifest=manifest,
        summary=summary,
        failures=failures,
    )
    return {
        "timestamp": timestamp_str,
        "json_file": str(json_filename),
        "bundle_file": str(bundle_filename),
        "summary": summary,
        "failures": failures,
        "manifest": manifest,
    }


def _store_generation_job(job: GenerationJobState) -> None:
    with GENERATION_JOBS_LOCK:
        GENERATION_JOBS[job.job_id] = job
    _persist_generation_job_snapshot(job.job_id)


def _load_persisted_generation_job_snapshot(job_id: str) -> dict | None:
    snapshot = read_job_snapshot(job_id, base_dir=REPO_ROOT)
    if snapshot is None:
        return None
    snapshot["event_history"] = read_job_events(job_id, base_dir=REPO_ROOT)
    snapshot["event_timeline"] = list(snapshot["event_history"])
    snapshot["snapshot_source"] = "disk"
    snapshot["worker_done"] = True
    return snapshot


def _persist_generation_job_snapshot(job_id: str) -> None:
    job = get_generation_job(job_id)
    if job is None:
        return
    try:
        snapshot = job.snapshot()
        snapshot["snapshot_source"] = "memory"
        write_job_snapshot(job_id, snapshot, base_dir=REPO_ROOT)
    except Exception:
        logger.warning("持久化生成任务快照失败: %s", job_id, exc_info=True)


def get_generation_job(job_id: str) -> GenerationJobState | None:
    with GENERATION_JOBS_LOCK:
        return GENERATION_JOBS.get(job_id)


def get_generation_job_snapshot(job_id: str) -> dict | None:
    job = get_generation_job(job_id)
    if job is None:
        return _load_persisted_generation_job_snapshot(job_id)
    snapshot = job.snapshot()
    snapshot["snapshot_source"] = "memory"
    return snapshot


def clear_generation_job(job_id: str) -> None:
    with GENERATION_JOBS_LOCK:
        GENERATION_JOBS.pop(job_id, None)


def _should_include_event_source(event_dict: dict) -> bool:
    source = str(event_dict.get("source", "") or "")
    kind = str(event_dict.get("kind", "") or "")
    return source not in {"PaperBananaDemo", "PaperVizProcessor", "GenerationUtils"} and kind not in {
        "stage",
        "preview_ready",
        "candidate_result",
        "artifact",
    }


def _render_event_status_line(event_dict: dict) -> str:
    event = coerce_runtime_event(event_dict, default_source="PaperBananaDemo")
    summary = event_summary_text(
        event,
        include_source=_should_include_event_source(event.to_dict()),
    )
    return f"[{event.ts}] {clean_text(summary)}"


def _append_generation_status_line(job: GenerationJobState, event_dict: dict) -> None:
    line = _render_event_status_line(event_dict)
    if job.status_history and job.status_history[-1] == line:
        return
    job.status_history.append(line)
    if len(job.status_history) > GENERATION_STATUS_LINE_LIMIT:
        job.status_history = job.status_history[-GENERATION_STATUS_LINE_LIMIT:]


def _append_refine_status_line(job: RefineJobState, event_dict: dict) -> None:
    line = _render_event_status_line(event_dict)
    if job.status_history and job.status_history[-1] == line:
        return
    job.status_history.append(line)
    if len(job.status_history) > REFINE_STATUS_LINE_LIMIT:
        job.status_history = job.status_history[-REFINE_STATUS_LINE_LIMIT:]


def _ensure_generation_candidate_snapshot(
    job: GenerationJobState,
    candidate_id: str,
    ts: str,
) -> dict:
    return job.candidate_snapshots.setdefault(
        candidate_id,
        {
            "candidate_id": candidate_id,
            "status": "queued",
            "stage": "等待开始",
            "preview_image": "",
            "preview_mime_type": "",
            "preview_label": "",
            "result": None,
            "error": "",
            "updated_at": ts,
        },
    )


def _count_generation_terminal_candidates(job: GenerationJobState) -> int:
    return sum(
        1
        for snapshot in job.candidate_snapshots.values()
        if str(snapshot.get("status", "") or "").strip() in {"completed", "failed", "cancelled"}
    )


def _sync_generation_job_progress_locked(job: GenerationJobState) -> None:
    total_candidates = max(
        int(job.progress_total or 0),
        int(job.requested_candidates or 0),
        len(job.candidate_snapshots),
    )
    terminal_candidates = _count_generation_terminal_candidates(job)
    job.progress_total = total_candidates
    job.progress_done = min(
        total_candidates,
        max(int(job.progress_done or 0), terminal_candidates),
    )


def _render_collapsible_event_timeline(
    event_history: list[dict],
    *,
    limit: int,
    label: str = "事件时间线",
) -> None:
    if not event_history:
        return

    safe_limit = max(1, int(limit or 1))
    latest_events = event_history[-safe_limit:]
    latest_lines = [_render_event_status_line(item) for item in latest_events]
    expander_label = f"{label}（最近 {len(latest_events)} / 共 {len(event_history)} 条）"

    with st.expander(expander_label, expanded=False):
        st.code("\n".join(latest_lines), language="text")
        if len(event_history) > len(latest_events):
            with st.expander(f"查看全部 {len(event_history)} 条事件", expanded=False):
                all_lines = [_render_event_status_line(item) for item in event_history]
                st.code("\n".join(all_lines), language="text")


def record_generation_job_event(job_id: str, event: dict | None) -> None:
    job = get_generation_job(job_id)
    if job is None or event is None:
        return

    payload = coerce_runtime_event(event, default_source="PaperBananaDemo").to_dict()
    if not payload.get("job_type"):
        payload["job_type"] = "generation"
    payload["message"] = clean_text(payload.get("message", ""))
    candidate_id = str(payload.get("candidate_id", "")).strip()
    stage_value = str(payload.get("stage", "")).strip()
    if not candidate_id or not stage_value:
        parsed_candidate = extract_generation_candidate_stage(payload.get("message", ""))
        if parsed_candidate:
            parsed_id, parsed_stage = parsed_candidate
            candidate_id = candidate_id or parsed_id
            stage_value = stage_value or parsed_stage
            payload["candidate_id"] = candidate_id
            payload["stage"] = stage_value
    if not payload.get("ts"):
        payload["ts"] = datetime.now().strftime("%H:%M:%S")

    with job.lock:
        job.event_history.append(dict(payload))
        if len(job.event_history) > GENERATION_EVENT_HISTORY_LIMIT:
            job.event_history = job.event_history[-GENERATION_EVENT_HISTORY_LIMIT:]
        _append_generation_status_line(job, payload)

        if candidate_id:
            snapshot = _ensure_generation_candidate_snapshot(job, candidate_id, payload["ts"])
            status_value = str(payload.get("status", "")).strip()
            if status_value:
                snapshot["status"] = status_value
            if stage_value:
                snapshot["stage"] = stage_value
                job.candidate_stage_map[candidate_id] = stage_value
            if payload.get("preview_image"):
                snapshot["preview_image"] = payload["preview_image"]
                snapshot["preview_mime_type"] = str(payload.get("preview_mime_type", "image/png"))
                snapshot["preview_label"] = str(
                    payload.get("preview_label", "") or stage_value or "最新预览"
                )
            if payload.get("kind") == "error":
                snapshot["status"] = "failed"
                snapshot["error"] = clean_text(payload.get("details") or payload.get("message", ""))
            elif stage_value:
                if "等待" in stage_value:
                    snapshot["status"] = snapshot.get("status") or "queued"
                elif "失败" in stage_value:
                    snapshot["status"] = "failed"
                elif "完成" in stage_value:
                    snapshot["status"] = "completed"
                elif snapshot.get("status") in {"", "queued"}:
                    snapshot["status"] = "running"
            snapshot["updated_at"] = payload["ts"]
            _sync_generation_job_progress_locked(job)
    append_job_event(job_id, payload, base_dir=REPO_ROOT)
    _persist_generation_job_snapshot(job_id)


def append_generation_job_status(job_id: str, message: str) -> None:
    if not message:
        return
    record_generation_job_event(
        job_id,
        create_runtime_event(
            level="INFO",
            kind="job",
            source="PaperBananaDemo",
            message=message,
            job_type="generation",
        ).to_dict(),
    )


def append_generation_job_result(job_id: str, result_data: dict) -> None:
    job = get_generation_job(job_id)
    if job is None or not isinstance(result_data, dict):
        return

    candidate_id = str(get_candidate_id(result_data, len(job.results))).strip()
    with job.lock:
        remaining_results = [
            existing
            for existing in job.results
            if str(get_candidate_id(existing, "")) != candidate_id
        ]
        remaining_results.append(dict(result_data))
        job.results = sort_results_stably(remaining_results)
        snapshot = job.candidate_snapshots.setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "status": "completed",
                "stage": "候选流程完成",
                "preview_image": "",
                "preview_mime_type": "",
                "preview_label": "",
                "result": None,
                "error": "",
                "updated_at": datetime.now().strftime("%H:%M:%S"),
            },
        )
        snapshot["result"] = dict(result_data)
        snapshot["status"] = "failed" if result_data.get("status") == "failed" else "completed"
        snapshot["stage"] = "候选失败" if result_data.get("status") == "failed" else "候选流程完成"
        if snapshot["status"] == "completed":
            final_image_key, final_desc_key = find_final_stage_keys(
                result_data,
                task_name=result_data.get("task_name", job.task_name),
                exp_mode=result_data.get("exp_mode", job.exp_mode),
            )
            if final_image_key and result_data.get(final_image_key):
                snapshot["preview_image"] = result_data.get(final_image_key, "")
                snapshot["preview_mime_type"] = "image/png"
                snapshot["preview_label"] = "✅ 最终结果预览"
        snapshot["updated_at"] = datetime.now().strftime("%H:%M:%S")
        _sync_generation_job_progress_locked(job)
    _persist_generation_job_snapshot(job_id)


def update_generation_job_progress(
    job_id: str,
    done_count: int,
    total_count: int,
    effective_concurrent: int,
) -> None:
    job = get_generation_job(job_id)
    if job is None:
        return
    with job.lock:
        job.progress_done = int(done_count)
        job.progress_total = int(total_count)
        job.effective_concurrent = int(effective_concurrent)
        job.estimated_batches = math.ceil(total_count / max(1, effective_concurrent))
        _sync_generation_job_progress_locked(job)
    _persist_generation_job_snapshot(job_id)


def request_generation_job_cancel(job_id: str) -> None:
    job = get_generation_job(job_id)
    if job is None:
        return
    with job.lock:
        job.cancel_requested = True
        job.cancel_event.set()
    _persist_generation_job_snapshot(job_id)
    append_generation_job_status(job_id, "[生成] 已收到停止请求，当前进行中的候选会继续完成，未开始的候选将被跳过。")


class JobEventHandler(logging.Handler):
    """Bridge selected logger output into background job event history."""

    def __init__(self, job_id: str, job_type: str, logger_names: set[str]):
        super().__init__(level=logging.INFO)
        self.job_id = job_id
        self.job_type = job_type
        self.logger_names = set(logger_names)

    def emit(self, record: logging.LogRecord) -> None:
        if record.name not in self.logger_names:
            return
        if getattr(record, "paperbanana_event", None) is not None:
            return
        if record.levelno < logging.INFO:
            return
        try:
            payload = runtime_event_from_log_record(record).to_dict()
        except Exception:
            payload = create_runtime_event(
                ts=datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
                level=record.levelno,
                kind="error" if record.levelno >= logging.ERROR else "warning" if record.levelno >= logging.WARNING else "job",
                source=record.name,
                message=record.getMessage(),
                job_type=self.job_type,
            ).to_dict()
        if not payload.get("job_type"):
            payload["job_type"] = self.job_type
        if self.job_type == "generation":
            record_generation_job_event(self.job_id, payload)
        else:
            record_refine_job_event(self.job_id, payload)


@contextmanager
def capture_job_logs(job_id: str, job_type: str):
    logger_names = GENERATION_LOGGER_NAMES if job_type == "generation" else REFINE_LOGGER_NAMES
    handler = JobEventHandler(job_id, job_type, logger_names)
    logger_objects = [logging.getLogger(name) for name in logger_names]
    original_levels = {logger_obj: logger_obj.level for logger_obj in logger_objects}
    for logger_obj in logger_objects:
        logger_obj.addHandler(handler)
        if logger_obj.getEffectiveLevel() > logging.INFO:
            logger_obj.setLevel(logging.INFO)
    try:
        yield
    finally:
        for logger_obj in logger_objects:
            logger_obj.removeHandler(handler)
            logger_obj.setLevel(original_levels[logger_obj])
        handler.close()


def persist_generation_job_results(
    snapshot: dict,
    *,
    source_label: str = "后台生成任务",
    reset_candidate_workspace: bool = True,
) -> None:
    st.session_state["results"] = sort_results_stably(snapshot.get("results", []))
    st.session_state["task_name"] = normalize_task_name(snapshot.get("task_name", "diagram"))
    st.session_state["dataset_name"] = snapshot.get("dataset_name", DEFAULT_DATASET_NAME)
    st.session_state["exp_mode"] = snapshot.get("exp_mode", "")
    st.session_state["concurrency_mode"] = snapshot.get("concurrency_mode", "auto")
    st.session_state["max_concurrent"] = int(snapshot.get("max_concurrent", 0) or 0)
    st.session_state["effective_concurrent"] = int(snapshot.get("effective_concurrent", 0) or 0)
    st.session_state["estimated_batches"] = int(snapshot.get("estimated_batches", 0) or 0)
    st.session_state["timestamp"] = snapshot.get("timestamp") or snapshot.get("created_at", "N/A")
    st.session_state["json_file"] = snapshot.get("json_file", "")
    st.session_state["bundle_file"] = snapshot.get("bundle_file", "")
    st.session_state["generation_summary"] = snapshot.get("summary", {})
    st.session_state["generation_failures"] = snapshot.get("failures", [])
    st.session_state["requested_candidates"] = int(
        snapshot.get("requested_candidates", len(snapshot.get("results", []))) or 0
    )
    st.session_state["result_source_label"] = source_label
    if reset_candidate_workspace:
        st.session_state["generation_candidate_decisions"] = {}
        st.session_state["generation_final_candidate_id"] = ""


def list_demo_bundle_files(
    task_name: str | None = None,
    *,
    limit: int | None = None,
) -> list[Path]:
    root = get_demo_results_root()
    if not root.exists():
        return []
    if task_name:
        search_root = root / normalize_task_name(task_name)
        pattern = "*.bundle.json"
    else:
        search_root = root
        pattern = "*.bundle.json"
    if not search_root.exists():
        return []
    bundle_files = list(search_root.rglob(pattern))
    bundle_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if limit is None:
        return bundle_files
    return bundle_files[: max(1, int(limit or 1))]


def load_generation_history_snapshot(bundle_path: str | Path) -> dict:
    payload = load_result_bundle(bundle_path)
    manifest = payload.get("manifest", {})
    results = sort_results_stably(payload.get("results", []))
    requested_candidates = int(manifest.get("requested_candidates", len(results)) or len(results))
    effective_concurrent = int(manifest.get("effective_concurrent", manifest.get("max_concurrent", 0)) or 0)
    estimated_batches = math.ceil(requested_candidates / max(1, effective_concurrent)) if effective_concurrent else 0
    json_path = Path(bundle_path)
    legacy_json_path = (
        Path(str(json_path).replace(".bundle.json", ".json"))
        if str(json_path).endswith(".bundle.json")
        else json_path
    )
    return {
        "results": results,
        "summary": payload.get("summary", {}),
        "failures": payload.get("failures", []),
        "dataset_name": manifest.get("dataset_name", DEFAULT_DATASET_NAME),
        "task_name": manifest.get("task_name", "diagram"),
        "exp_mode": manifest.get("exp_mode", ""),
        "concurrency_mode": manifest.get("concurrency_mode", "auto"),
        "max_concurrent": int(manifest.get("max_concurrent", 0) or 0),
        "effective_concurrent": effective_concurrent,
        "estimated_batches": estimated_batches,
        "requested_candidates": requested_candidates,
        "timestamp": manifest.get("timestamp") or manifest.get("created_at", ""),
        "json_file": str(legacy_json_path) if legacy_json_path.exists() else "",
        "bundle_file": str(bundle_path),
        "status": manifest.get("run_status", "completed"),
        "manifest": manifest,
    }


def _read_bundle_manifest_fast(bundle_path: str | Path) -> dict:
    """Read only the manifest from a bundle file without loading the full results array.

    This avoids parsing multi-MB base64 image data just to extract metadata.
    Falls back to an empty dict on any error.
    """
    import json as _json
    try:
        with open(bundle_path, "r", encoding="utf-8") as f:
            payload = _json.load(f)
        if isinstance(payload, dict):
            return payload.get("manifest", {})
    except Exception:
        pass
    return {}


def format_demo_bundle_label(bundle_path: str | Path) -> str:
    manifest = _read_bundle_manifest_fast(bundle_path)
    stamp = manifest.get("timestamp") or manifest.get("created_at") or Path(bundle_path).stem
    task_name = normalize_task_name(manifest.get("task_name", "diagram"))
    provider = manifest.get("provider", "-")
    exp_mode = manifest.get("exp_mode", "-")
    result_count = int(manifest.get("result_count", 0) or 0)
    return f"{stamp} | {task_name} | {provider} | {exp_mode} | {result_count} results"


def build_demo_bundle_display_labels(bundle_files: list[Path]) -> dict[str, str]:
    """为历史 bundle 构建稳定且唯一的显示标签。"""
    base_labels = [format_demo_bundle_label(path) for path in bundle_files]
    duplicate_counts = Counter(base_labels)
    seen_counts: dict[str, int] = {}
    display_labels: dict[str, str] = {}
    for path, base_label in zip(bundle_files, base_labels, strict=False):
        duplicate_count = duplicate_counts.get(base_label, 0)
        if duplicate_count <= 1:
            display_labels[str(path)] = base_label
            continue
        seen_counts[base_label] = seen_counts.get(base_label, 0) + 1
        display_labels[str(path)] = f"{base_label} | {path.name} | #{seen_counts[base_label]}"
    return display_labels


def normalize_candidate_token(candidate_id: int | str | None) -> str:
    return str(candidate_id if candidate_id not in (None, "") else "").strip()


def get_generation_candidate_decision_map() -> dict[str, str]:
    payload = st.session_state.get("generation_candidate_decisions", {})
    if not isinstance(payload, dict):
        payload = {}
    normalized = {
        normalize_candidate_token(candidate_id): str(decision or "default")
        for candidate_id, decision in payload.items()
        if normalize_candidate_token(candidate_id)
    }
    st.session_state["generation_candidate_decisions"] = normalized
    return normalized


def get_generation_final_candidate_token() -> str:
    return normalize_candidate_token(st.session_state.get("generation_final_candidate_id", ""))


def get_generation_candidate_decision(candidate_id: int | str | None) -> str:
    candidate_token = normalize_candidate_token(candidate_id)
    if not candidate_token:
        return "default"
    if get_generation_final_candidate_token() == candidate_token:
        return "final"
    return get_generation_candidate_decision_map().get(candidate_token, "default")


def set_generation_candidate_decision(candidate_id: int | str | None, decision: str) -> None:
    candidate_token = normalize_candidate_token(candidate_id)
    if not candidate_token:
        return
    decision_map = get_generation_candidate_decision_map()
    normalized_decision = str(decision or "default").strip() or "default"
    if normalized_decision == "final":
        previous_final = get_generation_final_candidate_token()
        if previous_final and previous_final != candidate_token and decision_map.get(previous_final) == "final":
            decision_map[previous_final] = "default"
        st.session_state["generation_final_candidate_id"] = candidate_token
        decision_map[candidate_token] = "final"
    elif normalized_decision == "discarded":
        if get_generation_final_candidate_token() == candidate_token:
            st.session_state["generation_final_candidate_id"] = ""
        decision_map[candidate_token] = "discarded"
    elif normalized_decision == "favorite":
        if get_generation_final_candidate_token() == candidate_token:
            st.session_state["generation_final_candidate_id"] = ""
        decision_map[candidate_token] = "favorite"
    else:
        if get_generation_final_candidate_token() == candidate_token:
            st.session_state["generation_final_candidate_id"] = ""
        decision_map.pop(candidate_token, None)
    st.session_state["generation_candidate_decisions"] = decision_map


def filter_generation_results_by_scope(
    results: list[dict],
    scope: str,
) -> list[dict]:
    normalized_scope = str(scope or "全部候选").strip() or "全部候选"
    filtered_results = []
    for fallback_index, result in enumerate(results):
        candidate_id = get_candidate_id(result, fallback_index)
        decision = get_generation_candidate_decision(candidate_id)
        if normalized_scope == "仅未淘汰" and decision == "discarded":
            continue
        if normalized_scope == "仅收藏" and decision not in {"favorite", "final"}:
            continue
        if normalized_scope == "仅最终候选" and decision != "final":
            continue
        filtered_results.append(result)
    return filtered_results


def find_generation_result_by_candidate_id(
    results: list[dict],
    candidate_id: int | str | None,
) -> dict | None:
    candidate_token = normalize_candidate_token(candidate_id)
    if not candidate_token:
        return None
    for fallback_index, result in enumerate(results):
        if normalize_candidate_token(get_candidate_id(result, fallback_index)) == candidate_token:
            return result
    return None


def _compute_image_digest(image_bytes: bytes) -> str:
    return hashlib.sha1(bytes(image_bytes or b"")).hexdigest()


def get_refine_version_history() -> list[dict]:
    history = st.session_state.get("refine_version_history", [])
    if not isinstance(history, list):
        history = []
    st.session_state["refine_version_history"] = history
    return history


def _next_refine_version_key() -> tuple[str, int]:
    counter = int(st.session_state.get("refine_version_counter", 0) or 0) + 1
    st.session_state["refine_version_counter"] = counter
    return f"v{counter:02d}", counter


def ensure_refine_source_version(
    image_bytes: bytes,
    *,
    input_mime_type: str,
    source_label: str,
) -> str:
    if not image_bytes:
        return ""
    validated_bytes, validated_mime_type, validation_error = validate_refine_image_bytes(
        image_bytes,
        input_mime_type=input_mime_type,
        file_name=source_label,
    )
    if validation_error:
        logger.warning("跳过写入损坏的精修来源版本: %s", validation_error)
        return ""
    history = get_refine_version_history()
    image_digest = _compute_image_digest(validated_bytes)
    for entry in history:
        if entry.get("image_digest") == image_digest:
            return str(entry.get("version_key", "") or "")
    version_key, _ = _next_refine_version_key()
    history.append(
        {
            "version_key": version_key,
            "parent_version_key": "",
            "label": "原图" if len(history) == 0 else source_label,
            "source_label": source_label,
            "edit_prompt": "",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_bytes": validated_bytes,
            "input_mime_type": validated_mime_type,
            "image_digest": image_digest,
            "provider": "",
            "image_model_name": "",
            "resolution": "",
            "variant_index": 0,
        }
    )
    return version_key


def find_refine_version_entry(version_key: str) -> dict | None:
    normalized_key = str(version_key or "").strip()
    if not normalized_key:
        return None
    for entry in get_refine_version_history():
        if str(entry.get("version_key", "") or "") == normalized_key:
            return entry
    return None


def activate_refine_version(version_key: str) -> bool:
    entry = find_refine_version_entry(version_key)
    if not entry:
        return False
    display_label = get_refine_version_display_label(entry=entry)
    stage_refine_source_image(
        entry.get("image_bytes", b""),
        input_mime_type=entry.get("input_mime_type", "image/png"),
        source_label=display_label or "历史版本",
        default_prompt=entry.get("edit_prompt", ""),
    )
    st.session_state["refine_active_version_key"] = version_key
    return True


def append_refine_snapshot_to_version_history(
    snapshot: dict,
    *,
    edit_prompt: str,
) -> list[str]:
    refined_images = snapshot.get("refined_images", []) or []
    if not refined_images:
        return []

    original_bytes = snapshot.get("original_image_bytes", b"") or b""
    original_input_mime_type = snapshot.get("input_mime_type", "image/png")
    source_label = (
        st.session_state.get("refine_selected_source_label")
        or st.session_state.get("refine_staged_source_label")
        or "原图"
    )
    parent_version_key = ensure_refine_source_version(
        original_bytes,
        input_mime_type=original_input_mime_type,
        source_label=source_label,
    )
    history = get_refine_version_history()
    created_version_keys: list[str] = []
    for variant_index, item in enumerate(refined_images, start=1):
        image_bytes = item.get("bytes", b"") or b""
        if not image_bytes:
            continue
        validated_bytes, validated_mime_type, validation_error = validate_refine_image_bytes(
            image_bytes,
            input_mime_type="image/png",
            file_name=f"精修结果 {variant_index}",
        )
        if validation_error:
            logger.warning("跳过写入损坏的精修结果版本: %s", validation_error)
            continue
        image_digest = _compute_image_digest(validated_bytes)
        if any(entry.get("image_digest") == image_digest for entry in history):
            continue
        version_key, _ = _next_refine_version_key()
        display_label = get_next_refine_generated_label(history)
        history.append(
            {
                "version_key": version_key,
                "parent_version_key": parent_version_key,
                "label": display_label,
                "source_label": source_label,
                "edit_prompt": clean_text(edit_prompt),
                "created_at": snapshot.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "image_bytes": validated_bytes,
                "input_mime_type": validated_mime_type,
                "image_digest": image_digest,
                "provider": snapshot.get("provider", ""),
                "image_model_name": snapshot.get("image_model_name", ""),
                "resolution": snapshot.get("resolution", ""),
                "variant_index": variant_index,
            }
        )
        created_version_keys.append(version_key)
    if created_version_keys:
        st.session_state["refine_active_version_key"] = created_version_keys[0]
        st.session_state["refine_latest_version_keys"] = created_version_keys
    return created_version_keys


def stage_refine_source_image(
    image_bytes: bytes,
    *,
    input_mime_type: str = "image/png",
    source_label: str,
    default_prompt: str = "",
) -> None:
    st.session_state["refine_staged_image_bytes"] = image_bytes
    st.session_state["refine_staged_input_mime_type"] = normalize_image_mime_type(input_mime_type)
    st.session_state["refine_staged_source_label"] = source_label
    _queue_refine_widget_state_updates(
        {
            "refine_input_source": "候选方案",
            "refine_workspace_view": REFINE_WORKSPACE_VIEW_OPTIONS[0],
        }
    )
    if default_prompt and not st.session_state.get("edit_prompt", "").strip():
        _queue_refine_widget_state_updates({"edit_prompt": default_prompt})


def clear_staged_refine_source() -> None:
    for key in (
        "refine_staged_image_bytes",
        "refine_staged_input_mime_type",
        "refine_staged_source_label",
    ):
        st.session_state.pop(key, None)
    if st.session_state.get("refine_input_source") == "候选方案":
        _queue_refine_widget_state_updates(
            {
                "refine_input_source": "上传图像",
                "refine_workspace_view": REFINE_WORKSPACE_VIEW_OPTIONS[0],
            }
        )


def extract_result_image_payload(
    result: dict,
    *,
    exp_mode: str,
    task_name: str,
) -> tuple[bytes | None, str]:
    final_image_key, _ = find_final_stage_keys(
        result,
        task_name=task_name,
        exp_mode=exp_mode,
    )
    if not final_image_key or not result.get(final_image_key):
        return None, "image/png"
    image = base64_to_image(result[final_image_key])
    if image is None:
        return None, "image/png"
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"


def stage_candidate_for_refine(
    result: dict,
    *,
    candidate_id: int | str,
    exp_mode: str,
    task_name: str,
) -> bool:
    image_bytes, input_mime_type = extract_result_image_payload(
        result,
        exp_mode=exp_mode,
        task_name=task_name,
    )
    if not image_bytes:
        return False
    default_prompt = "保持内容与语义不变，优化清晰度、对齐、版式与视觉层次。"
    try:
        fallback_index = int(candidate_id)
    except Exception:
        fallback_index = 0
    candidate_label = format_candidate_display_label(result, fallback_index=fallback_index)
    stage_refine_source_image(
        image_bytes,
        input_mime_type=input_mime_type,
        source_label=candidate_label,
        default_prompt=default_prompt,
    )
    return True


def stage_plot_code_for_rerender(result: dict, *, candidate_id: int | str, exp_mode: str) -> bool:
    _, final_desc_key = find_final_stage_keys(
        result,
        task_name="plot",
        exp_mode=exp_mode,
    )
    if not final_desc_key:
        return False
    code_key = final_desc_key if final_desc_key == "vanilla_plot_code" else f"{final_desc_key}_code"
    code_text = clean_text(result.get(code_key, ""))
    if not code_text:
        return False
    try:
        fallback_index = int(candidate_id)
    except Exception:
        fallback_index = 0
    candidate_label = format_candidate_display_label(result, fallback_index=fallback_index)
    st.session_state["plot_rerender_code"] = code_text
    st.session_state["plot_rerender_code_editor"] = code_text
    st.session_state["plot_rerender_candidate_id"] = candidate_id
    st.session_state["plot_rerender_candidate_label"] = candidate_label
    st.session_state["plot_rerender_source_desc_key"] = final_desc_key
    st.session_state.pop("plot_rerender_preview", None)
    return True


def _sanitize_zip_component(text: str, *, fallback: str = "未命名") -> str:
    cleaned = clean_text(text).strip()
    cleaned = re.sub(r"[<>:\"/\\\\|?*]+", "_", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def _normalize_export_timestamp_token(timestamp: str | None = None) -> str:
    raw = clean_text(timestamp or "").strip()
    if raw:
        token = re.sub(r"[^0-9]", "", raw)
        if len(token) >= 14:
            return f"{token[:8]}_{token[8:14]}"
        if len(token) >= 8:
            return token[:8]
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _format_candidate_export_folder_name(
    candidate_id: int | str,
    candidate_index: int,
    *,
    task_display_name_cn: str,
) -> str:
    suffix = _sanitize_zip_component(task_display_name_cn, fallback="候选结果")
    display_candidate = max(1, int(candidate_index))
    return f"候选{display_candidate:02d}_{suffix}"


def _normalize_stage_export_label(stage_name: str, *, stage_index: int) -> str:
    stage_text = clean_text(stage_name)
    if "规划" in stage_text:
        return "规划器"
    if "风格" in stage_text:
        return "风格增强"
    if "评审" in stage_text:
        round_match = re.search(r"第\s*(\d+)\s*轮", stage_text)
        if round_match:
            return f"评审第{int(round_match.group(1)):02d}轮"
        return "评审轮"
    if "基础直出" in stage_text:
        return "基础直出"
    if "精修" in stage_text:
        return "精修成稿"
    return _sanitize_zip_component(stage_text, fallback=f"阶段{stage_index:02d}")


def _decode_image_bytes(image_b64: str) -> bytes:
    encoded = str(image_b64 or "")
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    return base64.b64decode(encoded)


def _infer_image_extension_from_bytes(raw_bytes: bytes, *, fallback: str = "png") -> str:
    try:
        with Image.open(BytesIO(raw_bytes)) as img:
            image_format = (img.format or "").upper()
    except Exception:
        return fallback
    ext_map = {
        "JPEG": "jpg",
        "JPG": "jpg",
        "PNG": "png",
        "WEBP": "webp",
        "GIF": "gif",
    }
    return ext_map.get(image_format, fallback)


def _build_candidate_overview_text(
    *,
    candidate_id: int | str,
    candidate_index: int,
    task_display_name_cn: str,
    exp_mode: str,
    final_caption: str,
    final_desc_key: str,
    stages: list[dict],
) -> str:
    display_candidate = max(1, int(candidate_index))
    lines = [
        f"候选编号：{display_candidate:02d}（内部 ID: {candidate_id}）",
        f"任务类型：{task_display_name_cn}",
        f"流水线：{exp_mode}",
        f"最终结果：{final_caption}",
        f"最终描述键：{final_desc_key or '无'}",
        f"演化阶段数：{len(stages)}",
    ]
    if stages:
        lines.append("")
        lines.append("阶段顺序：")
        for idx, stage in enumerate(stages, start=1):
            lines.append(f"{idx:02d}. {clean_text(stage.get('name', '阶段'))}")
    return "\n".join(lines)


def _write_text_to_zip(zip_file, archive_path: str, text: str, *, encoding: str = "utf-8") -> None:
    zip_file.writestr(archive_path, clean_text(text).encode(encoding))


def build_final_results_zip(
    results: list[dict],
    *,
    task_name: str,
    exp_mode: str,
) -> tuple[bytes, int, list[str]]:
    import zipfile

    task_name = normalize_task_name(task_name)
    zip_buffer = BytesIO()
    zip_export_failures: list[str] = []
    exported_count = 0

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for fallback_idx, result in enumerate(results):
            candidate_id = get_candidate_id(result, fallback_idx)
            candidate_label = f"候选 {fallback_idx + 1:02d}"
            if isinstance(result, dict) and result.get("status") == "failed":
                zip_export_failures.append(
                    f"{candidate_label}: 流水线执行失败，无法导出"
                )
                continue

            final_image_key, final_desc_key = find_final_stage_keys(
                result,
                task_name=task_name,
                exp_mode=exp_mode,
            )

            exported_any = False
            if final_image_key and final_image_key in result:
                try:
                    raw_bytes = _decode_image_bytes(result[final_image_key])
                    image_ext = _infer_image_extension_from_bytes(raw_bytes, fallback="bin")
                    zip_file.writestr(
                        f"candidate_{candidate_id}.{image_ext}",
                        raw_bytes,
                    )
                    exported_any = True
                except Exception as export_err:
                    zip_export_failures.append(
                        f"{candidate_label}: 图像导出失败 ({export_err})"
                    )
            else:
                zip_export_failures.append(
                    f"{candidate_label}: 未找到最终图像"
                )

            if task_name == "plot" and final_desc_key:
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
    return zip_buffer.getvalue(), exported_count, zip_export_failures


def build_full_process_zip(
    results: list[dict],
    *,
    task_name: str,
    exp_mode: str,
    dataset_name: str,
    timestamp: str,
    source_label: str,
    json_file_path: Path | None = None,
    bundle_file_path: Path | None = None,
) -> tuple[bytes, int, list[str]]:
    import zipfile

    normalized_task = normalize_task_name(task_name)
    task_config = get_task_ui_config(normalized_task)
    timestamp_token = _normalize_export_timestamp_token(timestamp)
    root_dir = _sanitize_zip_component(
        f"PaperBanana_全流程总览_{task_config['display_name_cn']}_{timestamp_token}",
        fallback=f"PaperBanana_全流程总览_{timestamp_token}",
    )

    zip_buffer = BytesIO()
    export_failures: list[str] = []
    exported_count = 0

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        overview_dir = f"{root_dir}/00_运行总览"
        overview_lines = [
            f"任务类型：{task_config['display_name_cn']}",
            f"数据集：{dataset_name}",
            f"流水线：{exp_mode}",
            f"结果来源：{source_label}",
            f"导出时间：{timestamp or 'N/A'}",
            f"候选总数：{len(results)}",
        ]
        _write_text_to_zip(
            zip_file,
            f"{overview_dir}/运行总览.txt",
            "\n".join(overview_lines),
        )
        _write_text_to_zip(
            zip_file,
            f"{overview_dir}/如何查看本压缩包.txt",
            (
                "1. 先看 00_运行总览，快速了解本次任务配置。\n"
                "2. 再进入 01_候选方案，每个候选都按“最终结果 -> 演化过程 -> 原始记录”组织。\n"
                "3. 重点文件优先看每个候选目录下的 01_最终结果，以及 02_演化过程 中各阶段图片。"
            ),
        )
        if json_file_path and json_file_path.exists():
            zip_file.writestr(
                f"{overview_dir}/原始结果.json",
                json_file_path.read_bytes(),
            )
        if bundle_file_path and bundle_file_path.exists():
            zip_file.writestr(
                f"{overview_dir}/结果Bundle.bundle.json",
                bundle_file_path.read_bytes(),
            )

        for fallback_idx, result in enumerate(results):
            candidate_id = get_candidate_id(result, fallback_idx)
            candidate_index = fallback_idx + 1
            candidate_label = f"候选 {candidate_index:02d}"
            candidate_folder_name = _format_candidate_export_folder_name(
                candidate_id,
                candidate_index,
                task_display_name_cn=task_config["display_name_cn"],
            )
            candidate_root = f"{root_dir}/01_候选方案/{candidate_folder_name}"

            if isinstance(result, dict) and result.get("status") == "failed":
                failed_dir = f"{root_dir}/02_失败候选"
                _write_text_to_zip(
                    zip_file,
                    f"{failed_dir}/{candidate_folder_name}_失败信息.txt",
                    (
                        f"候选内部 ID：{candidate_id}\n"
                        f"状态：failed\n"
                        f"错误：{result.get('error', 'Unknown error')}\n\n"
                        f"{clean_text(result.get('error_detail', ''))}"
                    ),
                )
                export_failures.append(f"{candidate_label}: 流水线执行失败，仅导出失败信息")
                continue

            final_image_key, final_desc_key = find_final_stage_keys(
                result,
                task_name=normalized_task,
                exp_mode=exp_mode,
            )
            stages = build_evolution_stages(result, exp_mode, task_name=normalized_task)
            _write_text_to_zip(
                zip_file,
                f"{candidate_root}/00_候选总览.txt",
                _build_candidate_overview_text(
                    candidate_id=candidate_id,
                    candidate_index=candidate_index,
                    task_display_name_cn=task_config["display_name_cn"],
                    exp_mode=exp_mode,
                    final_caption=task_config["final_caption"],
                    final_desc_key=final_desc_key,
                    stages=stages,
                ),
            )

            exported_any = False
            final_result_dir = f"{candidate_root}/01_最终结果"
            if final_image_key and result.get(final_image_key):
                try:
                    final_bytes = _decode_image_bytes(result[final_image_key])
                    final_ext = _infer_image_extension_from_bytes(final_bytes)
                    zip_file.writestr(
                        f"{final_result_dir}/01_{_sanitize_zip_component(task_config['final_caption'], fallback='最终结果')}.{final_ext}",
                        final_bytes,
                    )
                    exported_any = True
                except Exception as export_err:
                    export_failures.append(
                        f"{candidate_label}: 最终图像导出失败 ({export_err})"
                    )
            else:
                export_failures.append(f"{candidate_label}: 未找到最终图像")

            if final_desc_key and result.get(final_desc_key):
                _write_text_to_zip(
                    zip_file,
                    f"{final_result_dir}/02_最终描述.md",
                    result[final_desc_key],
                )
                exported_any = True

            if normalized_task == "plot" and final_desc_key:
                final_code_key = (
                    final_desc_key
                    if final_desc_key == "vanilla_plot_code"
                    else f"{final_desc_key}_code"
                )
                if result.get(final_code_key):
                    _write_text_to_zip(
                        zip_file,
                        f"{final_result_dir}/03_最终Matplotlib代码.py",
                        result[final_code_key],
                    )
                    exported_any = True

            process_dir = f"{candidate_root}/02_演化过程"
            for stage_idx, stage in enumerate(stages, start=1):
                stage_label = _normalize_stage_export_label(stage.get("name", ""), stage_index=stage_idx)
                stage_dir = f"{process_dir}/{stage_idx:02d}_{stage_label}"
                _write_text_to_zip(
                    zip_file,
                    f"{stage_dir}/00_阶段说明.txt",
                    stage.get("description", ""),
                )
                stage_image_key = stage.get("image_key")
                if stage_image_key and result.get(stage_image_key):
                    try:
                        stage_bytes = _decode_image_bytes(result[stage_image_key])
                        stage_ext = _infer_image_extension_from_bytes(stage_bytes)
                        zip_file.writestr(
                            f"{stage_dir}/01_阶段图像.{stage_ext}",
                            stage_bytes,
                        )
                        exported_any = True
                    except Exception as export_err:
                        export_failures.append(
                            f"{candidate_label}: 阶段 {stage_label} 图像导出失败 ({export_err})"
                        )
                if stage.get("desc_key") and result.get(stage["desc_key"]):
                    _write_text_to_zip(
                        zip_file,
                        f"{stage_dir}/02_阶段描述.md",
                        result[stage["desc_key"]],
                    )
                    exported_any = True
                if stage.get("code_key") and result.get(stage["code_key"]):
                    _write_text_to_zip(
                        zip_file,
                        f"{stage_dir}/03_阶段代码.py",
                        result[stage["code_key"]],
                    )
                    exported_any = True
                if stage.get("suggestions_key") and result.get(stage["suggestions_key"]):
                    _write_text_to_zip(
                        zip_file,
                        f"{stage_dir}/04_评审建议.md",
                        result[stage["suggestions_key"]],
                    )
                    exported_any = True

            _write_text_to_zip(
                zip_file,
                f"{candidate_root}/99_原始记录/候选完整结果.json",
                json.dumps(result, ensure_ascii=False, indent=2),
            )
            if exported_any:
                exported_count += 1

    zip_buffer.seek(0)
    return zip_buffer.getvalue(), exported_count, export_failures


def _store_refine_job(job: RefineJobState) -> None:
    with REFINE_JOBS_LOCK:
        REFINE_JOBS[job.job_id] = job
    _persist_refine_job_snapshot(job.job_id)


def _load_persisted_refine_job_snapshot(job_id: str) -> dict | None:
    snapshot = read_job_snapshot(job_id, base_dir=REPO_ROOT)
    if snapshot is None:
        return None
    snapshot["event_history"] = read_job_events(job_id, base_dir=REPO_ROOT)
    snapshot["snapshot_source"] = "disk"
    snapshot["worker_done"] = True
    return snapshot


def _persist_refine_job_snapshot(job_id: str) -> None:
    job = get_refine_job(job_id)
    if job is None:
        return
    try:
        snapshot = job.snapshot()
        snapshot["snapshot_source"] = "memory"
        write_job_snapshot(job_id, snapshot, base_dir=REPO_ROOT)
    except Exception:
        logger.warning("持久化精修任务快照失败: %s", job_id, exc_info=True)


def get_refine_job(job_id: str) -> RefineJobState | None:
    with REFINE_JOBS_LOCK:
        return REFINE_JOBS.get(job_id)


def get_refine_job_snapshot(job_id: str) -> dict | None:
    job = get_refine_job(job_id)
    if job is None:
        return _load_persisted_refine_job_snapshot(job_id)
    snapshot = job.snapshot()
    snapshot["snapshot_source"] = "memory"
    return snapshot


def clear_refine_job(job_id: str) -> None:
    with REFINE_JOBS_LOCK:
        REFINE_JOBS.pop(job_id, None)


def record_refine_job_event(job_id: str, event: dict | None) -> None:
    job = get_refine_job(job_id)
    if job is None or event is None:
        return

    payload = coerce_runtime_event(event, default_source="PaperBananaDemo").to_dict()
    if not payload.get("job_type"):
        payload["job_type"] = "refine"
    payload["message"] = clean_text(payload.get("message", ""))
    if not payload.get("ts"):
        payload["ts"] = datetime.now().strftime("%H:%M:%S")

    with job.lock:
        job.event_history.append(dict(payload))
        if len(job.event_history) > REFINE_EVENT_HISTORY_LIMIT:
            job.event_history = job.event_history[-REFINE_EVENT_HISTORY_LIMIT:]
        _append_refine_status_line(job, payload)
    append_job_event(job_id, payload, base_dir=REPO_ROOT)
    _persist_refine_job_snapshot(job_id)


def append_refine_job_status(job_id: str, message: str) -> None:
    if not message:
        return
    record_refine_job_event(
        job_id,
        create_runtime_event(
            level="INFO",
            kind="job",
            source="PaperBananaDemo",
            message=message,
            job_type="refine",
        ).to_dict(),
    )


def update_refine_job_progress(job_id: str, done_count: int, total_count: int) -> None:
    job = get_refine_job(job_id)
    if job is None:
        return
    with job.lock:
        job.progress_done = done_count
        job.progress_total = total_count
    _persist_refine_job_snapshot(job_id)


def request_refine_job_cancel(job_id: str) -> None:
    job = get_refine_job(job_id)
    if job is None:
        return
    with job.lock:
        job.cancel_requested = True
        job.cancel_event.set()
    _persist_refine_job_snapshot(job_id)
    append_refine_job_status(job_id, "[精修] 已收到停止请求，将在当前请求结束后停止后续重试。")


def persist_refine_job_results(snapshot: dict) -> None:
    st.session_state["refined_images"] = snapshot.get("refined_images", [])
    st.session_state["refine_failed_results"] = snapshot.get("failed_results", [])
    st.session_state["refine_timestamp"] = snapshot.get("created_at", "N/A")
    st.session_state["refine_result_resolution"] = snapshot.get("resolution", "N/A")
    st.session_state["refine_count"] = snapshot.get("num_images", 0)
    st.session_state["refine_provider_used"] = (
        snapshot.get("provider_display_name")
        or snapshot.get("connection_id")
        or snapshot.get("provider", "N/A")
    )
    st.session_state["refine_image_model_used"] = snapshot.get("image_model_name", "N/A")
    st.session_state["refine_original_image_bytes"] = snapshot.get("original_image_bytes", b"")
    st.session_state["refine_original_input_mime_type"] = snapshot.get("input_mime_type", "image/png")
    st.session_state["refine_failed_results"] = snapshot.get("failed_results", [])
    append_refine_snapshot_to_version_history(
        snapshot,
        edit_prompt=st.session_state.get("edit_prompt", ""),
    )


def start_generation_background_job(
    *,
    dataset_name: str,
    task_name: str,
    exp_mode: str,
    retrieval_setting: str,
    curated_profile: str,
    provider: str,
    connection_id: str = "",
    api_key: str,
    model_name: str,
    image_model_name: str,
    base_url: str = "",
    extra_headers: dict[str, str] | None = None,
    concurrency_mode: str,
    max_concurrent: int,
    num_candidates: int,
    max_critic_rounds: int,
    aspect_ratio: str,
    image_resolution: str,
    image_generation_options: dict[str, Any] | None = None,
    content: str,
    visual_intent: str,
) -> str:
    normalized_task_name = normalize_task_name(task_name)
    retrieval_setting = normalize_retrieval_setting(retrieval_setting)
    curated_profile = normalize_curated_profile_name(curated_profile)
    runtime_settings = resolve_runtime_settings(
        provider,
        connection_id=connection_id,
        api_key=api_key,
        model_name=model_name,
        image_model_name=image_model_name,
        base_url=base_url,
        extra_headers=extra_headers,
        concurrency_mode=concurrency_mode,
        max_concurrent=max_concurrent,
        max_critic_rounds=max_critic_rounds,
        base_dir=REPO_ROOT,
        model_config_data=model_config_data,
    )
    requested_candidates = max(1, int(num_candidates))
    effective_concurrent = compute_effective_concurrency(
        concurrency_mode=runtime_settings.concurrency_mode,
        max_concurrent=runtime_settings.max_concurrent,
        total_candidates=requested_candidates,
        task_name=normalized_task_name,
        retrieval_setting=retrieval_setting,
        exp_mode=exp_mode,
        provider=runtime_settings.provider,
    )
    estimated_batches = math.ceil(requested_candidates / max(1, effective_concurrent))
    job_id = f"generate_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job = GenerationJobState(
        job_id=job_id,
        dataset_name=dataset_name,
        task_name=normalized_task_name,
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        curated_profile=curated_profile,
        provider=runtime_settings.provider,
        model_name=runtime_settings.model_name,
        image_model_name=runtime_settings.image_model_name,
        connection_id=runtime_settings.connection_id,
        provider_display_name=runtime_settings.provider_display_name,
        concurrency_mode=runtime_settings.concurrency_mode,
        max_concurrent=runtime_settings.max_concurrent,
        requested_candidates=requested_candidates,
        max_critic_rounds=runtime_settings.max_critic_rounds,
        aspect_ratio=aspect_ratio,
        image_resolution=image_resolution,
        content=content,
        visual_intent=visual_intent,
        progress_total=requested_candidates,
        effective_concurrent=effective_concurrent,
        estimated_batches=estimated_batches,
    )
    _store_generation_job(job)
    record_generation_job_event(
        job_id,
        create_runtime_event(
            level="INFO",
            kind="job",
            source="PaperBananaDemo",
            message=(
                f"[生成] 已创建后台任务：任务={normalized_task_name} | 数据集={dataset_name} | "
                f"候选={requested_candidates} | Provider={runtime_settings.provider}"
            ),
            job_type="generation",
            provider=runtime_settings.provider,
            model=runtime_settings.model_name,
            status="running",
        ).to_dict(),
    )

    def worker():
        started_at = time.perf_counter()
        with capture_job_logs(job_id, "generation"):
            try:
                input_data_list = create_sample_inputs(
                    content=content,
                    visual_intent=visual_intent,
                    task_name=normalized_task_name,
                    aspect_ratio=aspect_ratio,
                    num_copies=requested_candidates,
                    max_critic_rounds=runtime_settings.max_critic_rounds,
                    image_resolution=image_resolution,
                    image_generation_options=image_generation_options,
                )

                def on_progress(done_count: int, total_count: int, effective_count: int):
                    update_generation_job_progress(job_id, done_count, total_count, effective_count)

                def on_status(message: str):
                    append_generation_job_status(job_id, message)

                def on_event(event: dict):
                    record_generation_job_event(job_id, event)

                def on_result(result_data: dict):
                    append_generation_job_result(job_id, result_data)

                results, used_concurrency = asyncio.run(
                    process_parallel_candidates(
                        input_data_list,
                        dataset_name=dataset_name,
                        task_name=normalized_task_name,
                        exp_mode=exp_mode,
                        retrieval_setting=retrieval_setting,
                        curated_profile=curated_profile,
                        model_name=runtime_settings.model_name,
                        image_model_name=runtime_settings.image_model_name,
                        provider=runtime_settings.provider,
                        connection_id=runtime_settings.connection_id,
                        api_key=runtime_settings.api_key,
                        base_url=runtime_settings.base_url,
                        extra_headers=runtime_settings.extra_headers,
                        concurrency_mode=runtime_settings.concurrency_mode,
                        max_concurrent=runtime_settings.max_concurrent,
                        progress_callback=on_progress,
                        status_callback=on_status,
                        event_callback=on_event,
                        result_callback=on_result,
                        cancel_check=job.cancel_event.is_set,
                    )
                )
                results = sort_results_stably(results)
                run_status = "cancelled" if job.cancel_requested else "completed"
                artifact_payload = {}
                try:
                    artifact_payload = save_demo_generation_artifacts(
                        results=results,
                        dataset_name=dataset_name,
                        task_name=normalized_task_name,
                        exp_mode=exp_mode,
                        retrieval_setting=retrieval_setting,
                        curated_profile=curated_profile,
                        provider=runtime_settings.provider,
                        connection_id=runtime_settings.connection_id,
                        provider_display_name=runtime_settings.provider_display_name,
                        model_name=runtime_settings.model_name,
                        image_model_name=runtime_settings.image_model_name,
                        concurrency_mode=runtime_settings.concurrency_mode,
                        max_concurrent=runtime_settings.max_concurrent,
                        max_critic_rounds=runtime_settings.max_critic_rounds,
                        requested_candidates=requested_candidates,
                        effective_concurrent=used_concurrency,
                        timestamp_str=job.created_at,
                        run_status=run_status,
                    )
                    record_generation_job_event(
                        job_id,
                        create_runtime_event(
                            level="INFO",
                            kind="artifact",
                            source="PaperBananaDemo",
                            message=(
                                f"[生成] 结果已保存：JSON={Path(artifact_payload.get('json_file', '')).name or 'N/A'} | "
                                f"Bundle={Path(artifact_payload.get('bundle_file', '')).name or 'N/A'}"
                            ),
                            job_type="generation",
                            status=run_status,
                        ).to_dict(),
                    )
                except Exception as save_err:
                    record_generation_job_event(
                        job_id,
                        create_runtime_event(
                            level="WARNING",
                            kind="warning",
                            source="PaperBananaDemo",
                            message="[生成] 结果写盘失败",
                            job_type="generation",
                            status="running",
                            details=safe_log_text(save_err),
                        ).to_dict(),
                    )

                with job.lock:
                    job.results = results
                    job.summary = artifact_payload.get("summary", build_result_summary(results))
                    job.failures = artifact_payload.get("failures", build_failure_manifest(results))
                    job.json_file = artifact_payload.get("json_file", "")
                    job.bundle_file = artifact_payload.get("bundle_file", "")
                    job.progress_done = len(results)
                    job.progress_total = requested_candidates
                    job.effective_concurrent = int(used_concurrency)
                    job.estimated_batches = math.ceil(requested_candidates / max(1, used_concurrency))
                    job.elapsed_seconds = time.perf_counter() - started_at
                    job.status = run_status
                record_generation_job_event(
                    job_id,
                    create_runtime_event(
                        level="INFO",
                        kind="job",
                        source="PaperBananaDemo",
                        message=(
                            f"[生成] 后台任务结束：状态={run_status} | "
                            f"完成={len(results)}/{requested_candidates}"
                        ),
                        job_type="generation",
                        status=run_status,
                    ).to_dict(),
                )
            except Exception as exc:
                with job.lock:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.elapsed_seconds = time.perf_counter() - started_at
                record_generation_job_event(
                    job_id,
                    create_runtime_event(
                        level="ERROR",
                        kind="error",
                        source="PaperBananaDemo",
                        message="[生成] 后台任务失败",
                        job_type="generation",
                        status="failed",
                        details=job.error,
                    ).to_dict(),
                )

    job.future = GENERATION_JOB_EXECUTOR.submit(worker)
    return job_id


async def process_parallel_candidates(
    data_list,
    dataset_name=DEFAULT_DATASET_NAME,
    task_name="diagram",
    exp_mode="dev_planner_critic",
    retrieval_setting="auto",
    curated_profile=DEFAULT_CURATED_PROFILE,
    model_name="",
    image_model_name="",
    provider=DEFAULT_PROVIDER,
    connection_id="",
    api_key="",
    base_url="",
    extra_headers: dict[str, str] | None = None,
    concurrency_mode="auto",
    max_concurrent=20,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    event_callback: Optional[Callable[[dict], None]] = None,
    result_callback: Optional[Callable[[dict], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
):
    """使用 PaperVizProcessor 并行处理多个候选方案。"""
    task_name = normalize_task_name(task_name)
    retrieval_setting = normalize_retrieval_setting(retrieval_setting)
    curated_profile = normalize_curated_profile_name(curated_profile)
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
    logger.debug(
        "process_parallel_candidates start | task=%s provider=%s model=%s image_model=%s mode=%s retrieval=%s candidates=%s concurrency=%s/%s -> %s",
        task_name,
        provider,
        model_name,
        image_model_name,
        exp_mode,
        retrieval_setting,
        total_candidates,
        concurrency_mode,
        max_concurrent,
        effective_concurrent,
    )
    emit_generation_event(
        message=(
            f"[生成] 已启动：task={task_name} | provider={provider} | "
            f"流水线={exp_mode} | 检索={retrieval_setting} | 候选={total_candidates}"
        ),
        event_callback=event_callback,
        status_callback=status_callback,
        kind="job",
        level="INFO",
        status="running",
        provider=provider,
        model=model_name,
    )
    emit_generation_event(
        message=(
            f"[生成] 运行配置：文本模型={model_name or 'N/A'} | "
            f"图像模型={image_model_name or 'N/A'} | 有效并发={effective_concurrent}"
        ),
        event_callback=event_callback,
        status_callback=status_callback,
        kind="job",
        level="INFO",
        status="running",
        provider=provider,
        model=model_name or image_model_name,
    )

    if progress_callback:
        try:
            progress_callback(0, total_candidates, effective_concurrent)
        except Exception as cb_error:
            logger.warning("生成进度回调初始化失败: %s", safe_log_text(cb_error))
    if status_callback and event_callback is None:
        try:
            status_callback(
                f"任务启动：候选数={total_candidates}, 并发={effective_concurrent}, 流水线={exp_mode}"
            )
        except Exception as cb_error:
            logger.warning("生成文本状态回调初始化失败: %s", safe_log_text(cb_error))

    from utils import generation_utils
    runtime_settings = resolve_runtime_settings(
        provider,
        connection_id=connection_id,
        api_key=api_key,
        model_name=model_name,
        image_model_name=image_model_name,
        base_url=base_url,
        extra_headers=extra_headers,
        concurrency_mode=concurrency_mode,
        max_concurrent=max_concurrent,
        base_dir=REPO_ROOT,
        model_config_data=model_config_data,
    )
    runtime_context = build_runtime_context(
        runtime_settings,
        event_hook=event_callback,
        status_hook=status_callback if event_callback is None else None,
        cancel_check=cancel_check,
    )
    if not runtime_settings.api_key:
        emit_generation_event(
            message="[生成] 未提供 API Key，Provider 可能无法正常工作",
            event_callback=event_callback,
            status_callback=status_callback,
            kind="warning",
            level="WARNING",
            status="running",
            provider=runtime_settings.provider,
        )

    # 创建实验配置
    exp_config = config.ExpConfig(
        dataset_name=dataset_name,
        task_name=task_name,
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        concurrency_mode=runtime_settings.concurrency_mode,
        max_concurrent=runtime_settings.max_concurrent,
        curated_profile=curated_profile,
        model_name=runtime_settings.model_name,
        image_model_name=runtime_settings.image_model_name,
        provider=runtime_settings.provider,
        connection_id=runtime_settings.connection_id,
        provider_display_name=runtime_settings.provider_display_name,
        work_dir=REPO_ROOT,
    )
    emit_generation_event(
        message=(
            f"[生成] ExpConfig 就绪：provider={exp_config.provider} | "
            f"text={exp_config.model_name or 'N/A'} | image={exp_config.image_model_name or 'N/A'}"
        ),
        event_callback=event_callback,
        status_callback=status_callback,
        kind="job",
        level="INFO",
        status="running",
        provider=exp_config.provider,
        model=exp_config.model_name,
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
            emit_generation_event(
                message="[生成] 处理器已就绪，开始并发调度候选",
                event_callback=event_callback,
                status_callback=status_callback,
                kind="job",
                level="INFO",
                status="running",
                provider=exp_config.provider,
                model=exp_config.model_name,
            )
            async for result_data in processor.process_queries_batch(
                data_list,
                max_concurrent=concurrent_num,
                do_eval=False,
                status_callback=status_callback if event_callback is None else None,
                event_callback=event_callback,
                cancel_check=cancel_check,
            ):
                results.append(result_data)
                if result_callback:
                    try:
                        result_callback(result_data)
                    except Exception as cb_error:
                        logger.warning("生成结果回调失败: %s", safe_log_text(cb_error))
                if progress_callback:
                    try:
                        progress_callback(len(results), total_candidates, effective_concurrent)
                    except Exception as cb_error:
                        logger.warning("生成进度回调更新失败: %s", safe_log_text(cb_error))
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
    provider=DEFAULT_PROVIDER,
    connection_id="",
    image_model_name="",
    base_url="",
    extra_headers: dict[str, str] | None = None,
    image_generation_options: dict[str, Any] | None = None,
    task_id: int = 1,
    event_callback: Optional[Callable[[dict], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    input_mime_type: str = "image/png",
    max_attempts: Optional[int] = None,
    max_total_seconds: Optional[float] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    runtime_context=None,
):
    """
    使用图像编辑 API 精修图像，支持 Gemini、Evolink 和 OpenAI 图像 Provider。

    参数：
        image_bytes: 图像字节数据
        edit_prompt: 描述所需修改的文本
        aspect_ratio: 输出宽高比 (21:9, 16:9, 3:2)
        image_size: 输出分辨率 (2K 或 4K)
        api_key: API 密钥
        provider: "gemini"、"evolink"、"openai" 或 "openai_compatible"

    返回：
        元组 (编辑后的图像字节数据, 成功消息)
    """
    from utils import generation_utils

    attempt = 0
    sleep_seconds = 2.0
    started_at = time.perf_counter()
    task_prefix = f"task#{task_id}"
    normalized_mime_type = normalize_image_mime_type(input_mime_type)
    runtime_settings = resolve_runtime_settings(
        provider,
        connection_id=connection_id,
        api_key=api_key,
        image_model_name=image_model_name,
        base_url=base_url,
        extra_headers=extra_headers,
        base_dir=REPO_ROOT,
        model_config_data=model_config_data,
    )
    timeout_seconds = get_refine_request_timeout_seconds(runtime_settings.provider)
    attempt_limit = max_attempts or get_refine_max_attempts(runtime_settings.provider)
    total_time_limit = max_total_seconds or get_refine_total_timeout_seconds(runtime_settings.provider)
    active_runtime_context = runtime_context or build_runtime_context(
        runtime_settings,
        event_hook=event_callback,
        status_hook=status_callback if event_callback is None else None,
        cancel_check=cancel_check,
    )
    owns_runtime_context = runtime_context is None

    try:
        with generation_utils.use_runtime_context(active_runtime_context):
            while attempt < attempt_limit:
                if cancel_check and cancel_check():
                    emit_refine_event(
                        message=f"[精修][任务 {task_id}] 已取消，未开始第 {attempt + 1} 次尝试",
                        event_callback=event_callback,
                        status_callback=status_callback,
                        kind="warning",
                        level="WARNING",
                        status="cancelled",
                        provider=runtime_settings.provider,
                        model=runtime_settings.image_model_name,
                    )
                    return None, "⛔ 已取消精修任务"
                elapsed = time.perf_counter() - started_at
                if elapsed >= total_time_limit:
                    break
                attempt += 1
                try:
                    if runtime_settings.provider == "gemini":
                        # ====== Gemini 路径：复用统一的阶梯降级与可取消无限重试 ======
                        from google.genai import types

                        emit_refine_event(
                            message=f"[精修][任务 {task_id}] 开始请求，模型={runtime_settings.image_model_name}",
                            event_callback=event_callback,
                            status_callback=status_callback,
                            kind="job",
                            level="INFO",
                            status="running",
                            provider=runtime_settings.provider,
                            model=runtime_settings.image_model_name,
                        )
                        response_list = await generation_utils.call_gemini_with_retry_async(
                            model_name=runtime_settings.image_model_name,
                            contents=[
                                {
                                    "type": "text",
                                    "text": image_utils.build_gemini_image_prompt(
                                        edit_prompt,
                                        aspect_ratio=aspect_ratio,
                                        image_size=image_size,
                                    ),
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": normalized_mime_type,
                                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                                    },
                                },
                            ],
                            config=types.GenerateContentConfig(
                                temperature=1.0,
                                max_output_tokens=8192,
                                response_modalities=["IMAGE"],
                            ),
                            max_attempts=max(2, int(max_attempts or 2)),
                            retry_delay=5,
                            error_context=f"refine-image[{task_prefix}]",
                        )

                        if response_list and response_list[0] and response_list[0] != "Error":
                            emit_refine_event(
                                message=f"[精修][任务 {task_id}] 已成功生成结果，模型={runtime_settings.image_model_name}",
                                event_callback=event_callback,
                                status_callback=status_callback,
                                kind="job",
                                level="INFO",
                                status="completed",
                                provider=runtime_settings.provider,
                                model=runtime_settings.image_model_name,
                            )
                            return base64.b64decode(response_list[0]), "✅ 图像精修成功！"

                        emit_refine_event(
                            message=f"[精修][任务 {task_id}] Gemini 返回不可恢复错误",
                            event_callback=event_callback,
                            status_callback=status_callback,
                            kind="error",
                            level="ERROR",
                            status="failed",
                            provider=runtime_settings.provider,
                            model=runtime_settings.image_model_name,
                        )
                        return None, "❌ 图像精修失败：Gemini 返回不可恢复错误"

                    if runtime_settings.provider == "evolink":
                        # ====== Evolink 路径：上传图片获取 URL → image_urls ======
                        evolink_provider = generation_utils.get_evolink_provider()
                        if evolink_provider is None:
                            await asyncio.sleep(min(sleep_seconds, 10.0))
                            sleep_seconds = min(sleep_seconds * 1.2, 15.0)
                            continue

                        image_model = runtime_settings.image_model_name
                        emit_refine_event(
                            message=f"[精修][任务 {task_id}] 第 {attempt} 次尝试，模型={image_model}，超时={int(timeout_seconds)}s",
                            event_callback=event_callback,
                            status_callback=status_callback,
                            kind="job",
                            level="INFO",
                            status="running",
                            provider=runtime_settings.provider,
                            model=image_model,
                            attempt=attempt,
                        )

                        # 步骤 1：上传原始图片到 Evolink 文件服务
                        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                        ref_image_url = await generation_utils.upload_image_to_evolink(
                            image_b64,
                            media_type=normalized_mime_type,
                        )
                        logger.debug(
                            "refine upload complete | task=%s attempt=%s ref=%s",
                            task_prefix,
                            attempt,
                            safe_log_text(ref_image_url[:80]),
                        )

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
                            emit_refine_event(
                                message=f"[精修][{task_prefix}] success on attempt={attempt} model={image_model}",
                                event_callback=event_callback,
                                status_callback=status_callback,
                                kind="job",
                                level="INFO",
                                status="completed",
                                provider=runtime_settings.provider,
                                model=image_model,
                                attempt=attempt,
                            )
                            return edited_image_data, f"✅ 图像精修成功！（第 {attempt} 次尝试）"

                        raise RuntimeError("Evolink 未返回有效图像数据")

                    if runtime_settings.provider in {"openai", "openai_compatible"}:
                        emit_refine_event(
                            message=f"[精修][任务 {task_id}] 开始请求，模型={runtime_settings.image_model_name}",
                            event_callback=event_callback,
                            status_callback=status_callback,
                            kind="job",
                            level="INFO",
                            status="running",
                            provider=runtime_settings.provider,
                            model=runtime_settings.image_model_name,
                            attempt=attempt,
                        )
                        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                        openai_refine_options = {
                            "aspect_ratio": aspect_ratio,
                            "image_resolution": image_size,
                            "output_format": "png",
                            **dict(image_generation_options or {}),
                        }
                        response_list = await asyncio.wait_for(
                            generation_utils.call_openai_image_generation_with_retry_async(
                                model_name=runtime_settings.image_model_name,
                                prompt=edit_prompt,
                                config=openai_refine_options,
                                contents=[
                                    {"type": "text", "text": edit_prompt},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": normalized_mime_type,
                                            "data": image_b64,
                                        },
                                    },
                                ],
                                provider_type=runtime_settings.provider,
                                max_attempts=max(1, attempt_limit - attempt + 1),
                                retry_delay=3,
                                error_context=f"refine-image[{task_prefix}]",
                            ),
                            timeout=min(total_time_limit, timeout_seconds * max(1, attempt_limit - attempt + 1)),
                        )
                        if response_list and response_list[0] and response_list[0] != "Error":
                            emit_refine_event(
                                message=f"[精修][任务 {task_id}] 已成功生成结果，模型={runtime_settings.image_model_name}",
                                event_callback=event_callback,
                                status_callback=status_callback,
                                kind="job",
                                level="INFO",
                                status="completed",
                                provider=runtime_settings.provider,
                                model=runtime_settings.image_model_name,
                                attempt=attempt,
                            )
                            return base64.b64decode(response_list[0]), "✅ 图像精修成功！"
                        return None, "❌ 图像精修失败：OpenAI 未返回有效图像"

                    raise RuntimeError(
                        f"当前精修链路仅支持 Gemini、Evolink 或 OpenAI 图像编辑接口，暂不支持 {runtime_settings.provider}。"
                    )

                except asyncio.TimeoutError:
                    err_text = (
                        f"{runtime_settings.provider} request timed out after {int(timeout_seconds)}s "
                        f"(attempt={attempt}, {task_prefix})"
                    )
                    delay = min(max(sleep_seconds, 3.0), 20.0)
                    emit_refine_event(
                        message=f"[精修][任务 {task_id}] 请求超时，{delay:.1f}s 后重试",
                        event_callback=event_callback,
                        status_callback=status_callback,
                        kind="retry",
                        level="WARNING",
                        status="retrying",
                        provider=runtime_settings.provider,
                        model=runtime_settings.image_model_name,
                        attempt=attempt,
                        details=err_text,
                    )
                    await asyncio.sleep(delay)
                    sleep_seconds = min(max(delay * 1.2, sleep_seconds * 1.25), 30.0)
                except Exception as e:
                    # 不向前端抛错，持续重试直到成功
                    error_text = safe_log_text(e, max_len=2000)
                    lower_error = error_text.lower()
                    if runtime_settings.provider in {"openai", "openai_compatible"} and "openai 图像生成在" in lower_error:
                        emit_refine_event(
                            message=f"[精修][任务 {task_id}] OpenAI 图像精修失败：{error_text[:160]}",
                            event_callback=event_callback,
                            status_callback=status_callback,
                            kind="error",
                            level="ERROR",
                            status="failed",
                            provider=runtime_settings.provider,
                            model=runtime_settings.image_model_name,
                            attempt=attempt,
                            details=error_text,
                        )
                        return None, f"❌ 图像精修失败：{error_text[:300]}"

                    # Windows 套接字异常自愈：重建 client/provider 后继续。
                    if "winerror 10038" in lower_error:
                        try:
                            generation_utils.reinitialize_runtime_context(active_runtime_context)
                        except Exception as reinit_error:
                            logger.warning("精修 socket 自愈重建失败: %s", safe_log_text(reinit_error))

                    suggested_delay = extract_retry_delay_seconds(error_text)
                    delay = min(sleep_seconds, 20.0)
                    if suggested_delay is not None:
                        delay = min(max(delay, suggested_delay), 60.0)
                    if "limit: 0" in lower_error:
                        delay = max(delay, 30.0)
                    emit_refine_event(
                        message=f"[精修][任务 {task_id}] 第 {attempt} 次失败，{delay:.1f}s 后重试 | {error_text[:160]}",
                        event_callback=event_callback,
                        status_callback=status_callback,
                        kind="retry",
                        level="WARNING",
                        status="retrying",
                        provider=runtime_settings.provider,
                        model=runtime_settings.image_model_name,
                        attempt=attempt,
                        details=error_text,
                    )
                    await asyncio.sleep(delay)
                    sleep_seconds = min(max(delay * 1.1, sleep_seconds * 1.25), 30.0)

        elapsed = time.perf_counter() - started_at
        if cancel_check and cancel_check():
            emit_refine_event(
                message=f"[精修][任务 {task_id}] 已取消，累计耗时 {elapsed:.1f}s",
                event_callback=event_callback,
                status_callback=status_callback,
                kind="warning",
                level="WARNING",
                status="cancelled",
                provider=runtime_settings.provider,
                model=runtime_settings.image_model_name,
            )
            return None, "⛔ 已取消精修任务"
        failure_message = (
            f"❌ 图像精修失败：已达到重试上限（attempts={attempt_limit}, elapsed={elapsed:.1f}s）"
        )
        emit_refine_event(
            message=f"[精修][任务 {task_id}] 已达到重试上限，累计耗时 {elapsed:.1f}s",
            event_callback=event_callback,
            status_callback=status_callback,
            kind="error",
            level="ERROR",
            status="failed",
            provider=runtime_settings.provider,
            model=runtime_settings.image_model_name,
            details=failure_message,
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
    provider=DEFAULT_PROVIDER,
    connection_id="",
    image_model_name="",
    base_url="",
    extra_headers: dict[str, str] | None = None,
    image_generation_options: dict[str, Any] | None = None,
    input_mime_type="image/png",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    event_callback: Optional[Callable[[dict], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
):
    """并发精修多张图像，按完成先后收集，避免被单一慢任务阻塞观感。"""
    safe_count = max(1, int(num_images))
    results = [None] * safe_count
    tasks = []
    runtime_settings = resolve_runtime_settings(
        provider,
        connection_id=connection_id,
        api_key=api_key,
        image_model_name=image_model_name,
        base_url=base_url,
        extra_headers=extra_headers,
        base_dir=REPO_ROOT,
        model_config_data=model_config_data,
    )
    runtime_context = build_runtime_context(
        runtime_settings,
        event_hook=event_callback,
        status_hook=status_callback if event_callback is None else None,
        cancel_check=cancel_check,
    )
    semaphore = asyncio.Semaphore(min(safe_count, 3 if runtime_settings.provider == "gemini" else 2))

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
                                connection_id=runtime_settings.connection_id,
                                image_model_name=runtime_settings.image_model_name,
                                base_url=runtime_settings.base_url,
                                extra_headers=runtime_settings.extra_headers,
                                image_generation_options=image_generation_options,
                                task_id=task_idx + 1,
                                event_callback=event_callback,
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
                        logger.warning("精修进度回调失败: %s", safe_log_text(cb_error))
                emit_refine_event(
                    message=f"[精修] 已完成 {done_count}/{safe_count}（任务 {task_idx + 1}）",
                    event_callback=event_callback,
                    status_callback=status_callback,
                    kind="job",
                    level="INFO",
                    status="running",
                    provider=runtime_settings.provider,
                    model=runtime_settings.image_model_name,
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
    connection_id: str = "",
    image_model_name: str,
    base_url: str = "",
    extra_headers: dict[str, str] | None = None,
    image_generation_options: dict[str, Any] | None = None,
    input_mime_type: str,
) -> str:
    job_id = f"refine_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    runtime_settings = resolve_runtime_settings(
        provider,
        connection_id=connection_id,
        api_key=api_key,
        image_model_name=image_model_name,
        base_url=base_url,
        extra_headers=extra_headers,
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
        connection_id=runtime_settings.connection_id,
        provider_display_name=runtime_settings.provider_display_name,
    )
    _store_refine_job(job)
    record_refine_job_event(
        job_id,
        create_runtime_event(
            level="INFO",
            kind="job",
            source="PaperBananaDemo",
            message=f"[精修] 输入图像：格式={input_mime_type} | 大小={len(image_bytes)} bytes",
            job_type="refine",
            provider=runtime_settings.provider,
            model=runtime_settings.image_model_name,
            status="running",
        ).to_dict(),
    )
    record_refine_job_event(
        job_id,
        create_runtime_event(
            level="INFO",
            kind="job",
            source="PaperBananaDemo",
            message=(
                f"[精修] 已启动：provider={runtime_settings.provider} | "
                f"model={runtime_settings.image_model_name} | 分辨率={image_size} | "
                f"宽高比={aspect_ratio} | 目标张数={int(num_images)}"
            ),
            job_type="refine",
            provider=runtime_settings.provider,
            model=runtime_settings.image_model_name,
            status="running",
        ).to_dict(),
    )

    def worker():
        started_at = time.perf_counter()
        with capture_job_logs(job_id, "refine"):
            try:
                def on_progress(done_count: int, total_count: int):
                    update_refine_job_progress(job_id, done_count, total_count)

                def on_status(message: str):
                    append_refine_job_status(job_id, message)

                def on_event(event: dict):
                    record_refine_job_event(job_id, event)

                refined_results = asyncio.run(
                    refine_images_with_count(
                        image_bytes=image_bytes,
                        edit_prompt=edit_prompt,
                        num_images=int(num_images),
                        aspect_ratio=aspect_ratio,
                        image_size=image_size,
                        api_key=runtime_settings.api_key,
                        provider=runtime_settings.provider,
                        connection_id=runtime_settings.connection_id,
                        image_model_name=runtime_settings.image_model_name,
                        base_url=runtime_settings.base_url,
                        extra_headers=runtime_settings.extra_headers,
                        image_generation_options=image_generation_options,
                        input_mime_type=input_mime_type,
                        progress_callback=on_progress,
                        event_callback=on_event,
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
                    elif refined_images:
                        job.status = "completed"
                    else:
                        job.status = "failed"
                        if not job.error and failed_refine_results:
                            job.error = "; ".join(
                                str(item.get("message", "未知精修错误"))
                                for item in failed_refine_results[:3]
                            )
                record_refine_job_event(
                    job_id,
                    create_runtime_event(
                        level="INFO",
                        kind="job",
                        source="PaperBananaDemo",
                        message=(
                            f"[精修] 后台任务结束：状态={job.status} | "
                            f"成功={len(refined_images)} | 失败={len(failed_refine_results)}"
                        ),
                        job_type="refine",
                        provider=runtime_settings.provider,
                        model=runtime_settings.image_model_name,
                        status=job.status,
                    ).to_dict(),
                )
            except Exception as exc:
                with job.lock:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.elapsed_seconds = time.perf_counter() - started_at
                record_refine_job_event(
                    job_id,
                    create_runtime_event(
                        level="ERROR",
                        kind="error",
                        source="PaperBananaDemo",
                        message="[精修] 后台任务失败",
                        job_type="refine",
                        provider=runtime_settings.provider,
                        model=runtime_settings.image_model_name,
                        status="failed",
                        details=job.error,
                    ).to_dict(),
                )

    job.future = REFINE_JOB_EXECUTOR.submit(worker)
    return job_id


def display_candidate_result(
    result,
    candidate_id,
    exp_mode,
    *,
    task_name="diagram",
    candidate_index: int = 0,
):
    """展示单个候选方案的结果。"""
    task_name = normalize_task_name(task_name)
    task_config = get_task_ui_config(task_name)
    candidate_label = format_candidate_display_label(
        result,
        fallback_index=candidate_index,
    )

    if isinstance(result, dict) and result.get("status") == "failed":
        st.error(f"{candidate_label} 失败：{result.get('error', 'Unknown error')}")
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

    # 展示最终图像（缩略图 + 点击查看原图）
    if final_image_key and final_image_key in result:
        img = base64_to_image(result[final_image_key])
        if img:
            # 网格预览使用 JPEG 缩略图，减少前端传输量
            thumb_bytes = image_to_jpeg_thumbnail(img)
            st.image(
                thumb_bytes,
                width="stretch",
                caption=f"{candidate_label}（{task_config['final_caption']}）",
            )

            # 点击展开查看原图
            with st.expander("🔍 查看原图", expanded=False):
                st.image(img, width="stretch")

            # 添加下载按钮（下载原始 PNG）
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            st.download_button(
                label="📥 下载图片",
                data=buffered.getvalue(),
                file_name=f"candidate_{candidate_id}.png",
                mime="image/png",
                key=f"download_candidate_{candidate_id}",
                width="stretch"
            )
        else:
            st.error(f"{candidate_label} 的图像解码失败")
    else:
        st.warning(f"{candidate_label} 未生成图像")

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
                label="📥 下载代码",
                data=code_text.encode("utf-8"),
                file_name=f"candidate_{candidate_id}.py",
                mime="text/x-python",
                key=f"download_candidate_code_{candidate_id}",
                width="stretch",
            )

    decision_state = get_generation_candidate_decision(candidate_id)
    st.caption(f"当前决策：{GENERATION_CANDIDATE_DECISION_LABELS.get(decision_state, '待决策')}")
    decision_cols = st.columns(3)
    with decision_cols[0]:
        favorite_label = "取消收藏" if decision_state == "favorite" else "⭐ 收藏"
        if st.button(
            favorite_label,
            key=f"candidate_favorite_{task_name}_{candidate_id}",
            width="stretch",
        ):
            set_generation_candidate_decision(
                candidate_id,
                "default" if decision_state == "favorite" else "favorite",
            )
            persist_demo_ui_state()
            st.rerun()
    with decision_cols[1]:
        final_label = "取消最终" if decision_state == "final" else "✅ 设为最终"
        if st.button(
            final_label,
            key=f"candidate_final_{task_name}_{candidate_id}",
            width="stretch",
        ):
            set_generation_candidate_decision(
                candidate_id,
                "default" if decision_state == "final" else "final",
            )
            persist_demo_ui_state()
            st.rerun()
    with decision_cols[2]:
        discard_label = "恢复候选" if decision_state == "discarded" else "🚫 淘汰"
        if st.button(
            discard_label,
            key=f"candidate_discard_{task_name}_{candidate_id}",
            width="stretch",
        ):
            set_generation_candidate_decision(
                candidate_id,
                "default" if decision_state == "discarded" else "discarded",
            )
            persist_demo_ui_state()
            st.rerun()

    action_column_count = 2 if task_name == "plot" and final_code_key and result.get(final_code_key) else 1
    action_cols = st.columns(action_column_count)
    with action_cols[0]:
        if st.button(
            "✨ 送去精修",
            key=f"stage_candidate_for_refine_{task_name}_{candidate_id}",
            width="stretch",
        ):
            if stage_candidate_for_refine(
                result,
                candidate_id=candidate_id,
                exp_mode=exp_mode,
                task_name=task_name,
            ):
                persist_demo_ui_state()
                st.success(f"已将 {candidate_label} 载入精修工作台。")
            else:
                st.warning(f"当前{candidate_label}缺少可用于精修的最终图像。")
    if action_column_count > 1:
        with action_cols[1]:
            if st.button(
                "↺ 载入代码重渲染",
                key=f"stage_plot_rerender_{candidate_id}",
                width="stretch",
            ):
                if stage_plot_code_for_rerender(
                    result,
                    candidate_id=candidate_id,
                    exp_mode=exp_mode,
                ):
                    st.success(f"已将 {candidate_label} 的代码载入重渲染工作台。")
                else:
                    st.warning(f"当前{candidate_label}没有可编辑的最终绘图代码。")

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
                    st.image(stage_img, width="stretch")

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


def render_plot_rerender_workspace() -> None:
    staged_code = clean_text(st.session_state.get("plot_rerender_code", ""))
    if not staged_code:
        return

    candidate_id = st.session_state.get("plot_rerender_candidate_id", "N/A")
    candidate_label = st.session_state.get(
        "plot_rerender_candidate_label",
        format_candidate_slot_label(candidate_id),
    )
    st.divider()
    st.markdown("## 🧪 绘图代码重渲染工作台")
    st.caption(f"来源：{candidate_label}。你可以直接编辑 Matplotlib 代码并本地预览。")

    if "plot_rerender_code_editor" not in st.session_state:
        st.session_state["plot_rerender_code_editor"] = staged_code

    st.text_area(
        "Matplotlib 代码",
        height=260,
        key="plot_rerender_code_editor",
        help="支持直接粘贴或修改候选方案的最终绘图代码，然后点击重新渲染。",
    )

    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        if st.button("🔄 重新渲染预览", width="stretch"):
            current_code = clean_text(st.session_state.get("plot_rerender_code_editor", ""))
            st.session_state["plot_rerender_code"] = current_code
            st.session_state["plot_rerender_preview"] = execute_plot_code_with_details(current_code)
            st.rerun()
    with action_col2:
        if st.button("✨ 预览送去精修", width="stretch"):
            preview = st.session_state.get("plot_rerender_preview", {})
            if preview.get("success") and preview.get("base64_jpg"):
                stage_refine_source_image(
                    base64.b64decode(preview["base64_jpg"]),
                    input_mime_type="image/jpeg",
                    source_label=f"Plot 重渲染预览（{candidate_label}）",
                    default_prompt="保持语义不变，优化布局、标签清晰度、留白和整体视觉层次。",
                )
                st.success("已将重渲染预览送入精修工作台。")
            else:
                st.warning("当前还没有成功的预览结果，请先执行一次重渲染。")
    with action_col3:
        if st.button("🧹 清空工作台", width="stretch"):
            for key in (
                "plot_rerender_code",
                "plot_rerender_code_editor",
                "plot_rerender_candidate_id",
                "plot_rerender_candidate_label",
                "plot_rerender_source_desc_key",
                "plot_rerender_preview",
            ):
                st.session_state.pop(key, None)
            st.rerun()

    preview = st.session_state.get("plot_rerender_preview")
    if not isinstance(preview, dict):
        return

    if preview.get("success") and preview.get("base64_jpg"):
        preview_image = base64_to_image(preview.get("base64_jpg"))
        if preview_image is not None:
            st.image(preview_image, width="stretch", caption="重渲染预览")
            st.download_button(
                label="📥 下载预览 JPEG",
                data=base64.b64decode(preview["base64_jpg"]),
                file_name=f"plot_rerender_preview_{candidate_id}.jpg",
                mime="image/jpeg",
                width="stretch",
            )
    elif preview:
        st.error("重渲染失败，请检查代码输出和报错信息。")

    if preview:
        with st.expander("查看重渲染诊断", expanded=not preview.get("success", False)):
            if preview.get("exception"):
                st.code(clean_text(preview["exception"]))
            if preview.get("stdout"):
                st.markdown("**stdout**")
                st.code(clean_text(preview["stdout"]))
            if preview.get("stderr"):
                st.markdown("**stderr**")
                st.code(clean_text(preview["stderr"]))


def render_generation_history_panel(task_name: str) -> None:
    with st.expander("📚 历史运行回放", expanded=False):
        bundle_files = list_demo_bundle_files(task_name)
        if not bundle_files:
            st.info("当前任务还没有历史 bundle。先运行一次生成任务后，这里会自动出现可回放记录。")
            return

        display_labels = build_demo_bundle_display_labels(bundle_files)
        bundle_options = [str(path) for path in bundle_files]
        selected_path_str = st.selectbox(
            "选择要回放的 bundle",
            bundle_options,
            key=f"history_bundle_select_{normalize_task_name(task_name)}",
            format_func=lambda bundle_path: display_labels.get(bundle_path, bundle_path),
        )
        st.caption(f"共发现 {len(bundle_files)} 条历史记录，可在下拉列表中继续向下滚动查看更早结果。")
        selected_path = Path(selected_path_str)
        st.caption(f"文件：`{format_repo_relative_path(selected_path)}`")

        normalized_tn = normalize_task_name(task_name)
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("📥 载入历史结果", key=f"history_load_{normalized_tn}", width="stretch"):
                snapshot = load_generation_history_snapshot(selected_path)
                persist_generation_job_results(
                    snapshot,
                    source_label=f"历史回放：{selected_path.name}",
                )
                persist_demo_ui_state()
                st.rerun()
        with action_col2:
            if st.button("🧹 清除当前结果", key=f"history_clear_{normalized_tn}", width="stretch"):
                for key in (
                    "results",
                    "task_name",
                    "dataset_name",
                    "exp_mode",
                    "concurrency_mode",
                    "max_concurrent",
                    "effective_concurrent",
                    "estimated_batches",
                    "timestamp",
                    "json_file",
                    "bundle_file",
                    "generation_summary",
                    "generation_failures",
                    "requested_candidates",
                    "result_source_label",
                    "generation_candidate_decisions",
                    "generation_final_candidate_id",
                ):
                    st.session_state.pop(key, None)
                for key in (
                    f"generation_display_scope_{normalize_task_name(task_name)}",
                    f"generation_export_scope_{normalize_task_name(task_name)}",
                ):
                    st.session_state.pop(key, None)
                persist_demo_ui_state()
                st.rerun()


def render_generation_runtime_panel(snapshot: dict | None, *, requested_candidates: int) -> str | None:
    """在首屏展示后台生成任务的实时状态、候选阶段和最近日志。"""
    if not snapshot:
        return None

    status = snapshot.get("status", "running")
    status_titles = {
        "running": "🚧 生成任务运行中",
        "completed": "✅ 生成任务已完成",
        "cancelled": "⛔ 生成任务已停止",
        "failed": "❌ 生成任务失败",
        "interrupted": "🧭 已恢复历史生成快照",
    }
    status_labels = {
        "running": "运行中",
        "completed": "已完成",
        "cancelled": "已停止",
        "failed": "失败",
        "interrupted": "已恢复",
    }
    progress_done = int(snapshot.get("progress_done", 0) or 0)
    progress_total = int(snapshot.get("progress_total", requested_candidates) or requested_candidates or 0)
    progress_total = max(progress_total, 1)
    ratio = min(progress_done / progress_total, 1.0)
    candidate_stage_map = snapshot.get("candidate_stage_map", {}) or {}
    stage_counter = Counter(candidate_stage_map.values())
    status_lines = snapshot.get("status_history", []) or []
    latest_lines = status_lines[-GENERATION_LOG_RENDER_LIMIT:]

    def _candidate_sort_key(item):
        candidate_id = str(item[0])
        try:
            return (0, int(candidate_id))
        except ValueError:
            return (1, candidate_id)

    with st.container(border=True):
        st.markdown(f"### {status_titles.get(status, '🧭 生成任务状态')}")
        metric_cols = st.columns(4)
        metric_cols[0].metric("状态", status_labels.get(status, status))
        metric_cols[1].metric("进度", f"{progress_done}/{progress_total}")
        metric_cols[2].metric("有效并发", int(snapshot.get("effective_concurrent", 0) or 0))
        metric_cols[3].metric("最近日志", len(status_lines))

        st.caption(
            f"Provider：{snapshot.get('provider')} | 文本模型：{snapshot.get('model_name') or 'N/A'} | "
            f"图像模型：{snapshot.get('image_model_name') or 'N/A'} | "
            f"流水线：{snapshot.get('exp_mode')} | 检索：{snapshot.get('retrieval_setting')}"
        )
        st.progress(
            ratio,
            text=(
                f"总体进度：已完成 {progress_done}/{progress_total} 个候选 | "
                f"预计批次 {int(snapshot.get('estimated_batches', 0) or 0)}"
            ),
        )

        if stage_counter:
            summary_text = " | ".join(
                f"{stage} ×{count}"
                for stage, count in stage_counter.most_common()
            )
            st.caption(f"候选阶段汇总：{summary_text}")

        if candidate_stage_map:
            st.markdown("**候选实时阶段**")
            candidate_items = sorted(candidate_stage_map.items(), key=_candidate_sort_key)
            num_cols = min(3, len(candidate_items))
            for row_start in range(0, len(candidate_items), num_cols):
                cols = st.columns(num_cols)
                for offset, (col, (candidate_id, stage)) in enumerate(
                    zip(cols, candidate_items[row_start: row_start + num_cols]),
                    start=row_start,
                ):
                    with col:
                        with st.container(border=True):
                            st.caption(
                                format_candidate_slot_label(
                                    candidate_id,
                                    fallback_index=offset,
                                )
                            )
                            st.write(stage)

        if latest_lines:
            st.markdown("**实时日志**")
            st.code("\n".join(latest_lines), language="text")
            if len(status_lines) > len(latest_lines):
                with st.expander(f"查看全部最近 {len(status_lines)} 条日志", expanded=False):
                    st.code("\n".join(status_lines), language="text")

        if status in {"failed", "interrupted"} and snapshot.get("error"):
            st.error(f"后台任务报错：{snapshot.get('error')}")

        if status in {"completed", "cancelled"}:
            file_bits = []
            json_file = snapshot.get("json_file")
            bundle_file = snapshot.get("bundle_file")
            if json_file:
                file_bits.append(f"JSON: {Path(json_file).name}")
            if bundle_file:
                file_bits.append(f"Bundle: {Path(bundle_file).name}")
            if file_bits:
                st.caption("结果文件：" + " | ".join(file_bits))

        if status == "running":
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if st.button(
                    "🛑 停止生成",
                    key=f"stop_generation_panel_{snapshot.get('job_id', 'active')}",
                    width="stretch",
                ):
                    return "cancel"
            with action_col2:
                if st.button(
                    "🔄 刷新状态",
                    key=f"refresh_generation_panel_{snapshot.get('job_id', 'active')}",
                    width="stretch",
                ):
                    return "refresh"
    return None


def render_generation_live_stream(snapshot: dict | None) -> None:
    """在按钮下方展示候选流式卡片和事件时间线。"""
    if not snapshot:
        return

    candidate_snapshots = snapshot.get("candidate_snapshots", {}) or {}
    event_timeline = snapshot.get("event_history", snapshot.get("event_timeline", [])) or []
    if not candidate_snapshots and not event_timeline:
        return

    def _candidate_sort_key(item):
        candidate_id = str(item[0])
        try:
            return (0, int(candidate_id))
        except ValueError:
            return (1, candidate_id)

    st.markdown("### ⚡ 流式生成展示")
    st.caption("任何阶段变化都会立即写入这里；只要首张预览图生成成功，就不必等整条候选完全结束。")

    if candidate_snapshots:
        candidate_items = sorted(candidate_snapshots.items(), key=_candidate_sort_key)
        preview_width = get_generation_live_preview_width(len(candidate_items))
        row_layout = get_generation_live_row_layout(len(candidate_items))

        def _render_candidate_card(candidate_id, candidate_snapshot, *, fallback_index: int) -> None:
            with st.container(border=True):
                st.caption(
                    format_candidate_slot_label(
                        candidate_id,
                        fallback_index=fallback_index,
                    )
                )
                st.markdown(f"**{candidate_snapshot.get('stage', '等待开始')}**")
                status_label = candidate_snapshot.get("status", "queued")
                updated_at = candidate_snapshot.get("updated_at", "")
                st.caption(f"状态：{status_label} | 更新时间：{updated_at}")
                preview_image = candidate_snapshot.get("preview_image", "")
                if preview_image:
                    preview = base64_to_image(preview_image)
                    if preview is not None:
                        st.image(
                            preview,
                            width=preview_width,
                            caption=candidate_snapshot.get("preview_label", "最新预览"),
                        )
                else:
                    st.info("当前候选还没有可展示的图像预览。")

                if candidate_snapshot.get("error"):
                    st.error(candidate_snapshot["error"])

        if len(candidate_items) == 1:
            candidate_id, candidate_snapshot = candidate_items[0]
            _, center_col, _ = st.columns(row_layout, gap="large")
            with center_col:
                _render_candidate_card(candidate_id, candidate_snapshot, fallback_index=0)
        elif len(candidate_items) == 2:
            _, left_col, right_col, _ = st.columns(row_layout, gap="large")
            for offset, (col, (candidate_id, candidate_snapshot)) in enumerate(
                zip((left_col, right_col), candidate_items),
            ):
                with col:
                    _render_candidate_card(
                        candidate_id,
                        candidate_snapshot,
                        fallback_index=offset,
                    )
        else:
            num_cols = min(3, len(candidate_items))
            for row_start in range(0, len(candidate_items), num_cols):
                cols = st.columns(num_cols)
                for offset, (col, (candidate_id, candidate_snapshot)) in enumerate(
                    zip(cols, candidate_items[row_start: row_start + num_cols]),
                    start=row_start,
                ):
                    with col:
                        _render_candidate_card(
                            candidate_id,
                            candidate_snapshot,
                            fallback_index=offset,
                        )

    if event_timeline:
        _render_collapsible_event_timeline(
            event_timeline,
            limit=GENERATION_EVENT_RENDER_LIMIT,
            label="事件时间线",
        )


def render_refine_runtime_panel(snapshot: dict | None, *, requested_images: int) -> str | None:
    """在精修页首屏展示后台精修任务的实时状态和日志。"""
    if not snapshot:
        return None

    status = snapshot.get("status", "running")
    status_titles = {
        "running": "✨ 精修任务运行中",
        "completed": "✅ 精修任务已完成",
        "cancelled": "⛔ 精修任务已停止",
        "failed": "❌ 精修任务失败",
        "interrupted": "🧭 已恢复历史精修快照",
    }
    status_labels = {
        "running": "运行中",
        "completed": "已完成",
        "cancelled": "已停止",
        "failed": "失败",
        "interrupted": "已恢复",
    }
    progress_done = int(snapshot.get("progress_done", 0) or 0)
    progress_total = int(snapshot.get("progress_total", requested_images) or requested_images or 0)
    progress_total = max(progress_total, 1)
    ratio = min(progress_done / progress_total, 1.0)
    status_lines = snapshot.get("status_history", []) or []
    latest_lines = status_lines[-REFINE_LOG_RENDER_LIMIT:]
    event_history = snapshot.get("event_history", []) or []
    success_count = len(snapshot.get("refined_images", []))
    failed_count = len(snapshot.get("failed_results", []))

    with st.container(border=True):
        st.markdown(f"### {status_titles.get(status, '🧭 精修任务状态')}")
        metric_cols = st.columns(4)
        metric_cols[0].metric("状态", status_labels.get(status, status))
        metric_cols[1].metric("进度", f"{progress_done}/{progress_total}")
        metric_cols[2].metric("成功", success_count)
        metric_cols[3].metric("失败", failed_count)

        st.caption(
            f"Provider：{snapshot.get('provider')} | 模型：{snapshot.get('image_model_name') or 'N/A'} | "
            f"分辨率：{snapshot.get('resolution')} | 宽高比：{snapshot.get('aspect_ratio')} | "
            f"输入格式：{snapshot.get('input_mime_type')}"
        )
        st.progress(ratio, text=f"总体进度：已完成 {progress_done}/{progress_total} 张精修图")

        if latest_lines:
            st.markdown("**实时日志**")
            st.code("\n".join(latest_lines), language="text")
            if len(status_lines) > len(latest_lines):
                with st.expander(f"查看全部最近 {len(status_lines)} 条日志", expanded=False):
                    st.code("\n".join(status_lines), language="text")
        if event_history:
            _render_collapsible_event_timeline(
                event_history,
                limit=REFINE_EVENT_RENDER_LIMIT,
                label="事件时间线",
            )

        if status in {"failed", "interrupted"} and snapshot.get("error"):
            st.error(f"后台精修报错：{snapshot.get('error')}")

        if status == "running":
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if st.button(
                    "🛑 停止精修",
                    key=f"stop_refine_panel_{snapshot.get('job_id', 'active')}",
                    width="stretch",
                ):
                    return "cancel"
            with action_col2:
                if st.button(
                    "🔄 刷新状态",
                    key=f"refresh_refine_panel_{snapshot.get('job_id', 'active')}",
                    width="stretch",
                ):
                    return "refresh"
    return None


def render_refine_version_history_panel() -> None:
    removed_labels = sanitize_refine_version_history()
    history = get_refine_version_history()
    if removed_labels:
        preview_labels = "、".join(removed_labels[:3])
        if len(removed_labels) > 3:
            preview_labels += f" 等 {len(removed_labels)} 个版本"
        st.warning(f"已自动清理损坏的精修历史版本：{preview_labels}")
    if not history:
        return

    st.divider()
    st.markdown("## 🧬 精修版本历史")
    st.caption("每次精修结果都会形成独立版本分支。你可以从任意历史版本继续精修，或回退到它的父版本。")
    display_map = build_refine_version_display_map(history)

    history_options = {
        f"{get_refine_version_display_label(entry=entry, display_map=display_map, history=history)} | {entry.get('created_at', 'N/A')}": str(
            entry.get("version_key", "")
        )
        for entry in history
    }
    active_version_key = str(
        st.session_state.get("refine_active_version_key", history[-1].get("version_key", ""))
        or history[-1].get("version_key", "")
    )
    selected_history_label = st.selectbox(
        "选择要继续编辑的版本",
        list(history_options.keys()),
        index=max(
            0,
            next(
                (
                    idx
                    for idx, version_key in enumerate(history_options.values())
                    if version_key == active_version_key
                ),
                len(history_options) - 1,
            ),
        ),
        key="refine_history_version_picker",
    )
    selected_version_key = history_options[selected_history_label]
    selected_entry = find_refine_version_entry(selected_version_key)
    if selected_entry is None:
        return

    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("↺ 设为当前输入版本", width="stretch", key="refine_use_selected_version"):
            if activate_refine_version(selected_version_key):
                persist_demo_ui_state()
                st.rerun()
    with action_cols[1]:
        parent_version_key = str(selected_entry.get("parent_version_key", "") or "")
        if st.button(
            "⤴ 回退到父版本",
            width="stretch",
            key="refine_use_parent_version",
            disabled=not parent_version_key,
        ):
            if activate_refine_version(parent_version_key):
                persist_demo_ui_state()
                st.rerun()
    with action_cols[2]:
        if st.button("🧹 清空版本历史", width="stretch", key="refine_clear_version_history"):
            for key in (
                "refine_version_history",
                "refine_version_counter",
                "refine_active_version_key",
                "refine_latest_version_keys",
            ):
                st.session_state.pop(key, None)
            persist_demo_ui_state()
            st.rerun()

    preview_cols = get_refine_history_grid_columns(len(history))
    preview_width = get_refine_history_preview_width(len(history))
    for row_start in range(0, len(history), preview_cols):
        if preview_cols == 1:
            _, center_col, _ = st.columns([1, 2, 1], gap="large")
            cols = [center_col]
        else:
            cols = st.columns(preview_cols, gap="medium")
        for col_offset in range(preview_cols):
            history_index = row_start + col_offset
            if history_index >= len(history):
                continue
            entry = history[history_index]
            with cols[col_offset]:
                with st.container(border=True):
                    display_label = get_refine_version_display_label(
                        entry=entry,
                        display_map=display_map,
                        history=history,
                    )
                    parent_version_key = str(entry.get("parent_version_key", "") or "")
                    parent_text = (
                        get_refine_version_display_label(
                            version_key=parent_version_key,
                            display_map=display_map,
                            history=history,
                        )
                        if parent_version_key
                        else "原始输入"
                    )
                    st.markdown(f"**{display_label}**")
                    st.caption(f"来源：{entry.get('source_label', 'N/A')} | 父版本：{parent_text}")
                    image_bytes = entry.get("image_bytes", b"")
                    if image_bytes:
                        preview_image, preview_error = load_refine_preview_image(
                            image_bytes,
                            source_label=str(display_label or "历史版本"),
                        )
                        if preview_image is not None:
                            st.image(preview_image, width=preview_width)
                        elif preview_error:
                            st.caption(preview_error)
                    if entry.get("edit_prompt"):
                        with st.expander("查看编辑指令", expanded=False):
                            st.write(clean_text(entry["edit_prompt"]))


def render_refine_results_section(
    *,
    fallback_original_bytes: bytes = b"",
    fallback_resolution: str = "2K",
    fallback_provider: str = "gemini",
    fallback_image_model_name: str = "",
) -> None:
    refined_images = st.session_state.get("refined_images", [])
    if not refined_images:
        return

    st.divider()
    st.markdown("## 🎨 精修结果")
    final_resolution = st.session_state.get(
        "refine_result_resolution",
        fallback_resolution,
    )
    final_count = st.session_state.get("refine_count", len(refined_images))
    refine_provider_used = st.session_state.get(
        "refine_provider_used",
        fallback_provider,
    )
    refine_image_model_used = st.session_state.get(
        "refine_image_model_used",
        fallback_image_model_name,
    )
    failed_refine_results = st.session_state.get("refine_failed_results", [])
    latest_version_keys = list(st.session_state.get("refine_latest_version_keys", []) or [])
    history = get_refine_version_history()
    display_map = build_refine_version_display_map(history)
    st.caption(
        f"生成时间：{st.session_state.get('refine_timestamp', 'N/A')} | "
        f"分辨率：{final_resolution} | 张数：{final_count} | "
        f"Provider：{refine_provider_used} | 模型：{refine_image_model_used}"
    )
    if latest_version_keys:
        latest_labels = []
        for version_key in latest_version_keys:
            entry = find_refine_version_entry(version_key)
            if entry is not None:
                latest_labels.append(
                    get_refine_version_display_label(
                        entry=entry,
                        display_map=display_map,
                        history=history,
                    )
                )
        if latest_labels:
            st.info(
                f"当前结果已写入版本链：{' / '.join(latest_labels)}。你可以直接继续精修某个版本，或在下方版本历史里切换分支。"
            )
    if failed_refine_results:
        st.warning(f"有 {len(failed_refine_results)} 张精修失败。")
        with st.expander("查看失败详情", expanded=False):
            for item in failed_refine_results:
                st.write(f"结果 {item['index']}: {item['message']}")

    st.markdown("### 精修前")
    original_preview_bytes = st.session_state.get(
        "refine_original_image_bytes",
        fallback_original_bytes,
    )
    if original_preview_bytes:
        original_preview_image, original_preview_error = load_refine_preview_image(
            original_preview_bytes,
            source_label="精修前图像",
        )
        if original_preview_image is not None:
            _, center_col, _ = st.columns([1, 2, 1], gap="large")
            with center_col:
                st.image(original_preview_image, width=REFINE_ORIGINAL_PREVIEW_WIDTH)
        elif original_preview_error:
            st.warning(original_preview_error)

    st.markdown(f"### 精修后（{final_resolution}）")

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

    preview_cols = get_refine_result_grid_columns(len(refined_images))
    preview_width = get_refine_result_preview_width(len(refined_images))
    for row_start in range(0, len(refined_images), preview_cols):
        if preview_cols == 1:
            _, center_col, _ = st.columns([1, 2, 1], gap="large")
            cols = [center_col]
        else:
            cols = st.columns(preview_cols, gap="medium")
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
                refined_image, refined_preview_error = load_refine_preview_image(
                    img_bytes,
                    source_label=f"精修结果 {idx}",
                )
                if refined_image is not None:
                    st.image(refined_image, width=preview_width)
                elif refined_preview_error:
                    st.warning(refined_preview_error)
                version_key = latest_version_keys[item_pos] if item_pos < len(latest_version_keys) else ""
                if version_key:
                    st.caption(
                        f"版本：{get_refine_version_display_label(version_key=version_key, display_map=display_map, history=history)}"
                    )
                    if st.button(
                        "↺ 继续精修此版本",
                        key=f"refine_continue_{version_key}_{item_pos}",
                        width="stretch",
                    ):
                        if activate_refine_version(version_key):
                            persist_demo_ui_state()
                            st.rerun()

                file_name = f"refined_{final_resolution}_{idx}.png"
                st.download_button(
                    label=f"📥 下载结果 {idx}",
                    data=img_bytes,
                    file_name=file_name,
                    mime="image/png",
                    key=f"download_refined_{idx}_{final_resolution}_{item_pos}",
                    width="stretch"
                )

    zip_buffer.seek(0)
    st.download_button(
        label="📥 一键下载全部结果（ZIP）",
        data=zip_buffer.getvalue(),
        file_name=zip_name,
        mime="application/zip",
        width="stretch",
        key="download_refined_zip"
    )


def _build_generation_effective_settings(
    advanced_settings: dict,
    *,
    task_name: str,
) -> dict:
    effective_settings = dict(advanced_settings)
    if not get_task_ui_config(task_name)["uses_render_controls"]:
        effective_settings["image_resolution"] = ""
        effective_settings["image_generation_options"] = {}
    return effective_settings


def _queue_generation_widget_state_updates(updates: dict) -> None:
    pending_updates = st.session_state.get("_pending_generation_widget_updates", {})
    if not isinstance(pending_updates, dict):
        pending_updates = {}
    pending_updates.update({str(key): value for key, value in dict(updates or {}).items()})
    st.session_state["_pending_generation_widget_updates"] = pending_updates


def _apply_pending_generation_widget_state_updates() -> None:
    pending_updates = st.session_state.pop("_pending_generation_widget_updates", None)
    if not isinstance(pending_updates, dict):
        return
    for key, value in pending_updates.items():
        st.session_state[str(key)] = value


def _queue_refine_widget_state_updates(updates: dict) -> None:
    pending_updates = st.session_state.get("_pending_refine_widget_updates", {})
    if not isinstance(pending_updates, dict):
        pending_updates = {}
    pending_updates.update({str(key): value for key, value in dict(updates or {}).items()})
    st.session_state["_pending_refine_widget_updates"] = pending_updates


def _apply_pending_refine_widget_state_updates() -> None:
    pending_updates = st.session_state.pop("_pending_refine_widget_updates", None)
    if not isinstance(pending_updates, dict):
        return
    for key, value in pending_updates.items():
        st.session_state[str(key)] = value


def _normalize_generation_example_selector_value(value: Any) -> str:
    normalized = clean_text(str(value or EXAMPLE_SELECTOR_NONE_OPTION)).strip()
    return normalized or EXAMPLE_SELECTOR_NONE_OPTION


def _get_generation_example_selector_previous_key(selector_key: str) -> str:
    return f"{selector_key}__previous"


def _prime_generation_example_selector_state(selector_key: str) -> None:
    previous_key = _get_generation_example_selector_previous_key(selector_key)
    if previous_key in st.session_state:
        return
    st.session_state[previous_key] = _normalize_generation_example_selector_value(
        st.session_state.get(selector_key, EXAMPLE_SELECTOR_NONE_OPTION)
    )


def _apply_generation_example_selection(
    *,
    selector_key: str,
    selected_value: str,
    editor_key: str,
    example_name: str,
    example_value: str,
) -> bool:
    normalized_selected = _normalize_generation_example_selector_value(selected_value)
    normalized_example_name = _normalize_generation_example_selector_value(example_name)
    previous_key = _get_generation_example_selector_previous_key(selector_key)
    previous_value = _normalize_generation_example_selector_value(
        st.session_state.get(previous_key, EXAMPLE_SELECTOR_NONE_OPTION)
    )
    st.session_state[previous_key] = normalized_selected

    # 示例加载必须是一次性动作，避免 rerun 持续覆盖用户刚编辑的大段文本。
    if normalized_selected != normalized_example_name or previous_value == normalized_example_name:
        return False

    st.session_state[editor_key] = example_value
    _queue_generation_widget_state_updates(
        {
            selector_key: EXAMPLE_SELECTOR_NONE_OPTION,
            previous_key: EXAMPLE_SELECTOR_NONE_OPTION,
        }
    )
    return True


def _infer_generation_cost_label(
    retrieval_setting: str,
    max_critic_rounds: int,
    num_candidates: int,
) -> str:
    if retrieval_setting == "none" and int(max_critic_rounds) == 0 and int(num_candidates) <= 2:
        return "低成本"
    if retrieval_setting == "auto-full" or int(max_critic_rounds) >= 3 or int(num_candidates) >= 8:
        return "高成本"
    return "中等成本"


def build_generation_preflight_report(
    *,
    task_name: str,
    input_content: str,
    visual_intent: str,
    content_for_generation: str,
    allow_raw_plot_input: bool,
    num_candidates: int,
    effective_settings: dict,
    retrieval_ref_path: Path,
    resolved_profile_path: Path | None,
    generation_is_running: bool,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    if generation_is_running:
        errors.append("当前已有后台生成任务运行中，请先等待完成或停止当前任务。")
    if task_name == "plot" and not allow_raw_plot_input and content_for_generation == input_content:
        errors.append("plot 输入尚未通过结构化解析；请修正格式，或勾选“按原始文本继续”。")
    if bool(effective_settings.get("connection_pending_save")):
        errors.append("当前选择的是未保存的自定义连接草稿；请先点击“保存连接”，再启动正式生成。")
    if str(effective_settings.get("extra_headers_error", "") or "").strip():
        errors.append(str(effective_settings["extra_headers_error"]).strip())

    retrieval_setting = effective_settings["retrieval_setting"]
    if not str(effective_settings.get("api_key", "") or "").strip():
        warnings.append("当前没有可用的 API Key，任务可能无法正常发起。")
    if retrieval_setting in {"auto", "auto-full", "random"} and not retrieval_ref_path.exists():
        warnings.append("当前运行环境未找到参考样例库，本次会自动回退到“不使用参考”。")
    if retrieval_setting == "curated" and resolved_profile_path is None:
        warnings.append("当前固定参考集不存在，本次会自动回退到“不使用参考”。")

    notes.append(
        "当前参数："
        f"流水线={effective_settings['exp_mode']} | "
        f"检索={retrieval_setting} | 评审轮次={int(effective_settings['max_critic_rounds'])} | "
        f"候选数={int(num_candidates)}"
    )
    notes.append(
        f"成本预估：{_infer_generation_cost_label(retrieval_setting, int(effective_settings['max_critic_rounds']), int(num_candidates))}"
    )
    return {
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
    }


def render_preflight_summary(report: dict) -> None:
    errors = report.get("errors", [])
    warnings = report.get("warnings", [])
    notes = report.get("notes", [])
    if errors:
        for message in errors:
            st.error(message, icon="⛔")
    elif warnings:
        for message in warnings:
            st.warning(message, icon="⚠️")
    else:
        st.success("✅ 已通过检查，可以直接启动任务。", icon="🚀")
    if notes:
        st.caption(" · ".join(notes))


def render_sidebar_section_styles() -> None:
    """渲染侧边栏统一样式。"""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem; }
        [data-testid="stSidebar"] hr { margin: 0.25rem 0; }
        [data-testid="stSidebar"] .stButton > button {
            min-height: 2.5rem;
            padding: 0.35rem 0.6rem;
            border-radius: 14px;
            font-weight: 600;
        }
        .sb-section {
            font-size: 0.7rem; font-weight: 700;
            color: #c47d15;
            text-transform: uppercase; letter-spacing: 0.08em;
            margin: 14px 0 4px 0; padding: 0 0 4px 0;
            border-bottom: 2px solid #f5deb3;
        }
        .sb-section:first-of-type { margin-top: 6px; }
        .pb-inline-status {
            margin: 0.15rem 0 0.45rem 0;
            padding: 0.55rem 0.72rem;
            border-radius: 12px;
            border: 1px solid #e6dccf;
            background: #fffaf2;
            color: #6a5532;
            font-size: 0.84rem;
            line-height: 1.35;
        }
        .pb-inline-status strong { font-weight: 700; }
        .pb-inline-status.is-running {
            background: #fff6e7;
            border-color: #f0d6a6;
            color: #8b5d12;
        }
        .pb-inline-status.is-success {
            background: #edf8f0;
            border-color: #c7e2cf;
            color: #1f6f43;
        }
        .pb-inline-status.is-error {
            background: #fff0f0;
            border-color: #ecc8c8;
            color: #a24040;
        }
        .pb-probe-panel {
            margin-top: 0.45rem;
            padding: 0.75rem 0.85rem 0.4rem 0.85rem;
            border: 1px solid #e7dfd3;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.72);
        }
        .pb-probe-title {
            font-size: 0.94rem;
            font-weight: 700;
            color: #2f3443;
            margin-bottom: 0.35rem;
        }
        .pb-probe-row {
            padding: 0.5rem 0;
            border-top: 1px solid rgba(47, 52, 67, 0.08);
        }
        .pb-probe-row:first-of-type {
            border-top: none;
            padding-top: 0.2rem;
        }
        .pb-probe-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.6rem;
        }
        .pb-probe-label {
            font-size: 0.9rem;
            font-weight: 700;
            color: #2f3443;
        }
        .pb-probe-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.14rem 0.5rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            line-height: 1.1;
            white-space: nowrap;
        }
        .pb-probe-badge-success {
            color: #1f6f43;
            background: #e9f7ee;
        }
        .pb-probe-badge-warning {
            color: #8b5d12;
            background: #fff1d2;
        }
        .pb-probe-badge-failed {
            color: #a24040;
            background: #ffe5e5;
        }
        .pb-probe-badge-skipped {
            color: #707787;
            background: #f1f3f6;
        }
        .pb-probe-summary {
            margin-top: 0.24rem;
            font-size: 0.84rem;
            color: #556072;
            line-height: 1.4;
        }
        .pb-probe-meta {
            margin-top: 0.18rem;
            font-size: 0.76rem;
            color: #8a91a0;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_generation_sidebar_controls() -> dict:
    with st.sidebar:
        render_sidebar_section_styles()
        st.markdown('<p class="sb-section">基本设置</p>', unsafe_allow_html=True)
        ensure_session_choice_state(
            "tab1_task_name",
            ["diagram", "plot"],
            normalize_task_name(st.session_state.get("tab1_task_name", "diagram")),
        )
        task_name = st.selectbox(
            "生成任务",
            ["diagram", "plot"],
            key="tab1_task_name",
            format_func=lambda x: TASK_OPTION_LABELS[x],
            help="选择当前要生成的是论文方法图解，还是统计图表。",
        )
        task_config = get_task_ui_config(task_name)

        ensure_session_int_state(
            "tab1_num_candidates",
            5,
            min_value=1,
            max_value=20,
        )
        num_candidates = int(
            st.number_input(
                "候选方案数量",
                min_value=1,
                max_value=20,
                key="tab1_num_candidates",
                help="要并行生成多少个候选方案。",
            )
        )

        st.markdown('<p class="sb-section">流水线配置</p>', unsafe_allow_html=True)

        if not str(st.session_state.get("tab1_dataset_name", "") or "").strip():
            st.session_state["tab1_dataset_name"] = DEFAULT_DATASET_NAME
        dataset_name = st.text_input(
            "数据集名称",
            key="tab1_dataset_name",
            help="用于定位参考样例、GT 资源和数据集内的相对路径。",
        ).strip() or DEFAULT_DATASET_NAME

        exp_mode = st.selectbox(
            "生成流程",
            ["demo_planner_critic", "demo_full"],
            key="tab1_exp_mode",
            format_func=lambda x: PIPELINE_OPTION_LABELS[x],
            help="选择本次生成采用的多 Agent 流程。",
        )
        st.caption(GENERATION_MODE_INFO[exp_mode])

        retrieval_setting_key = "tab1_retrieval_setting"
        retrieval_options = ["auto", "auto-full", "curated", "random", "none"]
        ensure_session_choice_state(
            retrieval_setting_key,
            retrieval_options,
            normalize_retrieval_setting(st.session_state.get(retrieval_setting_key, "auto")),
        )
        retrieval_setting = st.selectbox(
            "参考样例策略",
            retrieval_options,
            key=retrieval_setting_key,
            help="决定系统如何选择用于 few-shot 提示的参考样例。",
            format_func=get_retrieval_setting_label,
        )

        retrieval_target_label = "可视化意图" if task_name == "plot" else "图注"
        retrieval_ref_path = get_reference_file_path(
            dataset_name,
            task_name,
            work_dir=REPO_ROOT,
        )
        retrieval_notice = RETRIEVAL_NOTICE_TEXT[retrieval_setting]
        if retrieval_setting == "auto":
            retrieval_notice = (
                f"默认推荐。只把你的{retrieval_target_label}发给模型做参考匹配，成本低、速度快，适合大多数试跑。"
            )
        st.caption(retrieval_notice)

        curated_profile_key = "tab1_curated_profile"
        curated_profile_input_key = "tab1_curated_profile_input"
        current_curated_profile = initialize_curated_profile_state(
            profile_key=curated_profile_key,
            input_key=curated_profile_input_key,
        )
        curated_profile = current_curated_profile
        resolved_profile_path = None
        if retrieval_setting == "curated":
            curated_profile_input = st.text_input(
                "固定参考集名称",
                key=curated_profile_input_key,
                help=(
                    "填写要使用的固定参考集名称。系统优先读取 "
                    "`manual_profiles/<name>.json`；当名称为 `default` 时，也兼容旧的 "
                    "`agent_selected_12.json`。"
                ),
            )
            curated_profile = resolve_curated_profile_input(
                curated_profile_input,
                profile_key=curated_profile_key,
            )
            if curated_profile != str(curated_profile_input or "").strip():
                st.caption(f"运行时会使用规范化名称：`{curated_profile}`")
            resolved_profile_path = find_curated_profile_path(
                dataset_name,
                task_name,
                profile_name=curated_profile,
                work_dir=REPO_ROOT,
            )
            if resolved_profile_path is not None:
                source_note = ""
                if resolved_profile_path.name == "agent_selected_12.json":
                    source_note = "（兼容旧版 agent_selected_12.json）"
                st.caption(
                    f"当前固定参考集文件：`{format_repo_relative_path(resolved_profile_path)}`{source_note}"
                )
            else:
                expected_profile_path = get_curated_profile_path(
                    dataset_name,
                    task_name,
                    profile_name=curated_profile,
                    work_dir=REPO_ROOT,
                )
                if curated_profile == DEFAULT_CURATED_PROFILE:
                    legacy_profile_path = get_legacy_manual_reference_path(
                        dataset_name,
                        task_name,
                        work_dir=REPO_ROOT,
                    )
                    st.warning(
                        "当前未发现默认固定参考集。系统会优先查找 "
                        f"`{format_repo_relative_path(expected_profile_path)}`，并兼容旧路径 "
                        f"`{format_repo_relative_path(legacy_profile_path)}`。"
                    )
                else:
                    st.warning(
                        f"当前未找到固定参考集：`{format_repo_relative_path(expected_profile_path)}`。运行时会自动回退到“不使用参考”。"
                    )

        st.markdown('<p class="sb-section">并发 & 渲染</p>', unsafe_allow_html=True)
        ensure_session_choice_state(
            "tab1_concurrency_mode",
            ["auto", "manual"],
            str(st.session_state.get("tab1_concurrency_mode", "auto") or "auto"),
        )
        concurrency_mode = st.selectbox(
            "并发策略",
            ["auto", "manual"],
            key="tab1_concurrency_mode",
            help="auto：自动并发（默认）| manual：使用固定并发上限",
        )
        ensure_session_int_state(
            "tab1_max_concurrent",
            20,
            min_value=1,
            max_value=100,
        )
        max_concurrent = st.number_input(
            "并发上限",
            min_value=1,
            max_value=100,
            step=1,
            key="tab1_max_concurrent",
            help="候选任务并发上限，默认 20",
        )
        effective_concurrency_preview = compute_effective_concurrency(
            concurrency_mode=concurrency_mode,
            max_concurrent=int(max_concurrent),
            total_candidates=num_candidates,
            task_name=task_name,
            retrieval_setting=retrieval_setting,
            exp_mode=exp_mode,
            provider=st.session_state.get("tab1_provider", "gemini"),
        )
        estimated_batches_preview = math.ceil(
            max(1, num_candidates) / max(1, effective_concurrency_preview)
        )
        st.caption(f"预计并发：{effective_concurrency_preview} | 批次：{estimated_batches_preview}")

        if task_config["uses_render_controls"]:
            ensure_session_choice_state(
                "tab1_aspect_ratio",
                COMMON_ASPECT_RATIOS,
                str(st.session_state.get("tab1_aspect_ratio", "16:9") or "16:9"),
            )
            aspect_ratio = st.selectbox(
                "宽高比",
                COMMON_ASPECT_RATIOS,
                key="tab1_aspect_ratio",
                help="生成图表的宽高比",
            )
            selected_connection = find_connection_by_id(
                str(st.session_state.get("tab1_provider", DEFAULT_PROVIDER) or DEFAULT_PROVIDER)
            )
            provider_for_resolution = (
                selected_connection.provider_type if selected_connection is not None else DEFAULT_PROVIDER
            )
            image_model_for_resolution = (
                selected_connection.image_model if selected_connection is not None else ""
            )
            resolution_capabilities = get_image_model_capabilities(
                provider_for_resolution,
                image_model_for_resolution,
            )
            resolution_options = list(resolution_capabilities.legacy_resolution_options)
            default_resolution = st.session_state.get("tab1_image_resolution", "2K")
            if default_resolution not in resolution_options:
                default_resolution = "2K" if "2K" in resolution_options else resolution_options[0]
            ensure_session_choice_state(
                "tab1_image_resolution",
                resolution_options,
                str(default_resolution),
            )
            image_resolution = st.selectbox(
                "图像分辨率",
                resolution_options,
                key="tab1_image_resolution",
                help=f"生成图像的分辨率（当前 Provider 支持：{', '.join(resolution_options)}）",
            )
        else:
            aspect_ratio = "16:9"
            image_resolution = "2K"
            st.info("当前任务使用 Matplotlib 代码渲染，宽高比和图像分辨率控件暂不生效。")

        ensure_session_int_state(
            "tab1_max_critic_rounds",
            3,
            min_value=0,
            max_value=5,
        )
        max_critic_rounds = st.number_input(
            "最大评审轮次",
            min_value=0,
            max_value=5,
            key="tab1_max_critic_rounds",
            help="评审优化迭代的最大轮次；设为 0 可做低成本试跑。",
        )
        st.markdown('<p class="sb-section">Provider & 模型</p>', unsafe_allow_html=True)
        connection_options = get_connection_options()
        connection_ids = [item[0] for item in connection_options]
        ensure_session_choice_state(
            "tab1_provider",
            connection_ids,
            str(st.session_state.get("tab1_provider", DEFAULT_PROVIDER) or DEFAULT_PROVIDER),
        )
        provider = st.selectbox(
            "生成连接",
            connection_ids,
            key="tab1_provider",
            format_func=lambda value: dict(connection_options).get(value, value),
            help="可选择内置连接，或创建自定义 OpenAI 兼容连接。",
        )

        provider_defaults = ensure_connection_editor_state("tab1", provider)
        state_keys = _build_connection_state_keys("tab1")
        runtime_connection_id = get_selected_connection_runtime_id("tab1", provider)
        connection_pending_save = provider == CUSTOM_CONNECTION_CREATE_OPTION
        is_builtin_connection = bool(provider_defaults.get("builtin", False))
        sync_connection_runtime_input_state(
            prefix="tab1",
            selected_connection_id=provider,
            provider_defaults=provider_defaults,
        )
        if state_keys["persist_secret"] not in st.session_state:
            st.session_state[state_keys["persist_secret"]] = True

        if not is_builtin_connection:
            st.text_input(
                "连接显示名",
                key=state_keys["display_name"],
                disabled=False,
                help="自定义连接可编辑显示名，方便在生成页和精修页复用。",
            )
            st.text_input(
                "连接 ID",
                key=state_keys["connection_id"],
                disabled=False,
                help="仅自定义连接可编辑，建议使用小写英文和连字符。",
            )
        if is_builtin_connection:
            with st.expander("高级连接参数", expanded=False):
                base_url = st.text_input(
                    "Base URL",
                    key=state_keys["base_url"],
                    help="OpenAI 兼容服务请填写完整 base URL；内置连接也可临时覆盖。",
                )
                extra_headers_json = st.text_area(
                    "额外请求头（JSON）",
                    key=state_keys["extra_headers_json"],
                    height=100,
                    help="用于某些自建网关的附加请求头，例如组织信息或路由标记。",
                )
        else:
            base_url = st.text_input(
                "Base URL",
                key=state_keys["base_url"],
                help="OpenAI 兼容服务请填写完整 base URL；内置连接也可临时覆盖。",
            )
        if not is_builtin_connection:
            st.text_input(
                "API Key 环境变量",
                key=state_keys["api_key_env_var"],
                disabled=False,
                help="可选。若配置了环境变量名，将优先读取该环境变量中的密钥。",
            )
            extra_headers_json = st.text_area(
                "额外请求头（JSON）",
                key=state_keys["extra_headers_json"],
                height=100,
                help="用于某些自建网关的附加请求头，例如组织信息或路由标记。",
            )
        persist_secret = st.checkbox(
            "将 API Key 保存到本地",
            key=state_keys["persist_secret"],
            help="关闭后，本次输入的密钥仅在当前会话生效。",
        )
        allow_local_api_key_persist = is_builtin_connection or not connection_pending_save

        api_key = render_provider_api_key_controls(
            provider=runtime_connection_id,
            provider_defaults={
                **provider_defaults,
                "api_key_label": provider_defaults.get("api_key_label", "API Key"),
                "api_key_help": provider_defaults.get("api_key_help", "当前连接使用的 API 密钥"),
            },
            session_key="tab1_api_key",
            clear_request_key="tab1_api_key_clear_requested",
            clear_button_key="tab1_clear_provider_api_key",
            persist_secret=bool(persist_secret),
            allow_local_persist=allow_local_api_key_persist,
        )
        if provider_defaults.get("provider_type") == "gemini":
            model_name = render_preset_or_custom_model_input(
                "文本模型",
                GEMINI_TEXT_MODELS,
                value_key="tab1_model_name",
                selector_key="tab1_model_name_selector",
                custom_value_key="tab1_model_name_custom",
                default_value=provider_defaults["model_name"],
                select_help="用于推理/规划/评审的模型名称。可选择预设模型，或选“自定义”后手动输入。",
                custom_help="请输入用于推理/规划/评审的自定义文本模型名称。",
            )
        else:
            model_name = st.text_input(
                "文本模型",
                key="tab1_model_name",
                help="用于推理/规划/评审的模型名称",
            )

        if task_config["uses_image_model"]:
            if provider_defaults.get("provider_type") == "gemini":
                image_model_name = render_preset_or_custom_model_input(
                    "图像模型",
                    GEMINI_IMAGE_MODELS,
                    value_key="tab1_image_model_name",
                    selector_key="tab1_image_model_name_selector",
                    custom_value_key="tab1_image_model_name_custom",
                    default_value=provider_defaults["image_model_name"],
                    select_help="用于图像生成的模型名称。可选择预设模型，或选“自定义”后手动输入。",
                    custom_help="请输入用于图像生成的自定义模型名称。",
                )
            else:
                image_model_name = st.text_input(
                    "图像模型",
                    key="tab1_image_model_name",
                    help="用于图像生成的模型名称",
                )
            if provider_defaults.get("provider_type") in {"openai", "openai_compatible"}:
                image_generation_options = render_image_generation_option_controls(
                    provider_type=str(provider_defaults.get("provider_type", "openai") or "openai"),
                    image_model_name=image_model_name,
                    aspect_ratio=aspect_ratio,
                    image_resolution=image_resolution,
                )
            else:
                image_generation_options = normalize_image_generation_options(
                    provider_type=str(provider_defaults.get("provider_type", DEFAULT_PROVIDER) or DEFAULT_PROVIDER),
                    model_name=image_model_name,
                    aspect_ratio=aspect_ratio,
                    image_resolution=image_resolution,
                ).to_dict()
        else:
            image_model_name = ""
            image_generation_options = {}
            st.caption("当前任务不会调用图像生成模型，最终图像由文本模型生成的 Matplotlib 代码渲染。")

        supports_image = bool(st.session_state.get(state_keys["supports_image"], provider_defaults.get("supports_image", True)))
        if not is_builtin_connection:
            supports_image = st.checkbox(
                "启用图像能力",
                key=state_keys["supports_image"],
                disabled=False,
                help="仅对自定义连接生效；开启后才会执行图像探针。",
            )
            st.checkbox(
                "启用文本能力",
                key=state_keys["supports_text"],
                disabled=False,
                help="仅对自定义连接生效。",
            )
            st.selectbox(
                "模型发现模式",
                ["manual", "hybrid", "fetch", "static"],
                key=state_keys["model_discovery_mode"],
                disabled=False,
                help="hybrid 表示优先尝试远端发现，失败后回退到手动模型名。",
            )
            st.text_area(
                "模型白名单（每行一个）",
                key=state_keys["model_allowlist"],
                height=80,
                disabled=False,
                help="用于 `/models` 不可用时的手动回退列表。",
            )

        action_cols = st.columns([1, 1, 1], gap="small")
        with action_cols[0]:
            save_clicked = st.button("💾 保存", key="tab1_save_connection", width="stretch", help="保存当前连接草稿。")
        with action_cols[1]:
            refresh_clicked = st.button("↻ 刷新", key="tab1_refresh_models", width="stretch", help="重新拉取或更新可用模型列表。")
        with action_cols[2]:
            test_clicked = st.button("🧪 测试", key="tab1_test_connection", width="stretch", help="验证文本链路与图像链路是否可用。")

        if save_clicked:
            ok, message, _ = save_connection_draft(
                prefix="tab1",
                selected_connection_id=provider,
                api_key=api_key,
                model_name=model_name,
                image_model_name=image_model_name,
            )
            if ok:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

        if refresh_clicked:
            _, message = run_refresh_models_with_status(
                prefix="tab1",
                connection_id=provider,
                api_key=api_key,
                model_name=model_name,
                image_model_name=image_model_name,
            )
            emit_connection_action_feedback(message)

        if test_clicked:
            _, message = run_connection_probe_with_status(
                prefix="tab1",
                connection_id=provider,
                api_key=api_key,
                model_name=model_name,
                image_model_name=image_model_name,
            )
            emit_connection_action_feedback(message)

        if provider not in BUILTIN_CONNECTION_IDS and provider != CUSTOM_CONNECTION_CREATE_OPTION:
            delete_col, _ = st.columns([1, 2])
            with delete_col:
                delete_clicked = st.button("删除自定义连接", key="tab1_delete_custom_connection", width="stretch")
            if delete_clicked:
                delete_custom_connection(provider, base_dir=REPO_ROOT)
                st.session_state["tab1_provider"] = DEFAULT_PROVIDER
                st.rerun()

        if extra_headers_json:
            st.caption("已填写额外请求头草稿。")
        if not persist_secret:
            st.caption("当前 API Key 仅在本次会话中使用。")
        if not is_builtin_connection and not supports_image:
            st.caption("当前连接未启用图像能力，图像探针会自动跳过。")
        if connection_pending_save:
            st.caption("当前是未保存的自定义连接草稿；可先测试，但正式运行前需要先保存连接。")
        render_connection_probe_results("tab1")

    extra_headers, extra_headers_error = parse_extra_headers_json_safe(extra_headers_json)
    return {
        "task_name": task_name,
        "num_candidates": int(num_candidates),
        "dataset_name": dataset_name,
        "exp_mode": exp_mode,
        "retrieval_setting": retrieval_setting,
        "curated_profile": curated_profile,
        "resolved_profile_path": resolved_profile_path,
        "retrieval_ref_path": retrieval_ref_path,
        "concurrency_mode": concurrency_mode,
        "max_concurrent": int(max_concurrent),
        "aspect_ratio": aspect_ratio,
        "image_resolution": image_resolution,
        "image_generation_options": image_generation_options,
        "max_critic_rounds": int(max_critic_rounds),
        "provider": runtime_connection_id,
        "connection_id": runtime_connection_id,
        "provider_type": str(provider_defaults.get("provider_type", DEFAULT_PROVIDER) or DEFAULT_PROVIDER),
        "provider_display_name": str(st.session_state.get(state_keys["display_name"], provider_defaults.get("display_name", runtime_connection_id)) or runtime_connection_id),
        "connection_pending_save": connection_pending_save,
        "api_key": api_key,
        "base_url": base_url,
        "extra_headers": extra_headers,
        "extra_headers_error": extra_headers_error,
        "model_name": model_name,
        "image_model_name": image_model_name,
    }


def render_refine_sidebar_controls() -> dict:
    """渲染精修页侧边栏设置。"""
    with st.sidebar:
        render_sidebar_section_styles()
        st.markdown("## 高级设置")
        st.caption("当前处于精修工作台，生成侧高级设置已隐藏。")

        st.markdown('<p class="sb-section">输出设置</p>', unsafe_allow_html=True)
        ensure_session_choice_state("refine_resolution", ["2K", "4K"], "2K")
        refine_resolution = st.selectbox(
            "目标分辨率",
            ["2K", "4K"],
            key="refine_resolution",
            help="更高分辨率会增加耗时，但通常能得到更细致的结果。",
        )
        ensure_session_choice_state(
            "refine_aspect_ratio",
            COMMON_ASPECT_RATIOS,
            COMMON_ASPECT_RATIOS[0],
        )
        refine_aspect_ratio = st.selectbox(
            "宽高比",
            COMMON_ASPECT_RATIOS,
            key="refine_aspect_ratio",
            help="指定精修后图像的目标宽高比。",
        )
        ensure_session_int_state(
            "refine_num_images",
            3,
            min_value=1,
            max_value=12,
        )
        refine_num_images = int(
            st.number_input(
                "精修张数",
                min_value=1,
                max_value=12,
                step=1,
                key="refine_num_images",
                help="并发生成多少张不同版本，便于横向挑选。",
            )
        )
        st.caption(
            f"当前任务将输出 {refine_num_images} 张 {refine_resolution} 结果，宽高比为 {refine_aspect_ratio}。"
        )

        st.markdown('<p class="sb-section">Provider</p>', unsafe_allow_html=True)
        connection_options = get_connection_options()
        connection_ids = [item[0] for item in connection_options]
        ensure_session_choice_state(
            "refine_provider",
            connection_ids,
            str(st.session_state.get("refine_provider", DEFAULT_PROVIDER) or DEFAULT_PROVIDER),
        )
        refine_provider = st.selectbox(
            "精修连接",
            connection_ids,
            key="refine_provider",
            format_func=lambda value: dict(connection_options).get(value, value),
            help="精修链路可单独选择连接，不依赖生成页设置。",
        )
        refine_provider_defaults = ensure_connection_editor_state("refine", refine_provider)
        state_keys = _build_connection_state_keys("refine")
        runtime_connection_id = get_selected_connection_runtime_id("refine", refine_provider)
        connection_pending_save = refine_provider == CUSTOM_CONNECTION_CREATE_OPTION
        is_builtin_connection = bool(refine_provider_defaults.get("builtin", False))
        sync_connection_runtime_input_state(
            prefix="refine",
            selected_connection_id=refine_provider,
            provider_defaults=refine_provider_defaults,
        )
        if state_keys["persist_secret"] not in st.session_state:
            st.session_state[state_keys["persist_secret"]] = True

        if not is_builtin_connection:
            st.text_input(
                "连接显示名",
                key=state_keys["display_name"],
                disabled=False,
                help="自定义连接可编辑显示名。",
            )
            st.text_input(
                "连接 ID",
                key=state_keys["connection_id"],
                disabled=False,
                help="仅自定义连接可编辑。",
            )
        if is_builtin_connection:
            with st.expander("高级连接参数", expanded=False):
                refine_base_url = st.text_input(
                    "Base URL",
                    key=state_keys["base_url"],
                    help="当前精修连接的 base URL。",
                )
                refine_extra_headers_json = st.text_area(
                    "额外请求头（JSON）",
                    key=state_keys["extra_headers_json"],
                    height=90,
                    help="用于某些 OpenAI 兼容网关的附加请求头。",
                )
        else:
            refine_base_url = st.text_input(
                "Base URL",
                key=state_keys["base_url"],
                help="当前精修连接的 base URL。",
            )
        if not is_builtin_connection:
            st.text_input(
                "API Key 环境变量",
                key=state_keys["api_key_env_var"],
                disabled=False,
                help="可选。若配置，将优先读取该环境变量。",
            )
            refine_extra_headers_json = st.text_area(
                "额外请求头（JSON）",
                key=state_keys["extra_headers_json"],
                height=90,
                help="用于某些 OpenAI 兼容网关的附加请求头。",
            )
        st.checkbox(
            "将 API Key 保存到本地",
            key=state_keys["persist_secret"],
            help="关闭后，本次输入的密钥仅在当前会话中使用。",
        )
        allow_local_api_key_persist = is_builtin_connection or not connection_pending_save
        refine_api_key = render_provider_api_key_controls(
            provider=runtime_connection_id,
            provider_defaults={
                **refine_provider_defaults,
                "api_key_label": refine_provider_defaults.get("api_key_label", "API Key"),
                "api_key_help": refine_provider_defaults.get("api_key_help", "当前连接使用的 API 密钥"),
            },
            session_key="refine_api_key",
            clear_request_key="refine_api_key_clear_requested",
            clear_button_key="refine_clear_provider_api_key",
            persist_secret=bool(st.session_state.get(state_keys["persist_secret"], True)),
            allow_local_persist=allow_local_api_key_persist,
        )
        if refine_provider_defaults.get("provider_type") == "gemini":
            refine_image_model_name = render_preset_or_custom_model_input(
                "精修图像模型",
                GEMINI_IMAGE_MODELS,
                value_key="refine_image_model_name",
                selector_key="refine_image_model_name_selector",
                custom_value_key="refine_image_model_name_custom",
                default_value=refine_provider_defaults["image_model_name"],
                select_help="精修流程使用的图像模型。可选择预设模型，或选“自定义”后手动输入。",
                custom_help="请输入精修流程使用的自定义图像模型名称。",
            )
        else:
            refine_image_model_name = st.text_input(
                "精修图像模型",
                key="refine_image_model_name",
                help="精修流程使用的图像模型",
            )
        if refine_provider_defaults.get("provider_type") in {"openai", "openai_compatible"}:
            refine_image_generation_options = render_image_generation_option_controls(
                provider_type=str(refine_provider_defaults.get("provider_type", "openai") or "openai"),
                image_model_name=refine_image_model_name,
                aspect_ratio=refine_aspect_ratio,
                image_resolution=refine_resolution,
                widget_prefix="refine",
            )
        else:
            refine_image_generation_options = normalize_image_generation_options(
                provider_type=str(refine_provider_defaults.get("provider_type", DEFAULT_PROVIDER) or DEFAULT_PROVIDER),
                model_name=refine_image_model_name,
                aspect_ratio=refine_aspect_ratio,
                image_resolution=refine_resolution,
            ).to_dict()
        if not is_builtin_connection:
            st.checkbox(
                "启用图像能力",
                key=state_keys["supports_image"],
                disabled=False,
                help="仅对自定义连接生效；关闭时图像探针会自动跳过。",
            )
            st.selectbox(
                "模型发现模式",
                ["manual", "hybrid", "fetch", "static"],
                key=state_keys["model_discovery_mode"],
                disabled=False,
                help="hybrid 表示优先尝试远端发现，失败后回退到手动模型名。",
            )
            st.text_area(
                "模型白名单（每行一个）",
                key=state_keys["model_allowlist"],
                height=70,
                disabled=False,
                help="用于 `/models` 不可用时的手动回退列表。",
            )
        action_cols = st.columns([1, 1, 1], gap="small")
        with action_cols[0]:
            save_clicked = st.button("💾 保存", key="refine_save_connection", width="stretch", help="保存当前连接草稿。")
        with action_cols[1]:
            refresh_clicked = st.button("↻ 刷新", key="refine_refresh_models", width="stretch", help="重新拉取或更新可用模型列表。")
        with action_cols[2]:
            test_clicked = st.button("🧪 测试", key="refine_test_connection", width="stretch", help="验证当前连接的图像链路与兼容性。")

        if save_clicked:
            ok, message, _ = save_connection_draft(
                prefix="refine",
                selected_connection_id=refine_provider,
                api_key=refine_api_key,
                model_name="",
                image_model_name=refine_image_model_name,
            )
            if ok:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

        if refresh_clicked:
            _, message = run_refresh_models_with_status(
                prefix="refine",
                connection_id=refine_provider,
                api_key=refine_api_key,
                model_name="",
                image_model_name=refine_image_model_name,
            )
            emit_connection_action_feedback(message)

        if test_clicked:
            _, message = run_connection_probe_with_status(
                prefix="refine",
                connection_id=refine_provider,
                api_key=refine_api_key,
                model_name="",
                image_model_name=refine_image_model_name,
            )
            emit_connection_action_feedback(message)

        if refine_provider not in BUILTIN_CONNECTION_IDS and refine_provider != CUSTOM_CONNECTION_CREATE_OPTION:
            delete_clicked = st.button("删除自定义连接", key="refine_delete_custom_connection", width="stretch")
            if delete_clicked:
                delete_custom_connection(refine_provider, base_dir=REPO_ROOT)
                st.session_state["refine_provider"] = DEFAULT_PROVIDER
                st.rerun()
        if refine_extra_headers_json:
            st.caption("已填写额外请求头草稿。")
        if not bool(st.session_state.get(state_keys["persist_secret"], True)):
            st.caption("当前 API Key 仅在本次会话中使用。")
        if connection_pending_save:
            st.caption("当前是未保存的自定义连接草稿；可先测试，但正式运行前需要先保存连接。")
        render_connection_probe_results("refine")

        st.markdown('<p class="sb-section">当前会话</p>', unsafe_allow_html=True)
        staged_refine_bytes = bytes(st.session_state.get("refine_staged_image_bytes", b"") or b"")
        cached_uploaded_bytes = bytes(st.session_state.get("refine_uploaded_image_bytes", b"") or b"")
        active_refine_job_id = st.session_state.get("active_refine_job_id")
        st.caption(f"候选来源：{'已载入' if staged_refine_bytes else '未载入'}")
        st.caption(f"上传缓存：{'已缓存' if cached_uploaded_bytes else '未缓存'}")
        st.caption(f"后台任务：{'运行中' if active_refine_job_id else '空闲'}")

    extra_headers, extra_headers_error = parse_extra_headers_json_safe(refine_extra_headers_json)
    return {
        "resolution": refine_resolution,
        "aspect_ratio": refine_aspect_ratio,
        "num_images": refine_num_images,
        "provider": runtime_connection_id,
        "connection_id": runtime_connection_id,
        "provider_type": str(refine_provider_defaults.get("provider_type", DEFAULT_PROVIDER) or DEFAULT_PROVIDER),
        "provider_display_name": str(st.session_state.get(state_keys["display_name"], refine_provider_defaults.get("display_name", runtime_connection_id)) or runtime_connection_id),
        "connection_pending_save": connection_pending_save,
        "api_key": refine_api_key,
        "base_url": refine_base_url,
        "extra_headers": extra_headers,
        "extra_headers_error": extra_headers_error,
        "image_model_name": refine_image_model_name,
        "image_generation_options": refine_image_generation_options,
    }


def render_generation_results_panel(default_task_name: str) -> None:
    if "results" not in st.session_state or not st.session_state["results"]:
        return

    results = st.session_state["results"]
    current_task_name = normalize_task_name(
        st.session_state.get(
            "task_name",
            results[0].get("task_name", default_task_name) if results else default_task_name,
        )
    )
    current_dataset_name = str(
        st.session_state.get(
            "dataset_name",
            results[0].get("dataset_name", DEFAULT_DATASET_NAME) if results else DEFAULT_DATASET_NAME,
        )
    ).strip() or DEFAULT_DATASET_NAME
    current_task_config = get_task_ui_config(current_task_name)
    current_mode = st.session_state.get("exp_mode", "demo_planner_critic")
    timestamp = st.session_state.get("timestamp", "N/A")
    mode_used = st.session_state.get("concurrency_mode", "auto")
    max_used = st.session_state.get("max_concurrent", 0)
    requested_candidates = int(st.session_state.get("requested_candidates", len(results)) or len(results))
    result_source_label = st.session_state.get("result_source_label", "当前会话")
    effective_used = st.session_state.get(
        "effective_concurrent",
        compute_effective_concurrency(mode_used, int(max_used), len(results)),
    )

    st.divider()
    st.markdown(f"## 🎨 已生成的{current_task_config['display_name_cn']}候选方案")
    success_count = sum(
        1 for item in results if not (isinstance(item, dict) and item.get("status") == "failed")
    )
    failed_count = len(results) - success_count
    st.caption(
        f"生成时间：{timestamp} | 任务：{current_task_config['display_name_cn']} | 来源：{result_source_label} | "
        f"数据集：{current_dataset_name} | 流水线：{GENERATION_MODE_INFO.get(current_mode, current_mode)} | "
        f"并发：{mode_used} (max={max_used}, effective={effective_used}) | "
        f"保留结果：{len(results)}/{requested_candidates} | 成功/失败：{success_count}/{failed_count}"
    )

    favorite_count = 0
    discarded_count = 0
    final_candidate_token = get_generation_final_candidate_token()
    for fallback_index, result_item in enumerate(results):
        decision = get_generation_candidate_decision(get_candidate_id(result_item, fallback_index))
        if decision == "final":
            favorite_count += 1
        elif decision == "favorite":
            favorite_count += 1
        elif decision == "discarded":
            discarded_count += 1

    with st.container(border=True):
        st.markdown("### 🧭 候选决策工作台")
        metric_cols = st.columns(4)
        metric_cols[0].metric("全部候选", len(results))
        metric_cols[1].metric("已收藏", favorite_count)
        metric_cols[2].metric("已淘汰", discarded_count)
        metric_cols[3].metric("最终候选", 1 if final_candidate_token else 0)

        filter_col, export_col = st.columns(2)
        with filter_col:
            display_scope = st.selectbox(
                "当前展示范围",
                GENERATION_DECISION_FILTER_OPTIONS,
                key=f"generation_display_scope_{current_task_name}",
            )
        with export_col:
            export_scope = st.selectbox(
                "批量导出范围",
                GENERATION_EXPORT_SCOPE_OPTIONS,
                key=f"generation_export_scope_{current_task_name}",
            )

        final_candidate_result = find_generation_result_by_candidate_id(
            results,
            final_candidate_token,
        )
        if final_candidate_result is not None:
            final_candidate_label = format_candidate_display_label(final_candidate_result)
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                st.success(f"当前最终候选：{final_candidate_label}")
            with action_col2:
                if st.button(
                    "✨ 将最终候选送去精修",
                    key=f"stage_final_candidate_for_refine_{current_task_name}",
                    width="stretch",
                ):
                    if stage_candidate_for_refine(
                        final_candidate_result,
                        candidate_id=final_candidate_token,
                        exp_mode=current_mode,
                        task_name=current_task_name,
                    ):
                        persist_demo_ui_state()
                        st.success(f"已将 {final_candidate_label} 载入精修工作台。")
                        st.rerun()
                    else:
                        st.warning(f"{final_candidate_label} 当前缺少可用于精修的最终图像。")

    visible_results = filter_generation_results_by_scope(results, display_scope)

    json_file_path = Path(st.session_state["json_file"]) if st.session_state.get("json_file") else None
    bundle_file_path = Path(st.session_state["bundle_file"]) if st.session_state.get("bundle_file") else None
    if (json_file_path and json_file_path.exists()) or (bundle_file_path and bundle_file_path.exists()):
        button_cols = [3, 1]
        if json_file_path and json_file_path.exists() and bundle_file_path and bundle_file_path.exists():
            button_cols.append(1)
        columns = st.columns(button_cols)
        with columns[0]:
            file_label = ""
            if json_file_path and json_file_path.exists():
                file_label = f"📄 结果已保存至：`{format_repo_relative_path(json_file_path)}`"
            if bundle_file_path and bundle_file_path.exists():
                if file_label:
                    file_label += "\n\n"
                file_label += f"🧾 Bundle：`{format_repo_relative_path(bundle_file_path)}`"
            st.info(file_label)
        if json_file_path and json_file_path.exists():
            with columns[1]:
                st.download_button(
                    label="📥 下载结果 JSON",
                    data=json_file_path.read_text(encoding="utf-8"),
                    file_name=json_file_path.name,
                    mime="application/json",
                    width="stretch",
                )
        if bundle_file_path and bundle_file_path.exists():
            target_col = columns[2] if len(columns) > 2 else columns[1]
            with target_col:
                st.download_button(
                    label="📥 下载结果 Bundle",
                    data=bundle_file_path.read_text(encoding="utf-8"),
                    file_name=bundle_file_path.name,
                    mime="application/json",
                    width="stretch",
                )

    if not visible_results:
        st.info("当前筛选范围下没有可展示的候选。")
    else:
        num_cols = 3
        for row_start in range(0, len(visible_results), num_cols):
            cols = st.columns(num_cols)
            for col_idx in range(num_cols):
                result_idx = row_start + col_idx
                if result_idx >= len(visible_results):
                    continue
                result_item = visible_results[result_idx]
                candidate_id = get_candidate_id(result_item, result_idx)
                with cols[col_idx]:
                    display_candidate_result(
                        result_item,
                        candidate_id,
                        current_mode,
                        task_name=current_task_name,
                        candidate_index=result_idx,
                    )

    st.divider()
    st.markdown("### 💾 批量下载")
    export_results = filter_generation_results_by_scope(results, export_scope)
    if not export_results:
        st.info("当前导出范围下没有候选可打包。")
        return
    try:
        final_zip_bytes, final_exported_count, final_zip_failures = build_final_results_zip(
            export_results,
            task_name=current_task_name,
            exp_mode=current_mode,
        )
        full_zip_bytes, full_exported_count, full_zip_failures = build_full_process_zip(
            export_results,
            task_name=current_task_name,
            exp_mode=current_mode,
            dataset_name=current_dataset_name,
            timestamp=timestamp,
            source_label=f"{result_source_label} | 导出范围={export_scope}",
            json_file_path=json_file_path,
            bundle_file_path=bundle_file_path,
        )

        download_cols = st.columns(2)
        with download_cols[0]:
            if final_exported_count > 0:
                st.download_button(
                    label="📥 下载最终候选（ZIP）",
                    data=final_zip_bytes,
                    file_name=f"paperbanana_{current_task_name}_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    width="stretch",
                )
            else:
                st.button("📥 下载最终候选（ZIP）", disabled=True, width="stretch")

        with download_cols[1]:
            if full_exported_count > 0:
                st.download_button(
                    label="📥 下载全流程总览（ZIP）",
                    data=full_zip_bytes,
                    file_name=f"PaperBanana_全流程总览_{current_task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    width="stretch",
                )
            else:
                st.button("📥 下载全流程总览（ZIP）", disabled=True, width="stretch")

        st.caption(
            f"最终候选 ZIP：成功导出 {final_exported_count} 个候选 | "
            f"全流程总览 ZIP：成功导出 {full_exported_count} 个候选。"
        )
        st.info("全流程总览 ZIP 会按中文目录整理会话总览、每个候选的最终结果、阶段演化图、阶段描述、评审建议、绘图代码和原始结果 JSON。")
        if final_zip_failures or full_zip_failures:
            with st.expander("查看 ZIP 导出详情", expanded=False):
                if final_zip_failures:
                    st.markdown("**最终候选 ZIP**")
                    for item in final_zip_failures:
                        st.write(item)
                if full_zip_failures:
                    st.markdown("**全流程总览 ZIP**")
                    for item in full_zip_failures:
                        st.write(item)
    except Exception as export_error:
        st.error(f"创建 ZIP 压缩包失败：{export_error}")


def _build_generation_terminal_notice(snapshot: dict) -> tuple[str, str]:
    completed_count = len(snapshot.get("results", []))
    requested_count = int(snapshot.get("requested_candidates", completed_count) or completed_count)
    failed_count = sum(
        1
        for item in snapshot.get("results", [])
        if isinstance(item, dict) and item.get("status") == "failed"
    )
    status = snapshot.get("status")
    if status == "completed":
        return "success", f"✅ 生成任务完成：完成 {completed_count}/{requested_count} 个候选，失败 {failed_count} 个。"
    if status == "cancelled":
        return "warning", f"⛔ 生成任务已停止：保留 {completed_count}/{requested_count} 个候选，失败 {failed_count} 个。"
    if status == "interrupted":
        return "warning", f"🧭 已恢复最近一次历史快照：当前保留 {completed_count}/{requested_count} 个候选，失败 {failed_count} 个。"
    return "error", f"❌ 后台生成失败：{snapshot.get('error', 'Unknown error')}"


def _build_refine_terminal_notice(snapshot: dict) -> tuple[str, str]:
    completed = len(snapshot.get("refined_images", []))
    failed = len(snapshot.get("failed_results", []))
    status = snapshot.get("status")
    if status == "completed":
        return "success", f"✅ 后台精修完成：成功 {completed} 张，失败 {failed} 张。"
    if status == "cancelled":
        return "warning", f"⛔ 精修已停止：成功 {completed} 张，失败/取消 {failed} 张。"
    if status == "interrupted":
        return "warning", f"🧭 已恢复最近一次历史精修快照：成功 {completed} 张，失败 {failed} 张。"
    return "error", f"❌ 后台精修失败：{snapshot.get('error', 'Unknown error')}"


def _render_background_terminal_notice(session_key: str) -> None:
    notice = st.session_state.pop(session_key, None)
    if not isinstance(notice, dict):
        return
    level = str(notice.get("level", "info") or "info").lower()
    message = str(notice.get("message", "") or "").strip()
    if not message:
        return
    renderer = getattr(st, level, None)
    if callable(renderer):
        renderer(message)
    else:
        st.info(message)


def _render_generation_activity_content(*, requested_candidates: int, allow_fragment_rerun: bool) -> None:
    active_generation_job_id = st.session_state.get("active_generation_job_id")
    active_generation_snapshot = None
    if active_generation_job_id:
        active_generation_snapshot = hydrate_persisted_job_snapshot(
            get_generation_job_snapshot(active_generation_job_id),
            job_kind="generation",
        )

    finalized_generation_snapshot = None
    if (
        active_generation_snapshot
        and active_generation_snapshot.get("status") in BACKGROUND_TERMINAL_STATUSES
        and active_generation_snapshot.get("worker_done", True)
    ):
        finalized_generation_snapshot = active_generation_snapshot
        if st.session_state.get("last_generation_completed_job_id") != active_generation_job_id:
            snapshot_status = active_generation_snapshot.get("status")
            if snapshot_status in {"completed", "cancelled", "interrupted"} and active_generation_snapshot.get("results"):
                source_label = "后台生成任务"
                if snapshot_status == "cancelled":
                    source_label = "后台生成任务（已停止）"
                elif snapshot_status == "interrupted":
                    source_label = "已恢复的历史生成快照"
                persist_generation_job_results(active_generation_snapshot, source_label=source_label)
            elif snapshot_status == "failed":
                st.session_state["generation_failures"] = active_generation_snapshot.get("failures", [])
            st.session_state["last_generation_completed_job_id"] = active_generation_job_id
        st.session_state.pop("active_generation_job_id", None)
        clear_generation_job(active_generation_job_id)
        active_generation_snapshot = None
        persist_demo_ui_state()
        if allow_fragment_rerun:
            notice_level, notice_message = _build_generation_terminal_notice(finalized_generation_snapshot)
            st.session_state["generation_terminal_notice"] = {
                "level": notice_level,
                "message": notice_message,
            }
            request_streamlit_rerun(scope="app")
            return

    runtime_panel_action = render_generation_runtime_panel(
        active_generation_snapshot or finalized_generation_snapshot,
        requested_candidates=requested_candidates,
    )
    if runtime_panel_action == "cancel" and active_generation_snapshot:
        request_generation_job_cancel(active_generation_snapshot["job_id"])
        st.warning("已发送停止请求，当前进行中的候选会继续完成，未开始的候选将被跳过。")
        persist_demo_ui_state()
        request_streamlit_rerun(scope="fragment" if allow_fragment_rerun else "app")
        return
    if runtime_panel_action == "refresh":
        persist_demo_ui_state()
        request_streamlit_rerun(scope="fragment" if allow_fragment_rerun else "app")
        return

    if finalized_generation_snapshot:
        notice_level, notice_message = _build_generation_terminal_notice(finalized_generation_snapshot)
        renderer = getattr(st, notice_level, None)
        if callable(renderer):
            renderer(notice_message)
        else:
            st.info(notice_message)

    render_generation_live_stream(active_generation_snapshot or finalized_generation_snapshot)


@streamlit_fragment(run_every=BACKGROUND_FRAGMENT_AUTO_REFRESH_INTERVAL)
def _render_generation_activity_fragment_running(*, requested_candidates: int, default_task_name: str) -> None:
    _render_generation_activity_content(
        requested_candidates=requested_candidates,
        allow_fragment_rerun=True,
    )


def render_generation_activity_fragment(*, requested_candidates: int, default_task_name: str) -> None:
    active_generation_job_id = st.session_state.get("active_generation_job_id")
    active_generation_snapshot = hydrate_persisted_job_snapshot(
        get_generation_job_snapshot(active_generation_job_id),
        job_kind="generation",
    ) if active_generation_job_id else None
    if (
        supports_streamlit_fragment()
        and active_generation_snapshot
        and active_generation_snapshot.get("status") == "running"
    ):
        _render_generation_activity_fragment_running(
            requested_candidates=requested_candidates,
            default_task_name=default_task_name,
        )
        return
    _render_generation_activity_content(
        requested_candidates=requested_candidates,
        allow_fragment_rerun=False,
    )


def _render_refine_activity_content(
    *,
    requested_images: int,
    fallback_original_bytes: bytes,
    fallback_resolution: str,
    fallback_provider: str,
    fallback_image_model_name: str,
    allow_fragment_rerun: bool,
) -> None:
    active_refine_job_id = st.session_state.get("active_refine_job_id")
    active_refine_snapshot = None
    if active_refine_job_id:
        active_refine_snapshot = hydrate_persisted_job_snapshot(
            get_refine_job_snapshot(active_refine_job_id),
            job_kind="refine",
        )

    finalized_refine_snapshot = None
    if (
        active_refine_snapshot
        and active_refine_snapshot.get("status") in BACKGROUND_TERMINAL_STATUSES
        and active_refine_snapshot.get("worker_done", True)
    ):
        finalized_refine_snapshot = active_refine_snapshot
        if st.session_state.get("last_refine_completed_job_id") != active_refine_job_id:
            persist_refine_job_results(active_refine_snapshot)
            st.session_state["last_refine_completed_job_id"] = active_refine_job_id
        st.session_state.pop("active_refine_job_id", None)
        clear_refine_job(active_refine_job_id)
        active_refine_snapshot = None
        persist_demo_ui_state()
        if allow_fragment_rerun:
            notice_level, notice_message = _build_refine_terminal_notice(finalized_refine_snapshot)
            st.session_state["refine_terminal_notice"] = {
                "level": notice_level,
                "message": notice_message,
            }
            request_streamlit_rerun(scope="app")
            return

    refine_panel_action = render_refine_runtime_panel(
        active_refine_snapshot or finalized_refine_snapshot,
        requested_images=requested_images,
    )
    if refine_panel_action == "cancel" and active_refine_snapshot:
        request_refine_job_cancel(active_refine_snapshot["job_id"])
        st.warning("已发送停止请求，系统会在当前请求结束后停止后续重试。")
        persist_demo_ui_state()
        request_streamlit_rerun(scope="fragment" if allow_fragment_rerun else "app")
        return
    if refine_panel_action == "refresh":
        persist_demo_ui_state()
        request_streamlit_rerun(scope="fragment" if allow_fragment_rerun else "app")
        return

    if finalized_refine_snapshot:
        notice_level, notice_message = _build_refine_terminal_notice(finalized_refine_snapshot)
        renderer = getattr(st, notice_level, None)
        if callable(renderer):
            renderer(notice_message)
        else:
            st.info(notice_message)

    render_refine_results_section(
        fallback_original_bytes=fallback_original_bytes,
        fallback_resolution=fallback_resolution,
        fallback_provider=fallback_provider,
        fallback_image_model_name=fallback_image_model_name,
    )


@streamlit_fragment(run_every=BACKGROUND_FRAGMENT_AUTO_REFRESH_INTERVAL)
def _render_refine_activity_fragment_running(
    *,
    requested_images: int,
    fallback_original_bytes: bytes,
    fallback_resolution: str,
    fallback_provider: str,
    fallback_image_model_name: str,
) -> None:
    _render_refine_activity_content(
        requested_images=requested_images,
        fallback_original_bytes=fallback_original_bytes,
        fallback_resolution=fallback_resolution,
        fallback_provider=fallback_provider,
        fallback_image_model_name=fallback_image_model_name,
        allow_fragment_rerun=True,
    )


def render_refine_activity_fragment(
    *,
    requested_images: int,
    fallback_original_bytes: bytes,
    fallback_resolution: str,
    fallback_provider: str,
    fallback_image_model_name: str,
) -> None:
    active_refine_job_id = st.session_state.get("active_refine_job_id")
    active_refine_snapshot = hydrate_persisted_job_snapshot(
        get_refine_job_snapshot(active_refine_job_id),
        job_kind="refine",
    ) if active_refine_job_id else None
    if (
        supports_streamlit_fragment()
        and active_refine_snapshot
        and active_refine_snapshot.get("status") == "running"
    ):
        _render_refine_activity_fragment_running(
            requested_images=requested_images,
            fallback_original_bytes=fallback_original_bytes,
            fallback_resolution=fallback_resolution,
            fallback_provider=fallback_provider,
            fallback_image_model_name=fallback_image_model_name,
        )
        return
    _render_refine_activity_content(
        requested_images=requested_images,
        fallback_original_bytes=fallback_original_bytes,
        fallback_resolution=fallback_resolution,
        fallback_provider=fallback_provider,
        fallback_image_model_name=fallback_image_model_name,
        allow_fragment_rerun=False,
    )


def render_generation_workspace() -> None:
    _apply_pending_generation_widget_state_updates()

    active_generation_job_id = st.session_state.get("active_generation_job_id")
    active_generation_snapshot = hydrate_persisted_job_snapshot(
        get_generation_job_snapshot(active_generation_job_id),
        job_kind="generation",
    ) if active_generation_job_id else None
    generation_is_running = bool(
        active_generation_snapshot and active_generation_snapshot.get("status") == "running"
    )

    advanced_settings = render_generation_sidebar_controls()
    task_name = normalize_task_name(advanced_settings["task_name"])
    num_candidates = int(advanced_settings["num_candidates"])
    task_config = get_task_ui_config(task_name)
    effective_settings = _build_generation_effective_settings(
        advanced_settings,
        task_name=task_name,
    )

    st.markdown(f"#### 📝 {task_config['intro']}")

    content_state_key = f"tab1_{task_name}_content"
    visual_state_key = f"tab1_{task_name}_visual_intent"
    content_example_key = f"tab1_{task_name}_content_example_selector"
    visual_example_key = f"tab1_{task_name}_visual_example_selector"
    example_options = [EXAMPLE_SELECTOR_NONE_OPTION, task_config["example_name"]]

    content_editor_key = f"{content_state_key}_editor"
    visual_editor_key = f"{visual_state_key}_editor"

    if content_editor_key not in st.session_state:
        st.session_state[content_editor_key] = st.session_state.get(content_state_key, "")
    if visual_editor_key not in st.session_state:
        st.session_state[visual_editor_key] = st.session_state.get(visual_state_key, "")
    _prime_generation_example_selector_state(content_example_key)
    _prime_generation_example_selector_state(visual_example_key)

    example_col1, example_col2 = st.columns([3, 2])
    with example_col1:
        content_example = st.selectbox(
            task_config["content_selector_label"],
            example_options,
            key=content_example_key,
        )
    with example_col2:
        visual_example = st.selectbox(
            task_config["visual_selector_label"],
            example_options,
            key=visual_example_key,
        )

    content_example_applied = _apply_generation_example_selection(
        selector_key=content_example_key,
        selected_value=content_example,
        editor_key=content_editor_key,
        example_name=task_config["example_name"],
        example_value=task_config["example_content"],
    )
    visual_example_applied = _apply_generation_example_selection(
        selector_key=visual_example_key,
        selected_value=visual_example,
        editor_key=visual_editor_key,
        example_name=task_config["example_name"],
        example_value=task_config["example_visual_intent"],
    )
    if content_example_applied or visual_example_applied:
        request_streamlit_rerun()
        return

    with st.form("generation_request_form", clear_on_submit=False):
        col_input1, col_input2 = st.columns([3, 2])
        with col_input1:
            input_content = st.text_area(
                task_config["content_label"],
                height=180,
                max_chars=GENERATION_TEXTAREA_MAX_CHARS,
                placeholder=task_config["content_placeholder"],
                help=task_config["content_help"],
                key=content_editor_key,
            )

        with col_input2:
            visual_intent = st.text_area(
                task_config["visual_label"],
                height=180,
                max_chars=GENERATION_TEXTAREA_MAX_CHARS,
                placeholder=task_config["visual_placeholder"],
                help=task_config["visual_help"],
                key=visual_editor_key,
            )

        content_for_generation = input_content
        allow_raw_plot_input = False
        if task_name == "plot":
            parsed_plot_input = parse_plot_input_text(input_content)
            if parsed_plot_input["ok"]:
                content_for_generation = parsed_plot_input["normalized_content"]
                st.caption(
                    f"已解析 plot 输入：格式={parsed_plot_input['format']} | 行数={parsed_plot_input['row_count']} | "
                    f"字段={', '.join(parsed_plot_input['columns']) or 'N/A'}"
                )
                if parsed_plot_input["preview_rows"]:
                    with st.expander("🔎 查看结构化数据预览", expanded=False):
                        st.dataframe(parsed_plot_input["preview_rows"], width="stretch")
            else:
                st.warning(parsed_plot_input["error"])
                allow_raw_plot_input = st.checkbox(
                    "按原始文本继续（跳过结构化解析）",
                    key="plot_allow_raw_input",
                    help="仅在你的数据不是 JSON、CSV 或 Markdown 表格时使用。",
                )

        preflight_report = build_generation_preflight_report(
            task_name=task_name,
            input_content=input_content,
            visual_intent=visual_intent,
            content_for_generation=content_for_generation,
            allow_raw_plot_input=allow_raw_plot_input,
            num_candidates=int(num_candidates),
            effective_settings=effective_settings,
            retrieval_ref_path=advanced_settings["retrieval_ref_path"],
            resolved_profile_path=advanced_settings["resolved_profile_path"],
            generation_is_running=generation_is_running,
        )
        render_preflight_summary(preflight_report)
        submit_generation = st.form_submit_button(
            "🚀 生成候选方案",
            type="primary",
            width="stretch",
            disabled=generation_is_running,
        )

    if submit_generation:
        if preflight_report["errors"]:
            st.error("当前请求未通过启动前检查，请先修正上面的错误项。")
        else:
            st.session_state[content_state_key] = input_content
            st.session_state[visual_state_key] = visual_intent
            job_id = start_generation_background_job(
                dataset_name=advanced_settings["dataset_name"],
                task_name=task_name,
                exp_mode=effective_settings["exp_mode"],
                retrieval_setting=effective_settings["retrieval_setting"],
                curated_profile=advanced_settings["curated_profile"],
                provider=advanced_settings["provider"],
                connection_id=advanced_settings["connection_id"],
                api_key=advanced_settings["api_key"],
                model_name=advanced_settings["model_name"],
                image_model_name=advanced_settings["image_model_name"],
                base_url=advanced_settings["base_url"],
                extra_headers=advanced_settings["extra_headers"],
                concurrency_mode=advanced_settings["concurrency_mode"],
                max_concurrent=int(advanced_settings["max_concurrent"]),
                num_candidates=int(num_candidates),
                max_critic_rounds=int(effective_settings["max_critic_rounds"]),
                aspect_ratio=advanced_settings["aspect_ratio"],
                image_resolution=effective_settings["image_resolution"],
                image_generation_options=effective_settings.get("image_generation_options", {}),
                content=content_for_generation,
                visual_intent=visual_intent,
            )
            st.session_state["active_generation_job_id"] = job_id
            st.session_state["last_generation_completed_job_id"] = None
            persist_demo_ui_state()
            st.rerun()

    if task_name == "plot":
        render_plot_rerender_workspace()

    _render_background_terminal_notice("generation_terminal_notice")
    render_generation_activity_fragment(
        requested_candidates=int(num_candidates),
        default_task_name=task_name,
    )
    render_generation_results_panel(task_name)
    render_generation_history_panel(task_name)


def render_refine_workbench_panel(
    *,
    refine_is_running: bool,
    refine_resolution: str,
    refine_aspect_ratio: str,
    refine_num_images: int,
    refine_provider: str,
    refine_provider_type: str,
    refine_api_key: str,
    refine_image_model_name: str,
    refine_connection_id: str = "",
    refine_base_url: str = "",
    refine_extra_headers: dict[str, str] | None = None,
    refine_image_generation_options: dict[str, Any] | None = None,
    refine_connection_pending_save: bool = False,
    refine_extra_headers_error: str = "",
) -> tuple[bytes, str]:
    """渲染精修工作台主面板，并返回当前选中的输入图像。"""
    staged_refine_bytes = bytes(st.session_state.get("refine_staged_image_bytes", b"") or b"")
    cached_uploaded_bytes = bytes(st.session_state.get("refine_uploaded_image_bytes", b"") or b"")
    current_source_kind = "候选方案" if staged_refine_bytes else "上传图像" if cached_uploaded_bytes else "未选择"
    version_count = len(get_refine_version_history())
    result_count = len(st.session_state.get("refined_images", []) or [])

    with st.container(border=True):
        st.markdown("### 🧭 精修工作台")
        metric_cols = st.columns(4)
        metric_cols[0].metric("当前来源", current_source_kind)
        metric_cols[1].metric("版本数", version_count)
        metric_cols[2].metric("已有结果", result_count)
        metric_cols[3].metric("后台状态", "运行中" if refine_is_running else "空闲")

    if refine_is_running:
        st.info("后台精修任务运行中；当前工作台仍可查看输入图像，实时进度请切换到“结果与状态”。")

    preview_image = None
    preview_error = None
    selected_image_bytes = b""
    selected_input_mime_type = "image/png"
    selected_source_label = "上传图像"
    submit_refine = False
    edit_prompt = st.session_state.get("edit_prompt", "")

    workbench_col1, workbench_col2 = st.columns([1.08, 0.92], gap="large")
    with workbench_col1:
        with st.container(border=True):
            st.markdown("### 输入图像")
            st.caption("支持上传图片，或直接复用生成页/版本历史送来的候选版本。")

            upload_widget_nonce = int(st.session_state.get("refine_upload_widget_nonce", 0) or 0)
            upload_widget_key = f"refine_uploaded_file_input_{upload_widget_nonce}"
            previous_upload_present = bool(st.session_state.get("refine_uploaded_widget_present", False))
            source_options = ["上传图像"]
            if staged_refine_bytes:
                source_options.append("候选方案")
            default_refine_source = st.session_state.get(
                "refine_input_source",
                "候选方案" if staged_refine_bytes else "上传图像",
            )
            if default_refine_source not in source_options:
                default_refine_source = source_options[0]
            refine_input_source = st.radio(
                "图像来源",
                source_options,
                index=source_options.index(default_refine_source),
                key="refine_input_source",
                horizontal=True,
                label_visibility="collapsed",
            )
            uploaded_file = st.file_uploader(
                "选择一个图像文件",
                type=["png", "jpg", "jpeg"],
                help="上传您想要精修的图表。",
                key=upload_widget_key,
            )
            if uploaded_file is not None:
                upload_error = cache_refine_uploaded_file(uploaded_file)
                if upload_error:
                    persist_demo_ui_state()
                    st.rerun()
            elif previous_upload_present:
                clear_cached_refine_upload(clear_error=True, reset_widget=False)

            cached_uploaded_bytes = bytes(st.session_state.get("refine_uploaded_image_bytes", b"") or b"")
            cached_uploaded_filename = str(st.session_state.get("refine_uploaded_filename", "") or "")
            cached_uploaded_size = int(st.session_state.get("refine_uploaded_file_size", 0) or 0)
            cached_upload_error = str(st.session_state.get("refine_upload_error", "") or "")
            if cached_uploaded_bytes:
                upload_col1, upload_col2 = st.columns([4, 1])
                with upload_col1:
                    size_mb = cached_uploaded_size / (1024 * 1024) if cached_uploaded_size else 0.0
                    display_name = cached_uploaded_filename or "已上传图像"
                    st.caption(f"已缓存上传图像：{display_name}（{size_mb:.2f} MB）")
                with upload_col2:
                    if st.button("清除上传", width="stretch", key="clear_refine_uploaded_image"):
                        clear_cached_refine_upload(clear_error=True, reset_widget=True)
                        persist_demo_ui_state()
                        st.rerun()
            elif cached_upload_error:
                st.error(cached_upload_error)

            if staged_refine_bytes:
                staged_label = st.session_state.get("refine_staged_source_label", "候选方案")
                staged_col1, staged_col2 = st.columns([4, 1])
                with staged_col1:
                    st.caption(f"已载入候选来源：{staged_label}")
                with staged_col2:
                    if st.button("清除候选", width="stretch", key="clear_refine_staged_source"):
                        clear_staged_refine_source()
                        persist_demo_ui_state()
                        st.rerun()

            if refine_input_source == "候选方案" and staged_refine_bytes:
                selected_image_bytes = staged_refine_bytes
                selected_input_mime_type = st.session_state.get("refine_staged_input_mime_type", "image/png")
                selected_source_label = st.session_state.get("refine_staged_source_label", "候选方案")
            elif cached_uploaded_bytes:
                selected_image_bytes = cached_uploaded_bytes
                selected_input_mime_type = st.session_state.get("refine_uploaded_input_mime_type", "image/png")
                selected_source_label = "上传图像"

            st.session_state["refine_selected_image_bytes"] = selected_image_bytes
            st.session_state["refine_selected_input_mime_type"] = selected_input_mime_type
            st.session_state["refine_selected_source_label"] = selected_source_label

            if selected_image_bytes:
                preview_image, preview_error = load_refine_preview_image(
                    selected_image_bytes,
                    source_label=selected_source_label,
                )
                if preview_image is not None:
                    active_version_key = ensure_refine_source_version(
                        selected_image_bytes,
                        input_mime_type=selected_input_mime_type,
                        source_label=selected_source_label,
                    )
                    current_active_entry = find_refine_version_entry(
                        st.session_state.get("refine_active_version_key", "")
                    )
                    if active_version_key and (
                        current_active_entry is None
                        or current_active_entry.get("image_digest") != _compute_image_digest(selected_image_bytes)
                    ):
                        st.session_state["refine_active_version_key"] = active_version_key
                else:
                    st.error(preview_error or "当前图像无法预览，请重新选择。")
                    recovery_col1, recovery_col2 = st.columns(2)
                    with recovery_col1:
                        if selected_source_label == "上传图像" and st.button(
                            "清除损坏上传",
                            width="stretch",
                            key="clear_invalid_refine_upload",
                        ):
                            clear_cached_refine_upload(clear_error=True, reset_widget=True)
                            persist_demo_ui_state()
                            st.rerun()
                    with recovery_col2:
                        if selected_source_label != "上传图像" and st.button(
                            "清除当前候选",
                            width="stretch",
                            key="clear_invalid_refine_source",
                        ):
                            clear_staged_refine_source()
                            persist_demo_ui_state()
                            st.rerun()

            if preview_image is not None:
                st.caption(f"当前预览：{selected_source_label}")
                _, preview_col, _ = st.columns([0.05, 0.9, 0.05])
                with preview_col:
                    st.image(preview_image, width=REFINE_WORKSPACE_PREVIEW_WIDTH)
            else:
                st.info("请上传图像，或先在生成结果/版本历史中点击“送去精修”载入候选方案。")

    with workbench_col2:
        with st.container(border=True):
            st.markdown("### 编辑指令")
            st.caption(
                f"本次输出：{refine_resolution} · {refine_aspect_ratio} · {int(refine_num_images)} 张 · Provider={refine_provider}"
            )
            with st.form("refine_request_form", clear_on_submit=False):
                edit_prompt = st.text_area(
                    "描述您想要的修改",
                    height=220,
                    placeholder="例如：保持语义不变，增强标题和标签清晰度，优化对齐、留白和整体视觉层次。",
                    help="描述您想要的修改，或使用“保持内容不变”仅进行放大与清晰化。",
                    key="edit_prompt",
                )
                submit_refine = st.form_submit_button(
                    "✨ 精修图像",
                    type="primary",
                    width="stretch",
                    disabled=refine_is_running or preview_image is None,
                )

            with st.expander("查看指令建议", expanded=False):
                st.markdown(
                    "- 保持语义不变，优化清晰度、对齐、留白和视觉层次。\n"
                    "- 放大标题与图例文字，确保论文截图中仍清晰可读。\n"
                    "- 保持主色不变，仅修复边缘锯齿和文本模糊。"
                )

            if preview_image is None:
                st.info("先选择一张可预览的输入图像，右侧按钮才会启用。")

    if submit_refine:
        if not edit_prompt:
            st.error("请提供编辑指令！")
        elif refine_is_running:
            st.warning("当前已有精修任务在后台运行，请先等待完成或停止当前任务。")
        elif refine_connection_pending_save:
            st.error("当前选择的是未保存的自定义连接草稿；请先点击“保存连接”，再启动正式精修。")
        elif refine_extra_headers_error:
            st.error(refine_extra_headers_error)
        elif refine_provider_type not in {"gemini", "evolink", "openai", "openai_compatible"}:
            st.error("当前精修链路仅支持 Gemini、Evolink 或 OpenAI 图像编辑接口；如需验证其他连接，请先使用“测试连接”。")
        else:
            job_id = start_refine_background_job(
                image_bytes=selected_image_bytes,
                edit_prompt=edit_prompt,
                num_images=int(refine_num_images),
                aspect_ratio=refine_aspect_ratio,
                image_size=refine_resolution,
                api_key=refine_api_key,
                provider=refine_provider,
                connection_id=refine_connection_id,
                image_model_name=refine_image_model_name,
                base_url=refine_base_url,
                extra_headers=refine_extra_headers,
                image_generation_options=refine_image_generation_options,
                input_mime_type=selected_input_mime_type,
            )
            st.session_state["refined_images"] = []
            st.session_state["refine_failed_results"] = []
            st.session_state["active_refine_job_id"] = job_id
            st.session_state["last_refine_completed_job_id"] = None
            _queue_refine_widget_state_updates(
                {"refine_workspace_view": REFINE_WORKSPACE_VIEW_OPTIONS[1]}
            )
            persist_demo_ui_state()
            st.rerun()

    return selected_image_bytes, selected_input_mime_type


def render_refine_workspace() -> None:
    _apply_pending_refine_widget_state_updates()
    refine_settings = render_refine_sidebar_controls()
    st.markdown("### 精修并放大图表（2K / 4K）")
    st.caption("布局已切成和主页面一致的“侧边栏参数 + 主工作台”体验，减少长页面滚动。")

    active_refine_job_id = st.session_state.get("active_refine_job_id")
    active_refine_snapshot = hydrate_persisted_job_snapshot(
        get_refine_job_snapshot(active_refine_job_id),
        job_kind="refine",
    ) if active_refine_job_id else None
    refine_is_running = bool(
        active_refine_snapshot and active_refine_snapshot.get("status") == "running"
    )
    _render_background_terminal_notice("refine_terminal_notice")
    default_refine_view = st.session_state.get(
        "refine_workspace_view",
        REFINE_WORKSPACE_VIEW_OPTIONS[1] if refine_is_running else REFINE_WORKSPACE_VIEW_OPTIONS[0],
    )
    ensure_session_choice_state(
        "refine_workspace_view",
        REFINE_WORKSPACE_VIEW_OPTIONS,
        default_refine_view if default_refine_view in REFINE_WORKSPACE_VIEW_OPTIONS else REFINE_WORKSPACE_VIEW_OPTIONS[0],
    )
    current_refine_view = st.radio(
        "精修视图",
        REFINE_WORKSPACE_VIEW_OPTIONS,
        key="refine_workspace_view",
        horizontal=True,
        label_visibility="collapsed",
    )

    selected_image_bytes = bytes(st.session_state.get("refine_selected_image_bytes", b"") or b"")
    selected_input_mime_type = st.session_state.get("refine_selected_input_mime_type", "image/png")
    if current_refine_view == REFINE_WORKSPACE_VIEW_OPTIONS[0]:
        selected_image_bytes, selected_input_mime_type = render_refine_workbench_panel(
            refine_is_running=refine_is_running,
            refine_resolution=refine_settings["resolution"],
            refine_aspect_ratio=refine_settings["aspect_ratio"],
            refine_num_images=int(refine_settings["num_images"]),
            refine_provider=refine_settings["provider"],
            refine_provider_type=refine_settings["provider_type"],
            refine_api_key=refine_settings["api_key"],
            refine_connection_id=refine_settings["connection_id"],
            refine_image_model_name=refine_settings["image_model_name"],
            refine_base_url=refine_settings["base_url"],
            refine_extra_headers=refine_settings["extra_headers"],
            refine_image_generation_options=refine_settings.get("image_generation_options", {}),
            refine_connection_pending_save=bool(refine_settings.get("connection_pending_save")),
            refine_extra_headers_error=str(refine_settings.get("extra_headers_error", "") or ""),
        )
    elif current_refine_view == REFINE_WORKSPACE_VIEW_OPTIONS[1]:
        if not active_refine_job_id and not st.session_state.get("refined_images"):
            st.info("当前还没有后台精修任务或结果；先去“工作台”选择图像并发起一次精修。")
        else:
            render_refine_activity_fragment(
                requested_images=int(refine_settings["num_images"]),
                fallback_original_bytes=selected_image_bytes,
                fallback_resolution=refine_settings["resolution"],
                fallback_provider=refine_settings["provider"],
                fallback_image_model_name=refine_settings["image_model_name"],
            )
    else:
        if not get_refine_version_history():
            st.info("当前还没有可切换的历史版本；先去“工作台”启动一次精修，或从生成结果送入候选图像。")
        else:
            render_refine_version_history_panel()


def main():
    restore_persisted_demo_ui_state()
    st.markdown(
        """
        <style>
        .pb-header { display:flex; align-items:center; gap:12px; padding:4px 0 0 0; }
        .pb-header .title { font-size:1.75rem; font-weight:700; margin:0; }
        .pb-header .badge {
            font-size:0.72rem; font-weight:600;
            background: linear-gradient(135deg, #f7c948 0%, #f2994a 100%);
            color: #fff; padding:3px 10px; border-radius:12px;
            letter-spacing:0.03em;
        }
        .pb-sub { color:#888; font-size:0.88rem; margin:0 0 2px 0; }
        .pb-sep { border:none; border-top:1px solid #e0e0e0; margin:4px 0 0 0; }
        </style>
        <div class="pb-header">
            <p class="title">🍌 PaperBanana-Pro</p>
            <span class="badge">工作台</span>
        </div>
        <p class="pb-sub">AI 驱动的科学图表生成与精修</p>
        """,
        unsafe_allow_html=True,
    )

    workspace_mode = st.radio(
        "工作区",
        WORKSPACE_MODE_OPTIONS,
        key="workspace_mode",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown('<hr class="pb-sep">', unsafe_allow_html=True)

    if workspace_mode == WORKSPACE_MODE_OPTIONS[0]:
        render_generation_workspace()
    else:
        render_refine_workspace()

    persist_demo_ui_state()


if __name__ == "__main__":
    main()
