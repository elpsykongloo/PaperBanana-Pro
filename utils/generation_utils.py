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
API 调用工具函数，支持 Evolink、Gemini、Claude、OpenAI 等多种 Provider。
"""

import json
import asyncio
import base64
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any, Callable, Optional, Tuple

from PIL import Image

import os
from pathlib import Path

from utils.config_loader import load_model_config, get_config_val
from utils.image_generation_options import (
    build_openai_image_request_params,
    get_image_model_capabilities,
    normalize_image_generation_options,
)
from utils.log_config import get_logger
from utils.runtime_events import create_runtime_event

logger = get_logger("GenerationUtils")

_SENSITIVE_TOKEN_RE = re.compile(r"\b(sk-[A-Za-z0-9_\-]{5})[A-Za-z0-9_\-]{8,}([A-Za-z0-9_\-]{4})\b")
_BEARER_TOKEN_RE = re.compile(r"(?i)(bearer\s+)([A-Za-z0-9_\-\.]{12,})")
_SECRET_FIELD_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|secret)(['\"\s:=]+)([^,'\"\s}{]{8,})"
)

# ==================== 配置加载 ====================

REPO_ROOT = Path(__file__).parent.parent
model_config = load_model_config(REPO_ROOT)


# ==================== 运行时事件回调（用于 UI 实时反馈） ====================

runtime_event_hook: Optional[Callable[[dict[str, Any]], None]] = None
runtime_status_hook: Optional[Callable[[str], None]] = None

DEFAULT_GEMINI_IMAGE_FALLBACK_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_GEMINI_TEXT_FALLBACK_MODELS = (
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
)

evolink_base_url = get_config_val(
    model_config,
    "evolink",
    "base_url",
    "EVOLINK_BASE_URL",
    "https://api.evolink.ai",
    base_dir=REPO_ROOT,
)


@dataclass
class RuntimeContext:
    connection_id: str = ""
    provider: str = ""
    api_key: str = ""
    base_url: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)
    event_hook: Optional[Callable[[dict[str, Any]], None]] = None
    status_hook: Optional[Callable[[str], None]] = None
    cancel_check: Optional[Callable[[], bool]] = None
    gemini_client: Any = None
    anthropic_client: Any = None
    openai_client: Any = None
    evolink_provider: Any = None
    owns_evolink_provider: bool = False


_active_runtime_context: ContextVar[RuntimeContext | None] = ContextVar(
    "paperbanana_runtime_context",
    default=None,
)
_default_runtime_context: RuntimeContext | None = None


def _sync_legacy_runtime_globals_from_default() -> None:
    global evolink_provider, gemini_client, anthropic_client, openai_client
    context = _default_runtime_context
    evolink_provider = context.evolink_provider if context else None
    gemini_client = context.gemini_client if context else None
    anthropic_client = context.anthropic_client if context else None
    openai_client = context.openai_client if context else None


def _create_evolink_provider(api_key: str, base_url: str = ""):
    if not api_key:
        return None
    try:
        from providers import create_provider
    except ImportError:
        logger.warning("⚠️  未安装 providers.evolink，Evolink Provider 不可用")
        return None
    url = base_url or evolink_base_url
    return create_provider("evolink", api_key=api_key, base_url=url)


def _create_gemini_client(api_key: str):
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        logger.warning("⚠️  未安装 google-genai，Gemini Client 不可用。请运行 pip install google-genai")
        return None


def _create_anthropic_client(api_key: str):
    if not api_key:
        return None
    try:
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(api_key=api_key)
    except ImportError:
        logger.warning("⚠️  未安装 anthropic，Anthropic Client 不可用")
        return None


def _create_openai_client(api_key: str, base_url: str = "", extra_headers: Optional[dict[str, str]] = None):
    if not api_key:
        return None
    try:
        from openai import AsyncOpenAI
        kwargs = {"api_key": api_key}
        if str(base_url or "").strip():
            kwargs["base_url"] = str(base_url).strip()
        if extra_headers:
            kwargs["default_headers"] = extra_headers
        return AsyncOpenAI(**kwargs)
    except ImportError:
        logger.warning("⚠️  未安装 openai，OpenAI Client 不可用")
        return None


def _create_openrouter_client(
    api_key: str,
    base_url: str = "",
    extra_headers: Optional[dict[str, str]] = None,
):
    """OpenRouter 使用 OpenAI 兼容接口，只需指定不同的 base_url。"""
    if not api_key:
        return None
    resolved_url = str(base_url or "https://openrouter.ai/api/v1").strip()
    return _create_openai_client(
        api_key,
        base_url=resolved_url,
        extra_headers=extra_headers,
    )


def get_default_runtime_context() -> RuntimeContext | None:
    return _default_runtime_context


def set_default_runtime_context(context: RuntimeContext | None) -> RuntimeContext | None:
    global _default_runtime_context
    _default_runtime_context = context
    _sync_legacy_runtime_globals_from_default()
    return _default_runtime_context


def get_active_runtime_context() -> RuntimeContext | None:
    return _active_runtime_context.get() or _default_runtime_context


def _runtime_cancel_requested() -> bool:
    context = get_active_runtime_context()
    if context is None or context.cancel_check is None:
        return False
    try:
        return bool(context.cancel_check())
    except Exception:
        return False


def get_evolink_provider():
    context = get_active_runtime_context()
    return context.evolink_provider if context else None


def get_gemini_client():
    context = get_active_runtime_context()
    return context.gemini_client if context else None


def get_anthropic_client():
    context = get_active_runtime_context()
    return context.anthropic_client if context else None


def get_openai_client():
    context = get_active_runtime_context()
    return context.openai_client if context else None


def create_runtime_context(
    *,
    connection_id: str = "",
    provider: str = "",
    api_key: str = "",
    event_hook: Optional[Callable[[dict[str, Any]], None]] = None,
    status_hook: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    base_url: str = "",
    extra_headers: Optional[dict[str, str]] = None,
) -> RuntimeContext:
    normalized_provider = str(provider or "").strip().lower()
    resolved_base_url = str(base_url or "").strip()
    if normalized_provider == "evolink" and not resolved_base_url:
        resolved_base_url = evolink_base_url
    context = RuntimeContext(
        connection_id=str(connection_id or "").strip(),
        provider=normalized_provider,
        api_key=str(api_key or "").strip(),
        base_url=resolved_base_url,
        extra_headers=dict(extra_headers or {}),
        event_hook=event_hook,
        status_hook=status_hook,
        cancel_check=cancel_check,
    )

    if normalized_provider == "evolink" and context.api_key:
        context.evolink_provider = _create_evolink_provider(context.api_key, resolved_base_url)
        context.owns_evolink_provider = context.evolink_provider is not None
    elif normalized_provider == "gemini" and context.api_key:
        context.gemini_client = _create_gemini_client(context.api_key)
    elif normalized_provider == "anthropic" and context.api_key:
        context.anthropic_client = _create_anthropic_client(context.api_key)
    elif normalized_provider == "openai" and context.api_key:
        context.openai_client = _create_openai_client(
            context.api_key,
            base_url=context.base_url,
            extra_headers=context.extra_headers,
        )
    elif normalized_provider in {"openrouter", "openai_compatible"} and context.api_key:
        context.openai_client = _create_openrouter_client(
            context.api_key,
            resolved_base_url,
            context.extra_headers,
        )

    return context


@contextmanager
def use_runtime_context(context: RuntimeContext | None):
    token = _active_runtime_context.set(context)
    try:
        yield context
    finally:
        _active_runtime_context.reset(token)


async def close_runtime_context(context: RuntimeContext | None) -> None:
    if context is None:
        return
    provider = context.evolink_provider
    if context.owns_evolink_provider and provider is not None and hasattr(provider, "close"):
        try:
            await provider.close()
        except Exception as err:
            _safe_log(f"[DEBUG] [WARN] close_runtime_context 失败: {err}")


def reinitialize_runtime_context(context: RuntimeContext | None) -> RuntimeContext | None:
    if context is None:
        return None
    if context.provider == "evolink" and context.api_key:
        context.evolink_provider = _create_evolink_provider(context.api_key, context.base_url)
        context.owns_evolink_provider = context.evolink_provider is not None
    elif context.provider == "gemini" and context.api_key:
        context.gemini_client = _create_gemini_client(context.api_key)
    elif context.provider == "anthropic" and context.api_key:
        context.anthropic_client = _create_anthropic_client(context.api_key)
    elif context.provider == "openai" and context.api_key:
        context.openai_client = _create_openai_client(
            context.api_key,
            base_url=context.base_url,
            extra_headers=context.extra_headers,
        )
    elif context.provider in {"openrouter", "openai_compatible"} and context.api_key:
        context.openai_client = _create_openrouter_client(
            context.api_key,
            context.base_url,
            context.extra_headers,
        )
    return context


def set_runtime_event_hook(hook: Optional[Callable[[dict[str, Any]], None]]) -> None:
    """设置运行时结构化事件回调。"""
    global runtime_event_hook
    runtime_event_hook = hook
    context = _active_runtime_context.get() or _default_runtime_context
    if context is not None:
        context.event_hook = hook


def set_runtime_status_hook(hook: Optional[Callable[[str], None]]) -> None:
    """兼容旧接口：设置纯文本状态回调。"""
    global runtime_status_hook
    runtime_status_hook = hook
    context = _active_runtime_context.get() or _default_runtime_context
    if context is not None:
        context.status_hook = hook


def _safe_text_for_log(value: Any, max_len: int = 6000) -> str:
    """Convert arbitrary value to a printable string safe for Windows stdout."""
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        text = repr(value)
    text = text.replace("\x00", "\\x00")
    text = _redact_sensitive_text(text)
    try:
        safe = text.encode("utf-8", errors="backslashreplace").decode("utf-8", errors="ignore")
    except Exception:
        safe = repr(text)
    if len(safe) > max_len:
        return safe[:max_len] + f"...(truncated {len(safe) - max_len} chars)"
    return safe


def _redact_sensitive_text(text: str) -> str:
    """脱敏 API key、Bearer token 等敏感字段，避免上游错误体原样进日志。"""
    safe = str(text or "")
    safe = _SENSITIVE_TOKEN_RE.sub(r"\1***\2", safe)
    safe = _BEARER_TOKEN_RE.sub(r"\1***", safe)
    safe = _SECRET_FIELD_RE.sub(r"\1\2***", safe)
    return safe


def _safe_log(message: Any) -> None:
    """尽力输出日志，永远不会向调用方抛出异常。"""
    try:
        logger.debug(_safe_text_for_log(message))
    except Exception:
        pass


def _emit_runtime_event(
    *,
    level: str = "INFO",
    kind: str = "job",
    message: str,
    source: str = "GenerationUtils",
    job_type: str = "",
    candidate_id: Any = "",
    stage: str = "",
    status: str = "",
    provider: str = "",
    model: str = "",
    attempt: int | None = None,
    error_code: int | None = None,
    preview_image: str = "",
    preview_mime_type: str = "",
    preview_label: str = "",
    details: Any = "",
) -> dict[str, Any]:
    """Emit one structured runtime event without raising to callers."""
    event = create_runtime_event(
        level=level,
        kind=kind,
        source=source,
        message=message,
        job_type=job_type,
        candidate_id=candidate_id,
        stage=stage,
        status=status,
        provider=provider,
        model=model,
        attempt=attempt,
        error_code=error_code,
        preview_image=preview_image,
        preview_mime_type=preview_mime_type,
        preview_label=preview_label,
        details=_safe_text_for_log(details, max_len=3000) if details else "",
    )
    payload = event.to_dict()
    try:
        logger.log(
            getattr(logging, event.level, logging.INFO),
            payload["message"],
            extra={"paperbanana_event": dict(payload)},
        )
    except Exception as err:
        _safe_log(f"runtime_event logger emit 失败: {err}")
    hook = runtime_event_hook
    context = get_active_runtime_context()
    if context is not None and context.event_hook is not None:
        hook = context.event_hook
    if hook is not None:
        try:
            hook(payload)
        except Exception as err:
            _safe_log(f"runtime_event_hook 调用失败: {err}")
    return payload


def _emit_runtime_status(message: str) -> None:
    """兼容旧接口：发送纯文本状态，同时优先转成结构化事件。"""
    payload = _emit_runtime_event(
        level="INFO",
        kind="job",
        message=message,
        source="GenerationUtils",
    )
    hook = runtime_status_hook
    context = get_active_runtime_context()
    if context is not None and context.status_hook is not None:
        hook = context.status_hook
    if hook is None:
        return
    try:
        hook(message)
    except Exception as err:
        _safe_log(f"runtime_status_hook 调用失败: {err}")

# ==================== 原始 Provider 初始化（保留兼容性） ====================

evolink_api_key = get_config_val(
    model_config,
    "evolink",
    "api_key",
    "EVOLINK_API_KEY",
    "",
    base_dir=REPO_ROOT,
)
gemini_api_key = get_config_val(
    model_config,
    "api_keys",
    "google_api_key",
    "GOOGLE_API_KEY",
    "",
    base_dir=REPO_ROOT,
)
anthropic_api_key = get_config_val(
    model_config,
    "api_keys",
    "anthropic_api_key",
    "ANTHROPIC_API_KEY",
    "",
    base_dir=REPO_ROOT,
)
openai_api_key = get_config_val(
    model_config,
    "api_keys",
    "openai_api_key",
    "OPENAI_API_KEY",
    "",
    base_dir=REPO_ROOT,
)

set_default_runtime_context(
    RuntimeContext(
        provider="",
        api_key="",
        base_url=evolink_base_url,
        event_hook=runtime_event_hook,
        status_hook=runtime_status_hook,
        gemini_client=_create_gemini_client(gemini_api_key),
        anthropic_client=_create_anthropic_client(anthropic_api_key),
        openai_client=_create_openai_client(openai_api_key),
        evolink_provider=_create_evolink_provider(evolink_api_key, evolink_base_url),
        owns_evolink_provider=bool(evolink_api_key),
    )
)

if get_default_runtime_context() and get_default_runtime_context().evolink_provider:
    logger.debug("默认 Evolink Provider 已初始化 (base_url=%s)", evolink_base_url)
else:
    logger.debug("未配置 Evolink API Key；仅当选择 evolink provider 时才会影响运行。")
if get_default_runtime_context() and get_default_runtime_context().gemini_client:
    logger.debug("默认 Gemini Client 已初始化")
if get_default_runtime_context() and get_default_runtime_context().anthropic_client:
    logger.debug("默认 Anthropic Client 已初始化")
if get_default_runtime_context() and get_default_runtime_context().openai_client:
    logger.debug("默认 OpenAI Client 已初始化")


def _cleanup_evolink_provider():
    """Shut down default Evolink provider session on process exit."""
    provider = get_default_runtime_context().evolink_provider if get_default_runtime_context() else None
    if provider is not None:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(provider.close())
            else:
                loop.run_until_complete(provider.close())
        except Exception:
            pass


import atexit
atexit.register(_cleanup_evolink_provider)


def init_evolink_provider(api_key: str, base_url: str = ""):
    """用指定的 API Key 初始化或更新默认 Evolink Provider（保留兼容性）。"""
    if not api_key:
        return
    context = get_default_runtime_context() or RuntimeContext()
    context.provider = "evolink"
    context.api_key = api_key
    context.base_url = base_url or evolink_base_url
    context.evolink_provider = _create_evolink_provider(api_key, context.base_url)
    context.owns_evolink_provider = context.evolink_provider is not None
    set_default_runtime_context(context)
    if context.evolink_provider is not None:
        logger.debug("运行时 Evolink Provider 已初始化")


def init_gemini_client(api_key: str):
    """用指定的 API Key 初始化或更新默认 Gemini Client（保留兼容性）。"""
    if not api_key:
        return
    context = get_default_runtime_context() or RuntimeContext()
    context.provider = "gemini"
    context.api_key = api_key
    context.gemini_client = _create_gemini_client(api_key)
    set_default_runtime_context(context)
    if context.gemini_client is not None:
        logger.debug("运行时 Gemini Client 已初始化")


# ==================== Evolink 调用函数 ====================

async def call_evolink_text_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    通过 Evolink Provider 进行文本生成。

    Args:
        model_name: 模型名称（如 "gemini-2.5-flash"）
        contents: 通用内容列表
        config: 配置字典或对象，需包含 system_instruction, temperature, max_output_tokens
        max_attempts: 最大重试次数
        retry_delay: 重试间隔
        error_context: 错误上下文
    """
    provider = get_evolink_provider()
    logger.debug(f"📤 call_evolink_text: model={model_name}, provider={'已初始化' if provider else '未初始化'}")
    if provider is None:
        raise RuntimeError("Evolink Provider 未初始化，请检查 EVOLINK_API_KEY 配置。")

    # 从 config 中提取参数（兼容 types.GenerateContentConfig 和 dict）
    if hasattr(config, 'system_instruction'):
        system_prompt = config.system_instruction or ""
        temperature = config.temperature
        max_output_tokens = config.max_output_tokens
        logger.debug("📋 call_evolink_text: 从 GenerateContentConfig 提取参数")
    elif isinstance(config, dict):
        system_prompt = config.get("system_prompt", "")
        temperature = config.get("temperature", 1.0)
        max_output_tokens = config.get("max_output_tokens", 50000)
        logger.debug("📋 call_evolink_text: 从 dict 提取参数")
    else:
        system_prompt = ""
        temperature = 1.0
        max_output_tokens = 50000
        logger.debug(f"📋 call_evolink_text: 使用默认参数, config type={type(config)}")

    return await provider.generate_text(
        model_name=model_name,
        contents=contents,
        system_prompt=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


async def upload_image_to_evolink(image_b64: str, media_type: str = "image/jpeg") -> str:
    """
    将 base64 图片上传到 Evolink 文件服务，返回可访问的 URL。

    用于 image-to-image 场景（如 Polish Agent），需要先把本地 base64 图片
    上传为 URL，才能传给图像生成 API 的 image_urls 参数。
    """
    provider = get_evolink_provider()
    if provider is None:
        raise RuntimeError("Evolink Provider 未初始化，请检查 EVOLINK_API_KEY 配置。")
    url = await provider.upload_image_base64(image_b64, media_type)
    if not url:
        raise RuntimeError("图片上传到 Evolink 文件服务失败")
    return url


async def call_evolink_image_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    通过 Evolink Provider 进行图像生成。

    Args:
        model_name: 图像模型名称（如 "nano-banana-2-lite"，通过 /v1/images/generations）
        prompt: 图像描述提示词
        config: 配置字典，需包含 aspect_ratio, quality 等
        max_attempts: 最大重试次数
        retry_delay: 重试间隔
        error_context: 错误上下文
    """
    provider = get_evolink_provider()
    logger.debug(f"🖼️ call_evolink_image: model={model_name}, config={config}, provider={'已初始化' if provider else '未初始化'}")
    if provider is None:
        raise RuntimeError("Evolink Provider 未初始化，请检查 EVOLINK_API_KEY 配置。")

    aspect_ratio = config.get("aspect_ratio", "16:9")
    quality = config.get("quality", "2K")
    image_urls = config.get("image_urls", None)

    return await provider.generate_image(
        model_name=model_name,
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        quality=quality,
        image_urls=image_urls,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )


# ==================== 原始 Gemini 调用函数（保留兼容性） ====================

def _convert_to_gemini_parts(contents):
    """将通用内容列表转换为 Gemini 的 Part 对象列表"""
    from google.genai import types
    gemini_parts = []
    for item in contents:
        if item.get("type") == "text":
            gemini_parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
    return gemini_parts


def _is_gemini_image_request(model_name: str, config: Any) -> bool:
    """判断当前是否为图像生成请求。"""
    lower_model = (model_name or "").lower()
    if "image" in lower_model or "nanoviz" in lower_model:
        return True

    modalities = getattr(config, "response_modalities", None)
    if modalities:
        for mod in modalities:
            if str(mod).upper() == "IMAGE":
                return True
    return False


def _build_gemini_model_ladder(
    model_name: str,
    *,
    is_image_request: bool,
    image_fallback_model: str = DEFAULT_GEMINI_IMAGE_FALLBACK_MODEL,
) -> List[str]:
    """Build an ordered retry ladder from expensive/fragile models to cheaper/stabler ones."""
    normalized = str(model_name or "").strip()
    lower_model = normalized.lower()
    ladder: List[str] = []

    def _push(name: str) -> None:
        candidate = str(name or "").strip()
        if candidate and candidate not in ladder:
            ladder.append(candidate)

    _push(normalized)
    if is_image_request:
        if "pro-image" in lower_model:
            _push(image_fallback_model)
        elif "flash-image" not in lower_model:
            _push(image_fallback_model)
    else:
        if "gemini-3.1-pro" in lower_model or "gemini-3-pro" in lower_model:
            for fallback_model in DEFAULT_GEMINI_TEXT_FALLBACK_MODELS:
                _push(fallback_model)
        elif "flash-lite" in lower_model:
            _push("gemini-3-flash-preview")

    return ladder


def _should_retry_gemini_forever(error_text: str) -> bool:
    """Retry indefinitely only for provider instability and capacity issues."""
    lower = str(error_text or "").lower()
    retry_signals = [
        "503 unavailable",
        "high demand",
        "resource_exhausted",
        "quota exceeded",
        "not found",
        "empty candidates",
        "empty text response",
        "empty image candidates",
        "empty image response",
        "timed out",
        "timeout",
        "deadline exceeded",
        "connection reset",
        "internal",
        "service unavailable",
        "unavailable",
        "too many requests",
        "rate limit",
    ]
    return any(sig in lower for sig in retry_signals)


def _is_gemini_non_retryable_error(error_text: str) -> bool:
    """Avoid infinite loops on auth/config/safety/input failures."""
    lower = str(error_text or "").lower()
    non_retryable_signals = [
        "api key not valid",
        "invalid api key",
        "permission denied",
        "unauthenticated",
        "forbidden",
        "invalid argument",
        "bad request",
        "400 bad request",
        "401 unauthorized",
        "403 forbidden",
        "safety",
        "blocked",
        "unsupported",
    ]
    return any(sig in lower for sig in non_retryable_signals)


def _stage_retry_budget(
    *,
    stage_model_name: str,
    primary_model_name: str,
    is_image_request: bool,
    cycle_index: int,
    requested_attempts: int,
) -> int:
    """Balance persistence with API cost by demoting flaky pro models after the first cycle."""
    safe_requested = max(1, int(requested_attempts or 1))
    lower_stage = str(stage_model_name or "").lower()
    lower_primary = str(primary_model_name or "").lower()
    is_primary = lower_stage == lower_primary

    if is_image_request:
        if "pro-image" in lower_stage:
            if cycle_index == 0:
                return min(2, safe_requested)
            return 1 if cycle_index % 4 == 0 else 0
        if "flash-image" in lower_stage:
            return min(max(2, safe_requested), 4)
        return min(max(2, safe_requested), 3)

    if "3.1-pro" in lower_stage or ("3-pro" in lower_stage and "image" not in lower_stage):
        if cycle_index == 0:
            return min(2, safe_requested)
        return 1 if cycle_index % 3 == 0 else 0
    if "flash-lite" in lower_stage:
        return min(max(2, safe_requested), 3)
    if "flash-preview" in lower_stage:
        return min(max(2, safe_requested + 1), 4)

    if is_primary:
        return min(2, safe_requested)
    return min(max(2, safe_requested), 3)


def _compute_cycle_cooldown_seconds(last_error_text: str, retry_delay: float, cycle_index: int) -> float:
    base_delay = _compute_retry_delay_seconds(
        error_text=last_error_text,
        retry_delay=retry_delay,
        attempt=min(cycle_index, 4),
    )
    if _is_gemini_permanent_quota_block(last_error_text):
        long_cooldown = min(300.0 * max(1, cycle_index + 1), 1800.0)
        return float(max(base_delay, long_cooldown))
    return float(min(max(base_delay, retry_delay), 60.0))


def _should_try_text_fallback(error_text: str) -> bool:
    """是否应触发文本模型兜底。"""
    lower = error_text.lower()
    fallback_signals = [
        "503 unavailable",
        "high demand",
        "resource_exhausted",
        "quota exceeded",
        "404",
        "not found",
        "empty candidates",
        "empty text response",
        "nonetype' object is not iterable",
        "timed out",
        "timeout",
    ]
    return any(sig in lower for sig in fallback_signals)


def _is_gemini_permanent_quota_block(error_text: str) -> bool:
    """判断是否是基本不可恢复的配额阻塞（当前会话内重试价值很低）。"""
    lower = error_text.lower()
    if "resource_exhausted" not in lower and "quota exceeded" not in lower:
        return False
    if "limit: 0" in lower:
        return True
    if "generaterequestsperday" in lower or "generatecontentinputtokenspermodelperday" in lower:
        return True
    return False


def _compute_retry_delay_seconds(error_text: str, retry_delay: float, attempt: int) -> float:
    """基于指数退避与服务端建议计算最终等待时长。"""
    local_backoff = float(min(retry_delay * (2 ** attempt), 30))
    lower = error_text.lower()
    delay_candidates: List[float] = []

    for m in re.findall(r"retrydelay[\"']?\s*[:=]\s*[\"']([\d.]+)s[\"']", error_text, flags=re.IGNORECASE):
        try:
            delay_candidates.append(float(m))
        except ValueError:
            pass

    for m in re.findall(r"please retry in\s+([\d.]+)s", lower):
        try:
            delay_candidates.append(float(m))
        except ValueError:
            pass

    if not delay_candidates:
        return local_backoff
    # 兼顾服务端建议，但避免超长等待
    return float(min(max(local_backoff, max(delay_candidates)), 90.0))


def _parse_gemini_error_metadata(error_text: str) -> Dict[str, Any]:
    """Extract structured metadata from Gemini exception text."""
    lower = (error_text or "").lower()
    code = None
    status = None

    code_match = re.search(r"\b(\d{3})\s+[A-Z_]+\b", error_text or "")
    if code_match:
        try:
            code = int(code_match.group(1))
        except Exception:
            code = None
    if code is None:
        code_match = re.search(r"[\"']code[\"']\s*:\s*(\d+)", error_text or "")
        if code_match:
            try:
                code = int(code_match.group(1))
            except Exception:
                code = None

    status_match = re.search(r"\b\d{3}\s+([A-Z_]+)\b", error_text or "")
    if status_match:
        status = status_match.group(1)
    if not status:
        status_match = re.search(r"[\"']status[\"']\s*:\s*[\"']([A-Z_]+)[\"']", error_text or "")
        if status_match:
            status = status_match.group(1)

    retry_delay = None
    for m in re.findall(r"please retry in\s+([\d.]+)s", lower):
        try:
            retry_delay = float(m)
            break
        except ValueError:
            pass
    if retry_delay is None:
        for m in re.findall(r"retrydelay[\"']?\s*[:=]\s*[\"']([\d.]+)s[\"']", error_text or "", flags=re.IGNORECASE):
            try:
                retry_delay = float(m)
                break
            except ValueError:
                pass

    is_quota = (
        code == 429
        or "resource_exhausted" in lower
        or "quota exceeded" in lower
    )
    is_limit_zero = "limit: 0" in lower

    return {
        "code": code,
        "status": status,
        "is_quota": is_quota,
        "is_limit_zero": is_limit_zero,
        "retry_delay_hint": retry_delay,
    }


def _build_retry_status_line(
    *,
    stage: str,
    model: str,
    attempt: int,
    max_attempts: int,
    error_code: Optional[int],
    error_status: Optional[str],
    retry_delay: float,
    error_context: str,
) -> str:
    """Create concise retry log line for demo realtime status."""
    code_str = str(error_code) if error_code is not None else "-"
    status_str = error_status or "-"
    context_suffix = f" | {error_context}" if error_context else ""
    return (
        f"[RETRY] stage={stage} model={model} attempt={attempt}/{max_attempts} "
        f"code={code_str} status={status_str} wait={retry_delay:.1f}s{context_suffix}"
    )


def _get_gemini_request_timeout_seconds(is_image_request: bool) -> float:
    """获取 Gemini 单次请求超时，避免请求长时间卡住。"""
    if is_image_request:
        env_val = os.getenv("GEMINI_IMAGE_TIMEOUT_SEC", "").strip()
        if env_val:
            try:
                return max(float(env_val), 30.0)
            except ValueError:
                pass
        return 240.0

    env_val = os.getenv("GEMINI_TEXT_TIMEOUT_SEC", "").strip()
    if env_val:
        try:
            return max(float(env_val), 10.0)
        except ValueError:
            pass
    return 45.0


async def call_gemini_with_retry_async(
    model_name,
    contents,
    config,
    max_attempts=5,
    retry_delay=5,
    error_context="",
    image_fallback_model: str = DEFAULT_GEMINI_IMAGE_FALLBACK_MODEL,
    image_fallback_max_attempts: int = 5,
):
    """Gemini 调用：激进并发场景下优先降级，再对可恢复错误做可取消的无限重试。"""
    if get_gemini_client() is None:
        raise RuntimeError("Gemini Client 未初始化，请检查 Google API Key。")

    result_list: List[str] = []
    target_candidate_count = int(getattr(config, "candidate_count", 1) or 1)
    current_candidate_count = getattr(config, "candidate_count", None) if hasattr(config, "candidate_count") else None
    if isinstance(current_candidate_count, int) and current_candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    is_image_request = _is_gemini_image_request(model_name, config)
    request_timeout_seconds = _get_gemini_request_timeout_seconds(is_image_request)
    model_ladder = _build_gemini_model_ladder(
        model_name,
        is_image_request=is_image_request,
        image_fallback_model=image_fallback_model,
    )

    async def _run_stage(
        stage_name: str,
        stage_model_name: str,
        stage_max_attempts: int,
    ) -> Tuple[List[str], Dict[str, Any], bool]:
        """Run one retry stage and return (results, last_error_meta, success)."""
        stage_attempts = max(1, int(stage_max_attempts))
        stage_results: List[str] = []
        last_error_meta: Dict[str, Any] = {}

        for attempt_idx in range(stage_attempts):
            if _runtime_cancel_requested():
                raise asyncio.CancelledError()
            try:
                client = get_gemini_client()
                if client is None:
                    raise RuntimeError("Gemini Client 未初始化，请检查 Google API Key。")
                gemini_contents = _convert_to_gemini_parts(current_contents)
                _emit_runtime_event(
                    level="INFO",
                    kind="job",
                    source="GenerationUtils",
                    job_type="generation",
                    provider="gemini",
                    model=stage_model_name,
                    attempt=attempt_idx + 1,
                    stage=stage_name,
                    status="running",
                    message=(
                        f"Gemini 请求开始：stage={stage_name} "
                        f"attempt={attempt_idx + 1}/{stage_attempts}"
                    ),
                    details=error_context,
                )
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=stage_model_name,
                        contents=gemini_contents,
                        config=config,
                    ),
                    timeout=request_timeout_seconds,
                )

                raw_response_list: List[str] = []
                if _is_gemini_image_request(stage_model_name, config):
                    if not response.candidates or not response.candidates[0].content.parts:
                        raise RuntimeError(f"Gemini returned empty image candidates for model {stage_model_name}")
                    for part in response.candidates[0].content.parts:
                        if getattr(part, "inline_data", None) and part.inline_data.data:
                            raw_response_list.append(
                                base64.b64encode(part.inline_data.data).decode("utf-8")
                            )
                            break
                    if not raw_response_list:
                        raise RuntimeError(f"Gemini returned empty image response for model {stage_model_name}")
                else:
                    if not response.candidates:
                        raise RuntimeError(f"Gemini returned empty candidates for model {stage_model_name}")

                    for candidate in response.candidates:
                        if not candidate.content or not candidate.content.parts:
                            continue
                        for part in candidate.content.parts:
                            part_text = getattr(part, "text", None)
                            if part_text:
                                raw_response_list.append(part_text)
                    if not raw_response_list:
                        raise RuntimeError(f"Gemini returned empty text response for model {stage_model_name}")

                stage_results.extend([r for r in raw_response_list if isinstance(r, str) and r.strip() != ""])
                if len(stage_results) >= target_candidate_count:
                    return stage_results[:target_candidate_count], {}, True

            except Exception as e:
                if isinstance(e, asyncio.TimeoutError):
                    error_text = (
                        f"Gemini request timed out after {request_timeout_seconds:.0f}s "
                        f"for model {stage_model_name}"
                    )
                else:
                    error_text = str(e)

                parsed_meta = _parse_gemini_error_metadata(error_text)
                current_delay = _compute_retry_delay_seconds(
                    error_text=error_text,
                    retry_delay=retry_delay,
                    attempt=attempt_idx,
                )
                retry_line = _build_retry_status_line(
                    stage=stage_name,
                    model=stage_model_name,
                    attempt=attempt_idx + 1,
                    max_attempts=stage_attempts,
                    error_code=parsed_meta.get("code"),
                    error_status=parsed_meta.get("status"),
                    retry_delay=current_delay,
                    error_context=error_context,
                )
                _emit_runtime_event(
                    level="WARNING",
                    kind="retry",
                    source="GenerationUtils",
                    job_type="generation",
                    provider="gemini",
                    model=stage_model_name,
                    attempt=attempt_idx + 1,
                    stage=stage_name,
                    status="retrying",
                    error_code=parsed_meta.get("code"),
                    message=retry_line,
                    details=error_text,
                )
                _safe_log(f"[Gemini] {retry_line} | error={_safe_text_for_log(error_text, max_len=1200)}")

                last_error_meta = {
                    "stage": stage_name,
                    "model": stage_model_name,
                    "attempt": attempt_idx + 1,
                    "max_attempts": stage_attempts,
                    "error_text": _safe_text_for_log(error_text, max_len=3000),
                    **parsed_meta,
                }

                if attempt_idx < stage_attempts - 1:
                    if _runtime_cancel_requested():
                        raise asyncio.CancelledError()
                    await asyncio.sleep(current_delay)
                else:
                    _emit_runtime_event(
                        level="WARNING",
                        kind="warning",
                        source="GenerationUtils",
                        job_type="generation",
                        provider="gemini",
                        model=stage_model_name,
                        stage=stage_name,
                        status="rotate",
                        message=(
                            f"Gemini 当前阶段重试已耗尽：stage={stage_name} "
                            f"model={stage_model_name} attempts={stage_attempts}"
                        ),
                    )

        return stage_results[:target_candidate_count], last_error_meta, len(stage_results) >= target_candidate_count

    final_error_meta: Dict[str, Any] = {}
    cycle_index = 0

    while True:
        for ladder_index, stage_model_name in enumerate(model_ladder):
            stage_requested_attempts = (
                int(max_attempts)
                if stage_model_name == model_name
                else int(image_fallback_max_attempts if is_image_request else max_attempts)
            )
            stage_attempt_budget = _stage_retry_budget(
                stage_model_name=stage_model_name,
                primary_model_name=model_name,
                is_image_request=is_image_request,
                cycle_index=cycle_index,
                requested_attempts=stage_requested_attempts,
            )
            if stage_attempt_budget <= 0:
                continue

            if stage_model_name != model_name or cycle_index > 0:
                _emit_runtime_event(
                    level="INFO",
                    kind="warning",
                    source="GenerationUtils",
                    job_type="generation",
                    provider="gemini",
                    model=stage_model_name,
                    status="fallback",
                    message=(
                        f"Gemini 模型降级：cycle={cycle_index + 1} "
                        f"step={ladder_index + 1}/{len(model_ladder)} model={stage_model_name}"
                    ),
                )

            stage_results, stage_error_meta, stage_success = await _run_stage(
                stage_name=f"cycle{cycle_index + 1}_step{ladder_index + 1}",
                stage_model_name=stage_model_name,
                stage_max_attempts=stage_attempt_budget,
            )
            if stage_success and stage_results:
                result_list = stage_results
                return result_list[:target_candidate_count]
            if stage_error_meta:
                final_error_meta = stage_error_meta
                error_text = str(stage_error_meta.get("error_text", ""))
                if _is_gemini_non_retryable_error(error_text):
                    _emit_runtime_event(
                        level="ERROR",
                        kind="error",
                        source="GenerationUtils",
                        job_type="generation",
                        provider="gemini",
                        model=stage_model_name,
                        status="failed",
                        error_code=stage_error_meta.get("code"),
                        message=(
                            f"Gemini 不可恢复错误：model={stage_model_name} "
                            f"code={stage_error_meta.get('code')} status={stage_error_meta.get('status')}"
                        ),
                        details=stage_error_meta.get("error_text", ""),
                    )
                    _safe_log(
                        f"[Gemini] 非可恢复错误 ({error_context}): {stage_error_meta.get('error_text', '')}"
                    )
                    result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
                    return result_list

        if _runtime_cancel_requested():
            raise asyncio.CancelledError()

        last_error_text = str(final_error_meta.get("error_text", ""))
        if final_error_meta and not _should_retry_gemini_forever(last_error_text):
            _emit_runtime_event(
                level="ERROR",
                kind="error",
                source="GenerationUtils",
                job_type="generation",
                provider="gemini",
                model=final_error_meta.get("model", ""),
                stage=final_error_meta.get("stage", ""),
                status="failed",
                error_code=final_error_meta.get("code"),
                message=(
                    "Gemini 停止重试："
                    f"last_stage={final_error_meta.get('stage')} "
                    f"model={final_error_meta.get('model')}"
                ),
                details=final_error_meta.get("error_text", ""),
            )
            _safe_log(
                f"[Gemini] 停止重试 ({error_context}): {final_error_meta.get('error_text', '')}"
            )
            result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
            return result_list

        cycle_delay = _compute_cycle_cooldown_seconds(last_error_text, retry_delay, cycle_index)
        _emit_runtime_event(
            level="WARNING",
            kind="retry",
            source="GenerationUtils",
            job_type="generation",
            provider="gemini",
            model=model_name,
            status="retrying",
            message=(
                f"Gemini 阶梯模型已轮询完成，{cycle_delay:.1f}s 后继续重试 "
                f"(cycle={cycle_index + 1}, ladder={','.join(model_ladder)})"
            ),
            details=last_error_text,
        )
        if _runtime_cancel_requested():
            raise asyncio.CancelledError()
        await asyncio.sleep(cycle_delay)
        cycle_index += 1


# ==================== 原始 Claude/OpenAI 调用函数（保留兼容性） ====================

def _convert_to_claude_format(contents):
    return contents

def _convert_to_openai_format(contents):
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            data = ""
            media_type = "image/jpeg"
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
            elif item.get("image_base64"):
                media_type = item.get("mime_type", "image/jpeg")
                data = item.get("image_base64", "")
            if data:
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
    return openai_contents


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """原始 Claude API 异步调用（保留兼容性）"""
    client = get_anthropic_client()
    if client is None:
        raise RuntimeError("Anthropic Client 未初始化，请检查 ANTHROPIC_API_KEY。")
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []

    current_contents = contents
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            claude_contents = _convert_to_claude_format(current_contents)
            first_response = await client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break
        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            _emit_runtime_event(
                level="WARNING",
                kind="retry",
                source="GenerationUtils",
                job_type="generation",
                provider="anthropic",
                model=model_name,
                attempt=attempt + 1,
                status="retrying",
                message=f"Anthropic 验证失败，第 {attempt + 1}/{max_attempts} 次重试前等待 {retry_delay}s",
                details=f"{context_msg}: {error_str}",
            )
            logger.warning(
                "Anthropic 验证第 %s 次尝试失败%s，%ss 后重试",
                attempt + 1,
                context_msg,
                retry_delay,
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    if not is_input_valid:
        _emit_runtime_event(
            level="ERROR",
            kind="error",
            source="GenerationUtils",
            job_type="generation",
            provider="anthropic",
            model=model_name,
            status="failed",
            message=f"Anthropic 在 {max_attempts} 次尝试后仍然失败",
        )
        logger.error("Anthropic 全部 %s 次验证尝试失败，返回错误", max_attempts)
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": valid_claude_contents}],
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list


async def call_openai_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """原始 OpenAI API 异步调用（保留兼容性）"""
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI Client 未初始化，请检查 OPENAI_API_KEY。")
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    current_contents = contents
    is_input_valid = False
    last_exception: Exception | None = None
    last_error_text = ""
    for attempt in range(max_attempts):
        try:
            openai_contents = _convert_to_openai_format(current_contents)
            first_response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            response_text_list.append(first_response.choices[0].message.content)
            is_input_valid = True
            break
        except Exception as e:
            last_exception = e
            last_error_text = str(e)
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            _emit_runtime_event(
                level="WARNING",
                kind="retry",
                source="GenerationUtils",
                job_type="generation",
                provider="openai",
                model=model_name,
                attempt=attempt + 1,
                status="retrying",
                message=f"OpenAI 验证失败，第 {attempt + 1}/{max_attempts} 次重试前等待 {retry_delay}s",
                details=f"{context_msg}: {error_str}",
            )
            logger.warning(
                "OpenAI 验证第 %s 次尝试失败%s，%ss 后重试",
                attempt + 1,
                context_msg,
                retry_delay,
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    if not is_input_valid:
        context_msg = f" for {error_context}" if error_context else ""
        failure_message = f"OpenAI 文本生成在 {max_attempts} 次尝试后仍然失败{context_msg}"
        if last_error_text:
            failure_message = f"{failure_message}: {last_error_text}"
        _emit_runtime_event(
            level="ERROR",
            kind="error",
            source="GenerationUtils",
            job_type="generation",
            provider="openai",
            model=model_name,
            status="failed",
            message=f"OpenAI 在 {max_attempts} 次尝试后仍然失败",
            details=failure_message,
        )
        logger.error("OpenAI 全部 %s 次验证尝试失败，抛出异常", max_attempts)
        raise RuntimeError(failure_message) from last_exception

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content)

    return response_text_list


def _extract_base64_from_data_url(data_url: str) -> str:
    """从 data URL 中提取纯 base64 字符串。"""
    value = str(data_url or "").strip()
    if not value:
        return ""
    if value.startswith("data:") and ";base64," in value:
        return value.split(";base64,", 1)[1]
    return value


def _extract_openai_input_images(contents: Optional[List[Dict[str, Any]]] = None) -> list[tuple[bytes, str, str]]:
    images: list[tuple[bytes, str, str]] = []
    for index, item in enumerate(contents or []):
        if item.get("type") != "image":
            continue
        source = item.get("source", {}) or {}
        data = ""
        media_type = "image/png"
        if source.get("type") == "base64":
            data = str(source.get("data", "") or "")
            media_type = str(source.get("media_type", "image/png") or "image/png")
        elif item.get("image_base64"):
            data = str(item.get("image_base64", "") or "")
            media_type = str(item.get("mime_type", "image/png") or "image/png")
        if not data:
            continue
        suffix = media_type.split("/", 1)[-1].split(";", 1)[0] or "png"
        try:
            images.append((base64.b64decode(data), media_type, f"reference_{index}.{suffix}"))
        except Exception:
            logger.warning("OpenAI 图像编辑跳过无法解码的参考图 index=%s", index)
    return images


def _iter_openai_image_payloads(response: Any) -> list[Any]:
    payloads: list[Any] = []
    for attr_name in ("data", "output", "images"):
        values = getattr(response, attr_name, None)
        if values:
            payloads.extend(list(values))
    if isinstance(response, dict):
        for key in ("data", "output", "images"):
            values = response.get(key)
            if values:
                payloads.extend(list(values if isinstance(values, list) else [values]))
    if not payloads:
        payloads.append(response)
    return payloads


def _extract_url_field(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        value = payload.strip()
        if value.startswith("data:") or value.startswith("http://") or value.startswith("https://"):
            return value
        return ""
    if isinstance(payload, dict):
        for key in ("url", "image_url", "file_url"):
            value = payload.get(key)
            if isinstance(value, dict):
                nested_url = _extract_url_field(value)
                if nested_url:
                    return nested_url
            elif isinstance(value, str):
                nested_url = _extract_url_field(value)
                if nested_url:
                    return nested_url
        for key in ("image", "content", "result"):
            nested_url = _extract_url_field(payload.get(key))
            if nested_url:
                return nested_url
        return ""
    for key in ("url", "image_url", "file_url", "image", "content", "result"):
        nested_url = _extract_url_field(getattr(payload, key, None))
        if nested_url:
            return nested_url
    if hasattr(payload, "model_extra"):
        return _extract_url_field(getattr(payload, "model_extra", None) or {})
    return ""


async def _fetch_image_url_as_base64(url: str) -> str:
    normalized_url = str(url or "").strip()
    if not normalized_url:
        return ""
    if normalized_url.startswith("data:"):
        return _extract_base64_from_data_url(normalized_url)
    if not (normalized_url.startswith("http://") or normalized_url.startswith("https://")):
        return ""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(normalized_url)
            response.raise_for_status()
            content_type = str(response.headers.get("content-type", "") or "").lower()
            if content_type and not content_type.startswith("image/"):
                logger.warning("OpenAI 图像 URL 返回非图像内容：content-type=%s", _safe_text_for_log(content_type))
            return base64.b64encode(response.content).decode("utf-8")
    except Exception as exc:
        logger.warning("OpenAI 图像 URL 下载失败：%s", _safe_text_for_log(exc))
        return ""


async def _extract_openai_response_images(response: Any) -> List[str]:
    response_images: List[str] = []
    seen_values: set[str] = set()
    pending_urls: list[str] = []

    for item in _iter_openai_image_payloads(response):
        b64_json = _extract_base64_field(item)
        if b64_json and not b64_json.startswith(("http://", "https://")) and b64_json not in seen_values:
            response_images.append(str(b64_json))
            seen_values.add(str(b64_json))
            continue
        image_url = _extract_url_field(item)
        if image_url and image_url not in pending_urls:
            pending_urls.append(image_url)

    for image_url in pending_urls:
        fetched_b64 = await _fetch_image_url_as_base64(image_url)
        if fetched_b64 and fetched_b64 not in seen_values:
            response_images.append(fetched_b64)
            seen_values.add(fetched_b64)

    return response_images


def _extract_base64_field(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return _extract_base64_from_data_url(payload)
    if isinstance(payload, dict):
        for key in ("b64_json", "result", "image_base64", "partial_image_b64", "partial_image"):
            value = payload.get(key)
            extracted = _extract_base64_field(value)
            if extracted:
                return extracted
        return ""
    for key in ("b64_json", "result", "image_base64", "partial_image_b64", "partial_image"):
        value = getattr(payload, key, None)
        extracted = _extract_base64_field(value)
        if extracted:
            return extracted
    if hasattr(payload, "model_extra"):
        return _extract_base64_field(getattr(payload, "model_extra", None) or {})
    return ""


def _extract_candidate_id_from_context(error_context: str = "") -> str:
    match = re.search(r"candidate=([^,\]\)]+)", str(error_context or ""))
    return match.group(1).strip() if match else ""


def _extract_preview_slot_from_context(error_context: str = "") -> str:
    candidate_id = _extract_candidate_id_from_context(error_context)
    if candidate_id:
        return candidate_id
    task_match = re.search(r"task#([A-Za-z0-9_\-]+)", str(error_context or ""))
    if task_match:
        return f"task#{task_match.group(1).strip()}"
    return ""


def _job_type_from_context(error_context: str = "") -> str:
    return "refine" if str(error_context or "").startswith("refine-image") else "generation"


async def _extract_openai_stream_images(
    stream: Any,
    *,
    provider: str = "openai",
    model_name: str = "",
    error_context: str = "",
    job_type: str = "generation",
    preview_label_prefix: str = "OpenAI",
) -> List[str]:
    final_images: List[str] = []
    partial_images: List[str] = []
    candidate_id = _extract_preview_slot_from_context(error_context)
    async for event in stream:
        event_type = str(
            getattr(event, "type", "")
            or (event.get("type", "") if isinstance(event, dict) else "")
            or ""
        )
        image_b64 = _extract_base64_field(event)
        if image_b64 and image_b64.startswith(("http://", "https://")):
            image_b64 = await _fetch_image_url_as_base64(image_b64)
        if not image_b64:
            image_url = _extract_url_field(event)
            if image_url:
                image_b64 = await _fetch_image_url_as_base64(image_url)
        if not image_b64:
            continue
        if "partial" in event_type:
            partial_images.append(image_b64)
            _emit_runtime_event(
                level="INFO",
                kind="preview_ready",
                source="GenerationUtils",
                job_type=job_type,
                candidate_id=candidate_id,
                stage=f"{preview_label_prefix} partial image",
                status="running",
                provider=provider,
                model=model_name,
                preview_image=image_b64,
                preview_mime_type="image/png",
                preview_label=f"{preview_label_prefix} 流式预览 {len(partial_images)}",
                message=f"{preview_label_prefix} 图像流式预览已更新（slot={candidate_id or 'active'}, partial={len(partial_images)}）",
            )
        else:
            final_images.append(image_b64)
            _emit_runtime_event(
                level="INFO",
                kind="preview_ready",
                source="GenerationUtils",
                job_type=job_type,
                candidate_id=candidate_id,
                stage=f"{preview_label_prefix} final image",
                status="running",
                provider=provider,
                model=model_name,
                preview_image=image_b64,
                preview_mime_type="image/png",
                preview_label=f"{preview_label_prefix} 最终图像",
                message=f"{preview_label_prefix} 图像流式最终图已到达（slot={candidate_id or 'active'}）",
            )
    return final_images or partial_images


def _is_openai_image_retryable_error(exc: Exception) -> bool:
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            NotFoundError,
            PermissionDeniedError,
            RateLimitError,
        )
    except Exception:  # pragma: no cover
        APIConnectionError = APIStatusError = APITimeoutError = AuthenticationError = BadRequestError = NotFoundError = PermissionDeniedError = RateLimitError = tuple()  # type: ignore

    message = str(exc or "").lower()
    if AuthenticationError and isinstance(exc, AuthenticationError):
        return False
    if BadRequestError and isinstance(exc, BadRequestError):
        return False
    if NotFoundError and isinstance(exc, NotFoundError):
        return False
    if PermissionDeniedError and isinstance(exc, PermissionDeniedError):
        return False
    if APITimeoutError and isinstance(exc, APITimeoutError):
        return True
    if APIConnectionError and isinstance(exc, APIConnectionError):
        return True
    if RateLimitError and isinstance(exc, RateLimitError):
        if any(token in message for token in ("insufficient_quota", "insufficient quota", "credit", "billing")):
            return False
        return True
    status_code = 0
    if APIStatusError and isinstance(exc, APIStatusError):
        status_code = int(getattr(exc, "status_code", 0) or 0)
    elif hasattr(exc, "status_code"):
        try:
            status_code = int(getattr(exc, "status_code", 0) or 0)
        except Exception:
            status_code = 0
    if status_code in {400, 401, 403, 404}:
        return False
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    if any(token in message for token in ("invalid_request", "model_not_found", "not found", "unsupported parameter")):
        return False
    if any(token in message for token in ("timeout", "timed out", "connection", "rate limit", "too many requests", "server error", "temporarily unavailable")):
        return True
    return True


def _should_try_openai_responses_fallback(
    exc: Exception,
    *,
    provider_type: str,
    model_name: str,
    fallback_mode: str,
) -> bool:
    if str(fallback_mode or "auto").lower() == "never":
        return False
    provider = str(provider_type or "").strip().lower()
    model = str(model_name or "").strip().lower()
    if provider not in {"openai", "openai_compatible"}:
        return False
    if not (model.startswith("gpt-image") or provider == "openai_compatible"):
        return False
    if str(fallback_mode or "").lower() == "always":
        return True
    message = str(exc or "").lower()
    status_code = 0
    try:
        status_code = int(getattr(exc, "status_code", 0) or 0)
    except Exception:
        status_code = 0
    if status_code in {401, 403}:
        return False
    if any(token in message for token in ("invalid api key", "incorrect api key", "insufficient_quota", "insufficient quota", "credit", "billing")):
        return False
    return status_code in {400, 404, 408, 409, 425, 429, 500, 502, 503, 504} or any(
        token in message
        for token in (
            "unsupported parameter",
            "not supported",
            "not found",
            "upstream",
            "server error",
            "temporarily unavailable",
            "timeout",
        )
    )


def _build_openai_responses_image_tool(
    *,
    model_name: str,
    options: Any,
    capabilities: Any,
    edit: bool,
) -> dict[str, Any]:
    tool: dict[str, Any] = {
        "type": "image_generation",
        "model": model_name,
        "action": "edit" if edit else "generate",
        "size": options.size,
        "quality": options.quality,
    }
    if capabilities.model_family.startswith("gpt-image"):
        tool["background"] = options.background
        tool["output_format"] = options.output_format
        if capabilities.supports_moderation:
            tool["moderation"] = options.moderation
    if options.output_compression is not None:
        tool["output_compression"] = options.output_compression
    if edit and capabilities.supports_input_fidelity and options.input_fidelity != "auto":
        tool["input_fidelity"] = options.input_fidelity
    if options.stream and options.partial_images:
        tool["partial_images"] = options.partial_images
    return tool


def _build_openai_responses_image_input(
    *,
    prompt: str,
    input_images: list[tuple[bytes, str, str]],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": str(prompt or "")}]
    for image_bytes, media_type, _filename in input_images:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{image_b64}",
            }
        )
    return [{"role": "user", "content": content}]


async def _call_openai_responses_image_generation_async(
    *,
    client: Any,
    model_name: str,
    prompt: str,
    options: Any,
    capabilities: Any,
    input_images: list[tuple[bytes, str, str]],
    provider_type: str,
    error_context: str = "",
) -> List[str]:
    responses_model = str(options.responses_model or "").strip() or str(os.environ.get("OPENAI_RESPONSES_MODEL", "") or "").strip()
    if not responses_model:
        responses_model = "gpt-4.1-mini"
    tool = _build_openai_responses_image_tool(
        model_name=model_name,
        options=options,
        capabilities=capabilities,
        edit=bool(input_images),
    )
    response_params: dict[str, Any] = {
        "model": responses_model,
        "input": _build_openai_responses_image_input(prompt=prompt, input_images=input_images),
        "tools": [tool],
        "tool_choice": {"type": "image_generation"},
    }
    if options.stream:
        response_params["stream"] = True
    response = await client.responses.create(**response_params)
    if options.stream:
        return await _extract_openai_stream_images(
            response,
            provider=provider_type,
            model_name=model_name,
            error_context=error_context,
            job_type=_job_type_from_context(error_context),
            preview_label_prefix="OpenAI Responses",
        )
    return await _extract_openai_response_images(response)


def _extract_openrouter_message_images(message: Any) -> List[str]:
    """从 OpenRouter chat completion 消息中提取生成图像。"""
    if message is None:
        return []

    images = getattr(message, "images", None)
    if not images and hasattr(message, "model_extra"):
        model_extra = getattr(message, "model_extra", None) or {}
        if isinstance(model_extra, dict):
            images = model_extra.get("images")
    if not images and isinstance(message, dict):
        images = message.get("images")

    extracted_images: List[str] = []
    for item in images or []:
        image_payload = None
        if isinstance(item, dict):
            image_payload = item.get("image_url") or item.get("imageUrl")
        else:
            image_payload = (
                getattr(item, "image_url", None)
                or getattr(item, "imageUrl", None)
                or (
                    getattr(item, "model_extra", None) or {}
                ).get("image_url")
                or (
                    getattr(item, "model_extra", None) or {}
                ).get("imageUrl")
            )

        image_url = None
        if isinstance(image_payload, dict):
            image_url = image_payload.get("url")
        elif image_payload is not None:
            image_url = getattr(image_payload, "url", None)
            if image_url is None and hasattr(image_payload, "model_extra"):
                image_url = (getattr(image_payload, "model_extra", None) or {}).get("url")

        image_b64 = _extract_base64_from_data_url(str(image_url or ""))
        if image_b64:
            extracted_images.append(image_b64)

    return extracted_images


def _openrouter_image_modalities_for_model(model_name: str, config: dict[str, Any]) -> List[str]:
    explicit_modalities = config.get("modalities")
    if explicit_modalities:
        return [str(item).strip() for item in explicit_modalities if str(item).strip()]

    normalized_model = str(model_name or "").strip().lower()
    if "gemini" in normalized_model:
        return ["image", "text"]
    return ["image"]


async def call_openrouter_image_generation_with_retry_async(
    model_name,
    prompt,
    config,
    *,
    contents: Optional[List[Dict[str, Any]]] = None,
    system_prompt: str = "",
    max_attempts=5,
    retry_delay=30,
    error_context="",
):
    """通过 OpenRouter 的 chat/completions 多模态接口进行图像生成。"""
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OpenRouter Client 未初始化，请检查 OPENROUTER_API_KEY。")

    prompt_text = str(prompt or "").strip()
    request_contents = list(contents or [])
    if request_contents:
        normalized_contents: List[Dict[str, Any]] = []
        prompt_inserted = False
        for item in request_contents:
            if item.get("type") == "text" and not prompt_inserted:
                normalized_contents.append({"type": "text", "text": prompt_text or str(item.get("text", "") or "")})
                prompt_inserted = True
            else:
                normalized_contents.append(item)
        if not prompt_inserted:
            normalized_contents.insert(0, {"type": "text", "text": prompt_text})
    else:
        normalized_contents = [{"type": "text", "text": prompt_text}]

    messages: List[Dict[str, Any]] = []
    if str(system_prompt or "").strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})
    messages.append(
        {
            "role": "user",
            "content": _convert_to_openai_format(normalized_contents),
        }
    )

    image_config: Dict[str, Any] = {}
    aspect_ratio = str(config.get("aspect_ratio", "") or "").strip()
    image_size = str(config.get("image_size", "") or "").strip()
    if aspect_ratio:
        image_config["aspect_ratio"] = aspect_ratio
    if image_size:
        image_config["image_size"] = image_size
    if config.get("font_inputs"):
        image_config["font_inputs"] = config["font_inputs"]
    if config.get("super_resolution_references"):
        image_config["super_resolution_references"] = config["super_resolution_references"]

    extra_body: Dict[str, Any] = {
        "modalities": _openrouter_image_modalities_for_model(model_name, config),
    }
    if image_config:
        extra_body["image_config"] = image_config
    output_format = str(config.get("output_format", "") or "").strip()
    if output_format:
        extra_body["output_format"] = output_format

    last_exception: Exception | None = None
    last_error_text = ""
    for attempt in range(max_attempts):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False,
                extra_body=extra_body,
            )
            response_images: List[str] = []
            for choice in getattr(response, "choices", []) or []:
                message = getattr(choice, "message", None)
                response_images.extend(_extract_openrouter_message_images(message))
            if response_images:
                return response_images

            last_error_text = "OpenRouter 图像生成未返回图片字段"
            _emit_runtime_event(
                level="WARNING",
                kind="warning",
                source="GenerationUtils",
                job_type="generation",
                provider="openrouter",
                model=model_name,
                attempt=attempt + 1,
                status="retrying",
                message="OpenRouter 图像生成未返回图片字段，将继续重试",
            )
            logger.warning("OpenRouter 图像生成失败，未返回图片字段")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            continue
        except Exception as e:
            last_exception = e
            last_error_text = str(e)
            context_msg = f" for {error_context}" if error_context else ""
            _emit_runtime_event(
                level="WARNING",
                kind="retry",
                source="GenerationUtils",
                job_type="generation",
                provider="openrouter",
                model=model_name,
                attempt=attempt + 1,
                status="retrying",
                message=f"OpenRouter 图像生成第 {attempt + 1}/{max_attempts} 次尝试失败，{retry_delay}s 后重试",
                details=f"{context_msg}: {e}",
            )
            logger.warning(
                "OpenRouter 图像生成第 %s 次尝试失败%s，%ss 后重试",
                attempt + 1,
                context_msg,
                retry_delay,
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    context_msg = f" for {error_context}" if error_context else ""
    failure_message = f"OpenRouter 图像生成在 {max_attempts} 次尝试后仍然失败{context_msg}"
    if last_error_text:
        failure_message = f"{failure_message}: {last_error_text}"
    _emit_runtime_event(
        level="ERROR",
        kind="error",
        source="GenerationUtils",
        job_type="generation",
        provider="openrouter",
        model=model_name,
        status="failed",
        message=f"OpenRouter 图像生成在 {max_attempts} 次尝试后仍然失败",
        details=failure_message,
    )
    logger.error("OpenRouter 图像生成全部 %s 次尝试失败%s", max_attempts, context_msg)
    raise RuntimeError(failure_message) from last_exception


async def call_openai_image_generation_with_retry_async(
    model_name,
    prompt,
    config,
    *,
    contents: Optional[List[Dict[str, Any]]] = None,
    provider_type: str = "openai",
    max_attempts=5,
    retry_delay=30,
    error_context="",
):
    """通过 OpenAI Images API 进行图像生成或参考图编辑。"""
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI Client 未初始化，请检查 OPENAI_API_KEY。")

    options = normalize_image_generation_options(
        provider_type=provider_type,
        model_name=model_name,
        aspect_ratio=str(config.get("aspect_ratio", "1:1") or "1:1"),
        image_resolution=str(config.get("image_resolution", "2K") or "2K"),
        raw_options=config,
    )
    capabilities = get_image_model_capabilities(provider_type, model_name)
    input_images = _extract_openai_input_images(contents)
    is_edit_request = bool(input_images)
    request_params = build_openai_image_request_params(options, capabilities, edit=is_edit_request)
    gen_params: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        **request_params,
    }
    if is_edit_request:
        gen_params["image"] = [
            (filename, BytesIO(image_bytes), media_type)
            for image_bytes, media_type, filename in input_images
        ]

    safe_max_attempts = max(1, int(max_attempts or 1))
    last_exception: Exception | None = None
    last_error_text = ""
    attempts_used = 0
    for attempt in range(safe_max_attempts):
        attempts_used = attempt + 1
        if _runtime_cancel_requested():
            raise asyncio.CancelledError()
        try:
            if is_edit_request:
                response = await client.images.edit(**gen_params)
            else:
                response = await client.images.generate(**gen_params)
            if request_params.get("stream"):
                response_images = await _extract_openai_stream_images(
                    response,
                    provider=provider_type,
                    model_name=model_name,
                    error_context=error_context,
                    job_type=_job_type_from_context(error_context),
                )
            else:
                response_images = await _extract_openai_response_images(response)
            if response_images:
                return response_images
            else:
                last_error_text = "OpenAI 图像生成未返回数据"
                _emit_runtime_event(
                    level="WARNING",
                    kind="warning",
                    source="GenerationUtils",
                    job_type="generation",
                    provider="openai",
                    model=model_name,
                    attempt=attempt + 1,
                    status="retrying",
                    message="OpenAI 图像生成未返回数据，将继续重试",
                )
                logger.warning("OpenAI 图像生成失败，未返回数据")
                if attempt < safe_max_attempts - 1:
                    if _runtime_cancel_requested():
                        raise asyncio.CancelledError()
                    await asyncio.sleep(retry_delay)
                continue
        except Exception as e:
            last_exception = e
            last_error_text = _safe_text_for_log(e)
            context_msg = f" for {error_context}" if error_context else ""
            retryable = _is_openai_image_retryable_error(e)
            if _should_try_openai_responses_fallback(
                e,
                provider_type=provider_type,
                model_name=model_name,
                fallback_mode=options.responses_fallback,
            ):
                try:
                    _emit_runtime_event(
                        level="WARNING",
                        kind="retry",
                        source="GenerationUtils",
                        job_type="generation",
                        provider="openai",
                        model=model_name,
                        attempt=attempt + 1,
                        status="retrying",
                        message="OpenAI Images API 失败，改用 Responses 图像工具兜底",
                        details=f"{context_msg}: {_safe_text_for_log(e)}",
                    )
                    response_images = await _call_openai_responses_image_generation_async(
                        client=client,
                        model_name=model_name,
                        prompt=prompt,
                        options=options,
                        capabilities=capabilities,
                        input_images=input_images,
                        provider_type=provider_type,
                        error_context=error_context,
                    )
                    if response_images:
                        return response_images
                    raise RuntimeError("OpenAI Responses 图像工具未返回图像数据")
                except Exception as fallback_error:
                    last_exception = fallback_error
                    last_error_text = (
                        f"Images API: {_safe_text_for_log(e, max_len=1200)}; "
                        f"Responses fallback: {_safe_text_for_log(fallback_error, max_len=1200)}"
                    )
                    fallback_status_code = 0
                    try:
                        fallback_status_code = int(getattr(fallback_error, "status_code", 0) or 0)
                    except Exception:
                        fallback_status_code = 0
                    retryable = retryable or (
                        fallback_status_code in {408, 409, 425, 429, 500, 502, 503, 504}
                        and _is_openai_image_retryable_error(fallback_error)
                    )
            if not retryable:
                _emit_runtime_event(
                    level="ERROR",
                    kind="error",
                    source="GenerationUtils",
                    job_type="generation",
                    provider="openai",
                    model=model_name,
                    attempt=attempt + 1,
                    status="failed",
                    message="OpenAI 图像生成遇到不可恢复错误，停止重试",
                    details=f"{context_msg}: {e}",
                )
                logger.error("OpenAI 图像生成遇到不可恢复错误%s: %s", context_msg, e)
                break
            _emit_runtime_event(
                level="WARNING",
                kind="retry",
                source="GenerationUtils",
                job_type="generation",
                provider="openai",
                model=model_name,
                attempt=attempt + 1,
                status="retrying",
                message=f"OpenAI 图像生成第 {attempt + 1}/{safe_max_attempts} 次尝试失败，将继续使用同一模型，{retry_delay}s 后重试",
                details=f"{context_msg}: {e}",
            )
            logger.warning(
                "OpenAI 图像生成第 %s 次尝试失败%s，将继续使用同一模型，%ss 后重试",
                attempt + 1,
                context_msg,
                retry_delay,
            )
            if attempt < safe_max_attempts - 1:
                if _runtime_cancel_requested():
                    raise asyncio.CancelledError()
                await asyncio.sleep(retry_delay)
    context_msg = f" for {error_context}" if error_context else ""
    failure_message = f"OpenAI 图像生成在 {attempts_used or safe_max_attempts} 次同模型尝试后仍然失败{context_msg}"
    if last_error_text:
        failure_message = f"{failure_message}: {last_error_text}"
    _emit_runtime_event(
        level="ERROR",
        kind="error",
        source="GenerationUtils",
        job_type="generation",
        provider="openai",
        model=model_name,
        status="failed",
        message=f"OpenAI 图像生成在 {attempts_used or safe_max_attempts} 次同模型尝试后仍然失败",
        details=failure_message,
    )
    logger.error("OpenAI 图像生成全部 %s 次同模型尝试失败%s", attempts_used or safe_max_attempts, context_msg)
    raise RuntimeError(failure_message) from last_exception
