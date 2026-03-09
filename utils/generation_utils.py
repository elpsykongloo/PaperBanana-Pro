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
import re
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any, Callable, Optional, Tuple

from PIL import Image

import os
from pathlib import Path

from utils.config_loader import load_model_config, get_config_val
from utils.log_config import get_logger

logger = get_logger("GenerationUtils")

# ==================== 配置加载 ====================

REPO_ROOT = Path(__file__).parent.parent
model_config = load_model_config(REPO_ROOT)


# ==================== 运行时状态回调（用于 UI 实时反馈） ====================

runtime_status_hook: Optional[Callable[[str], None]] = None

DEFAULT_GEMINI_IMAGE_FALLBACK_MODEL = "gemini-3.1-flash-image-preview"


def set_runtime_status_hook(hook: Optional[Callable[[str], None]]) -> None:
    """设置运行时状态回调。"""
    global runtime_status_hook
    runtime_status_hook = hook


def _safe_text_for_log(value: Any, max_len: int = 6000) -> str:
    """Convert arbitrary value to a printable string safe for Windows stdout."""
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
        return safe[:max_len] + f"...(truncated {len(safe) - max_len} chars)"
    return safe


def _safe_log(message: Any) -> None:
    """尽力输出日志，永远不会向调用方抛出异常。"""
    try:
        logger.debug(_safe_text_for_log(message))
    except Exception:
        pass


def _emit_runtime_status(message: str) -> None:
    """发送运行时状态（失败不影响主流程）。"""
    if runtime_status_hook is None:
        return
    try:
        runtime_status_hook(message)
    except Exception as err:
        _safe_log(f"[DEBUG] [WARN] runtime_status_hook 调用失败: {err}")

# ==================== Evolink Provider 初始化 ====================

evolink_provider = None

evolink_api_key = get_config_val(
    model_config,
    "evolink",
    "api_key",
    "EVOLINK_API_KEY",
    "",
    base_dir=REPO_ROOT,
)
evolink_base_url = get_config_val(
    model_config,
    "evolink",
    "base_url",
    "EVOLINK_BASE_URL",
    "https://api.evolink.ai",
    base_dir=REPO_ROOT,
)

if evolink_api_key:
    try:
        from providers.evolink import EvolinkProvider
        evolink_provider = EvolinkProvider(api_key=evolink_api_key, base_url=evolink_base_url)
        logger.info(f"✅ 已初始化 Evolink Provider (base_url={evolink_base_url})")
    except ImportError:
        logger.warning("⚠️  未安装 providers.evolink，Evolink Provider 不可用")
else:
    logger.debug("未配置 Evolink API Key；仅当选择 evolink provider 时才会影响运行。")


def _cleanup_evolink_provider():
    """Shut down Evolink provider session on process exit."""
    global evolink_provider
    if evolink_provider is not None:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(evolink_provider.close())
            else:
                loop.run_until_complete(evolink_provider.close())
        except Exception:
            pass


import atexit
atexit.register(_cleanup_evolink_provider)


def init_evolink_provider(api_key: str, base_url: str = ""):
    """用指定的 API Key 初始化或更新 Evolink Provider（供界面动态传入）。"""
    global evolink_provider
    if not api_key:
        return
    url = base_url or evolink_base_url
    from providers.evolink import EvolinkProvider
    evolink_provider = EvolinkProvider(api_key=api_key, base_url=url)
    logger.info(f"✅ 已通过界面初始化 Evolink Provider (base_url={url})")


def init_gemini_client(api_key: str):
    """用指定的 API Key 初始化或更新 Gemini Client（供界面动态传入）。"""
    global gemini_client
    if not api_key:
        return
    try:
        from google import genai
        gemini_client = genai.Client(api_key=api_key)
        logger.info("✅ 已通过界面初始化 Gemini Client")
    except ImportError:
        logger.warning("⚠️  未安装 google-genai，Gemini Client 不可用。请运行 pip install google-genai")


# ==================== 原始 Provider 初始化（保留兼容性） ====================

gemini_client = None
anthropic_client = None
openai_client = None

api_key = get_config_val(
    model_config,
    "api_keys",
    "google_api_key",
    "GOOGLE_API_KEY",
    "",
    base_dir=REPO_ROOT,
)
if api_key:
    try:
        from google import genai
        from google.genai import types
        gemini_client = genai.Client(api_key=api_key)
        logger.info("✅ 已初始化 Gemini Client")
    except ImportError:
        logger.warning("⚠️  未安装 google-genai，Gemini Client 不可用")

anthropic_api_key = get_config_val(
    model_config,
    "api_keys",
    "anthropic_api_key",
    "ANTHROPIC_API_KEY",
    "",
    base_dir=REPO_ROOT,
)
if anthropic_api_key:
    try:
        from anthropic import AsyncAnthropic
        anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)
        logger.info("✅ 已初始化 Anthropic Client")
    except ImportError:
        logger.warning("⚠️  未安装 anthropic，Anthropic Client 不可用")

openai_api_key = get_config_val(
    model_config,
    "api_keys",
    "openai_api_key",
    "OPENAI_API_KEY",
    "",
    base_dir=REPO_ROOT,
)
if openai_api_key:
    try:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=openai_api_key)
        logger.info("✅ 已初始化 OpenAI Client")
    except ImportError:
        logger.warning("⚠️  未安装 openai，OpenAI Client 不可用")


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
    logger.debug(f"📤 call_evolink_text: model={model_name}, provider={'已初始化' if evolink_provider else '未初始化'}")
    if evolink_provider is None:
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

    return await evolink_provider.generate_text(
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
    if evolink_provider is None:
        raise RuntimeError("Evolink Provider 未初始化，请检查 EVOLINK_API_KEY 配置。")
    url = await evolink_provider.upload_image_base64(image_b64, media_type)
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
    logger.debug(f"🖼️ call_evolink_image: model={model_name}, config={config}, provider={'已初始化' if evolink_provider else '未初始化'}")
    if evolink_provider is None:
        raise RuntimeError("Evolink Provider 未初始化，请检查 EVOLINK_API_KEY 配置。")

    aspect_ratio = config.get("aspect_ratio", "16:9")
    quality = config.get("quality", "2K")
    image_urls = config.get("image_urls", None)

    return await evolink_provider.generate_image(
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


def _choose_gemini_text_fallback(model_name: str) -> Optional[str]:
    """为高风险文本模型选择兜底模型。"""
    lower_model = (model_name or "").lower()
    if "gemini-3.1-pro" in lower_model or "gemini-3-pro" in lower_model:
        return "gemini-2.5-flash"
    if "gemini-2.0-flash" in lower_model:
        return "gemini-2.5-flash"
    return None


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
    """原始 Gemini API 异步调用（保留兼容性）"""
    if gemini_client is None:
        raise RuntimeError("Gemini Client 未初始化，请检查 Google API Key。")

    result_list: List[str] = []
    target_candidate_count = int(getattr(config, "candidate_count", 1) or 1)
    if hasattr(config, "candidate_count") and config.candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    is_image_request = _is_gemini_image_request(model_name, config)
    request_timeout_seconds = _get_gemini_request_timeout_seconds(is_image_request)

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
            try:
                client = gemini_client
                gemini_contents = _convert_to_gemini_parts(current_contents)
                context_msg = f" ({error_context})" if error_context else ""
                _emit_runtime_status(
                    f"[TRY] stage={stage_name} model={stage_model_name} "
                    f"attempt={attempt_idx + 1}/{stage_attempts}{context_msg}"
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
                _emit_runtime_status(retry_line)
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
                    await asyncio.sleep(current_delay)
                else:
                    _emit_runtime_status(
                        f"[FAIL] stage={stage_name} model={stage_model_name} exhausted {stage_attempts} attempts"
                    )

        return stage_results[:target_candidate_count], last_error_meta, len(stage_results) >= target_candidate_count

    final_error_meta: Dict[str, Any] = {}
    # Stage 1: primary model
    primary_results, primary_error_meta, primary_success = await _run_stage(
        stage_name="primary",
        stage_model_name=model_name,
        stage_max_attempts=int(max_attempts),
    )
    if primary_success and primary_results:
        result_list = primary_results
    else:
        if primary_error_meta:
            final_error_meta = primary_error_meta

        # Stage 2 for image requests: fixed fallback model + fixed retries
        if is_image_request:
            fallback_image_model = (image_fallback_model or "").strip()
            if fallback_image_model and fallback_image_model != model_name:
                _emit_runtime_status(
                    f"[FALLBACK] image model switch: {model_name} -> {fallback_image_model}"
                )
                fallback_results, fallback_error_meta, fallback_success = await _run_stage(
                    stage_name="fallback",
                    stage_model_name=fallback_image_model,
                    stage_max_attempts=int(image_fallback_max_attempts),
                )
                if fallback_success and fallback_results:
                    result_list = fallback_results
                elif fallback_error_meta:
                    final_error_meta = fallback_error_meta
        else:
            # Text requests fallback only when primary error indicates instability/quota.
            fallback_text_model = _choose_gemini_text_fallback(model_name)
            if (
                fallback_text_model
                and fallback_text_model != model_name
                and _should_try_text_fallback(str(primary_error_meta.get("error_text", "")))
            ):
                _emit_runtime_status(
                    f"[FALLBACK] text model switch: {model_name} -> {fallback_text_model}"
                )
                fallback_results, fallback_error_meta, fallback_success = await _run_stage(
                    stage_name="fallback_text",
                    stage_model_name=fallback_text_model,
                    stage_max_attempts=int(max_attempts),
                )
                if fallback_success and fallback_results:
                    result_list = fallback_results
                elif fallback_error_meta:
                    final_error_meta = fallback_error_meta

    if len(result_list) < target_candidate_count:
        if final_error_meta:
            _emit_runtime_status(
                f"[FAIL] Gemini 全部阶段失败: "
                f"last_stage={final_error_meta.get('stage')} model={final_error_meta.get('model')} "
                f"code={final_error_meta.get('code')} status={final_error_meta.get('status')}"
            )
            _safe_log(
                f"[Gemini] 全部阶段失败 ({error_context}): {final_error_meta.get('error_text', '')}"
            )
        result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
    return result_list


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
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
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
            first_response = await anthropic_client.messages.create(
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
            logger.warning(f"⚠️  验证第 {attempt + 1} 次尝试失败{context_msg}: {error_str}。{retry_delay}s 后重试...")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    if not is_input_valid:
        logger.error(f"❌ 全部 {max_attempts} 次验证尝试失败，返回错误")
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            anthropic_client.messages.create(
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
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    current_contents = contents
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            openai_contents = _convert_to_openai_format(current_contents)
            first_response = await openai_client.chat.completions.create(
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
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            logger.warning(f"⚠️  验证第 {attempt + 1} 次尝试失败{context_msg}: {error_str}。{retry_delay}s 后重试...")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    if not is_input_valid:
        logger.error(f"❌ 全部 {max_attempts} 次验证尝试失败，返回错误")
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            openai_client.chat.completions.create(
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


async def call_openai_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """原始 OpenAI 图像生成 API 异步调用（保留兼容性）"""
    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")

    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": quality,
        "background": background,
        "output_format": output_format,
    }

    for attempt in range(max_attempts):
        try:
            response = await openai_client.images.generate(**gen_params)
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                logger.warning("⚠️  OpenAI 图像生成失败，未返回数据")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue
        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            logger.warning(f"⚠️  OpenAI 图像生成第 {attempt + 1} 次尝试失败{context_msg}: {e}。{retry_delay}s 后重试...")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"❌ OpenAI 图像生成全部 {max_attempts} 次尝试失败{context_msg}")
                return ["Error"]

    return ["Error"]
