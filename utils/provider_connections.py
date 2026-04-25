"""Provider 连接注册、解析与探针工具。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from utils.config_loader import (
    get_provider_api_key,
    load_model_config,
    load_provider_defaults,
)
from utils.image_generation_options import normalize_image_generation_options


DEFAULT_PROVIDER_REGISTRY_VERSION = 1
DEFAULT_PROVIDER_REGISTRY_FILE = "provider_registry.yaml"
DEFAULT_CONNECTION_META_FILE = "provider_connection_meta.json"
CUSTOM_PROVIDER_DIRNAME = "providers"
BUILTIN_CONNECTION_IDS = ("gemini", "evolink", "openrouter", "openai")
CUSTOM_PROVIDER_TYPE = "openai_compatible"
SUPPORTED_PROVIDER_TYPES = (*BUILTIN_CONNECTION_IDS, CUSTOM_PROVIDER_TYPE)
CONNECTION_ID_RE = re.compile(r"[^a-z0-9-]+")


@dataclass(frozen=True)
class ProbeResult:
    target: str
    stage: str
    status: str
    error_type: str = ""
    http_status: int = 0
    message: str = ""
    raw_excerpt: str = ""
    discovered_models: tuple[str, ...] = ()
    latency_ms: int = 0
    tested_model: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["discovered_models"] = list(self.discovered_models)
        return payload


@dataclass(frozen=True)
class ProviderConnection:
    connection_id: str
    display_name: str
    provider_type: str
    protocol_family: str
    base_url: str = ""
    api_key_env_var: str = ""
    text_model: str = ""
    image_model: str = ""
    model_discovery_mode: str = "manual"
    model_allowlist: tuple[str, ...] = ()
    extra_headers: dict[str, str] = field(default_factory=dict)
    supports_text: bool = True
    supports_image: bool = True
    enabled: bool = True
    builtin: bool = False
    api_key: str = ""
    probe_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_registry_dict(self) -> dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "display_name": self.display_name,
            "provider_type": self.provider_type,
            "protocol_family": self.protocol_family,
            "base_url": self.base_url,
            "api_key_env_var": self.api_key_env_var,
            "text_model": self.text_model,
            "image_model": self.image_model,
            "model_discovery_mode": self.model_discovery_mode,
            "model_allowlist": list(self.model_allowlist),
            "extra_headers": dict(self.extra_headers),
            "supports_text": bool(self.supports_text),
            "supports_image": bool(self.supports_image),
            "enabled": bool(self.enabled),
            "builtin": bool(self.builtin),
        }


def normalize_connection_id(value: str | None, *, default: str = "custom-openai") -> str:
    text = CONNECTION_ID_RE.sub("-", str(value or "").strip().lower()).strip("-")
    return text or default


def _repo_root(base_dir: Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    return Path(__file__).resolve().parent.parent


def _config_dir(base_dir: Path | None = None) -> Path:
    return _repo_root(base_dir) / "configs"


def _local_dir(base_dir: Path | None = None) -> Path:
    return _config_dir(base_dir) / "local"


def _registry_path(base_dir: Path | None = None) -> Path:
    return _config_dir(base_dir) / DEFAULT_PROVIDER_REGISTRY_FILE


def _meta_path(base_dir: Path | None = None) -> Path:
    return _local_dir(base_dir) / DEFAULT_CONNECTION_META_FILE


def get_custom_provider_secret_path(
    connection_id: str,
    *,
    base_dir: Path | None = None,
) -> Path:
    return _local_dir(base_dir) / CUSTOM_PROVIDER_DIRNAME / f"{normalize_connection_id(connection_id)}.txt"


def _read_yaml_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml_payload(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return path


def _read_json_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json_payload(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def read_custom_provider_api_key(
    connection_id: str,
    *,
    base_dir: Path | None = None,
) -> str:
    secret_path = get_custom_provider_secret_path(connection_id, base_dir=base_dir)
    if not secret_path.exists():
        return ""
    return secret_path.read_text(encoding="utf-8").strip()


def write_custom_provider_api_key(
    connection_id: str,
    api_key: str,
    *,
    base_dir: Path | None = None,
) -> Path | None:
    normalized_value = str(api_key or "").strip()
    secret_path = get_custom_provider_secret_path(connection_id, base_dir=base_dir)
    if not normalized_value:
        return secret_path if secret_path.exists() else None
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    secret_path.write_text(normalized_value + "\n", encoding="utf-8")
    return secret_path


def delete_custom_provider_api_key(
    connection_id: str,
    *,
    base_dir: Path | None = None,
) -> Path | None:
    secret_path = get_custom_provider_secret_path(connection_id, base_dir=base_dir)
    if secret_path.exists():
        secret_path.unlink()
    return secret_path


def load_provider_registry(base_dir: Path | None = None) -> dict[str, Any]:
    path = _registry_path(base_dir)
    payload = _read_yaml_payload(path)
    version = int(payload.get("version", DEFAULT_PROVIDER_REGISTRY_VERSION) or DEFAULT_PROVIDER_REGISTRY_VERSION)
    connections = payload.get("connections", [])
    if not isinstance(connections, list):
        connections = []
    return {
        "version": version,
        "connections": connections,
    }


def save_provider_registry(
    connections: list[ProviderConnection],
    *,
    base_dir: Path | None = None,
) -> Path:
    payload = {
        "version": DEFAULT_PROVIDER_REGISTRY_VERSION,
        "connections": [item.to_registry_dict() for item in connections if not item.builtin],
    }
    return _write_yaml_payload(_registry_path(base_dir), payload)


def load_connection_metadata(base_dir: Path | None = None) -> dict[str, Any]:
    return _read_json_payload(_meta_path(base_dir))


def write_connection_probe_result(
    connection_id: str,
    result: ProbeResult,
    *,
    base_dir: Path | None = None,
) -> Path:
    metadata = load_connection_metadata(base_dir)
    connection_meta = dict(metadata.get(connection_id, {}) or {})
    probe_results = dict(connection_meta.get("probe_results", {}) or {})
    probe_results[result.target] = result.to_dict()
    connection_meta["probe_results"] = probe_results
    metadata[connection_id] = connection_meta
    return _write_json_payload(_meta_path(base_dir), metadata)


def _extract_connection_probe_results(
    connection_id: str,
    *,
    base_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    metadata = load_connection_metadata(base_dir)
    connection_meta = metadata.get(connection_id, {})
    probe_results = connection_meta.get("probe_results", {})
    return probe_results if isinstance(probe_results, dict) else {}


def _build_builtin_connection(
    connection_id: str,
    *,
    base_dir: Path | None = None,
    model_config_data: dict[str, Any] | None = None,
) -> ProviderConnection:
    repo_root = _repo_root(base_dir)
    config_data = model_config_data if model_config_data is not None else load_model_config(repo_root)
    defaults = load_provider_defaults(connection_id, config_data, base_dir=repo_root)
    protocol_family = "openai" if connection_id in {"openrouter", "openai"} else connection_id
    model_discovery_mode = {
        "gemini": "static",
        "evolink": "manual",
        "openrouter": "hybrid",
        "openai": "hybrid",
    }.get(connection_id, "manual")
    display_name = {
        "gemini": "Gemini",
        "evolink": "Evolink",
        "openrouter": "OpenRouter",
        "openai": "OpenAI",
    }.get(connection_id, connection_id)
    supports_image = connection_id in {"gemini", "evolink", "openrouter", "openai"}
    allowlist = []
    for item in (defaults.get("model_name", ""), defaults.get("image_model_name", "")):
        normalized = str(item or "").strip()
        if normalized and normalized not in allowlist:
            allowlist.append(normalized)
    return ProviderConnection(
        connection_id=connection_id,
        display_name=display_name,
        provider_type=connection_id,
        protocol_family=protocol_family,
        base_url=str(defaults.get("base_url", "") or "").strip(),
        api_key_env_var={
            "gemini": "GOOGLE_API_KEY",
            "evolink": "EVOLINK_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
        }[connection_id],
        text_model=str(defaults.get("model_name", "") or "").strip(),
        image_model=str(defaults.get("image_model_name", "") or "").strip(),
        model_discovery_mode=model_discovery_mode,
        model_allowlist=tuple(allowlist),
        extra_headers={},
        supports_text=True,
        supports_image=supports_image,
        enabled=True,
        builtin=True,
        api_key=get_provider_api_key(connection_id, config_data, base_dir=repo_root),
        probe_results=_extract_connection_probe_results(connection_id, base_dir=repo_root),
    )


def _coerce_connection_payload(payload: dict[str, Any]) -> ProviderConnection:
    connection_id = normalize_connection_id(payload.get("connection_id"))
    provider_type = str(payload.get("provider_type") or CUSTOM_PROVIDER_TYPE).strip().lower()
    if provider_type not in SUPPORTED_PROVIDER_TYPES:
        provider_type = CUSTOM_PROVIDER_TYPE
    allowlist = []
    for item in payload.get("model_allowlist", []) or []:
        normalized = str(item or "").strip()
        if normalized and normalized not in allowlist:
            allowlist.append(normalized)
    extra_headers = payload.get("extra_headers", {}) or {}
    if not isinstance(extra_headers, dict):
        extra_headers = {}
    return ProviderConnection(
        connection_id=connection_id,
        display_name=str(payload.get("display_name") or connection_id).strip() or connection_id,
        provider_type=provider_type,
        protocol_family=str(payload.get("protocol_family") or "openai").strip() or "openai",
        base_url=str(payload.get("base_url") or "").strip(),
        api_key_env_var=str(payload.get("api_key_env_var") or "").strip(),
        text_model=str(payload.get("text_model") or "").strip(),
        image_model=str(payload.get("image_model") or "").strip(),
        model_discovery_mode=str(payload.get("model_discovery_mode") or "manual").strip().lower() or "manual",
        model_allowlist=tuple(allowlist),
        extra_headers={str(key): str(value) for key, value in extra_headers.items() if str(key).strip()},
        supports_text=bool(payload.get("supports_text", True)),
        supports_image=bool(payload.get("supports_image", False)),
        enabled=bool(payload.get("enabled", True)),
        builtin=bool(payload.get("builtin", False)),
    )


def list_provider_connections(
    *,
    base_dir: Path | None = None,
    model_config_data: dict[str, Any] | None = None,
    include_disabled: bool = True,
) -> list[ProviderConnection]:
    repo_root = _repo_root(base_dir)
    config_data = model_config_data if model_config_data is not None else load_model_config(repo_root)
    registry = load_provider_registry(repo_root)
    connections: list[ProviderConnection] = [
        _build_builtin_connection(connection_id, base_dir=repo_root, model_config_data=config_data)
        for connection_id in BUILTIN_CONNECTION_IDS
    ]
    for item in registry.get("connections", []):
        if not isinstance(item, dict):
            continue
        connection = _coerce_connection_payload(item)
        api_key = ""
        env_var = str(connection.api_key_env_var or "").strip()
        if env_var:
            api_key = str(os.getenv(env_var, "") or "").strip()
        if not api_key:
            api_key = read_custom_provider_api_key(connection.connection_id, base_dir=repo_root)
        connections.append(
            ProviderConnection(
                **{
                    **asdict(connection),
                    "api_key": api_key,
                    "probe_results": _extract_connection_probe_results(connection.connection_id, base_dir=repo_root),
                }
            )
        )
    if include_disabled:
        return connections
    return [item for item in connections if item.enabled]


def get_provider_connection(
    connection_id: str,
    *,
    base_dir: Path | None = None,
    model_config_data: dict[str, Any] | None = None,
) -> ProviderConnection:
    normalized_id = normalize_connection_id(connection_id, default=str(connection_id or ""))
    for connection in list_provider_connections(base_dir=base_dir, model_config_data=model_config_data):
        if connection.connection_id == normalized_id:
            return connection
    raise ValueError(f"Unsupported provider connection: {connection_id!r}")


def resolve_connection(
    connection_id: str,
    *,
    api_key: str = "",
    text_model: str = "",
    image_model: str = "",
    base_url: str = "",
    extra_headers: dict[str, str] | None = None,
    enabled: bool | None = None,
    base_dir: Path | None = None,
    model_config_data: dict[str, Any] | None = None,
) -> ProviderConnection:
    connection = get_provider_connection(
        connection_id,
        base_dir=base_dir,
        model_config_data=model_config_data,
    )
    resolved_headers = dict(connection.extra_headers)
    if extra_headers:
        for key, value in extra_headers.items():
            normalized_key = str(key or "").strip()
            if not normalized_key:
                continue
            resolved_headers[normalized_key] = str(value or "").strip()
    return ProviderConnection(
        **{
            **asdict(connection),
            "api_key": str(api_key or connection.api_key or "").strip(),
            "text_model": str(text_model or connection.text_model or "").strip(),
            "image_model": str(image_model or connection.image_model or "").strip(),
            "base_url": str(base_url or connection.base_url or "").strip(),
            "extra_headers": resolved_headers,
            "enabled": connection.enabled if enabled is None else bool(enabled),
        }
    )


def upsert_custom_connection(
    payload: dict[str, Any],
    *,
    api_key: str = "",
    persist_secret: bool = True,
    base_dir: Path | None = None,
) -> ProviderConnection:
    repo_root = _repo_root(base_dir)
    normalized_payload = dict(payload)
    normalized_payload["connection_id"] = normalize_connection_id(payload.get("connection_id"))
    normalized_payload["provider_type"] = CUSTOM_PROVIDER_TYPE
    normalized_payload["protocol_family"] = str(payload.get("protocol_family") or "openai").strip() or "openai"
    connection = _coerce_connection_payload(normalized_payload)
    if connection.connection_id in BUILTIN_CONNECTION_IDS:
        raise ValueError("自定义连接 ID 不能与内置连接同名。")

    registry = load_provider_registry(repo_root)
    existing_connections = []
    replaced = False
    for item in registry.get("connections", []):
        if not isinstance(item, dict):
            continue
        if normalize_connection_id(item.get("connection_id")) == connection.connection_id:
            existing_connections.append(connection)
            replaced = True
        else:
            existing_connections.append(_coerce_connection_payload(item))
    if not replaced:
        existing_connections.append(connection)
    save_provider_registry(existing_connections, base_dir=repo_root)

    normalized_api_key = str(api_key or "").strip()
    if persist_secret and normalized_api_key:
        write_custom_provider_api_key(connection.connection_id, normalized_api_key, base_dir=repo_root)
    elif not persist_secret:
        delete_custom_provider_api_key(connection.connection_id, base_dir=repo_root)

    return resolve_connection(
        connection.connection_id,
        api_key=normalized_api_key,
        base_dir=repo_root,
    )


def delete_custom_connection(
    connection_id: str,
    *,
    remove_secret: bool = True,
    base_dir: Path | None = None,
) -> None:
    normalized_id = normalize_connection_id(connection_id)
    if normalized_id in BUILTIN_CONNECTION_IDS:
        raise ValueError("不能删除内置连接。")
    repo_root = _repo_root(base_dir)
    registry = load_provider_registry(repo_root)
    kept_connections = []
    for item in registry.get("connections", []):
        if not isinstance(item, dict):
            continue
        if normalize_connection_id(item.get("connection_id")) == normalized_id:
            continue
        kept_connections.append(_coerce_connection_payload(item))
    save_provider_registry(kept_connections, base_dir=repo_root)
    if remove_secret:
        delete_custom_provider_api_key(normalized_id, base_dir=repo_root)
    metadata = load_connection_metadata(repo_root)
    if normalized_id in metadata:
        metadata.pop(normalized_id, None)
        _write_json_payload(_meta_path(repo_root), metadata)


def parse_extra_headers_json(raw_text: str) -> dict[str, str]:
    normalized_text = str(raw_text or "").strip()
    if not normalized_text:
        return {}
    payload = json.loads(normalized_text)
    if not isinstance(payload, dict):
        raise ValueError("额外请求头必须是 JSON 对象。")
    return {
        str(key): str(value)
        for key, value in payload.items()
        if str(key or "").strip()
    }


def format_extra_headers_json(headers: dict[str, str] | None) -> str:
    if not headers:
        return ""
    return json.dumps(headers, ensure_ascii=False, indent=2)


def _clip_raw_excerpt(value: Any, *, max_length: int = 280) -> str:
    text = str(value or "").strip()
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def _gemini_finish_reason_name(response: Any) -> str:
    try:
        candidates = list(getattr(response, "candidates", []) or [])
        if not candidates:
            return ""
        finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason is None:
            return ""
        return str(getattr(finish_reason, "name", finish_reason) or "").strip()
    except Exception:
        return ""


def _extract_gemini_text_response(response: Any) -> list[str]:
    texts: list[str] = []
    parts = list(getattr(response, "parts", []) or [])
    if not parts:
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts.extend(list(getattr(content, "parts", []) or []))
    for part in parts:
        text = str(getattr(part, "text", "") or "").strip()
        if text:
            texts.append(text)
    return texts


def _extract_gemini_inline_images(response: Any) -> list[tuple[str, bytes]]:
    images: list[tuple[str, bytes]] = []
    parts = list(getattr(response, "parts", []) or [])
    if not parts:
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts.extend(list(getattr(content, "parts", []) or []))
    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        raw_bytes = getattr(inline_data, "data", None) if inline_data is not None else None
        if raw_bytes:
            images.append((str(getattr(inline_data, "mime_type", "") or "").strip(), raw_bytes))
    return images


def _build_probe_latency_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _classify_gemini_finish_reason(
    finish_reason: str,
    *,
    target: str,
) -> tuple[str, str]:
    normalized_reason = str(finish_reason or "").strip().upper()
    if not normalized_reason:
        return "response_incompatible", f"Gemini {target} 探针未返回可用结果。"
    if normalized_reason == "NO_IMAGE":
        return "response_incompatible", "Gemini 图像探针未返回图片内容。"
    if normalized_reason == "MALFORMED_FUNCTION_CALL":
        return "response_incompatible", "Gemini 返回了不兼容的函数调用响应。"
    if normalized_reason == "UNEXPECTED_TOOL_CALL":
        return "response_incompatible", "Gemini 返回了未预期的工具调用响应。"
    if normalized_reason == "IMAGE_RECITATION":
        return "response_incompatible", "Gemini 图像响应触发了图片引用限制。"
    if normalized_reason in {"IMAGE_SAFETY", "SAFETY", "BLOCKLIST"}:
        return "provider_unavailable", f"Gemini 响应被安全策略拦截：{normalized_reason}。"
    return "response_incompatible", f"Gemini {target} 探针未返回可用结果：{normalized_reason}。"


async def _gemini_generate_content_once(
    connection: ProviderConnection,
    *,
    model_name: str,
    prompt: str,
    system_instruction: str = "",
    max_output_tokens: int = 0,
    response_modalities: list[str] | None = None,
    image_aspect_ratio: str = "",
    image_size: str = "",
) -> Any:
    from google import genai
    from google.genai import types

    config_kwargs: dict[str, Any] = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if max_output_tokens > 0:
        config_kwargs["max_output_tokens"] = max_output_tokens
    if response_modalities:
        config_kwargs["response_modalities"] = response_modalities
    if image_aspect_ratio or image_size:
        image_config_kwargs: dict[str, Any] = {}
        if image_aspect_ratio:
            image_config_kwargs["aspect_ratio"] = image_aspect_ratio
        if image_size:
            image_config_kwargs["image_size"] = image_size
        config_kwargs["image_config"] = types.ImageConfig(**image_config_kwargs)
    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
    client = genai.Client(api_key=connection.api_key)
    return await asyncio.to_thread(
        client.models.generate_content,
        model=model_name,
        contents=prompt,
        config=config,
    )


def classify_probe_error(exc: Exception) -> tuple[str, int, str]:
    http_status = 0
    message = str(exc or "").strip()
    error_type = "unknown"

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

    if APIStatusError and isinstance(exc, APIStatusError):
        http_status = int(getattr(exc, "status_code", 0) or 0)
    elif hasattr(exc, "status_code"):
        try:
            http_status = int(getattr(exc, "status_code", 0) or 0)
        except Exception:
            http_status = 0
    elif hasattr(exc, "status"):
        try:
            http_status = int(getattr(exc, "status", 0) or 0)
        except Exception:
            http_status = 0

    if APITimeoutError and isinstance(exc, APITimeoutError):
        return "timeout", http_status, message
    if APIConnectionError and isinstance(exc, APIConnectionError):
        return "base_url_error", http_status, message
    if AuthenticationError and isinstance(exc, AuthenticationError):
        return "invalid_credentials", http_status or 401, message
    if RateLimitError and isinstance(exc, RateLimitError):
        return "rate_limited", http_status or 429, message
    if PermissionDeniedError and isinstance(exc, PermissionDeniedError):
        return "provider_unavailable", http_status or 403, message
    if NotFoundError and isinstance(exc, NotFoundError):
        lowered = message.lower()
        if "model" in lowered and ("not found" in lowered or "unknown" in lowered):
            return "model_not_found", http_status or 404, message
        if "<!doctype html" in lowered or "<html" in lowered:
            return "response_incompatible", http_status or 404, message
        return "base_url_error", http_status or 404, message
    if BadRequestError and isinstance(exc, BadRequestError):
        lowered = message.lower()
        if "model" in lowered and ("not found" in lowered or "unknown" in lowered):
            return "model_not_found", http_status or 400, message
        return "response_incompatible", http_status or 400, message

    lowered = message.lower()
    if http_status == 401:
        error_type = "invalid_credentials"
    elif http_status == 404 and ("<!doctype html" in lowered or "<html" in lowered):
        error_type = "response_incompatible"
    elif "invalid api key" in lowered or (
        "unauthorized" in lowered and http_status in {0, 400, 401, 403}
    ):
        error_type = "invalid_credentials"
    elif http_status == 402 or "insufficient" in lowered or "credit" in lowered or "quota" in lowered:
        error_type = "insufficient_credits"
    elif (http_status == 404 or "404" in lowered) and (
        "model" in lowered or "not_found" in lowered or "not found" in lowered
    ):
        error_type = "model_not_found"
    elif http_status == 429 or "rate limit" in lowered or "too many requests" in lowered:
        error_type = "rate_limited"
    elif http_status in {502, 503, 504} or "service unavailable" in lowered:
        error_type = "provider_unavailable"
    elif "timed out" in lowered or "timeout" in lowered:
        error_type = "timeout"
    elif "connection" in lowered or "base url" in lowered or "dns" in lowered:
        error_type = "base_url_error"
    elif http_status == 400:
        error_type = "response_incompatible"
    return error_type, http_status, message


async def discover_models(
    connection: ProviderConnection,
    *,
    timeout_seconds: float = 20.0,
) -> ProbeResult:
    started_at = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()
    allowlist = list(connection.model_allowlist)
    if connection.provider_type == "gemini":
        discovered = []
        for item in (connection.text_model, connection.image_model, *allowlist):
            normalized = str(item or "").strip()
            if normalized and normalized not in discovered:
                discovered.append(normalized)
        return ProbeResult(
            target="discovery",
            stage="static_models",
            status="success",
            message="Gemini 使用内置静态模型列表。",
            discovered_models=tuple(discovered),
            latency_ms=_build_probe_latency_ms(started_at),
            timestamp=timestamp,
        )

    if connection.provider_type == "evolink":
        discovered = []
        for item in (connection.text_model, connection.image_model, *allowlist):
            normalized = str(item or "").strip()
            if normalized and normalized not in discovered:
                discovered.append(normalized)
        return ProbeResult(
            target="discovery",
            stage="manual_fallback",
            status="skipped",
            message="Evolink 当前未接入标准 /models 发现，保留手动模型输入。",
            discovered_models=tuple(discovered),
            latency_ms=_build_probe_latency_ms(started_at),
            timestamp=timestamp,
        )

    try:
        from openai import AsyncOpenAI
    except Exception as exc:  # pragma: no cover
        error_type, http_status, message = classify_probe_error(exc)
        return ProbeResult(
            target="discovery",
            stage="import_client",
            status="failed",
            error_type=error_type,
            http_status=http_status,
            message=message,
            raw_excerpt=_clip_raw_excerpt(message),
            latency_ms=_build_probe_latency_ms(started_at),
            timestamp=timestamp,
        )

    try:
        client = AsyncOpenAI(
            api_key=connection.api_key or "no-api-key",
            base_url=connection.base_url or None,
            timeout=timeout_seconds,
            max_retries=0,
            default_headers=connection.extra_headers or None,
        )
        response = await client.models.list()
        models = []
        for item in getattr(response, "data", []) or []:
            model_id = str(getattr(item, "id", "") or "").strip()
            if model_id and model_id not in models:
                models.append(model_id)
        await client.close()
        message = "已成功获取远端模型列表。" if models else "模型列表接口返回为空。"
        status = "success" if models else "warning"
        return ProbeResult(
            target="discovery",
            stage="models_list",
            status=status,
            message=message,
            discovered_models=tuple(models),
            latency_ms=_build_probe_latency_ms(started_at),
            timestamp=timestamp,
        )
    except Exception as exc:
        error_type, http_status, message = classify_probe_error(exc)
        return ProbeResult(
            target="discovery",
            stage="models_list",
            status="failed",
            error_type=error_type,
            http_status=http_status,
            message="模型发现失败，可继续手动填写模型名。",
            raw_excerpt=_clip_raw_excerpt(message),
            discovered_models=tuple(allowlist),
            latency_ms=_build_probe_latency_ms(started_at),
            timestamp=timestamp,
        )


async def probe_text(connection: ProviderConnection) -> ProbeResult:
    started_at = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()
    tested_model = str(connection.text_model or "").strip()
    if not connection.supports_text:
        return ProbeResult(
            target="text",
            stage="capability_check",
            status="skipped",
            message="该连接未启用文本能力。",
            latency_ms=_build_probe_latency_ms(started_at),
            timestamp=timestamp,
        )
    if not tested_model:
        return ProbeResult(
            target="text",
            stage="config_check",
            status="failed",
            error_type="model_not_found",
            message="缺少文本模型名称。",
            latency_ms=_build_probe_latency_ms(started_at),
            timestamp=timestamp,
        )

    from utils import generation_utils

    context = None
    try:
        if connection.provider_type == "gemini":
            response = await _gemini_generate_content_once(
                connection,
                model_name=tested_model,
                prompt="只回复 OK。",
                system_instruction="你是连接探针，只回复 OK。",
                max_output_tokens=8,
            )
            response_texts = _extract_gemini_text_response(response)
            finish_reason = _gemini_finish_reason_name(response)
            if response_texts:
                return ProbeResult(
                    target="text",
                    stage="chat_completion",
                    status="success",
                    message="文本链路探针成功。",
                    raw_excerpt=_clip_raw_excerpt(" | ".join(response_texts)),
                    tested_model=tested_model,
                    latency_ms=_build_probe_latency_ms(started_at),
                    timestamp=timestamp,
                )
            error_type, failure_message = _classify_gemini_finish_reason(finish_reason, target="文本")
            raw_excerpt = _clip_raw_excerpt(
                " | ".join(
                    item
                    for item in (
                        f"finish_reason={finish_reason}" if finish_reason else "",
                        *response_texts,
                    )
                    if item
                )
            )
            return ProbeResult(
                target="text",
                stage="chat_completion",
                status="failed",
                error_type=error_type,
                message=failure_message,
                raw_excerpt=raw_excerpt,
                tested_model=tested_model,
                latency_ms=_build_probe_latency_ms(started_at),
                timestamp=timestamp,
            )

        context = generation_utils.create_runtime_context(
            provider=connection.provider_type,
            api_key=connection.api_key,
            base_url=connection.base_url,
            extra_headers=connection.extra_headers,
        )
        with generation_utils.use_runtime_context(context):
            if connection.provider_type == "evolink":
                await generation_utils.call_evolink_text_with_retry_async(
                    model_name=tested_model,
                    contents=[{"type": "text", "text": "返回 OK"}],
                    config={
                        "system_prompt": "只回复 OK。",
                        "temperature": 0,
                        "max_output_tokens": 8,
                    },
                    max_attempts=1,
                    retry_delay=0,
                    error_context="provider_probe[text]",
                )
            else:
                await generation_utils.call_openai_with_retry_async(
                    model_name=tested_model,
                    contents=[{"type": "text", "text": "返回 OK"}],
                    config={
                        "system_prompt": "只回复 OK。",
                        "temperature": 0,
                        "candidate_num": 1,
                        "max_completion_tokens": 8,
                    },
                    max_attempts=1,
                    retry_delay=0,
                    error_context="provider_probe[text]",
                )
        return ProbeResult(
            target="text",
            stage="chat_completion",
            status="success",
            message="文本链路探针成功。",
            tested_model=tested_model,
            latency_ms=_build_probe_latency_ms(started_at),
            timestamp=timestamp,
        )
    except Exception as exc:
        error_type, http_status, message = classify_probe_error(exc)
        return ProbeResult(
            target="text",
            stage="chat_completion",
            status="failed",
            error_type=error_type,
            http_status=http_status,
            message="文本链路探针失败。",
            raw_excerpt=_clip_raw_excerpt(message),
            tested_model=tested_model,
            latency_ms=_build_probe_latency_ms(started_at),
            timestamp=timestamp,
        )
    finally:
        from utils import generation_utils

        if context is not None:
            await generation_utils.close_runtime_context(context)


async def probe_image(connection: ProviderConnection) -> ProbeResult:
    started_at = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()
    tested_model = str(connection.image_model or "").strip()
    if not connection.supports_image:
        return ProbeResult(
            target="image",
            stage="capability_check",
            status="skipped",
            message="该连接未启用图像能力。",
            latency_ms=_build_probe_latency_ms(started_at),
            timestamp=timestamp,
        )
    if not tested_model:
        return ProbeResult(
            target="image",
            stage="config_check",
            status="skipped",
            message="未配置图像模型，跳过图像探针。",
            latency_ms=_build_probe_latency_ms(started_at),
            timestamp=timestamp,
        )

    from utils import generation_utils

    context = None
    try:
        if connection.provider_type == "gemini":
            # 按 Gemini 官方图像生成文档，使用稳定的 generate_content 路径与简洁英文提示词。
            prompt = "Create a simple blue circle icon on a white background. No text."
            compatibility_attempts = [
                {
                    "name": "generate_content_plain",
                    "kwargs": {},
                },
                {
                    "name": "generate_content_image_only",
                    "kwargs": {"response_modalities": ["IMAGE"]},
                },
            ]
            last_finish_reason = ""
            last_response_texts: list[str] = []
            last_strategy = compatibility_attempts[0]["name"]
            for attempt in compatibility_attempts:
                last_strategy = str(attempt["name"])
                response = await _gemini_generate_content_once(
                    connection,
                    model_name=tested_model,
                    prompt=prompt,
                    **dict(attempt["kwargs"]),
                )
                images = _extract_gemini_inline_images(response)
                finish_reason = _gemini_finish_reason_name(response)
                response_texts = _extract_gemini_text_response(response)
                if images:
                    detail = f"官方兼容探测 `{last_strategy}` 已返回 {len(images)} 张测试图片。"
                    return ProbeResult(
                        target="image",
                        stage="image_generation",
                        status="success",
                        message="图像链路探针成功。",
                        raw_excerpt=_clip_raw_excerpt(detail),
                        tested_model=tested_model,
                        latency_ms=_build_probe_latency_ms(started_at),
                        timestamp=timestamp,
                    )
                last_finish_reason = finish_reason
                last_response_texts = response_texts
                if finish_reason not in {
                    "",
                    "NO_IMAGE",
                    "MALFORMED_FUNCTION_CALL",
                    "UNEXPECTED_TOOL_CALL",
                    "MAX_TOKENS",
                }:
                    break

            error_type, failure_message = _classify_gemini_finish_reason(last_finish_reason, target="图像")
            raw_excerpt = _clip_raw_excerpt(
                " | ".join(
                    item
                    for item in (
                        f"strategy={last_strategy}",
                        f"finish_reason={last_finish_reason}" if last_finish_reason else "",
                        *last_response_texts,
                    )
                    if item
                )
            )
            return ProbeResult(
                target="image",
                stage="image_generation",
                status="failed",
                error_type=error_type,
                message=failure_message,
                raw_excerpt=raw_excerpt,
                tested_model=tested_model,
                latency_ms=_build_probe_latency_ms(started_at),
                timestamp=timestamp,
            )

        context = generation_utils.create_runtime_context(
            provider=connection.provider_type,
            api_key=connection.api_key,
            base_url=connection.base_url,
            extra_headers=connection.extra_headers,
        )
        with generation_utils.use_runtime_context(context):
            if connection.provider_type == "evolink":
                await generation_utils.call_evolink_image_with_retry_async(
                    model_name=tested_model,
                    prompt="A simple blue circle icon on white background.",
                    config={
                        "aspect_ratio": "1:1",
                        "quality": "2K",
                    },
                    max_attempts=1,
                    retry_delay=0,
                    error_context="provider_probe[image]",
                )
            elif connection.provider_type == "openrouter":
                await generation_utils.call_openrouter_image_generation_with_retry_async(
                    model_name=tested_model,
                    prompt="A simple blue circle icon on white background.",
                    config={
                        "aspect_ratio": "1:1",
                        "image_size": "1K",
                        "output_format": "png",
                    },
                    max_attempts=1,
                    retry_delay=0,
                    error_context="provider_probe[image]",
                )
            else:
                probe_options = normalize_image_generation_options(
                    provider_type=connection.provider_type,
                    model_name=tested_model,
                    aspect_ratio="1:1",
                    image_resolution="1K",
                ).to_dict()
                probe_options["responses_model"] = connection.text_model
                await generation_utils.call_openai_image_generation_with_retry_async(
                    model_name=tested_model,
                    prompt="A simple blue circle icon on white background.",
                    config=probe_options,
                    provider_type=connection.provider_type,
                    max_attempts=1,
                    retry_delay=0,
                    error_context="provider_probe[image]",
                )
        return ProbeResult(
            target="image",
            stage="image_generation",
            status="success",
            message="图像链路探针成功。",
            tested_model=tested_model,
            latency_ms=_build_probe_latency_ms(started_at),
            timestamp=timestamp,
        )
    except Exception as exc:
        error_type, http_status, message = classify_probe_error(exc)
        return ProbeResult(
            target="image",
            stage="image_generation",
            status="failed",
            error_type=error_type,
            http_status=http_status,
            message="图像链路探针失败。",
            raw_excerpt=_clip_raw_excerpt(message),
            tested_model=tested_model,
            latency_ms=_build_probe_latency_ms(started_at),
            timestamp=timestamp,
        )
    finally:
        from utils import generation_utils

        if context is not None:
            await generation_utils.close_runtime_context(context)


async def probe_connection(
    connection: ProviderConnection,
    *,
    include_discovery: bool = True,
    stage_callback: Callable[[str, str], None] | None = None,
) -> dict[str, ProbeResult]:
    results: dict[str, ProbeResult] = {}
    if include_discovery:
        if stage_callback is not None:
            stage_callback("discovery", "running")
        results["discovery"] = await discover_models(connection)
        if stage_callback is not None:
            stage_callback("discovery", results["discovery"].status)
    if stage_callback is not None:
        stage_callback("text", "running")
    results["text"] = await probe_text(connection)
    if stage_callback is not None:
        stage_callback("text", results["text"].status)
    if stage_callback is not None:
        stage_callback("image", "running")
    results["image"] = await probe_image(connection)
    if stage_callback is not None:
        stage_callback("image", results["image"].status)
    return results


def run_async_probe(coro):
    return asyncio.run(coro)
