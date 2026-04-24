"""Shared runtime-settings helpers for CLI, demo, and tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.provider_connections import (
    BUILTIN_CONNECTION_IDS,
    ProviderConnection,
    get_provider_connection,
    list_provider_connections,
    resolve_connection,
)


PROVIDER_UI_META = {
    "gemini": {
        "api_key_label": "Google API Key",
        "api_key_help": "Google AI Studio API 密钥",
    },
    "evolink": {
        "api_key_label": "API Key",
        "api_key_help": "Evolink API 密钥（Bearer Token）",
    },
    "openrouter": {
        "api_key_label": "OpenRouter API Key",
        "api_key_help": "OpenRouter API 密钥，从 openrouter.ai 获取",
    },
    "openai": {
        "api_key_label": "OpenAI API Key",
        "api_key_help": "OpenAI 官方 API 密钥；默认用于 GPT Image 2。",
    },
    "openai_compatible": {
        "api_key_label": "兼容 API Key",
        "api_key_help": "任意 OpenAI 兼容服务的 API 密钥；若服务无需密钥，可留空。",
    },
}

DEFAULT_PROVIDER = "gemini"
SUPPORTED_PROVIDERS = tuple(BUILTIN_CONNECTION_IDS)


def normalize_provider_name(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    if not normalized:
        return DEFAULT_PROVIDER
    if normalized in SUPPORTED_PROVIDERS:
        return normalized
    supported = ", ".join(SUPPORTED_PROVIDERS)
    raise ValueError(f"Unsupported provider: {provider!r}. Expected one of: {supported}.")


@dataclass(frozen=True)
class RuntimeSettings:
    provider: str
    api_key: str
    model_name: str
    image_model_name: str
    base_url: str = ""
    connection_id: str = ""
    provider_display_name: str = ""
    provider_connection: ProviderConnection | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    concurrency_mode: str = "auto"
    max_concurrent: int = 20
    max_critic_rounds: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "api_key": self.api_key,
            "model_name": self.model_name,
            "image_model_name": self.image_model_name,
            "base_url": self.base_url,
            "connection_id": self.connection_id,
            "provider_display_name": self.provider_display_name,
            "extra_headers": dict(self.extra_headers),
            "concurrency_mode": self.concurrency_mode,
            "max_concurrent": self.max_concurrent,
            "max_critic_rounds": self.max_critic_rounds,
        }


def resolve_runtime_settings(
    provider: str,
    *,
    connection_id: str = "",
    api_key: str = "",
    model_name: str = "",
    image_model_name: str = "",
    base_url: str = "",
    extra_headers: dict[str, str] | None = None,
    concurrency_mode: str = "auto",
    max_concurrent: int = 20,
    max_critic_rounds: int = 3,
    base_dir: Path | None = None,
    model_config_data: dict[str, Any] | None = None,
) -> RuntimeSettings:
    repo_root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent.parent
    resolved_connection = resolve_connection(
        connection_id or provider,
        api_key=api_key,
        text_model=model_name,
        image_model=image_model_name,
        base_url=base_url,
        extra_headers=extra_headers,
        base_dir=repo_root,
        model_config_data=model_config_data,
    )

    resolved_concurrency_mode = str(concurrency_mode or "auto").strip().lower() or "auto"
    if resolved_concurrency_mode not in {"auto", "manual"}:
        resolved_concurrency_mode = "auto"

    return RuntimeSettings(
        provider=resolved_connection.provider_type,
        api_key=str(resolved_connection.api_key or "").strip(),
        model_name=str(resolved_connection.text_model or "").strip(),
        image_model_name=str(resolved_connection.image_model or "").strip(),
        base_url=str(resolved_connection.base_url or "").strip(),
        connection_id=resolved_connection.connection_id,
        provider_display_name=resolved_connection.display_name,
        provider_connection=resolved_connection,
        extra_headers=dict(resolved_connection.extra_headers),
        concurrency_mode=resolved_concurrency_mode,
        max_concurrent=max(1, int(max_concurrent)),
        max_critic_rounds=max(0, int(max_critic_rounds)),
    )


def build_provider_ui_defaults(
    provider: str,
    *,
    base_dir: Path | None = None,
    model_config_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    connection = get_provider_connection(
        provider,
        base_dir=base_dir,
        model_config_data=model_config_data,
    )
    settings = resolve_runtime_settings(
        provider,
        base_dir=base_dir,
        model_config_data=model_config_data,
    )
    ui_meta = PROVIDER_UI_META.get(connection.provider_type, PROVIDER_UI_META["openai_compatible"])
    return {
        "connection_id": connection.connection_id,
        "display_name": connection.display_name,
        "provider_type": connection.provider_type,
        "api_key_label": ui_meta["api_key_label"],
        "api_key_help": ui_meta["api_key_help"],
        "api_key_default": settings.api_key,
        "model_name": settings.model_name,
        "image_model_name": settings.image_model_name,
        "base_url": settings.base_url,
        "api_key_env_var": connection.api_key_env_var,
        "extra_headers": dict(connection.extra_headers),
        "model_discovery_mode": connection.model_discovery_mode,
        "model_allowlist": list(connection.model_allowlist),
        "supports_text": connection.supports_text,
        "supports_image": connection.supports_image,
        "enabled": connection.enabled,
        "builtin": connection.builtin,
        "probe_results": dict(connection.probe_results),
    }


def build_all_provider_ui_defaults(
    *,
    base_dir: Path | None = None,
    model_config_data: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        connection.connection_id: build_provider_ui_defaults(
            connection.connection_id,
            base_dir=base_dir,
            model_config_data=model_config_data,
        )
        for connection in list_provider_connections(
            base_dir=base_dir,
            model_config_data=model_config_data,
        )
    }


def list_runtime_connections(
    *,
    base_dir: Path | None = None,
    model_config_data: dict[str, Any] | None = None,
    include_disabled: bool = False,
) -> list[ProviderConnection]:
    return list_provider_connections(
        base_dir=base_dir,
        model_config_data=model_config_data,
        include_disabled=include_disabled,
    )


def initialize_provider_runtime(settings: RuntimeSettings) -> None:
    """Initialize the default runtime context for compatibility paths."""
    from utils import generation_utils

    generation_utils.set_default_runtime_context(build_runtime_context(settings))


def build_runtime_context(
    settings: RuntimeSettings,
    *,
    status_hook=None,
    event_hook=None,
    cancel_check=None,
):
    """Build an isolated runtime context for one run/session."""
    from utils import generation_utils

    return generation_utils.create_runtime_context(
        connection_id=settings.connection_id,
        provider=settings.provider,
        api_key=settings.api_key,
        base_url=settings.base_url,
        extra_headers=settings.extra_headers,
        event_hook=event_hook,
        status_hook=status_hook,
        cancel_check=cancel_check,
    )
