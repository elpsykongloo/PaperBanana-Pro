"""Shared runtime-settings helpers for CLI, demo, and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.config_loader import load_model_config, load_provider_defaults


PROVIDER_UI_META = {
    "evolink": {
        "api_key_label": "API Key",
        "api_key_help": "Evolink API 密钥（Bearer Token）",
    },
    "gemini": {
        "api_key_label": "Google API Key",
        "api_key_help": "Google AI Studio API 密钥",
    },
}

SUPPORTED_PROVIDERS = tuple(PROVIDER_UI_META.keys())


def normalize_provider_name(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized in SUPPORTED_PROVIDERS:
        return normalized
    return "gemini"


@dataclass(frozen=True)
class RuntimeSettings:
    provider: str
    api_key: str
    model_name: str
    image_model_name: str
    concurrency_mode: str = "auto"
    max_concurrent: int = 20
    max_critic_rounds: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "api_key": self.api_key,
            "model_name": self.model_name,
            "image_model_name": self.image_model_name,
            "concurrency_mode": self.concurrency_mode,
            "max_concurrent": self.max_concurrent,
            "max_critic_rounds": self.max_critic_rounds,
        }


def resolve_runtime_settings(
    provider: str,
    *,
    api_key: str = "",
    model_name: str = "",
    image_model_name: str = "",
    concurrency_mode: str = "auto",
    max_concurrent: int = 20,
    max_critic_rounds: int = 3,
    base_dir: Path | None = None,
    model_config_data: dict[str, Any] | None = None,
) -> RuntimeSettings:
    repo_root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent.parent
    normalized_provider = normalize_provider_name(provider)
    config_data = model_config_data if model_config_data is not None else load_model_config(repo_root)
    defaults = load_provider_defaults(normalized_provider, config_data, base_dir=repo_root)

    resolved_concurrency_mode = str(concurrency_mode or "auto").strip().lower() or "auto"
    if resolved_concurrency_mode not in {"auto", "manual"}:
        resolved_concurrency_mode = "auto"

    return RuntimeSettings(
        provider=normalized_provider,
        api_key=str(api_key or defaults["api_key"]).strip(),
        model_name=str(model_name or defaults["model_name"]).strip(),
        image_model_name=str(image_model_name or defaults["image_model_name"]).strip(),
        concurrency_mode=resolved_concurrency_mode,
        max_concurrent=max(1, int(max_concurrent)),
        max_critic_rounds=max(0, int(max_critic_rounds)),
    )


def build_provider_ui_defaults(
    provider: str,
    *,
    base_dir: Path | None = None,
    model_config_data: dict[str, Any] | None = None,
) -> dict[str, str]:
    normalized_provider = normalize_provider_name(provider)
    settings = resolve_runtime_settings(
        normalized_provider,
        base_dir=base_dir,
        model_config_data=model_config_data,
    )
    ui_meta = PROVIDER_UI_META[normalized_provider]
    return {
        "api_key_label": ui_meta["api_key_label"],
        "api_key_help": ui_meta["api_key_help"],
        "api_key_default": settings.api_key,
        "model_name": settings.model_name,
        "image_model_name": settings.image_model_name,
    }


def build_all_provider_ui_defaults(
    *,
    base_dir: Path | None = None,
    model_config_data: dict[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    return {
        provider: build_provider_ui_defaults(
            provider,
            base_dir=base_dir,
            model_config_data=model_config_data,
        )
        for provider in SUPPORTED_PROVIDERS
    }


def initialize_provider_runtime(settings: RuntimeSettings) -> None:
    """Initialize the default runtime context for compatibility paths."""
    from utils import generation_utils

    generation_utils.set_default_runtime_context(
        build_runtime_context(settings)
    )


def build_runtime_context(
    settings: RuntimeSettings,
    *,
    status_hook=None,
):
    """Build an isolated runtime context for one run/session."""
    from utils import generation_utils

    return generation_utils.create_runtime_context(
        provider=settings.provider,
        api_key=settings.api_key,
        status_hook=status_hook,
    )
