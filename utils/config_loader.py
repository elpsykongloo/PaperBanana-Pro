"""Helpers for loading local configuration and secrets."""

from pathlib import Path
from typing import Any

import os
import yaml


CONFIG_DIRNAME = "configs"
LOCAL_SECRET_DIRNAME = "local"

SECRET_FILE_MAP = {
    ("api_keys", "google_api_key"): "google_api_key.txt",
    ("api_keys", "openai_api_key"): "openai_api_key.txt",
    ("api_keys", "anthropic_api_key"): "anthropic_api_key.txt",
    ("evolink", "api_key"): "evolink_api_key.txt",
    ("openrouter", "api_key"): "openrouter_api_key.txt",
}

PROVIDER_CONFIG_MAP = {
    "gemini": {
        "model_section": "defaults",
        "api_section": "api_keys",
        "api_key": "google_api_key",
        "api_env": "GOOGLE_API_KEY",
        "default_model_name": "gemini-3.1-flash-lite-preview",
        "default_image_model_name": "gemini-3.1-flash-image-preview",
        "base_url_section": "",
        "base_url_key": "",
        "base_url_env": "",
        "default_base_url": "",
    },
    "evolink": {
        "model_section": "evolink",
        "api_section": "evolink",
        "api_key": "api_key",
        "api_env": "EVOLINK_API_KEY",
        "default_model_name": "gemini-2.5-flash",
        "default_image_model_name": "nano-banana-2-lite",
        "base_url_section": "evolink",
        "base_url_key": "base_url",
        "base_url_env": "EVOLINK_BASE_URL",
        "default_base_url": "https://api.evolink.ai",
    },
    "openrouter": {
        "model_section": "openrouter",
        "api_section": "openrouter",
        "api_key": "api_key",
        "api_env": "OPENROUTER_API_KEY",
        "default_model_name": "google/gemini-3.1-flash-lite-preview",
        "default_image_model_name": "google/gemini-3.1-flash-image-preview",
        "base_url_section": "openrouter",
        "base_url_key": "base_url",
        "base_url_env": "OPENROUTER_BASE_URL",
        "default_base_url": "https://openrouter.ai/api/v1",
    },
}


def _normalize_provider_name(provider: str) -> str:
    return str(provider or "").strip().lower()


def _get_provider_config(provider: str) -> dict[str, str]:
    normalized_provider = _normalize_provider_name(provider)
    provider_config = PROVIDER_CONFIG_MAP.get(normalized_provider)
    if provider_config is None:
        supported = ", ".join(sorted(PROVIDER_CONFIG_MAP))
        raise ValueError(
            f"Unsupported provider: {provider!r}. Expected one of: {supported}."
        )
    return provider_config


def get_repo_root(base_dir: Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    return Path(__file__).resolve().parent.parent


def get_config_dir(base_dir: Path | None = None) -> Path:
    return get_repo_root(base_dir) / CONFIG_DIRNAME


def get_local_secret_dir(base_dir: Path | None = None) -> Path:
    return get_config_dir(base_dir) / LOCAL_SECRET_DIRNAME


def get_local_secret_path(section: str, key: str, base_dir: Path | None = None) -> Path | None:
    filename = SECRET_FILE_MAP.get((section, key))
    if not filename:
        return None
    return get_local_secret_dir(base_dir) / filename


def read_local_secret(section: str, key: str, base_dir: Path | None = None) -> str:
    secret_path = get_local_secret_path(section, key, base_dir=base_dir)
    if not secret_path or not secret_path.exists():
        return ""
    return secret_path.read_text(encoding="utf-8").strip()


def write_local_secret(
    section: str,
    key: str,
    value: str,
    base_dir: Path | None = None,
) -> Path | None:
    secret_path = get_local_secret_path(section, key, base_dir=base_dir)
    if secret_path is None:
        return None

    normalized_value = str(value or "").strip()
    if not normalized_value:
        return secret_path if secret_path.exists() else None

    secret_path.parent.mkdir(parents=True, exist_ok=True)
    current_value = ""
    if secret_path.exists():
        current_value = secret_path.read_text(encoding="utf-8").strip()
    if current_value != normalized_value:
        secret_path.write_text(normalized_value + "\n", encoding="utf-8")
    return secret_path


def delete_local_secret(
    section: str,
    key: str,
    base_dir: Path | None = None,
) -> Path | None:
    secret_path = get_local_secret_path(section, key, base_dir=base_dir)
    if secret_path is None:
        return None
    if secret_path.exists():
        secret_path.unlink()
    return secret_path


def load_model_config(base_dir: Path | None = None) -> dict[str, Any]:
    config_path = get_config_dir(base_dir) / "model_config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config_val(
    model_config: dict[str, Any],
    section: str,
    key: str,
    env_var: str,
    default: str = "",
    base_dir: Path | None = None,
) -> str:
    val = os.getenv(env_var, "").strip()
    if val:
        return val

    val = read_local_secret(section, key, base_dir=base_dir)
    if val:
        return val

    if section in model_config:
        val = model_config[section].get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return default


def get_provider_model_defaults(
    provider: str,
    model_config: dict[str, Any],
) -> dict[str, str]:
    provider_config = _get_provider_config(provider)
    section_config = model_config.get(provider_config["model_section"], {})
    model_name = section_config.get("model_name") or provider_config["default_model_name"]
    image_model_name = (
        section_config.get("image_model_name")
        or provider_config["default_image_model_name"]
    )
    return {
        "model_name": str(model_name).strip(),
        "image_model_name": str(image_model_name).strip(),
    }


def get_provider_api_key(
    provider: str,
    model_config: dict[str, Any],
    base_dir: Path | None = None,
) -> str:
    provider_config = _get_provider_config(provider)
    return get_config_val(
        model_config,
        provider_config["api_section"],
        provider_config["api_key"],
        provider_config["api_env"],
        "",
        base_dir=base_dir,
    )


def get_provider_base_url(
    provider: str,
    model_config: dict[str, Any],
    base_dir: Path | None = None,
) -> str:
    provider_config = _get_provider_config(provider)
    if not provider_config["base_url_section"]:
        return ""
    return get_config_val(
        model_config,
        provider_config["base_url_section"],
        provider_config["base_url_key"],
        provider_config["base_url_env"],
        provider_config["default_base_url"],
        base_dir=base_dir,
    )


def load_provider_defaults(
    provider: str,
    model_config: dict[str, Any],
    base_dir: Path | None = None,
) -> dict[str, str]:
    defaults = get_provider_model_defaults(provider, model_config)
    defaults["api_key"] = get_provider_api_key(
        provider,
        model_config,
        base_dir=base_dir,
    )
    defaults["base_url"] = get_provider_base_url(
        provider,
        model_config,
        base_dir=base_dir,
    )
    return defaults


def write_provider_api_key(
    provider: str,
    api_key: str,
    base_dir: Path | None = None,
) -> Path | None:
    provider_config = _get_provider_config(provider)
    return write_local_secret(
        provider_config["api_section"],
        provider_config["api_key"],
        api_key,
        base_dir=base_dir,
    )


def delete_provider_api_key(
    provider: str,
    base_dir: Path | None = None,
) -> Path | None:
    provider_config = _get_provider_config(provider)
    return delete_local_secret(
        provider_config["api_section"],
        provider_config["api_key"],
        base_dir=base_dir,
    )
