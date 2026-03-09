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
}

PROVIDER_CONFIG_MAP = {
    "gemini": {
        "model_section": "defaults",
        "api_section": "api_keys",
        "api_key": "google_api_key",
        "api_env": "GOOGLE_API_KEY",
        "default_model_name": "gemini-3.1-pro-preview",
        "default_image_model_name": "gemini-3-pro-image-preview",
    },
    "evolink": {
        "model_section": "evolink",
        "api_section": "evolink",
        "api_key": "api_key",
        "api_env": "EVOLINK_API_KEY",
        "default_model_name": "gemini-2.5-flash",
        "default_image_model_name": "nano-banana-2-lite",
    },
}


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
    provider_config = PROVIDER_CONFIG_MAP.get(provider, PROVIDER_CONFIG_MAP["gemini"])
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
    provider_config = PROVIDER_CONFIG_MAP.get(provider, PROVIDER_CONFIG_MAP["gemini"])
    return get_config_val(
        model_config,
        provider_config["api_section"],
        provider_config["api_key"],
        provider_config["api_env"],
        "",
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
    return defaults
