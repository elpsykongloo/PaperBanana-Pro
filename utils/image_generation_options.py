"""图像生成模型能力与参数归一化。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ImageModelCapabilities:
    provider_type: str
    model_family: str
    size_options: tuple[str, ...]
    quality_options: tuple[str, ...]
    background_options: tuple[str, ...] = ("opaque",)
    output_format_options: tuple[str, ...] = ("png",)
    supports_output_compression: bool = False
    supports_moderation: bool = False
    supports_input_fidelity: bool = False
    supports_stream: bool = False
    supports_partial_images: bool = False
    supports_edit: bool = False
    transparent_supported: bool = False
    legacy_resolution_options: tuple[str, ...] = ("2K", "4K")


@dataclass(frozen=True)
class ImageGenerationOptions:
    size: str = "auto"
    quality: str = "auto"
    background: str = "auto"
    output_format: str = "png"
    output_compression: int | None = None
    moderation: str = "auto"
    input_fidelity: str = "auto"
    stream: bool = False
    partial_images: int = 0
    aspect_ratio: str = "1:1"
    image_resolution: str = "2K"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GPT_IMAGE_2_SIZES = (
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2560x1440",
    "1440x2560",
)
GPT_IMAGE_1_SIZES = ("auto", "1024x1024", "1536x1024", "1024x1536")
DALLE3_SIZES = ("1024x1024", "1792x1024", "1024x1792")
DALLE2_SIZES = ("256x256", "512x512", "1024x1024")


CAPABILITY_PRESETS: dict[str, ImageModelCapabilities] = {
    "gemini": ImageModelCapabilities(
        provider_type="gemini",
        model_family="gemini",
        size_options=("1K", "2K", "4K"),
        quality_options=("auto",),
        background_options=("auto", "opaque"),
        legacy_resolution_options=("1K", "2K", "4K"),
    ),
    "evolink": ImageModelCapabilities(
        provider_type="evolink",
        model_family="evolink",
        size_options=("2K", "4K"),
        quality_options=("2K", "4K"),
        background_options=("opaque",),
        legacy_resolution_options=("2K", "4K"),
    ),
    "openrouter": ImageModelCapabilities(
        provider_type="openrouter",
        model_family="openrouter",
        size_options=("1K", "2K", "4K"),
        quality_options=("auto",),
        background_options=("auto", "opaque"),
        output_format_options=("png", "jpeg", "webp"),
        legacy_resolution_options=("1K", "2K", "4K"),
    ),
    "gpt-image-2": ImageModelCapabilities(
        provider_type="openai",
        model_family="gpt-image-2",
        size_options=GPT_IMAGE_2_SIZES,
        quality_options=("auto", "low", "medium", "high"),
        background_options=("auto", "opaque"),
        output_format_options=("png", "jpeg", "webp"),
        supports_output_compression=True,
        supports_moderation=True,
        supports_input_fidelity=True,
        supports_stream=True,
        supports_partial_images=True,
        supports_edit=True,
        transparent_supported=False,
        legacy_resolution_options=("auto", "1K", "2K", "4K"),
    ),
    "gpt-image-1": ImageModelCapabilities(
        provider_type="openai",
        model_family="gpt-image-1",
        size_options=GPT_IMAGE_1_SIZES,
        quality_options=("auto", "low", "medium", "high"),
        background_options=("auto", "opaque", "transparent"),
        output_format_options=("png", "jpeg", "webp"),
        supports_output_compression=True,
        supports_moderation=True,
        supports_input_fidelity=True,
        supports_stream=True,
        supports_partial_images=True,
        supports_edit=True,
        transparent_supported=True,
        legacy_resolution_options=("auto", "1K", "2K"),
    ),
    "dall-e-3": ImageModelCapabilities(
        provider_type="openai",
        model_family="dall-e-3",
        size_options=DALLE3_SIZES,
        quality_options=("standard", "hd"),
        background_options=("opaque",),
        output_format_options=("png",),
        legacy_resolution_options=("1K", "2K"),
    ),
    "dall-e-2": ImageModelCapabilities(
        provider_type="openai",
        model_family="dall-e-2",
        size_options=DALLE2_SIZES,
        quality_options=("standard",),
        background_options=("opaque",),
        output_format_options=("png",),
        supports_edit=True,
        legacy_resolution_options=("1K",),
    ),
}


ASPECT_RATIO_SIZE_MAP = {
    "1:1": {"1K": "1024x1024", "2K": "2048x2048", "4K": "2048x2048"},
    "16:9": {"1K": "1536x1024", "2K": "2560x1440", "4K": "2560x1440"},
    "21:9": {"1K": "1536x1024", "2K": "2560x1440", "4K": "2560x1440"},
    "3:2": {"1K": "1536x1024", "2K": "1536x1024", "4K": "2560x1440"},
    "4:3": {"1K": "1536x1024", "2K": "1536x1024", "4K": "2560x1440"},
    "2:3": {"1K": "1024x1536", "2K": "1024x1536", "4K": "1440x2560"},
    "3:4": {"1K": "1024x1536", "2K": "1024x1536", "4K": "1440x2560"},
    "9:16": {"1K": "1024x1536", "2K": "1440x2560", "4K": "1440x2560"},
    "4:5": {"1K": "1024x1536", "2K": "1024x1536", "4K": "1440x2560"},
    "5:4": {"1K": "1536x1024", "2K": "1536x1024", "4K": "2560x1440"},
}


def _first_supported(preferred: str, options: tuple[str, ...], fallback: str = "") -> str:
    if preferred and preferred in options:
        return preferred
    if fallback and fallback in options:
        return fallback
    return options[0] if options else preferred


def get_image_model_capabilities(provider_type: str, model_name: str = "") -> ImageModelCapabilities:
    provider = str(provider_type or "").strip().lower()
    model = str(model_name or "").strip().lower()
    if provider in {"gemini", "evolink", "openrouter"}:
        return CAPABILITY_PRESETS[provider]
    if model.startswith("gpt-image-2"):
        return CAPABILITY_PRESETS["gpt-image-2"]
    if model.startswith("gpt-image-1") or model == "chatgpt-image-latest":
        return CAPABILITY_PRESETS["gpt-image-1"]
    if model.startswith("dall-e-3"):
        return CAPABILITY_PRESETS["dall-e-3"]
    if model.startswith("dall-e-2"):
        return CAPABILITY_PRESETS["dall-e-2"]
    if provider in {"openai", "openai_compatible"}:
        return CAPABILITY_PRESETS["gpt-image-2"]
    return CAPABILITY_PRESETS["evolink"]


def resolve_legacy_size(aspect_ratio: str, image_resolution: str, capabilities: ImageModelCapabilities) -> str:
    resolution = str(image_resolution or "").strip()
    if resolution in capabilities.size_options:
        return resolution
    if resolution.upper() in {"1K", "2K", "4K"}:
        mapped = ASPECT_RATIO_SIZE_MAP.get(
            str(aspect_ratio or "").strip(),
            ASPECT_RATIO_SIZE_MAP["1:1"],
        ).get(resolution.upper(), "")
        if mapped in capabilities.size_options:
            return mapped
    return _first_supported("auto", capabilities.size_options, "1024x1024")


def normalize_image_generation_options(
    *,
    provider_type: str,
    model_name: str,
    aspect_ratio: str = "1:1",
    image_resolution: str = "2K",
    raw_options: dict[str, Any] | None = None,
) -> ImageGenerationOptions:
    capabilities = get_image_model_capabilities(provider_type, model_name)
    raw = dict(raw_options or {})
    resolved_aspect_ratio = str(raw.get("aspect_ratio", aspect_ratio) or "1:1").strip() or "1:1"
    resolved_resolution = str(raw.get("image_resolution", image_resolution) or "2K").strip() or "2K"

    size = str(raw.get("size") or raw.get("image_size") or "").strip()
    if not size:
        size = resolve_legacy_size(resolved_aspect_ratio, resolved_resolution, capabilities)
    size = _first_supported(size, capabilities.size_options, "auto")

    default_quality = "auto" if "auto" in capabilities.quality_options else capabilities.quality_options[0]
    quality = _first_supported(str(raw.get("quality") or default_quality).strip(), capabilities.quality_options, default_quality)

    default_background = "auto" if "auto" in capabilities.background_options else capabilities.background_options[0]
    background = str(raw.get("background") or default_background).strip() or default_background
    if background == "transparent" and not capabilities.transparent_supported:
        background = default_background
    background = _first_supported(background, capabilities.background_options, default_background)

    output_format = _first_supported(
        str(raw.get("output_format") or "png").strip(),
        capabilities.output_format_options,
        "png",
    )

    compression_value = raw.get("output_compression")
    output_compression: int | None = None
    if capabilities.supports_output_compression and output_format in {"jpeg", "webp"} and compression_value not in (None, ""):
        output_compression = max(0, min(100, int(compression_value)))

    moderation = str(raw.get("moderation") or "auto").strip() or "auto"
    if not capabilities.supports_moderation or moderation not in {"auto", "low"}:
        moderation = "auto"

    input_fidelity = str(raw.get("input_fidelity") or "auto").strip() or "auto"
    if not capabilities.supports_input_fidelity or input_fidelity not in {"auto", "low", "high"}:
        input_fidelity = "auto"

    stream = bool(raw.get("stream", False)) and capabilities.supports_stream
    partial_images = int(raw.get("partial_images", 0) or 0)
    if not capabilities.supports_partial_images:
        partial_images = 0
    partial_images = max(0, min(3, partial_images))

    return ImageGenerationOptions(
        size=size,
        quality=quality,
        background=background,
        output_format=output_format,
        output_compression=output_compression,
        moderation=moderation,
        input_fidelity=input_fidelity,
        stream=stream,
        partial_images=partial_images,
        aspect_ratio=resolved_aspect_ratio,
        image_resolution=resolved_resolution,
    )


def build_openai_image_request_params(
    options: ImageGenerationOptions,
    capabilities: ImageModelCapabilities,
    *,
    edit: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "size": options.size,
        "quality": options.quality,
    }
    if capabilities.model_family.startswith("gpt-image"):
        payload["background"] = options.background
        payload["output_format"] = options.output_format
        if capabilities.supports_moderation:
            payload["moderation"] = options.moderation
    if options.output_compression is not None:
        payload["output_compression"] = options.output_compression
    if edit and options.input_fidelity != "auto":
        payload["input_fidelity"] = options.input_fidelity
    if options.stream:
        payload["stream"] = True
        payload["partial_images"] = options.partial_images
    return payload
