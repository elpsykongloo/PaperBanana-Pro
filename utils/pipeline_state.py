"""Typed helpers for pipeline state and artifact keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping


def normalize_task_name(task_name: str) -> str:
    return "plot" if str(task_name or "").strip().lower() == "plot" else "diagram"


def detect_task_type_from_result(result: dict[str, Any] | list[dict[str, Any]] | None) -> str:
    """Infer task type from one result item or a list of items."""
    if isinstance(result, list):
        if not result:
            return "diagram"
        result = result[0]

    if not isinstance(result, dict):
        return "diagram"

    if any(key.startswith("target_plot_") for key in result.keys()):
        return "plot"

    explicit_task = str(result.get("task_name", "")).strip().lower()
    if explicit_task in {"plot", "diagram"}:
        return explicit_task

    content = result.get("content")
    if isinstance(content, (dict, list)):
        return "plot"
    return "diagram"


def planner_desc_key(task_name: str) -> str:
    return f"target_{normalize_task_name(task_name)}_desc0"


def stylist_desc_key(task_name: str) -> str:
    return f"target_{normalize_task_name(task_name)}_stylist_desc0"


def critic_desc_key(task_name: str, round_idx: int) -> str:
    return f"target_{normalize_task_name(task_name)}_critic_desc{int(round_idx)}"


def critic_suggestions_key(task_name: str, round_idx: int) -> str:
    return f"target_{normalize_task_name(task_name)}_critic_suggestions{int(round_idx)}"


def critic_status_key(task_name: str, round_idx: int) -> str:
    return f"target_{normalize_task_name(task_name)}_critic_status{int(round_idx)}"


def critic_raw_response_key(task_name: str, round_idx: int) -> str:
    return f"target_{normalize_task_name(task_name)}_critic_raw_response{int(round_idx)}"


def image_key_for_desc(desc_key: str) -> str:
    return f"{desc_key}_base64_jpg"


def mime_key_for_desc(desc_key: str) -> str:
    return f"{desc_key}_mime_type"


def code_key_for_desc(desc_key: str) -> str:
    return f"{desc_key}_code"


def plot_exec_key_for_desc(desc_key: str) -> str:
    return f"{desc_key}_plot_exec"


def desc_key_from_image_key(image_key: str) -> str | None:
    if isinstance(image_key, str) and image_key.endswith("_base64_jpg"):
        return image_key[: -len("_base64_jpg")]
    return None


def get_available_critic_rounds(result: dict[str, Any], task_name: str) -> list[int]:
    prefix = f"target_{normalize_task_name(task_name)}_critic_desc"
    suffix = "_base64_jpg"
    rounds = []
    for key, value in (result or {}).items():
        if not value or not key.startswith(prefix) or not key.endswith(suffix):
            continue
        round_text = key[len(prefix) : -len(suffix)]
        if round_text.isdigit():
            rounds.append(int(round_text))
    return sorted(set(rounds))


def find_final_stage_keys(
    result: dict[str, Any],
    task_name: str,
    exp_mode: str = "demo_planner_critic",
) -> tuple[str, str | None]:
    eval_image_field = result.get("eval_image_field")
    if isinstance(eval_image_field, str) and result.get(eval_image_field):
        return eval_image_field, desc_key_from_image_key(eval_image_field)

    normalized_task = normalize_task_name(task_name)
    critic_rounds = get_available_critic_rounds(result, task_name=normalized_task)
    if critic_rounds:
        final_round = critic_rounds[-1]
        desc_key = critic_desc_key(normalized_task, final_round)
        return image_key_for_desc(desc_key), desc_key

    if exp_mode == "demo_full":
        desc_key = stylist_desc_key(normalized_task)
        return image_key_for_desc(desc_key), desc_key

    desc_key = planner_desc_key(normalized_task)
    return image_key_for_desc(desc_key), desc_key


def collect_parse_error_round_keys(result: dict[str, Any]) -> list[str]:
    rounds = []
    for key, value in (result or {}).items():
        if "_critic_status" in key and value == "parse_error":
            rounds.append(key)
    return sorted(rounds)


@dataclass(frozen=True)
class RenderOptions:
    aspect_ratio: str
    image_resolution: str


def get_render_options(
    data: dict[str, Any],
    *,
    default_aspect_ratio: str = "1:1",
    default_image_resolution: str = "2K",
) -> RenderOptions:
    additional_info = data.get("additional_info", {})
    if not isinstance(additional_info, dict):
        additional_info = {}
    return RenderOptions(
        aspect_ratio=str(additional_info.get("rounded_ratio", default_aspect_ratio)),
        image_resolution=str(additional_info.get("image_resolution", default_image_resolution)),
    )


class PipelineState:
    """Thin typed wrapper around the mutable pipeline result dict."""

    def __init__(self, data: MutableMapping[str, Any], task_name: str):
        self.data = data
        self.task_name = normalize_task_name(task_name)

    @property
    def current_critic_round(self) -> int:
        return int(self.data.get("current_critic_round", 0) or 0)

    @current_critic_round.setter
    def current_critic_round(self, round_idx: int) -> None:
        self.data["current_critic_round"] = int(round_idx)

    @property
    def eval_image_field(self) -> str | None:
        value = self.data.get("eval_image_field")
        return value if isinstance(value, str) else None

    @eval_image_field.setter
    def eval_image_field(self, image_key: str | None) -> None:
        self.data["eval_image_field"] = image_key

    @property
    def max_critic_rounds(self) -> int:
        return int(self.data.get("max_critic_rounds", 0) or 0)

    def planner_desc_key(self) -> str:
        return planner_desc_key(self.task_name)

    def stylist_desc_key(self) -> str:
        return stylist_desc_key(self.task_name)

    def critic_desc_key(self, round_idx: int) -> str:
        return critic_desc_key(self.task_name, round_idx)

    def critic_suggestions_key(self, round_idx: int) -> str:
        return critic_suggestions_key(self.task_name, round_idx)

    def critic_status_key(self, round_idx: int) -> str:
        return critic_status_key(self.task_name, round_idx)

    def critic_raw_response_key(self, round_idx: int) -> str:
        return critic_raw_response_key(self.task_name, round_idx)

    @staticmethod
    def image_key(desc_key: str) -> str:
        return image_key_for_desc(desc_key)

    @staticmethod
    def mime_key(desc_key: str) -> str:
        return mime_key_for_desc(desc_key)

    @staticmethod
    def code_key(desc_key: str) -> str:
        return code_key_for_desc(desc_key)

    @staticmethod
    def plot_exec_key(desc_key: str) -> str:
        return plot_exec_key_for_desc(desc_key)

    def current_desc_key_for_critic(self, source: str, round_idx: int) -> str:
        if round_idx == 0:
            if source == "stylist":
                return self.stylist_desc_key()
            if source == "planner":
                return self.planner_desc_key()
            raise ValueError(f"Invalid source '{source}'. Must be 'stylist' or 'planner'.")
        return self.critic_desc_key(round_idx - 1)

    def available_critic_rounds(self) -> list[int]:
        return get_available_critic_rounds(self.data, self.task_name)
