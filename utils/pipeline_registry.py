"""Registry of supported pipeline modes and their stage layouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineSpec:
    stages: tuple[str, ...]
    eval_image_source: str | None = None
    critic_source: str | None = None
    disable_eval: bool = False


PIPELINE_SPECS: dict[str, PipelineSpec] = {
    "vanilla": PipelineSpec(
        stages=("vanilla",),
        eval_image_source="vanilla",
    ),
    "dev_planner": PipelineSpec(
        stages=("retriever", "planner", "visualizer"),
        eval_image_source="planner",
    ),
    "dev_planner_stylist": PipelineSpec(
        stages=("retriever", "planner", "stylist", "visualizer"),
        eval_image_source="stylist",
    ),
    "dev_planner_critic": PipelineSpec(
        stages=("retriever", "planner", "visualizer", "critic"),
        critic_source="planner",
    ),
    "demo_planner_critic": PipelineSpec(
        stages=("retriever", "planner", "visualizer", "critic"),
        critic_source="planner",
        disable_eval=True,
    ),
    "dev_full": PipelineSpec(
        stages=("retriever", "planner", "stylist", "visualizer", "critic"),
        critic_source="stylist",
    ),
    "demo_full": PipelineSpec(
        stages=("retriever", "planner", "stylist", "visualizer", "critic"),
        critic_source="stylist",
        disable_eval=True,
    ),
    "dev_polish": PipelineSpec(
        stages=("polish",),
        eval_image_source="polish",
    ),
    "dev_retriever": PipelineSpec(
        stages=("retriever",),
        disable_eval=True,
    ),
}


def get_pipeline_spec(exp_mode: str) -> PipelineSpec:
    try:
        return PIPELINE_SPECS[exp_mode]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment name: {exp_mode}") from exc


def get_supported_exp_modes() -> tuple[str, ...]:
    return tuple(PIPELINE_SPECS.keys())
