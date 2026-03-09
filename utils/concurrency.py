"""Shared concurrency heuristics for demo/CLI workflows."""

from __future__ import annotations

from utils.pipeline_state import normalize_task_name


def compute_effective_concurrency(
    concurrency_mode: str,
    max_concurrent: int,
    total_candidates: int,
    *,
    task_name: str = "diagram",
    retrieval_setting: str = "auto",
    exp_mode: str = "dev_planner_critic",
    provider: str = "gemini",
) -> int:
    """Compute a practical concurrency limit for the current workload."""
    safe_max = max(1, int(max_concurrent))
    safe_total = max(1, int(total_candidates))
    requested = min(safe_max, safe_total)

    if str(concurrency_mode or "").strip().lower() != "auto":
        return requested

    normalized_task = normalize_task_name(task_name)
    normalized_retrieval = str(retrieval_setting or "auto").strip().lower()
    normalized_mode = str(exp_mode or "").strip().lower()
    normalized_provider = str(provider or "gemini").strip().lower()

    recommended = requested

    retrieval_caps = {
        "auto-full": 2,
        "auto": 4,
        "manual": 6,
        "random": 6,
        "none": 8,
    }
    recommended = min(recommended, retrieval_caps.get(normalized_retrieval, 4))

    if normalized_task == "diagram":
        recommended = min(recommended, 4 if normalized_provider == "gemini" else 3)
    else:
        recommended = min(recommended, 6)

    if normalized_mode in {"dev_full", "demo_full"}:
        recommended = min(recommended, 3)
    elif "critic" in normalized_mode:
        recommended = min(recommended, 4)
    elif normalized_mode == "vanilla":
        recommended = min(recommended, 6)

    return max(1, recommended)
