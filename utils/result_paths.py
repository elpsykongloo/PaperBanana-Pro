"""Shared helpers for resolving result-linked local assets."""

from __future__ import annotations

import os
from pathlib import Path

from utils.pipeline_state import normalize_task_name


def resolve_gt_image_path(
    raw_path: str | None,
    task_type: str,
    results_path: str | None = None,
    work_dir: str | os.PathLike[str] | None = None,
) -> Path | None:
    if not raw_path:
        return None

    normalized_task = normalize_task_name(task_type)
    repo_root = Path(work_dir).resolve() if work_dir is not None else Path(os.getcwd()).resolve()

    candidates = [Path(raw_path)]
    if results_path:
        candidates.append(Path(results_path).resolve().parent / raw_path)
    candidates.append(repo_root / "data" / "PaperBananaBench" / normalized_task / raw_path)

    for candidate in candidates:
        resolved_candidate = Path(candidate)
        if resolved_candidate.exists():
            return resolved_candidate
    return None
