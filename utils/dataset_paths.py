"""Shared helpers for dataset-aware path resolution."""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path
from typing import Any

from utils.pipeline_state import normalize_task_name


DEFAULT_DATASET_NAME = "PaperBananaBench"


def get_repo_root(work_dir: str | os.PathLike[str] | None = None) -> Path:
    if work_dir is not None:
        return Path(work_dir).resolve()
    return Path(__file__).resolve().parent.parent


def get_data_root(work_dir: str | os.PathLike[str] | None = None) -> Path:
    return get_repo_root(work_dir) / "data"


def normalize_dataset_name(
    dataset_name: str | None,
    *,
    default: str = DEFAULT_DATASET_NAME,
) -> str:
    normalized = str(dataset_name or "").strip()
    return normalized or default


def resolve_dataset_name(
    data: dict[str, Any] | None = None,
    *,
    default: str = DEFAULT_DATASET_NAME,
) -> str:
    if isinstance(data, dict):
        return normalize_dataset_name(data.get("dataset_name"), default=default)
    return normalize_dataset_name("", default=default)


def get_dataset_dir(
    dataset_name: str | None,
    *,
    work_dir: str | os.PathLike[str] | None = None,
) -> Path:
    return get_data_root(work_dir) / normalize_dataset_name(dataset_name)


def get_task_data_dir(
    dataset_name: str | None,
    task_name: str,
    *,
    work_dir: str | os.PathLike[str] | None = None,
) -> Path:
    return get_dataset_dir(dataset_name, work_dir=work_dir) / normalize_task_name(task_name)


def get_dataset_split_path(
    dataset_name: str | None,
    task_name: str,
    split_name: str,
    *,
    work_dir: str | os.PathLike[str] | None = None,
) -> Path:
    return get_task_data_dir(dataset_name, task_name, work_dir=work_dir) / f"{split_name}.json"


def get_reference_file_path(
    dataset_name: str | None,
    task_name: str,
    *,
    work_dir: str | os.PathLike[str] | None = None,
) -> Path:
    return get_task_data_dir(dataset_name, task_name, work_dir=work_dir) / "ref.json"


def get_manual_reference_file_path(
    dataset_name: str | None,
    task_name: str,
    *,
    work_dir: str | os.PathLike[str] | None = None,
) -> Path:
    return get_task_data_dir(dataset_name, task_name, work_dir=work_dir) / "agent_selected_12.json"


def resolve_data_asset_path(
    raw_path: str | None,
    task_name: str,
    *,
    dataset_name: str | None = None,
    work_dir: str | os.PathLike[str] | None = None,
    results_path: str | os.PathLike[str] | None = None,
) -> Path | None:
    if not raw_path:
        return None

    repo_root = get_repo_root(work_dir)
    normalized_task = normalize_task_name(task_name)
    normalized_dataset = normalize_dataset_name(dataset_name)
    data_root = get_data_root(repo_root)

    candidates = [Path(raw_path)]
    if results_path:
        candidates.append(Path(results_path).resolve().parent / raw_path)

    candidates.extend(
        [
            get_task_data_dir(normalized_dataset, normalized_task, work_dir=repo_root) / raw_path,
            get_dataset_dir(normalized_dataset, work_dir=repo_root) / raw_path,
            data_root / raw_path,
        ]
    )

    if normalized_dataset != DEFAULT_DATASET_NAME:
        candidates.extend(
            [
                get_task_data_dir(
                    DEFAULT_DATASET_NAME,
                    normalized_task,
                    work_dir=repo_root,
                )
                / raw_path,
                get_dataset_dir(DEFAULT_DATASET_NAME, work_dir=repo_root) / raw_path,
            ]
        )

    if dataset_name is None and data_root.exists():
        for dataset_dir in sorted(data_root.iterdir()):
            if dataset_dir.is_dir():
                candidates.append(dataset_dir / normalized_task / raw_path)

    seen: set[str] = set()
    for candidate in candidates:
        candidate_path = Path(candidate)
        try:
            dedupe_key = str(candidate_path.resolve(strict=False)).lower()
        except Exception:
            dedupe_key = str(candidate_path).lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        if candidate_path.exists():
            return candidate_path
        for variant_name in _filename_variants(candidate_path.name):
            variant_path = candidate_path.with_name(variant_name)
            if variant_path.exists():
                return variant_path
    return None


def _filename_variants(name: str) -> list[str]:
    """
    文件名的等价写法：Unicode NFC/NFD 规范化，以及 Windows 上解压未标记 UTF-8 的 zip 时
    产生的 cp437 乱码名（例如 "inference‑time" 变成 "inferenceΓÇætime"）。
    """
    if not name or name.isascii():
        return []
    variants: list[str] = []
    for value in (
        unicodedata.normalize("NFC", name),
        unicodedata.normalize("NFD", name),
        name.encode("utf-8").decode("cp437"),
    ):
        if value != name and value not in variants:
            variants.append(value)
    return variants
