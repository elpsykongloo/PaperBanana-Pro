"""CLI 批处理任务的 checkpoint / resume 合同。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.result_order import get_candidate_id, prepare_input_payload, sort_results_stably


CLI_CHECKPOINT_SCHEMA = "paperbanana.cli_checkpoint"
CLI_CHECKPOINT_VERSION = 1


def checkpoint_path_for_output(output_path: str | Path) -> Path:
    return Path(output_path).with_suffix(".checkpoint.json")


def checkpoint_event_log_path(checkpoint_path: str | Path) -> Path:
    checkpoint_file = Path(checkpoint_path)
    return checkpoint_file.with_name(f"{checkpoint_file.stem}.events.jsonl")


def _normalize_timestamp(value: str | None = None) -> str:
    return value or datetime.now(timezone.utc).isoformat()


def get_result_input_index(
    result: dict[str, Any] | None,
    fallback_index: int = 0,
) -> int:
    if isinstance(result, dict):
        for key in ("input_index", "candidate_id", "id"):
            value = result.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
    return int(fallback_index)


def dedupe_results_by_input_index(
    results: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    ordered = sort_results_stably(results or [])
    deduped: dict[int, dict[str, Any]] = {}
    for fallback_index, item in enumerate(ordered):
        result_index = get_result_input_index(item, fallback_index)
        deduped[result_index] = item
    return [deduped[index] for index in sorted(deduped)]


def collect_completed_input_indices(
    results: list[dict[str, Any]] | None,
) -> list[int]:
    completed_indices = []
    for fallback_index, item in enumerate(dedupe_results_by_input_index(results)):
        completed_indices.append(get_result_input_index(item, fallback_index))
    return completed_indices


def prepare_pending_inputs(
    data_list: list[dict[str, Any]] | None,
    completed_input_indices: list[int] | set[int] | None,
) -> list[dict[str, Any]]:
    completed_index_set = {int(item) for item in (completed_input_indices or [])}
    pending_inputs = []
    for input_index, raw_item in enumerate(data_list or []):
        prepared_item = prepare_input_payload(raw_item, input_index)
        if int(prepared_item.get("input_index", input_index)) in completed_index_set:
            continue
        pending_inputs.append(prepared_item)
    return pending_inputs


def build_cli_checkpoint_payload(
    *,
    manifest: dict[str, Any],
    input_file: str | Path,
    output_file: str | Path,
    bundle_file: str | Path,
    summary_file: str | Path,
    failures_file: str | Path,
    total_inputs: int,
    results: list[dict[str, Any]] | None,
    status: str,
    error: str = "",
    resume_source: str = "",
    updated_at: str | None = None,
) -> dict[str, Any]:
    normalized_results = dedupe_results_by_input_index(results)
    completed_input_indices = collect_completed_input_indices(normalized_results)
    completed_candidate_ids = [
        get_candidate_id(item, fallback_index)
        for fallback_index, item in enumerate(normalized_results)
    ]
    return {
        "schema": CLI_CHECKPOINT_SCHEMA,
        "schema_version": CLI_CHECKPOINT_VERSION,
        "updated_at": _normalize_timestamp(updated_at),
        "status": str(status or "running"),
        "error": str(error or ""),
        "resume_source": str(resume_source or ""),
        "input_file": str(Path(input_file)),
        "output_file": str(Path(output_file)),
        "bundle_file": str(Path(bundle_file)),
        "summary_file": str(Path(summary_file)),
        "failures_file": str(Path(failures_file)),
        "total_inputs": int(total_inputs or 0),
        "result_count": len(normalized_results),
        "completed_input_indices": completed_input_indices,
        "completed_candidate_ids": completed_candidate_ids,
        "manifest": dict(manifest or {}),
    }


def write_cli_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return checkpoint_path


def read_cli_checkpoint(path: str | Path) -> dict[str, Any] | None:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CLI checkpoint 文件必须是 JSON 对象。")
    return payload


def append_cli_checkpoint_event(
    path: str | Path,
    *,
    event_type: str,
    status: str,
    message: str,
    details: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> Path:
    event_path = Path(path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": _normalize_timestamp(timestamp),
        "event_type": str(event_type or "update"),
        "status": str(status or ""),
        "message": str(message or ""),
        "details": details if isinstance(details, dict) else {},
    }
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")
    return event_path
