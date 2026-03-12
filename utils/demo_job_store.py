"""后台任务与安全 UI 状态的持久化辅助。"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


JOB_STORE_DIRNAME = "_job_state"
UI_STATE_DIRNAME = "_ui_state"
UI_STATE_FILENAME = "session_state.json"
SNAPSHOT_SUFFIX = ".snapshot.json"
EVENT_LOG_SUFFIX = ".events.jsonl"


def _resolve_demo_results_root(base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    return root / "results" / "demo"


def get_job_store_root(base_dir: str | Path | None = None) -> Path:
    root = _resolve_demo_results_root(base_dir) / JOB_STORE_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_ui_state_path(base_dir: str | Path | None = None) -> Path:
    root = _resolve_demo_results_root(base_dir) / UI_STATE_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root / UI_STATE_FILENAME


def get_job_snapshot_path(job_id: str, base_dir: str | Path | None = None) -> Path:
    return get_job_store_root(base_dir) / f"{job_id}{SNAPSHOT_SUFFIX}"


def get_job_event_log_path(job_id: str, base_dir: str | Path | None = None) -> Path:
    return get_job_store_root(base_dir) / f"{job_id}{EVENT_LOG_SUFFIX}"


def _serialize_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {
            "__type__": "bytes",
            "data": base64.b64encode(value).decode("utf-8"),
        }
    if isinstance(value, Path):
        return {
            "__type__": "path",
            "data": str(value),
        }
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return {
            "__type__": "tuple",
            "data": [_serialize_value(item) for item in value],
        }
    return value


def _deserialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        value_type = value.get("__type__")
        if value_type == "bytes":
            encoded = str(value.get("data", "") or "")
            return base64.b64decode(encoded) if encoded else b""
        if value_type == "path":
            return Path(str(value.get("data", "") or ""))
        if value_type == "tuple":
            return tuple(_deserialize_value(item) for item in value.get("data", []))
        return {str(key): _deserialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]
    return value


def write_job_snapshot(
    job_id: str,
    snapshot: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> Path:
    snapshot_path = get_job_snapshot_path(job_id, base_dir)
    payload = _serialize_value(snapshot)
    snapshot_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return snapshot_path


def read_job_snapshot(
    job_id: str,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    snapshot_path = get_job_snapshot_path(job_id, base_dir)
    if not snapshot_path.exists():
        return None
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return _deserialize_value(payload)


def append_job_event(
    job_id: str,
    event: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> Path:
    event_path = get_job_event_log_path(job_id, base_dir)
    payload = _serialize_value(event)
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return event_path


def read_job_events(
    job_id: str,
    *,
    base_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    event_path = get_job_event_log_path(job_id, base_dir)
    if not event_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in event_path.read_text(encoding="utf-8").splitlines():
        raw = str(line or "").strip()
        if not raw:
            continue
        events.append(_deserialize_value(json.loads(raw)))
    return events


def write_ui_state(
    state_payload: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> Path:
    ui_state_path = get_ui_state_path(base_dir)
    ui_state_path.write_text(
        json.dumps(_serialize_value(state_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ui_state_path


def read_ui_state(
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    ui_state_path = get_ui_state_path(base_dir)
    if not ui_state_path.exists():
        return {}
    payload = json.loads(ui_state_path.read_text(encoding="utf-8"))
    return _deserialize_value(payload)
