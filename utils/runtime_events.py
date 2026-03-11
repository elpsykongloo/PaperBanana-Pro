"""Structured runtime events shared by CLI, demo, and background jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import logging
from typing import Any


RUNTIME_EVENT_KINDS = {
    "job",
    "stage",
    "retry",
    "preview_ready",
    "candidate_result",
    "warning",
    "error",
    "artifact",
}


def _normalize_level_name(level: str | int | None) -> str:
    if isinstance(level, int):
        return logging.getLevelName(level)
    if not level:
        return "INFO"
    normalized = str(level).strip().upper()
    return normalized if normalized in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "INFO"


def _normalize_kind(kind: str | None, *, level: str) -> str:
    normalized = str(kind or "").strip().lower()
    if normalized in RUNTIME_EVENT_KINDS:
        return normalized
    if level in {"ERROR", "CRITICAL"}:
        return "error"
    if level == "WARNING":
        return "warning"
    return "job"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = str(value)
        except Exception:
            text = repr(value)
    return text.replace("\x00", "\\x00")


@dataclass(frozen=True)
class RuntimeEvent:
    ts: str
    level: str
    kind: str
    source: str
    message: str
    job_type: str = ""
    candidate_id: str = ""
    stage: str = ""
    status: str = ""
    provider: str = ""
    model: str = ""
    attempt: int | None = None
    error_code: int | None = None
    preview_image: str = ""
    preview_mime_type: str = ""
    preview_label: str = ""
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_runtime_event(
    *,
    level: str | int = "INFO",
    kind: str = "job",
    source: str,
    message: str,
    job_type: str = "",
    candidate_id: Any = "",
    stage: str = "",
    status: str = "",
    provider: str = "",
    model: str = "",
    attempt: int | None = None,
    error_code: int | None = None,
    preview_image: str = "",
    preview_mime_type: str = "",
    preview_label: str = "",
    details: Any = "",
    ts: str | None = None,
) -> RuntimeEvent:
    level_name = _normalize_level_name(level)
    return RuntimeEvent(
        ts=str(ts or datetime.now().strftime("%H:%M:%S")),
        level=level_name,
        kind=_normalize_kind(kind, level=level_name),
        source=_safe_text(source) or "PaperBanana",
        message=_safe_text(message),
        job_type=_safe_text(job_type),
        candidate_id=_safe_text(candidate_id),
        stage=_safe_text(stage),
        status=_safe_text(status),
        provider=_safe_text(provider),
        model=_safe_text(model),
        attempt=attempt if isinstance(attempt, int) else None,
        error_code=error_code if isinstance(error_code, int) else None,
        preview_image=_safe_text(preview_image),
        preview_mime_type=_safe_text(preview_mime_type),
        preview_label=_safe_text(preview_label),
        details=_safe_text(details),
    )


def coerce_runtime_event(value: Any, *, default_source: str = "PaperBanana") -> RuntimeEvent:
    if isinstance(value, RuntimeEvent):
        return value

    if isinstance(value, dict):
        data = dict(value)
        details = data.get("details", "")
        if data.get("error") and not details:
            details = data.get("error")
        return create_runtime_event(
            ts=data.get("ts"),
            level=data.get("level", "INFO"),
            kind=data.get("kind", "job"),
            source=data.get("source", default_source),
            message=data.get("message", ""),
            job_type=data.get("job_type", ""),
            candidate_id=data.get("candidate_id", ""),
            stage=data.get("stage", ""),
            status=data.get("status", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            attempt=data.get("attempt"),
            error_code=data.get("error_code"),
            preview_image=data.get("preview_image", ""),
            preview_mime_type=data.get("preview_mime_type", ""),
            preview_label=data.get("preview_label", ""),
            details=details,
        )

    return create_runtime_event(
        source=default_source,
        message=_safe_text(value),
    )


def runtime_event_from_log_record(record: logging.LogRecord) -> RuntimeEvent:
    payload = getattr(record, "paperbanana_event", None)
    if payload is not None:
        return coerce_runtime_event(payload, default_source=record.name)
    return create_runtime_event(
        ts=datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
        level=record.levelno,
        kind=getattr(record, "paperbanana_kind", ""),
        source=getattr(record, "paperbanana_source", record.name),
        message=record.getMessage(),
        job_type=getattr(record, "paperbanana_job_type", ""),
        candidate_id=getattr(record, "paperbanana_candidate_id", ""),
        stage=getattr(record, "paperbanana_stage", ""),
        status=getattr(record, "paperbanana_status", ""),
        provider=getattr(record, "paperbanana_provider", ""),
        model=getattr(record, "paperbanana_model", ""),
        attempt=getattr(record, "paperbanana_attempt", None),
        error_code=getattr(record, "paperbanana_error_code", None),
        preview_image=getattr(record, "paperbanana_preview_image", ""),
        preview_mime_type=getattr(record, "paperbanana_preview_mime_type", ""),
        preview_label=getattr(record, "paperbanana_preview_label", ""),
        details=getattr(record, "paperbanana_details", ""),
    )


def event_summary_text(event: RuntimeEvent, *, include_source: bool = True) -> str:
    summary = event.message.strip()
    if event.level == "DEBUG" and event.details:
        summary = f"{summary} | {event.details}".strip(" |")
    if include_source and event.source:
        return f"[{event.source}] {summary}" if summary else f"[{event.source}]"
    return summary

