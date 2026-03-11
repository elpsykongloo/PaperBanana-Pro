# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Central logging configuration for PaperBanana."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import time
from typing import Callable, Any

from utils.runtime_events import event_summary_text, runtime_event_from_log_record


_THIRD_PARTY_QUIET = (
    "streamlit",
    "streamlit.runtime.scriptrunner_utils.script_run_context",
    "urllib3",
    "httpcore",
    "httpx",
    "google",
    "google_genai",
    "PIL",
    "watchdog",
    "tornado",
    "blinker",
    "fsevents",
    "asyncio",
    "matplotlib",
)

_ICON_REPLACEMENTS = {
    "✅": "[OK]",
    "⚠️": "[WARN]",
    "⚠": "[WARN]",
    "❌": "[ERROR]",
    "🚀": "[RUN]",
    "💾": "[SAVE]",
    "📈": "[PROGRESS]",
    "🧾": "[SUMMARY]",
    "📁": "[FILE]",
    "🔍": "[SEARCH]",
    "📝": "[PLAN]",
    "📋": "[DETAIL]",
    "⏭️": "[SKIP]",
    "⏭": "[SKIP]",
    "🔧": "[STEP]",
    "⚙️": "[CONFIG]",
    "⚙": "[CONFIG]",
    "🖼️": "[IMAGE]",
    "🖼": "[IMAGE]",
    "🎨": "[STYLE]",
    "📤": "[UPLOAD]",
    "✨": "[DONE]",
    "📊": "[DATA]",
}

_setup_done = False
_configured_mode = ""


def _normalize_console_text(text: str) -> str:
    normalized = str(text or "")
    for old, new in _ICON_REPLACEMENTS.items():
        normalized = normalized.replace(old, new)
    return normalized


class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that never crashes on Windows console encoding."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
                self.flush()
            except UnicodeEncodeError:
                encoded = msg.encode(stream.encoding or "utf-8", errors="backslashreplace")
                stream.buffer.write(encoded + self.terminator.encode(stream.encoding or "utf-8"))
                self.flush()
        except Exception:
            self.handleError(record)


class RuntimeEventFormatter(logging.Formatter):
    """Render structured runtime events as single-line high-signal logs."""

    def format(self, record: logging.LogRecord) -> str:
        event = runtime_event_from_log_record(record)
        message = event_summary_text(
            event,
            include_source=getattr(record, "paperbanana_event", None) is None,
        )
        if event.level == "DEBUG" and event.details:
            message = f"{message} | {event.details}".strip(" |")
        record.message = _normalize_console_text(message)
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        formatted = self.formatMessage(record)
        if record.exc_info:
            formatted = f"{formatted}\n{self.formatException(record.exc_info)}"
        return formatted


class DuplicateEventFilter(logging.Filter):
    """Collapse bursty duplicate log lines across reruns and noisy libraries."""

    def __init__(self, window_seconds: float = 1.5):
        super().__init__()
        self.window_seconds = max(window_seconds, 0.1)
        self._last_seen: dict[tuple[str, str, str, str, str], float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        event = runtime_event_from_log_record(record)
        key = (
            event.level,
            event.source or record.name,
            event.message,
            event.candidate_id,
            event.stage,
        )
        now = time.monotonic()
        last = self._last_seen.get(key)
        self._last_seen[key] = now
        stale_before = now - (self.window_seconds * 4)
        self._last_seen = {
            existing_key: ts
            for existing_key, ts in self._last_seen.items()
            if ts >= stale_before
        }
        return last is None or (now - last) > self.window_seconds


class EventSinkHandler(logging.Handler):
    """Forward structured runtime events to an in-process sink."""

    def __init__(self, sink: Callable[[dict[str, Any]], None]):
        super().__init__(level=logging.INFO)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = runtime_event_from_log_record(record)
            self._sink(event.to_dict())
        except Exception:
            self.handleError(record)


def _apply_third_party_policy() -> None:
    for prefix in _THIRD_PARTY_QUIET:
        level = logging.ERROR if prefix == "streamlit.runtime.scriptrunner_utils.script_run_context" else logging.WARNING
        logging.getLogger(prefix).setLevel(level)


def _resolve_file_log_path() -> Path | None:
    explicit = os.environ.get("PAPERBANANA_LOG_FILE", "").strip()
    if explicit:
        return Path(explicit)
    enabled = os.environ.get("PAPERBANANA_LOG_TO_FILE", "").strip().lower()
    if enabled in {"1", "true", "yes", "on"}:
        return Path("logs") / "paperbanana.log"
    return None


def setup_logging(
    level: str = "INFO",
    *,
    mode: str = "cli",
    force: bool = False,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
    third_party_policy: bool = True,
) -> None:
    """Configure root logging once unless force=True."""

    global _setup_done, _configured_mode
    if _setup_done and not force:
        if mode == _configured_mode and event_sink is None:
            return
        force = True

    env_level = os.environ.get("PAPERBANANA_LOG_LEVEL", "").strip().upper()
    effective_level = env_level if env_level in ("DEBUG", "INFO", "WARNING", "ERROR") else level.upper()
    log_level = getattr(logging, effective_level, logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    console_handler = SafeStreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.addFilter(DuplicateEventFilter())
    console_handler.setFormatter(
        RuntimeEventFormatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root_logger.addHandler(console_handler)

    if event_sink is not None:
        sink_handler = EventSinkHandler(event_sink)
        sink_handler.setLevel(log_level)
        sink_handler.addFilter(DuplicateEventFilter())
        root_logger.addHandler(sink_handler)

    file_log_path = _resolve_file_log_path()
    if file_log_path is not None:
        file_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            RuntimeEventFormatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

    if third_party_policy:
        _apply_third_party_policy()

    _setup_done = True
    _configured_mode = mode


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
