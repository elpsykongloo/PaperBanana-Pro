import logging
import os
import re
import unittest
from pathlib import Path

from utils.log_config import DuplicateEventFilter, RuntimeEventFormatter, setup_logging
from utils.runtime_events import create_runtime_event


class LogConfigTest(unittest.TestCase):
    def setUp(self):
        self._env_backup = {
            "PAPERBANANA_LOG_LEVEL": os.environ.get("PAPERBANANA_LOG_LEVEL"),
            "PAPERBANANA_LOG_FILE": os.environ.get("PAPERBANANA_LOG_FILE"),
            "PAPERBANANA_LOG_TO_FILE": os.environ.get("PAPERBANANA_LOG_TO_FILE"),
        }
        for key in self._env_backup:
            os.environ.pop(key, None)
        setup_logging("INFO", mode="cli", force=True)

    def tearDown(self):
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        setup_logging("INFO", mode="cli", force=True)

    def test_runtime_event_formatter_renders_single_line_with_timestamp(self):
        formatter = RuntimeEventFormatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        payload = create_runtime_event(
            level="WARNING",
            kind="retry",
            source="GenerationUtils",
            message="Gemini 请求重试",
            model="gemini-test",
            attempt=2,
        ).to_dict()
        record = logging.LogRecord(
            name="GenerationUtils",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg=payload["message"],
            args=(),
            exc_info=None,
        )
        record.paperbanana_event = payload

        text = formatter.format(record)

        self.assertRegex(text, r"^\d{2}:\d{2}:\d{2} \[WARNING\] GenerationUtils \| ")
        self.assertIn("Gemini 请求重试", text)
        self.assertNotIn("\n", text)

    def test_duplicate_event_filter_collapses_same_event_within_window(self):
        duplicate_filter = DuplicateEventFilter(window_seconds=60.0)
        payload = create_runtime_event(
            level="INFO",
            kind="job",
            source="PaperBananaDemo",
            message="同一条日志",
            candidate_id="3",
            stage="planner 规划中",
        ).to_dict()

        first = logging.LogRecord("PaperBananaDemo", logging.INFO, __file__, 1, payload["message"], (), None)
        first.paperbanana_event = payload
        second = logging.LogRecord("PaperBananaDemo", logging.INFO, __file__, 1, payload["message"], (), None)
        second.paperbanana_event = dict(payload)

        self.assertTrue(duplicate_filter.filter(first))
        self.assertFalse(duplicate_filter.filter(second))

    def test_setup_logging_force_replaces_handlers_without_accumulating(self):
        setup_logging("INFO", mode="cli", force=True)
        first_count = len(logging.getLogger().handlers)

        setup_logging("INFO", mode="cli", force=True)
        second_count = len(logging.getLogger().handlers)

        self.assertEqual(first_count, second_count)
        self.assertGreaterEqual(second_count, 1)

    def test_setup_logging_event_sink_receives_structured_events(self):
        captured_events = []
        setup_logging("INFO", mode="streamlit", force=True, event_sink=captured_events.append)
        payload = create_runtime_event(
            level="ERROR",
            kind="error",
            source="PaperBananaDemo",
            message="后台任务失败",
            job_type="generation",
            status="failed",
            details="RuntimeError: boom",
        ).to_dict()

        logging.getLogger("PaperBananaDemo").error(
            payload["message"],
            extra={"paperbanana_event": payload},
        )

        self.assertEqual(len(captured_events), 1)
        self.assertEqual(captured_events[0]["kind"], "error")
        self.assertEqual(captured_events[0]["details"], "RuntimeError: boom")


class LoggingSourceRegressionTest(unittest.TestCase):
    def test_demo_has_no_bare_print_calls(self):
        text = Path("demo.py").read_text(encoding="utf-8")
        self.assertNotRegex(text, re.compile(r"\bprint\s*\("))

    def test_streamlit_width_api_is_updated(self):
        for path in (
            Path("demo.py"),
            Path("visualize/show_pipeline_evolution.py"),
            Path("visualize/show_referenced_eval.py"),
        ):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("use_container_width", text, str(path))


if __name__ == "__main__":
    unittest.main()
