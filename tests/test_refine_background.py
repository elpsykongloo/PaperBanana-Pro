import asyncio
import importlib
import sys
import tempfile
import time
import types
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from utils.demo_job_store import get_job_store_root

if "streamlit" not in sys.modules:
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.set_page_config = lambda *args, **kwargs: None
    fake_streamlit.session_state = {}
    sys.modules["streamlit"] = fake_streamlit

demo = importlib.import_module("demo")


def _build_png_bytes() -> bytes:
    image = Image.new("RGB", (8, 8), color=(12, 34, 56))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _DummyContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeRenderStreamlit:
    def __init__(self):
        self.session_state = {}
        self.markdown_calls = []
        self.image_calls = []
        self.download_calls = 0
        self.columns_calls = []

    def divider(self):
        return None

    def markdown(self, text, **kwargs):
        self.markdown_calls.append(text)

    def caption(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def image(self, *args, **kwargs):
        self.image_calls.append({"args": args, "kwargs": dict(kwargs)})

    def download_button(self, *args, **kwargs):
        self.download_calls += 1
        return False

    def expander(self, *args, **kwargs):
        return _DummyContextManager()

    def columns(self, spec, **kwargs):
        normalized_spec = spec if isinstance(spec, int) else list(spec)
        self.columns_calls.append(
            {
                "spec": normalized_spec,
                "kwargs": dict(kwargs),
            }
        )
        count = spec if isinstance(spec, int) else len(spec)
        return [_DummyContextManager() for _ in range(count)]


class RefineBackgroundJobTest(unittest.TestCase):
    def setUp(self):
        self._drain_background_jobs()
        self._original_repo_root = demo.REPO_ROOT
        self._repo_root_tempdir = tempfile.TemporaryDirectory()
        demo.REPO_ROOT = Path(self._repo_root_tempdir.name)
        job_store_root = get_job_store_root(base_dir=demo.REPO_ROOT)
        for child in job_store_root.glob("*"):
            if child.is_file():
                child.unlink()

    def tearDown(self):
        try:
            self._drain_background_jobs()
        finally:
            demo.REPO_ROOT = self._original_repo_root
            self._repo_root_tempdir.cleanup()

    def _drain_background_jobs(self, timeout: float = 5.0):
        generation_ids = list(demo.GENERATION_JOBS.keys())
        refine_ids = list(demo.REFINE_JOBS.keys())

        for job_id in generation_ids:
            job = demo.get_generation_job(job_id)
            if job and job.future is not None and not job.future.done():
                demo.request_generation_job_cancel(job_id)
        for job_id in refine_ids:
            job = demo.get_refine_job(job_id)
            if job and job.future is not None and not job.future.done():
                demo.request_refine_job_cancel(job_id)

        deadline = time.time() + timeout
        for job_id in generation_ids:
            job = demo.get_generation_job(job_id)
            if job and job.future is not None:
                remaining = max(0.01, deadline - time.time())
                job.future.result(timeout=remaining)
            demo.clear_generation_job(job_id)
        for job_id in refine_ids:
            job = demo.get_refine_job(job_id)
            if job and job.future is not None:
                remaining = max(0.01, deadline - time.time())
                job.future.result(timeout=remaining)
            demo.clear_refine_job(job_id)

    def _wait_for_terminal_snapshot(self, job_id: str, timeout: float = 5.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            snapshot = demo.get_refine_job_snapshot(job_id)
            if (
                snapshot
                and snapshot.get("status") in {"completed", "cancelled", "failed"}
                and snapshot.get("worker_done", False)
            ):
                return snapshot
            time.sleep(0.05)
        self.fail(f"refine job {job_id} did not finish within {timeout}s")

    def test_background_refine_job_records_results(self):
        original = demo.refine_images_with_count

        async def fake_refine_images_with_count(status_callback=None, progress_callback=None, event_callback=None, **kwargs):
            if event_callback:
                event_callback(
                    {
                        "kind": "job",
                        "status": "running",
                        "message": "[精修][任务 1] 开始请求，模型=test-image-model",
                        "job_type": "refine",
                        "model": "test-image-model",
                    }
                )
            elif status_callback:
                status_callback("[精修][任务 1] 开始请求，模型=test-image-model")
            await asyncio.sleep(0.01)
            if progress_callback:
                progress_callback(1, 1)
            return [(b"image-bytes", "ok")]

        demo.refine_images_with_count = fake_refine_images_with_count
        try:
            job_id = demo.start_refine_background_job(
                image_bytes=b"input",
                edit_prompt="make it better",
                num_images=1,
                aspect_ratio="16:9",
                image_size="2K",
                api_key="local-test-key",
                provider="gemini",
                image_model_name="gemini-3.1-flash-image-preview",
                input_mime_type="image/png",
            )
            snapshot = self._wait_for_terminal_snapshot(job_id)

            self.assertEqual(snapshot["status"], "completed")
            self.assertEqual(len(snapshot["refined_images"]), 1)
            self.assertTrue(snapshot["event_history"])
            self.assertTrue(
                any("模型=test-image-model" in line for line in snapshot["status_history"])
            )
        finally:
            demo.refine_images_with_count = original
            demo.clear_refine_job(job_id)

    def test_background_refine_job_passes_connection_metadata_to_runtime(self):
        original_resolve = demo.resolve_runtime_settings
        original_refine = demo.refine_images_with_count
        captured = {}
        job_id = None

        demo.resolve_runtime_settings = lambda *args, **kwargs: types.SimpleNamespace(
            api_key="local-test-key",
            provider="evolink",
            connection_id="custom-image-gateway",
            provider_display_name="自定义图像网关",
            image_model_name="custom-image-model",
            base_url="https://example.com/v1",
            extra_headers={"X-Test": "demo"},
        )

        async def fake_refine_images_with_count(**kwargs):
            captured.update(kwargs)
            return [(b"image-bytes", "ok")]

        demo.refine_images_with_count = fake_refine_images_with_count
        try:
            job_id = demo.start_refine_background_job(
                image_bytes=b"input",
                edit_prompt="make it better",
                num_images=1,
                aspect_ratio="16:9",
                image_size="2K",
                api_key="local-test-key",
                provider="custom-image-gateway",
                connection_id="custom-image-gateway",
                image_model_name="custom-image-model",
                base_url="https://example.com/v1",
                extra_headers={"X-Test": "demo"},
                input_mime_type="image/png",
            )
            snapshot = self._wait_for_terminal_snapshot(job_id)

            self.assertEqual(snapshot["connection_id"], "custom-image-gateway")
            self.assertEqual(snapshot["provider_display_name"], "自定义图像网关")
            self.assertEqual(captured["connection_id"], "custom-image-gateway")
            self.assertEqual(captured["base_url"], "https://example.com/v1")
            self.assertEqual(captured["extra_headers"], {"X-Test": "demo"})
        finally:
            demo.resolve_runtime_settings = original_resolve
            demo.refine_images_with_count = original_refine
            if job_id:
                demo.clear_refine_job(job_id)

    def test_request_cancel_marks_job_cancelled(self):
        original = demo.refine_images_with_count

        async def fake_refine_images_with_count(cancel_check=None, **kwargs):
            for _ in range(50):
                if cancel_check and cancel_check():
                    return [(None, "⛔ 已取消精修任务")]
                await asyncio.sleep(0.01)
            return [(b"late-result", "ok")]

        demo.refine_images_with_count = fake_refine_images_with_count
        try:
            job_id = demo.start_refine_background_job(
                image_bytes=b"input",
                edit_prompt="make it better",
                num_images=1,
                aspect_ratio="16:9",
                image_size="2K",
                api_key="local-test-key",
                provider="gemini",
                image_model_name="gemini-3.1-flash-image-preview",
                input_mime_type="image/png",
            )
            demo.request_refine_job_cancel(job_id)
            snapshot = self._wait_for_terminal_snapshot(job_id)

            self.assertEqual(snapshot["status"], "cancelled")
            self.assertTrue(snapshot["cancel_requested"])
        finally:
            demo.refine_images_with_count = original
            demo.clear_refine_job(job_id)

    def test_background_refine_job_marks_all_failures_as_failed(self):
        original = demo.refine_images_with_count

        async def fake_refine_images_with_count(**kwargs):
            await asyncio.sleep(0.01)
            return [(None, "❌ mock refine failure")]

        demo.refine_images_with_count = fake_refine_images_with_count
        try:
            job_id = demo.start_refine_background_job(
                image_bytes=b"input",
                edit_prompt="make it better",
                num_images=1,
                aspect_ratio="16:9",
                image_size="2K",
                api_key="local-test-key",
                provider="gemini",
                image_model_name="gemini-3.1-flash-image-preview",
                input_mime_type="image/png",
            )
            snapshot = self._wait_for_terminal_snapshot(job_id)

            self.assertEqual(snapshot["status"], "failed")
            self.assertEqual(len(snapshot["refined_images"]), 0)
            self.assertEqual(len(snapshot["failed_results"]), 1)
            self.assertIn("mock refine failure", snapshot["error"])
        finally:
            demo.refine_images_with_count = original
            demo.clear_refine_job(job_id)

    def test_refine_images_with_count_uses_module_runtime_helpers(self):
        original_resolve = demo.resolve_runtime_settings
        original_build = demo.build_runtime_context
        original_generation_utils = demo.generation_utils
        original_refine_one = demo.refine_image_with_nanoviz

        runtime_context = object()
        closed_contexts = []

        class _FakeRuntimeContextManager:
            def __init__(self, ctx):
                self.ctx = ctx

            def __enter__(self):
                return self.ctx

            def __exit__(self, exc_type, exc, tb):
                return False

        class _FakeGenerationUtils:
            @staticmethod
            def use_runtime_context(ctx):
                return _FakeRuntimeContextManager(ctx)

            @staticmethod
            async def close_runtime_context(ctx):
                closed_contexts.append(ctx)

        async def fake_refine_image_with_nanoviz(*, runtime_context=None, **kwargs):
            self.assertIs(runtime_context, runtime_context_obj)
            return b"refined", "ok"

        runtime_context_obj = runtime_context
        demo.resolve_runtime_settings = lambda *args, **kwargs: types.SimpleNamespace(
            api_key="local-test-key",
            provider="gemini",
            connection_id="gemini",
            provider_display_name="Gemini",
            image_model_name="gemini-3.1-flash-image-preview",
            base_url="",
            extra_headers={},
        )
        demo.build_runtime_context = lambda *args, **kwargs: runtime_context_obj
        demo.generation_utils = _FakeGenerationUtils()
        demo.refine_image_with_nanoviz = fake_refine_image_with_nanoviz

        try:
            results = asyncio.run(
                demo.refine_images_with_count(
                    image_bytes=b"input",
                    edit_prompt="make it better",
                    num_images=1,
                    aspect_ratio="16:9",
                    image_size="2K",
                    api_key="local-test-key",
                    provider="gemini",
                    image_model_name="gemini-3.1-flash-image-preview",
                    input_mime_type="image/png",
                )
            )
            self.assertEqual(results, [(b"refined", "ok")])
            self.assertEqual(closed_contexts, [runtime_context_obj])
        finally:
            demo.resolve_runtime_settings = original_resolve
            demo.build_runtime_context = original_build
            demo.generation_utils = original_generation_utils
            demo.refine_image_with_nanoviz = original_refine_one

    def test_refine_image_with_nanoviz_routes_openai_to_images_edit(self):
        original_resolve = demo.resolve_runtime_settings
        original_generation_utils = demo.generation_utils
        original_use_runtime_context = original_generation_utils.use_runtime_context
        original_close_runtime_context = original_generation_utils.close_runtime_context
        original_call_openai = original_generation_utils.call_openai_image_generation_with_retry_async
        captured = {}

        class _FakeRuntimeContextManager:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        async def fake_close_runtime_context(ctx):
            return None

        async def fake_call_openai_image_generation_with_retry_async(**kwargs):
            captured.update(kwargs)
            return ["ZmFrZS1pbWFnZQ=="]

        demo.resolve_runtime_settings = lambda *args, **kwargs: types.SimpleNamespace(
            api_key="local-test-key",
            provider="openai",
            connection_id="openai",
            provider_display_name="OpenAI",
            image_model_name="gpt-image-2",
            base_url="https://api.openai.com/v1",
            extra_headers={},
        )
        original_generation_utils.use_runtime_context = lambda ctx: _FakeRuntimeContextManager()
        original_generation_utils.close_runtime_context = fake_close_runtime_context
        original_generation_utils.call_openai_image_generation_with_retry_async = fake_call_openai_image_generation_with_retry_async

        try:
            result = asyncio.run(
                demo.refine_image_with_nanoviz(
                    image_bytes=_build_png_bytes(),
                    edit_prompt="make it sharper",
                    aspect_ratio="16:9",
                    image_size="2K",
                    provider="openai",
                    image_model_name="gpt-image-2",
                    input_mime_type="image/png",
                    runtime_context=object(),
                    image_generation_options={
                        "quality": "high",
                        "size": "2304x1024",
                        "output_format": "webp",
                        "output_compression": 75,
                    },
                    max_attempts=2,
                )
            )
        finally:
            demo.resolve_runtime_settings = original_resolve
            original_generation_utils.use_runtime_context = original_use_runtime_context
            original_generation_utils.close_runtime_context = original_close_runtime_context
            original_generation_utils.call_openai_image_generation_with_retry_async = original_call_openai

        self.assertEqual(result[0], b"fake-image")
        self.assertEqual(captured["model_name"], "gpt-image-2")
        self.assertEqual(captured["provider_type"], "openai")
        self.assertEqual(captured["config"]["image_resolution"], "2K")
        self.assertEqual(captured["config"]["quality"], "high")
        self.assertEqual(captured["config"]["size"], "2304x1024")
        self.assertEqual(captured["config"]["output_format"], "webp")
        self.assertEqual(captured["config"]["output_compression"], 75)
        self.assertEqual(captured["contents"][1]["source"]["media_type"], "image/png")

    def test_refine_job_snapshot_falls_back_to_disk_store(self):
        job_id = "refine_disk_snapshot"
        job = demo.RefineJobState(
            job_id=job_id,
            provider="gemini",
            image_model_name="gemini-3.1-flash-image-preview",
            resolution="2K",
            aspect_ratio="16:9",
            num_images=1,
            input_mime_type="image/png",
            original_image_bytes=b"input-image",
        )
        demo._store_refine_job(job)
        demo.record_refine_job_event(
            job_id,
            {
                "kind": "job",
                "status": "running",
                "message": "后台精修任务已启动",
            },
        )

        demo.clear_refine_job(job_id)
        snapshot = demo.get_refine_job_snapshot(job_id)

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["job_id"], job_id)
        self.assertEqual(snapshot["snapshot_source"], "disk")
        self.assertTrue(snapshot["event_history"])

    def test_refine_activity_wrapper_prefers_fragment_runner_while_job_running(self):
        original_supports = demo.supports_streamlit_fragment
        original_get_snapshot = demo.get_refine_job_snapshot
        original_hydrate = demo.hydrate_persisted_job_snapshot
        original_fragment_runner = demo._render_refine_activity_fragment_running
        original_plain_renderer = demo._render_refine_activity_content
        demo.st.session_state["active_refine_job_id"] = "refine_fragment_running"
        calls = []

        demo.supports_streamlit_fragment = lambda: True
        demo.get_refine_job_snapshot = lambda job_id: {
            "job_id": job_id,
            "status": "running",
            "worker_done": False,
        }
        demo.hydrate_persisted_job_snapshot = lambda snapshot, job_kind: snapshot
        demo._render_refine_activity_fragment_running = lambda **kwargs: calls.append(("fragment", kwargs))
        demo._render_refine_activity_content = lambda **kwargs: calls.append(("plain", kwargs))

        try:
            demo.render_refine_activity_fragment(
                requested_images=1,
                fallback_original_bytes=b"input",
                fallback_resolution="2K",
                fallback_provider="gemini",
                fallback_image_model_name="gemini-3.1-flash-image-preview",
            )
        finally:
            demo.supports_streamlit_fragment = original_supports
            demo.get_refine_job_snapshot = original_get_snapshot
            demo.hydrate_persisted_job_snapshot = original_hydrate
            demo._render_refine_activity_fragment_running = original_fragment_runner
            demo._render_refine_activity_content = original_plain_renderer

        self.assertEqual(
            calls,
            [(
                "fragment",
                {
                    "requested_images": 1,
                    "fallback_original_bytes": b"input",
                    "fallback_resolution": "2K",
                    "fallback_provider": "gemini",
                    "fallback_image_model_name": "gemini-3.1-flash-image-preview",
                },
            )],
        )

    def test_render_refine_results_section_renders_even_with_uploaded_source(self):
        original_st = demo.st
        fake_st = _FakeRenderStreamlit()
        fake_st.session_state.update(
            {
                "refined_images": [
                    {"index": 1, "bytes": _build_png_bytes()},
                ],
                "refine_original_image_bytes": _build_png_bytes(),
                "refine_timestamp": "2026-03-10 03:00:00",
            }
        )

        class _UploadedFile:
            @staticmethod
            def getvalue():
                return _build_png_bytes()

        demo.st = fake_st
        try:
            demo.render_refine_results_section(
                fallback_original_bytes=_UploadedFile.getvalue(),
                fallback_resolution="2K",
                fallback_provider="gemini",
                fallback_image_model_name="gemini-3.1-flash-image-preview",
            )
        finally:
            demo.st = original_st

        self.assertIn("## 🎨 精修结果", fake_st.markdown_calls)
        self.assertGreaterEqual(len(fake_st.image_calls), 2)
        self.assertEqual(fake_st.columns_calls[0]["spec"], [1, 2, 1])
        self.assertEqual(
            fake_st.image_calls[0]["kwargs"].get("width"),
            demo.REFINE_ORIGINAL_PREVIEW_WIDTH,
        )
        self.assertEqual(fake_st.columns_calls[1]["spec"], [1, 2, 1])
        self.assertEqual(
            fake_st.image_calls[1]["kwargs"].get("width"),
            demo.SINGLE_REFINE_RESULT_PREVIEW_WIDTH,
        )
        self.assertGreaterEqual(fake_st.download_calls, 1)

    def test_render_refine_results_section_uses_dense_grid_for_multiple_results(self):
        original_st = demo.st
        fake_st = _FakeRenderStreamlit()
        fake_st.session_state.update(
            {
                "refined_images": [
                    {"index": 1, "bytes": _build_png_bytes()},
                    {"index": 2, "bytes": _build_png_bytes()},
                    {"index": 3, "bytes": _build_png_bytes()},
                ],
                "refine_original_image_bytes": _build_png_bytes(),
                "refine_timestamp": "2026-03-10 03:00:00",
            }
        )

        demo.st = fake_st
        try:
            demo.render_refine_results_section(
                fallback_original_bytes=b"",
                fallback_resolution="2K",
                fallback_provider="gemini",
                fallback_image_model_name="gemini-3.1-flash-image-preview",
            )
        finally:
            demo.st = original_st

        self.assertGreaterEqual(len(fake_st.columns_calls), 2)
        self.assertEqual(fake_st.columns_calls[1]["spec"], 3)
        self.assertEqual(fake_st.columns_calls[1]["kwargs"].get("gap"), "medium")
        result_image_widths = [call["kwargs"].get("width") for call in fake_st.image_calls[1:]]
        self.assertTrue(all(width == "stretch" for width in result_image_widths))


if __name__ == "__main__":
    unittest.main()
