import asyncio
import importlib
import sys
import time
import types
import unittest
from io import BytesIO

from PIL import Image


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
        self.image_calls = 0
        self.download_calls = 0

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
        self.image_calls += 1

    def download_button(self, *args, **kwargs):
        self.download_calls += 1
        return False

    def expander(self, *args, **kwargs):
        return _DummyContextManager()

    def columns(self, spec, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [_DummyContextManager() for _ in range(count)]


class RefineBackgroundJobTest(unittest.TestCase):
    def _wait_for_terminal_snapshot(self, job_id: str, timeout: float = 5.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            snapshot = demo.get_refine_job_snapshot(job_id)
            if snapshot and snapshot.get("status") in {"completed", "cancelled", "failed"}:
                return snapshot
            time.sleep(0.05)
        self.fail(f"refine job {job_id} did not finish within {timeout}s")

    def test_background_refine_job_records_results(self):
        original = demo.refine_images_with_count

        async def fake_refine_images_with_count(**kwargs):
            await asyncio.sleep(0.01)
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
        finally:
            demo.refine_images_with_count = original
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
            image_model_name="gemini-3.1-flash-image-preview",
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
                uploaded_file=_UploadedFile(),
                fallback_resolution="2K",
                fallback_provider="gemini",
                fallback_image_model_name="gemini-3.1-flash-image-preview",
            )
        finally:
            demo.st = original_st

        self.assertIn("## 🎨 精修结果", fake_st.markdown_calls)
        self.assertGreaterEqual(fake_st.image_calls, 1)
        self.assertGreaterEqual(fake_st.download_calls, 1)


if __name__ == "__main__":
    unittest.main()
