import asyncio
import importlib
import sys
import time
import types
import unittest


if "streamlit" not in sys.modules:
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.set_page_config = lambda *args, **kwargs: None
    fake_streamlit.session_state = {}
    sys.modules["streamlit"] = fake_streamlit

demo = importlib.import_module("demo")


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


if __name__ == "__main__":
    unittest.main()
