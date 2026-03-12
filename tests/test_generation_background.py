import asyncio
import base64
import importlib
import json
import logging
import sys
import tempfile
import time
import types
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from utils.demo_job_store import get_job_store_root, get_ui_state_path
from utils.result_bundle import build_run_manifest, write_result_bundle


if "streamlit" not in sys.modules:
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.set_page_config = lambda *args, **kwargs: None
    fake_streamlit.session_state = {}
    sys.modules["streamlit"] = fake_streamlit

demo = importlib.import_module("demo")


def _build_png_base64() -> str:
    image = Image.new("RGB", (8, 8), color=(12, 34, 56))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class GenerationBackgroundJobTest(unittest.TestCase):
    def setUp(self):
        demo.st.session_state.clear()
        with demo.DEMO_UI_STATE_LOCK:
            demo.DEMO_UI_STATE.clear()
        ui_state_path = get_ui_state_path(base_dir=demo.REPO_ROOT)
        if ui_state_path.exists():
            ui_state_path.unlink()
        job_store_root = get_job_store_root(base_dir=demo.REPO_ROOT)
        for child in job_store_root.glob("*"):
            if child.is_file():
                child.unlink()

    def test_background_job_runtime_is_shared_resource(self):
        runtime_a = demo.get_background_job_runtime()
        runtime_b = demo.get_background_job_runtime()

        self.assertIs(runtime_a, runtime_b)
        self.assertIs(runtime_a["generation_jobs"], demo.GENERATION_JOBS)
        self.assertIs(runtime_a["refine_jobs"], demo.REFINE_JOBS)
        self.assertIs(runtime_a["demo_ui_state"], demo.DEMO_UI_STATE)

    def test_background_job_runtime_hot_reload_normalizes_missing_keys(self):
        stale_runtime = {
            "generation_executor": object(),
            "generation_jobs_lock": object(),
            "generation_jobs": {},
            "refine_executor": object(),
            "refine_jobs_lock": object(),
            "refine_jobs": {},
        }

        normalized = demo._normalize_background_job_runtime(stale_runtime)

        self.assertIn("demo_ui_state_lock", normalized)
        self.assertIn("demo_ui_state", normalized)

    def test_demo_ui_state_round_trip_survives_session_reset(self):
        demo.st.session_state["tab1_provider"] = "gemini"
        demo.st.session_state["tab1_api_key"] = "runtime-key"
        demo.st.session_state["tab1_model_name"] = "gemini-3.1-pro-preview"
        demo.st.session_state["tab1_model_name_selector"] = demo.CUSTOM_MODEL_OPTION
        demo.st.session_state["tab1_model_name_custom"] = "custom-text-model"
        demo.st.session_state["workspace_mode"] = "✨ 精修图像"
        demo.st.session_state["tab1_curated_profile"] = "paper-profile"
        demo.st.session_state["tab1_curated_profile_input"] = "paper-profile"
        demo.st.session_state["tab1_num_candidates"] = 8
        demo.st.session_state["refine_staged_image_bytes"] = b"preview-bytes"
        demo.st.session_state["active_generation_job_id"] = "generate_demo_running"

        demo.persist_demo_ui_state()
        with demo.DEMO_UI_STATE_LOCK:
            demo.DEMO_UI_STATE.clear()
        demo.st.session_state.clear()

        demo.restore_persisted_demo_ui_state()

        self.assertEqual(demo.st.session_state["tab1_provider"], "gemini")
        self.assertEqual(demo.st.session_state["tab1_model_name"], "gemini-3.1-pro-preview")
        self.assertEqual(demo.st.session_state["tab1_model_name_selector"], demo.CUSTOM_MODEL_OPTION)
        self.assertEqual(demo.st.session_state["tab1_model_name_custom"], "custom-text-model")
        self.assertEqual(demo.st.session_state["workspace_mode"], "✨ 精修图像")
        self.assertEqual(demo.st.session_state["tab1_curated_profile"], "paper-profile")
        self.assertEqual(demo.st.session_state["tab1_curated_profile_input"], "paper-profile")
        self.assertEqual(demo.st.session_state["tab1_num_candidates"], 8)
        self.assertEqual(demo.st.session_state["refine_staged_image_bytes"], b"preview-bytes")
        self.assertNotIn("tab1_api_key", demo.st.session_state)
        self.assertNotIn("active_generation_job_id", demo.st.session_state)

    def test_generation_job_snapshot_falls_back_to_disk_store(self):
        job_id = "generate_disk_snapshot"
        job = demo.GenerationJobState(
            job_id=job_id,
            dataset_name="PaperBananaBench",
            task_name="diagram",
            exp_mode="demo_planner_critic",
            retrieval_setting="none",
            curated_profile="default",
            provider="gemini",
            model_name="gemini-3.1-flash-lite-preview",
            image_model_name="gemini-3.1-flash-image-preview",
            concurrency_mode="manual",
            max_concurrent=1,
            requested_candidates=1,
            max_critic_rounds=0,
            aspect_ratio="16:9",
            image_resolution="2K",
            content="paper method",
            visual_intent="draw a pipeline",
        )
        demo._store_generation_job(job)
        demo.record_generation_job_event(
            job_id,
            {
                "kind": "job",
                "status": "running",
                "message": "后台任务已启动",
            },
        )

        demo.clear_generation_job(job_id)
        snapshot = demo.get_generation_job_snapshot(job_id)

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["job_id"], job_id)
        self.assertEqual(snapshot["snapshot_source"], "disk")
        self.assertTrue(snapshot["event_history"])

    def _wait_for_terminal_snapshot(self, job_id: str, timeout: float = 5.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            snapshot = demo.get_generation_job_snapshot(job_id)
            if snapshot and snapshot.get("status") in {"completed", "cancelled", "failed"}:
                return snapshot
            time.sleep(0.05)
        self.fail(f"generation job {job_id} did not finish within {timeout}s")

    def test_background_generation_job_records_results_and_artifacts(self):
        original_process = demo.process_parallel_candidates
        original_save = demo.save_demo_generation_artifacts

        async def fake_process_parallel_candidates(data_list, progress_callback=None, status_callback=None, **kwargs):
            if progress_callback:
                progress_callback(0, 2, 1)
            if status_callback:
                status_callback("候选 0: 开始处理")
            await asyncio.sleep(0.01)
            results = [
                {
                    "candidate_id": 0,
                    "task_name": "diagram",
                    "dataset_name": "PaperBananaBench",
                    "exp_mode": "demo_planner_critic",
                    "eval_image_field": "target_diagram_desc0_base64_jpg",
                    "target_diagram_desc0_base64_jpg": _build_png_base64(),
                },
                {
                    "candidate_id": 1,
                    "task_name": "diagram",
                    "dataset_name": "PaperBananaBench",
                    "exp_mode": "demo_planner_critic",
                    "eval_image_field": "target_diagram_desc0_base64_jpg",
                    "target_diagram_desc0_base64_jpg": _build_png_base64(),
                },
            ]
            if progress_callback:
                progress_callback(2, 2, 1)
            return results, 1

        def fake_save_demo_generation_artifacts(**kwargs):
            return {
                "summary": {"total_candidates": 2},
                "failures": [],
                "json_file": "D:/tmp/demo_generation.json",
                "bundle_file": "D:/tmp/demo_generation.bundle.json",
                "manifest": {},
            }

        demo.process_parallel_candidates = fake_process_parallel_candidates
        demo.save_demo_generation_artifacts = fake_save_demo_generation_artifacts
        job_id = None
        try:
            job_id = demo.start_generation_background_job(
                dataset_name="PaperBananaBench",
                task_name="diagram",
                exp_mode="demo_planner_critic",
                retrieval_setting="none",
                curated_profile="default",
                provider="gemini",
                api_key="local-test-key",
                model_name="gemini-3.1-flash-lite-preview",
                image_model_name="gemini-3.1-flash-image-preview",
                concurrency_mode="manual",
                max_concurrent=1,
                num_candidates=2,
                max_critic_rounds=1,
                aspect_ratio="16:9",
                image_resolution="2K",
                content="paper method",
                visual_intent="draw a pipeline",
            )
            snapshot = self._wait_for_terminal_snapshot(job_id)

            self.assertEqual(snapshot["status"], "completed")
            self.assertEqual(len(snapshot["results"]), 2)
            self.assertTrue(snapshot["event_history"])
            self.assertEqual(snapshot["json_file"], "D:/tmp/demo_generation.json")
            self.assertEqual(snapshot["bundle_file"], "D:/tmp/demo_generation.bundle.json")
            self.assertEqual(snapshot["curated_profile"], "default")
        finally:
            demo.process_parallel_candidates = original_process
            demo.save_demo_generation_artifacts = original_save
            if job_id:
                demo.clear_generation_job(job_id)

    def test_request_generation_cancel_marks_job_cancelled(self):
        original_process = demo.process_parallel_candidates
        original_save = demo.save_demo_generation_artifacts

        async def fake_process_parallel_candidates(data_list, cancel_check=None, progress_callback=None, **kwargs):
            results = []
            total = 3
            if progress_callback:
                progress_callback(0, total, 1)
            for idx in range(total):
                await asyncio.sleep(0.02)
                if cancel_check and cancel_check():
                    break
                results.append(
                    {
                        "candidate_id": idx,
                        "task_name": "diagram",
                        "dataset_name": "PaperBananaBench",
                        "exp_mode": "demo_planner_critic",
                        "eval_image_field": "target_diagram_desc0_base64_jpg",
                        "target_diagram_desc0_base64_jpg": _build_png_base64(),
                    }
                )
                if progress_callback:
                    progress_callback(len(results), total, 1)
            return results, 1

        def fake_save_demo_generation_artifacts(**kwargs):
            return {
                "summary": {"total_candidates": len(kwargs["results"])},
                "failures": [],
                "json_file": "D:/tmp/cancelled_generation.json",
                "bundle_file": "D:/tmp/cancelled_generation.bundle.json",
                "manifest": {},
            }

        demo.process_parallel_candidates = fake_process_parallel_candidates
        demo.save_demo_generation_artifacts = fake_save_demo_generation_artifacts
        job_id = None
        try:
            job_id = demo.start_generation_background_job(
                dataset_name="PaperBananaBench",
                task_name="diagram",
                exp_mode="demo_planner_critic",
                retrieval_setting="none",
                curated_profile="default",
                provider="gemini",
                api_key="local-test-key",
                model_name="gemini-3.1-flash-lite-preview",
                image_model_name="gemini-3.1-flash-image-preview",
                concurrency_mode="manual",
                max_concurrent=1,
                num_candidates=3,
                max_critic_rounds=1,
                aspect_ratio="16:9",
                image_resolution="2K",
                content="paper method",
                visual_intent="draw a pipeline",
            )
            time.sleep(0.03)
            demo.request_generation_job_cancel(job_id)
            snapshot = self._wait_for_terminal_snapshot(job_id)

            self.assertEqual(snapshot["status"], "cancelled")
            self.assertTrue(snapshot["cancel_requested"])
            self.assertLess(len(snapshot["results"]), 3)
        finally:
            demo.process_parallel_candidates = original_process
            demo.save_demo_generation_artifacts = original_save
            if job_id:
                demo.clear_generation_job(job_id)

    def test_background_generation_job_captures_stage_updates_and_logger_output(self):
        original_process = demo.process_parallel_candidates
        original_save = demo.save_demo_generation_artifacts

        async def fake_process_parallel_candidates(
            data_list,
            progress_callback=None,
            status_callback=None,
            event_callback=None,
            **kwargs,
        ):
            if progress_callback:
                progress_callback(0, 1, 1)
            if event_callback:
                event_callback(
                    {
                        "candidate_id": 0,
                        "kind": "stage",
                        "status": "running",
                        "stage": "📝 生成规划草案",
                        "message": "候选 01: 📝 生成规划草案",
                    }
                )
            elif status_callback:
                status_callback("候选 01: 📝 生成规划草案")
            logging.getLogger("PlannerAgent").info("测试规划日志已同步")
            await asyncio.sleep(0.01)
            result = {
                "candidate_id": 0,
                "task_name": "diagram",
                "dataset_name": "PaperBananaBench",
                "exp_mode": "demo_planner_critic",
                "eval_image_field": "target_diagram_desc0_base64_jpg",
                "target_diagram_desc0_base64_jpg": _build_png_base64(),
            }
            if progress_callback:
                progress_callback(1, 1, 1)
            return [result], 1

        def fake_save_demo_generation_artifacts(**kwargs):
            return {
                "summary": {"total_candidates": 1},
                "failures": [],
                "json_file": "D:/tmp/logged_generation.json",
                "bundle_file": "D:/tmp/logged_generation.bundle.json",
                "manifest": {},
            }

        demo.process_parallel_candidates = fake_process_parallel_candidates
        demo.save_demo_generation_artifacts = fake_save_demo_generation_artifacts
        job_id = None
        try:
            job_id = demo.start_generation_background_job(
                dataset_name="PaperBananaBench",
                task_name="diagram",
                exp_mode="demo_planner_critic",
                retrieval_setting="none",
                curated_profile="default",
                provider="gemini",
                api_key="local-test-key",
                model_name="gemini-3.1-flash-lite-preview",
                image_model_name="gemini-3.1-flash-image-preview",
                concurrency_mode="manual",
                max_concurrent=1,
                num_candidates=1,
                max_critic_rounds=1,
                aspect_ratio="16:9",
                image_resolution="2K",
                content="paper method",
                visual_intent="draw a pipeline",
            )
            snapshot = self._wait_for_terminal_snapshot(job_id)

            self.assertEqual(snapshot["candidate_stage_map"]["0"], "📝 生成规划草案")
            self.assertTrue(
                any(event.get("stage") == "📝 生成规划草案" for event in snapshot["event_history"])
            )
            self.assertTrue(
                any("测试规划日志已同步" in line for line in snapshot["status_history"])
            )
        finally:
            demo.process_parallel_candidates = original_process
            demo.save_demo_generation_artifacts = original_save
            if job_id:
                demo.clear_generation_job(job_id)

    def test_generation_job_records_preview_events_for_live_stream(self):
        original_process = demo.process_parallel_candidates
        original_save = demo.save_demo_generation_artifacts

        async def fake_process_parallel_candidates(
            data_list,
            progress_callback=None,
            status_callback=None,
            event_callback=None,
            result_callback=None,
            **kwargs,
        ):
            preview_b64 = _build_png_base64()
            if progress_callback:
                progress_callback(0, 1, 1)
            if event_callback:
                event_callback(
                    {
                        "candidate_id": 0,
                        "kind": "preview_ready",
                        "status": "running",
                        "stage": "🖼️ 首张预览已生成",
                        "message": "候选 01: 首张预览已生成",
                        "preview_image": preview_b64,
                        "preview_mime_type": "image/png",
                        "preview_label": "📝 规划草案",
                    }
                )
            result = {
                "candidate_id": 0,
                "task_name": "diagram",
                "dataset_name": "PaperBananaBench",
                "exp_mode": "demo_planner_critic",
                "eval_image_field": "target_diagram_desc0_base64_jpg",
                "target_diagram_desc0_base64_jpg": preview_b64,
            }
            if result_callback:
                result_callback(result)
            if progress_callback:
                progress_callback(1, 1, 1)
            return [result], 1

        def fake_save_demo_generation_artifacts(**kwargs):
            return {
                "summary": {"total_candidates": 1},
                "failures": [],
                "json_file": "D:/tmp/live_generation.json",
                "bundle_file": "D:/tmp/live_generation.bundle.json",
                "manifest": {},
            }

        demo.process_parallel_candidates = fake_process_parallel_candidates
        demo.save_demo_generation_artifacts = fake_save_demo_generation_artifacts
        job_id = None
        try:
            job_id = demo.start_generation_background_job(
                dataset_name="PaperBananaBench",
                task_name="diagram",
                exp_mode="demo_planner_critic",
                retrieval_setting="none",
                curated_profile="default",
                provider="gemini",
                api_key="local-test-key",
                model_name="gemini-3.1-flash-lite-preview",
                image_model_name="gemini-3.1-flash-image-preview",
                concurrency_mode="manual",
                max_concurrent=1,
                num_candidates=1,
                max_critic_rounds=1,
                aspect_ratio="16:9",
                image_resolution="2K",
                content="paper method",
                visual_intent="draw a pipeline",
            )
            snapshot = self._wait_for_terminal_snapshot(job_id)

            candidate_snapshot = snapshot["candidate_snapshots"]["0"]
            self.assertTrue(
                any(event.get("kind") == "preview_ready" for event in snapshot["event_history"])
            )
            self.assertEqual(candidate_snapshot["preview_label"], "✅ 最终结果预览")
            self.assertTrue(
                any(event.get("preview_label") == "📝 规划草案" for event in snapshot["event_history"])
            )
            self.assertTrue(candidate_snapshot["preview_image"])
            self.assertEqual(candidate_snapshot["status"], "completed")
        finally:
            demo.process_parallel_candidates = original_process
            demo.save_demo_generation_artifacts = original_save
            if job_id:
                demo.clear_generation_job(job_id)

    def test_generation_progress_tracks_terminal_candidates_without_ordered_yield_blocking(self):
        job_id = "generate_progress_sync"
        job = demo.GenerationJobState(
            job_id=job_id,
            dataset_name="PaperBananaBench",
            task_name="diagram",
            exp_mode="demo_planner_critic",
            retrieval_setting="auto",
            curated_profile="default",
            provider="gemini",
            model_name="gemini-3.1-flash-lite-preview",
            image_model_name="gemini-3.1-flash-image-preview",
            concurrency_mode="auto",
            max_concurrent=5,
            requested_candidates=5,
            max_critic_rounds=1,
            aspect_ratio="16:9",
            image_resolution="2K",
            content="paper method",
            visual_intent="draw a pipeline",
            progress_total=5,
        )
        demo._store_generation_job(job)
        try:
            demo.update_generation_job_progress(job_id, 1, 5, 5)
            for candidate_id in (0, 2, 3, 4):
                demo.record_generation_job_event(
                    job_id,
                    {
                        "candidate_id": candidate_id,
                        "kind": "candidate_result",
                        "status": "completed",
                        "stage": "候选流程完成",
                        "message": f"候选 {candidate_id}: 候选已完成并可展示",
                    },
                )

            snapshot = demo.get_generation_job_snapshot(job_id)
            self.assertEqual(snapshot["progress_done"], 4)
            self.assertEqual(snapshot["progress_total"], 5)

            demo.update_generation_job_progress(job_id, 2, 5, 5)
            snapshot = demo.get_generation_job_snapshot(job_id)
            self.assertEqual(snapshot["progress_done"], 4)
        finally:
            demo.clear_generation_job(job_id)

    def test_stage_candidate_for_refine_stores_session_image(self):
        result = {
            "target_diagram_desc0_base64_jpg": _build_png_base64(),
        }

        success = demo.stage_candidate_for_refine(
            result,
            candidate_id=7,
            exp_mode="demo_planner_critic",
            task_name="diagram",
        )

        self.assertTrue(success)
        self.assertTrue(demo.st.session_state["refine_staged_image_bytes"])
        self.assertEqual(demo.st.session_state["refine_staged_source_label"], "候选 08")

    def test_build_full_process_zip_contains_stage_images_and_metadata(self):
        result = {
            "candidate_id": 0,
            "task_name": "diagram",
            "exp_mode": "demo_planner_critic",
            "target_diagram_desc0": "planner desc",
            "target_diagram_desc0_base64_jpg": _build_png_base64(),
            "target_diagram_critic_desc0": "critic desc",
            "target_diagram_critic_desc0_base64_jpg": _build_png_base64(),
            "target_diagram_critic_suggestions0": "make arrows clearer",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "session.json"
            bundle_path = Path(tmp_dir) / "session.bundle.json"
            json_path.write_text("{}", encoding="utf-8")
            bundle_path.write_text("{}", encoding="utf-8")

            zip_bytes, exported_count, failures = demo.build_full_process_zip(
                [result],
                task_name="diagram",
                exp_mode="demo_planner_critic",
                dataset_name="PaperBananaBench",
                timestamp="2026-03-10 18:00:28",
                source_label="后台生成任务",
                json_file_path=json_path,
                bundle_file_path=bundle_path,
            )

        self.assertEqual(exported_count, 1)
        self.assertEqual(failures, [])
        with zipfile.ZipFile(BytesIO(zip_bytes), "r") as archive:
            names = set(archive.namelist())

        self.assertTrue(any(name.endswith("/00_运行总览/运行总览.txt") for name in names))
        self.assertTrue(any(name.endswith("/00_运行总览/原始结果.json") for name in names))
        self.assertTrue(any(name.endswith("/00_运行总览/结果Bundle.bundle.json") for name in names))
        self.assertTrue(any("候选01_学术图解/01_最终结果/01_最终图解.png" in name for name in names))
        self.assertTrue(any("候选01_学术图解/02_演化过程/01_规划器/01_阶段图像.png" in name for name in names))
        self.assertTrue(any("候选01_学术图解/02_演化过程/02_评审第01轮/04_评审建议.md" in name for name in names))
        self.assertTrue(any(name.endswith("/99_原始记录/候选完整结果.json") for name in names))

    def test_build_full_process_zip_exports_plot_code_files(self):
        result = {
            "candidate_id": 0,
            "task_name": "plot",
            "exp_mode": "demo_planner_critic",
            "target_plot_desc0": "planner desc",
            "target_plot_desc0_base64_jpg": _build_png_base64(),
            "target_plot_desc0_code": "print('planner')",
            "target_plot_critic_desc0": "critic desc",
            "target_plot_critic_desc0_base64_jpg": _build_png_base64(),
            "target_plot_critic_desc0_code": "print('critic')",
            "target_plot_critic_suggestions0": "tighten the layout",
        }

        zip_bytes, exported_count, failures = demo.build_full_process_zip(
            [result],
            task_name="plot",
            exp_mode="demo_planner_critic",
            dataset_name="PaperBananaBench",
            timestamp="2026-03-10 18:00:28",
            source_label="后台生成任务",
        )

        self.assertEqual(exported_count, 1)
        self.assertEqual(failures, [])
        with zipfile.ZipFile(BytesIO(zip_bytes), "r") as archive:
            names = set(archive.namelist())

        self.assertTrue(any("候选01_统计图/01_最终结果/03_最终Matplotlib代码.py" in name for name in names))
        self.assertTrue(any("候选01_统计图/02_演化过程/01_规划器/03_阶段代码.py" in name for name in names))
        self.assertTrue(any("候选01_统计图/02_演化过程/02_评审第01轮/03_阶段代码.py" in name for name in names))

    def test_load_generation_history_snapshot_reads_bundle_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "history.bundle.json"
            results = [
                {
                    "candidate_id": 0,
                    "dataset_name": "CustomBench",
                    "task_name": "plot",
                    "exp_mode": "demo_planner_critic",
                    "eval_image_field": "target_plot_desc0_base64_jpg",
                    "target_plot_desc0_base64_jpg": _build_png_base64(),
                }
            ]
            manifest = build_run_manifest(
                producer="demo",
                dataset_name="CustomBench",
                task_name="plot",
                exp_mode="demo_planner_critic",
                provider="gemini",
                model_name="text-model",
                image_model_name="",
                concurrency_mode="manual",
                max_concurrent=2,
                max_critic_rounds=0,
                result_count=1,
                extra={
                    "requested_candidates": 4,
                    "effective_concurrent": 2,
                    "run_status": "completed",
                },
            )
            write_result_bundle(bundle_path, results, manifest=manifest)

            snapshot = demo.load_generation_history_snapshot(bundle_path)

            self.assertEqual(snapshot["dataset_name"], "CustomBench")
            self.assertEqual(snapshot["task_name"], "plot")
            self.assertEqual(snapshot["requested_candidates"], 4)
            self.assertEqual(snapshot["effective_concurrent"], 2)
            self.assertEqual(snapshot["bundle_file"], str(bundle_path))

    def test_list_demo_bundle_files_reads_latest_history_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_root = Path(tmp_dir) / "results" / "demo" / "diagram"
            results_root.mkdir(parents=True, exist_ok=True)
            older_path = results_root / "older.bundle.json"
            newer_path = results_root / "newer.bundle.json"
            payload = {
                "schema": "paperbanana.result_bundle",
                "schema_version": 1,
                "manifest": {
                    "dataset_name": "PaperBananaBench",
                    "task_name": "diagram",
                    "exp_mode": "demo_planner_critic",
                    "provider": "gemini",
                    "result_count": 0,
                },
                "summary": {},
                "failures": [],
                "results": [],
            }
            older_path.write_text(json.dumps(payload), encoding="utf-8")
            time.sleep(0.02)
            newer_path.write_text(json.dumps(payload), encoding="utf-8")
            original_get_root = demo.get_demo_results_root
            demo.get_demo_results_root = lambda: Path(tmp_dir) / "results" / "demo"
            try:
                bundle_files = demo.list_demo_bundle_files("diagram", limit=5)
            finally:
                demo.get_demo_results_root = original_get_root

            self.assertEqual(bundle_files[0].name, "newer.bundle.json")
            self.assertEqual(bundle_files[1].name, "older.bundle.json")


if __name__ == "__main__":
    unittest.main()
