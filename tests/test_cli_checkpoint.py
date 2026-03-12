import json
import tempfile
import unittest
from pathlib import Path

import main
from utils.cli_checkpoint import (
    CLI_CHECKPOINT_SCHEMA,
    append_cli_checkpoint_event,
    build_cli_checkpoint_payload,
    checkpoint_event_log_path,
    checkpoint_path_for_output,
    prepare_pending_inputs,
    read_cli_checkpoint,
    write_cli_checkpoint,
)
from utils.result_bundle import build_run_manifest, write_result_bundle


class CliCheckpointTest(unittest.TestCase):
    def test_prepare_pending_inputs_skips_completed_input_indices(self):
        data_list = [
            {"id": "sample-0"},
            {"id": "sample-1"},
            {"id": "sample-2"},
        ]

        pending = prepare_pending_inputs(data_list, {1})

        self.assertEqual([item["input_index"] for item in pending], [0, 2])
        self.assertEqual([item["candidate_id"] for item in pending], [0, 2])

    def test_build_and_read_cli_checkpoint_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = checkpoint_path_for_output(root / "run.json")
            manifest = build_run_manifest(
                producer="cli",
                result_count=2,
                dataset_name="SmokeBench",
                task_name="diagram",
            )
            payload = build_cli_checkpoint_payload(
                manifest=manifest,
                input_file=root / "input.json",
                output_file=root / "run.json",
                bundle_file=root / "run.bundle.json",
                summary_file=root / "run.summary.json",
                failures_file=root / "run.failures.json",
                total_inputs=3,
                results=[
                    {"input_index": 0, "candidate_id": 0, "status": "ok"},
                    {"input_index": 2, "candidate_id": 2, "status": "failed"},
                ],
                status="running",
            )

            write_cli_checkpoint(checkpoint_path, payload)
            restored = read_cli_checkpoint(checkpoint_path)

            self.assertEqual(restored["schema"], CLI_CHECKPOINT_SCHEMA)
            self.assertEqual(restored["status"], "running")
            self.assertEqual(restored["completed_input_indices"], [0, 2])
            self.assertEqual(restored["result_count"], 2)

    def test_append_cli_checkpoint_event_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            event_path = checkpoint_event_log_path(Path(temp_dir) / "run.checkpoint.json")

            append_cli_checkpoint_event(
                event_path,
                event_type="checkpoint_saved",
                status="running",
                message="已写入 checkpoint",
                details={"result_count": 3},
            )

            lines = event_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["event_type"], "checkpoint_saved")
            self.assertEqual(payload["details"]["result_count"], 3)

    def test_resolve_resume_source_path_prefers_checkpoint_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / "run.json"
            bundle_path = root / "run.bundle.json"
            checkpoint_path = checkpoint_path_for_output(output_path)
            write_result_bundle(
                bundle_path,
                [
                    {
                        "input_index": 0,
                        "candidate_id": 0,
                        "dataset_name": "SmokeBench",
                        "task_name": "diagram",
                        "target_diagram_desc0_base64_jpg": "abc",
                        "eval_image_field": "target_diagram_desc0_base64_jpg",
                    }
                ],
                manifest=build_run_manifest(
                    producer="cli",
                    result_count=1,
                    dataset_name="SmokeBench",
                    task_name="diagram",
                ),
            )
            write_cli_checkpoint(
                checkpoint_path,
                build_cli_checkpoint_payload(
                    manifest=build_run_manifest(
                        producer="cli",
                        result_count=1,
                        dataset_name="SmokeBench",
                        task_name="diagram",
                    ),
                    input_file=root / "input.json",
                    output_file=output_path,
                    bundle_file=bundle_path,
                    summary_file=root / "run.summary.json",
                    failures_file=root / "run.failures.json",
                    total_inputs=1,
                    results=[{"input_index": 0, "candidate_id": 0}],
                    status="running",
                ),
            )

            resolved_path, restored_checkpoint = main.resolve_resume_source_path(
                resume_flag=True,
                resume_from="",
                checkpoint_path=checkpoint_path,
                bundle_path=bundle_path,
                output_path=output_path,
            )

            self.assertEqual(resolved_path, bundle_path)
            self.assertEqual(restored_checkpoint["bundle_file"], str(bundle_path))


if __name__ == "__main__":
    unittest.main()
