import importlib
import sys
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


def make_test_png_bytes(color: tuple[int, int, int] = (12, 34, 56)) -> bytes:
    """生成一个最小可用 PNG，避免测试依赖伪造字节。"""
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


class DemoWorkbenchStateTest(unittest.TestCase):
    def setUp(self):
        demo.st.session_state.clear()

    def test_generation_example_selection_applies_once_and_queues_selector_reset(self):
        selector_key = "tab1_diagram_content_example_selector"
        editor_key = "tab1_diagram_content_editor"
        previous_key = demo._get_generation_example_selector_previous_key(selector_key)

        demo._prime_generation_example_selector_state(selector_key)
        applied = demo._apply_generation_example_selection(
            selector_key=selector_key,
            selected_value="PaperBanana 框架",
            editor_key=editor_key,
            example_name="PaperBanana 框架",
            example_value="示例方法章节",
        )

        self.assertTrue(applied)
        self.assertEqual(demo.st.session_state[editor_key], "示例方法章节")
        self.assertEqual(demo.st.session_state[previous_key], "PaperBanana 框架")
        self.assertEqual(
            demo.st.session_state["_pending_generation_widget_updates"][selector_key],
            demo.EXAMPLE_SELECTOR_NONE_OPTION,
        )
        self.assertEqual(
            demo.st.session_state["_pending_generation_widget_updates"][previous_key],
            demo.EXAMPLE_SELECTOR_NONE_OPTION,
        )

    def test_generation_example_selection_does_not_reapply_when_selector_stays_on_example(self):
        selector_key = "tab1_diagram_content_example_selector"
        editor_key = "tab1_diagram_content_editor"

        demo._prime_generation_example_selector_state(selector_key)
        first_applied = demo._apply_generation_example_selection(
            selector_key=selector_key,
            selected_value="PaperBanana 框架",
            editor_key=editor_key,
            example_name="PaperBanana 框架",
            example_value="第一次示例",
        )
        demo.st.session_state[editor_key] = "用户后续手动改写"
        second_applied = demo._apply_generation_example_selection(
            selector_key=selector_key,
            selected_value="PaperBanana 框架",
            editor_key=editor_key,
            example_name="PaperBanana 框架",
            example_value="第二次示例",
        )

        self.assertTrue(first_applied)
        self.assertFalse(second_applied)
        self.assertEqual(demo.st.session_state[editor_key], "用户后续手动改写")

    def test_generation_example_selection_ignores_stale_initial_example_state(self):
        selector_key = "tab1_diagram_content_example_selector"
        editor_key = "tab1_diagram_content_editor"
        demo.st.session_state[selector_key] = "PaperBanana 框架"

        demo._prime_generation_example_selector_state(selector_key)
        applied = demo._apply_generation_example_selection(
            selector_key=selector_key,
            selected_value="PaperBanana 框架",
            editor_key=editor_key,
            example_name="PaperBanana 框架",
            example_value="示例方法章节",
        )

        self.assertFalse(applied)
        self.assertNotIn(editor_key, demo.st.session_state)

    def test_generation_candidate_decision_filter_scopes(self):
        results = [
            {"candidate_id": 0},
            {"candidate_id": 1},
            {"candidate_id": 2},
        ]
        demo.set_generation_candidate_decision(0, "favorite")
        demo.set_generation_candidate_decision(1, "discarded")
        demo.set_generation_candidate_decision(2, "final")

        self.assertEqual(len(demo.filter_generation_results_by_scope(results, "全部候选")), 3)
        self.assertEqual(
            [item["candidate_id"] for item in demo.filter_generation_results_by_scope(results, "仅未淘汰")],
            [0, 2],
        )
        self.assertEqual(
            [item["candidate_id"] for item in demo.filter_generation_results_by_scope(results, "仅收藏")],
            [0, 2],
        )
        self.assertEqual(
            [item["candidate_id"] for item in demo.filter_generation_results_by_scope(results, "仅最终候选")],
            [2],
        )

    def test_setting_new_final_candidate_replaces_previous_final(self):
        demo.set_generation_candidate_decision(0, "final")
        demo.set_generation_candidate_decision(1, "final")

        self.assertEqual(demo.get_generation_candidate_decision(0), "default")
        self.assertEqual(demo.get_generation_candidate_decision(1), "final")

        demo.set_generation_candidate_decision(1, "discarded")
        self.assertEqual(demo.get_generation_candidate_decision(1), "discarded")
        self.assertEqual(demo.get_generation_final_candidate_token(), "")

    def test_append_refine_snapshot_to_version_history_creates_branch_versions(self):
        original_bytes = make_test_png_bytes((200, 10, 10))
        demo.stage_refine_source_image(
            original_bytes,
            input_mime_type="image/png",
            source_label="候选 01",
            default_prompt="保持语义不变",
        )
        parent_version_key = demo.ensure_refine_source_version(
            original_bytes,
            input_mime_type="image/png",
            source_label="候选 01",
        )
        demo.st.session_state["refine_active_version_key"] = parent_version_key

        created_keys = demo.append_refine_snapshot_to_version_history(
            {
                "created_at": "2026-03-12 18:00:00",
                "provider": "gemini",
                "image_model_name": "gemini-image",
                "resolution": "4K",
                "input_mime_type": "image/png",
                "original_image_bytes": original_bytes,
                "refined_images": [
                    {"index": 1, "bytes": make_test_png_bytes((10, 200, 10))},
                    {"index": 2, "bytes": make_test_png_bytes((10, 10, 200))},
                ],
            },
            edit_prompt="放大并优化布局",
        )

        self.assertEqual(len(created_keys), 2)
        self.assertEqual(demo.st.session_state["refine_active_version_key"], created_keys[0])
        history = demo.get_refine_version_history()
        child_nodes = [item for item in history if item.get("parent_version_key") == parent_version_key]
        self.assertEqual(len(child_nodes), 2)
        self.assertTrue(all(item.get("edit_prompt") == "放大并优化布局" for item in child_nodes))

    def test_stage_refine_source_image_queues_widget_state_updates(self):
        original_bytes = make_test_png_bytes((66, 77, 88))

        demo.stage_refine_source_image(
            original_bytes,
            input_mime_type="image/png",
            source_label="候选 02",
            default_prompt="保持内容不变，增强清晰度",
        )

        self.assertEqual(demo.st.session_state["refine_staged_image_bytes"], original_bytes)
        self.assertEqual(demo.st.session_state["refine_staged_source_label"], "候选 02")
        self.assertNotIn("refine_input_source", demo.st.session_state)
        self.assertEqual(
            demo.st.session_state["_pending_refine_widget_updates"]["refine_input_source"],
            "候选方案",
        )
        self.assertEqual(
            demo.st.session_state["_pending_refine_widget_updates"]["refine_workspace_view"],
            "工作台",
        )
        self.assertEqual(
            demo.st.session_state["_pending_refine_widget_updates"]["edit_prompt"],
            "保持内容不变，增强清晰度",
        )

    def test_activate_refine_version_stages_history_version_via_pending_widget_updates(self):
        original_bytes = make_test_png_bytes((90, 20, 30))
        version_key = demo.ensure_refine_source_version(
            original_bytes,
            input_mime_type="image/png",
            source_label="上传图像",
        )

        activated = demo.activate_refine_version(version_key)

        self.assertTrue(activated)
        self.assertEqual(demo.st.session_state["refine_active_version_key"], version_key)
        self.assertEqual(demo.st.session_state["refine_staged_image_bytes"], original_bytes)
        self.assertEqual(
            demo.st.session_state["_pending_refine_widget_updates"]["refine_input_source"],
            "候选方案",
        )
        self.assertEqual(
            demo.st.session_state["_pending_refine_widget_updates"]["refine_workspace_view"],
            "工作台",
        )

    def test_build_refine_version_display_map_uses_human_readable_labels(self):
        demo.st.session_state["refine_version_history"] = [
            {
                "version_key": "v16",
                "parent_version_key": "",
                "label": "原图",
                "source_label": "上传图像",
                "variant_index": 0,
            },
            {
                "version_key": "v17",
                "parent_version_key": "v16",
                "label": "v17",
                "source_label": "上传图像",
                "variant_index": 1,
            },
            {
                "version_key": "v20",
                "parent_version_key": "v16",
                "label": "v20",
                "source_label": "上传图像",
                "variant_index": 2,
            },
        ]

        display_map = demo.build_refine_version_display_map()

        self.assertEqual(display_map["v16"], "原图")
        self.assertEqual(display_map["v17"], "第1版")
        self.assertEqual(display_map["v20"], "第2版")

    def test_append_refine_snapshot_stores_human_readable_label_for_new_versions(self):
        original_bytes = make_test_png_bytes((210, 30, 30))
        demo.ensure_refine_source_version(
            original_bytes,
            input_mime_type="image/png",
            source_label="上传图像",
        )

        created_keys = demo.append_refine_snapshot_to_version_history(
            {
                "created_at": "2026-03-15 19:18:02",
                "provider": "gemini",
                "image_model_name": "gemini-image",
                "resolution": "4K",
                "input_mime_type": "image/png",
                "original_image_bytes": original_bytes,
                "refined_images": [
                    {"index": 1, "bytes": make_test_png_bytes((30, 210, 30))},
                ],
            },
            edit_prompt="增强标题清晰度",
        )

        created_entry = demo.find_refine_version_entry(created_keys[0])
        self.assertIsNotNone(created_entry)
        self.assertEqual(created_entry.get("label"), "第1版")

    def test_activate_refine_version_uses_display_label_as_source_label(self):
        original_bytes = make_test_png_bytes((11, 22, 33))
        refined_bytes = make_test_png_bytes((44, 55, 66))
        demo.st.session_state["refine_version_history"] = [
            {
                "version_key": "v16",
                "parent_version_key": "",
                "label": "原图",
                "source_label": "上传图像",
                "input_mime_type": "image/png",
                "image_bytes": original_bytes,
                "variant_index": 0,
            },
            {
                "version_key": "v20",
                "parent_version_key": "v16",
                "label": "v20",
                "source_label": "上传图像",
                "input_mime_type": "image/png",
                "image_bytes": refined_bytes,
                "edit_prompt": "增强标题",
                "variant_index": 1,
            },
        ]

        activated = demo.activate_refine_version("v20")

        self.assertTrue(activated)
        self.assertEqual(demo.st.session_state["refine_staged_source_label"], "第1版")

    def test_append_refine_snapshot_prefers_selected_source_label(self):
        original_bytes = make_test_png_bytes((123, 45, 67))
        demo.st.session_state["refine_staged_source_label"] = "候选 01"
        demo.st.session_state["refine_selected_source_label"] = "上传图像"

        created_keys = demo.append_refine_snapshot_to_version_history(
            {
                "created_at": "2026-03-12 19:00:00",
                "provider": "gemini",
                "image_model_name": "gemini-image",
                "resolution": "2K",
                "input_mime_type": "image/png",
                "original_image_bytes": original_bytes,
                "refined_images": [
                    {"index": 1, "bytes": make_test_png_bytes((88, 99, 120))},
                ],
            },
            edit_prompt="只做清晰度增强",
        )

        self.assertEqual(len(created_keys), 1)
        created_entry = demo.find_refine_version_entry(created_keys[0])
        self.assertIsNotNone(created_entry)
        self.assertEqual(created_entry.get("source_label"), "上传图像")

    def test_persist_generation_results_can_preserve_candidate_workspace(self):
        demo.st.session_state["generation_candidate_decisions"] = {"1": "favorite"}
        demo.st.session_state["generation_final_candidate_id"] = "2"

        demo.persist_generation_job_results(
            {
                "results": [{"candidate_id": 1}, {"candidate_id": 2}],
                "task_name": "diagram",
                "dataset_name": "PaperBananaBench",
                "exp_mode": "demo_planner_critic",
                "summary": {},
                "failures": [],
            },
            source_label="历史回放",
            reset_candidate_workspace=False,
        )

        self.assertEqual(demo.st.session_state["generation_candidate_decisions"], {"1": "favorite"})
        self.assertEqual(demo.st.session_state["generation_final_candidate_id"], "2")

    def test_complex_ui_state_round_trip_handles_version_history_bytes(self):
        payload = demo._serialize_ui_state_value(
            "refine_version_history",
            [
                {
                    "version_key": "v01",
                    "image_bytes": make_test_png_bytes((1, 2, 3)),
                }
            ],
        )

        restored = demo._deserialize_ui_state_value("refine_version_history", payload)

        self.assertEqual(restored[0]["version_key"], "v01")
        self.assertEqual(restored[0]["image_bytes"], make_test_png_bytes((1, 2, 3)))

    def test_validate_refine_image_bytes_rejects_invalid_bytes(self):
        validated_bytes, validated_mime_type, validation_error = demo.validate_refine_image_bytes(
            b"not-an-image",
            input_mime_type="image/png",
            file_name="broken.png",
        )

        self.assertEqual(validated_bytes, b"")
        self.assertEqual(validated_mime_type, "image/png")
        self.assertIn("broken.png", validation_error or "")
        self.assertIn("不是可识别", validation_error or "")

    def test_sanitize_refine_version_history_removes_invalid_entries(self):
        valid_bytes = make_test_png_bytes((90, 80, 70))
        demo.st.session_state["refine_version_history"] = [
            {
                "version_key": "v01",
                "label": "原图",
                "image_bytes": valid_bytes,
                "input_mime_type": "image/png",
            },
            {
                "version_key": "v02",
                "label": "损坏版本",
                "image_bytes": b"broken-image",
                "input_mime_type": "image/png",
            },
        ]
        demo.st.session_state["refine_active_version_key"] = "v02"
        demo.st.session_state["refine_latest_version_keys"] = ["v01", "v02"]

        removed_labels = demo.sanitize_refine_version_history()

        self.assertEqual(removed_labels, ["损坏版本"])
        self.assertEqual(len(demo.st.session_state["refine_version_history"]), 1)
        self.assertEqual(demo.st.session_state["refine_active_version_key"], "v01")
        self.assertEqual(demo.st.session_state["refine_latest_version_keys"], ["v01"])

    def test_apply_pending_generation_widget_state_updates_flushes_queue(self):
        demo.st.session_state["_pending_generation_widget_updates"] = {
            "tab1_exp_mode": "demo_planner_critic",
            "tab1_image_resolution": "2K",
        }

        demo._apply_pending_generation_widget_state_updates()

        self.assertEqual(demo.st.session_state["tab1_exp_mode"], "demo_planner_critic")
        self.assertEqual(demo.st.session_state["tab1_image_resolution"], "2K")
        self.assertNotIn("_pending_generation_widget_updates", demo.st.session_state)

    def test_apply_pending_refine_widget_state_updates_flushes_queue(self):
        demo.st.session_state["_pending_refine_widget_updates"] = {
            "refine_input_source": "候选方案",
            "edit_prompt": "继续优化层次",
            "refine_workspace_view": "结果与状态",
        }

        demo._apply_pending_refine_widget_state_updates()

        self.assertEqual(demo.st.session_state["refine_input_source"], "候选方案")
        self.assertEqual(demo.st.session_state["edit_prompt"], "继续优化层次")
        self.assertEqual(demo.st.session_state["refine_workspace_view"], "结果与状态")
        self.assertNotIn("_pending_refine_widget_updates", demo.st.session_state)

    def test_refine_history_layout_helpers_match_compact_workspace_expectation(self):
        self.assertEqual(demo.get_refine_history_grid_columns(1), 1)
        self.assertEqual(demo.get_refine_history_grid_columns(7), demo.REFINE_HISTORY_PREVIEW_COLS)
        self.assertEqual(demo.get_refine_history_preview_width(1), demo.REFINE_HISTORY_PREVIEW_WIDTH)
        self.assertEqual(demo.get_refine_history_preview_width(3), "stretch")


if __name__ == "__main__":
    unittest.main()
