import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


if "streamlit" not in sys.modules:
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.set_page_config = lambda *args, **kwargs: None
    fake_streamlit.session_state = {}
    sys.modules["streamlit"] = fake_streamlit

demo = importlib.import_module("demo")


_TEXT_INPUT_SENTINEL = object()


class _FakeInteractiveStreamlit:
    def __init__(self):
        self.session_state = {}
        self.selectbox_calls = []
        self.text_input_calls = []
        self.html_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def selectbox(self, label, options, **kwargs):
        self.selectbox_calls.append(
            {
                "label": label,
                "options": list(options),
                "kwargs": dict(kwargs),
            }
        )
        index = kwargs.get("index", 0)
        key = kwargs.get("key")
        if key not in self.session_state:
            self.session_state[key] = options[index]
        return self.session_state[key]

    def text_input(self, label, *args, value=_TEXT_INPUT_SENTINEL, key=None, help=None, **kwargs):
        explicit_value = bool(args) or value is not _TEXT_INPUT_SENTINEL
        resolved_value = args[0] if args else ("" if value is _TEXT_INPUT_SENTINEL else value)
        self.text_input_calls.append(
            {
                "label": label,
                "value": resolved_value,
                "value_provided": explicit_value,
                "key": key,
                "help": help,
                "kwargs": dict(kwargs),
            }
        )
        if key not in self.session_state:
            self.session_state[key] = resolved_value
        return self.session_state[key]

    def columns(self, spec, **kwargs):
        return [self for _ in spec]

    def caption(self, *args, **kwargs):
        return None

    def button(self, *args, **kwargs):
        return False

    def html(self, body, **kwargs):
        self.html_calls.append({"body": body, "kwargs": dict(kwargs)})
        return None

    def checkbox(self, label, value=False, key=None, **kwargs):
        if key not in self.session_state:
            self.session_state[key] = value
        return self.session_state[key]

    def slider(self, label, min_value=None, max_value=None, value=None, key=None, **kwargs):
        if key not in self.session_state:
            self.session_state[key] = value
        return self.session_state[key]

    def warning(self, *args, **kwargs):
        return None


class DemoModelInputTest(unittest.TestCase):
    def setUp(self):
        self.original_streamlit = demo.st
        self.fake_streamlit = _FakeInteractiveStreamlit()
        demo.st = self.fake_streamlit

    def tearDown(self):
        demo.st = self.original_streamlit

    def _render_openai_image_options(self, *, aspect_ratio, image_resolution, model="gpt-image-1.5"):
        return demo.render_image_generation_option_controls(
            provider_type="openai",
            image_model_name=model,
            aspect_ratio=aspect_ratio,
            image_resolution=image_resolution,
            widget_prefix="t",
        )

    def test_openai_size_follows_aspect_ratio_changes_but_keeps_manual_choice(self):
        first = self._render_openai_image_options(aspect_ratio="16:9", image_resolution="2K")
        self.assertEqual(first["size"], "1536x1024")

        # 宽高比变化后，尺寸要跟着变（之前会一直停留在首次渲染的值）
        changed = self._render_openai_image_options(aspect_ratio="9:16", image_resolution="2K")
        self.assertEqual(changed["size"], "1024x1536")

        # 输入不变时保留用户手选的尺寸
        self.fake_streamlit.session_state["t_image_size"] = "1024x1024"
        kept = self._render_openai_image_options(aspect_ratio="9:16", image_resolution="2K")
        self.assertEqual(kept["size"], "1024x1024")

        # 尺寸下拉框不再同时传 index，避免与 Session State 冲突
        size_calls = [call for call in self.fake_streamlit.selectbox_calls if call["label"] == "OpenAI 输出尺寸"]
        self.assertTrue(size_calls)
        self.assertNotIn("index", size_calls[-1]["kwargs"])

    def test_gpt_image_2_custom_size_follows_aspect_ratio(self):
        first = self._render_openai_image_options(aspect_ratio="16:9", image_resolution="1K", model="gpt-image-2")
        self.assertEqual(first["size"], "1536x864")
        self.assertEqual(self.fake_streamlit.session_state["t_image_size"], "自定义像素尺寸")

        changed = self._render_openai_image_options(aspect_ratio="1:1", image_resolution="2K", model="gpt-image-2")
        self.assertEqual(changed["size"], "2048x2048")
        self.assertEqual(self.fake_streamlit.session_state["t_image_size"], "2048x2048")

    def test_gpt_image_2_custom_size_survives_hidden_text_input(self):
        state = self.fake_streamlit.session_state
        first = self._render_openai_image_options(aspect_ratio="16:9", image_resolution="1K", model="gpt-image-2")
        self.assertEqual(first["size"], "1536x864")

        # 用户改选预设尺寸：自定义输入框这一轮不渲染，真实 Streamlit 会删除它的状态
        state["t_image_size"] = "1024x1024"
        self._render_openai_image_options(aspect_ratio="16:9", image_resolution="1K", model="gpt-image-2")
        del state["t_image_custom_size"]

        # 再切回自定义：应恢复默认的 1536x864，而不是空值退回 auto
        state["t_image_size"] = "自定义像素尺寸"
        back = self._render_openai_image_options(aspect_ratio="16:9", image_resolution="1K", model="gpt-image-2")
        self.assertEqual(back["size"], "1536x864")

    def test_custom_model_value_initializes_selector_and_text_buffer(self):
        self.fake_streamlit.session_state["model_value"] = "vendor/custom-text-model"

        resolved = demo.render_preset_or_custom_model_input(
            "文本模型",
            ["preset-a", "preset-b"],
            value_key="model_value",
            selector_key="model_selector",
            custom_value_key="model_custom",
            default_value="preset-a",
            select_help="help",
        )

        self.assertEqual(resolved, "vendor/custom-text-model")
        self.assertEqual(
            self.fake_streamlit.session_state["model_selector"],
            demo.CUSTOM_MODEL_OPTION,
        )
        self.assertEqual(
            self.fake_streamlit.session_state["model_custom"],
            "vendor/custom-text-model",
        )

    def test_custom_selector_uses_manual_input_value(self):
        self.fake_streamlit.session_state.update(
            {
                "model_value": "preset-a",
                "model_selector": demo.CUSTOM_MODEL_OPTION,
                "model_custom": "vendor/manual-image-model",
            }
        )

        resolved = demo.render_preset_or_custom_model_input(
            "图像模型",
            ["preset-a", "preset-b"],
            value_key="model_value",
            selector_key="model_selector",
            custom_value_key="model_custom",
            default_value="preset-a",
            select_help="help",
        )

        self.assertEqual(resolved, "vendor/manual-image-model")
        self.assertEqual(
            self.fake_streamlit.session_state["model_value"],
            "vendor/manual-image-model",
        )

    def test_model_selector_widget_does_not_pass_default_index_when_key_is_managed(self):
        self.fake_streamlit.session_state["model_value"] = "preset-b"

        resolved = demo.render_preset_or_custom_model_input(
            "文本模型",
            ["preset-a", "preset-b"],
            value_key="model_value",
            selector_key="model_selector",
            custom_value_key="model_custom",
            default_value="preset-a",
            select_help="help",
        )

        self.assertEqual(resolved, "preset-b")
        self.assertEqual(len(self.fake_streamlit.selectbox_calls), 1)
        self.assertNotIn("index", self.fake_streamlit.selectbox_calls[0]["kwargs"])

    def test_initialize_curated_profile_state_separates_widget_key_from_canonical_key(self):
        self.fake_streamlit.session_state["curated_profile"] = " paper profile "

        normalized = demo.initialize_curated_profile_state(
            profile_key="curated_profile",
            input_key="curated_profile_input",
        )

        self.assertEqual(normalized, "paper-profile")
        self.assertEqual(
            self.fake_streamlit.session_state["curated_profile"],
            "paper-profile",
        )
        self.assertEqual(
            self.fake_streamlit.session_state["curated_profile_input"],
            "paper-profile",
        )

    def test_resolve_curated_profile_input_updates_only_canonical_key(self):
        self.fake_streamlit.session_state["curated_profile"] = "default"
        self.fake_streamlit.session_state["curated_profile_input"] = " custom profile "

        normalized = demo.resolve_curated_profile_input(
            self.fake_streamlit.session_state["curated_profile_input"],
            profile_key="curated_profile",
        )

        self.assertEqual(normalized, "custom-profile")
        self.assertEqual(
            self.fake_streamlit.session_state["curated_profile"],
            "custom-profile",
        )
        self.assertEqual(
            self.fake_streamlit.session_state["curated_profile_input"],
            " custom profile ",
        )

    def test_hydrate_api_key_session_state_restores_local_default_when_blank(self):
        self.fake_streamlit.session_state["tab1_api_key"] = ""

        restored = demo.hydrate_api_key_session_state(
            session_key="tab1_api_key",
            provider_defaults={"api_key_default": "local-google-key"},
        )

        self.assertEqual(restored, "local-google-key")
        self.assertEqual(
            self.fake_streamlit.session_state["tab1_api_key"],
            "local-google-key",
        )

    def test_prepare_api_key_widget_state_initializes_widget_key_from_canonical_value(self):
        self.fake_streamlit.session_state["tab1_api_key"] = "saved-google-key"

        restored = demo.prepare_api_key_widget_state(
            session_key="tab1_api_key",
            clear_request_key="tab1_api_key_clear_requested",
            provider_defaults={"api_key_default": "saved-google-key"},
        )

        self.assertEqual(restored, "saved-google-key")
        self.assertEqual(
            self.fake_streamlit.session_state[demo.get_api_key_widget_key("tab1_api_key")],
            "saved-google-key",
        )

    def test_prepare_api_key_widget_state_does_not_overwrite_existing_widget_buffer(self):
        self.fake_streamlit.session_state["tab1_api_key"] = ""
        self.fake_streamlit.session_state[demo.get_api_key_widget_key("tab1_api_key")] = "typed-not-yet-applied"

        restored = demo.prepare_api_key_widget_state(
            session_key="tab1_api_key",
            clear_request_key="tab1_api_key_clear_requested",
            provider_defaults={"api_key_default": ""},
        )

        self.assertEqual(restored, "typed-not-yet-applied")
        self.assertEqual(
            self.fake_streamlit.session_state[demo.get_api_key_widget_key("tab1_api_key")],
            "typed-not-yet-applied",
        )
        self.assertEqual(self.fake_streamlit.session_state["tab1_api_key"], "")

    def test_prepare_api_key_widget_state_honors_pending_clear_request(self):
        self.fake_streamlit.session_state["tab1_api_key"] = "persisted-key"
        self.fake_streamlit.session_state[demo.get_api_key_widget_key("tab1_api_key")] = "persisted-key"
        self.fake_streamlit.session_state["tab1_api_key_clear_requested"] = True

        restored = demo.prepare_api_key_widget_state(
            session_key="tab1_api_key",
            clear_request_key="tab1_api_key_clear_requested",
            provider_defaults={"api_key_default": ""},
        )

        self.assertEqual(restored, "")
        self.assertEqual(self.fake_streamlit.session_state["tab1_api_key"], "")
        self.assertEqual(
            self.fake_streamlit.session_state[demo.get_api_key_widget_key("tab1_api_key")],
            "",
        )
        self.assertNotIn("tab1_api_key_clear_requested", self.fake_streamlit.session_state)

    def test_render_provider_api_key_controls_uses_separate_widget_key(self):
        self.fake_streamlit.session_state["tab1_api_key"] = "saved-google-key"
        original_persist = demo.persist_provider_api_key_input
        captured_calls = []
        demo.persist_provider_api_key_input = lambda provider, api_key: captured_calls.append((provider, api_key))

        try:
            restored = demo.render_provider_api_key_controls(
                provider="gemini",
                provider_defaults={
                    "api_key_label": "Google API Key",
                    "api_key_help": "Google AI Studio API 密钥",
                    "api_key_default": "saved-google-key",
                },
                session_key="tab1_api_key",
                clear_request_key="tab1_api_key_clear_requested",
                clear_button_key="tab1_clear_provider_api_key",
            )
        finally:
            demo.persist_provider_api_key_input = original_persist

        self.assertEqual(restored, "saved-google-key")
        self.assertEqual(
            self.fake_streamlit.session_state[demo.get_api_key_widget_key("tab1_api_key")],
            "saved-google-key",
        )
        self.assertEqual(len(self.fake_streamlit.text_input_calls), 1)
        self.assertFalse(self.fake_streamlit.text_input_calls[0]["value_provided"])
        self.assertEqual(captured_calls, [("gemini", "saved-google-key")])

    def test_build_api_key_storage_notice_reflects_local_secret_state(self):
        self.assertEqual(
            demo.build_api_key_storage_notice({"api_key_default": "saved-key"}),
            "已在本机保存当前 Provider 的密钥，刷新页面后仍会保留。",
        )
        self.assertEqual(
            demo.build_api_key_storage_notice({"api_key_default": ""}),
            "密钥只保存在当前电脑；输入后会自动写入本地 txt。",
        )
        self.assertEqual(
            demo.build_api_key_storage_notice({"api_key_default": "saved-key"}, persist_secret=False),
            "当前输入仅在本次会话生效，不会写入本地 txt。",
        )
        self.assertEqual(
            demo.build_api_key_storage_notice(
                {"api_key_default": ""},
                persist_secret=True,
                allow_local_persist=False,
            ),
            "当前是未保存的自定义连接草稿；API Key 会先保留在本次会话，保存连接后才会写入本地 txt。",
        )

    def test_render_provider_api_key_controls_skips_persist_when_session_only(self):
        self.fake_streamlit.session_state["tab1_api_key"] = "session-only-key"
        original_persist = demo.persist_provider_api_key_input
        captured_calls = []
        demo.persist_provider_api_key_input = lambda provider, api_key: captured_calls.append((provider, api_key))

        try:
            restored = demo.render_provider_api_key_controls(
                provider="gemini",
                provider_defaults={
                    "api_key_label": "Google API Key",
                    "api_key_help": "Google AI Studio API 密钥",
                    "api_key_default": "session-only-key",
                },
                session_key="tab1_api_key",
                clear_request_key="tab1_api_key_clear_requested",
                clear_button_key="tab1_clear_provider_api_key",
                persist_secret=False,
            )
        finally:
            demo.persist_provider_api_key_input = original_persist

        self.assertEqual(restored, "session-only-key")
        self.assertEqual(captured_calls, [])

    def test_render_provider_api_key_controls_skips_persist_for_unsaved_custom_draft(self):
        self.fake_streamlit.session_state["tab1_api_key"] = "draft-key"
        original_persist = demo.persist_provider_api_key_input
        captured_calls = []
        demo.persist_provider_api_key_input = lambda provider, api_key: captured_calls.append((provider, api_key))

        try:
            restored = demo.render_provider_api_key_controls(
                provider="custom-openai",
                provider_defaults={
                    "api_key_label": "兼容 API Key",
                    "api_key_help": "OpenAI 兼容接口密钥",
                    "api_key_default": "draft-key",
                },
                session_key="tab1_api_key",
                clear_request_key="tab1_api_key_clear_requested",
                clear_button_key="tab1_clear_provider_api_key",
                persist_secret=True,
                allow_local_persist=False,
            )
        finally:
            demo.persist_provider_api_key_input = original_persist

        self.assertEqual(restored, "draft-key")
        self.assertEqual(captured_calls, [])

    def test_sync_connection_runtime_input_state_resets_inputs_when_selection_changes(self):
        self.fake_streamlit.session_state.update(
            {
                "tab1_api_key": "old-key",
                demo.get_api_key_widget_key("tab1_api_key"): "old-key",
                "tab1_model_name": "old-text",
                "tab1_image_model_name": "old-image",
                "tab1_model_name_selector": "旧选择器",
                "tab1_model_name_custom": "旧自定义文本",
                "tab1_image_model_name_selector": "旧图像选择器",
                "tab1_image_model_name_custom": "旧自定义图像",
                "tab1_runtime_input_connection_id": "gemini",
            }
        )

        demo.sync_connection_runtime_input_state(
            prefix="tab1",
            selected_connection_id="custom-openai",
            provider_defaults={
                "api_key_default": "new-key",
                "model_name": "new-text",
                "image_model_name": "new-image",
            },
        )

        self.assertEqual(self.fake_streamlit.session_state["tab1_api_key"], "new-key")
        self.assertEqual(
            self.fake_streamlit.session_state[demo.get_api_key_widget_key("tab1_api_key")],
            "new-key",
        )
        self.assertEqual(self.fake_streamlit.session_state["tab1_model_name"], "new-text")
        self.assertEqual(self.fake_streamlit.session_state["tab1_image_model_name"], "new-image")
        self.assertNotIn("tab1_model_name_selector", self.fake_streamlit.session_state)
        self.assertNotIn("tab1_model_name_custom", self.fake_streamlit.session_state)
        self.assertNotIn("tab1_image_model_name_selector", self.fake_streamlit.session_state)
        self.assertNotIn("tab1_image_model_name_custom", self.fake_streamlit.session_state)
        self.assertEqual(
            self.fake_streamlit.session_state["tab1_runtime_input_connection_id"],
            "custom-openai",
        )

    def test_sync_connection_runtime_input_state_keeps_current_values_when_selection_unchanged(self):
        self.fake_streamlit.session_state.update(
            {
                "tab1_api_key": "keep-key",
                demo.get_api_key_widget_key("tab1_api_key"): "keep-key",
                "tab1_model_name": "keep-text",
                "tab1_runtime_input_connection_id": "gemini",
            }
        )

        demo.sync_connection_runtime_input_state(
            prefix="tab1",
            selected_connection_id="gemini",
            provider_defaults={
                "api_key_default": "new-key",
                "model_name": "new-text",
                "image_model_name": "new-image",
            },
        )

        self.assertEqual(self.fake_streamlit.session_state["tab1_api_key"], "keep-key")
        self.assertEqual(self.fake_streamlit.session_state["tab1_model_name"], "keep-text")

    def test_save_connection_draft_deletes_builtin_secret_when_persist_disabled(self):
        state_keys = demo._build_connection_state_keys("tab1")
        self.fake_streamlit.session_state[state_keys["persist_secret"]] = False
        deleted_providers = []

        with patch.object(demo, "delete_provider_api_key", side_effect=lambda provider, base_dir=None: deleted_providers.append(provider)):
            with patch.object(demo, "write_provider_api_key") as mocked_write:
                ok, message, defaults = demo.save_connection_draft(
                    prefix="tab1",
                    selected_connection_id="gemini",
                    api_key="temporary-key",
                    model_name="gemini-3.1-pro-preview",
                    image_model_name="gemini-3-pro-image-preview",
                )

        self.assertTrue(ok)
        self.assertEqual(message, "已保存内置连接配置。")
        self.assertIn("connection_id", defaults)
        self.assertEqual(deleted_providers, ["gemini"])
        mocked_write.assert_not_called()

    def test_save_connection_draft_for_builtin_connection_queues_widget_updates(self):
        state_keys = demo._build_connection_state_keys("tab1")
        self.fake_streamlit.session_state.update(
            {
                state_keys["persist_secret"]: True,
                state_keys["base_url"]: "https://draft.invalid/v1",
                state_keys["probe_results"]: {"text": {"status": "failed"}},
            }
        )

        with patch.object(demo, "write_provider_api_key"):
            ok, message, defaults = demo.save_connection_draft(
                prefix="tab1",
                selected_connection_id="openrouter",
                api_key="temporary-key",
                model_name="google/gemini-3.1-flash-lite-preview",
                image_model_name="google/gemini-3.1-flash-image-preview",
            )

        self.assertTrue(ok)
        self.assertEqual(message, "已保存内置连接配置。")
        self.assertEqual(
            self.fake_streamlit.session_state[state_keys["base_url"]],
            "https://draft.invalid/v1",
        )
        self.assertEqual(
            self.fake_streamlit.session_state["_pending_generation_widget_updates"][state_keys["base_url"]],
            defaults["base_url"],
        )

    def test_save_connection_draft_for_custom_connection_queues_provider_selection(self):
        state_keys = demo._build_connection_state_keys("tab1")
        self.fake_streamlit.session_state.update(
            {
                state_keys["connection_id"]: "saved-openai",
                state_keys["display_name"]: "已保存连接",
                state_keys["base_url"]: "https://example.com/v1",
                state_keys["api_key_env_var"]: "SAVED_API_KEY",
                state_keys["extra_headers_json"]: "",
                state_keys["supports_text"]: True,
                state_keys["supports_image"]: True,
                state_keys["enabled"]: True,
                state_keys["persist_secret"]: True,
                state_keys["model_discovery_mode"]: "hybrid",
                state_keys["model_allowlist"]: "saved-text\nsaved-image",
            }
        )

        with patch.object(
            demo,
            "upsert_custom_connection",
            return_value=types.SimpleNamespace(connection_id="saved-openai", display_name="已保存连接"),
        ):
            with patch.object(
                demo,
                "_apply_connection_defaults_to_session",
                return_value={"connection_id": "saved-openai"},
            ):
                ok, message, defaults = demo.save_connection_draft(
                    prefix="tab1",
                    selected_connection_id=demo.CUSTOM_CONNECTION_CREATE_OPTION,
                    api_key="draft-key",
                    model_name="saved-text",
                    image_model_name="saved-image",
                )

        self.assertTrue(ok)
        self.assertEqual(message, "已保存自定义连接：已保存连接")
        self.assertEqual(defaults["connection_id"], "saved-openai")
        self.assertEqual(
            self.fake_streamlit.session_state["_pending_generation_widget_updates"][state_keys["select_key"]],
            "saved-openai",
        )

    def test_build_connection_draft_supports_unsaved_custom_connection(self):
        state_keys = demo._build_connection_state_keys("tab1")
        self.fake_streamlit.session_state.update(
            {
                state_keys["connection_id"]: "draft-openai",
                state_keys["display_name"]: "草稿连接",
                state_keys["base_url"]: "https://example.com/v1",
                state_keys["api_key_env_var"]: "DRAFT_API_KEY",
                state_keys["extra_headers_json"]: '{\"X-Test\":\"demo\"}',
                state_keys["supports_text"]: True,
                state_keys["supports_image"]: False,
                state_keys["enabled"]: True,
                state_keys["model_discovery_mode"]: "hybrid",
                state_keys["model_allowlist"]: "draft-text\ndraft-image",
                state_keys["probe_results"]: {},
            }
        )

        connection = demo.build_connection_draft(
            prefix="tab1",
            selected_connection_id=demo.CUSTOM_CONNECTION_CREATE_OPTION,
            api_key="draft-key",
            model_name="draft-text",
            image_model_name="draft-image",
        )

        self.assertEqual(connection.connection_id, "draft-openai")
        self.assertEqual(connection.provider_type, demo.CUSTOM_PROVIDER_TYPE)
        self.assertEqual(connection.base_url, "https://example.com/v1")
        self.assertEqual(connection.extra_headers, {"X-Test": "demo"})
        self.assertEqual(connection.api_key, "draft-key")

    def test_parse_extra_headers_json_safe_returns_error_instead_of_raising(self):
        headers, error_message = demo.parse_extra_headers_json_safe("{not-json}")

        self.assertEqual(headers, {})
        self.assertIn("额外请求头格式错误", error_message)

    def test_inject_refine_tab_sidebar_autocollapse_hook_registers_dom_script(self):
        demo.inject_refine_tab_sidebar_autocollapse_hook()

        self.assertEqual(len(self.fake_streamlit.html_calls), 1)
        html_call = self.fake_streamlit.html_calls[0]
        self.assertIn("section.stSidebar", html_call["body"])
        self.assertIn("button[role=\"tab\"]", html_call["body"])
        self.assertIn("精修图像", html_call["body"])
        self.assertIn("stExpandSidebarButton", html_call["body"])
        self.assertIn("autoCollapsed", html_call["body"])
        self.assertTrue(html_call["kwargs"]["unsafe_allow_javascript"])

    def test_format_repo_relative_path_prefers_repo_relative_display(self):
        absolute_path = Path("D:/PaperBanana/data/PaperBananaBench/diagram/manual_profiles/default.json")

        formatted = demo.format_repo_relative_path(absolute_path, base_dir=Path("D:/PaperBanana"))

        self.assertEqual(
            formatted,
            "data/PaperBananaBench/diagram/manual_profiles/default.json",
        )

    def test_format_repo_relative_path_falls_back_to_absolute_display(self):
        outside_path = Path("C:/Users/86166/AppData/Roaming/uv/tools/paperbanana-pro/Lib/site-packages/results/demo/sample.bundle.json")

        formatted = demo.format_repo_relative_path(outside_path, base_dir=Path("D:/PaperBanana"))

        self.assertEqual(
            formatted,
            "C:/Users/86166/AppData/Roaming/uv/tools/paperbanana-pro/Lib/site-packages/results/demo/sample.bundle.json",
        )

    def test_ensure_session_choice_state_repairs_invalid_value(self):
        self.fake_streamlit.session_state["tab1_retrieval_setting"] = "mystery"

        resolved = demo.ensure_session_choice_state(
            "tab1_retrieval_setting",
            ["auto", "auto-full", "none"],
            "auto",
        )

        self.assertEqual(resolved, "auto")
        self.assertEqual(self.fake_streamlit.session_state["tab1_retrieval_setting"], "auto")

    def test_ensure_session_int_state_clamps_invalid_input(self):
        self.fake_streamlit.session_state["tab1_max_critic_rounds"] = "99"

        resolved = demo.ensure_session_int_state(
            "tab1_max_critic_rounds",
            1,
            min_value=0,
            max_value=5,
        )

        self.assertEqual(resolved, 5)
        self.assertEqual(self.fake_streamlit.session_state["tab1_max_critic_rounds"], 5)

    def test_resolve_demo_base_dir_prefers_direct_url_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            install_root = temp_root / "site-packages"
            install_root.mkdir()
            workspace_root = temp_root / "workspace"
            workspace_root.mkdir()
            (workspace_root / "demo.py").write_text("", encoding="utf-8")
            (workspace_root / "agents").mkdir()
            (workspace_root / "utils").mkdir()
            (workspace_root / "data").mkdir()

            dist_info = install_root / "paperbanana_pro-0.1.0.dist-info"
            dist_info.mkdir()
            (dist_info / "direct_url.json").write_text(
                json.dumps({"url": workspace_root.resolve().as_uri()}, ensure_ascii=False),
                encoding="utf-8",
            )

            resolved = demo.resolve_demo_base_dir(
                install_root,
                cwd=temp_root / "outside-cwd",
            )

            self.assertEqual(resolved, workspace_root.resolve())
