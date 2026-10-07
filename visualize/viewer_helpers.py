"""共享的 viewer 输入与结果筛选辅助逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from utils.result_bundle import (
    ResultBundleLoadError,
    load_result_bundle,
    load_result_bundle_bytes,
)
from utils.result_order import get_candidate_id


VIEWER_UPLOAD_TYPES = ["json", "jsonl"]


@dataclass(frozen=True)
class ViewerBundleSource:
    bundle: dict[str, Any]
    source_label: str
    source_name: str
    source_kind: str


@st.cache_data(show_spinner=False)
def load_bundle_from_path_cached(path: str) -> dict[str, Any]:
    return load_result_bundle(path)


@st.cache_data(show_spinner=False)
def load_bundle_from_upload_cached(
    source_name: str,
    payload_bytes: bytes,
) -> dict[str, Any]:
    return load_result_bundle_bytes(payload_bytes, source_name=source_name)


def clear_viewer_bundle_cache() -> None:
    load_bundle_from_path_cached.clear()
    load_bundle_from_upload_cached.clear()


def get_result_identifier(item: dict[str, Any], fallback_index: int = 0) -> str:
    candidate_value = get_candidate_id(item, fallback_index)
    if candidate_value not in (None, ""):
        return str(candidate_value)
    legacy_id = str(item.get("id", "") or "").strip()
    if legacy_id:
        return legacy_id
    filename = str(item.get("filename", "") or "").strip()
    return filename or "Unknown"


def matches_result_search(
    item: dict[str, Any],
    query: str,
    fallback_index: int = 0,
) -> bool:
    lowered = str(query or "").strip().lower()
    if not lowered:
        return True
    tokens = {
        get_result_identifier(item, fallback_index),
        str(item.get("id", "") or ""),
        str(item.get("filename", "") or ""),
        str(item.get("visual_intent", "") or ""),
        str(item.get("brief_desc", "") or ""),
    }
    return any(lowered in token.lower() for token in tokens if token)


def render_bundle_load_error(
    *,
    source_label: str,
    error: Exception,
) -> None:
    hint = getattr(error, "hint", "") or ""
    st.error(f"无法打开结果文件：{source_label}")
    st.markdown(f"- 原因：{error}")
    if hint:
        st.markdown(f"- 建议：{hint}")


def render_bundle_manifest_sidebar(
    manifest: dict[str, Any],
    *,
    result_count: int,
    source_label: str,
) -> None:
    st.write(f"**来源：** {source_label}")
    manifest_fields = [
        ("生产方", "producer"),
        ("数据集", "dataset_name"),
        ("任务", "task_name"),
        ("切分", "split_name"),
        ("模式", "exp_mode"),
        ("Provider", "provider"),
        ("文本模型", "model_name"),
        ("图像模型", "image_model_name"),
        ("检索策略", "retrieval_setting"),
    ]
    for label, key in manifest_fields:
        value = manifest.get(key)
        if value:
            st.write(f"**{label}：** {value}")
    if manifest.get("retrieval_setting") in ("curated", "manual") and manifest.get("curated_profile"):
        st.write(f"**固定参考集：** {manifest['curated_profile']}")
    st.write(f"**结果数：** {manifest.get('result_count', result_count)}")


def render_viewer_empty_state(search_query: str = "") -> None:
    if search_query:
        st.warning(f"没有找到匹配 “{search_query}” 的样本。")
        return
    st.warning("当前结果里没有可展示的候选。请确认上传的是包含 `results` 的 bundle。")


def load_viewer_bundle_source(
    *,
    source_kind: str,
    source_name: str,
    path_value: str | None = None,
    payload_bytes: bytes | None = None,
) -> ViewerBundleSource | None:
    try:
        if source_kind == "upload":
            bundle = load_bundle_from_upload_cached(
                source_name,
                bytes(payload_bytes or b""),
            )
            return ViewerBundleSource(
                bundle=bundle,
                source_label=f"上传文件：{source_name}",
                source_name=source_name,
                source_kind=source_kind,
            )

        normalized_path = str(path_value or "").strip()
        if not normalized_path:
            return None
        bundle = load_bundle_from_path_cached(normalized_path)
        return ViewerBundleSource(
            bundle=bundle,
            source_label=f"本地路径：{Path(normalized_path).name}",
            source_name=normalized_path,
            source_kind=source_kind,
        )
    except (FileNotFoundError, ResultBundleLoadError, ValueError) as error:
        render_bundle_load_error(source_label=source_name, error=error)
        return None


def render_bundle_source_picker(
    *,
    viewer_key: str,
    sidebar_title: str,
) -> ViewerBundleSource | None:
    st.sidebar.title(sidebar_title)
    st.sidebar.caption("优先上传 `.bundle.json`；旧版结果 JSON / JSONL 也兼容。")

    uploaded_file = st.sidebar.file_uploader(
        "上传结果 Bundle",
        type=VIEWER_UPLOAD_TYPES,
        key=f"{viewer_key}_bundle_upload",
        help="推荐直接上传 `.bundle.json`。这样不需要手动记住结果目录结构。",
    )

    with st.sidebar.expander("高级：从本地路径读取", expanded=False):
        path_value = st.text_input(
            "本地结果路径",
            key=f"{viewer_key}_bundle_path",
            placeholder="例如：results/demo/run.bundle.json",
            help="仅在你需要读取本地同目录资源时再使用路径模式。",
        )

    if uploaded_file is not None:
        return load_viewer_bundle_source(
            source_kind="upload",
            source_name=uploaded_file.name,
            payload_bytes=uploaded_file.getvalue(),
        )

    normalized_path = str(path_value or "").strip()
    if normalized_path:
        return load_viewer_bundle_source(
            source_kind="path",
            source_name=normalized_path,
            path_value=normalized_path,
        )

    return None
