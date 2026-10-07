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

"""
Streamlit Visualizer for Pipeline Evolution
Shows the progression of diagrams through Planner → Stylist → Critic stages
"""

import streamlit as st
import json
import base64
from io import BytesIO
from PIL import Image
import os
import sys

# Ensure local imports work
sys.path.append(os.getcwd())

from utils.pipeline_state import (
    critic_suggestions_key,
    build_render_stage_entries,
    detect_task_type_from_result,
    get_available_critic_rounds,
    stage_display_label,
)
from utils.demo_task_utils import collect_candidate_references, summarize_retrieval_meta
from utils.result_order import format_candidate_display_label
from utils.result_paths import resolve_gt_image_path
from visualize.viewer_helpers import (
    clear_viewer_bundle_cache,
    get_result_identifier,
    matches_result_search,
    render_bundle_manifest_sidebar,
    render_bundle_source_picker,
    render_viewer_empty_state,
)

st.set_page_config(layout="wide", page_title="PaperBanana-Pro 流程回放", page_icon="🍌")

def base64_to_image(b64_str):
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data))
    except Exception:
        return None

def detect_task_type(item):
    """Detect whether data is for diagram or plot task."""
    return detect_task_type_from_result(item)


def get_latest_review_notes(item):
    task_type = detect_task_type(item)
    rounds = get_available_critic_rounds(item, task_type)
    if rounds:
        latest_key = critic_suggestions_key(task_type, rounds[-1])
        latest_notes = str(item.get(latest_key, "") or "").strip()
        if latest_notes:
            return latest_notes
    for legacy_key in ("critique0", "suggestions_diagram", "suggestions_plot"):
        legacy_notes = str(item.get(legacy_key, "") or "").strip()
        if legacy_notes:
            return legacy_notes
    return ""


def display_stage_comparison(item, results_path):
    """Display 2x2 grid comparison: Ground Truth + three pipeline stages."""
    st.markdown("### 📊 流水线演化对比")
    
    task_type = detect_task_type(item)
    
    # Detect available stages dynamically
    available_stages = []
    
    # Human (Ground Truth) - always first
    available_stages.append({
        "title": "🎯 人工参考图",
        "desc_key": None,
        "img_key": "annotation_info",
        "color": "orange",
        "is_human": True
    })

    stage_colors = {
        "vanilla": "blue",
        "planner": "blue",
        "stylist": "violet",
        "critic": "green",
        "polish": "orange",
    }
    for stage_entry in build_render_stage_entries(
        item,
        task_type,
        item.get("exp_mode"),
    ):
        available_stages.append({
            "title": stage_display_label(
                stage_entry["stage_name"],
                stage_entry.get("round_idx"),
            ),
            "desc_key": stage_entry.get("text_key"),
            "img_key": stage_entry["image_key"],
            "suggestions_key": stage_entry.get("suggestions_key"),
            "color": stage_colors.get(stage_entry["stage_name"], "blue"),
            "is_human": False,
            "round_idx": stage_entry.get("round_idx"),
            "stage_name": stage_entry["stage_name"],
            "code_key": stage_entry.get("code_key"),
        })
            
    # Create dynamic grid based on number of stages
    num_stages = len(available_stages)
    cols_per_row = 2
    stages = available_stages
    
    # Display stages in a grid
    for row_start in range(0, num_stages, cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            stage_idx = row_start + col_idx
            if stage_idx >= num_stages:
                break
            
            stage = stages[stage_idx]
            with cols[col_idx]:
                st.markdown(f"**{stage['title']}**")
                
                # Display image
                if stage["is_human"]:
                    # Handle Human (Ground Truth) image
                    human_path = item.get("path_to_gt_image")
                    resolved_human_path = resolve_gt_image_path(
                        human_path,
                        task_type=task_type,
                        results_path=results_path,
                        work_dir=os.getcwd(),
                        dataset_name=item.get("dataset_name"),
                    )
                    if resolved_human_path:
                        try:
                            img = Image.open(resolved_human_path)
                            st.image(img, width="stretch")
                        except Exception as e:
                            st.error(f"加载人工参考图失败：{e}")
                    else:
                        st.info("暂无人工参考图")
                    
                    # Show caption instead of description
                    caption = item.get("brief_desc", "暂无图注")
                    with st.expander("查看图注", expanded=False):
                        st.write(caption)
                else:
                    # Handle pipeline stage images
                    img_b64 = item.get(stage["img_key"])
                    if img_b64:
                        img = base64_to_image(img_b64)
                        if img:
                            st.image(img, width="stretch")
                        else:
                            st.error("图像解码失败")
                    else:
                        st.info("暂无图像")
                    
                    # Display description in expander
                    desc = item.get(stage["desc_key"], "暂无描述")
                    with st.expander("查看描述", expanded=False):
                        st.write(desc)

                    if task_type == "plot" and stage.get("code_key") and item.get(stage["code_key"]):
                        with st.expander("查看 Matplotlib 代码", expanded=False):
                            st.code(item[stage["code_key"]], language="python")
                    
                    # Display critic suggestions if this is a critic stage
                    if "suggestions_key" in stage:
                        suggestions = item.get(stage["suggestions_key"], "")
                        if suggestions and suggestions.strip() != "No changes needed.":
                            with st.expander("💬 评审建议", expanded=False):
                                st.write(suggestions)

def display_references(item):
    """展示检索到的参考 id、图注与检索摘要（旧结果只有 id 时同样可用）。"""
    references = collect_candidate_references(item)
    meta_summary = summarize_retrieval_meta(item.get("retrieval_meta"))
    if not references and not meta_summary:
        return
    with st.expander(f"📚 使用的参考（{len(references)} 条）", expanded=False):
        if meta_summary:
            st.caption(meta_summary)
        for reference in references:
            caption = reference.get("caption") or ""
            if len(caption) > 160:
                caption = caption[:160] + "…"
            st.markdown(f"- **{reference['id']}** {caption}")


def display_critique(item):
    """Display the critique if available."""
    critique = get_latest_review_notes(item)
    if critique:
        st.markdown("### 💬 评审反馈")
        with st.expander("查看评审内容", expanded=False):
            st.write(critique)

def display_evaluation_results(item):
    """Display evaluation results if available."""
    dimensions = ["Faithfulness", "Conciseness", "Readability", "Aesthetics", "Overall"]
    
    has_eval = any(f"{dim.lower()}_outcome" in item for dim in dimensions)
    
    if has_eval:
        st.markdown("### 📈 评估结果")
        cols = st.columns(len(dimensions))
        
        for i, dim in enumerate(dimensions):
            outcome_key = f"{dim.lower()}_outcome"
            reasoning_key = f"{dim.lower()}_reasoning"
            outcome = item.get(outcome_key, "N/A")
            reasoning = item.get(reasoning_key, "N/A")
            
            with cols[i]:
                st.markdown(f"**{dim}**")
                if outcome == "Model":
                    st.success(outcome)
                elif outcome == "Human":
                    st.info(outcome)
                elif outcome == "Tie":
                    st.warning(outcome)
                else:
                    st.text(outcome)
                
                with st.expander("查看理由", expanded=False):
                    st.write(reasoning)

def main():
    bundle_source = render_bundle_source_picker(
        viewer_key="pipeline_evolution",
        sidebar_title="🍌 流程回放",
    )

    if st.sidebar.button("🔄 刷新数据"):
        clear_viewer_bundle_cache()
        st.rerun()

    if bundle_source is None:
        st.info("请先上传结果 Bundle；只有在需要读取本地同目录资源时，才使用高级路径模式。")
        st.stop()

    bundle = bundle_source.bundle
    data = bundle.get("results", [])
    manifest = bundle.get("manifest", {})

    with st.sidebar.expander("🧾 运行清单", expanded=False):
        render_bundle_manifest_sidebar(
            manifest,
            result_count=len(data),
            source_label=bundle_source.source_label,
        )
    
    # --- Search Functionality ---
    search_query = st.sidebar.text_input(
        "🔍 搜索候选编号 / ID",
        value="",
        help="优先按 candidate_id 搜索，也兼容旧结果里的 id、filename 和可视化意图。",
    )
    if search_query:
        data = [
            item
            for idx, item in enumerate(data)
            if matches_result_search(item, search_query, idx)
        ]
        st.sidebar.caption(f"找到 {len(data)} 条匹配结果")
    
    total_items = len(data)
    
    if total_items == 0:
        render_viewer_empty_state(search_query)
        return
    
    st.title("🍌 PaperBanana-Pro 流程回放")
    st.markdown("按候选回看规划、渲染、评审修正等中间阶段，不再直接暴露内部 artifact key。")
    if bundle_source.source_kind != "path":
        st.caption("当前使用上传模式。若某些人工参考图依赖本地同目录资源，可在左侧“高级：从本地路径读取”中切换到路径模式。")
    
    st.divider()
    
    # --- Global Statistics ---
    with st.expander("📊 全局统计", expanded=False):
        total = len(data)
        multi_stage_cases = sum(
            1
            for item in data
            if len(
                build_render_stage_entries(
                    item,
                    detect_task_type(item),
                    item.get("exp_mode"),
                )
            )
            >= 2
        )
        
        col1, col2, col3 = st.columns(3)
        col1.metric("样本总数", total)
        col2.metric("多阶段样本", multi_stage_cases)
        col3.metric("覆盖率", f"{multi_stage_cases/total*100:.1f}%")
    
    st.divider()
    
    # --- Pagination ---
    PAGE_SIZE = 10  # Changed from 5 to 10
    if "page" not in st.session_state:
        st.session_state.page = 0
    
    total_pages = max((total_items + PAGE_SIZE - 1) // PAGE_SIZE, 1)
    
    # Navigation buttons
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_left:
        if st.button("⬅️ 上一页", disabled=(st.session_state.page == 0)):
            st.session_state.page -= 1
            st.rerun()
    
    with col_center:
        page_input = st.number_input(
            "页码", 
            min_value=1, 
            max_value=total_pages, 
            value=st.session_state.page + 1,
            label_visibility="collapsed"
        )
        if page_input != st.session_state.page + 1:
            st.session_state.page = page_input - 1
            st.rerun()
        st.caption(f"第 {st.session_state.page + 1} / {total_pages} 页")
    
    with col_right:
        if st.button("下一页 ➡️", disabled=(st.session_state.page >= total_pages - 1)):
            st.session_state.page += 1
            st.rerun()
    
    start_idx = st.session_state.page * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, total_items)
    batch = data[start_idx:end_idx]
    
    st.markdown(f"**正在显示第 {start_idx + 1} - {end_idx} 条，共 {total_items} 条**")
    
    # --- Display Samples ---
    for i, item in enumerate(batch):
        idx = start_idx + i
        anno = item  # Flattened structure
        
        with st.container(border=True):
            # Header
            candidate_label = format_candidate_display_label(item, fallback_index=idx)
            identifier = get_result_identifier(item, idx)
            st.subheader(f"{candidate_label}: {item.get('visual_intent', 'N/A')}")
            st.caption(f"标识：`{identifier}`")
            
            # Method/Data section
            task_type = detect_task_type(item)
            label = "📚 原始数据" if task_type == "plot" else "📚 方法章节"
            
            with st.expander(label, expanded=False):
                if task_type == "plot":
                    st.code(json.dumps(item.get('content', {}), indent=2), language="json")
                else:
                    method_content = item.get('content', 'N/A')
                    st.markdown(method_content)
            
            # Pipeline comparison
            display_stage_comparison(
                item,
                bundle_source.source_name if bundle_source.source_kind == "path" else None,
            )
            
            # 本样本使用的参考
            display_references(item)

            # Critique
            display_critique(item)
            
            # Evaluation results
            display_evaluation_results(item)
            
            st.divider()

if __name__ == "__main__":
    main()

