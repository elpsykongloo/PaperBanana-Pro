"""Shared demo task metadata and result helpers."""

from copy import deepcopy

from utils.pipeline_state import (
    code_key_for_desc,
    critic_desc_key,
    critic_suggestions_key,
    find_final_stage_keys as find_pipeline_final_stage_keys,
    get_available_critic_rounds as get_pipeline_available_critic_rounds,
    image_key_for_desc,
    normalize_task_name,
    planner_desc_key,
    stylist_desc_key,
)


TASK_UI_CONFIGS = {
    "diagram": {
        "display_name": "diagram",
        "display_name_cn": "学术图解",
        "intro": "从论文方法章节和图注生成多个图解候选方案",
        "content_selector_label": "加载示例（方法章节）",
        "visual_selector_label": "加载示例（图注）",
        "content_label": "方法章节内容（建议使用 Markdown 格式）",
        "content_placeholder": "在此粘贴方法章节内容...",
        "content_help": "论文中描述方法的章节内容。建议使用 Markdown 格式。",
        "visual_label": "图注（建议使用 Markdown 格式）",
        "visual_placeholder": "输入图注...",
        "visual_help": "要生成的图解说明。建议使用 Markdown 格式。",
        "content_input_name": "方法内容",
        "visual_input_name": "图注",
        "uses_image_model": True,
        "uses_render_controls": True,
        "example_name": "PaperBanana 框架",
        "example_content": r"""## Methodology: The PaperBanana Framework

In this section, we present the architecture of PaperBanana, a reference-driven agentic framework for automated academic illustration. As illustrated in Figure \ref{fig:methodology_diagram}, PaperBanana orchestrates a collaborative team of five specialized agents—Retriever, Planner, Stylist, Visualizer, and Critic—to transform raw scientific content into publication-quality diagrams and plots. (See Appendix \ref{app_sec:agent_prompts} for prompts)

### Retriever Agent

Given the source context $S$ and the communicative intent $C$, the Retriever Agent identifies $N$ most relevant examples $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ from the fixed reference set $\mathcal{R}$ to guide the downstream agents. As defined in Section \ref{sec:task_formulation}, each example $E_i \in \mathcal{R}$ is a triplet $(S_i, C_i, I_i)$.
To leverage the reasoning capabilities of VLMs, we adopt a generative retrieval approach where the VLM performs selection over candidate metadata:
$$
\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)
$$

### Planner Agent

The Planner Agent serves as the cognitive core of the system. It takes the source context $S$, communicative intent $C$, and retrieved examples $\mathcal{E}$ as inputs. By performing in-context learning from the demonstrations in $\mathcal{E}$, the Planner translates the unstructured or structured data in $S$ into a comprehensive and detailed textual description $P$ of the target illustration.

### Stylist Agent

To ensure the output adheres to the aesthetic standards of modern academic manuscripts, the Stylist Agent acts as a design consultant. It refines the initial description into a stylistically optimized version while preserving semantic accuracy.

### Visualizer Agent

The Visualizer Agent transforms the description into visual output. In each iteration $t$, given a description $P_t$, the Visualizer generates:
$$
I_t = \text{Image-Gen}(P_t)
$$

### Critic Agent

The Critic Agent examines the generated image $I_t$ and provides a refined description $P_{t+1}$ that addresses factual misalignments and visual glitches. The Visualizer-Critic loop iterates for $T=3$ rounds.""",
        "example_visual_intent": "Figure 1: Overview of our PaperBanana framework. Given the source context and communicative intent, we first apply a Linear Planning Phase to retrieve relevant reference examples and synthesize a stylistically optimized description. We then use an Iterative Refinement Loop (consisting of Visualizer and Critic agents) to transform the description into visual output and conduct multi-round refinements to produce the final academic illustration.",
        "final_caption": "最终图解",
        "planner_stage_description": "基于方法内容和图注生成的初始图解规划",
        "stylist_stage_description": "经过风格优化的图解描述",
    },
    "plot": {
        "display_name": "plot",
        "display_name_cn": "统计图",
        "intro": "从原始数据和可视化意图生成多个统计图候选方案",
        "content_selector_label": "加载示例（原始数据）",
        "visual_selector_label": "加载示例（可视化意图）",
        "content_label": "原始数据（推荐 JSON / CSV / Markdown 表格）",
        "content_placeholder": "在此粘贴原始数据...",
        "content_help": "可直接粘贴 JSON、CSV 或 Markdown 表格，尽量保留完整字段和值。",
        "visual_label": "可视化意图（建议写清图类型、坐标、分组和风格）",
        "visual_placeholder": "输入希望生成的统计图说明...",
        "visual_help": "说明图表类型、字段映射、分组方式、强调重点和视觉风格。",
        "content_input_name": "原始数据",
        "visual_input_name": "可视化意图",
        "uses_image_model": False,
        "uses_render_controls": False,
        "example_name": "模型缩放曲线",
        "example_content": """[
  {"family": "PaperBanana", "tokens_billion": 10, "accuracy": 58.4},
  {"family": "PaperBanana", "tokens_billion": 25, "accuracy": 62.7},
  {"family": "PaperBanana", "tokens_billion": 50, "accuracy": 66.1},
  {"family": "PaperBanana", "tokens_billion": 100, "accuracy": 68.8},
  {"family": "Baseline", "tokens_billion": 10, "accuracy": 55.2},
  {"family": "Baseline", "tokens_billion": 25, "accuracy": 59.1},
  {"family": "Baseline", "tokens_billion": 50, "accuracy": 61.8},
  {"family": "Baseline", "tokens_billion": 100, "accuracy": 64.0}
]""",
        "example_visual_intent": "Create a publication-quality line chart comparing model accuracy against training tokens for two model families. Use training tokens (billions) on the x-axis and accuracy (%) on the y-axis. Show one line per family with circular markers, annotate the final point of each line, place the legend in the lower right, and use a clean NeurIPS-style layout with subtle grid lines.",
        "final_caption": "最终统计图",
        "planner_stage_description": "基于原始数据和可视化意图生成的初始绘图规划",
        "stylist_stage_description": "经过风格优化的绘图描述",
    },
}


def get_task_ui_config(task_name: str) -> dict:
    return TASK_UI_CONFIGS[normalize_task_name(task_name)]


def create_sample_inputs(
    content,
    visual_intent,
    task_name: str = "diagram",
    aspect_ratio: str = "16:9",
    num_copies: int = 10,
    max_critic_rounds: int = 3,
    image_resolution: str = "2K",
):
    """Create demo inputs for parallel candidate generation."""
    normalized_task = normalize_task_name(task_name)
    base_input = {
        "filename": "demo_input",
        "task_name": normalized_task,
        "caption": visual_intent,
        "content": content,
        "visual_intent": visual_intent,
        "additional_info": {
            "rounded_ratio": aspect_ratio,
            "image_resolution": image_resolution,
        },
        "max_critic_rounds": max_critic_rounds,
    }

    inputs = []
    for idx in range(max(1, int(num_copies))):
        input_copy = deepcopy(base_input)
        input_copy["filename"] = f"demo_input_candidate_{idx}"
        input_copy["candidate_id"] = idx
        inputs.append(input_copy)
    return inputs


def get_available_critic_rounds(result, task_name: str = "diagram"):
    """Return critic rounds that produced rendered outputs."""
    return get_pipeline_available_critic_rounds(result or {}, task_name)


def find_final_stage_keys(result, task_name: str = "diagram", exp_mode: str = "demo_planner_critic"):
    """Resolve the final rendered image/description keys for a candidate result."""
    return find_pipeline_final_stage_keys(result or {}, task_name, exp_mode)


def build_evolution_stages(result, exp_mode: str, task_name: str = "diagram"):
    """Build a stage timeline for demo result display."""
    normalized_task = normalize_task_name(task_name)
    task_config = get_task_ui_config(normalized_task)
    stages = []

    planner_key = planner_desc_key(normalized_task)
    planner_img_key = image_key_for_desc(planner_key)
    if result.get(planner_img_key):
        stage = {
            "name": "📋 规划器",
            "image_key": planner_img_key,
            "desc_key": planner_key,
            "description": task_config["planner_stage_description"],
        }
        planner_code_key = code_key_for_desc(planner_key)
        if normalized_task == "plot" and result.get(planner_code_key):
            stage["code_key"] = planner_code_key
        stages.append(stage)

    if exp_mode == "demo_full":
        stylist_key = stylist_desc_key(normalized_task)
        stylist_img_key = image_key_for_desc(stylist_key)
        if result.get(stylist_img_key):
            stage = {
                "name": "✨ 风格化器",
                "image_key": stylist_img_key,
                "desc_key": stylist_key,
                "description": task_config["stylist_stage_description"],
            }
            stylist_code_key = code_key_for_desc(stylist_key)
            if normalized_task == "plot" and result.get(stylist_code_key):
                stage["code_key"] = stylist_code_key
            stages.append(stage)

    for round_idx in get_available_critic_rounds(result, task_name=normalized_task):
        critic_key = critic_desc_key(normalized_task, round_idx)
        critic_img_key = image_key_for_desc(critic_key)
        if not result.get(critic_img_key):
            continue
        stage = {
            "name": f"🔍 评审第 {round_idx + 1} 轮",
            "image_key": critic_img_key,
            "desc_key": critic_key,
            "suggestions_key": critic_suggestions_key(normalized_task, round_idx),
            "description": f"根据评审反馈进行优化（第 {round_idx + 1} 次迭代）",
        }
        critic_code_key = code_key_for_desc(critic_key)
        if normalized_task == "plot" and result.get(critic_code_key):
            stage["code_key"] = critic_code_key
        stages.append(stage)

    return stages
