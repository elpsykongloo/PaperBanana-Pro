# 评估工具

用于在改检索、提示、模型或流程之前和之后，按同一口径做对比。所有命令都在仓库根目录运行。

## 准备

- 数据集：`data/PaperBananaBench/`（见主 README）。
- 真实调用需要环境变量 `GOOGLE_API_KEY`。离线汇总（`retrieval_eval report`）不需要。
- 输出写到 `results/eval/`（已被 git 忽略），图片缓存在 `results/eval/cache/`。

```powershell
$env:GOOGLE_API_KEY = "<你的 key>"
```

## 工具

| 命令 | 用途 | 量级（按标价估算） |
|---|---|---|
| `python -m scripts.eval.retrieval_eval report [--file ...]` | 离线汇总检索基准：各方案 top-3/top-10 平均有用度、≥2 分占比、配对差值置信区间，以及"已评分参考里最好的 10 条"这一上限 | 免费 |
| `python -m scripts.eval.retrieval_eval run --arm <名字> --model <模型> [--no-thumbnails]` | 用仓库 `RetrieverAgent` 跑一个新方案，只给新出现的参考补评分，另存新文件 | 40 条约 $1–3 |
| `python -m scripts.eval.pipeline_measure --n 3 --include-zh --leak-check` | 整链跑若干条：各阶段的模型、耗时（含最长）、token、费用；每一轮成图导出到 `<输出目录>/<字段名>/<id>.png` | 每条约 $0.4（3.8 Flash、2K、3 轮 Critic） |
| `python -m scripts.eval.pairwise_eval --x <目录> --y <目录> [--leak-check]` | 两组同名成图的成对盲评（gemini-3.1-pro，看原图，两种顺序都胜才算胜），可选检查非内容文字 | 每条约 $0.05–0.1 |
| `python -m scripts.eval.critic_probe --run <pipeline 输出目录> --models a,b` | 换文本模型前检查它当 Critic 时会不会总回"无需修改" | 很少 |

常见组合：

- **比较 Critic 轮次**：`pipeline_measure --n 20 --out results/eval/critic_rounds` 之后，`pairwise_eval --x .../target_diagram_critic_desc2 --y .../target_diagram_desc0`。
- **换文本模型**：先 `retrieval_eval run` 看检索，再 `critic_probe` 看 Critic，最后 `pipeline_measure` 看耗时和费用。

## 基准数据（`baselines/`）

| 文件 | 内容 |
|---|---|
| `retrieval_40.json` | 测试集 40 条（4 类各 10 条），仓库检索各方案的 10 条参考与 0–3 分有用度；含当前默认（`flash-3.8-thumbs`） |
| `retrieval_80.json` | 测试集 80 条（4 类各 20 条），六种检索方法；用于比较方法与估计参考池上限 |
| `image_prompts_16.json` | 测试集 16 条的 Planner 描述与修正版 Stylist 描述，以及生图提示改动（V0–V4）、生图模型（NB2 / NB2.1）的成对盲评记录 |

三份文件的查询互不重叠。`pipeline_measure` 默认抽样时会排除它们。

## 判分口径

- **参考有用度**：评审（gemini-3-flash-preview）同时看目标原图和候选参考，按"能否作为画这张图的版式模板"打 0–3 分：
  - 3 分：同类型、布局很像；
  - 2 分：同类型、布局不同；
  - 1 分：弱相关；
  - 0 分：无用。
- **成对盲评**：评审（gemini-3.1-pro-preview）看原图（只作参照）和两张候选图，按整体、忠实度、可读性、简洁、美观分别判 A / B / 平。两种顺序各评一次，两次都判被测方胜才算胜。
- **非内容文字**：gemini-3-flash-preview 列出图中不属于图内容的文字，例如颜色名、风格术语、布局指令。

注意：

- 改了评审提示或评审模型后，新结果不能与基准直接比较。
- 补评的参考与原有参考不在同一批评审里。
- 样本量小（16–80 条），结论用于定方向。
