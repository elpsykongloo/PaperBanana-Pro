"""把成对盲评结果抽样做成一个离线网页，供人工复核评审是否可信。

网页里每对图随机左右放置、不显示模型判决；人工选完后点"导出"，把得到的 JSON 存成文件，再用 agree 子命令算一致率。
  python -m scripts.eval.review_page build --pairwise results/eval/pairwise_a.json,results/eval/pairwise_b.json --n 30
  python -m scripts.eval.review_page agree --page results/eval/review_xxx.json --answers 人工结果.json
"""

from __future__ import annotations

import argparse
import base64
import html
import random
from pathlib import Path

from scripts.eval.common import OUTPUT_DIR, jpeg_bytes, load_bench, load_json, resolve_image, save_json, timestamp


def b64(path: Path, side: int) -> str:
    return base64.b64encode(jpeg_bytes(path, side)).decode()


def build(args: argparse.Namespace) -> None:
    test, _ = load_bench()
    pool = []
    for f in args.pairwise.split(","):
        data = load_json(Path(f))
        for row in data["rows"]:
            x_img, y_img = Path(data["x"]) / f"{row['id']}.png", Path(data["y"]) / f"{row['id']}.png"
            if x_img.read_bytes() == y_img.read_bytes():  # 例如选择器挑中的正是最后一轮，两边是同一张图
                continue
            pool.append({"file": f, "x": data["x"], "y": data["y"], "id": row["id"], "model": row["verdict"]["overall"]})
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    picked = pool[: args.n]
    stamp = timestamp()
    page_meta = []
    cards = []
    for k, p in enumerate(picked):
        left_is_x = rng.random() < 0.5
        x_img, y_img = Path(p["x"]) / f"{p['id']}.png", Path(p["y"]) / f"{p['id']}.png"
        left, right = (x_img, y_img) if left_is_x else (y_img, x_img)
        item = test[p["id"]]
        gt = resolve_image(item["path_to_gt_image"])
        page_meta.append({**p, "k": k, "left_is_x": left_is_x})
        cards.append(f"""
<section class="card" data-k="{k}">
  <h2>{k + 1}. {html.escape(str(item['visual_intent'])[:300])}</h2>
  <details><summary>原图（仅作参照）与方法摘要</summary><img class="gt" src="data:image/jpeg;base64,{b64(gt, 800)}">
  <p class="method">{html.escape(str(item['content'])[:2500])}</p></details>
  <div class="pair"><figure><figcaption>左</figcaption><img src="data:image/jpeg;base64,{b64(left, 1200)}"></figure>
  <figure><figcaption>右</figcaption><img src="data:image/jpeg;base64,{b64(right, 1200)}"></figure></div>
  <div class="choice">整体更好：
    <label><input type="radio" name="c{k}" value="left">左</label>
    <label><input type="radio" name="c{k}" value="tie">差不多</label>
    <label><input type="radio" name="c{k}" value="right">右</label>
    <input class="note" name="n{k}" placeholder="可选：理由（例如右图漏了某模块）">
  </div>
</section>""")
    page = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>成图人工复核</title><style>
:root{{--bg:#fafafa;--fg:#1b1b1b;--card:#fff;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--card:#1f1f1f;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px;max-width:1500px;margin:auto}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:16px 0}}
h2{{font-size:15px;margin:0 0 8px}} .pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
@media (max-width:800px){{.pair{{grid-template-columns:1fr}}}}
figure{{margin:0}} img{{max-width:100%;border:1px solid var(--line);background:#fff}} .gt{{max-width:600px}}
.method{{white-space:pre-wrap;font-size:13px;max-height:200px;overflow:auto}} .choice{{margin-top:8px;display:flex;gap:12px;flex-wrap:wrap;align-items:center}}
.note{{flex:1;min-width:200px;padding:4px}} button{{font-size:15px;padding:8px 16px}} textarea{{width:100%;height:120px}}
</style></head><body>
<h1>成图人工复核（{len(picked)} 对）</h1>
<p>每对图是同一篇论文、同一图注的两个生成结果，左右随机。请按"忠实于方法、可读、专业"判断哪张整体更好。全部选完后点底部"导出"，把文本框内容存成 JSON 文件发回。</p>
{''.join(cards)}
<button id="export">导出</button><textarea id="out" readonly></textarea>
<script>
const KEY='review-{stamp}';
function save(){{const d={{}};document.querySelectorAll('input').forEach(i=>{{if((i.type==='radio'&&i.checked)||i.type!=='radio')d[i.name]=i.value}});try{{localStorage.setItem(KEY,JSON.stringify(d))}}catch(e){{}}}}
try{{const d=JSON.parse(localStorage.getItem(KEY)||'{{}}');for(const [k,v] of Object.entries(d)){{document.querySelectorAll(`[name="${{k}}"]`).forEach(i=>{{if(i.type==='radio')i.checked=i.value===v;else i.value=v}})}}}}catch(e){{}}
document.addEventListener('change',save);document.addEventListener('input',save);
document.getElementById('export').onclick=()=>{{const r=[];document.querySelectorAll('.card').forEach(c=>{{const k=c.dataset.k;const s=c.querySelector(`input[name="c${{k}}"]:checked`);r.push({{k:+k,choice:s?s.value:null,note:c.querySelector(`[name="n${{k}}"]`).value}})}});document.getElementById('out').value=JSON.stringify({{page:'{stamp}',answers:r}},null,1)}};
</script></body></html>"""
    out_html = Path(args.out) if args.out else OUTPUT_DIR / f"review_{stamp}.html"
    out_html.write_text(page, encoding="utf-8")
    save_json(out_html.with_suffix(".json"), {"page": stamp, "pairs": page_meta})
    print(f"网页：{out_html}（{out_html.stat().st_size / 1e6:.1f} MB）\n对照表：{out_html.with_suffix('.json')}")


def agree(args: argparse.Namespace) -> None:
    meta = {p["k"]: p for p in load_json(Path(args.page))["pairs"]}
    answers = load_json(Path(args.answers))["answers"]
    both = exact = strict_disagree = n = 0
    for a in answers:
        if not a.get("choice"):
            continue
        p = meta[a["k"]]
        human = "tie" if a["choice"] == "tie" else ("win" if (a["choice"] == "left") == p["left_is_x"] else "loss")
        n += 1
        exact += human == p["model"]
        strict_disagree += {human, p["model"]} == {"win", "loss"}
        both += human != "tie" and p["model"] != "tie" and human == p["model"]
    print(f"已作答 {n} 对：完全一致 {exact}/{n}；方向相反（一方说胜、另一方说负）{strict_disagree}/{n}；双方都有倾向且一致 {both}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--pairwise", required=True, help="逗号分隔的 pairwise_eval 结果文件")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    p = sub.add_parser("agree")
    p.add_argument("--page", required=True)
    p.add_argument("--answers", required=True)
    args = parser.parse_args(argv)
    {"build": build, "agree": agree}[args.cmd](args)


if __name__ == "__main__":
    main()
