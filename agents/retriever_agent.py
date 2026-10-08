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
Retriever Agent - 检索相关参考示例。
"""

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
Retriever Agent - 检索相关参考示例。
"""


import asyncio
import copy
import hashlib
import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from utils.dataset_paths import (
    get_reference_file_path,
    resolve_data_asset_path,
)
from utils.retrieval_index import ReferencePoolIndex, cjk_ratio, stringify_payload
from utils.retrieval_profiles import (
    find_curated_profile_path,
    iter_curated_profile_candidate_paths,
    load_curated_reference_profile,
)
from utils.retrieval_settings import normalize_retrieval_setting
from utils import generation_utils, image_utils
from .base_agent import BaseAgent

from utils.log_config import get_logger

logger = get_logger("RetrieverAgent")

TOP_K = 10
# 查询中文占比超过该比例，或没有任何英文有效词时，先把查询改写成英文关键词再做词法预筛
CJK_REWRITE_RATIO = 0.3
# 每个 RetrieverAgent 实例最多缓存的检索结果数（按输入去重）
SHARED_RETRIEVAL_CACHE_LIMIT = 128
# 查询没有可用英文词时，参考池不超过该规模才把全部 caption 交给 LLM
ALL_CAPTIONS_FALLBACK_LIMIT = 400
# diagram 的 auto 模式：预筛候选附缩略图，让模型按图类型和布局挑选。
# 小样本实验（对照原图的 0–3 分有用度盲评，top-10 平均分）：gemini-3-flash 80 条 1.60 → 1.66；
# gemini-3.1-flash-lite 40 条 1.48 → 1.62。两组配对差的 95% 置信区间都不含 0。
# 同 40 条附缩略图：gemini-3.8-flash 1.68，gemini-3.5-flash-lite 1.61。
VISUAL_RERANK_THUMBNAIL_MAX_SIDE = 512
VISUAL_RERANK_TARGET_CHAR_LIMIT = 6000
VISUAL_RERANK_CAPTION_CHAR_LIMIT = 300


@lru_cache(maxsize=8)
def _load_indexed_pool(
    ref_file: str,
    mtime_ns: int,
    file_size: int,
    caption_weight: float,
    overlap_weight: float,
) -> tuple[tuple[dict, ...], ReferencePoolIndex]:
    """按文件路径、修改时间和大小缓存参考池及其 BM25 索引。"""
    with open(ref_file, "r", encoding="utf-8") as f:
        raw_pool = json.load(f)
    pool = tuple(
        item
        for item in raw_pool
        if isinstance(item, dict) and str(item.get("id", "") or "").strip()
    )
    return pool, ReferencePoolIndex(pool, caption_weight=caption_weight, overlap_weight=overlap_weight)


class RetrieverAgent(BaseAgent):
    """Retriever Agent to retrieve relevant reference examples"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.exp_config.model_name
        self._shared_retrievals: dict[str, Any] = {}

        if self.exp_config.task_name == "plot":
            self.system_prompt = PLOT_RETRIEVER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "plot",
                "lite_prefilter_limit": 48,
                "full_prefilter_limit": 20,
                # plot 的类别就是图表类型，caption 共有词数（不加 IDF）对图表类型词更敏感
                "index_weights": {"caption_weight": 1.0, "overlap_weight": 2.0},
                # 未在 plot 上做过对比实验，保持只发 caption
                "visual_rerank": False,
                "target_labels": ["Visual Intent", "Raw Data"],
                "candidate_labels": ["Plot ID", "Visual Intent", "Raw Data"],
                "candidate_type": "Plot",
                "output_key": "top10_references",
                "instruction_suffix": 'select the Top 10 most relevant plots according to the instructions provided. Your output must be a strictly valid JSON object with exactly one key, "top10_references", whose value is a list of the exact ids of the top 10 selected plots. Copy each id exactly as written after "Plot ID:"; never use a candidate\'s position.',
            }
        else:
            self.system_prompt = DIAGRAM_RETRIEVER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "diagram",
                "lite_prefilter_limit": 40,
                "full_prefilter_limit": 18,
                "index_weights": {"caption_weight": 0.3, "overlap_weight": 0.0},
                "visual_rerank": True,
                "target_labels": ["Caption", "Methodology section"],
                "candidate_labels": ["Diagram ID", "Caption", "Methodology section"],
                "candidate_type": "Diagram",
                "output_key": "top10_references",
                "instruction_suffix": 'select the Top 10 most relevant diagrams according to the instructions provided. Your output must be a strictly valid JSON object with exactly one key, "top10_references", whose value is a list of the exact ids of the top 10 selected diagrams. Copy each id exactly as written after "Diagram ID:"; never use a candidate\'s position.',
            }

    async def process(self, data: Dict[str, Any], retrieval_setting: str = "auto") -> Dict[str, Any]:
        cfg = self.task_config
        candidate_id = data.get("candidate_id", "N/A")
        requested_setting = normalize_retrieval_setting(retrieval_setting)
        retrieval_setting = requested_setting
        logger.debug(f"🔍 开始处理, setting={retrieval_setting}, task={cfg['task_name']}, provider={self.exp_config.provider}")

        ref_file = self._reference_file(cfg)
        if retrieval_setting in ["auto", "auto-full", "random"] and not ref_file.exists():
            logger.warning(f"⚠️  参考文件未找到: {ref_file}，回退到 retrieval_setting='none'")
            retrieval_setting = "none"

        if retrieval_setting == "curated":
            profile_path = find_curated_profile_path(
                self.exp_config.dataset_name,
                cfg["task_name"],
                profile_name=self.exp_config.curated_profile,
                work_dir=self.exp_config.work_dir,
            )
            if profile_path is None:
                candidate_paths = iter_curated_profile_candidate_paths(
                    self.exp_config.dataset_name,
                    cfg["task_name"],
                    profile_name=self.exp_config.curated_profile,
                    work_dir=self.exp_config.work_dir,
                )
                logger.warning(
                    "⚠️  未找到 curated profile（profile=%s, looked_at=%s），回退到 retrieval_setting='none'",
                    self.exp_config.curated_profile,
                    [str(path) for path in candidate_paths],
                )
                retrieval_setting = "none"

        meta: dict[str, Any] = {"requested_setting": requested_setting, "setting": retrieval_setting}

        if retrieval_setting == "none":
            data["top10_references"] = []
            data["retrieved_examples"] = []
            meta["method"] = "none"
            logger.debug("⏭️  跳过检索 (setting=none)")

        elif retrieval_setting == "curated":
            profile = self._load_curated_references(cfg)
            ids = profile.selected_ids
            examples = profile.examples
            data["top10_references"] = ids
            data["retrieved_examples"] = examples
            data["curated_profile"] = profile.profile_name
            data["curated_profile_source"] = profile.source_path.name
            meta["method"] = "curated"
            meta["missing_ids"] = list(profile.missing_ids or [])[:10]
            logger.info(
                "✅ curated 检索完成, profile=%s, source=%s, %s 个参考",
                profile.profile_name,
                profile.source_path.name,
                len(ids),
            )
            if profile.missing_ids:
                logger.warning(
                    "⚠️  curated profile 中有 %s 个 id 在 ref.json 中不存在: %s",
                    len(profile.missing_ids),
                    profile.missing_ids,
                )

        elif retrieval_setting == "random":
            data["top10_references"] = self._load_random_references(cfg)
            data["retrieved_examples"] = []
            meta["method"] = "random"
            logger.info(f"✅ 随机检索完成, {len(data['top10_references'])} 个参考")

        elif retrieval_setting in ("auto", "auto-full"):
            lite = retrieval_setting == "auto"
            retrieved_ids, retrieval_meta, shared = await self._shared_auto_retrieval(
                data,
                cfg,
                candidate_id=candidate_id,
                lite=lite,
            )
            pool_by_id = {str(item["id"]).strip(): item for item in self._load_candidate_pool(cfg)}
            data["top10_references"] = list(retrieved_ids)
            # 参考池条目在多个候选间共享，交给下游前复制一份，避免相互影响
            data["retrieved_examples"] = [
                copy.deepcopy(pool_by_id[ref_id]) for ref_id in retrieved_ids if ref_id in pool_by_id
            ]
            meta.update(retrieval_meta)
            meta["shared"] = shared
            logger.info(
                "✅ 自动检索完成 (%s), %s 个参考%s",
                "lite" if lite else "full",
                len(data["top10_references"]),
                "（复用本任务已有检索结果）" if shared else "",
            )
            logger.debug("auto reference ids=%s", data["top10_references"])
        else:
            raise ValueError(f"Unknown retrieval_setting: {retrieval_setting}")

        data["retrieval_meta"] = meta
        return data

    def _reference_file(self, cfg: dict) -> Path:
        return get_reference_file_path(
            self.exp_config.dataset_name,
            cfg["task_name"],
            work_dir=self.exp_config.work_dir,
        )

    def _load_indexed_pool(self, cfg: dict) -> tuple[tuple[dict, ...], ReferencePoolIndex]:
        ref_file = self._reference_file(cfg)
        stat = ref_file.stat()
        weights = cfg.get("index_weights") or {}
        return _load_indexed_pool(
            str(ref_file.resolve()),
            stat.st_mtime_ns,
            stat.st_size,
            float(weights.get("caption_weight", 0.5)),
            float(weights.get("overlap_weight", 0.0)),
        )

    def _load_curated_references(self, cfg: dict):
        return load_curated_reference_profile(
            self.exp_config.dataset_name,
            cfg["task_name"],
            profile_name=self.exp_config.curated_profile,
            work_dir=self.exp_config.work_dir,
            limit=TOP_K,
        )

    def _load_random_references(self, cfg: dict) -> list:
        id_list = [str(item["id"]).strip() for item in self._load_candidate_pool(cfg)]
        sample_size = min(TOP_K, len(id_list))
        return random.sample(id_list, sample_size) if sample_size > 0 else []

    @staticmethod
    def _normalize_retrieved_ids(values: Any) -> list[str]:
        if values is None:
            return []

        if isinstance(values, dict):
            for key in ["id", "reference_id", "diagram_id", "plot_id"]:
                if key in values:
                    return RetrieverAgent._normalize_retrieved_ids(values.get(key))
            return []

        if isinstance(values, (list, tuple, set)):
            normalized_ids: list[str] = []
            seen: set[str] = set()
            for value in values:
                for ref_id in RetrieverAgent._normalize_retrieved_ids(value):
                    if ref_id not in seen:
                        normalized_ids.append(ref_id)
                        seen.add(ref_id)
            return normalized_ids

        text = str(values or "").strip()
        return [text] if text else []

    @staticmethod
    def _json_payload_candidates(raw_response: str) -> list[str]:
        text = str(raw_response or "").strip()
        candidates: list[str] = []
        seen: set[str] = set()

        def add_candidate(candidate: str) -> None:
            candidate = str(candidate or "").strip()
            if candidate and candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

        add_candidate(text)
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
            add_candidate(match.group(1))

        for opener, closer in [("{", "}"), ("[", "]")]:
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end > start:
                add_candidate(text[start : end + 1])

        return candidates

    def _extract_retrieval_ids_from_payload(self, payload: Any, task_name: str) -> list[str]:
        if isinstance(payload, list):
            return self._normalize_retrieved_ids(payload)

        if not isinstance(payload, dict):
            return self._normalize_retrieved_ids(payload)

        if task_name == "plot":
            task_keys = ["top10_plots", "top_plots", "plots", "plot_ids"]
        elif task_name == "diagram":
            task_keys = ["top10_diagrams", "top_diagrams", "diagrams", "diagram_ids"]
        else:
            raise ValueError(f"Unknown task_name: {task_name}")

        common_keys = [
            "top10_references",
            "top_10_references",
            "selected_ids",
            "reference_ids",
            "ids",
            "top10",
            "top_10",
            "references",
            "results",
            "items",
        ]
        for key in [*task_keys, *common_keys]:
            if key not in payload:
                continue
            ids = self._extract_retrieval_ids_from_payload(payload.get(key), task_name)
            if ids:
                return ids

        return self._normalize_retrieved_ids(payload)

    @staticmethod
    def _extract_reference_ids_from_text(raw_response: str) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for ref_id in re.findall(r"\bref_[A-Za-z0-9_.:-]+\b", str(raw_response or "")):
            if ref_id not in seen:
                ids.append(ref_id)
                seen.add(ref_id)
        return ids

    def _load_candidate_pool(self, cfg: dict) -> list[dict]:
        pool, _ = self._load_indexed_pool(cfg)
        return list(pool)

    def _rank_candidate_pool(
        self,
        data: Dict[str, Any],
        cfg: dict,
        *,
        lite: bool,
        query_text: str | None = None,
    ) -> tuple[list[dict], dict[str, Any]]:
        """BM25 预筛：对整个参考池打分，返回 (短名单, 预筛元数据)。"""
        pool, index = self._load_indexed_pool(cfg)
        shortlist_limit = max(TOP_K, int(cfg["lite_prefilter_limit"] if lite else cfg["full_prefilter_limit"]))
        meta: dict[str, Any] = {"pool_size": len(pool)}
        if len(pool) <= TOP_K:
            meta["shortlist_size"] = len(pool)
            return list(pool), meta

        visual_intent = str(data.get("visual_intent", "") or "")
        content = stringify_payload(data.get("content", ""))
        if query_text:
            query_caption = f"{query_text}\n{visual_intent}"
            query_content = f"{query_text}\n{content}"
        else:
            query_caption, query_content = visual_intent, content
        order, token_count = index.rank(query_caption, query_content)
        meta["query_token_count"] = token_count
        shortlisted = [pool[idx] for idx in order[:shortlist_limit]]
        meta["shortlist_size"] = len(shortlisted)
        logger.debug(
            "📚 预筛完成: task=%s lite=%s total=%s shortlist=%s query_tokens=%s",
            cfg["task_name"],
            lite,
            len(pool),
            len(shortlisted),
            token_count,
        )
        return shortlisted, meta

    def _prefilter_candidate_pool(
        self,
        data: Dict[str, Any],
        cfg: dict,
        *,
        lite: bool,
        query_text: str | None = None,
    ) -> list[dict]:
        shortlist, _ = self._rank_candidate_pool(data, cfg, lite=lite, query_text=query_text)
        return shortlist

    def _needs_query_rewrite(self, data: Dict[str, Any], cfg: dict) -> bool:
        """目标输入以中文为主或没有英文有效词时，词法预筛无法工作，需要先改写成英文。"""
        _, index = self._load_indexed_pool(cfg)
        visual_intent = str(data.get("visual_intent", "") or "")
        content = stringify_payload(data.get("content", ""))
        caption_tokens, body_tokens = index.query_tokens(visual_intent, content)
        if not caption_tokens and not body_tokens:
            return True
        return cjk_ratio(f"{visual_intent}\n{content[:6000]}") > CJK_REWRITE_RATIO

    async def _rewrite_query_to_english(self, data: Dict[str, Any], cfg: dict, *, candidate_id: Any) -> str:
        prompt = QUERY_REWRITE_USER_PROMPT.format(
            target_type=cfg["candidate_type"].lower(),
            caption=str(data.get("visual_intent", "") or "")[:2000],
            content=stringify_payload(data.get("content", ""))[:6000],
        )
        try:
            response_list = await self.call_text_api(
                [{"type": "text", "text": prompt}],
                model_name=self.model_name,
                system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
                max_output_tokens=8192,
                max_attempts=2,
                retry_delay=5,
                error_context=f"retriever-query-rewrite[candidate={candidate_id}]",
            )
        except Exception as exc:
            logger.warning("⚠️  检索查询改写失败，将不做词法预筛: %s", exc)
            return ""
        text = str(response_list[0]).strip() if response_list else ""
        return "" if text == "Error" else text

    def _shared_cache_key(self, data: Dict[str, Any], cfg: dict, *, lite: bool) -> str:
        payload = json.dumps(
            [
                cfg["task_name"],
                str(self.exp_config.dataset_name),
                "lite" if lite else "full",
                str(self.model_name),
                stringify_payload(data.get("content", "")),
                str(data.get("visual_intent", "") or ""),
            ],
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _trim_shared_cache(self) -> None:
        done_keys = [key for key, value in self._shared_retrievals.items() if isinstance(value, tuple)]
        for key in done_keys[: max(0, len(done_keys) - SHARED_RETRIEVAL_CACHE_LIMIT)]:
            self._shared_retrievals.pop(key, None)

    async def _shared_auto_retrieval(
        self,
        data: Dict[str, Any],
        cfg: dict,
        *,
        candidate_id: Any,
        lite: bool,
    ) -> tuple[list[str], dict[str, Any], bool]:
        """同一输入在同一个 Agent 内只检索一次：并发的候选共用一个检索任务。"""
        key = self._shared_cache_key(data, cfg, lite=lite)
        loop = asyncio.get_running_loop()
        entry = self._shared_retrievals.get(key)
        if isinstance(entry, tuple):
            return list(entry[0]), dict(entry[1]), True
        if isinstance(entry, asyncio.Future) and entry.get_loop() is loop:
            ids, meta = await asyncio.shield(entry)
            return list(ids), dict(meta), True

        task = loop.create_task(self._retrieve_and_parse(data, cfg, candidate_id=candidate_id, lite=lite))
        self._shared_retrievals[key] = task

        def _finalize(done: asyncio.Task) -> None:
            if done.cancelled() or done.exception() is not None:
                if self._shared_retrievals.get(key) is done:
                    self._shared_retrievals.pop(key, None)
                return
            ids, meta = done.result()
            self._shared_retrievals[key] = (tuple(ids), dict(meta))
            self._trim_shared_cache()

        task.add_done_callback(_finalize)
        ids, meta = await asyncio.shield(task)
        return list(ids), dict(meta), False

    async def _retrieve_and_parse(
        self,
        data: Dict[str, Any],
        cfg: dict,
        candidate_id: Any = "N/A",
        lite: bool = True,
    ) -> tuple[list[str], dict[str, Any]]:
        """
        BM25 预筛 + LLM 选择。

        Args:
            lite: True = 候选只附 caption；False = 候选附完整正文（候选数更少）。目标输入始终完整发送。

        Returns:
            (参考 id 列表, 检索元数据)
        """
        meta: dict[str, Any] = {"method": "bm25+llm", "mode": "lite" if lite else "full"}
        query_text = None
        if self._needs_query_rewrite(data, cfg):
            query_text = await self._rewrite_query_to_english(data, cfg, candidate_id=candidate_id)
            meta["query_rewritten"] = bool(query_text)
            if query_text:
                logger.info("🌐 检索查询以中文为主，已改写为英文关键词用于预筛")

        candidate_pool, prefilter_meta = self._rank_candidate_pool(data, cfg, lite=lite, query_text=query_text)
        meta.update(prefilter_meta)
        if meta.get("query_token_count", 1) == 0:
            full_pool, _ = self._load_indexed_pool(cfg)
            if len(full_pool) <= ALL_CAPTIONS_FALLBACK_LIMIT:
                # 没有可用的英文查询词：不做预筛，改为把整个参考池的 caption 交给 LLM 选择
                candidate_pool = list(full_pool)
                lite = True
                meta.update({"fallback": "no_query_tokens_all_captions", "mode": "lite", "shortlist_size": len(candidate_pool)})
                logger.warning("⚠️  查询缺少可用英文词，跳过预筛，改为发送全部 %s 条参考的 caption", len(candidate_pool))
            else:
                meta["fallback"] = "no_query_tokens_file_order"
                logger.warning("⚠️  查询缺少可用英文词且参考池较大，预筛结果按文件顺序截取")

        raw_response = ""
        if lite and cfg.get("visual_rerank") and not meta.get("fallback"):
            raw_response = await self._select_with_thumbnails(data, cfg, candidate_pool, meta, candidate_id=candidate_id)
        if not raw_response:
            raw_response = await self._select_with_captions(data, cfg, candidate_pool, lite=lite, candidate_id=candidate_id)

        parsed_ids = self._parse_retrieval_result(raw_response, cfg["task_name"])
        shortlist_ids = [str(item.get("id", "") or "").strip() for item in candidate_pool]
        shortlist_id_set = set(shortlist_ids)
        parsed_ids, repaired = self._repair_numeric_ids(parsed_ids, shortlist_ids)
        if repaired:
            meta["repaired_ids"] = repaired
            logger.info("ℹ️  检索模型返回了 %s 个只含数字的 id，已按短名单补全", repaired)
        retrieved_ids = [ref_id for ref_id in parsed_ids if ref_id in shortlist_id_set][:TOP_K]
        missing_ids = [ref_id for ref_id in parsed_ids if ref_id not in shortlist_id_set]
        meta["selected"] = len(retrieved_ids)
        if missing_ids:
            meta["invalid_ids"] = missing_ids[:TOP_K]
            logger.warning(
                "⚠️  检索模型返回了 %s 个不在候选池中的 id，已忽略: %s",
                len(missing_ids),
                missing_ids[:5],
            )

        # 不足 TOP_K 时按预筛顺序补齐；一个有效 id 都没有时整组使用预筛结果
        target_count = min(TOP_K, len(shortlist_ids))
        topped_up = 0
        for ref_id in shortlist_ids:
            if len(retrieved_ids) >= target_count:
                break
            if ref_id not in retrieved_ids:
                retrieved_ids.append(ref_id)
                topped_up += 1
        if topped_up:
            meta["topped_up"] = topped_up
            if meta["selected"] == 0:
                meta.setdefault("fallback", "shortlist_order")
                logger.warning("⚠️  检索模型未返回可匹配的参考 id，使用预筛候选兜底: %s", retrieved_ids)
                logger.debug("检索模型原始响应片段: %s", raw_response[:500])
            else:
                logger.info("ℹ️  检索模型只返回 %s 个有效 id，已按预筛顺序补齐 %s 个", meta["selected"], topped_up)

        return retrieved_ids, meta

    @staticmethod
    def _repair_numeric_ids(parsed_ids: list[str], shortlist_ids: list[str]) -> tuple[list[str], int]:
        """
        模型有时只返回 id 的数字部分（如 "296" 而不是 "ref_296"）。
        短名单里恰好只有一个 id 以该数字结尾时补回完整 id；返回 (去重后的 id 列表, 补全个数)。
        """
        shortlist_id_set = set(shortlist_ids)
        by_number: dict[str, list[str]] = {}
        for ref_id in shortlist_ids:
            match = re.search(r"(\d+)$", ref_id)
            if match:
                by_number.setdefault(match.group(1), []).append(ref_id)

        repaired_ids: list[str] = []
        repaired = 0
        for ref_id in parsed_ids:
            if ref_id not in shortlist_id_set and ref_id.isdigit() and len(by_number.get(ref_id, [])) == 1:
                ref_id = by_number[ref_id][0]
                repaired += 1
            if ref_id not in repaired_ids:
                repaired_ids.append(ref_id)
        return repaired_ids, repaired

    def _load_candidate_thumbnail(self, item: dict, cfg: dict) -> str:
        image_path = resolve_data_asset_path(
            item.get("path_to_gt_image"),
            cfg["task_name"],
            dataset_name=self.exp_config.dataset_name,
            work_dir=self.exp_config.work_dir,
        )
        if image_path is None:
            return ""
        try:
            return image_utils.load_image_as_jpeg_base64(str(image_path), VISUAL_RERANK_THUMBNAIL_MAX_SIDE)
        except Exception as exc:
            logger.warning("⚠️  参考缩略图读取失败，跳过该图: %s (%s)", image_path, exc)
            return ""

    async def _select_with_thumbnails(
        self,
        data: Dict[str, Any],
        cfg: dict,
        candidate_pool: list[dict],
        meta: dict[str, Any],
        *,
        candidate_id: Any = "N/A",
    ) -> str:
        """
        预筛候选附缩略图，由模型按图类型、布局与风格挑选。

        失败（异常、空响应或没有可用缩略图）时返回空串，由调用方改走只发 caption 的选择。
        """
        content = stringify_payload(data["content"])[:VISUAL_RERANK_TARGET_CHAR_LIMIT]
        content_list: list[dict[str, Any]] = [{
            "type": "text",
            "text": f"TARGET caption: {data['visual_intent']}\nTARGET methodology (truncated): {content}\n\nCANDIDATES:",
        }]
        attached = 0
        for item in candidate_pool:
            caption = str(item.get("visual_intent", "") or "")[:VISUAL_RERANK_CAPTION_CHAR_LIMIT]
            content_list.append({
                "type": "text",
                "text": f"\nCandidate {item['id']} ({cfg['candidate_labels'][0]}: {item['id']}; caption: {caption}):",
            })
            thumbnail = self._load_candidate_thumbnail(item, cfg)
            if thumbnail:
                content_list.append({
                    "type": "image",
                    "source": {"type": "base64", "data": thumbnail, "media_type": "image/jpeg"},
                })
                attached += 1
        content_list.append({
            "type": "text",
            "text": (
                '\nReturn JSON {"top10_references": [10 candidate ids, best first]}. '
                f'Copy each id exactly as written after "{cfg["candidate_labels"][0]}:"; never shorten it to a number.'
            ),
        })
        if attached == 0:
            return ""

        try:
            response_list = await self.call_text_api(
                content_list,
                model_name=self.model_name,
                system_prompt=VISUAL_RERANK_SYSTEM_PROMPT,
                temperature=self.exp_config.temperature,
                max_output_tokens=50000,
                max_attempts=3,
                retry_delay=30,
                error_context=f"retriever-visual[candidate={candidate_id}]",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("⚠️  附缩略图的检索选择失败，改为只发 caption: %s", exc)
            meta["visual_rerank_failed"] = True
            return ""

        raw_response = str(response_list[0]).strip() if response_list else ""
        if not raw_response or raw_response == "Error":
            logger.warning("⚠️  附缩略图的检索选择没有返回结果，改为只发 caption")
            meta["visual_rerank_failed"] = True
            return ""
        meta["method"] = "bm25+vlm"
        meta["thumbnails"] = attached
        return raw_response

    async def _select_with_captions(
        self,
        data: Dict[str, Any],
        cfg: dict,
        candidate_pool: list[dict],
        *,
        lite: bool,
        candidate_id: Any = "N/A",
    ) -> str:
        """候选只附 caption（full 模式再附正文），由模型挑选。"""
        raw_content = data["content"]
        content = stringify_payload(raw_content)
        visual_intent = data["visual_intent"]

        user_prompt = f"**Target Input**\n- {cfg['target_labels'][0]}: {visual_intent}\n- {cfg['target_labels'][1]}: {content}\n\n**Candidate Pool**\n"
        for item in candidate_pool:
            # 候选只用 id 标识，不给序号，避免模型把序号当成 id 返回
            user_prompt += f"Candidate {cfg['candidate_type']} [{item['id']}]:\n"
            user_prompt += f"- {cfg['candidate_labels'][0]}: {item['id']}\n"
            user_prompt += f"- {cfg['candidate_labels'][1]}: {item.get('visual_intent', '')}\n"
            if not lite:
                user_prompt += f"- {cfg['candidate_labels'][2]}: {stringify_payload(item.get('content', ''))}\n"
            user_prompt += "\n"

        user_prompt += f"Now, based on the Target Input and the Candidate Pool, {cfg['instruction_suffix']}"
        content_list = [{"type": "text", "text": user_prompt}]

        prompt_chars = len(user_prompt)
        logger.debug(f"📊 auto 检索 prompt: {prompt_chars:,} 字符 (~{prompt_chars//4:,} tokens), lite={lite}")

        response_list = await self.call_text_api(
            content_list,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            temperature=self.exp_config.temperature,
            max_output_tokens=50000,
            max_attempts=3,
            retry_delay=30,
            error_context=f"retriever[candidate={candidate_id},lite={lite}]",
        )
        return str(response_list[0]).strip() if response_list else ""

    def _parse_retrieval_result(self, raw_response: str, task_name: str) -> list:
        import json_repair

        last_error = None
        for candidate in self._json_payload_candidates(raw_response):
            try:
                parsed = json_repair.loads(candidate)
                ids = self._extract_retrieval_ids_from_payload(parsed, task_name)
                if ids:
                    return ids
            except Exception as e:
                last_error = e

        ids_from_text = self._extract_reference_ids_from_text(raw_response)
        if ids_from_text:
            logger.warning("⚠️  检索结果不是标准 JSON，已从文本中提取参考 id")
            return ids_from_text

        if last_error is not None:
            logger.warning(f"⚠️  解析检索结果失败: {last_error}")
        else:
            logger.warning("⚠️  检索结果中没有可用参考 id")
        logger.debug(f"   原始响应: {raw_response[:500]}...")
        return []


# 与对比实验使用的提示一致
VISUAL_RERANK_SYSTEM_PROMPT = (
    "You select reference figures that will be given to an image-generation model as visual templates. "
    "Choose candidates whose diagram type, layout/composition and visual style would best serve as a template "
    "for drawing the target figure. Structure and layout matter more than research topic. Avoid near-duplicates."
)

QUERY_REWRITE_SYSTEM_PROMPT = "You turn figure requests written in any language into English search keywords."

QUERY_REWRITE_USER_PROMPT = """Below are the caption and the source content of a target {target_type} for an academic paper. They may be written in Chinese or another language.
Write 40-80 English keywords and short phrases covering: the figure type and layout, the main components or modules, and the research topic.
Output only the keywords, separated by commas.

Caption: {caption}

Content: {content}
"""


DIAGRAM_RETRIEVER_AGENT_SYSTEM_PROMPT = """
# Background & Goal
We are building an **AI system to automatically generate method diagrams for academic papers**. Given a paper's methodology section and a figure caption, the system needs to create a high-quality illustrative diagram that visualizes the described method.

To help the AI learn how to generate appropriate diagrams, we use a **few-shot learning approach**: we provide it with reference examples of similar diagrams. The AI will learn from these examples to understand what kind of diagram to create for the target.

# Your Task
**You are the Retrieval Agent.** Your job is to select the most relevant reference diagrams from a candidate pool that will serve as few-shot examples for the diagram generation model.

You will receive:
- **Target Input:** The methodology section and caption of the diagram we need to generate
- **Candidate Pool:** A shortlist of existing diagrams (each with an ID and a caption; in some modes also the methodology section)

You must select the **Top 10 candidates** that would be most helpful as examples for teaching the AI how to draw the target diagram.

# Selection Logic (Topic + Intent)

Your goal is to find examples that match the Target in both **Domain** and **Diagram Type**.

**1. Match Research Topic (Use Methodology & Caption):**
* What is the domain? (e.g., Agent & Reasoning, Vision & Perception, Generative & Learning, Science & Applications).
* Select candidates that belong to the **same research domain**.
* *Why?* Similar domains share similar terminology (e.g., "Actor-Critic" in RL).

**2. Match Visual Intent (Use Caption & Keywords):**
* What type of diagram is implied? (e.g., "Framework", "Pipeline", "Detailed Module", "Performance Chart").
* Select candidates with **similar visual structures**.
* *Why?* A "Framework" diagram example is useless for drawing a "Performance Bar Chart", even if they are in the same domain.

**Ranking Priority:**
1.  **Best Match:** Same Topic AND Same Visual Intent (e.g., Target is "Agent Framework" -> Candidate is "Agent Framework", Target is "Dataset Construction Pipeline" -> Candidate is "Dataset Construction Pipeline").
2.  **Second Best:** Same Visual Intent (e.g., Target is "Agent Framework" -> Candidate is "Vision Framework"). *Structure is more important than Topic for drawing.*
3.  **Avoid:** Different Visual Intent (e.g., Target is "Pipeline" -> Candidate is "Bar Chart").

# Input Data

## Target Input
-   **Caption:** [Caption of the target diagram]
-   **Methodology section:** [Methodology section of the target paper]

## Candidate Pool
List of candidate diagrams, each structured as follows:

Candidate Diagram [<id>]:
-   **Diagram ID:** [Exact ID string of the candidate diagram, also shown in brackets after "Candidate Diagram"]
-   **Caption:** [Caption of the candidate diagram]
-   **Methodology section:** [Methodology section of the candidate's paper]


# Output Format
Provide your output strictly in the following JSON format, containing only the **exact IDs** of the Top 10 selected diagrams. Copy each ID exactly as written after "Diagram ID:" in the Candidate Pool; never use a candidate's position in the list and never invent IDs:
```json
{
  "top10_references": [
    "<best candidate id>",
    "<2nd candidate id>",
    "...",
    "<10th candidate id>"
  ]
}```
"""

PLOT_RETRIEVER_AGENT_SYSTEM_PROMPT = """
# Background & Goal
We are building an **AI system to automatically generate statistical plots**. Given a plot's raw data and the visual intent, the system needs to create a high-quality visualization that effectively presents the data.

To help the AI learn how to generate appropriate plots, we use a **few-shot learning approach**: we provide it with reference examples of similar plots. The AI will learn from these examples to understand what kind of plot to create for the target data.

# Your Task
**You are the Retrieval Agent.** Your job is to select the most relevant reference plots from a candidate pool that will serve as few-shot examples for the plot generation model.

You will receive:
- **Target Input:** The raw data and visual intent of the plot we need to generate
- **Candidate Pool:** Reference plots (each with raw data and visual intent)

You must select the **Top 10 candidates** that would be most helpful as examples for teaching the AI how to create the target plot.

# Selection Logic (Data Type + Visual Intent)

Your goal is to find examples that match the Target in both **Data Characteristics** and **Plot Type**.

**1. Match Data Characteristics (Use Raw Data & Visual Intent):**
* What type of data is it? (e.g., categorical vs numerical, single series vs multi-series, temporal vs comparative).
* What are the data dimensions? (e.g., 1D, 2D, 3D).
* Select candidates with **similar data structures and characteristics**.
* *Why?* Different data types require different visualization approaches.

**2. Match Visual Intent (Use Visual Intent):**
* What type of plot is implied? (e.g., "bar chart", "scatter plot", "line chart", "pie chart", "heatmap", "radar chart").
* Select candidates with **similar plot types**.
* *Why?* A "bar chart" example is more useful for generating another bar chart than a "scatter plot" example, even if the data domains are similar.

**Ranking Priority:**
1.  **Best Match:** Same Data Type AND Same Plot Type (e.g., Target is "multi-series line chart" -> Candidate is "multi-series line chart").
2.  **Second Best:** Same Plot Type with compatible data (e.g., Target is "bar chart with 5 categories" -> Candidate is "bar chart with 6 categories").
3.  **Avoid:** Different Plot Type (e.g., Target is "bar chart" -> Candidate is "pie chart"), unless there are no more candidates with the same plot type.

# Input Data

## Target Input
-   **Visual Intent:** [Visual intent of the target plot]
-   **Raw Data:** [Raw data to be visualized]

## Candidate Pool
List of candidate plots, each structured as follows:

Candidate Plot [<id>]:
-   **Plot ID:** [Exact ID string of the candidate plot, also shown in brackets after "Candidate Plot"]
-   **Visual Intent:** [Visual intent of the candidate plot]
-   **Raw Data:** [Raw data of the candidate plot]


# Output Format
Provide your output strictly in the following JSON format, containing only the **exact Plot IDs** of the Top 10 selected plots. Copy each ID exactly as written after "Plot ID:" in the Candidate Pool; never use a candidate's position in the list and never invent IDs:
```json
{
  "top10_references": [
    "<best candidate id>",
    "<2nd candidate id>",
    "...",
    "<10th candidate id>"
  ]
}```
"""
