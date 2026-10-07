"""参考池词法检索：BM25 索引与多字段融合排序。

只依赖 numpy；索引按参考池构建一次，可在多次检索间复用。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

_WORD_RE = re.compile(r"[a-z][a-z0-9+\-]{1,}")
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")

# 英文常用虚词 + 本任务里几乎每条参考都会出现的通用词
STOPWORDS = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
        "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how",
        "if", "in", "into", "is", "it", "its", "itself", "just", "may", "me", "more", "most", "much",
        "must", "my", "no", "nor", "not", "now", "of", "off", "on", "once", "one", "only", "or", "other",
        "our", "ours", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
        "that", "the", "their", "theirs", "them", "then", "there", "these", "they", "this", "those",
        "through", "to", "too", "under", "until", "up", "upon", "us", "use", "used", "uses", "using",
        "very", "via", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
        "why", "will", "with", "within", "without", "would", "you", "your", "yours", "et", "al", "eq",
        "i.e", "e.g", "etc", "let", "denote", "denotes", "denoted", "given", "thus", "hence", "however",
        "therefore", "where", "respectively", "first", "second", "two", "three", "new", "based",
        # 通用领域词
        "figure", "fig", "diagram", "plot", "method", "methods", "section", "visual", "intent", "data",
        "raw", "create", "overview", "illustration", "paper", "propose", "proposed", "approach", "shown",
        "show", "shows",
    }
)


def stringify_payload(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def tokenize(text: Any) -> list[str]:
    """英文小写分词：以字母开头、长度 ≥2，去掉停用词。"""
    return [token for token in _WORD_RE.findall(str(text or "").lower()) if token not in STOPWORDS]


def cjk_ratio(text: Any) -> float:
    """中日韩字符在“中日韩字符 + 英文字母”中的占比，用于判断查询是否以中文为主。"""
    value = str(text or "")
    cjk = len(_CJK_RE.findall(value))
    if cjk == 0:
        return 0.0
    ascii_letters = len(_ASCII_LETTER_RE.findall(value))
    return cjk / float(cjk + ascii_letters)


class BM25Field:
    """单字段 BM25（Okapi），倒排索引实现；查询词去重后计分。"""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75):
        self.size = len(docs)
        self.k1 = k1
        self.b = b
        lengths = np.array([len(doc) for doc in docs], dtype=np.float64)
        avg_length = float(lengths.mean()) if self.size and lengths.mean() > 0 else 1.0
        self._norm = k1 * (1.0 - b + b * lengths / avg_length)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, doc in enumerate(docs):
            for term, freq in Counter(doc).items():
                postings[term].append((doc_idx, freq))
        self._postings: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for term, entries in postings.items():
            doc_ids = np.fromiter((entry[0] for entry in entries), dtype=np.int64, count=len(entries))
            freqs = np.fromiter((entry[1] for entry in entries), dtype=np.float64, count=len(entries))
            df = len(entries)
            idf = math.log(1.0 + (self.size - df + 0.5) / (df + 0.5))
            self._postings[term] = (doc_ids, freqs, idf)

    def score(self, query_tokens: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.size, dtype=np.float64)
        for term in set(query_tokens):
            posting = self._postings.get(term)
            if posting is None:
                continue
            doc_ids, freqs, idf = posting
            scores[doc_ids] += idf * freqs * (self.k1 + 1.0) / (freqs + self._norm[doc_ids])
        return scores

    def overlap(self, query_tokens: Sequence[str]) -> np.ndarray:
        """每篇文档包含多少个不同的查询词（不加 IDF 权重）。"""
        counts = np.zeros(self.size, dtype=np.float64)
        for term in set(query_tokens):
            posting = self._postings.get(term)
            if posting is not None:
                counts[posting[0]] += 1.0
        return counts


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std


class ReferencePoolIndex:
    """参考池的双字段索引：caption（visual_intent）与正文（content）。

    融合分数 = caption_weight * z(BM25 caption) + (1 - caption_weight) * z(BM25 正文)
             + overlap_weight * z(caption 共有词数)。
    diagram 的 caption 很长，共有词数会偏向长 caption，因此只在 plot 上使用该项。
    """

    CONTENT_CHAR_LIMIT = 20000
    QUERY_CAPTION_BODY_TOKENS = 400
    QUERY_BODY_TOKENS = 2000

    def __init__(self, items: Sequence[dict], *, caption_weight: float = 0.5, overlap_weight: float = 0.0):
        self.size = len(items)
        self.caption_weight = float(caption_weight)
        self.overlap_weight = float(overlap_weight)
        self._caption = BM25Field([tokenize(item.get("visual_intent", "")) for item in items])
        self._content = BM25Field(
            [tokenize(stringify_payload(item.get("content", ""))[: self.CONTENT_CHAR_LIMIT]) for item in items]
        )

    def query_tokens(self, query_caption: Any, query_content: Any) -> tuple[list[str], list[str]]:
        caption_tokens = tokenize(query_caption)
        body_tokens = tokenize(stringify_payload(query_content)[: self.CONTENT_CHAR_LIMIT])
        return caption_tokens, body_tokens

    def rank(self, query_caption: Any, query_content: Any) -> tuple[list[int], int]:
        """返回 (按相关度降序的参考下标, 有效查询词数)。分数相同按原顺序。"""
        caption_tokens, body_tokens = self.query_tokens(query_caption, query_content)
        caption_query = caption_tokens + body_tokens[: self.QUERY_CAPTION_BODY_TOKENS]
        body_query = body_tokens[: self.QUERY_BODY_TOKENS]
        token_count = len(set(caption_query) | set(body_query))
        if self.size == 0:
            return [], token_count
        fused = self.caption_weight * _zscore(self._caption.score(caption_query))
        if self.caption_weight < 1.0:
            fused = fused + (1.0 - self.caption_weight) * _zscore(self._content.score(body_query))
        if self.overlap_weight:
            fused = fused + self.overlap_weight * _zscore(self._caption.overlap(caption_tokens))
        order = np.argsort(-fused, kind="stable")
        return [int(idx) for idx in order], token_count
