"""Mock 后端:无需 GPU / 网络,用于验证评测流程。

- MockEmbeddingBackend:字符 n-gram 哈希词袋向量(确定性、L2 归一化);
- MockRerankerBackend:字符 bigram 重合度打分。
仅用于流程演示与单元测试,**不代表任何真实模型效果**。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List

from .base import EmbeddingBackend, RerankerBackend

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")


def _char_ngrams(text: str, n: int = 3) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    joined = " ".join(tokens)
    grams = [joined[i : i + n] for i in range(max(0, len(joined) - n + 1))]
    return grams or tokens or ["<empty>"]


def _bucket(key: str, dim: int) -> int:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % dim


class MockEmbeddingBackend(EmbeddingBackend):
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"mock-embedding(dim={dim})"

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for gram in _char_ngrams(text):
            vec[_bucket(gram, self.dim)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]


class MockRerankerBackend(RerankerBackend):
    def __init__(self):
        self.name = "mock-reranker(bigram-overlap)"

    def score(self, query: str, documents: List[str]) -> List[float]:
        q_grams = set(_char_ngrams(query, n=2))
        scores = []
        for doc in documents:
            d_grams = set(_char_ngrams(doc, n=2))
            union = len(q_grams | d_grams) or 1
            scores.append(len(q_grams & d_grams) / union)
        return scores
