"""OpenAI 兼容 API 后端(vLLM / TEI / Xinference 等),仅需 requests。

- Embedding:POST {base_url}/embeddings;query 侧自动拼接 Qwen3 官方指令模板
  `Instruct: {instruction}\\nQuery: {query}`,document 不加前缀;可选 MRL `dimensions` 截断。
- Reranker:POST {server}/rerank(Jina/Cohere 兼容),失败时回退 /score 成对打分。
"""

from __future__ import annotations

import time
from typing import List, Optional

import requests

from .base import EmbeddingBackend, RerankerBackend

DEFAULT_QWEN3_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def _server_root(base_url: str) -> str:
    """http://host:port/v1 -> http://host:port"""
    root = base_url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


class OpenAIEmbeddingBackend(EmbeddingBackend):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        instruction: Optional[str] = DEFAULT_QWEN3_INSTRUCTION,
        dimensions: Optional[int] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.api_key = api_key
        self.instruction = instruction
        self.dimensions = dimensions
        self.timeout = timeout
        self.max_retries = max_retries
        self.name = f"openai-embedding:{model}"

    def _post(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()["data"]
                data = sorted(data, key=lambda d: d["index"])
                return [d["embedding"] for d in data]
            except Exception as err:  # noqa: BLE001
                last_err = err
                time.sleep(2**attempt)
        raise RuntimeError(f"embedding 请求失败({self.url}):{last_err}")

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        if self.instruction:
            texts = [f"Instruct: {self.instruction}\nQuery: {t}" for t in texts]
        return self._post(texts)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._post(texts)


class OpenAIRerankerBackend(RerankerBackend):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        root = _server_root(base_url)
        self.rerank_url = root + "/rerank"
        self.score_url = root + "/score"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.name = f"openai-reranker:{model}"

    def _post(self, url: str, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as err:  # noqa: BLE001
                last_err = err
                time.sleep(2**attempt)
        raise RuntimeError(f"rerank 请求失败({url}):{last_err}")

    def score(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []
        # 优先 /rerank(Jina/Cohere 兼容,一次请求返回全部文档分数)
        try:
            data = self._post(
                self.rerank_url,
                {"model": self.model, "query": query, "documents": documents},
            )
            scores = [0.0] * len(documents)
            for item in data["results"]:
                scores[item["index"]] = float(item["relevance_score"])
            return scores
        except Exception:  # noqa: BLE001 — 回退 /score 成对打分
            data = self._post(
                self.score_url,
                {"model": self.model, "queries": [query] * len(documents), "documents": documents},
            )
            return [float(item["score"]) for item in data["data"]]
