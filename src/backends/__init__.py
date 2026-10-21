"""后端工厂:按名称构建 Embedding / Reranker 后端。"""

from __future__ import annotations

from typing import Optional

from .base import EmbeddingBackend, RerankerBackend
from .mock_backend import MockEmbeddingBackend, MockRerankerBackend
from .openai_backend import (
    DEFAULT_QWEN3_INSTRUCTION,
    OpenAIEmbeddingBackend,
    OpenAIRerankerBackend,
)

__all__ = [
    "EmbeddingBackend",
    "RerankerBackend",
    "MockEmbeddingBackend",
    "MockRerankerBackend",
    "OpenAIEmbeddingBackend",
    "OpenAIRerankerBackend",
    "build_embedding_backend",
    "build_reranker_backend",
    "DEFAULT_QWEN3_INSTRUCTION",
]


def build_embedding_backend(
    backend: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: str = "EMPTY",
    instruction: Optional[str] = DEFAULT_QWEN3_INSTRUCTION,
    dimensions: Optional[int] = None,
    device: Optional[str] = None,
    batch_size: int = 32,
) -> EmbeddingBackend:
    if backend == "mock":
        return MockEmbeddingBackend()
    if backend == "openai":
        if not (base_url and model):
            raise ValueError("openai 后端需要 --base-url 与 --model")
        return OpenAIEmbeddingBackend(
            base_url=base_url,
            model=model,
            api_key=api_key,
            instruction=instruction,
            dimensions=dimensions,
        )
    if backend == "st":
        if not model:
            raise ValueError("st 后端需要 --model(本地模型路径)")
        from .st_backend import STEmbeddingBackend

        return STEmbeddingBackend(model, device=device, batch_size=batch_size)
    raise ValueError(f"未知 embedding 后端:{backend}(可选 mock / openai / st)")


def build_reranker_backend(
    backend: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: str = "EMPTY",
    device: Optional[str] = None,
) -> RerankerBackend:
    if backend == "mock":
        return MockRerankerBackend()
    if backend == "openai":
        if not (base_url and model):
            raise ValueError("openai rerank 后端需要 --rerank-base-url 与 --rerank-model")
        return OpenAIRerankerBackend(base_url=base_url, model=model, api_key=api_key)
    if backend == "st":
        if not model:
            raise ValueError("st rerank 后端需要 --rerank-model(本地模型路径)")
        from .st_backend import STRerankerBackend

        return STRerankerBackend(model, device=device)
    raise ValueError(f"未知 reranker 后端:{backend}(可选 mock / openai / st)")
