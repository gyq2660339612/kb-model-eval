"""sentence-transformers 本地推理后端(离线批处理 / 无服务化场景)。

- Embedding:SentenceTransformer,query 用 prompt_name="query"(Qwen3 内置指令模板);
- Reranker:CrossEncoder,默认加 sigmoid 输出 0-1 概率分。
"""

from __future__ import annotations

from typing import List, Optional

from .base import EmbeddingBackend, RerankerBackend


class STEmbeddingBackend(EmbeddingBackend):
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        batch_size: int = 32,
        use_query_prompt: bool = True,
    ):
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self.model = SentenceTransformer(model_path, device=device)
        self.batch_size = batch_size
        self.use_query_prompt = use_query_prompt
        self.name = f"st-embedding:{model_path}"

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        kwargs = {"batch_size": self.batch_size, "normalize_embeddings": True}
        if self.use_query_prompt:
            kwargs["prompt_name"] = "query"
        return self.model.encode(texts, **kwargs).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True
        ).tolist()


class STRerankerBackend(RerankerBackend):
    def __init__(self, model_path: str, device: Optional[str] = None, sigmoid: bool = True):
        import torch  # 延迟导入
        from sentence_transformers import CrossEncoder

        activation = torch.nn.Sigmoid() if sigmoid else None
        self.model = CrossEncoder(model_path, device=device, activation_fn=activation)
        self.name = f"st-reranker:{model_path}"

    def score(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]
