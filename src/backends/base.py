"""模型后端抽象:Embedding(双塔编码)与 Reranker(交叉打分)。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingBackend(ABC):
    """文本嵌入后端。query 与 document 分开编码(query 侧可能需要指令前缀)。"""

    name: str = "abstract"

    @abstractmethod
    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """编码查询列表(实现方负责拼接指令模板)"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """编码文档列表(不应加任何指令前缀)"""


class RerankerBackend(ABC):
    """重排序后端:对 (query, documents) 逐一给出相关性分数(越高越相关)。"""

    name: str = "abstract"

    @abstractmethod
    def score(self, query: str, documents: List[str]) -> List[float]:
        """返回与 documents 等长的相关性分数列表"""
