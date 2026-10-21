"""检索(召回)评估:embedding 全库编码 → 余弦相似度 top-k → 指标计算 + 分层指标 + 性能统计。"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .backends.base import EmbeddingBackend
from .data_loader import Corpus, Queries, doc_full_text
from .metrics import Qrels, evaluate_all, per_query_metrics


def _batched(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def compute_by_category(
    qrels: Qrels,
    results: Dict[str, List[str]],
    labels: Optional[Dict[str, Dict[str, Any]]],
    k_values: Iterable[int],
) -> Dict[str, Dict]:
    """按查询标签 category 分层计算指标(防止总分掩盖关键类别缺陷)"""
    if not labels:
        return {}
    groups: Dict[str, Qrels] = {}
    for qid in qrels:
        category = labels.get(qid, {}).get("category")
        if category:
            groups.setdefault(str(category), {})[qid] = qrels[qid]
    by_category: Dict[str, Dict] = {}
    for category, sub_qrels in sorted(groups.items()):
        if any(r > 0 for qr in sub_qrels.values() for r in qr.values()):
            by_category[category] = evaluate_all(sub_qrels, results, k_values)
    return by_category


def run_retrieval_eval(
    backend: EmbeddingBackend,
    corpus: Corpus,
    queries: Queries,
    qrels: Qrels,
    k_values: Iterable[int] = (1, 3, 5, 10),
    batch_size: int = 32,
    top_k: int = 100,
    labels: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict:
    """返回 {"metrics","by_category","per_query","performance","results"}"""
    doc_ids = list(corpus.keys())
    doc_texts = [doc_full_text(corpus[d]) for d in doc_ids]

    t0 = time.perf_counter()
    doc_vecs: List[List[float]] = []
    for chunk in _batched(doc_texts, batch_size):
        doc_vecs.extend(backend.embed_documents(chunk))
    doc_seconds = time.perf_counter() - t0

    qids = list(queries.keys())
    t0 = time.perf_counter()
    query_vecs: List[List[float]] = []
    for chunk in _batched([queries[q] for q in qids], batch_size):
        query_vecs.extend(backend.embed_queries(chunk))
    query_seconds = time.perf_counter() - t0

    doc_mat = _l2_normalize(np.asarray(doc_vecs, dtype=np.float32))
    query_mat = _l2_normalize(np.asarray(query_vecs, dtype=np.float32))
    sim = query_mat @ doc_mat.T

    eff_k = max(1, min(top_k, len(doc_ids)))
    top_idx = np.argpartition(-sim, eff_k - 1, axis=1)[:, :eff_k]
    results: Dict[str, List[str]] = {}
    for row, qid in enumerate(qids):
        order = top_idx[row][np.argsort(-sim[row, top_idx[row]])]
        results[qid] = [doc_ids[j] for j in order]

    metrics = evaluate_all(qrels, results, k_values)
    performance = {
        "num_documents": len(doc_ids),
        "num_queries": len(qids),
        "doc_embed_seconds": round(doc_seconds, 3),
        "doc_throughput_per_sec": round(len(doc_ids) / doc_seconds, 2) if doc_seconds > 0 else None,
        "query_embed_seconds": round(query_seconds, 3),
        "query_avg_latency_ms": round(query_seconds / len(qids) * 1000, 2) if qids else None,
        "vector_dim": int(doc_mat.shape[1]),
        "top_k": eff_k,
    }
    return {
        "metrics": metrics,
        "by_category": compute_by_category(qrels, results, labels, k_values),
        "per_query": per_query_metrics(qrels, results, k_values),
        "performance": performance,
        "results": results,
    }
