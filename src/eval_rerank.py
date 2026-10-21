"""重排序评估:对召回 top-k 逐 query 打分重排,对比重排前后指标。"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List

from .backends.base import RerankerBackend
from .data_loader import Corpus, Queries, doc_full_text
from .metrics import Qrels, evaluate_all


def run_rerank_eval(
    reranker: RerankerBackend,
    corpus: Corpus,
    queries: Queries,
    qrels: Qrels,
    base_results: Dict[str, List[str]],
    k_values: Iterable[int] = (1, 3, 5, 10),
    rerank_top_k: int = 20,
) -> Dict:
    """base_results:召回阶段的排序结果 {qid: [doc_id,...]}(eval_retrieval 的输出)。

    返回 {"before":..., "after":..., "delta_pct":..., "performance":..., "results":...}
    """
    reranked: Dict[str, List[str]] = {}
    latencies_ms: List[float] = []
    for qid, ranked in base_results.items():
        candidates = ranked[:rerank_top_k]
        docs = [doc_full_text(corpus[d]) for d in candidates]
        t0 = time.perf_counter()
        scores = reranker.score(queries[qid], docs)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        reranked[qid] = [candidates[i] for i in order] + ranked[rerank_top_k:]

    before = evaluate_all(qrels, base_results, k_values)
    after = evaluate_all(qrels, reranked, k_values)

    delta_pct = {}
    for key, val in after["metrics"].items():
        base_val = before["metrics"][key]
        delta_pct[key] = round((val - base_val) / base_val * 100, 2) if base_val else None

    lat_sorted = sorted(latencies_ms)
    n = len(lat_sorted) or 1
    performance = {
        "rerank_top_k": rerank_top_k,
        "num_queries": len(latencies_ms),
        "avg_latency_ms": round(sum(latencies_ms) / n, 2),
        "p95_latency_ms": round(lat_sorted[min(n - 1, int(0.95 * n))], 2),
    }
    return {
        "before": before,
        "after": after,
        "delta_pct": delta_pct,
        "performance": performance,
        "results": reranked,
    }
