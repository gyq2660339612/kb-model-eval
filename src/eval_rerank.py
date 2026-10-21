"""重排序评估:对召回 top-k 逐 query 打分重排,对比重排前后指标。

支持两种候选来源(总框架 §7.5 要求候选集固定且可复现):
1. 实时:直接取召回结果的前 rerank_top_k(默认);
2. 冻结:从 candidates.jsonl 读取预先冻结的候选集及其初始排名。
"""

from __future__ import annotations

import json
import time
from typing import Dict, Iterable, List, Optional

from .backends.base import RerankerBackend
from .data_loader import Corpus, Queries, doc_full_text
from .metrics import Qrels, evaluate_all, per_query_metrics


def load_candidates(path: str) -> Dict[str, List[str]]:
    """加载冻结候选集 candidates.jsonl:

    {"query_id": "q1", "candidates": [{"doc_id": "d1", "initial_rank": 1}, ...]}
    返回 {qid: [doc_id, ...]}(按 initial_rank 升序)
    """
    candidates: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id", obj.get("qid", obj.get("_id"))))
            docs = sorted(
                obj["candidates"], key=lambda c: c.get("initial_rank", 1 << 30)
            )
            candidates[qid] = [str(c["doc_id"]) for c in docs]
    if not candidates:
        raise ValueError(f"候选集为空:{path}")
    return candidates


def dump_candidates(
    results: Dict[str, List[str]], top_k: int, path: str
) -> None:
    """把召回结果冻结为 candidates.jsonl(供后续固定候选集重排实验复用)"""
    with open(path, "w", encoding="utf-8") as f:
        for qid, ranked in results.items():
            row = {
                "query_id": qid,
                "candidates": [
                    {"doc_id": d, "initial_rank": i}
                    for i, d in enumerate(ranked[:top_k], start=1)
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_rerank_eval(
    reranker: RerankerBackend,
    corpus: Corpus,
    queries: Queries,
    qrels: Qrels,
    base_results: Dict[str, List[str]],
    k_values: Iterable[int] = (1, 3, 5, 10),
    rerank_top_k: int = 20,
    candidates: Optional[Dict[str, List[str]]] = None,
    labels: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """base_results:召回排序结果 {qid: [doc_id,...]};candidates 提供时以冻结候选为准。

    返回 {"before","after","delta","delta_pct","by_category_after","per_query",
          "performance","results"}
    """
    base_for_eval = candidates if candidates is not None else base_results

    reranked: Dict[str, List[str]] = {}
    latencies_ms: List[float] = []
    for qid, ranked in base_for_eval.items():
        cand = ranked[:rerank_top_k]
        docs = [doc_full_text(corpus[d]) for d in cand]
        t0 = time.perf_counter()
        scores = reranker.score(queries[qid], docs)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        order = sorted(range(len(cand)), key=lambda i: scores[i], reverse=True)
        reranked[qid] = [cand[i] for i in order] + ranked[rerank_top_k:]

    before = evaluate_all(qrels, base_for_eval, k_values)
    after = evaluate_all(qrels, reranked, k_values)

    delta = {key: round(after["metrics"][key] - before["metrics"][key], 6)
             for key in after["metrics"]}
    delta_pct = {}
    for key, val in after["metrics"].items():
        base_val = before["metrics"][key]
        delta_pct[key] = round((val - base_val) / base_val * 100, 2) if base_val else None

    from .eval_retrieval import compute_by_category

    lat_sorted = sorted(latencies_ms)
    n = len(lat_sorted) or 1
    performance = {
        "rerank_top_k": rerank_top_k,
        "num_queries": len(latencies_ms),
        "avg_latency_ms": round(sum(latencies_ms) / n, 2),
        "p95_latency_ms": round(lat_sorted[min(n - 1, int(0.95 * n))], 2),
        "frozen_candidates": candidates is not None,
    }
    return {
        "before": before,
        "after": after,
        "delta": delta,
        "delta_pct": delta_pct,
        "by_category_after": compute_by_category(qrels, reranked, labels, k_values),
        "per_query": per_query_metrics(qrels, reranked, k_values),
        "performance": performance,
        "results": reranked,
    }
