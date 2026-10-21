"""检索/重排序评估指标(纯 Python,零第三方依赖)。

指标口径与 trec_eval / pytrec_eval / ranx 一致:
- nDCG 增益形式为 (2^rel - 1) / log2(rank + 1);
- 所有指标先对单查询计分,再对查询集取算术平均;
- 仅统计 qrels 中至少有一个相关文档(score > 0)的查询。

qrels 格式:  {query_id: {doc_id: relevance(int)}}
results 格式: {query_id: [doc_id, ...]}  按相关度降序排列
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

Qrels = Dict[str, Dict[str, int]]
RunResults = Dict[str, List[str]]


def _gain(rel: int) -> float:
    return float((2 ** rel) - 1)


def _relevant_set(qrels_q: Dict[str, int]) -> set:
    return {d for d, r in qrels_q.items() if r > 0}


def precision_at_k(qrels_q: Dict[str, int], ranked: List[str], k: int) -> float:
    """Precision@k = |Top-k ∩ 相关文档| / k"""
    if k <= 0:
        return 0.0
    hits = sum(1 for d in ranked[:k] if qrels_q.get(d, 0) > 0)
    return hits / k


def recall_at_k(qrels_q: Dict[str, int], ranked: List[str], k: int) -> float:
    """Recall@k = |Top-k ∩ 相关文档| / |全部相关文档|"""
    relevant = _relevant_set(qrels_q)
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / len(relevant)


def hit_rate_at_k(qrels_q: Dict[str, int], ranked: List[str], k: int) -> float:
    """HitRate@k:Top-k 中至少含 1 个相关文档记 1,否则 0(即 Success@k)"""
    relevant = _relevant_set(qrels_q)
    return 1.0 if any(d in relevant for d in ranked[:k]) else 0.0


def mrr_at_k(qrels_q: Dict[str, int], ranked: List[str], k: int) -> float:
    """MRR@k = 1 / 首个相关文档的名次;前 k 无相关记 0"""
    for i, d in enumerate(ranked[:k], start=1):
        if qrels_q.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def ap_at_k(qrels_q: Dict[str, int], ranked: List[str], k: int) -> float:
    """AP@k = (1/R) Σ_{i:第i位相关} P@i;对所有查询平均即 MAP@k"""
    relevant = _relevant_set(qrels_q)
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def dcg_at_k(rels_sorted: Iterable[int], k: int) -> float:
    """DCG@k = Σ_{i=1..k} (2^rel_i - 1) / log2(i + 1)"""
    return sum(
        _gain(rel) / math.log2(i + 1) for i, rel in enumerate(rels_sorted, start=1) if i <= k
    )


def ndcg_at_k(qrels_q: Dict[str, int], ranked: List[str], k: int) -> float:
    """nDCG@k = DCG@k / IDCG@k,支持分级相关度"""
    ideal = sorted((r for r in qrels_q.values() if r > 0), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    actual = [qrels_q.get(d, 0) for d in ranked]
    return dcg_at_k(actual, k) / idcg


_METRIC_FUNCS = {
    "recall": recall_at_k,
    "hit_rate": hit_rate_at_k,
    "precision": precision_at_k,
    "mrr": mrr_at_k,
    "map": ap_at_k,
    "ndcg": ndcg_at_k,
}

DEFAULT_K_VALUES: Tuple[int, ...] = (1, 3, 5, 10)


def evaluate_all(
    qrels: Qrels,
    results: RunResults,
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> Dict:
    """对整个查询集计算全部指标的平均值。

    返回 {"num_queries": n, "metrics": {"recall@1": ..., "ndcg@10": ..., ...}}
    """
    ks = sorted({int(k) for k in k_values if int(k) > 0})
    if not ks:
        raise ValueError("k_values 不能为空")
    sums = {f"{name}@{k}": 0.0 for k in ks for name in _METRIC_FUNCS}
    num_queries = 0
    for qid, qrels_q in qrels.items():
        if not _relevant_set(qrels_q):
            continue
        ranked = results.get(qid, [])
        num_queries += 1
        for k in ks:
            for name, func in _METRIC_FUNCS.items():
                sums[f"{name}@{k}"] += func(qrels_q, ranked, k)
    if num_queries == 0:
        raise ValueError("qrels 中没有任何带相关文档的查询")
    metrics = {key: val / num_queries for key, val in sums.items()}
    return {"num_queries": num_queries, "metrics": metrics}
