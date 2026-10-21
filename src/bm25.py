"""BM25 基线(纯 Python,Okapi BM25)。

总框架 §3.3 要求最低组合清单中包含 BM25-only 基线;
对中英文混合语料采用"英文数字按词、中文按字"的分词策略,无需第三方分词库。
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from .data_loader import Corpus, Queries, doc_full_text
from .eval_retrieval import compute_by_category
from .metrics import Qrels, evaluate_all, per_query_metrics

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Okapi:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, documents: List[List[str]]) -> "BM25Okapi":
        self.n_docs = len(documents)
        self.doc_len = [len(d) for d in documents]
        self.avgdl = sum(self.doc_len) / self.n_docs if self.n_docs else 1.0
        self.tf = [Counter(d) for d in documents]
        df: Counter = Counter()
        for d in documents:
            df.update(set(d))
        # Robertson/Spärck Jones idf(加 1 防负值)
        self.idf = {
            t: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for t, freq in df.items()
        }
        return self

    def score(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.n_docs
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.n_docs):
                f = self.tf[i].get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * f * (self.k1 + 1) / denom
        return scores


def run_bm25_eval(
    corpus: Corpus,
    queries: Queries,
    qrels: Qrels,
    k_values: Iterable[int] = (1, 3, 5, 10),
    top_k: int = 100,
    labels: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict:
    """与 run_retrieval_eval 返回结构一致(无向量维度/编码耗时,替换为建索引耗时)"""
    doc_ids = list(corpus.keys())
    t0 = time.perf_counter()
    index = BM25Okapi().fit([tokenize(doc_full_text(corpus[d])) for d in doc_ids])
    index_seconds = time.perf_counter() - t0

    eff_k = max(1, min(top_k, len(doc_ids)))
    results: Dict[str, List[str]] = {}
    t0 = time.perf_counter()
    for qid, qtext in queries.items():
        scores = index.score(tokenize(qtext))
        order = sorted(range(len(doc_ids)), key=lambda i: scores[i], reverse=True)
        results[qid] = [doc_ids[i] for i in order[:eff_k]]
    query_seconds = time.perf_counter() - t0

    metrics = evaluate_all(qrels, results, k_values)
    performance = {
        "num_documents": len(doc_ids),
        "num_queries": len(queries),
        "index_seconds": round(index_seconds, 3),
        "query_seconds": round(query_seconds, 3),
        "query_avg_latency_ms": round(query_seconds / len(queries) * 1000, 2) if queries else None,
        "vector_dim": None,
        "top_k": eff_k,
        "note": "纯 Python Okapi BM25 基线(k1=1.5, b=0.75),非生产检索引擎性能",
    }
    return {
        "metrics": metrics,
        "by_category": compute_by_category(qrels, results, labels, k_values),
        "per_query": per_query_metrics(qrels, results, k_values),
        "performance": performance,
        "results": results,
    }
