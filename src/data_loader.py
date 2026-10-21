"""测试数据加载(BEIR 格式为主,兼容总框架 §9 定义的扩展格式)。

目录结构(<data_dir>/):
    corpus.jsonl    必需。{"_id"|"id": "d1", "title": "...", "text": "...", "metadata": {...}}
    queries.jsonl   必需。{"_id"|"id": "q1", "text": "...",
                           "labels": {"category": "...", "language": "...", "difficulty": "..."},
                           "relevant_doc_ids": ["d1", ...](可选,无 qrels 文件时的兜底标注)}
    qrels.tsv       优先。query-id <TAB> corpus-id <TAB> score(表头可选,分级 0-3)
    qrels.jsonl     备选。{"query_id": "q1", "doc_id": "d1", "relevance": 2}

加载优先级:qrels.tsv > qrels.jsonl > queries.jsonl 中的 relevant_doc_ids(按二元相关=1)。
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

from .metrics import Qrels

Corpus = Dict[str, Dict[str, str]]
Queries = Dict[str, str]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: 非法 JSON:{exc}") from exc
    if not rows:
        raise ValueError(f"文件为空:{path}")
    return rows


def _row_id(obj: Dict[str, Any]) -> str:
    return str(obj.get("_id", obj.get("id")))


def load_corpus(data_dir: str) -> Corpus:
    path = os.path.join(data_dir, "corpus.jsonl")
    corpus: Corpus = {}
    for obj in _read_jsonl(path):
        doc_id = _row_id(obj)
        if doc_id in corpus:
            raise ValueError(f"corpus.jsonl 中文档 ID 重复:{doc_id}")
        corpus[doc_id] = {
            "title": obj.get("title", ""),
            "text": obj["text"],
            "metadata": obj.get("metadata", {}),
        }
    return corpus


def load_query_rows(data_dir: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(data_dir, "queries.jsonl")
    rows: Dict[str, Dict[str, Any]] = {}
    for obj in _read_jsonl(path):
        qid = _row_id(obj)
        if qid in rows:
            raise ValueError(f"queries.jsonl 中查询 ID 重复:{qid}")
        rows[qid] = obj
    return rows


def load_queries(data_dir: str) -> Queries:
    return {qid: str(row["text"]) for qid, row in load_query_rows(data_dir).items()}


def load_query_labels(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """返回 {qid: labels}(无 labels 字段的查询返回空 dict)"""
    return {qid: dict(row.get("labels", {})) for qid, row in load_query_rows(data_dir).items()}


def load_qrels(data_dir: str) -> Optional[Qrels]:
    """按优先级加载 qrels;都不存在时返回 None(由 load_dataset 兜底)"""
    tsv_path = os.path.join(data_dir, "qrels.tsv")
    if os.path.exists(tsv_path):
        qrels: Qrels = {}
        with open(tsv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) < 3:
                    continue
                qid, doc_id, score = row[0].strip(), row[1].strip(), row[2].strip()
                if not score.lstrip("-").isdigit():  # 跳过表头
                    continue
                qrels.setdefault(qid, {})[doc_id] = int(score)
        return qrels or None

    jsonl_path = os.path.join(data_dir, "qrels.jsonl")
    if os.path.exists(jsonl_path):
        qrels = {}
        for obj in _read_jsonl(jsonl_path):
            qid = str(obj.get("query_id", obj.get("qid")))
            doc_id = str(obj.get("doc_id"))
            rel = int(obj.get("relevance", obj.get("score", 0)))
            qrels.setdefault(qid, {})[doc_id] = rel
        return qrels or None
    return None


def _qrels_from_queries(rows: Dict[str, Dict[str, Any]]) -> Optional[Qrels]:
    qrels: Qrels = {}
    for qid, row in rows.items():
        for doc_id in row.get("relevant_doc_ids", []) or []:
            qrels.setdefault(qid, {})[str(doc_id)] = 1
    return qrels or None


def load_dataset(data_dir: str) -> tuple[Corpus, Queries, Qrels, Dict[str, Dict[str, Any]]]:
    """返回 (corpus, queries, qrels, labels)"""
    corpus = load_corpus(data_dir)
    rows = load_query_rows(data_dir)
    queries = {qid: str(row["text"]) for qid, row in rows.items()}
    labels = {qid: dict(row.get("labels", {})) for qid, row in rows.items()}

    qrels = load_qrels(data_dir)
    if qrels is None:
        qrels = _qrels_from_queries(rows)
    if not qrels:
        raise ValueError(
            f"{data_dir} 缺少相关性标注:需要 qrels.tsv、qrels.jsonl 之一,"
            "或 queries.jsonl 中的 relevant_doc_ids 字段"
        )

    missing_q = set(qrels) - set(queries)
    if missing_q:
        raise ValueError(f"qrels 中存在 queries.jsonl 没有的查询:{sorted(missing_q)}")
    unknown_d = {d for qr in qrels.values() for d in qr} - set(corpus)
    if unknown_d:
        raise ValueError(f"qrels 中存在 corpus.jsonl 没有的文档:{sorted(unknown_d)}")
    return corpus, queries, qrels, labels


def doc_full_text(doc: Dict[str, str]) -> str:
    """title + text 拼接(与 BEIR 默认做法一致)"""
    title = (doc.get("title") or "").strip()
    return f"{title}\n{doc['text']}" if title else doc["text"]
