"""BEIR 格式数据集加载。

目录结构:
    <data_dir>/corpus.jsonl   {"_id": "d1", "title": "...", "text": "..."}
    <data_dir>/queries.jsonl  {"_id": "q1", "text": "..."}
    <data_dir>/qrels.tsv      query-id <TAB> corpus-id <TAB> score(首行表头可选)
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict

from .metrics import Qrels

Corpus = Dict[str, Dict[str, str]]
Queries = Dict[str, str]


def load_corpus(data_dir: str) -> Corpus:
    path = os.path.join(data_dir, "corpus.jsonl")
    corpus: Corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("_id", obj.get("id")))
            corpus[doc_id] = {"title": obj.get("title", ""), "text": obj["text"]}
    if not corpus:
        raise ValueError(f"语料为空:{path}")
    return corpus


def load_queries(data_dir: str) -> Queries:
    path = os.path.join(data_dir, "queries.jsonl")
    queries: Queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("_id", obj.get("id")))
            queries[qid] = obj["text"]
    if not queries:
        raise ValueError(f"查询集为空:{path}")
    return queries


def load_qrels(data_dir: str) -> Qrels:
    path = os.path.join(data_dir, "qrels.tsv")
    qrels: Qrels = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            qid, doc_id, score = row[0].strip(), row[1].strip(), row[2].strip()
            if not score.lstrip("-").isdigit():  # 跳过表头(query-id corpus-id score)
                continue
            qrels.setdefault(qid, {})[doc_id] = int(score)
    if not qrels:
        raise ValueError(f"相关性标注为空:{path}")
    return qrels


def load_dataset(data_dir: str) -> tuple[Corpus, Queries, Qrels]:
    corpus = load_corpus(data_dir)
    queries = load_queries(data_dir)
    qrels = load_qrels(data_dir)
    missing = set(qrels) - set(queries)
    if missing:
        raise ValueError(f"qrels 中存在 queries.jsonl 没有的查询:{sorted(missing)}")
    return corpus, queries, qrels


def doc_full_text(doc: Dict[str, str]) -> str:
    """title + text 拼接(与 BEIR 默认做法一致)"""
    title = (doc.get("title") or "").strip()
    return f"{title}\n{doc['text']}" if title else doc["text"]
