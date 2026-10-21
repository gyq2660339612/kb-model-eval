"""管线冒烟测试(对应总框架 §10.4 SM-02/06/08/09)。

运行:python -m unittest tests.test_pipeline -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backends.mock_backend import MockEmbeddingBackend, MockRerankerBackend  # noqa: E402
from src.bm25 import BM25Okapi, tokenize  # noqa: E402
from src.data_loader import load_dataset  # noqa: E402
from src.eval_rerank import run_rerank_eval  # noqa: E402
from src.eval_retrieval import run_retrieval_eval  # noqa: E402
from src.metrics import pairwise_accuracy  # noqa: E402

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample")


class TestPipelineSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus, cls.queries, cls.qrels, cls.labels = load_dataset(SAMPLE_DIR)

    def test_sm09_mock_determinism(self):
        """SM-09:相同模型、输入、参数运行两次,排名必须一致"""
        backend = MockEmbeddingBackend()
        r1 = run_retrieval_eval(backend, self.corpus, self.queries, self.qrels)
        r2 = run_retrieval_eval(MockEmbeddingBackend(), self.corpus, self.queries, self.qrels)
        self.assertEqual(r1["results"], r2["results"])
        self.assertEqual(r1["metrics"], r2["metrics"])

    def test_sm02_mock_semantic_recall(self):
        """SM-02:mock 管线能跑通且总体命中合理(词面重叠的查询应召回正例)"""
        backend = MockEmbeddingBackend()
        out = run_retrieval_eval(backend, self.corpus, self.queries, self.qrels)
        self.assertGreaterEqual(out["metrics"]["metrics"]["hit_rate@10"], 0.5)
        self.assertIn("pairwise_accuracy", out["metrics"]["metrics"])
        self.assertEqual(out["metrics"]["num_unanswerable"], 1)  # q11 单独计数

    def test_rerank_pipeline(self):
        """SM-07:重排流程跑通,输出前后对比与逐 query 指标"""
        emb = MockEmbeddingBackend()
        ret = run_retrieval_eval(emb, self.corpus, self.queries, self.qrels)
        rr = run_rerank_eval(MockRerankerBackend(), self.corpus, self.queries,
                             self.qrels, ret["results"], rerank_top_k=10,
                             labels=self.labels)
        self.assertIn("before", rr)
        self.assertIn("after", rr)
        self.assertIn("ndcg@10", rr["delta"])
        self.assertTrue(rr["per_query"])

    def test_sm08_bad_data_raises(self):
        """SM-08:坏数据必须明确报错,不静默产生分数"""
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "corpus.jsonl"), "w", encoding="utf-8") as f:
                f.write('{"_id": "d1", "text": "hello"}\n{bad json}\n')
            with open(os.path.join(tmp, "queries.jsonl"), "w", encoding="utf-8") as f:
                f.write('{"_id": "q1", "text": "hi", "relevant_doc_ids": ["d1"]}\n')
            with self.assertRaises(ValueError):
                load_dataset(tmp)

    def test_loader_rejects_unknown_ids(self):
        """SM-08 补充:qrels 引用不存在的文档/查询必须报错"""
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "corpus.jsonl"), "w", encoding="utf-8") as f:
                f.write('{"_id": "d1", "text": "hello"}\n')
            with open(os.path.join(tmp, "queries.jsonl"), "w", encoding="utf-8") as f:
                f.write('{"_id": "q1", "text": "hi"}\n')
            with open(os.path.join(tmp, "qrels.tsv"), "w", encoding="utf-8") as f:
                f.write("q1\td999\t2\n")
            with self.assertRaises(ValueError):
                load_dataset(tmp)

    def test_bm25_baseline(self):
        """BM25 基线:含独特关键词的文档应排第一(总框架 §3.3 清单第 7 项)"""
        docs = [tokenize(t) for t in ["EUV 光刻 极紫外 光源", "清洗 工艺 颗粒 去除", "刻蚀 等离子体 各向异性"]]
        index = BM25Okapi().fit(docs)
        scores = index.score(tokenize("极紫外 光源"))
        self.assertEqual(scores.index(max(scores)), 0)

    def test_pairwise_accuracy_handcomputed(self):
        """Pairwise Accuracy 手工核算:qrels 2/1/0 三档"""
        qrels_q = {"d1": 2, "d2": 1, "d3": 0}
        self.assertAlmostEqual(pairwise_accuracy(qrels_q, ["d1", "d2", "d3"]), 1.0)
        # d2 排到 d1 前:3 对(d1>d2 错、d1>d3 对、d2>d3 对)→ 2/3
        self.assertAlmostEqual(pairwise_accuracy(qrels_q, ["d2", "d1", "d3"]), 2 / 3)


if __name__ == "__main__":
    unittest.main()
