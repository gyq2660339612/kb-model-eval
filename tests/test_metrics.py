"""指标计算单元测试(手工核算值)。

运行:python -m unittest tests.test_metrics -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import (  # noqa: E402
    ap_at_k,
    dcg_at_k,
    evaluate_all,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestMetrics(unittest.TestCase):
    """qrels: d1=2 分(高度相关), d2=1 分(部分相关);ranked 为模型排序输出"""

    qrels_q = {"d1": 2, "d2": 1}

    def test_perfect_ranking(self):
        ranked = ["d1", "d2", "d3"]
        self.assertAlmostEqual(recall_at_k(self.qrels_q, ranked, 1), 0.5)
        self.assertAlmostEqual(recall_at_k(self.qrels_q, ranked, 2), 1.0)
        self.assertAlmostEqual(precision_at_k(self.qrels_q, ranked, 2), 1.0)
        self.assertAlmostEqual(hit_rate_at_k(self.qrels_q, ranked, 1), 1.0)
        self.assertAlmostEqual(mrr_at_k(self.qrels_q, ranked, 10), 1.0)
        self.assertAlmostEqual(ap_at_k(self.qrels_q, ranked, 10), 1.0)
        self.assertAlmostEqual(ndcg_at_k(self.qrels_q, ranked, 2), 1.0)

    def test_imperfect_ranking(self):
        ranked = ["d3", "d1", "d2"]  # 首名不相关
        self.assertAlmostEqual(recall_at_k(self.qrels_q, ranked, 1), 0.0)
        self.assertAlmostEqual(recall_at_k(self.qrels_q, ranked, 2), 0.5)
        self.assertAlmostEqual(hit_rate_at_k(self.qrels_q, ranked, 1), 0.0)
        self.assertAlmostEqual(precision_at_k(self.qrels_q, ranked, 3), 2 / 3)
        self.assertAlmostEqual(mrr_at_k(self.qrels_q, ranked, 10), 0.5)
        # AP = (1/2 + 2/3) / 2
        self.assertAlmostEqual(ap_at_k(self.qrels_q, ranked, 10), (0.5 + 2 / 3) / 2)
        # nDCG@3:actual gains [0,2,1] -> 3/log2(3) + 1/log2(4);ideal [2,1]
        actual = 3 / 1.5849625007211562 + 1 / 2.0
        ideal = 3.0 + 1 / 1.5849625007211562
        self.assertAlmostEqual(ndcg_at_k(self.qrels_q, ranked, 3), actual / ideal, places=6)

    def test_dcg_gain_form(self):
        # trec_eval 增益形式:(2^rel - 1) / log2(i + 1)
        self.assertAlmostEqual(dcg_at_k([2], 1), 3.0)  # log2(2) = 1
        self.assertAlmostEqual(dcg_at_k([1, 1], 2), 1 + 1 / 1.5849625007211562)

    def test_empty_result(self):
        self.assertAlmostEqual(recall_at_k(self.qrels_q, [], 10), 0.0)
        self.assertAlmostEqual(mrr_at_k(self.qrels_q, [], 10), 0.0)

    def test_evaluate_all_aggregation(self):
        qrels = {"q1": {"d1": 2, "d2": 1}, "q2": {"d9": 0}}  # q2 无正例,应被跳过
        results = {"q1": ["d1", "d2"], "q2": ["d9"]}
        out = evaluate_all(qrels, results, k_values=(1, 2))
        self.assertEqual(out["num_queries"], 1)
        self.assertAlmostEqual(out["metrics"]["recall@2"], 1.0)
        self.assertAlmostEqual(out["metrics"]["ndcg@1"], 1.0)
        self.assertIn("map@2", out["metrics"])

    def test_evaluate_all_requires_positive(self):
        with self.assertRaises(ValueError):
            evaluate_all({"q1": {"d1": 0}}, {"q1": ["d1"]})


if __name__ == "__main__":
    unittest.main()
