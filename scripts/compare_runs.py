#!/usr/bin/env python3
"""两个评测 run 的逐 query 指标配对比较 + paired bootstrap 置信区间。

总框架 §8.3 要求:模型差异较小时不能凭 1 次运行下结论,需成对比较并给出置信区间。

用法:
    python scripts/compare_runs.py results/runA/per_query_results.jsonl \
        results/runB/per_query_results.jsonl --metric ndcg@10
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def load_per_query(path: str) -> dict[str, dict]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[str(row["query_id"])] = row["metrics"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="两个 run 的配对比较与 bootstrap 置信区间")
    parser.add_argument("run_a", help="run A 的 per_query_results.jsonl")
    parser.add_argument("run_b", help="run B 的 per_query_results.jsonl")
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    a = load_per_query(args.run_a)
    b = load_per_query(args.run_b)
    common = sorted(set(a) & set(b))
    if not common:
        sys.exit("两个 run 没有共同的 query_id")
    if len(common) < len(a) or len(common) < len(b):
        print(f"警告:仅 {len(common)} 条共同查询(A={len(a)}, B={len(b)})")

    va = np.array([a[q][args.metric] for q in common], dtype=float)
    vb = np.array([b[q][args.metric] for q in common], dtype=float)
    diff = vb - va

    rng = np.random.default_rng(args.seed)
    boot = np.empty(args.bootstrap)
    n = len(common)
    for i in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        boot[i] = diff[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])

    win = int((diff > 0).sum())
    tie = int((diff == 0).sum())
    loss = int((diff < 0).sum())
    summary = {
        "metric": args.metric,
        "num_queries": n,
        "mean_A": round(float(va.mean()), 6),
        "mean_B": round(float(vb.mean()), 6),
        "delta_abs": round(float(diff.mean()), 6),
        "delta_rel_pct": round(float(diff.mean() / va.mean() * 100), 2) if va.mean() else None,
        "bootstrap_95_ci": [round(float(lo), 6), round(float(hi), 6)],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "per_query": {"B_better": win, "tie": tie, "A_better": loss},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["ci_excludes_zero"]:
        better = "B" if diff.mean() > 0 else "A"
        print(f"\n结论:在 95% 置信水平下,{better} 显著优于另一方({args.metric})。")
    else:
        print("\n结论:95% 置信区间包含 0,差异不显著——按总框架 §8.3 应扩大测试集或重复实验。")


if __name__ == "__main__":
    main()
