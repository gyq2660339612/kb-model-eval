#!/usr/bin/env python3
"""无 GPU 流程演示:用 mock 后端在内置半导体示例集上跑通"召回 + 重排"全流程。

    python scripts/demo_mock.py

mock 后端基于词袋哈希,仅验证流程与指标计算,不代表真实模型效果。
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.run_evaluation import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = [
        "run_evaluation",
        "--backend", "mock",
        "--rerank-backend", "mock",
        "--data-dir", os.path.join(PROJECT_ROOT, "data", "sample"),
        "--k-values", "1,3,5,10",
        "--top-k", "22",
        "--rerank-top-k", "10",
    ]
    main()
