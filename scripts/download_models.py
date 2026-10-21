#!/usr/bin/env python3
"""模型下载脚本:从 ModelScope(默认,国内首选)或 HuggingFace 下载本项目涉及的模型权重。

用法:
    python scripts/download_models.py                      # 下载推荐集(8B 主力 + 轻量对照)
    python scripts/download_models.py --models all         # 全部模型
    python scripts/download_models.py --source hf          # 走 HuggingFace(自动用 hf-mirror 镜像)
    python scripts/download_models.py --dry-run            # 只打印命令不执行
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

MODEL_SETS = {
    # 主力:8B
    "qwen3-embedding-8b": "Qwen/Qwen3-Embedding-8B",
    "qwen3-reranker-8b": "Qwen/Qwen3-Reranker-8B",
    # 小参数
    "qwen3-embedding-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-reranker-4b": "Qwen/Qwen3-Reranker-4B",
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-reranker-0.6b": "Qwen/Qwen3-Reranker-0.6B",
    # BGE 对照
    "bge-m3": "BAAI/bge-m3",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
}

RECOMMENDED = [
    "qwen3-embedding-8b",
    "qwen3-reranker-8b",
    "qwen3-embedding-0.6b",
    "qwen3-reranker-0.6b",
    "bge-m3",
    "bge-reranker-v2-m3",
]


def build_command(source: str, repo_id: str, local_dir: str) -> list[str]:
    if source == "modelscope":
        return ["modelscope", "download", "--model", repo_id, "--local_dir", local_dir]
    return ["huggingface-cli", "download", repo_id, "--local-dir", local_dir]


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 embedding/reranker 模型权重")
    parser.add_argument("--source", choices=["modelscope", "hf"], default="modelscope")
    parser.add_argument("--models", default="recommended",
                        help=f"逗号分隔的模型名、all 或 recommended;可选:{','.join(MODEL_SETS)}")
    parser.add_argument("--output-dir", default="./models")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.models == "all":
        names = list(MODEL_SETS)
    elif args.models == "recommended":
        names = RECOMMENDED
    else:
        names = [n.strip() for n in args.models.split(",") if n.strip()]
    unknown = [n for n in names if n not in MODEL_SETS]
    if unknown:
        sys.exit(f"未知模型:{unknown};可选:{list(MODEL_SETS)}")

    if args.source == "hf":
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        print("已设置 HF_ENDPOINT=https://hf-mirror.com(社区镜像)")

    for name in names:
        repo_id = MODEL_SETS[name]
        local_dir = os.path.join(args.output_dir, repo_id.split("/")[-1])
        cmd = build_command(args.source, repo_id, local_dir)
        print(f"\n=== {name} -> {local_dir}")
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True, env=os.environ.copy())

    print("\n全部完成。模型权重请通过合规渠道拷入内网,部署见 docs/06-内网部署指南.md")


if __name__ == "__main__":
    main()
