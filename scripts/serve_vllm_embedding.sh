#!/usr/bin/env bash
# 启动 Qwen3 Embedding 服务(vLLM,OpenAI 兼容 /v1/embeddings)
# 用法: bash scripts/serve_vllm_embedding.sh [模型路径] [端口] [TP卡数]
set -euo pipefail

MODEL_PATH="${1:-/models/Qwen3-Embedding-8B}"
PORT="${2:-8000}"
TP="${3:-1}"

exec vllm serve "${MODEL_PATH}" \
  --runner pooling \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port "${PORT}"
