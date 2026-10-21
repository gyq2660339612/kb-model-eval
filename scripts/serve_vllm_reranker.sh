#!/usr/bin/env bash
# 启动 Qwen3 Reranker 服务(vLLM,/rerank 与 /score 端点)
# 用法: bash scripts/serve_vllm_reranker.sh [模型路径] [端口] [jinja模板路径]
# 注意:原始权重必须通过 hf_overrides 转换;模板文件在 vLLM 源码
#       examples/pooling/score/template/qwen3_reranker.jinja,内网需随代码拷贝。
set -euo pipefail

MODEL_PATH="${1:-/models/Qwen3-Reranker-8B}"
PORT="${2:-8001}"
CHAT_TEMPLATE="${3:-qwen3_reranker.jinja}"

exec vllm serve "${MODEL_PATH}" \
  --runner pooling \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
  --chat-template "${CHAT_TEMPLATE}" \
  --host 0.0.0.0 --port "${PORT}"
