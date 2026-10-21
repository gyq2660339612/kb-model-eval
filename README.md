# 知识库模型评估(KB Model Eval)

面向**企业内网知识库检索(RAG)场景**的 Embedding / Reranker 模型调研、评估与部署项目。

本项目完成四件事:

1. **调研**:系统梳理当前(2025–2026)主流的 embedding 与 reranker 模型 —— 以 **Qwen3-Embedding / Qwen3-Reranker(8B 为旗舰)** 为核心,Qwen3 小参数模型(0.6B / 4B)与 **BGE 系列**(bge-m3、bge-v1.5、bge-reranker-v2-m3 等)为对照,含模型介绍、参数对比、官方 benchmark 结论;
2. **标准**:构建一套完整的知识库检索**评价标准与指标体系**(Recall@k / Hit Rate@k / nDCG@k / MRR / MAP / 延迟 / 吞吐 / 显存),含验收门槛建议;
3. **代码**:提供可直接在内网运行的**评测代码**(测试用例格式、检索评估、重排评估、报告生成),支持 vLLM(OpenAI 兼容 API)与 sentence-transformers 两种后端;
4. **部署**:给出从模型下载(ModelScope / HF 镜像)到 vLLM / TEI / SGLang / Xinference / Ollama 的**内网离线部署全流程指南**。

## 选型结论速览

| 角色 | 推荐 | 说明 |
|---|---|---|
| Embedding 主力 | **Qwen3-Embedding-8B** | MTEB 多语言 70.58 / 英文 v2 75.22 / C-MTEB 73.84 / MTEB-Code 80.68,2025-06 发布时多语言榜第 1;4096 维、32K 上下文、支持 MRL 降维与指令感知,Apache 2.0 |
| Reranker 主力 | **Qwen3-Reranker-8B** | CMTEB-R 77.45 / MMTEB-R 72.94 / MTEB-Code 81.22;yes/no 二分类交叉编码 |
| 轻量 Embedding | Qwen3-Embedding-0.6B / 4B | 0.6B 全面超越同尺寸 BGE-M3;4B 性价比最高(多语言 69.45,接近 8B) |
| 轻量 Reranker | Qwen3-Reranker-0.6B / 4B | 0.6B 已超 bge-reranker-v2-m3;4B 在 MTEB-R 与 FollowIR 上反超 8B |
| 对照基线 | BGE-M3 / bge-reranker-v2-m3 | MIT/Apache 可商用、生态成熟、支持稠密+稀疏+多向量混合检索 |

> 详细数据见 [docs/05-官方Benchmark结论.md](docs/05-官方Benchmark结论.md)。

## 项目结构

```
知识库模型评估/
├── README.md                        # 本文件
├── requirements.txt                 # Python 依赖
├── docs/                            # 调研与标准文档
│   ├── 01-模型调研-Embedding.md      # 嵌入模型介绍(Qwen3 / BGE / 竞品)
│   ├── 02-模型调研-Reranker.md       # 重排序模型介绍
│   ├── 03-模型参数对比.md            # 全量参数对比表(维度/上下文/协议/显存)
│   ├── 04-评估指标体系.md            # 基准介绍 + 指标定义 + 验收标准
│   ├── 05-官方Benchmark结论.md       # 官方 benchmark 完整数据与结论
│   └── 06-内网部署指南.md            # 下载、vLLM/TEI/SGLang/Xinference/Ollama 部署
├── configs/
│   ├── models.json                  # 模型注册表(机器可读)
│   └── eval_config.json             # 评测默认配置与验收阈值
├── data/sample/                     # 示例测试集(半导体领域,BEIR 格式)
│   ├── corpus.jsonl                 # 文档库
│   ├── queries.jsonl                # 查询集
│   └── qrels.tsv                    # 人工相关性标注
├── src/                             # 评测代码
│   ├── metrics.py                   # 指标实现(纯 Python,零依赖)
│   ├── data_loader.py               # BEIR 格式数据加载
│   ├── backends/                    # 模型后端:vLLM(OpenAI API)/ sentence-transformers / mock
│   ├── eval_retrieval.py            # 检索(召回)评估
│   ├── eval_rerank.py               # 重排序评估(召回前后对比)
│   └── run_evaluation.py            # 命令行入口
├── scripts/
│   ├── download_models.py           # 模型下载(ModelScope / HF 镜像)
│   ├── serve_vllm_embedding.sh      # vLLM 启动 embedding 服务
│   ├── serve_vllm_reranker.sh       # vLLM 启动 reranker 服务
│   └── demo_mock.py                 # 无 GPU 流程演示(mock 后端)
└── tests/test_metrics.py            # 指标单元测试
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 如需 sentence-transformers 本地推理后端:
pip install sentence-transformers
```

### 2. 无 GPU 验证流程(mock 后端,使用内置半导体示例测试集)

```bash
python scripts/demo_mock.py
```

### 3. 内网真实评测(以 vLLM 为例)

```bash
# 步骤一:启动 embedding 服务(详见 docs/06)
bash scripts/serve_vllm_embedding.sh /path/to/Qwen3-Embedding-8B 8000
# 步骤二:启动 reranker 服务
bash scripts/serve_vllm_reranker.sh /path/to/Qwen3-Reranker-8B 8001

# 步骤三:运行评测(召回 + 重排,输出 JSON + Markdown 报告)
python -m src.run_evaluation \
  --backend openai --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-Embedding-8B \
  --rerank-backend openai --rerank-base-url http://127.0.0.1:8001 --rerank-model Qwen/Qwen3-Reranker-8B \
  --data-dir data/sample --top-k 10 --rerank-top-k 20 \
  --output results/report
```

### 4. 用自己的知识库数据评测

按 `data/sample/` 的 BEIR 格式准备三个文件(`corpus.jsonl` / `queries.jsonl` / `qrels.tsv`),把 `--data-dir` 指向你的目录即可。格式规范见 [docs/04-评估指标体系.md](docs/04-评估指标体系.md) 第 5 节。

## 文档索引

| 文档 | 内容 |
|---|---|
| [01-模型调研-Embedding](docs/01-模型调研-Embedding.md) | Qwen3-Embedding 架构/训练/用法,BGE 全系,竞品(Gemini / e5 / Harrier 等) |
| [02-模型调研-Reranker](docs/02-模型调研-Reranker.md) | Qwen3-Reranker 原理(yes/no 打分)、prompt 模板,BGE reranker,召回+重排管线 |
| [03-模型参数对比](docs/03-模型参数对比.md) | 参数量 / 维度 / MRL / 上下文 / 语言 / 协议 / 显存估算 全表 |
| [04-评估指标体系](docs/04-评估指标体系.md) | MTEB / C-MTEB / BEIR / MIRACL 介绍,指标严格定义,测试集构建规范,验收标准 |
| [05-官方Benchmark结论](docs/05-官方Benchmark结论.md) | 各 benchmark 完整分数表与结论分析 |
| [06-内网部署指南](docs/06-内网部署指南.md) | 离线下载、vLLM / TEI / SGLang / Xinference / Ollama 部署、NPU/CPU 备选 |

## 说明

- 所有 benchmark 数字均来自官方模型卡 / 官方博客 / 原始论文,来源 URL 见各文档末尾。
- `mock` 后端(词袋哈希向量)仅用于在无 GPU 环境下验证评测流程,**不代表任何真实模型效果**。
- 本项目调研时间为 2026 年 10 月;榜单分数随时间变化,复评时请以 [MTEB 官方榜单](https://huggingface.co/spaces/mteb/leaderboard) 最新数据为准。
