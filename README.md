# 知识库模型评估(KB Model Eval)

面向**企业内网知识库检索(RAG)场景**的 Embedding / Reranker 模型调研、评估与部署项目。

本项目完成五件事:

1. **体系框架**:一套完整的测评体系总纲(L1 召回 / L2 重排 / L3 系统三层分离,14 个能力维度,评分规则与验收门槛);
2. **调研**:主流模型梳理 —— 以 **Qwen3-Embedding / Qwen3-Reranker(8B 为旗舰)** 为核心,Qwen3 小参数(0.6B/4B)与 **BGE 系列**为对照,含模型介绍、参数对比、官方 benchmark 结论;
3. **标准**:指标口径(Recall@k / Hit Rate@k / nDCG@k / MRR / MAP / **Pairwise Accuracy** / Δ指标 / 延迟 / 吞吐 / 显存)+ 硬门槛与综合评分规则;
4. **代码**:可直接在内网运行的评测代码(BEIR/JSONL 双格式、**BM25 基线**、**冻结候选集**、**分层指标**、逐 query 输出、**bootstrap 置信区间对比**),支持 vLLM(OpenAI 兼容 API)与 sentence-transformers 后端;
5. **部署**:从模型下载(ModelScope / HF 镜像)到 vLLM / TEI / SGLang / Xinference / Ollama 的**内网离线部署全流程指南**。

## 选型结论速览

| 角色 | 推荐 | 说明 |
|---|---|---|
| Embedding 主力 | **Qwen3-Embedding-8B** | MTEB 多语言 70.58 / 英文 v2 75.22 / C-MTEB 73.84 / MTEB-Code 80.68;4096 维、32K 上下文、MRL 降维、指令感知,Apache 2.0 |
| Reranker 主力 | **Qwen3-Reranker-8B** | CMTEB-R 77.45 / MMTEB-R 72.94 / MTEB-Code 81.22(中文最强) |
| 必须对照 | **Qwen3-4B 系列** | Reranker-4B 在 MTEB-R 与 FollowIR 上反超 8B——"最大参数必然最佳"是错误假设 |
| 成本基线 | Qwen3-0.6B 系列 | 全面超越同尺寸 BGE;满足验收线时运营成本优势显著 |
| 对照基线 | BGE-M3 / bge-reranker-v2-m3 + **BM25** | MIT/Apache 可商用;dense/sparse/multi-vector 需分别测 |

> 公开分数是参考线而非内网保证值;最终决策以"内部测试集 + 固定候选重排 + 目标硬件压测"为准。详细数据见 [docs/05](docs/05-官方Benchmark结论.md),决策流程见 [docs/08](docs/08-实验流程与报告模板.md)。

## 项目结构

```
知识库模型评估/
├── README.md                        # 本文件
├── requirements.txt                 # Python 依赖(核心仅 numpy + requests)
├── docs/                            # 体系框架 + 调研 + 标准 + 部署
│   ├── 00-评估体系总框架.md          # 【总纲】三层测评体系、能力维度、评分与验收规则
│   ├── 01-模型调研-Embedding.md      # 嵌入模型介绍(Qwen3 / BGE / 竞品)
│   ├── 02-模型调研-Reranker.md       # 重排序模型介绍与管线
│   ├── 03-模型参数对比.md            # 全量参数对比表(维度/上下文/协议/显存)
│   ├── 04-评估指标体系.md            # 基准介绍 + 指标严格定义 + 验收门槛
│   ├── 05-官方Benchmark结论.md       # 官方 benchmark 完整数据与结论
│   ├── 06-内网部署指南.md            # 下载、vLLM/TEI/SGLang/Xinference/Ollama 部署
│   ├── 07-测试集构建与标注指南.md    # 规模/13类分层/标注标准/文件格式/冒烟用例
│   └── 08-实验流程与报告模板.md      # 组合清单、评分权重、复现准则、Phase 0-5
├── configs/
│   ├── models.json                  # 模型注册表(13 个模型,机器可读)
│   └── eval_config.json             # k 值、验收门槛、分类别门槛、综合评分权重
├── data/sample/                     # 示例测试集(半导体领域,26 文档 11 查询)
│   ├── corpus.jsonl                 # 文档库(含报警码/数值约束难负例)
│   ├── queries.jsonl                # 查询集(含 category 分层标签)
│   └── qrels.tsv                    # 0-2 分级人工标注(含显式 0 分难负例)
├── src/                             # 评测代码
│   ├── metrics.py                   # 指标(Recall/HitRate/Precision/MRR/MAP/nDCG/Pairwise,纯 Python)
│   ├── data_loader.py               # BEIR/JSONL 双格式加载 + 校验
│   ├── bm25.py                      # BM25-only 基线(纯 Python Okapi)
│   ├── backends/                    # 模型后端:vLLM(OpenAI API)/ sentence-transformers / mock
│   ├── eval_retrieval.py            # 召回评估(总体 + 分层 + 逐 query + 性能)
│   ├── eval_rerank.py               # 重排评估(前后对比 + Δ + 冻结候选集)
│   └── run_evaluation.py            # 命令行入口(标准结果目录输出)
├── scripts/
│   ├── download_models.py           # 模型下载(ModelScope / HF 镜像)
│   ├── serve_vllm_embedding.sh      # vLLM 启动 embedding 服务
│   ├── serve_vllm_reranker.sh       # vLLM 启动 reranker 服务(含 hf_overrides 转换)
│   ├── compare_runs.py              # 两个 run 的配对 bootstrap 置信区间对比
│   └── demo_mock.py                 # 无 GPU 流程演示(mock 后端)
└── tests/
    ├── test_metrics.py              # 指标单元测试(手工核算值)
    └── test_pipeline.py             # 管线冒烟测试(SM-02/08/09 等)
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 如需 sentence-transformers 本地推理后端:
pip install sentence-transformers
```

### 2. 无 GPU 验证流程(mock 后端 + 内置半导体示例集)

```bash
python scripts/demo_mock.py        # 召回 + 重排全流程
python -m unittest discover tests  # 单元测试 + 冒烟测试
```

### 3. BM25 基线(最低组合清单第 7 项)

```bash
python -m src.run_evaluation --backend bm25 --output-dir results/bm25-baseline
```

### 4. 内网真实评测(以 vLLM 为例)

```bash
# 步骤一:启动服务(详见 docs/06)
bash scripts/serve_vllm_embedding.sh /path/to/Qwen3-Embedding-8B 8000
bash scripts/serve_vllm_reranker.sh /path/to/Qwen3-Reranker-8B 8001

# 步骤二:召回评估,冻结候选集
python -m src.run_evaluation \
  --backend openai --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-Embedding-8B \
  --data-dir data/sample --output-dir results/qwen3-8b-r1 --save-candidates

# 步骤三:用冻结候选集公平对比各 reranker(总框架铁律)
python -m src.run_evaluation \
  --backend openai --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-Embedding-8B \
  --rerank-backend openai --rerank-base-url http://127.0.0.1:8001 --rerank-model Qwen/Qwen3-Reranker-8B \
  --data-dir data/sample --candidates results/qwen3-8b-r1/candidates.jsonl \
  --output-dir results/qwen3-8b-rr8b-r1

# 步骤四:两个 run 的显著性对比
python scripts/compare_runs.py results/qwen3-8b-r1/per_query_results.jsonl \
  results/qwen3-8b-rr8b-r1/per_query_results.jsonl --metric ndcg@10
```

### 5. 用自己的知识库数据评测

按 `data/sample/` 的格式准备 `corpus.jsonl` / `queries.jsonl` / `qrels.tsv`(也支持总框架的 JSONL 风格,见 [docs/07](docs/07-测试集构建与标注指南.md) §4),把 `--data-dir` 指向你的目录即可。

## 文档索引

| 文档 | 内容 |
|---|---|
| [00-评估体系总框架](docs/00-评估体系总框架.md) | **总纲**:L1/L2/L3 三层、14 能力维度、候选池、评分与验收规则、复现与报告规范 |
| [01-模型调研-Embedding](docs/01-模型调研-Embedding.md) | Qwen3-Embedding 架构/训练/用法,BGE 全系,竞品 |
| [02-模型调研-Reranker](docs/02-模型调研-Reranker.md) | Qwen3-Reranker 原理与模板,BGE reranker,召回+重排管线 |
| [03-模型参数对比](docs/03-模型参数对比.md) | 参数量/维度/MRL/上下文/语言/协议/显存 全表 |
| [04-评估指标体系](docs/04-评估指标体系.md) | 基准介绍,指标严格定义,验收门槛,统计可信度 |
| [05-官方Benchmark结论](docs/05-官方Benchmark结论.md) | 各 benchmark 完整分数表与结论分析 |
| [06-内网部署指南](docs/06-内网部署指南.md) | 离线下载、五种部署方案、NPU/CPU 备选、上线检查清单 |
| [07-测试集构建与标注指南](docs/07-测试集构建与标注指南.md) | 规模建议、13 类分层、标注标准、文件格式、冒烟用例 |
| [08-实验流程与报告模板](docs/08-实验流程与报告模板.md) | 最低组合清单、评分权重、复现准则、Phase 0-5、一页决策表 |

## 说明

- 所有 benchmark 数字均来自官方模型卡 / 官方博客 / 原始论文,来源 URL 见各文档末尾;**官方分数是参考线,不是内网保证值**。
- `mock` 后端(词袋哈希向量)仅用于无 GPU 环境验证评测流程,**不代表任何真实模型效果**。
- 本项目调研时间为 2026 年 10 月;榜单分数随时间变化,复评时请以 [MTEB 官方榜单](https://huggingface.co/spaces/mteb/leaderboard) 最新数据为准。
