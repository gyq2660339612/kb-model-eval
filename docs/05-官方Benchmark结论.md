# 05 · 官方 Benchmark 结论(完整数据)

> 数据来源:Qwen3 官方博客/GitHub/技术报告(arXiv:2506.05176)、BGE 官方模型卡与论文(arXiv:2402.03216)。Qwen3 对比分数取自 2025-05-24~06-06 的 MTEB 在线榜单快照。调研时间 2026-10。
>
> **重要声明**:公开榜单分数受版本、数据集、检索候选集、提示词、截断长度、硬件与实现影响。本文将"官方公开 benchmark 结果"与"内网实测结果"严格分开——**官方分数是参考线,不是内网保证值**,不允许把条件不同的分数直接当作严格的横向结论(总框架 §5)。

## 1. Embedding 官方数据

### 1.1 MTEB Multilingual(131 任务,Mean Task)

| 模型 | Mean(Task) | Mean(Type) |
|---|---|---|
| **Qwen3-Embedding-8B** | **70.58** | 61.69 |
| Qwen3-Embedding-4B | 69.45 | 60.86 |
| Gemini Embedding(exp-03-07) | 68.37 | 59.59 |
| Qwen3-Embedding-0.6B | 64.33 | 56.00 |
| multilingual-e5-large-instruct | 63.22 | 55.08 |
| gte-Qwen2-7B-Instruct | 62.51 | 55.93 |
| Cohere-embed-multilingual-v3.0 | 61.12 | 53.23 |
| GritLM-7B | 60.92 | 53.74 |
| **BGE-M3** | **59.56** | 52.18 |
| text-embedding-3-large | 58.93 | 51.41 |
| NV-Embed-v2 | 56.29 | 49.58 |

8B 分类型细分:Bitext Mining 80.89、Classification 74.00、Clustering 57.65、Rerank 65.63、Retrieval 70.88、STS 81.08、Pair Class 86.40。

### 1.2 MTEB (Eng, v2)

| 模型 | Mean(Task) | Mean(Type) |
|---|---|---|
| **Qwen3-Embedding-8B** | **75.22** | 68.71 |
| Qwen3-Embedding-4B | 74.60 | 68.10 |
| gemini-embedding-exp-03-07 | 73.30 | 67.67 |
| gte-Qwen2-7B-instruct | 70.72 | 65.77 |
| Qwen3-Embedding-0.6B | 70.70 | 64.88 |
| NV-Embed-v2 | 69.81 | 65.00 |

### 1.3 C-MTEB(中文,Mean Task)

| 模型 | Mean(Task) | Mean(Type) |
|---|---|---|
| **Qwen3-Embedding-8B** | **73.84** | 75.00 |
| ritrieve_zh_v1 (0.3B) | 72.71 | 73.85 |
| Qwen3-Embedding-4B | 72.27 | 73.51 |
| gte-Qwen2-7B-instruct | 71.62 | 72.19 |
| bge-multilingual-gemma2 | 67.64 | 68.52 |
| Qwen3-Embedding-0.6B | 66.33 | 67.45 |
| multilingual-e5-large-instruct | 58.08 | 58.24 |

(参照:bge-large-zh-v1.5 的 C-MTEB 平均为 64.53、Retrieval 70.46。)

### 1.4 官方模型卡 Retrieval / Rerank 分列(聚合口径)

模型卡除 Mean(Task) 外还给出检索/重排聚合列。**注意:这是官方 benchmark 套件内的聚合分数,不能误读成"企业知识库的 Recall@10"。**

MTEB Multilingual:

| 模型 | Mean(Task) | Retrieval | Rerank |
|---|---:|---:|---:|
| BGE-M3 | 59.56 | 54.60 | 62.79 |
| multilingual-e5-large-instruct | 63.22 | 57.12 | 62.61 |
| Qwen3-Embedding-0.6B | 64.33 | 64.64 | 61.41 |
| Qwen3-Embedding-4B | 69.45 | 69.60 | 65.08 |
| Qwen3-Embedding-8B | 70.58 | 70.88 | 65.63 |

MTEB (Eng, v2):

| 模型 | Mean(Task) | Retrieval | Rerank |
|---|---:|---:|---:|
| Qwen3-Embedding-0.6B | 70.70 | 61.83 | 48.18 |
| Qwen3-Embedding-4B | 74.60 | 68.46 | 50.76 |
| Qwen3-Embedding-8B | 75.22 | 69.44 | 51.56 |

C-MTEB:

| 模型 | Mean(Task) | Retrieval | Rerank |
|---|---:|---:|---:|
| Qwen3-Embedding-0.6B | 66.33 | 71.03 | 62.58 |
| Qwen3-Embedding-4B | 72.27 | 77.03 | 66.05 |
| Qwen3-Embedding-8B | 73.84 | 78.21 | 66.99 |

解读:公开口径下 8B 的中文 Retrieval 比 0.6B 高 7.18 分(78.21−71.03)——这是聚合分数差,**不代表内部检索 Recall 会提升相同比例**,内部增益必须实测。

### 1.5 MTEB-Code(nDCG@10)

| 模型 | 得分 |
|---|---|
| **Qwen3-Embedding-8B** | **80.68** |
| Qwen3-Embedding-4B | 80.06 |
| Qwen3-Embedding-0.6B | 75.41 |
| Gemini Embedding | 74.66 |
| GritLM-7B | 73.60 |
| multilingual-e5-large-instruct | 65.00 |
| text-embedding-3-large | 58.95 |
| gte-Qwen2-7B-instruct | 56.41 |

### 1.6 BGE 官方数据(各自论文/模型卡口径)

| 模型 | 基准 | 得分 |
|---|---|---|
| BGE-M3 | MIRACL nDCG@10 平均 | Dense 69.2 / Sparse 53.9 / Multi-vec 70.5 / **All 71.5**(mE5-large 66.6、BM25 31.9) |
| BGE-M3 | MKQA 跨语 Recall@100 | Dense 75.1 / All 75.5 |
| BGE-M3 | MLDR 长文档 nDCG@10 | Dense 52.5 / All 65.0(E5-mistral-7b 42.6) |
| BGE-M3 | NarrativeQA nDCG@10 | 61.7(text-embedding-3-large 51.6) |
| bge-large-en-v1.5 | MTEB 平均(56) | 64.23(Retrieval 54.29) |
| bge-large-zh-v1.5 | C-MTEB 平均 | 64.53(Retrieval 70.46、Reranking 65.84) |
| bge-multilingual-gemma2 | MIRACL / MTEB / C-MTEB | 74.1 / 69.88 / 68.44 |
| bge-code-v1 | CoIR / CodeRAG | 81.77 / 72.8 |

## 2. Reranker 官方数据

### 2.1 Qwen3-Reranker(召回基线 = Qwen3-Embedding-0.6B top-100)

| 模型 | MTEB-R | CMTEB-R | MMTEB-R | MLDR | MTEB-Code | FollowIR |
|---|---|---|---|---|---|---|
| 基线(无 rerank) | 61.82 | 71.02 | 64.64 | 50.26 | 75.41 | 5.09 |
| Jina-multilingual-reranker-v2-base (0.3B) | 58.22 | 63.37 | 63.73 | 39.66 | 58.98 | -0.68 |
| gte-multilingual-reranker-base (0.3B) | 59.51 | 74.08 | 59.44 | 66.33 | 54.18 | -1.64 |
| BGE-reranker-v2-m3 (0.6B) | 57.03 | 72.16 | 58.36 | 59.51 | 41.38 | -0.01 |
| Qwen3-Reranker-0.6B | 65.80 | 71.31 | 66.36 | 67.28 | 73.42 | 5.41 |
| Qwen3-Reranker-4B | **69.76** | 75.94 | 72.74 | 69.97 | 81.20 | **14.84** |
| **Qwen3-Reranker-8B** | 69.02 | **77.45** | **72.94** | **70.19** | **81.22** | 8.05 |

> 此表使用**固定候选集**(0.6B 召回 top-100)——这正是"评 reranker 必须固定候选集"的范例;其结论不能直接等价为"换上 8B 后生产链路必然提高相同分数"。

### 2.2 BGE reranker 官方数据(重排 top-100)

| 模型 | BEIR nDCG@10 | MIRACL nDCG@10 | C-MTEB Retrieval |
|---|---|---|---|
| (基线) | 54.31 | 67.91 | 70.46 |
| bge-reranker-v2-m3 | 55.36 | 72.84 | 71.82 |
| bge-reranker-v2-gemma | 60.71 | 73.39 | — |
| bge-reranker-v2-minicpm-28 | — | — | 73.51 |

### 2.3 召回 → 重排端到端提升(Qwen3 官方)

| 管线 | CMTEB-R | MLDR | MTEB-Code |
|---|---|---|---|
| Embedding-0.6B 召回 | 71.02 | 50.26 | 75.41 |
| + Reranker-4B | 75.94 | 69.97 | 81.20 |
| + Reranker-8B | **77.45** | **70.19** | **81.22** |

## 3. 2026 年榜单动态(参考)

- **Microsoft Harrier-OSS-v1**(2026-04,MIT):27B 版以 74.27 居 MTEB 多语言 v2 第 1(2026-04-06 口径);0.6B 版 69.01。
- gemini-embedding-001(GA):MTEB 多语言 68.32;2048 上下文;MRL 最大 3072。
- 腾讯 KaLM-Embedding 多语言 v2 第 2(72.32)、Conan-embedding-v2 英文 v1 第 1(74.22)(二级来源,未官方核实)。
- MTEB 榜单 2025 年起迁移至集中式 results 仓库统一跑分;**eng v1 与 v2、Multilingual v1 与 v2 之间分数不可直接比较**。

## 4. 结论

1. **开源可商用模型中,Qwen3-Embedding-8B 是中文/多语言知识库的一线候选**:多语言 70.58(发布时第 1)、中文 73.84、代码 80.68,32K 上下文、MRL 降维、Apache-2.0。**但它是"高精度候选"而非"预设胜者"**——是否采用取决于内部测试集收益与部署预算(总框架 §15)。
2. **Qwen3-Reranker-8B 中文最强**(CMTEB-R 77.45);**4B 在 MTEB-R 与 FollowIR 上反超 8B**——"最大参数必然最佳"是错误假设,4B 必须进入对照。
3. **0.6B 档全面碾压 BGE 同尺寸**:Embedding 64.33 vs 59.56;Reranker MTEB-R 65.80 vs 57.03。0.6B 是不可缺少的成本基线——若已满足内部验收线,应认真考虑其运营成本优势。
4. BGE-M3 的重点是**多功能召回**(dense/sparse/multi-vector),只测 dense 不代表其完整能力;混合检索(关键词/型号敏感)场景 sparse 通道仍有工程价值。
5. 公开榜单分数≠领域效果。**最终决策以"内部半导体测试集 + 固定候选重排评测 + 目标硬件压测"为准**(流程见 docs/08)。
6. **最需要防范的评测陷阱**:只看 Mean(Task)、候选集不一致、有无指令混测、截断方式不一致、相关文档池不完整、对泄漏文档过拟合、不同硬件比吞吐、用 reranker 原始 logit 跨模型作概率解释。

## 来源

1. https://qwenlm.github.io/blog/qwen3-embedding/
2. https://github.com/QwenLM/Qwen3-Embedding
3. https://arxiv.org/abs/2506.05176
4. https://arxiv.org/abs/2402.03216
5. https://github.com/FlagOpen/FlagEmbedding
6. https://huggingface.co/BAAI/bge-m3 等模型卡
7. https://blogs.bing.com/search/2026/4/Microsoft-Open-Sources-Industry-Leading-Embedding-Model/
8. https://huggingface.co/spaces/mteb/leaderboard
