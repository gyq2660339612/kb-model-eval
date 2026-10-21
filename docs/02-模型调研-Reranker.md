# 02 · Reranker 模型调研

> 调研时间:2026-10。数字均来自官方来源,URL 见文末。

## 1. 为什么需要 Reranker

Embedding(双塔)把 query 和 document 独立编码成向量,检索快但交互信息少;Reranker(交叉编码 cross-encoder)把 query+document **拼接后联合编码**,直接输出相关性分数,精度显著更高但无法预建索引、只能对召回的 top-k(典型 top-100)逐一打分。生产标准做法是**两段式管线**:

```
query → Embedding 召回 top-100(快)→ Reranker 精排取 top-5/10(准)
```

## 2. Qwen3-Reranker 系列(本项目主力)

### 2.1 工作原理

基于 Qwen3 因果语言模型的 **point-wise 交叉编码器**,把相关性判断建模为 yes/no 二分类:

```
score(q, d) = exp(P(yes | I, q, d)) / ( exp(P(yes|·)) + exp(P(no|·)) )
```

即取最后一个 token 位置处 "yes" / "no" 两个 token 的 logits,log-softmax 后 "yes" 的概率为相关性得分(0–1)。

**官方 prompt 模板(逐字,重要)**:

- system 前缀:`<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n`
- body:`<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}`
- suffix:`<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`(即关闭 thinking 模式)

默认 instruction:`Given a web search query, retrieve relevant passages that answer the query`,可按业务自定义(官方实测提升 1%–5%,多语言场景建议英文指令)。

**训练**:无弱监督预训练,直接高质量标注数据 SFT(正例 "yes"、负例 "no")+ slerp 模型合并;LoRA 微调保留基座能力。

### 2.2 规格

| 模型 | 参数量 | 层数 | 上下文 | 指令感知 |
|---|---|---|---|---|
| Qwen3-Reranker-0.6B | 0.6B | 28 | 32K | 是 |
| Qwen3-Reranker-4B | 4B | 36 | 32K | 是 |
| Qwen3-Reranker-8B | 8B | 36 | 32K | 是 |

支持 100+ 语言;**Apache 2.0**。Reranker 无向量维度/MRL 概念。

### 2.3 官方 benchmark(召回基线 = Qwen3-Embedding-0.6B 召回 top-100)

| 模型 | MTEB-R | CMTEB-R | MMTEB-R | MLDR | MTEB-Code | FollowIR |
|---|---|---|---|---|---|---|
| (基线,无 rerank) | 61.82 | 71.02 | 64.64 | 50.26 | 75.41 | 5.09 |
| Jina-multilingual-reranker-v2-base (0.3B) | 58.22 | 63.37 | 63.73 | 39.66 | 58.98 | -0.68 |
| gte-multilingual-reranker-base (0.3B) | 59.51 | 74.08 | 59.44 | 66.33 | 54.18 | -1.64 |
| BGE-reranker-v2-m3 (0.6B) | 57.03 | 72.16 | 58.36 | 59.51 | 41.38 | -0.01 |
| **Qwen3-Reranker-0.6B** | 65.80 | 71.31 | 66.36 | 67.28 | 73.42 | 5.41 |
| **Qwen3-Reranker-4B** | **69.76** | 75.94 | 72.74 | 69.97 | 81.20 | **14.84** |
| **Qwen3-Reranker-8B** | 69.02 | **77.45** | **72.94** | **70.19** | **81.22** | 8.05 |

结论:

- 0.6B 已全面超越同尺寸 BGE-reranker-v2-m3(MTEB-R 65.80 vs 57.03);
- **8B 并非全面最优**:MTEB-R 与 FollowIR 上 4B 反超 8B;中文(CMTEB-R 77.45)、多语言、长文档(MLDR 70.19)、代码(81.22)上 8B 最强——**中文知识库选 8B 合理**;
- 端到端提升显著:基线→+8B,CMTEB-R 71.02→77.45,MLDR 50.26→70.19(+19.93),MTEB-Code 75.41→81.22。

### 2.4 调用要点

- **transformers**:`AutoModelForCausalLM` 加载;`logits[:, -1, :]` 取 yes/no token id 两维做 log_softmax 后 exp;`padding_side='left'`;transformers>=4.51.0。
- **sentence-transformers**:`CrossEncoder("Qwen/Qwen3-Reranker-0.6B")`;`predict` 输出原始 logit 差值,加 `activation_fn=torch.nn.Sigmoid()` 得 0–1 分。
- **vLLM 关键坑点**:原始模型是 CausalLM,需对全部 151,669 词表算 logits,低效。官方两种解法:
  - 启动时加 `--hf_overrides '{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'` + 官方 jinja chat 模板;
  - 或先用官方脚本 `convert_model_to_seq_cls.py` 转成序列分类模型(社区现成版 `tomaarsen/Qwen3-Reranker-*-seq-cls`),直接支持 `/score`、`/rerank`。
- **SGLang**:部署 Qwen3-Reranker **不能加 `--is-embedding`**,否则无法计算 yes/no logprob。

## 3. BGE Reranker(对照)

### 3.1 BAAI/bge-reranker-v2-m3(2024-03)

- 基于 bge-m3 微调的 cross-encoder,约 **568M** 参数,多语言,**Apache-2.0**。
- 原理:`AutoModelForSequenceClassification`(num_labels=1),query+passage 拼接全注意力交互,直接输出**单个 logit**;分数无固定范围,可 sigmoid 映射到 [0,1]。官方评测推理长度 1024。
- 官方分数(重排 top-100):BEIR nDCG@10 平均 55.36(基线 54.31);MIRACL 72.84(基线 67.91);C-MTEB Retrieval 71.82(基线 70.46)。
- 用法:`FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True).compute_score([q, d])`;`normalize=True` 时输出 sigmoid 分。

### 3.2 其他 BGE reranker

- bge-reranker-v2-gemma(BEIR 60.71,更强但 Gemma 协议)、bge-reranker-v2-minicpm-layerwise(支持逐层早停加速,C-MTEB 73.51)等;多为研究版,生产常用仍是 v2-m3。

## 4. 管线组合建议

| 预算 | 组合 | 说明 |
|---|---|---|
| 最佳质量(中文知识库) | Qwen3-Embedding-8B 召回 top-100 → **Qwen3-Reranker-8B** 精排 | CMTEB-R 77.45,MLDR 70.19 |
| 均衡 | Qwen3-Embedding-4B → Qwen3-Reranker-4B | 单张 24GB 可同卡部署两者 |
| 低成本 | Qwen3-Embedding-0.6B → Qwen3-Reranker-0.6B | 已全面超过 BGE 同尺寸组合 |
| 混合检索 | BGE-M3(dense+sparse)→ bge-reranker-v2-m3 | 关键词敏感场景(型号、编号)有优势 |

## 来源

1. Qwen 官方博客:https://qwenlm.github.io/blog/qwen3-embedding/
2. GitHub:https://github.com/QwenLM/Qwen3-Embedding
3. 技术报告:https://arxiv.org/html/2506.05176v1
4. HF 模型卡:https://huggingface.co/Qwen/Qwen3-Reranker-0.6B(4B/8B 同路径)
5. vLLM 官方文档:https://docs.vllm.ai/en/v0.10.0/examples/offline_inference/qwen3_reranker.html
6. BGE reranker 模型卡:https://huggingface.co/BAAI/bge-reranker-v2-m3
7. FlagEmbedding reranker 评估:https://github.com/FlagOpen/FlagEmbedding/blob/master/research/llm_reranker/README.md
