# 01 · Embedding 模型调研

> 调研时间:2026-10。所有关键数字均来自官方模型卡 / 官方博客 / 原始论文,来源见文末。

## 1. Qwen3-Embedding 系列(本项目主力)

### 1.1 基本信息

- **发布**:2025-06-05,阿里云 Qwen 团队;技术报告 arXiv:2506.05176;**Apache 2.0** 开源(可商用)。
- **基座**:Qwen3 稠密基座模型(`Qwen3-0.6B/4B/8B-Base`),通过 **LoRA 微调**保留基座能力。
- **架构**:双塔(dual-encoder)。单文本独立编码,取向量方式为 **last token pooling**——输入末尾追加 `[EOS]`,取最后一层该位置的隐状态,之后 **L2 归一化**。
- **训练三阶段**:
  1. 用 Qwen3-32B 合成的约 **1.5 亿对**弱监督数据做对比预训练(改进 InfoNCE);
  2. 筛选约 **1200 万对**高质量数据(余弦相似度 >0.7 过滤)做监督微调;
  3. slerp 模型合并。

### 1.2 三个尺寸规格

| 项目 | Qwen3-Embedding-0.6B | Qwen3-Embedding-4B | Qwen3-Embedding-8B |
|---|---|---|---|
| 实际参数量(safetensors) | 595,776,512 | 4,021,774,336 | 7,567,295,488 |
| 层数 | 28 | 36 | 36 |
| hidden size | 1024 | 2560 | 4096 |
| 默认向量维度 | 1024 | 2560 | 4096 |
| MRL 可变维度 | 32–1024 自定义 | 32–2560 自定义 | 32–4096 自定义 |
| 最大上下文 | 32K | 32K | 32K |
| 语言 | 100+(含编程语言) | 同左 | 同左 |
| 指令感知 | 是(仅 query 侧) | 是 | 是 |

要点:

- **MRL(Matryoshka Representation Learning)**:向量可按需截断到任意 ≥32 的维度,精度近似无损地随维度降低而平缓下降。内网存储/检索成本敏感时,8B 模型也可以只存 1024 维。
- **指令感知**:query 侧拼接指令模板 `Instruct: {任务描述}\nQuery: {query}`,**document 侧不加任何前缀**。官方实测加指令提升约 **1%–5%**;多语言场景指令建议用英文书写。

### 1.3 官方 benchmark(摘要,完整数据见 docs/05)

| Benchmark | 0.6B | 4B | 8B | 关键对照 |
|---|---|---|---|---|
| MTEB Multilingual (Mean Task) | 64.33 | 69.45 | **70.58** | Gemini Embedding 68.37,BGE-M3 59.56 |
| MTEB (Eng, v2) | 70.70 | 74.60 | **75.22** | gemini-embedding-exp 73.30 |
| C-MTEB (中文) | 66.33 | 72.27 | **73.84** | gte-Qwen2-7B 71.62,mE5-large-instruct 58.08 |
| MTEB-Code (nDCG@10) | 75.41 | 80.06 | **80.68** | Gemini 74.66 |

发布时(2025-06-05 榜单)MTEB 多语言第 1。0.6B 已全面超越同尺寸 BGE-M3 与 multilingual-e5-large-instruct。

### 1.4 调用方式要点

- **transformers**:需 `transformers>=4.51.0`;`padding_side='left'`;建议 `attn_implementation="flash_attention_2"` + bf16。
- **sentence-transformers**:需 `>=2.7.0`;query 用 `model.encode(queries, prompt_name="query")`,文档直接 `encode`。
- **vLLM**:需 `vllm>=0.8.5`;`LLM(model=..., task="embed")` 或 `vllm serve --runner pooling`,OpenAI 兼容 `/v1/embeddings`,支持 `dimensions` 参数做 MRL 截断。
- **TEI**:`text-embeddings-inference >= 1.7.2` 官方支持 Qwen3-Embedding 全系。

## 2. BGE 系列(智源 FlagEmbedding,对照/轻量备选)

### 2.1 BAAI/bge-m3(2024-01,多语言旗舰)

- 参数量约 **569M**(XLM-RoBERTa 架构,hidden 1024,24 层);向量维度 **1024**;上下文 **8192**。
- 支持 **100+ 语言**(训练覆盖 170+)。
- 一模型三检索:**dense**([CLS] 归一化向量)+ **sparse**(类 BM25 词权重)+ **multi-vector**(ColBERT 式),支持加权混合 `s = w1·dense + w2·sparse + w3·colbert`。
- **不需要指令前缀**;协议 **MIT**(可商用)。
- 官方分数:MIRACL nDCG@10 平均 Dense 69.2 / 混合 **71.5**;MLDR 长文档 65.0;NarrativeQA 61.7;MTEB 多语言(第三方口径)59.56。

### 2.2 bge-large-zh-v1.5 / bge-large-en-v1.5(2023-09)

- 维度 1024,上下文 **仅 512 tokens**;zh 约 326M 参数,en 约 335M。
- 检索指令前缀:中文「为这个句子生成表示以用于检索相关文章:」,英文 "Represent this sentence for searching relevant passages:";**passage 永不加前缀**;v1.5 不加指令仅轻微掉点。
- C-MTEB:zh 平均 64.53 / Retrieval 70.46;MTEB:en 平均 64.23。**协议 MIT**。
- 局限:512 上下文对长文档知识库是硬伤,长 chunk 会被截断。

### 2.3 BAAI/bge-multilingual-gemma2(2024-07)

- 基于 google/gemma-2-9b(实际 9.24B 参数);维度 3584(last-token pooling);示例 max_length 4096。
- 指令格式 `<instruct>{任务描述}\n<query>{查询}`,文档不加。
- MIRACL 平均 **74.1**、MTEB 平均 69.88、C-MTEB 68.44。
- **协议为 Gemma 许可**(非 MIT/Apache,商用需审阅 Google Gemma 条款)——内网商用选型需注意。

### 2.4 BAAI/bge-code-v1(2025-05,代码检索)

- Qwen2 架构,约 1.54B 参数;维度 1536;上下文 32K;支持 20 种编程语言 + 中英文查询。
- CoIR 平均 **81.77**,CodeRAG 平均 72.8;**Apache-2.0**。

### 2.5 2025–2026 动态

- **BGE-VL**(2024-12 / 2025-03):多模态图文检索,MIT。
- **BGE-Reasoner**(2025-08/09):推理密集型检索框架;`BGE-Reasoner-Embed-Qwen3-8B-0923` 基于 Qwen3-8B,BRIGHT 原始 query nDCG@10 37.1。
- 未发现 bge-m4 或新一代 M 系列文本模型(截至 2026-10)。

## 3. 其他竞品速览(选型参考)

| 模型 | 规模 | 特点 | 协议 |
|---|---|---|---|
| Gemini Embedding / gemini-embedding-001 | API | MTEB 多语言 68.32–68.37;2048 上下文;MRL 3072 | 闭源 API |
| Gemini Embedding 2(2026) | API | 原生多模态;MTEB-Code 约 84.0(二级来源) | 闭源 API |
| multilingual-e5-large-instruct | 0.6B | MTEB 多语言 63.22;老牌多语言基线 | MIT |
| gte-Qwen2-7B-instruct | 7B | 多语言 62.51 / 中文 71.62 | Apache 2.0 |
| Microsoft Harrier-OSS-v1(2026-04) | 270M/0.6B/27B | 27B 以 74.27 居 MTEB 多语言 v2 第 1(2026-04 口径) | MIT |
| 腾讯 KaLM / Conan 系列 | — | 多语言 v2 第 2(72.32)、英文 v1 第 1(74.22)(二级来源) | 视具体模型 |
| voyage-3-large 等 | API | 未向 MTEB 官方榜单提交分数,仅厂商自测 | 闭源 API |

> 内网离线部署直接排除纯 API 模型;Harrier-OSS 虽新但 27B 部署成本远高于 Qwen3-8B,且中文场景 C-MTEB 数据缺乏,维持 Qwen3 为主的选型不变。

## 4. 选型建议(面向中文为主的企业知识库)

| 场景 | 推荐 |
|---|---|
| 追求最高召回质量(GPU ≥24GB) | **Qwen3-Embedding-8B**(可用 MRL 截到 1024 维省存储) |
| 性价比/单卡 16GB 级 | **Qwen3-Embedding-4B** |
| 高并发低成本 / CPU 边缘节点 | **Qwen3-Embedding-0.6B**(全面优于 BGE-M3) |
| 需要稀疏+稠密混合检索、生态成熟 | BGE-M3(FlagEmbedding 直接出 lexical weights) |
| 代码知识库 | Qwen3-Embedding 全系已很强(MTEB-Code 80.68);或 bge-code-v1 |

## 来源

1. Qwen 官方博客:https://qwenlm.github.io/blog/qwen3-embedding/
2. GitHub:https://github.com/QwenLM/Qwen3-Embedding
3. 技术报告:https://arxiv.org/abs/2506.05176
4. HF 模型卡:https://huggingface.co/Qwen/Qwen3-Embedding-0.6B(4B、8B 同路径)
5. BGE-M3 论文:https://arxiv.org/abs/2402.03216
6. FlagEmbedding:https://github.com/FlagOpen/FlagEmbedding
7. BGE 模型卡:https://huggingface.co/BAAI/bge-m3 、.../bge-large-zh-v1.5 、.../bge-multilingual-gemma2 、.../bge-code-v1
8. 微软 Harrier 博客:https://blogs.bing.com/search/2026/4/Microsoft-Open-Sources-Industry-Leading-Embedding-Model/
9. MTEB 榜单:https://huggingface.co/spaces/mteb/leaderboard
