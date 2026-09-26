# Lecture 12 — LLM 工程与 MLOps

> 把 ChatGPT 这类大语言模型"被制造出来的全过程"做成微缩模型，亲手复现一遍。
> 从打造分词器，到预训练、SFT、DPO、量化、服务与监控 —— 这是整套课程的顶点，
> 也是让你从"会用 LLM 的人"升级为"理解 LLM 是怎么造出来的人"的一门课。

## 这门课你将学到

- 大语言模型 (LLM) 像工厂流水线出产品一样被造出来的完整管线: 预训练 → 退火 → SFT → 偏好对齐 → 量化 → 部署
- BPE 分词器、数据打包 (packing)、去重与污染检查等"数据工厂"里的真实工序
- 亲手训练一个 2 层的迷你 GPT，见证"只靠预测下一个 token 就能长出语法"的时刻
- 用缩放法则 (scaling laws) 估算"训练这个模型要花多少钱"的量级直觉
- 不借助任何外部库，从零实现 LoRA、SFT、DPO，彻底吃透原理
- int8 量化、HTTP 推理服务器、漂移监控 —— 模型变成一项线上服务并持续运营的全过程
- 非科班出身者成长为 AI 人才的现实职业路线图 (整套课程的收官)

真实的大规模训练动辄数十亿到数万亿 token、数千张 GPU；我们把每个阶段
都缩小成**几万到几十万参数的迷你模型**来复现。所有关卡都设计成在笔记本电脑的
CPU 上 90 秒以内训练完毕。全程不使用 transformers 之类的库，一切亲手实现，
所以 LLM 对你来说将不再是黑箱，而是一台"摸得着的机器"。

## 先修课程

- **lecture08 — 深度学习基础** (需要能读懂张量、梯度下降、训练循环)
- **lecture10 — 自然语言处理 (NLP)** (token、嵌入、注意力的概念)
- **lecture11 — LLM 应用与 RAG** (有提示词和 LLM API 的使用经验，"制造方"的故事会生动得多)

像 level00、05、11 这种概念重于代码的关卡，即使先修知识不足也可以直接阅读。

## 关卡目录

| 关卡 | 标题 | 难度 |
|---|---|---|
| [level00](level00_the_big_picture/README.md) | 造一个模型意味着什么 — 全景地图 | ⭐ |
| [level01](level01_bpe_tokenizer/README.md) | 打造分词器 — 亲手实现 BPE | ⭐⭐⭐ |
| [level02](level02_data_pipeline_packing/README.md) | 训练数据流水线与打包 (Packing) | ⭐⭐⭐ |
| [level03](level03_data_quality_contamination/README.md) | 数据质量、去重与污染检查 | ⭐⭐⭐⭐ |
| [level04](level04_mini_gpt_pretraining/README.md) | 迷你 GPT 预训练 | ⭐⭐⭐⭐⭐ |
| [level05](level05_scaling_laws_costs/README.md) | 缩放法则与训练成本估算 | ⭐⭐⭐ |
| [level06](level06_finetuning_lora/README.md) | 微调与 LoRA | ⭐⭐⭐⭐ |
| [level07](level07_sft/README.md) | 指令微调 (SFT) | ⭐⭐⭐⭐ |
| [level08](level08_dpo/README.md) | 偏好对齐 (DPO) | ⭐⭐⭐⭐⭐ |
| [level09](level09_quantization/README.md) | 量化与轻量化 | ⭐⭐⭐⭐ |
| [level10](level10_serving_deployment/README.md) | 模型服务与部署 | ⭐⭐⭐⭐ |
| [level11](level11_monitoring_governance_career/README.md) | 监控、成本、治理与职业路线图 | ⭐⭐⭐ |

## 快速路线 (时间紧就只看这 5 个)

1. **level00** — 全景地图: 先从大局看清模型经过哪些阶段才诞生
2. **level04** — 迷你 GPT 预训练: 这门课的心脏。亲手做的模型学会造句的那一刻
3. **level05** — 缩放法则: 用数字回答"造一个 GPT 要花多少钱"
4. **level07** — SFT: 让"续写机器"变身"助理"的决定性一步
5. **level09** — 量化: 为什么公司发的笔记本电脑也能跑 LLM

## 这门课在实际工作中的用武之地

- **AI 引入决策会议**: 用分阶段的成本直觉判断"我们该自研、微调，还是直接调 API" (level00, 05)
- **与外部厂商、AI 团队协作**: 听懂"我们用 LoRA 做了微调，以 int4 上线服务"这样的汇报，并提出正确的问题 (level06, 09)
- **用公司数据定制模型**: 为"教模型学会本公司文档与口吻"的微调项目做数据准备与质量管理 (level02, 03, 07)
- **AI 服务运营**: 用仪表盘监控响应延迟、成本、质量漂移，及早发现异常 (level10, 11)
- **安全与合规审查**: 排查训练数据污染、日志里的个人信息、API 密钥管理等风险点 (level03, 11)
- **招聘与转型**: 用自己的话解释 LLM 工程岗位招聘启事里的术语 (预训练、RLHF/DPO、量化、服务) (全部关卡)

## 学习建议

- 建议按关卡顺序推进。这门课被设计成一次完整的"模型工厂参观"路线:
  level01~03 备料 (数据)，level04 烧制模型，level06~08 精修打磨，
  level09~11 打包、出厂、售后。
- 每个关卡都应边运行 `main.py` 边阅读。运行方式: 使用仓库根目录的虚拟环境执行
  `python3 main.py` (环境配置见根目录 `SETUP.md`)
- 训练数据用的是 `common/hjh_data.py` 里的迷你中文语料，全程不需要联网。
- 迷你模型生成的句子偏生硬是正常现象。重点是观察"它在一步步变好"的过程，
  而不是追求 ChatGPT 级别的质量。它与真实模型的差距，各关卡的笔记里都会用数字讲清楚。
