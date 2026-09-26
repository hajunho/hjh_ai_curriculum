# Lecture 10 — 自然语言处理 (NLP)

> 从零开始搭建计算机处理人类语言的整套技术。
> 从字符串清洗和正则表达式出发，经过 TF-IDF、情感分析、词嵌入，
> 亲手实现注意力机制和迷你 Transformer，一路抵达 GPT/BERT 的预训练范式。

## 这门课学什么

- 为什么语言对计算机来说格外难 — 歧义、上下文，以及"中文没有空格"这一特性
- 把脏乱的文本变成可分析形态的预处理流水线与正则表达式
- 把文本变成数字的三个世代: BoW/TF-IDF → 词嵌入 → 上下文嵌入
- 亲手做一个评论情感分析器，并解读模型是靠哪些词做出判断的
- 用 numpy 和 torch 把 RNN、注意力、Transformer 逐个零件拼装起来
- GPT (续写) 与 BERT (完形填空) 改变世界的预训练范式的核心

这门课**故意不使用** jieba、nltk、transformers 之类的 NLP 专用包。
只有亲手实现一遍分词器、TF-IDF、注意力，你才知道"库在里面到底干了什么"，
到下一门课 (lecture11 — LLM 应用与 RAG) 使用现成工具时，
它们对你就不再是黑箱，而是玻璃箱。

## 先修课程

- **lecture02 — Python 编程基础** (字符串、列表、字典、函数)
- **lecture03 — 数据处理** (NumPy 数组运算)
- **lecture06 — 机器学习入门** (分类、逻辑回归、训练/评估划分)
- **lecture08 — 深度学习基础** (level07 之后的 torch 实战需要；level00~06 不需要)

## 关卡目录

| 关卡 | 标题 | 难度 |
|---|---|---|
| [level00](level00_why_language_is_hard/README.md) | 计算机处理语言为什么难 | ⭐ |
| [level01](level01_text_preprocessing/README.md) | 文本预处理基础 | ⭐⭐ |
| [level02](level02_regex/README.md) | 正则表达式 | ⭐⭐ |
| [level03](level03_tokenization_korean/README.md) | 分词与中文的特殊性 | ⭐⭐⭐ |
| [level04](level04_bow_tfidf/README.md) | BoW 与 TF-IDF | ⭐⭐⭐ |
| [level05](level05_text_classification/README.md) | 文本分类实战 — 评论情感分析 | ⭐⭐⭐ |
| [level06](level06_word_embeddings/README.md) | 词嵌入 | ⭐⭐⭐ |
| [level07](level07_rnn_sequences/README.md) | RNN 与序列模型 | ⭐⭐⭐⭐ |
| [level08](level08_attention/README.md) | 注意力机制 | ⭐⭐⭐⭐ |
| [level09](level09_transformer_anatomy/README.md) | 解剖 Transformer 结构 | ⭐⭐⭐⭐ |
| [level10](level10_mini_transformer/README.md) | 亲手实现迷你 Transformer | ⭐⭐⭐⭐⭐ |
| [level11](level11_pretraining_paradigm/README.md) | 预训练范式 — BERT 与 GPT | ⭐⭐⭐⭐ |

## 快速路线 (没时间就只学这 5 个)

1. **level01 文本预处理** — 所有文本工作的起点。实务使用频率第一名。
2. **level04 BoW·TF-IDF** — "文本变数字"的标准做法。搜索、分类、关键词提取的地基。
3. **level05 文本分类** — 一次跑通评论情感分析的完整流水线。
4. **level08 注意力机制** — 如果只能选一把理解当代 AI 的钥匙，就是它。
5. **level11 预训练范式** — GPT/BERT 为什么了不起，LLM 时代的大局观。

## 这门课在实际工作中的用武之地

- **VOC (客户之声) 分析**: 把每天堆积上千条的评论、咨询自动分成好评/差评，
  提取投诉关键词，做成周报。(level01, 04, 05)
- **从文档里提取信息**: 从一摞合同、报价单中用正则表达式批量提取金额、日期、
  联系方式，整理进 Excel。(level02)
- **公司内部文档搜索与摘要的地基**: 回答"休假制度写在哪儿来着？"的内部搜索
  和 RAG 聊天机器人，全都建立在这门课的分词、TF-IDF、嵌入之上。(level03, 04, 06)
- **用好 LLM 的眼光**: 按 token 计费为什么是那样、上下文长度限制为什么存在、
  提示词为什么那样起作用 — 懂了 Transformer 结构就全能解释。(level08~11)

## 运行方法

```bash
cd lecture10_nlp_text/level00_why_language_is_hard
python3 main.py
```

所有实战练习都无需联网即可运行，数据由 `common/hjh_data.py` 现场生成。
用到 torch 的关卡 (07, 10, 11) 也设计得非常小，CPU 上 90 秒内就能跑完。
