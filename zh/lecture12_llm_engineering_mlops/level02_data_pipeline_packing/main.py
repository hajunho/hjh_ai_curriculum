"""
把预训练数据流水线做成微缩版。
先把文档用 <s>...</s> 首尾相接 (打包/packing) 成一条长 token 流，
存为 uint16 二进制分片 (shard)，再用 numpy memmap 秒开，
最后切出固定长度 (block_size) 的训练批次 —— 完整跑一遍全流程。
这正是大规模训练处理数 TB 数据方式的缩小版。
"""

import os
import sys
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"

# 特殊 token — 把 id 固定在最前面的编号。
BOS, EOS, PAD, UNK = "<s>", "</s>", "<pad>", "<unk>"
SPECIALS = [BOS, EOS, PAD, UNK]


def make_documents(n_docs: int = 200, sents_per_doc: int = 5):
    """把 tiny_corpus 按句号拆成句子，再 5 句一组拼成"文档"。"""
    corpus = hjh_data.tiny_corpus()
    sents = [s.strip() + "。" for s in corpus.split("。") if s.strip()]
    docs = []
    for i in range(n_docs):
        chunk = sents[i * sents_per_doc:(i + 1) * sents_per_doc]
        docs.append(" ".join(chunk))
    return docs


def build_vocab(docs):
    """字符级 vocab: 4 个特殊 token + 语料中的全部字符。"""
    chars = sorted(set("".join(docs)))
    itos = SPECIALS + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos


def tokenize_and_pack(docs, stoi):
    """把每篇文档包成 <s> tokens </s>，再接成一条长流。"""
    stream = []
    for doc in docs:
        stream.append(stoi[BOS])
        stream.extend(stoi.get(ch, stoi[UNK]) for ch in doc)
        stream.append(stoi[EOS])
    return np.array(stream, dtype=np.uint16)  # vocab < 65536，2 字节足够


def get_batch(mm, block_size: int, batch_size: int, rng):
    """从 memmap 流里随机挑起点，切出 (x, y) 批次。
    y 是 x 右移一格的结果 —— "猜下一个 token" 的标准答案。"""
    starts = rng.integers(0, len(mm) - block_size - 1, size=batch_size)
    x = np.stack([np.asarray(mm[s:s + block_size]) for s in starts])
    y = np.stack([np.asarray(mm[s + 1:s + block_size + 1]) for s in starts])
    return x, y


if __name__ == "__main__":
    rng = np.random.default_rng(42)  # 固定种子

    # [1] 原始文本 -> 文档列表
    docs = make_documents()
    total_chars = sum(len(d) for d in docs)
    print(f"[1] 文档准备: {len(docs)} 篇文档，共 {total_chars:,} 字")
    print(f"    示例文档: \"{docs[0][:40]}...\"")

    # [2] 构建 vocab (字符级 — 换用 level01 的 BPE 流会更短)
    stoi, itos = build_vocab(docs)
    print(f"\n[2] 构建 vocab: {len(itos)} 个"
          f" (特殊 token {len(SPECIALS)} + 字符 {len(itos) - len(SPECIALS)})")
    print(f"    特殊 token id: " +
          ", ".join(f"{t}={stoi[t]}" for t in SPECIALS))

    # [3] 分词 + <s>...</s> 打包
    stream = tokenize_and_pack(docs, stoi)
    print(f"\n[3] 打包完成: {len(docs)} 篇文档 -> token 流 {len(stream):,} 个")
    print("    直接按文档切块，短文档就会浪费大量 <pad>；")
    print("    先全部接起来再按固定长度切，GPU 才能被填得满满当当。")

    # [4] 保存为 uint16 二进制分片
    os.makedirs(OUT_DIR, exist_ok=True)
    shard_path = OUT_DIR / "shard_000.bin"
    stream.tofile(shard_path)
    disk = os.path.getsize(shard_path)
    text_bytes = sum(len(d.encode("utf-8")) for d in docs)
    print(f"\n[4] 分片保存: {shard_path}")
    print(f"    大小 {disk:,} 字节 (每 token 2 字节)"
          f" / 原文 UTF-8 {text_bytes:,} 字节")
    print("    真实流水线会造出几千个这样的分片，供并行读取。")

    # [5] memmap 加载 — 不把整个文件搬进 RAM，立即可用
    mm = np.memmap(shard_path, dtype=np.uint16, mode="r")
    print(f"\n[5] memmap 加载: {len(mm):,} token — 用到哪块才读哪块")
    print("    这就是几百 GB 的分片也能近乎 0 秒打开的原因。")

    # [6] 生成固定长度批次
    block_size, batch_size = 64, 8
    x, y = get_batch(mm, block_size, batch_size, rng)
    print(f"\n[6] 批次生成: x{x.shape}, y{y.shape}"
          f" (block_size={block_size}, batch_size={batch_size})")
    print(f"    x[0][:8] = {x[0][:8].tolist()}")
    print(f"    y[0][:8] = {y[0][:8].tolist()}  <- x 右移一格的答案")

    # [7] 解码验证 — 文档边界 token 会混进批次里。
    def decode(ids):
        return "".join(itos[i] for i in ids)

    row = decode(x[0])
    print(f"\n[7] 批次首行复原: \"{row[:60]}\"")
    boundary = [i for i in range(batch_size)
                if stoi[BOS] in x[i] or stoi[EOS] in x[i]]
    print(f"    含文档边界(<s>/</s>)的行: {len(boundary)}/{batch_size} 个")
    print("    跨越边界接起来的片段也照样参与训练 — <s> 会告诉模型")
    print("    '新文档从这里开始'，模型自己学会在这里换脑子。")
