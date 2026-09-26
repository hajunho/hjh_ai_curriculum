"""
事前学習データパイプラインをミニチュアで実装します。
文書を <s>...</s> でつなぎ合わせて(パッキング)1本の長いトークンストリームを作り、
uint16 のバイナリシャードに保存したあと numpy memmap で即座に読み込んで
固定長(block_size)の学習バッチを作り出す全工程を実行します。
実際の大規模学習で数 TB のデータを扱うやり方の縮小版です。
"""

import os
import sys
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"

# 特殊トークン — id を先頭の番号に固定しておきます。
BOS, EOS, PAD, UNK = "<s>", "</s>", "<pad>", "<unk>"
SPECIALS = [BOS, EOS, PAD, UNK]


def make_documents(n_docs: int = 200, sents_per_doc: int = 5):
    """tiny_corpus を文に分割したあと、5文ずつまとめて「文書」を作ります。"""
    corpus = hjh_data.tiny_corpus()
    sents = [s.strip() + "。" for s in corpus.split("。") if s.strip()]
    docs = []
    for i in range(n_docs):
        chunk = sents[i * sents_per_doc:(i + 1) * sents_per_doc]
        docs.append(" ".join(chunk))
    return docs


def build_vocab(docs):
    """文字単位の vocab: 特殊トークン4個 + コーパスのすべての文字。"""
    chars = sorted(set("".join(docs)))
    itos = SPECIALS + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos


def tokenize_and_pack(docs, stoi):
    """各文書を <s> トークン列 </s> で包み、1本の長いストリームにつなぎます。"""
    stream = []
    for doc in docs:
        stream.append(stoi[BOS])
        stream.extend(stoi.get(ch, stoi[UNK]) for ch in doc)
        stream.append(stoi[EOS])
    return np.array(stream, dtype=np.uint16)  # vocab < 65536 なので2バイトで十分


def get_batch(mm, block_size: int, batch_size: int, rng):
    """memmap ストリームからランダムな位置を選んで (x, y) バッチを作ります。
    y は x を1つずらしたもの — 「次のトークン当て」の答案です。"""
    starts = rng.integers(0, len(mm) - block_size - 1, size=batch_size)
    x = np.stack([np.asarray(mm[s:s + block_size]) for s in starts])
    y = np.stack([np.asarray(mm[s + 1:s + block_size + 1]) for s in starts])
    return x, y


if __name__ == "__main__":
    rng = np.random.default_rng(42)  # シード固定

    # [1] 生テキスト -> 文書リスト
    docs = make_documents()
    total_chars = sum(len(d) for d in docs)
    print(f"[1] 文書の準備: {len(docs)}件の文書、合計 {total_chars:,}文字")
    print(f"    文書の例: \"{docs[0][:40]}...\"")

    # [2] vocab の構築 (文字単位 — level01 の BPE を使えばもっと短くなります)
    stoi, itos = build_vocab(docs)
    print(f"\n[2] vocab の構築: {len(itos)}個"
          f" (特殊トークン {len(SPECIALS)} + 文字 {len(itos) - len(SPECIALS)})")
    print(f"    特殊トークンの id: " +
          ", ".join(f"{t}={stoi[t]}" for t in SPECIALS))

    # [3] トークン化 + <s>...</s> パッキング
    stream = tokenize_and_pack(docs, stoi)
    print(f"\n[3] パッキング完了: 文書 {len(docs)}件 -> トークンストリーム {len(stream):,}個")
    print("    文書をそのまま切ると短い文書ごとに <pad> の無駄が出るので、")
    print("    全部つないでから固定長で切る方が GPU をぎっしり埋められます。")

    # [4] uint16 バイナリシャードの保存
    os.makedirs(OUT_DIR, exist_ok=True)
    shard_path = OUT_DIR / "shard_000.bin"
    stream.tofile(shard_path)
    disk = os.path.getsize(shard_path)
    text_bytes = sum(len(d.encode("utf-8")) for d in docs)
    print(f"\n[4] シャードの保存: {shard_path}")
    print(f"    サイズ {disk:,}バイト (トークンあたり2バイト)"
          f" / 原文 UTF-8 {text_bytes:,}バイト")
    print("    実際のパイプラインはこういうシャードを数千個作って並列に読みます。")

    # [5] memmap ロード — ファイル全体を RAM に載せずに即座に使う
    mm = np.memmap(shard_path, dtype=np.uint16, mode="r")
    print(f"\n[5] memmap ロード: {len(mm):,}トークン — 必要な断片だけその都度読む")
    print("    数百 GB のシャードでもロードが0秒に近い理由です。")

    # [6] 固定長バッチの生成
    block_size, batch_size = 64, 8
    x, y = get_batch(mm, block_size, batch_size, rng)
    print(f"\n[6] バッチの生成: x{x.shape}, y{y.shape}"
          f" (block_size={block_size}, batch_size={batch_size})")
    print(f"    x[0][:8] = {x[0][:8].tolist()}")
    print(f"    y[0][:8] = {y[0][:8].tolist()}  <- x を1つずらした答案")

    # [7] デコードで検証 — 文書境界のトークンがバッチの中に混ざって入ってきます。
    def decode(ids):
        return "".join(itos[i] for i in ids)

    row = decode(x[0])
    print(f"\n[7] バッチ先頭行の復元: \"{row[:60]}\"")
    boundary = [i for i in range(batch_size)
                if stoi[BOS] in x[i] or stoi[EOS] in x[i]]
    print(f"    文書境界(<s>/</s>)を含む行: {len(boundary)}/{batch_size}個")
    print("    境界をまたいでつながった断片もそのまま学習します — <s> が「ここから")
    print("    新しい文書が始まる」と教えてくれるので、モデルが自分で文脈を切って読みます。")
