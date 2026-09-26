"""
level06 — 単語埋め込み (Word Embedding)

tiny_corpus (日本語ミニコーパス) から共起行列を作り、
PPMI 変換 + SVD 圧縮で単語埋め込みを「自分の手で」学習します。
類似単語の検索と、2 次元埋め込み地図の PNG 保存まで実演します。
word2vec のような専用ライブラリなし、numpy だけを使います。
"""

import os
import sys
import pathlib
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
JOSA = ["が", "を", "に"]


def set_japanese_font() -> None:
    """図の中の日本語が文字化けしないよう、システムにある日本語フォントを探して設定します。"""
    candidates = ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic",
                  "Meiryo", "Noto Sans CJK JP", "IPAexGothic"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def tokenize(corpus: str) -> list[str]:
    """句点の除去 + 語末の助詞 (が/を/に) の簡易分離。
    tiny_corpus は文節ごとに空白が入っているので、split() が使えます。"""
    tokens = []
    for word in corpus.replace("。", " ").split():
        for josa in JOSA:
            if word.endswith(josa) and len(word) - len(josa) >= 1:
                word = word[: -len(josa)]
                break
        tokens.append(word)
    return tokens


def build_cooccurrence(tokens: list[str], vocab: list[str], window: int = 2):
    """[2] 窓サイズ window の中で一緒に現れた回数を数える「同席記録帳」。"""
    index = {w: i for i, w in enumerate(vocab)}
    C = np.zeros((len(vocab), len(vocab)))
    for i, center in enumerate(tokens):
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if i != j:
                C[index[center], index[tokens[j]]] += 1
    return C


def ppmi(C: np.ndarray) -> np.ndarray:
    """[3] 正の自己相互情報量 — 「偶然以上の親密さ」だけを残す。"""
    total = C.sum()
    row = C.sum(axis=1, keepdims=True)
    col = C.sum(axis=0, keepdims=True)
    expected = row @ col / total                 # 偶然一緒に出る期待回数
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(C * total / (row * col))
    pmi[~np.isfinite(pmi)] = 0.0
    return np.maximum(pmi, 0.0)


def nearest(word: str, emb: np.ndarray, vocab: list[str], k: int = 4):
    """[5] コサイン類似度による最近傍 k 個。"""
    index = {w: i for i, w in enumerate(vocab)}
    normed = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = normed @ normed[index[word]]
    order = np.argsort(sims)[::-1]
    return [(vocab[i], float(sims[i])) for i in order if vocab[i] != word][:k]


if __name__ == "__main__":
    print("=" * 70)
    print("単語埋め込み — 共起 + PPMI + SVD で単語の地図を作る")
    print("=" * 70 + "\n")
    np.random.seed(0)
    set_japanese_font()

    # [1] コーパスの準備 ---------------------------------------------------
    corpus = hjh_data.tiny_corpus()
    tokens = tokenize(corpus)
    freq = Counter(tokens)
    vocab = sorted(freq)
    print(f"[1] コーパス: {len(corpus):,}文字 -> トークン {len(tokens):,}個、語彙 {len(vocab)}種")
    print(f"    頻度上位: {freq.most_common(5)}\n")

    # [2] 共起行列 ---------------------------------------------------------
    C = build_cooccurrence(tokens, vocab, window=2)
    print(f"[2] 共起行列 {C.shape} (窓サイズ 2)")
    idx = {w: i for i, w in enumerate(vocab)}
    student_row = C[idx["学生"]]
    top = np.argsort(student_row)[::-1][:5]
    print("    「学生」の主な同席者:",
          ", ".join(f"{vocab[i]}({int(student_row[i])}回)" for i in top))
    print("    -> 文型が「時間 主語 目的語 動詞」なので、主語の前後の窓には動詞がよく入ります。")
    print("       この「同席プロフィール」が、そのまま単語の指紋になります。\n")

    # [3] PPMI -------------------------------------------------------------
    P = ppmi(C)
    common_word = freq.most_common(1)[0][0]
    print(f"[3] PPMI 変換: 偶然の同席 (ありふれた単語) のスコアを割り引きます")
    print(f"    例: 「学生」-「{common_word}」の共起は {int(C[idx['学生'], idx[common_word]])}回もあるのに "
          f"PPMI {P[idx['学生'], idx[common_word]]:.2f}")
    print(f"    -> 回数が多くても「誰とでも一緒に出る」組み合わせは情報ではありません。\n")

    # [4] SVD 圧縮 ---------------------------------------------------------
    U, S, Vt = np.linalg.svd(P)
    dim = 8
    emb = U[:, :dim] * S[:dim]                   # 上位 dim 軸だけ残した座標
    explained = S[:dim].sum() / S.sum()
    print(f"[4] SVD 圧縮: {P.shape[1]}次元 -> {dim}次元の埋め込み")
    print(f"    上位 {dim}個の特異値が全体の情報の {explained:.0%} を担います。\n")

    # [5] 類似単語の検索 ---------------------------------------------------
    print("[5] 類似単語の検索 (コサイン類似度)")
    for query in ["学生", "報告書", "作った", "昨日"]:
        pairs = ", ".join(f"{w}({s:.2f})" for w, s in nearest(query, emb, vocab))
        print(f"    {query:6s} の近所: {pairs}")
    print("    -> 主語は主語同士、目的語は目的語同士、動詞は動詞同士で集まります。「意味 (役割) が近ければ距離も近い」!\n")

    # [6] 埋め込み地図の PNG ------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    coords = U[:, :2] * S[:2]                    # 図示用の 2 次元座標
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[:, 0], coords[:, 1], s=28, color="#3b6db4")
    for i, w in enumerate(vocab):
        ax.annotate(w, (coords[i, 0], coords[i, 1]), fontsize=9,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_title("Word embedding map (co-occurrence + PPMI + SVD)")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "embedding_map.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[6] 埋め込み地図を保存: {path}")
    print("    図の中で名詞・動詞・時間表現がそれぞれの領域を作っているか確認してみてください。")
    print("\n結論: 「単語の意味は付き合う友だちが決める」を、統計+線形代数で実装しました。")
