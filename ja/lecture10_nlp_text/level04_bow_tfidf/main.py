"""
level04 — BoW と TF-IDF

文書-単語行列 (BoW) と TF-IDF を numpy で自作し、
sklearn TfidfVectorizer の結果と小数点以下 12 桁まで一致することを検証します。
最後に SAMPLE_DOCS (社内規定・マニュアル) から文書ごとのキーワードを抽出し、
文書間のコサイン類似度を計算します。
"""

import re
import sys
import pathlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def tokenize(text: str) -> list[str]:
    """日本語の連続文字列は文字 2-gram に、英数字の連続はそのまま 1 トークンに。
    分かち書きのない日本語向けの level03「戦略3」を採用します。
    (sklearn 側にも同じ関数を analyzer として渡すので、条件は完全に同一です。)"""
    tokens = []
    for run in re.findall(r"[ぁ-んァ-ヶー一-龯々]+|[a-z0-9]+", text.lower()):
        if re.match(r"[a-z0-9]", run) or len(run) < 2:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


# ---------------------------------------------------------------------------
# [1] BoW — 文書-単語行列を手作業で
# ---------------------------------------------------------------------------

def build_bow(docs: list[str]) -> tuple[np.ndarray, list[str]]:
    """文書リスト -> (出現回数行列, 語彙リスト)。語彙は文字コード順に整列。"""
    vocab = sorted({tok for d in docs for tok in tokenize(d)})
    index = {w: i for i, w in enumerate(vocab)}
    counts = np.zeros((len(docs), len(vocab)), dtype=float)
    for row, doc in enumerate(docs):
        for tok in tokenize(doc):
            counts[row, index[tok]] += 1
    return counts, vocab


# ---------------------------------------------------------------------------
# [2] TF-IDF — sklearn の基本形と同じ式を numpy で
#     idf(t) = ln((1+N) / (1+df(t))) + 1   (スムージング)
#     tfidf  = tf * idf を文書ごとに L2 正規化
# ---------------------------------------------------------------------------

def tfidf_numpy(counts: np.ndarray) -> np.ndarray:
    n_docs = counts.shape[0]
    df = (counts > 0).sum(axis=0)                       # 単語ごとの出現文書数
    idf = np.log((1 + n_docs) / (1 + df)) + 1           # スムージング IDF
    weighted = counts * idf                             # TF x IDF
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return weighted / norms                             # 文書ごとの L2 正規化


# ---------------------------------------------------------------------------
# 実演
# ---------------------------------------------------------------------------

# 形態素解析 (level03) を通した後の姿を想定し、空白区切りにしてあります。
MINI_DOCS = [
    "配送 迅速 配送 担当 親切",
    "梱包 丁寧 価格 満足",
    "配送 遅延 価格 不満",
]


def demo_bow() -> None:
    print("[1] BoW: 文書をすりつぶして「単語の成分表」にする")
    counts, vocab = build_bow(MINI_DOCS)
    header = "         " + " ".join(f"{w:>4s}" for w in vocab)
    print(header)
    for i, row in enumerate(counts):
        cells = " ".join(f"{int(v):4d}" for v in row)
        print(f"    文書{i} {cells}   <- {MINI_DOCS[i]!r}")
    sparsity = (counts == 0).mean()
    print(f"    -> 行列 {counts.shape}、0 の割合 {sparsity:.0%} (疎行列)\n")


def demo_verify() -> None:
    print("[2]-[3] TF-IDF 自作実装 vs sklearn の突き合わせ検証")
    docs = [r["text"] for r in hjh_data.review_corpus(80, seed=3)]
    counts, vocab = build_bow(docs)
    mine = tfidf_numpy(counts)

    vec = TfidfVectorizer(analyzer=tokenize)             # 基本設定: スムージング+L2
    theirs = vec.fit_transform(docs).toarray()
    their_vocab = vec.get_feature_names_out().tolist()

    assert vocab == their_vocab, "語彙辞書の不一致"
    max_err = float(np.abs(mine - theirs).max())
    print(f"    文書 {len(docs)}件、語彙 {len(vocab)}種")
    print(f"    自作実装と sklearn の最大誤差: {max_err:.2e}")
    ok = max_err < 1e-10
    print(f"    -> {'一致! ライブラリの内部を正確に再現できました。' if ok else '不一致 — 式の確認が必要'}\n")
    assert ok


def demo_keywords() -> np.ndarray:
    print("[4] SAMPLE_DOCS 文書ごとのキーワード top5 (TF-IDF 基準)")
    names = list(hjh_data.SAMPLE_DOCS.keys())
    docs = list(hjh_data.SAMPLE_DOCS.values())
    counts, vocab = build_bow(docs)
    scores = tfidf_numpy(counts)
    for i, name in enumerate(names):
        top = np.argsort(scores[i])[::-1][:5]
        words = [f"{vocab[j]}({scores[i, j]:.2f})" for j in top]
        print(f"    {name:16s}: {', '.join(words)}")
    print("    -> 休暇の文書には年次・休暇、経費の文書には出張・経費 — 文書の「顔」が抽出されます。")
    print("       「暇は」「張が」のような境界をまたぐ 2-gram の混入は、この方式の代表的なノイズです。\n")
    return scores


def demo_similarity(scores: np.ndarray) -> None:
    print("[5] 文書間のコサイン類似度 (L2 正規化ベクトルの内積)")
    names = [n.replace(".txt", "").split("_")[-1] for n in hjh_data.SAMPLE_DOCS.keys()]
    sim = scores @ scores.T
    print("    " + " " * 14 + "  ".join(f"{n[:6]:>6s}" for n in names))
    for i, name in enumerate(names):
        cells = "  ".join(f"{sim[i, j]:6.2f}" for j in range(len(names)))
        print(f"    {name:14s}{cells}")
    # 自分自身を除いた最大類似度ペア
    mask = sim - np.eye(len(names))
    a, b = np.unravel_index(np.argmax(mask), mask.shape)
    print(f"    -> 最も似ているペア: {names[a]} <-> {names[b]} ({mask[a, b]:.2f})")
    print("       全体的に類似度が低い理由: 文書同士が「同じ単語」をほとんど使っていないからです。")
    print("       TF-IDF は文字が同じでなければ似ていると認めません — 「意味が近い別の単語」を")
    print("       見分ける方法が level06 の単語埋め込みです。")


if __name__ == "__main__":
    print("=" * 70)
    print("BoW と TF-IDF — テキストを数値ベクトルに、自作実装での検証まで")
    print("=" * 70 + "\n")
    np.random.seed(0)          # このレベルは乱数を使いませんが、慣例として固定
    demo_bow()
    demo_verify()
    scores = demo_keywords()
    demo_similarity(scores)
