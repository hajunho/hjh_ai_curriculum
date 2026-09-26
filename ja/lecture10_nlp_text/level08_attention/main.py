"""
level08 — アテンション機構

scaled dot-product attention を numpy で自作します。
おもちゃの例題で途中計算をすべて確認し、「あめ」(rain/candy) の
曖昧な 2 つの文のセルフアテンションを計算して、「同じ単語が文によって
違う場所に注目する」様子をヒートマップ PNG として保存します。
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEED = 7


def set_japanese_font() -> None:
    candidates = ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic",
                  "Meiryo", "Noto Sans CJK JP", "IPAexGothic"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# [1] アテンションの実装 — 核心は 5 行
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    """行方向の softmax。最大値を引いてオーバーフローを防止。"""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """scaled dot-product attention。戻り値: (出力, 重み行列)。"""
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)     # 1) クエリ-キーの関連度  2) スケール
    weights = softmax(scores)           # 3) 注目度の配分表 (行の合計 = 1)
    return weights @ V, weights         # 4) 値の加重平均


# ---------------------------------------------------------------------------
# [2] おもちゃの例題 — 数字を全部目で確認
# ---------------------------------------------------------------------------

def demo_toy() -> None:
    print("[1]-[2] アテンション 5 行実装 + トークン 3 個のおもちゃ例題")
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(3, 4))                    # トークン 3 個、4 次元埋め込み
    Wq, Wk, Wv = (rng.normal(size=(4, 4)) for _ in range(3))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv               # Q/K/V は X に行列積を 1 回ずつ
    scores = Q @ K.T
    scaled = scores / np.sqrt(K.shape[-1])
    weights = softmax(scaled)
    out = weights @ V
    np.set_printoptions(precision=2, suppress=True)
    print(f"    スコア QK^T:\n{scores}")
    print(f"    スケール後 (÷√4):\n{scaled}")
    print(f"    softmax 重み (各行の合計 = {weights.sum(axis=1)}):\n{weights}")
    print(f"    出力 (重み @ V) の shape: {out.shape}")
    print("    -> 「関連度の計算 -> 確率化 -> 加重平均」がすべてです。\n")


# ---------------------------------------------------------------------------
# [3] 「あめ」の曖昧な文のセルフアテンション
#     学習済みの埋め込みの代わりに、話題 (topic) 成分を仕込んだ手作りの埋め込みを使います。
#     あめ = rain 成分 + candy 成分を半々に持つ多義語として設計。
# ---------------------------------------------------------------------------

SENT_CANDY = ["あめが", "口で", "とけて", "甘い"]
SENT_RAIN = ["あめが", "ざあざあ", "降って", "道が", "混んだ"]


def build_embeddings() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    dim = 16
    candy_topic = rng.normal(size=dim)             # 「お菓子」の話題軸
    rain_topic = rng.normal(size=dim)              # 「天気」の話題軸
    def vec(topic_mix, noise_scale=0.45):
        return topic_mix + rng.normal(size=dim) * noise_scale
    return {
        "あめが": vec(0.5 * candy_topic + 0.5 * rain_topic),   # 多義語: 2 つの話題を半々
        "口で": vec(candy_topic), "とけて": vec(candy_topic),
        "甘い": vec(np.zeros(dim), 0.8),
        "ざあざあ": vec(rain_topic), "降って": vec(rain_topic),
        "道が": vec(np.zeros(dim), 0.8), "混んだ": vec(np.zeros(dim), 0.8),
    }


def self_attention_of(tokens: list[str], emb: dict) -> np.ndarray:
    """解釈を明確にするため、Q=K=V=X という最も単純なセルフアテンション。"""
    X = np.stack([emb[t] for t in tokens])
    _, weights = attention(X, X, X)
    return weights


def demo_ambiguity(emb: dict) -> tuple[np.ndarray, np.ndarray]:
    print("[3] 同じ「あめが」、文によって注目する場所が変わる (セルフアテンション)")
    results = []
    for tokens in (SENT_CANDY, SENT_RAIN):
        W = self_attention_of(tokens, emb)
        results.append(W)
        row = W[0]                                  # 「あめが」の行の注目度
        pairs = ", ".join(f"{t}({w:.2f})" for t, w in zip(tokens, row))
        print(f"    文: {' '.join(tokens)}")
        print(f"      「あめが」の注目度: {pairs}")
    print("    (参考: 自分自身への注目度が大きいのは、Q=K としたため自己類似度が")
    print("     最大になるからです。実際のモデルは W_Q, W_K を学習して調節します。)")
    print("    -> 飴の文では 口で/とけて に、雨の文では ざあざあ/降って に")
    print("       注目します。「あめ」の表現が文脈の単語で組み立て直される —")
    print("       level00 の曖昧性問題を、アテンションはこう扱います。\n")
    return results[0], results[1]


def save_heatmaps(W_candy: np.ndarray, W_rain: np.ndarray) -> None:
    print("[4] アテンション重みのヒートマップを保存")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, W, tokens, title in (
        (axes[0], W_candy, SENT_CANDY, "candy: あめが 口で とけて 甘い"),
        (axes[1], W_rain, SENT_RAIN, "rain: あめが ざあざあ 降って 道が 混んだ"),
    ):
        im = ax.imshow(W, cmap="Blues", vmin=0, vmax=W.max())
        ax.set_xticks(range(len(tokens)), tokens, rotation=45)
        ax.set_yticks(range(len(tokens)), tokens)
        ax.set_title(title, fontsize=10)
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if W[i, j] < 0.5 * W.max() else "white")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Self-attention weights (row = attending token)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "attention_heatmap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    保存: {path}")
    print("    読み方: 行 = 注目する側のトークン、列 = 注目される側、行の合計 = 1。\n")


def demo_cost() -> None:
    print("[5] アテンションのコスト — すべてのトークンがすべてのトークンを見る (n^2)")
    for n in (10, 100, 1000, 10000):
        cells = n * n
        mb = cells * 4 / 1e6                        # float32 換算
        print(f"    トークン {n:6,d}個 -> 重み行列 {cells:12,d}マス (~{mb:8.1f} MB/層/ヘッド)")
    print("    -> LLM のコンテキスト長制限と、長い入力の料金の根本原因です。")
    print("\n結論: 「関連度に応じた加重平均」の 1 行が、現代 AI の心臓部です。")


if __name__ == "__main__":
    print("=" * 70)
    print("アテンション — 会議で発言者ごとに注目度を変える")
    print("=" * 70 + "\n")
    np.random.seed(SEED)
    set_japanese_font()
    demo_toy()
    emb = build_embeddings()
    W_candy, W_rain = demo_ambiguity(emb)
    save_heatmaps(W_candy, W_rain)
    demo_cost()
