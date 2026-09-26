"""
Lecture 08 · Level 00 — ニューラルネットワークはなぜ登場したのか
線形モデルが AND/OR は完璧に解けるのに、XOR は精度 75% を
絶対に超えられないという事実を、最小二乗法と全数探索で証明します。
最後に座標変換(隠れ表現)を手で入れてやると、同じ線形モデルが
XOR を 100% で解くことを示し、表現学習という発想へつなげます。
"""

import numpy as np

np.random.seed(42)  # 再現性のため乱数シードを固定 (このレベルはほぼ乱数に依存しませんが、規則として固定)

# 入力: (x1, x2) の 4 通りの組み合わせ
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

# 正解表: 3 つの論理ルール
TARGETS = {
    "AND": np.array([0, 0, 0, 1]),
    "OR":  np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}


def fit_least_squares(features, y):
    """バイアス(bias)項を付けたうえで、最小二乗法により線形モデルの最適係数を求めます。"""
    A = np.hstack([features, np.ones((len(features), 1))])  # 最後の列 1 = バイアスの席
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef  # (w1, w2, ..., b)


def linear_accuracy(features, y, coef):
    """スコアが 0.5 以上なら 1 に分類したときの精度と予測を返します。"""
    A = np.hstack([features, np.ones((len(features), 1))])
    pred = (A @ coef >= 0.5).astype(int)
    return float((pred == y).mean()), pred


def best_linear_accuracy(features, y, n_w=61, n_b=61, limit=3.0):
    """重み・バイアスの格子を全数探索し、線形の決定境界が出せる最高精度を探します。

    スコア = w1*x1 + w2*x2 + b、スコア >= 0 なら 1 に分類します。
    numpy のブロードキャストで (n_w * n_w * n_b) 個の組み合わせを一度に評価します。
    """
    w_grid = np.linspace(-limit, limit, n_w)
    b_grid = np.linspace(-limit, limit, n_b)
    W1, W2, B = np.meshgrid(w_grid, w_grid, b_grid, indexing="ij")
    # scores の最後の軸がデータ 4 個: shape = (n_w, n_w, n_b, 4)
    scores = (W1[..., None] * features[:, 0]
              + W2[..., None] * features[:, 1]
              + B[..., None])
    pred = (scores >= 0.0).astype(int)
    acc = (pred == y).mean(axis=-1)
    best_idx = np.unravel_index(np.argmax(acc), acc.shape)
    best = float(acc[best_idx])
    n_tried = acc.size
    return best, n_tried


def hidden_features(features):
    """手で設計した隠れ表現(hidden representation)を 2 つ作ります。

    h1 は OR のように(合計が 0.5 以上なら 1)、h2 は AND のように(合計が 1.5 以上なら 1)動きます。
    後で学ぶ隠れ層のニューロン 2 個がやることとまったく同じです。
    """
    s = features[:, 0] + features[:, 1]
    h1 = (s >= 0.5).astype(float)  # OR の役割
    h2 = (s >= 1.5).astype(float)  # AND の役割
    return np.stack([h1, h2], axis=1)


def main():
    print("[1] 真理値表データの準備 — 入力 (x1, x2) の 4 通りの組み合わせと 3 つのルールの正解")
    print("    x1 x2 | AND OR XOR")
    for i in range(len(X)):
        print(f"     {int(X[i, 0])}  {int(X[i, 1])} |  {TARGETS['AND'][i]}   {TARGETS['OR'][i]}   {TARGETS['XOR'][i]}")

    print()
    print("[2] 最小二乗法で各問題に「最もよく合う」線形モデルを学習")
    for name, y in TARGETS.items():
        coef = fit_least_squares(X, y)
        acc, pred = linear_accuracy(X, y, coef)
        print(f"    {name:>3}: w1={coef[0]:+.3f}, w2={coef[1]:+.3f}, b={coef[2]:+.3f}"
              f" -> 精度 {acc * 100:5.1f}%  (予測 {pred.tolist()}, 正解 {y.tolist()})")
    print("    => AND/OR は 100%、XOR は直線 1 本では 4 点をすべて当てられません。")

    print()
    print("[3] 「もっと良い直線が隠れていないか?」 — 重み・バイアスの格子を全数探索")
    for name, y in TARGETS.items():
        best, n_tried = best_linear_accuracy(X, y)
        print(f"    {name:>3}: {n_tried:,}個の (w1, w2, b) 組み合わせをすべて試行 -> 最高精度 {best * 100:5.1f}%")
    print("    => XOR はどの直線を選んでも 75% が上限です。努力不足ではなく構造的な限界です。")

    print()
    print("[4] 座標変換(隠れ表現)を手で入れてやると?")
    H = hidden_features(X)
    print("    新しい座標 h1=OR(x1,x2), h2=AND(x1,x2) にデータを移します。")
    print("    x1 x2 -> h1 h2 | XOR 正解")
    for i in range(len(X)):
        print(f"     {int(X[i, 0])}  {int(X[i, 1])} ->  {int(H[i, 0])}  {int(H[i, 1])} |    {TARGETS['XOR'][i]}")
    coef_h = fit_least_squares(H, TARGETS["XOR"])
    acc_h, pred_h = linear_accuracy(H, TARGETS["XOR"], coef_h)
    best_h, _ = best_linear_accuracy(H, TARGETS["XOR"])
    print(f"    隠れ表現の上の線形モデル: w1={coef_h[0]:+.3f}, w2={coef_h[1]:+.3f}, b={coef_h[2]:+.3f}"
          f" -> 精度 {acc_h * 100:.1f}%")
    print(f"    (全数探索でも最高精度 {best_h * 100:.1f}% を確認 — 実質 y = h1 - h2 の 1 行)")

    print()
    print("[5] 結論")
    print("    - 同じ線形モデルなのに、座標(表現)を変えただけで XOR の精度が 75% -> 100% になりました。")
    print("    - つまりモデルが弱かったのではなく、データを眺める表現が悪かったのです。")
    print("    - この座標変換を、人間ではなく機械がデータから学ぶようにしようというのが")
    print("      表現学習(representation learning)、すなわちニューラルネットワークとディープラーニングの出発点です。")


if __name__ == "__main__":
    main()
