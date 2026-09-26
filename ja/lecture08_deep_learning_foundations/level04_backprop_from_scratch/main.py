"""
誤差逆伝播(backpropagation)を numpy だけでゼロから実装します。
2 層ニューラルネットワーク(2-8-1、tanh + sigmoid)の順伝播と逆伝播を手で書き、
線形モデルには解けなかった XOR の 4 点を 100% 分類できるまで学習させ、
数値微分(中央差分)で解析的な勾配が正しいかを検証します。
"""

import numpy as np

rng = np.random.default_rng(0)   # 再現性: シード固定


# ----------------------------- ニューラルネットワークの部品 -----------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def init_params():
    """2-8-1 ニューラルネットワークの重みを小さな乱数で初期化します。"""
    return {
        "W1": rng.normal(0, 0.8, size=(2, 8)),
        "b1": np.zeros(8),
        "W2": rng.normal(0, 0.8, size=(8, 1)),
        "b2": np.zeros(1),
    }


def forward(params, X):
    """順伝播: 入力 -> 隠れ(tanh) -> 出力(sigmoid)。中間値は逆伝播用に保管。"""
    z1 = X @ params["W1"] + params["b1"]      # (n, 8)
    h = np.tanh(z1)                           # 隠れ表現
    z2 = h @ params["W2"] + params["b2"]      # (n, 1)
    p = sigmoid(z2)                           # 解約確率のように 0~1
    cache = {"X": X, "z1": z1, "h": h, "p": p}
    return p, cache


def bce_loss(p, y):
    """二値クロスエントロピー: 確信に満ちた誤答に大きな罰点。"""
    eps = 1e-9
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def backward(params, cache, y):
    """誤差逆伝播: 出力の誤差の責任を連鎖律で逆向きに配分します。"""
    X, h, p = cache["X"], cache["h"], cache["p"]
    n = X.shape[0]

    # 出力層: BCE+sigmoid の組み合わせの微分は (p - y) にきれいに整理されます
    dz2 = (p - y) / n                          # (n, 1)
    grads = {
        "W2": h.T @ dz2,                       # 隠れ層の出力が寄与した分だけ責任を配分
        "b2": dz2.sum(axis=0),
    }
    # 隠れ層: 出力層から渡された責任 × tanh 通過時に縮む比率
    dh = dz2 @ params["W2"].T                  # (n, 8)
    dz1 = dh * (1.0 - h ** 2)                  # tanh'(z) = 1 - tanh(z)^2
    grads["W1"] = X.T @ dz1
    grads["b1"] = dz1.sum(axis=0)
    return grads


def numerical_grads(params, X, y, key, h_eps=1e-6):
    """数値微分による検算: パラメータをほんのわずか動かして loss の変化量を直接測ります。"""
    W = params[key]
    num = np.zeros_like(W)
    it = np.nditer(W, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = W[idx]
        W[idx] = orig + h_eps
        loss_plus = bce_loss(forward(params, X)[0], y)
        W[idx] = orig - h_eps
        loss_minus = bce_loss(forward(params, X)[0], y)
        W[idx] = orig
        num[idx] = (loss_plus - loss_minus) / (2 * h_eps)
        it.iternext()
    return num


# ------------------------------- 実習シナリオ -------------------------------

def main():
    # XOR: 線形モデル(level00)には絶対に解けなかった問題
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    params = init_params()

    print("[1] 問題: XOR — (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0")
    print("    2-8-1 ニューラルネットワーク(tanh + sigmoid)を、手作りの順伝播+逆伝播で学習します。\n")

    print("[2] 勾配の検証 — 逆伝播(解析) vs 数値微分(実測)")
    p, cache = forward(params, X)
    ana = backward(params, cache, y)
    worst = 0.0
    for key in ["W1", "b1", "W2", "b2"]:
        num = numerical_grads(params, X, y, key)
        # 相対誤差: |解析 - 実測| / (|解析| + |実測|)
        denom = np.abs(ana[key]) + np.abs(num) + 1e-12
        rel = float(np.max(np.abs(ana[key] - num) / denom))
        worst = max(worst, rel)
        print(f"    {key}: 最大相対誤差 = {rel:.2e}")
    assert worst < 1e-4, "逆伝播の実装にバグがあります!"
    print("    => すべて 1e-4 未満。手作りの逆伝播は正確です。\n")

    print("[3] 学習開始 (勾配降下、学習率 0.5)")
    lr, n_epochs = 0.5, 3000
    for epoch in range(1, n_epochs + 1):
        p, cache = forward(params, X)
        grads = backward(params, cache, y)
        for key in params:                       # すべての重みを勾配の逆方向へ
            params[key] -= lr * grads[key]
        if epoch in (1, 10, 100, 500, 1000, 2000, 3000):
            acc = float(np.mean((p > 0.5) == y))
            print(f"    epoch {epoch:4d}: loss = {bce_loss(p, y):.4f}, 精度 = {acc:.0%}")

    print("\n[4] 学習結果 — 4 つの入力に対する予測確率")
    p, _ = forward(params, X)
    for xi, yi, pi in zip(X, y, p):
        mark = "O" if (pi[0] > 0.5) == bool(yi[0]) else "X"
        print(f"    入力 {xi} -> 予測 p = {pi[0]:.4f} (正解 {int(yi[0])}) {mark}")
    acc = float(np.mean((p > 0.5) == y))
    assert acc == 1.0, "XOR の収束に失敗"
    print(f"    最終精度: {acc:.0%} — 線形モデルの 75% の壁を越えました。\n")

    print("[5] 隠れ層が作った「新しい座標」(表現) — 各入力の隠れ値の一部")
    _, cache = forward(params, X)
    for xi, hi in zip(X, cache["h"]):
        print(f"    {xi} -> h[:3] = [{hi[0]:+.2f}, {hi[1]:+.2f}, {hi[2]:+.2f}]")
    print("    この新しい座標では、XOR が直線 1 本で分かれます。これが表現学習です。")


if __name__ == "__main__":
    main()
