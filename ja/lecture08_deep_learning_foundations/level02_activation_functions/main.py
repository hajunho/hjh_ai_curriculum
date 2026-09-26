"""
Lecture 08 · Level 02 — 活性化関数
sigmoid/tanh/ReLU/Leaky ReLU の形と勾配(導関数)を PNG に保存して比較し、
飽和区間の勾配消失と死んだ ReLU を数字で確認します。
核心の証明: 線形の層 3 個を合成すると (W3 @ W2 @ W1) 1 枚と完全に同じだが、
層の間に ReLU を挟むと、どんな単一の線形層でも真似できないことを示します。
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 画面なしでファイルにだけ図を保存するバックエンド
import matplotlib.pyplot as plt

np.random.seed(42)  # 再現性のためのシード固定

# このファイルのあるフォルダの下の outputs/ に図を保存
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 関数ごとの色: 色覚多様性に配慮したパレット(Okabe-Ito)から固定で割り当て (実行ごと/図ごとに同一)
COLORS = {"sigmoid": "#0072B2", "tanh": "#E69F00",
          "ReLU": "#009E73", "Leaky ReLU": "#CC79A7"}


# ---- 活性化関数と解析的な導関数 --------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1.0 - s)          # 最大値 0.25 (z=0)


def tanh(z):
    return np.tanh(z)


def d_tanh(z):
    return 1.0 - np.tanh(z) ** 2  # 最大値 1.0 (z=0)、両端で 0 へ飽和


def relu(z):
    return np.maximum(0.0, z)


def d_relu(z):
    return (z > 0).astype(float)  # 正なら 1、負なら 0 (死んだ ReLU の原因)


def leaky_relu(z, slope=0.01):
    return np.where(z > 0, z, slope * z)


def d_leaky_relu(z, slope=0.01):
    return np.where(z > 0, 1.0, slope)  # 負の区間にも小さな勾配を残す


FUNCS = [("sigmoid", sigmoid, d_sigmoid),
         ("tanh", tanh, d_tanh),
         ("ReLU", relu, d_relu),
         ("Leaky ReLU", leaky_relu, d_leaky_relu)]


def plot_activations(x):
    """[1] 関数の形(実線)と導関数(点線)を 2x2 パネルで保存します。"""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
    for ax, (name, f, df) in zip(axes.ravel(), FUNCS):
        c = COLORS[name]
        ax.plot(x, f(x), color=c, linewidth=2, label="f(z)")
        ax.plot(x, df(x), color=c, linewidth=2, linestyle="--", alpha=0.55, label="f'(z)")
        ax.set_title(name)
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.axvline(0, color="#999999", linewidth=0.6)
        ax.grid(True, color="#dddddd", linewidth=0.5)
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Activation functions f(z) and derivatives f'(z)")
    path = os.path.join(OUT_DIR, "activations.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_gradients(x):
    """[2] 4 つの関数の導関数だけを一つの軸に重ねて描き、勾配消失を目で比較します。"""
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for name, _f, df in FUNCS:
        # ReLU と Leaky ReLU は正の区間で重なるため、Leaky ReLU だけ点線で区別
        style = "--" if name == "Leaky ReLU" else "-"
        width = 3.0 if name == "ReLU" else 2.0
        ax.plot(x, df(x), color=COLORS[name], linewidth=width, linestyle=style, label=name)
    ax.set_title("Derivatives compared: saturation vs. constant gradient")
    ax.set_xlabel("z")
    ax.set_ylabel("f'(z)")
    ax.grid(True, color="#dddddd", linewidth=0.5)
    ax.legend(loc="center right", frameon=False)
    path = os.path.join(OUT_DIR, "gradients.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def linear_stack_vs_single(rng):
    """[4] 線形の層 3 個の合成 = 単一の行列 W3@W2@W1 であることを数値で証明します。"""
    W1 = rng.normal(size=(5, 3))   # 3 次元入力 -> 5 次元
    W2 = rng.normal(size=(4, 5))   # 5 次元 -> 4 次元
    W3 = rng.normal(size=(2, 4))   # 4 次元 -> 2 次元出力
    x = rng.normal(size=(3, 8))    # 入力 8 個の束(各列がサンプル 1 個)
    deep = W3 @ (W2 @ (W1 @ x))    # 層をひとつずつ通過
    W_single = W3 @ W2 @ W1        # 先に掛けておいた「1 枚もの」の行列
    single = W_single @ x
    return W1, W2, W3, x, float(np.max(np.abs(deep - single)))


def relu_breaks_linearity(W1, W2, W3, x):
    """[5] 層の間に ReLU を挟むと単一の線形層と変わり、加法性も崩れることを示します。"""
    deep_relu = W3 @ relu(W2 @ relu(W1 @ x))
    single = (W3 @ W2 @ W1) @ x
    diff = float(np.max(np.abs(deep_relu - single)))

    # 線形関数なら必ず f(a+b) = f(a) + f(b) でなければなりません (加法性)。
    def net(v):
        return W3 @ relu(W2 @ relu(W1 @ v))

    a, b = x[:, :1], x[:, 1:2]
    additivity_gap = float(np.max(np.abs(net(a + b) - (net(a) + net(b)))))
    return diff, additivity_gap


def main():
    x = np.linspace(-6.0, 6.0, 601)

    path1 = plot_activations(x)
    print("[1] 活性化関数 4 種の形と導関数を描きました")
    print(f"    保存先: {path1}")

    path2 = plot_gradients(x)
    print("[2] 導関数だけを重ねて描いた比較図を保存しました")
    print(f"    保存先: {path2}")

    print()
    print("[3] 飽和と勾配消失、死んだ ReLU を数字で確認")
    print(f"    sigmoid'(0) = {d_sigmoid(0.0):.4f} (最大), sigmoid'(5) = {d_sigmoid(5.0):.6f}")
    print(f"    -> z が 5 になっただけで勾配が {d_sigmoid(0.0) / d_sigmoid(5.0):.0f}分の 1 のレベルに縮みます (飽和)")
    print(f"    sigmoid の最大勾配 0.25 を 10 層分掛けると: 0.25**10 = {0.25 ** 10:.2e}")
    print("    -> 下の層に届く信号は 100 万分の 1 レベル、これが勾配消失です")
    print(f"    ReLU'(-3) = {d_relu(np.array(-3.0)):.2f} -> 負の区間に囚われたニューロンは勾配 0、二度と目覚めない (死んだ ReLU)")
    print(f"    Leaky ReLU'(-3) = {d_leaky_relu(np.array(-3.0)):.2f} -> 小さな勾配を残して生き返る余地を作ります")

    print()
    print("[4] 証明: 活性化のない線形の層 3 個 = 線形の層 1 個")
    rng = np.random.default_rng(42)  # シード固定の乱数行列
    W1, W2, W3, xs, max_diff = linear_stack_vs_single(rng)
    print("    乱数行列 W1(5x3), W2(4x5), W3(2x4) と入力 8 個で比較します。")
    print(f"    max | W3@(W2@(W1@x)) - (W3@W2@W1)@x | = {max_diff:.2e}")
    print("    -> 浮動小数点の限界レベルの 0。3 つの層が 1 枚の行列に完全に圧縮されます。")

    print()
    print("[5] 層の間に ReLU を挟むと?")
    diff, gap = relu_breaks_linearity(W1, W2, W3, xs)
    print(f"    max | W3@relu(W2@relu(W1@x)) - (W3@W2@W1)@x | = {diff:.3f}")
    print(f"    線形なら 0 のはずの加法性誤差 max |f(a+b) - f(a) - f(b)| = {gap:.3f}")
    print("    -> もはやどんな単一の線形層でも真似できません。")
    print("    結論: 深さを本物の深さにしてくれる部品が活性化関数です。")


if __name__ == "__main__":
    main()
