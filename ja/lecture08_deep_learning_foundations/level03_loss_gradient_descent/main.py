"""
損失関数と勾配降下法を目で確かめる実習です。
1 変数関数 f(x) = (x-3)^2 + 0.7*sin(3x) の上で
学習率 (learning rate) 3 種 (小さすぎ/適度/大きすぎ) の勾配降下を回し、
ステップごとの移動経路を数字と図 (PNG) で比較します。
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def f(x):
    # 損失関数の役割: なめらかな谷 + ゆるやかな起伏 (最低点は x≈3 付近)
    return (x - 3.0) ** 2 + 0.7 * np.sin(3.0 * x)


def grad_f(x):
    # f の導関数(解析的微分): 足元の傾斜
    return 2.0 * (x - 3.0) + 2.1 * np.cos(3.0 * x)


def gradient_descent(x0, lr, n_steps):
    """x0 から始めて「勾配の逆方向へ lr の分だけ」移動を繰り返します。"""
    xs = [x0]
    x = x0
    for _ in range(n_steps):
        x = x - lr * grad_f(x)
        xs.append(x)
        if abs(x) > 1e6:          # 発散したら中断 (大きな学習率のデモ用)
            break
    return np.array(xs)


def numerical_grad(x, h=1e-5):
    # 中央差分による数値微分: 解析的微分が正しいかの検算
    return (f(x + h) - f(x - h)) / (2 * h)


def main():
    np.random.seed(42)                       # 再現性 (この実習はほぼ乱数なし)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1] 損失関数 f(x) = (x-3)^2 + 0.7*sin(3x) を「霧のかかった山」と考えます。")
    print("    現在位置の傾斜(勾配)だけを見て、谷(最低点)を探して降りていきます。\n")

    x_check = 7.0
    print("[2] 勾配の検算 — 解析的微分 vs 数値微分 (x=7.0)")
    print(f"    解析的: {grad_f(x_check):+.6f} / 数値: {numerical_grad(x_check):+.6f}")
    print("    2 つの値が同じなので、「足元の傾斜計」は信頼できます。\n")

    x0, n_steps = 9.0, 30
    settings = [("too_small", 0.01), ("good", 0.15), ("too_big", 1.05)]

    print(f"[3] 開始点 x0={x0}, {n_steps}ステップの勾配降下 — 学習率 3 種の比較")
    trajs = {}
    for name, lr in settings:
        xs = gradient_descent(x0, lr, n_steps)
        trajs[name] = (lr, xs)
        marks = [0, 1, 2, 5, 10, len(xs) - 1]
        print(f"\n    学習率 {lr} ({name})")
        for k in marks:
            if k < len(xs):
                print(f"      step {k:2d}: x = {xs[k]:+10.4f}, f(x) = {f(xs[k]):12.4f}")

    print("\n[4] 解釈")
    print("    - 0.01 : 方向は正しいが歩幅が短すぎて、30 ステップでは谷に到着できない。")
    print("    - 0.15 : 数ステップで最低点付近 (x≈3.5) に着地。適度な学習率。")
    print("    - 1.05 : 谷を毎回飛び越えて |x| が大きくなる → 発散 (loss の爆発)。")

    # 図: 関数の曲線の上に 3 つの軌跡を並べる
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    grid = np.linspace(-2, 11, 400)
    titles = {"too_small": "lr=0.01 (too small)",
              "good": "lr=0.15 (good)",
              "too_big": "lr=1.05 (too big → diverge)"}
    for ax, (name, (lr, xs)) in zip(axes, trajs.items()):
        ax.plot(grid, f(grid), color="#888888", lw=1.5)
        xs_plot = xs[np.abs(xs) < 12]        # 図の範囲内の点だけ
        ax.plot(xs_plot, f(xs_plot), "o-", color="#d62728", ms=4, lw=1)
        ax.plot(xs_plot[0], f(xs_plot[0]), "s", color="#1f77b4", ms=8, label="start")
        ax.set_title(titles[name])
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
    fig.suptitle("Gradient descent trajectories with 3 learning rates")
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "gd_learning_rates.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"\n[5] 軌跡の図を保存: {png_path}")

    print("\n[6] まとめ: 学習率は「歩幅」です。小さすぎればいつまでも、大きすぎれば崖っぷち。")
    print("    ディープラーニングの学習がうまくいかないとき、最初に疑うダイヤルが学習率です。")


if __name__ == "__main__":
    main()
