"""
level07 — 正态分布与中心极限定理(CLT)

1) 用 100 万个随机数验证 68-95-99.7 规则。
2) 在均匀分布、指数分布上以样本量 n=1/5/30 各抽 5,000 次样本平均，
   把"无论原始形状如何、平均数的分布都向钟形收敛"的过程
   保存为 outputs/clt_grid.png 一张图 (2x3 网格)。

※ 图内文字统一用英文，避免因环境缺字体而乱码。
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SAMPLE_SIZES = [1, 5, 30]
N_REPEAT = 5_000  # 抽取多少次样本平均


def setup_plot_style() -> None:
    plt.rcParams["axes.unicode_minus"] = False


def rule_68_95_997() -> None:
    """用 100 万个正态随机数清点 68-95-99.7 规则。"""
    rng = np.random.default_rng(707)
    z = rng.normal(0, 1, size=1_000_000)
    print("[1] 68-95-99.7 规则验证 (标准正态随机数 1,000,000 个)")
    for k, expect in [(1, 68.3), (2, 95.4), (3, 99.7)]:
        ratio = (np.abs(z) <= k).mean() * 100
        print(f"    ±{k}σ 以内的比例: {ratio:5.2f}%  (理论值约 {expect}%)")


def normal_pdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """正态分布概率密度函数 (不用 scipy，直接手写)。"""
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def clt_experiment() -> None:
    """把均匀、指数分布的样本平均分布画成网格。"""
    rng = np.random.default_rng(708)

    # (名称, 样本生成函数, 总体平均, 总体标准差)
    uniform_spec = ("Uniform U(0,1)", lambda size: rng.uniform(0, 1, size),
                    0.5, 1 / np.sqrt(12))
    expo_spec = ("Exponential (mean 1)", lambda size: rng.exponential(1.0, size),
                 1.0, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    print()
    print("[2][3] 样本平均实验 — 每格是把'n 个样本的平均'记录 5,000 次的分布")
    print(f"    {'原始分布':<22} {'n':>4} {'平均的平均':>12} {'平均的标准差':>14} {'理论值 σ/√n':>12}")

    for row, (name, sampler, mu, sigma) in enumerate([uniform_spec, expo_spec]):
        for col, n in enumerate(SAMPLE_SIZES):
            # 核心: 造出 (N_REPEAT, n) 的表、逐行取平均 -> N_REPEAT 个样本平均
            means = sampler((N_REPEAT, n)).mean(axis=1)
            se_theory = sigma / np.sqrt(n)
            print(f"    {name:<22} {n:>4} {means.mean():>12.4f} "
                  f"{means.std():>14.4f} {se_theory:>12.4f}")

            ax = axes[row][col]
            ax.hist(means, bins=50, density=True, color="#9fbce8", edgecolor="white")
            xs = np.linspace(means.min(), means.max(), 200)
            ax.plot(xs, normal_pdf(xs, mu, se_theory), color="#d1495b", lw=2,
                    label="CLT theoretical curve")
            ax.set_title(f"{name}, n={n}")
            if col == 0:
                ax.set_ylabel("Density")
            if row == 0 and col == 2:
                ax.legend(fontsize=9)

    fig.suptitle("Central Limit Theorem — whatever the source, the means gather into a bell", fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "clt_grid.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print()
    print(f"[4] CLT 网格图已保存: {path}")
    print("    左列(n=1)保持原始形状，右列(n=30)紧贴钟形曲线。")
    print("[5] 请在上表确认'平均的标准差'与理论值 σ/√n 一致。")
    print("    -> 样本得收集 4 倍，误差才减半 (√n 的经济学)。")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()
    rule_68_95_997()
    clt_experiment()


if __name__ == "__main__":
    main()
