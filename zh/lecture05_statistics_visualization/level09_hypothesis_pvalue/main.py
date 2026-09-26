"""
level09 — 假设检验与 p 值

把两家门店(浦东店 vs 天河店)每笔销售额的差异是否出于偶然送上法庭。
1) 置换检验: 把门店标签打乱 10,000 次，在'偶然差异'分布中直接数出 p 值
2) 与 scipy 的 Welch t 检验结果比较
3) 用真的没有差异的样本做对照实验 (第一类错误的含义)

※ 图内文字统一用英文，避免因环境缺字体而乱码。
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
N_PERM = 10_000


def setup_plot_style() -> None:
    plt.rcParams["axes.unicode_minus"] = False


def load_two_stores() -> tuple[np.ndarray, np.ndarray]:
    rows = hjh_data.sales_table(n_days=60, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    rng = np.random.default_rng(909)
    a = rng.choice(df[df["store"] == "浦东店"]["revenue"].to_numpy(), 150, replace=False)
    b = rng.choice(df[df["store"] == "天河店"]["revenue"].to_numpy(), 150, replace=False)
    return a.astype(float), b.astype(float)


def permutation_test(a: np.ndarray, b: np.ndarray, seed: int = 910) -> tuple[float, np.ndarray, float]:
    """打乱标签的检验: 亲手造出'仅凭偶然产生的差异'的分布。"""
    rng = np.random.default_rng(seed)
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n_a = len(a)
    fake_diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        rng.shuffle(pooled)                      # H0 的世界: 标签毫无意义
        fake_diffs[i] = pooled[:n_a].mean() - pooled[n_a:].mean()
    # 双侧检验: 偶然出现不小于观测差异绝对值的比例
    p = (np.abs(fake_diffs) >= abs(observed)).mean()
    return p, fake_diffs, observed


def trial_of_two_stores() -> None:
    a, b = load_two_stores()
    print("[1] 被告: '浦东店和天河店的销售额差异纯属偶然' (原假设 H0)")
    print(f"    浦东店样本平均 {a.mean() / 1e4:8.1f} 万韩元 (n={len(a)})")
    print(f"    天河店样本平均 {b.mean() / 1e4:8.1f} 万韩元 (n={len(b)})")
    print(f"    观测到的差异   {(a.mean() - b.mean()) / 1e4:+8.1f} 万韩元")

    p_perm, fake_diffs, observed = permutation_test(a, b)
    print()
    print(f"[2] 置换检验 — 与打乱标签 {N_PERM:,} 次造出的'偶然差异'分布比较")
    print(f"    偶然差异分布: 平均 {fake_diffs.mean() / 1e4:+.2f}, "
          f"标准差 {fake_diffs.std() / 1e4:.2f} (万韩元)")
    print(f"    p 值(置换) = {p_perm:.4f}")

    t_stat, p_scipy = stats.ttest_ind(a, b, equal_var=False)   # Welch t 检验
    print()
    print("[3] scipy 一行检验 — stats.ttest_ind(a, b, equal_var=False)")
    print(f"    t = {t_stat:.3f}, p 值(t 检验) = {p_scipy:.4f}")
    print(f"    -> 模拟 p({p_perm:.4f}) ≈ 公式 p({p_scipy:.4f})。算的是同一个东西。")
    verdict = "拒绝 — 难以视为偶然 (认定差异)" if p_scipy < 0.05 else "拒绝失败 — 用偶然也解释得通"
    print(f"    判决(显著性水平 5%): H0 {verdict}")

    # 图: 偶然分布 + 观测差异的位置
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(fake_diffs / 1e4, bins=60, color="#9fbce8", edgecolor="white",
            label="Differences by pure chance (10,000 shuffles)")
    ax.axvline(observed / 1e4, color="#d1495b", lw=2,
               label=f"Observed diff {observed / 1e4:+.1f} (10k KRW)")
    ax.axvline(-observed / 1e4, color="#d1495b", lw=1, ls="--")
    ax.set_title(f"p-value = tail area beyond red lines = {p_perm:.4f}")
    ax.set_xlabel("Mean difference between stores (10k KRW)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "permutation.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[4] 置换分布图已保存: {path}")


def control_experiment() -> None:
    """真的'没有'差异的两个样本 — 把同一家门店对半分再检验。"""
    rows = hjh_data.sales_table(n_days=60, seed=42)
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df = df[df["revenue"] > 0]
    rng = np.random.default_rng(911)
    hongdae = df[df["store"] == "海淀店"]["revenue"].to_numpy(dtype=float).copy()
    rng.shuffle(hongdae)
    half1, half2 = hongdae[:100], hongdae[100:200]

    t_stat, p = stats.ttest_ind(half1, half2, equal_var=False)
    print()
    print("[5] 对照实验 — 把同一家海淀店的销售额随机对半分再检验")
    print(f"    (定义上无真实差异) t = {t_stat:.3f}, p = {p:.4f}")
    print("    -> p 很大 = '用偶然完全解释得通'。但这不是'证明了没有差异'。")
    print("       无罪判决和证明清白是两回事。")
    print("    -> 这种'无差异检验'100 次里也约有 5 次会出 p<0.05。")
    print("       这就是显著性水平 5% = 认下第一类错误(误报)的契约含义。")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_plot_style()
    trial_of_two_stores()
    control_experiment()


if __name__ == "__main__":
    main()
